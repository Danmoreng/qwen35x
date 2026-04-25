bool parse_token_list_csv(
  const std::string & csv,
  std::vector<std::int32_t> & out_tokens,
  std::string & error_message) {
  out_tokens.clear();
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    const auto begin = token.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
      continue;
    }
    const auto end = token.find_last_not_of(" \t\r\n");
    const std::string trimmed = token.substr(begin, end - begin + 1);
    try {
      const long long parsed = std::stoll(trimmed);
      if (parsed < static_cast<long long>(std::numeric_limits<std::int32_t>::min()) ||
          parsed > static_cast<long long>(std::numeric_limits<std::int32_t>::max())) {
        error_message = "Token value is out of int32 range: " + trimmed;
        return false;
      }
      out_tokens.push_back(static_cast<std::int32_t>(parsed));
    } catch (...) {
      error_message = "Invalid token value in CSV list: " + trimmed;
      return false;
    }
  }

  if (out_tokens.empty()) {
    error_message = "Prompt token list is empty.";
    return false;
  }

  return true;
}

bool run_luce_qwen35_inference(
  const ReferenceInferenceOptions & options,
  ReferenceInferenceResult & result,
  std::string & error_message) {
  if (options.sampling.temperature > 0.0f) {
    error_message = "Luce decode backend currently supports only greedy decode (temperature <= 0).";
    return false;
  }
  if (options.gpu_decode_blocks < 0) {
    error_message = "gpu_decode_blocks must be >= 0.";
    return false;
  }

  const auto load_start = std::chrono::steady_clock::now();
  luce::LuceDecodeBackend backend;
  luce::LuceDecodeBackendConfig config;
  config.model_dir = options.model_dir;
  config.max_context = options.max_context;
  config.decode_blocks = options.gpu_decode_blocks;
  config.repetition_penalty = options.sampling.repetition_penalty;
  if (!backend.initialize(config, error_message)) {
    return false;
  }
  if (!backend.reset(error_message)) {
    return false;
  }
  const auto load_end = std::chrono::steady_clock::now();
  result.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

  std::unordered_set<std::int32_t> stop_token_set;
  stop_token_set.reserve(options.stop_token_ids.size());
  for (const std::int32_t token : options.stop_token_ids) {
    stop_token_set.insert(token);
  }

  DecodeProfilingAccumulator profiling;
  profiling.forward_pass_tokens = static_cast<int>(options.prompt_tokens.size());

  int first_token = 0;
  const auto prefill_start = std::chrono::steady_clock::now();
  if (options.luce_prefill_mode == LucePrefillMode::batched) {
    if (options.prefill_only) {
      if (!backend.run_prefill_only(options.prompt_tokens, error_message)) {
        return false;
      }
    } else if (!backend.run_prefill(options.prompt_tokens, first_token, error_message)) {
      return false;
    }
  } else {
    if (options.prefill_only) {
      error_message = "prefill_only is only supported with Luce batched prefill.";
      return false;
    }
    int prefill_position = 0;
    for (const std::int32_t prompt_token : options.prompt_tokens) {
      if (!backend.run_decode_step(prompt_token, prefill_position, first_token, error_message)) {
        return false;
      }
      ++prefill_position;
    }
  }
  const auto prefill_end = std::chrono::steady_clock::now();
  result.prefill_time_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
  result.prefill_tokens_per_second =
    (result.prefill_time_ms > 0.0)
      ? (static_cast<double>(options.prompt_tokens.size()) * 1000.0 / result.prefill_time_ms)
      : 0.0;

  result.generated_tokens.clear();
  if (options.prefill_only) {
    result.decode_time_ms = 0.0;
    result.tokens_per_second = 0.0;
    result.forward_pass_tokens = profiling.forward_pass_tokens;
    return true;
  }

  result.generated_tokens.reserve(static_cast<std::size_t>(options.max_new_tokens));

  const auto decode_start = std::chrono::steady_clock::now();
  int current = first_token;
  int position = static_cast<int>(options.prompt_tokens.size());
  for (int i = 0; i < options.max_new_tokens; ++i) {
    result.generated_tokens.push_back(current);

    const auto stop_checks_start = std::chrono::steady_clock::now();
    bool should_stop = false;
    std::size_t trim_count = 0;
    if (stop_token_set.find(current) != stop_token_set.end()) {
      should_stop = true;
      trim_count = std::max<std::size_t>(trim_count, 1);
    }
    for (const auto & stop_sequence : options.stop_token_sequences) {
      if (generated_ends_with_sequence(result.generated_tokens, stop_sequence)) {
        should_stop = true;
        trim_count = std::max(trim_count, stop_sequence.size());
      }
    }
    profiling.stop_checks_ms += elapsed_ms(stop_checks_start);
    if (should_stop) {
      if (trim_count > 0 && trim_count <= result.generated_tokens.size()) {
        result.generated_tokens.resize(result.generated_tokens.size() - trim_count);
      }
      break;
    }

    if (i + 1 >= options.max_new_tokens) {
      break;
    }

    int next = 0;
    if (!backend.run_decode_step(current, position, next, error_message)) {
      return false;
    }
    ++profiling.forward_pass_tokens;
    current = next;
    ++position;
  }

  const auto decode_end = std::chrono::steady_clock::now();
  result.decode_time_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
  result.tokens_per_second =
    (result.decode_time_ms > 0.0)
      ? (static_cast<double>(result.generated_tokens.size()) * 1000.0 / result.decode_time_ms)
      : 0.0;
  result.forward_pass_tokens = profiling.forward_pass_tokens;
  result.timing_breakdown.stop_checks_ms = profiling.stop_checks_ms;
  return true;
}

bool run_reference_qwen35_inference(
  const ModelProfile & profile,
  const ReferenceInferenceOptions & options,
  ReferenceInferenceResult & result,
  std::string & error_message) {
  result = ReferenceInferenceResult{};

  if (profile.family != "qwen3.5") {
    error_message = "Reference inference path currently supports only qwen3.5 family.";
    return false;
  }
  if (options.model_dir.empty()) {
    error_message = "Reference inference requires --hf-model-dir.";
    return false;
  }
  if (options.prompt_tokens.empty()) {
    error_message = "Reference inference requires non-empty prompt token list.";
    return false;
  }
  if (options.max_new_tokens < 0 || (!options.prefill_only && options.max_new_tokens == 0)) {
    error_message = "max_new_tokens must be > 0.";
    return false;
  }
  if (options.sampling.top_p <= 0.0f || options.sampling.top_p > 1.0f) {
    error_message = "top_p must be in (0, 1].";
    return false;
  }
  if (options.sampling.top_k < 0) {
    error_message = "top_k must be >= 0.";
    return false;
  }
  if (options.sampling.temperature < 0.0f) {
    error_message = "temperature must be >= 0.";
    return false;
  }
  if (options.sampling.repetition_penalty < 1.0f) {
    error_message = "repeat_penalty must be >= 1.0.";
    return false;
  }
  if (options.sampling.seed < -1) {
    error_message = "seed must be >= -1.";
    return false;
  }

#if !QWEN35X_HAS_CUDA
  if (options.use_cuda) {
    error_message = "GPU inference requested but this build has CUDA disabled.";
    return false;
  }
#endif

  RuntimeDims dims;
  if (!build_runtime_dims(profile, dims, error_message)) {
    return false;
  }

  if (options.max_context <= 0) {
    error_message = "max_context must be > 0.";
    return false;
  }
  const int required_context =
    static_cast<int>(options.prompt_tokens.size()) + (options.prefill_only ? 0 : options.max_new_tokens);
  if (required_context > options.max_context) {
    error_message = "prompt length + max_new_tokens exceeds max_context.";
    return false;
  }

  if (options.use_cuda && options.gpu_decode_backend == GpuDecodeBackend::luce) {
    return run_luce_qwen35_inference(options, result, error_message);
  }

  const auto load_start = std::chrono::steady_clock::now();
  const bool use_cuda_matvec_bf16 = options.use_cuda && options.use_cuda_matvec_bf16;

  ModelWeights weights;
  if (!load_model_weights(options.model_dir, dims, profile, weights, error_message)) {
    return false;
  }
  if (options.use_cuda && !upload_model_weights_to_cuda(weights, use_cuda_matvec_bf16, error_message)) {
    release_model_weights_cuda(weights);
    return false;
  }

  ModelState state;
  int full_layers = 0;
  int linear_layers = 0;
  for (const auto block : profile.fingerprint.attention_schedule) {
    if (block == AttentionBlock::linear) {
      ++linear_layers;
    } else {
      ++full_layers;
    }
  }

  state.full_states.resize(static_cast<std::size_t>(full_layers));
  for (auto & fs : state.full_states) {
    fs.k_cache.resize(
      static_cast<std::size_t>(options.max_context) * static_cast<std::size_t>(dims.n_kv_heads) *
      static_cast<std::size_t>(dims.head_dim));
    fs.v_cache.resize(
      static_cast<std::size_t>(options.max_context) * static_cast<std::size_t>(dims.n_kv_heads) *
      static_cast<std::size_t>(dims.head_dim));
    if (options.use_cuda) {
      if (!cuda::allocate_buffer_f32(fs.k_cache.size(), fs.k_cache_device, error_message) ||
          !cuda::allocate_buffer_f32(fs.v_cache.size(), fs.v_cache_device, error_message)) {
        release_model_state_cuda(state);
        release_model_weights_cuda(weights);
        return false;
      }
      fs.has_device_state = true;
    }
  }

  state.linear_states.resize(static_cast<std::size_t>(linear_layers));
  const int conv_hist = dims.linear_kernel - 1;
  for (auto & ls : state.linear_states) {
    ls.conv_state.resize(static_cast<std::size_t>(conv_hist * dims.linear_conv_channels), 0.0f);
    ls.recurrent_state.resize(
      static_cast<std::size_t>(dims.linear_num_v_heads) * static_cast<std::size_t>(dims.linear_head_v_dim) *
      static_cast<std::size_t>(dims.linear_head_v_dim),
      0.0f);
    if (options.use_cuda) {
      if (!cuda::allocate_buffer_f32(ls.conv_state.size(), ls.conv_state_device, error_message) ||
          !cuda::allocate_buffer_f32(ls.recurrent_state.size(), ls.recurrent_state_device, error_message)) {
        release_model_state_cuda(state);
        release_model_weights_cuda(weights);
        return false;
      }
      ls.has_device_state = true;
    }
  }

  CudaForwardWorkspace cuda_forward_workspace;
  CudaForwardWorkspace * cuda_workspace_ptr = nullptr;
  if (options.use_cuda) {
    const std::size_t max_input_count = std::max<std::size_t>({
      static_cast<std::size_t>(dims.hidden),
      static_cast<std::size_t>(dims.intermediate),
      static_cast<std::size_t>(dims.linear_conv_channels),
      static_cast<std::size_t>(dims.linear_v_dim),
      static_cast<std::size_t>(dims.linear_q_dim),
      static_cast<std::size_t>(dims.n_heads * dims.head_dim),
      static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim),
      1u
    });
    const std::size_t max_output_count = std::max<std::size_t>({
      static_cast<std::size_t>(dims.hidden),
      static_cast<std::size_t>(dims.intermediate),
      static_cast<std::size_t>(dims.vocab_size),
      static_cast<std::size_t>(dims.linear_conv_channels),
      static_cast<std::size_t>(dims.linear_v_dim),
      static_cast<std::size_t>(dims.n_heads * dims.head_dim * 2),
      static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim),
      1u
    });
    if (!cuda::begin_inference_session(max_input_count, max_output_count, error_message)) {
      release_model_state_cuda(state);
      release_model_weights_cuda(weights);
      return false;
    }
    cuda::set_prefer_bf16_matvec(use_cuda_matvec_bf16);
    if (!allocate_forward_workspace_cuda(dims, options.max_context, cuda_forward_workspace, error_message)) {
      cuda::end_inference_session();
      release_model_state_cuda(state);
      release_model_weights_cuda(weights);
      return false;
    }
    cuda_workspace_ptr = &cuda_forward_workspace;
  }

  RuntimeDecodeBackend decode_backend;
  auto release_cuda_resources = [&]() {
    release_runtime_decode_backend(decode_backend);
    if (options.use_cuda) {
      release_forward_workspace_cuda(cuda_forward_workspace);
      cuda::set_prefer_bf16_matvec(false);
      cuda::end_inference_session();
      release_model_state_cuda(state);
      release_model_weights_cuda(weights);
    }
  };

  const auto load_end = std::chrono::steady_clock::now();
  result.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

  if (options.use_cuda) {
    cuda::reset_transfer_stats();
  }

  std::mt19937 rng;
  if (options.sampling.seed >= 0) {
    rng.seed(static_cast<std::mt19937::result_type>(options.sampling.seed));
  } else {
    std::random_device random_device;
    rng.seed(random_device());
  }

  std::vector<int> token_counts(static_cast<std::size_t>(dims.vocab_size), 0);
  for (const std::int32_t token : options.prompt_tokens) {
    if (token >= 0 && token < dims.vocab_size) {
      token_counts[static_cast<std::size_t>(token)] += 1;
    }
  }
  std::unordered_set<std::int32_t> stop_token_set;
  stop_token_set.reserve(options.stop_token_ids.size());
  for (const std::int32_t token : options.stop_token_ids) {
    if (token >= 0 && token < dims.vocab_size) {
      stop_token_set.insert(token);
    }
  }

  const bool gpu_sampling_supported_by_config =
    options.sampling.temperature >= 0.0f && options.sampling.top_p > 0.0f && options.sampling.top_p <= 1.0f &&
    options.sampling.repetition_penalty > 0.0f &&
    ((options.sampling.temperature <= 0.0f) ||
     (options.sampling.top_k > 0 && options.sampling.top_k <= kCudaSamplingMaxTopK));
  const bool use_cuda_gpu_sampling =
    options.use_cuda && cuda_workspace_ptr != nullptr && cuda_workspace_ptr->has_gpu_sampling_buffers &&
    gpu_sampling_supported_by_config;
  if (options.use_cuda && !use_cuda_gpu_sampling) {
    error_message = "GPU sampling requires temperature >= 0, top_p in (0,1], repetition_penalty > 0, and top_k in [1, 64] when temperature > 0.";
    release_cuda_resources();
    return false;
  }
  if (!init_runtime_decode_backend(decode_backend, options, error_message) ||
      !reset_runtime_decode_backend(decode_backend, error_message)) {
    release_cuda_resources();
    return false;
  }
  if (use_cuda_gpu_sampling) {
    std::vector<float> seen_mask(static_cast<std::size_t>(dims.vocab_size), 0.0f);
    for (int token = 0; token < dims.vocab_size; ++token) {
      if (token_counts[static_cast<std::size_t>(token)] > 0) {
        seen_mask[static_cast<std::size_t>(token)] = 1.0f;
      }
    }
    if (!cuda::upload_to_buffer_f32(
          seen_mask.data(),
          seen_mask.size(),
          cuda_workspace_ptr->seen_token_mask,
          0,
          error_message)) {
      release_cuda_resources();
      return false;
    }
  }

  cuda::CudaDeviceBufferF32 sampled_token_device;
  cuda::CudaDeviceBufferF32 generated_tokens_device;
  auto release_cuda_decode_buffers = [&]() {
    cuda::free_buffer_f32(sampled_token_device);
    cuda::free_buffer_f32(generated_tokens_device);
  };

  DecodeProfilingAccumulator profiling;
  std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

  int position = 0;
  std::vector<float> predicted_logits;
  const auto prefill_start = std::chrono::steady_clock::now();
  for (std::size_t prompt_index = 0; prompt_index < options.prompt_tokens.size(); ++prompt_index) {
    const std::int32_t prompt_token = options.prompt_tokens[prompt_index];
    const bool compute_next_logits = !options.prefill_only && (prompt_index + 1 == options.prompt_tokens.size());
    if (!decode_step_with_runtime_backend(
          decode_backend,
          weights,
          dims,
          state,
          prompt_token,
          position,
          options.use_cuda,
          options.profile_cuda_sync,
          predicted_logits,
          use_cuda_gpu_sampling,
          compute_next_logits,
          cuda_workspace_ptr,
          &profiling,
          error_message)) {
      release_cuda_resources();
      return false;
    }
    ++position;
  }
  const auto prefill_end = std::chrono::steady_clock::now();
  result.prefill_time_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
  result.prefill_tokens_per_second =
    (result.prefill_time_ms > 0.0)
      ? (static_cast<double>(options.prompt_tokens.size()) * 1000.0 / result.prefill_time_ms)
      : 0.0;

  result.generated_tokens.clear();
  if (options.prefill_only) {
    result.decode_time_ms = 0.0;
    result.tokens_per_second = 0.0;
    result.forward_pass_tokens = profiling.forward_pass_tokens;
    result.timing_breakdown.embedding_ms = profiling.embedding_ms;
    result.timing_breakdown.attention_ms = profiling.attention_ms;
    result.timing_breakdown.mlp_ms = profiling.mlp_ms;
    result.timing_breakdown.logits_ms = profiling.logits_ms;
    result.timing_breakdown.sampling_ms = profiling.sampling_ms;
    result.timing_breakdown.stop_checks_ms = profiling.stop_checks_ms;
    release_cuda_decode_buffers();
    if (options.use_cuda) {
      release_cuda_resources();
    }
    return true;
  }

  result.generated_tokens.reserve(static_cast<std::size_t>(options.max_new_tokens));
  const auto decode_start = std::chrono::steady_clock::now();
  if (use_cuda_gpu_sampling) {
    const bool defer_stop_checks = stop_token_set.empty() && options.stop_token_sequences.empty();
    if (defer_stop_checks) {
      if (options.max_new_tokens > 0) {
        if (!cuda::allocate_buffer_f32(1, sampled_token_device, error_message) ||
            !cuda::allocate_buffer_f32(
              static_cast<std::size_t>(options.max_new_tokens),
              generated_tokens_device,
              error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
      }

      for (int i = 0; i < options.max_new_tokens; ++i) {
        if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        const auto sampling_start = std::chrono::steady_clock::now();
        const float random_u01 = options.sampling.temperature > 0.0f ? uniform01(rng) : 0.0f;
        if (!cuda::sample_token_from_logits_f32_device_to_buffer(
              cuda_workspace_ptr->logits,
              cuda_workspace_ptr->seen_token_mask,
              dims.vocab_size,
              options.sampling.temperature,
              options.sampling.top_p,
              options.sampling.top_k,
              options.sampling.repetition_penalty,
              random_u01,
              sampled_token_device,
              error_message) ||
            !cuda::copy_buffer_f32(
              sampled_token_device,
              1,
              0,
              generated_tokens_device,
              static_cast<std::size_t>(i),
              error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        profiling.sampling_ms += elapsed_ms(sampling_start);

        if (i + 1 < options.max_new_tokens) {
          ++profiling.forward_pass_tokens;
          if (!decode_step_with_runtime_backend_from_device_token(
                decode_backend,
                weights,
                dims,
                state,
                sampled_token_device,
                position,
                true,
                true,
                options.profile_cuda_sync,
                *cuda_workspace_ptr,
                predicted_logits,
                &profiling,
                error_message)) {
            release_cuda_decode_buffers();
            release_cuda_resources();
            return false;
          }
          ++position;
        }
      }

      std::vector<float> generated_token_values;
      if (options.max_new_tokens > 0 &&
          !cuda::download_from_buffer_f32(
            generated_tokens_device,
            static_cast<std::size_t>(options.max_new_tokens),
            0,
            generated_token_values,
            error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }

      for (int i = 0; i < options.max_new_tokens; ++i) {
        const int current = static_cast<int>(generated_token_values[static_cast<std::size_t>(i)]);
        if (current < 0 || current >= dims.vocab_size) {
          error_message = "CUDA sampling produced an out-of-range token id.";
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        result.generated_tokens.push_back(current);
      }
    } else {
      for (int i = 0; i < options.max_new_tokens; ++i) {
        if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        const auto sampling_start = std::chrono::steady_clock::now();
        int current = 0;
        const float random_u01 = options.sampling.temperature > 0.0f ? uniform01(rng) : 0.0f;
        if (!cuda::sample_token_from_logits_f32_device(
              cuda_workspace_ptr->logits,
              cuda_workspace_ptr->seen_token_mask,
              dims.vocab_size,
              options.sampling.temperature,
              options.sampling.top_p,
              options.sampling.top_k,
              options.sampling.repetition_penalty,
              random_u01,
              cuda_workspace_ptr->topk_values_scratch,
              cuda_workspace_ptr->topk_indices_scratch,
              current,
              error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        profiling.sampling_ms += elapsed_ms(sampling_start);

        result.generated_tokens.push_back(current);

        const auto stop_checks_start = std::chrono::steady_clock::now();
        bool should_stop = false;
        std::size_t trim_count = 0;
        if (stop_token_set.find(current) != stop_token_set.end()) {
          should_stop = true;
          trim_count = std::max<std::size_t>(trim_count, 1);
        }
        for (const auto & stop_sequence : options.stop_token_sequences) {
          if (generated_ends_with_sequence(result.generated_tokens, stop_sequence)) {
            should_stop = true;
            trim_count = std::max(trim_count, stop_sequence.size());
          }
        }
        profiling.stop_checks_ms += elapsed_ms(stop_checks_start);
        if (should_stop) {
          if (trim_count > 0 && trim_count <= result.generated_tokens.size()) {
            result.generated_tokens.resize(result.generated_tokens.size() - trim_count);
          }
          break;
        }

        if (!decode_step_with_runtime_backend(
              decode_backend,
              weights,
              dims,
              state,
              current,
              position,
              options.use_cuda,
              options.profile_cuda_sync,
              predicted_logits,
              true,
              true,
              cuda_workspace_ptr,
              &profiling,
              error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        ++position;
      }
    }
  } else {
    for (int i = 0; i < options.max_new_tokens; ++i) {
      if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }
      const auto sampling_start = std::chrono::steady_clock::now();
      int current = 0;
      if (!sample_token_from_logits(
            predicted_logits,
            options.sampling,
            token_counts,
            rng,
            current,
            error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }
      if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }
      profiling.sampling_ms += elapsed_ms(sampling_start);

      result.generated_tokens.push_back(current);
      if (current >= 0 && current < dims.vocab_size) {
        token_counts[static_cast<std::size_t>(current)] += 1;
      }

      const auto stop_checks_start = std::chrono::steady_clock::now();
      bool should_stop = false;
      std::size_t trim_count = 0;
      if (stop_token_set.find(current) != stop_token_set.end()) {
        should_stop = true;
        trim_count = std::max<std::size_t>(trim_count, 1);
      }
      for (const auto & stop_sequence : options.stop_token_sequences) {
        if (generated_ends_with_sequence(result.generated_tokens, stop_sequence)) {
          should_stop = true;
          trim_count = std::max(trim_count, stop_sequence.size());
        }
      }
      profiling.stop_checks_ms += elapsed_ms(stop_checks_start);
      if (should_stop) {
        if (trim_count > 0 && trim_count <= result.generated_tokens.size()) {
          result.generated_tokens.resize(result.generated_tokens.size() - trim_count);
        }
        break;
      }

      if (!decode_step_with_runtime_backend(
            decode_backend,
            weights,
            dims,
            state,
            current,
            position,
            options.use_cuda,
            options.profile_cuda_sync,
            predicted_logits,
            false,
            true,
            cuda_workspace_ptr,
            &profiling,
            error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }
      ++position;
    }
  }

  const auto decode_end = std::chrono::steady_clock::now();
  result.decode_time_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
  result.tokens_per_second =
    (result.decode_time_ms > 0.0)
      ? (static_cast<double>(result.generated_tokens.size()) * 1000.0 / result.decode_time_ms)
      : 0.0;
  result.forward_pass_tokens = profiling.forward_pass_tokens;
  result.timing_breakdown.embedding_ms = profiling.embedding_ms;
  result.timing_breakdown.attention_ms = profiling.attention_ms;
  result.timing_breakdown.mlp_ms = profiling.mlp_ms;
  result.timing_breakdown.logits_ms = profiling.logits_ms;
  result.timing_breakdown.sampling_ms = profiling.sampling_ms;
  result.timing_breakdown.stop_checks_ms = profiling.stop_checks_ms;
  if (options.use_cuda) {
    cuda::CudaTransferStats transfer_stats;
    cuda::get_transfer_stats(transfer_stats);
    result.transfer_breakdown.host_to_device_bytes = transfer_stats.host_to_device_bytes;
    result.transfer_breakdown.device_to_host_bytes = transfer_stats.device_to_host_bytes;
    result.transfer_breakdown.other_bytes = transfer_stats.other_bytes;
    result.transfer_breakdown.copy_calls = transfer_stats.copy_calls;
  }
  if (result.forward_pass_tokens > 0) {
    const double forward_tokens = static_cast<double>(result.forward_pass_tokens);
    result.host_to_device_bytes_per_forward_token =
      static_cast<double>(result.transfer_breakdown.host_to_device_bytes) / forward_tokens;
    result.device_to_host_bytes_per_forward_token =
      static_cast<double>(result.transfer_breakdown.device_to_host_bytes) / forward_tokens;
  }

  release_cuda_decode_buffers();
  release_cuda_resources();
  return true;
}

