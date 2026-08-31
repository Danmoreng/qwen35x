namespace {

struct CpuModelSessionState {
  mutable std::mutex mutex;
  std::string model_signature;
  ModelWeights weights;
  bool loaded = false;
};

[[nodiscard]] std::string cpu_model_session_signature(
  const ReferenceInferenceOptions & options,
  const ModelProfile & profile,
  const RuntimeDims & dims) {
  return cpu_prefix_model_signature(options, profile, dims) +
    "|backend=" + cpu::q8_0_backend_name(
      cpu::q8_0_resolve_backend(options.cpu_q8_backend)) +
    "|threads=" + std::to_string(options.cpu_threads);
}

} // namespace

ReferenceCpuModelSession::ReferenceCpuModelSession()
  : implementation_(std::make_shared<CpuModelSessionState>()) {}

void ReferenceCpuModelSession::clear() {
  if (implementation_ == nullptr) {
    implementation_ = std::make_shared<CpuModelSessionState>();
    return;
  }
  const auto state = std::static_pointer_cast<CpuModelSessionState>(implementation_);
  const std::lock_guard<std::mutex> lock(state->mutex);
  state->weights = ModelWeights{};
  state->model_signature.clear();
  state->loaded = false;
}

bool ReferenceCpuModelSession::empty() const {
  if (implementation_ == nullptr) {
    return true;
  }
  const auto state = std::static_pointer_cast<CpuModelSessionState>(implementation_);
  const std::lock_guard<std::mutex> lock(state->mutex);
  return !state->loaded;
}

namespace {

bool ensure_cpu_model_session_locked(
  const ModelProfile & profile,
  const ReferenceInferenceOptions & options,
  const RuntimeDims & dims,
  CpuModelSessionState & session,
  bool & cache_hit,
  std::string & error_message) {
  const std::string signature = cpu_model_session_signature(options, profile, dims);
  if (session.loaded && session.model_signature == signature) {
    cache_hit = true;
    return true;
  }

  ModelWeights candidate;
  if (!load_model_weights(
        options.model_dir,
        options.cpu_gguf_path,
        options.cpu_q4_h128_path,
        dims,
        profile,
        options.cpu_q8_backend,
        options.cpu_threads,
        candidate,
        error_message)) {
    return false;
  }

  session.weights = std::move(candidate);
  session.model_signature = signature;
  session.loaded = true;
  cache_hit = false;
  return true;
}

} // namespace

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

bool prepare_reference_cpu_model_session(
  const ModelProfile & profile,
  const ReferenceInferenceOptions & options,
  ReferenceCpuModelSession & session,
  std::string & error_message) {
  if (profile.family != "qwen3.5") {
    error_message = "Reference CPU model sessions currently support only qwen3.5 family.";
    return false;
  }
  if (options.model_dir.empty()) {
    error_message = "Reference CPU model session requires --hf-model-dir.";
    return false;
  }
  if (options.use_cuda) {
    error_message = "Reference CPU model sessions cannot be used with GPU inference.";
    return false;
  }
  if (options.cpu_threads < 0) {
    error_message = "cpu_threads must be >= 0 (zero selects the automatic default).";
    return false;
  }

  RuntimeDims dims;
  if (!build_runtime_dims(profile, dims, error_message)) {
    return false;
  }
  if (session.implementation_ == nullptr) {
    session.implementation_ = std::make_shared<CpuModelSessionState>();
  }
  const auto state = std::static_pointer_cast<CpuModelSessionState>(session.implementation_);
  const std::lock_guard<std::mutex> lock(state->mutex);
  bool cache_hit = false;
  return ensure_cpu_model_session_locked(
    profile, options, dims, *state, cache_hit, error_message);
}

bool run_qwen35x_cuda_inference(
  const ReferenceInferenceOptions & options,
  ReferenceInferenceResult & result,
  std::string & error_message) {
  if (options.sampling.temperature > 0.0f) {
    error_message = "Qwen35x CUDA backend currently supports only greedy decode (temperature <= 0).";
    return false;
  }
  if (options.gpu_decode_blocks < 0) {
    error_message = "gpu_decode_blocks must be >= 0.";
    return false;
  }

  const auto load_start = std::chrono::steady_clock::now();
  cuda_backend::Qwen35xCudaBackend backend;
  cuda_backend::Qwen35xCudaBackendConfig config;
  config.model_dir = options.model_dir;
  config.max_context = options.max_context;
  config.decode_blocks = options.gpu_decode_blocks;
  config.repetition_penalty = options.sampling.repetition_penalty;
  config.weight_precision = options.qwen35x_weight_precision;
  config.cache_precision = options.qwen35x_cache_precision;
  config.profile_enabled = options.profile_qwen35x;
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
  if (options.qwen35x_prefill_mode == Qwen35xPrefillMode::batched) {
    if (options.prefill_only) {
      if (!backend.run_prefill_only(options.prompt_tokens, error_message)) {
        return false;
      }
    } else if (!backend.run_prefill(options.prompt_tokens, first_token, error_message)) {
      return false;
    }
  } else {
    if (options.prefill_only) {
      error_message = "prefill_only is only supported with Qwen35x batched prefill.";
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
    result.qwen35x_profile = backend.profile();
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
  result.qwen35x_profile = backend.profile();
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
  if (options.cpu_threads < 0) {
    error_message = "cpu_threads must be >= 0 (zero selects the automatic default).";
    return false;
  }
  if (options.cpu_prefix_token_count > 0 && options.cpu_prefix_cache == nullptr) {
    error_message = "cpu_prefix_token_count requires a CPU prefix-cache handle.";
    return false;
  }
  if (options.cpu_prefix_token_count >= options.prompt_tokens.size() &&
      options.cpu_prefix_token_count > 0) {
    error_message = "CPU prefix cache must leave at least one uncached prompt token.";
    return false;
  }
  if (options.use_cuda && options.cpu_prefix_token_count > 0) {
    error_message = "The in-memory prefix cache currently supports CPU inference only.";
    return false;
  }
  if (options.use_cuda && options.cpu_model_session != nullptr) {
    error_message = "Persistent CPU model sessions cannot be used with GPU inference.";
    return false;
  }
  if ((options.logits_callback != nullptr || !options.forced_output_tokens.empty()) &&
      options.use_cuda) {
    error_message = "Teacher-forced logit observation currently supports CPU inference only.";
    return false;
  }
  if (options.logits_callback != nullptr && options.prefill_only) {
    error_message = "Logit observation cannot be combined with prefill-only inference.";
    return false;
  }
  if (!options.forced_output_tokens.empty() &&
      options.forced_output_tokens.size() !=
        static_cast<std::size_t>(options.max_new_tokens)) {
    error_message = "forced_output_tokens must contain exactly max_new_tokens entries.";
    return false;
  }
  if (!options.forced_output_tokens.empty() &&
      (!options.stop_token_ids.empty() || !options.stop_token_sequences.empty())) {
    error_message = "Teacher-forced inference cannot be combined with stop conditions.";
    return false;
  }
  if (!options.cpu_gguf_path.empty() && !options.cpu_q4_h128_path.empty()) {
    error_message = "--cpu-gguf and --cpu-q4-h128 are mutually exclusive.";
    return false;
  }
  if (options.use_cuda &&
      (!options.cpu_gguf_path.empty() || !options.cpu_q4_h128_path.empty())) {
    error_message = "CPU weight artifacts cannot be combined with GPU inference.";
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
  for (const std::int32_t token : options.forced_output_tokens) {
    if (token < 0 || token >= dims.vocab_size) {
      error_message = "Teacher-forced output token is outside the model vocabulary.";
      return false;
    }
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

  if (options.use_cuda && options.gpu_decode_backend == GpuDecodeBackend::qwen35x_cuda) {
    return run_qwen35x_cuda_inference(options, result, error_message);
  }

  const auto load_start = std::chrono::steady_clock::now();
  const bool use_cuda_matvec_bf16 = options.use_cuda && options.use_cuda_matvec_bf16;

  ModelWeights request_weights;
  ModelWeights * weights_ptr = &request_weights;
  std::shared_ptr<CpuModelSessionState> cpu_model_session_state;
  std::unique_lock<std::mutex> cpu_model_session_lock;
  if (options.cpu_model_session != nullptr) {
    if (options.cpu_model_session->implementation_ == nullptr) {
      options.cpu_model_session->implementation_ =
        std::make_shared<CpuModelSessionState>();
    }
    cpu_model_session_state = std::static_pointer_cast<CpuModelSessionState>(
      options.cpu_model_session->implementation_);
    cpu_model_session_lock = std::unique_lock<std::mutex>(cpu_model_session_state->mutex);
    if (!ensure_cpu_model_session_locked(
          profile,
          options,
          dims,
          *cpu_model_session_state,
          result.cpu_model_session_hit,
          error_message)) {
      return false;
    }
    weights_ptr = &cpu_model_session_state->weights;
  } else if (!load_model_weights(
               options.model_dir,
               options.cpu_gguf_path,
               options.cpu_q4_h128_path,
               dims,
               profile,
               options.cpu_q8_backend,
               options.cpu_threads,
               request_weights,
               error_message)) {
    return false;
  }
  ModelWeights & weights = *weights_ptr;
  if (options.use_cuda && !upload_model_weights_to_cuda(weights, use_cuda_matvec_bf16, error_message)) {
    release_model_weights_cuda(weights);
    return false;
  }

  ModelState state;
  const std::size_t rope_half = static_cast<std::size_t>(dims.rope_dim / 2);
  state.rope_inverse_frequency.resize(rope_half);
  for (std::size_t index = 0; index < rope_half; ++index) {
    state.rope_inverse_frequency[index] = std::pow(
      dims.rope_theta,
      -static_cast<float>(2 * index) / static_cast<float>(dims.rope_dim));
  }
  state.rope_cosine.resize(static_cast<std::size_t>(required_context) * rope_half);
  state.rope_sine.resize(static_cast<std::size_t>(required_context) * rope_half);
  for (int position = 0; position < required_context; ++position) {
    const std::size_t position_offset = static_cast<std::size_t>(position) * rope_half;
    for (std::size_t index = 0; index < rope_half; ++index) {
      const float angle =
        static_cast<float>(position) * state.rope_inverse_frequency[index];
      state.rope_cosine[position_offset + index] = std::cos(angle);
      state.rope_sine[position_offset + index] = std::sin(angle);
    }
  }
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
    if (!options.use_cuda && weights.cpu_q8_runtime != nullptr) {
      fs.k_cache_f16.resize(fs.k_cache.size());
      fs.v_cache_f16.resize(fs.v_cache.size());
    }
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
  CpuGreedySamplingState cpu_greedy_sampling{
    &token_counts,
    options.sampling.repetition_penalty,
    -1,
    !options.use_cuda && options.sampling.temperature <= 1.0e-6F &&
      options.capture_top_logits == 0 &&
      options.logits_callback == nullptr &&
      weights.embed_tokens.is_q4_0() && weights.cpu_q8_runtime != nullptr &&
      weights.cpu_q8_runtime->executor != nullptr,
  };
  CpuGreedySamplingState * cpu_greedy_sampling_ptr =
    cpu_greedy_sampling.enabled ? &cpu_greedy_sampling : nullptr;
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
  result.top_logits_by_step.clear();
  auto capture_top_logits = [&](const int step, const int selected_token) {
    if (options.capture_top_logits <= 0 || predicted_logits.empty()) {
      return;
    }
    std::vector<float> adjusted_logits = predicted_logits;
    apply_repetition_penalty_inplace(
      adjusted_logits, token_counts, options.sampling.repetition_penalty);
    std::vector<std::int32_t> candidate_ids;
    candidate_ids.reserve(adjusted_logits.size());
    for (std::size_t token_id = 0; token_id < adjusted_logits.size(); ++token_id) {
      candidate_ids.push_back(static_cast<std::int32_t>(token_id));
    }
    const std::size_t count = std::min(
      static_cast<std::size_t>(options.capture_top_logits), candidate_ids.size());
    std::partial_sort(
      candidate_ids.begin(), candidate_ids.begin() + static_cast<std::ptrdiff_t>(count),
      candidate_ids.end(),
      [&](const std::int32_t lhs, const std::int32_t rhs) {
        const float lhs_value = adjusted_logits[static_cast<std::size_t>(lhs)];
        const float rhs_value = adjusted_logits[static_cast<std::size_t>(rhs)];
        return lhs_value != rhs_value ? lhs_value > rhs_value : lhs < rhs;
      });
    ReferenceTopLogitsStep snapshot;
    snapshot.step = step;
    snapshot.selected_token_id = selected_token;
    snapshot.top_logits.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      const std::int32_t token_id = candidate_ids[index];
      snapshot.top_logits.push_back(
        ReferenceTopLogit{token_id, adjusted_logits[static_cast<std::size_t>(token_id)]});
    }
    result.top_logits_by_step.push_back(std::move(snapshot));
  };
  const auto prefill_start = std::chrono::steady_clock::now();
  const std::size_t prefix_token_count = options.cpu_prefix_token_count;
  const std::size_t kv_width =
    static_cast<std::size_t>(dims.n_kv_heads) * static_cast<std::size_t>(dims.head_dim);
  const std::string prefix_model_signature = prefix_token_count > 0
    ? cpu_prefix_model_signature(options, profile, dims) : std::string{};
  bool prefix_cache_hit = false;
  if (prefix_token_count > 0 && options.cpu_prefix_cache != nullptr &&
      options.cpu_prefix_cache->implementation_ != nullptr) {
    const auto restore_start = std::chrono::steady_clock::now();
    const auto snapshot = std::static_pointer_cast<CpuPrefixCacheSnapshot>(
      options.cpu_prefix_cache->implementation_);
    const bool tokens_match = snapshot->prefix_tokens.size() == prefix_token_count &&
      std::equal(
        snapshot->prefix_tokens.begin(), snapshot->prefix_tokens.end(),
        options.prompt_tokens.begin());
    if (snapshot->state_abi_version == kCpuPrefixCacheStateAbiVersion &&
        snapshot->model_signature == prefix_model_signature && tokens_match &&
        snapshot->backend == cpu::q8_0_resolve_backend(options.cpu_q8_backend) &&
        snapshot->prefill_mode == options.qwen35x_prefill_mode &&
        snapshot->hidden == dims.hidden && snapshot->head_dim == dims.head_dim &&
        snapshot->kv_heads == dims.n_kv_heads &&
        restore_cpu_prefix_state(
          *snapshot, state, prefix_token_count, kv_width)) {
      position = static_cast<int>(prefix_token_count);
      prefix_cache_hit = true;
      result.cached_prefix_tokens = position;
      result.prefix_cache_restore_time_ms = elapsed_ms(restore_start);
      result.prefix_cache_bytes = cpu_prefix_cache_size_bytes(*snapshot);
    } else {
      options.cpu_prefix_cache->clear();
    }
  }
  auto capture_prefix_if_ready = [&]() {
    if (prefix_token_count == 0 || prefix_cache_hit ||
        position != static_cast<int>(prefix_token_count) ||
        options.cpu_prefix_cache == nullptr) {
      return;
    }
    auto snapshot = std::make_shared<CpuPrefixCacheSnapshot>();
    snapshot->model_signature = prefix_model_signature;
    snapshot->prefix_tokens.assign(
      options.prompt_tokens.begin(),
      options.prompt_tokens.begin() + static_cast<std::ptrdiff_t>(prefix_token_count));
    snapshot->backend = cpu::q8_0_resolve_backend(options.cpu_q8_backend);
    snapshot->prefill_mode = options.qwen35x_prefill_mode;
    snapshot->hidden = dims.hidden;
    snapshot->head_dim = dims.head_dim;
    snapshot->kv_heads = dims.n_kv_heads;
    capture_cpu_prefix_state(
      state, prefix_token_count, kv_width,
      snapshot->backend == cpu::Q8_0Backend::avx2, *snapshot);
    result.prefix_cache_bytes = cpu_prefix_cache_size_bytes(*snapshot);
    options.cpu_prefix_cache->implementation_ = std::move(snapshot);
  };
  const bool use_cpu_q8_batch_prefill =
    !options.use_cuda && weights.cpu_q8_runtime != nullptr &&
    options.qwen35x_prefill_mode == Qwen35xPrefillMode::batched;
  if (use_cpu_q8_batch_prefill) {
    constexpr std::size_t cpu_prefill_chunk_size = 64;
    for (std::size_t chunk_begin = static_cast<std::size_t>(position);
         chunk_begin < options.prompt_tokens.size();) {
      std::size_t chunk_size = std::min(
        cpu_prefill_chunk_size, options.prompt_tokens.size() - chunk_begin);
      if (!prefix_cache_hit && chunk_begin < prefix_token_count &&
          chunk_begin + chunk_size > prefix_token_count) {
        chunk_size = prefix_token_count - chunk_begin;
      }
      const bool compute_next_logits =
        !options.prefill_only && (chunk_begin + chunk_size == options.prompt_tokens.size());
      if (!run_forward_cpu_q8_batch(
            weights,
            dims,
            state,
            options.prompt_tokens.data() + chunk_begin,
            chunk_size,
            position,
            compute_next_logits,
            predicted_logits,
            cpu_greedy_sampling_ptr,
            &profiling,
            error_message)) {
        release_cuda_resources();
        return false;
      }
      position += static_cast<int>(chunk_size);
      chunk_begin += chunk_size;
      capture_prefix_if_ready();
    }
  } else {
    for (std::size_t prompt_index = static_cast<std::size_t>(position);
         prompt_index < options.prompt_tokens.size(); ++prompt_index) {
      const std::int32_t prompt_token = options.prompt_tokens[prompt_index];
      const bool compute_next_logits =
        !options.prefill_only && (prompt_index + 1 == options.prompt_tokens.size());
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
            cpu_greedy_sampling_ptr,
            &profiling,
            error_message)) {
        release_cuda_resources();
        return false;
      }
      ++position;
      capture_prefix_if_ready();
    }
  }
  const auto prefill_end = std::chrono::steady_clock::now();
  result.prefill_time_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
  result.prefill_tokens_per_second =
    (result.prefill_time_ms > 0.0)
      ? (static_cast<double>(options.prompt_tokens.size() -
          static_cast<std::size_t>(result.cached_prefix_tokens)) * 1000.0 /
          result.prefill_time_ms)
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

        if (i + 1 >= options.max_new_tokens) {
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
              cpu_greedy_sampling_ptr,
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
    auto observe_logits = [&](const int output_index) {
      if (options.logits_callback == nullptr) {
        return true;
      }
      if (predicted_logits.size() != static_cast<std::size_t>(dims.vocab_size)) {
        error_message = "Observed CPU logits do not match the model vocabulary size.";
        return false;
      }
      const std::int32_t target_token = options.forced_output_tokens.empty()
        ? -1
        : options.forced_output_tokens[static_cast<std::size_t>(output_index)];
      return options.logits_callback(
        options.logits_callback_context,
        static_cast<std::size_t>(output_index),
        target_token,
        predicted_logits.data(),
        predicted_logits.size(),
        error_message);
    };
    for (int i = 0; i < options.max_new_tokens; ++i) {
      if (!observe_logits(i)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }
      if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }
      const auto sampling_start = std::chrono::steady_clock::now();
      int current = 0;
      if (!options.forced_output_tokens.empty()) {
        current = options.forced_output_tokens[static_cast<std::size_t>(i)];
      } else if (cpu_greedy_sampling_ptr != nullptr) {
        current = cpu_greedy_sampling.next_token;
        cpu_greedy_sampling.next_token = -1;
        if (current < 0 || current >= dims.vocab_size) {
          error_message = "Fused greedy Q4 selection produced an invalid token.";
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
      } else {
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
      }
      if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }
      profiling.sampling_ms += elapsed_ms(sampling_start);

      capture_top_logits(i, current);

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


      if (i + 1 >= options.max_new_tokens) {
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
            cpu_greedy_sampling_ptr,
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
