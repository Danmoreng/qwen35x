namespace {

void rms_norm_qwen3next_batch(
  const std::vector<float> & input,
  const std::size_t batch_size,
  const std::size_t width,
  const TensorData & weight,
  const float eps,
  std::vector<float> & output) {
  output.resize(input.size());
  cpu::rms_norm_f32(
    input.data(), weight.data.data(), output.data(), batch_size, width,
    eps, 1.0F);
}

void apply_rope_from_tables_inplace(
  std::vector<float> & values,
  const int head_count,
  const int head_dim,
  const int rope_dim,
  const float * cosine,
  const float * sine) {
  const int half = rope_dim / 2;
  for (int head = 0; head < head_count; ++head) {
    const std::size_t base =
      static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
    for (int index = 0; index < half; ++index) {
      const std::size_t first = base + static_cast<std::size_t>(index);
      const std::size_t second = first + static_cast<std::size_t>(half);
      const float x0 = values[first];
      const float x1 = values[second];
      values[first] = x0 * cosine[index] - x1 * sine[index];
      values[second] = x1 * cosine[index] + x0 * sine[index];
    }
  }
}

struct GatedDeltaNetBatchCpuJob {
  float * state = nullptr;
  const float * q = nullptr;
  const float * k = nullptr;
  const float * v = nullptr;
  const float * alpha = nullptr;
  const float * beta = nullptr;
  float * output = nullptr;
  std::size_t batch_size = 0;
  std::size_t head_count = 0;
  std::size_t key_dim = 0;
  std::size_t value_dim = 0;
  cpu::Q8_0Backend backend = cpu::Q8_0Backend::auto_select;
};

struct CausalConvBatchCpuJob {
  float * state = nullptr;
  const float * input = nullptr;
  const float * weights = nullptr;
  float * output = nullptr;
  std::size_t batch_size = 0;
  std::size_t input_stride = 0;
  std::size_t channel_count = 0;
  int kernel_size = 0;
};

void run_causal_conv_batch_cpu_channels(
  void * opaque_context,
  const std::size_t channel_begin,
  const std::size_t channel_end) noexcept {
  auto & job = *static_cast<CausalConvBatchCpuJob *>(opaque_context);
  const int history = job.kernel_size - 1;
  for (std::size_t channel = channel_begin; channel < channel_end; ++channel) {
    const float * weight =
      job.weights + channel * static_cast<std::size_t>(job.kernel_size);
    for (std::size_t token = 0; token < job.batch_size; ++token) {
      const float input_value = job.input[token * job.input_stride + channel];
      float sum = input_value * weight[static_cast<std::size_t>(history)];
      for (int kernel = 0; kernel < history; ++kernel) {
        sum += job.state[static_cast<std::size_t>(kernel) * job.channel_count + channel] *
          weight[static_cast<std::size_t>(kernel)];
      }
      for (int kernel = 0; kernel + 1 < history; ++kernel) {
        job.state[static_cast<std::size_t>(kernel) * job.channel_count + channel] =
          job.state[static_cast<std::size_t>(kernel + 1) * job.channel_count + channel];
      }
      if (history > 0) {
        job.state[static_cast<std::size_t>(history - 1) * job.channel_count + channel] =
          input_value;
      }
      job.output[token * job.channel_count + channel] = sum;
    }
  }
}

void run_gated_delta_net_batch_cpu_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<GatedDeltaNetBatchCpuJob *>(opaque_context);
  cpu::gated_delta_net_update_batch_rows(
    job.state,
    job.q,
    job.k,
    job.v,
    job.alpha,
    job.beta,
    job.output,
    job.batch_size,
    job.head_count,
    job.key_dim,
    job.value_dim,
    row_begin,
    row_end,
    job.backend);
}

bool run_linear_attention_batch_cpu_q8(
  const LayerWeights & layer,
  const RuntimeDims & dims,
  LinearAttentionState & state,
  const std::vector<float> & input,
  const std::size_t batch_size,
  std::vector<float> & output,
  std::string & error_message) {
  if (!layer.linear.in_proj_all_cpu.is_q8_0() || !layer.linear.out_proj.is_q8_0()) {
    error_message = "Batched CPU linear attention requires packed Q8_0 projections.";
    return false;
  }

  const std::size_t conv_channels = static_cast<std::size_t>(dims.linear_conv_channels);
  const std::size_t value_width = static_cast<std::size_t>(dims.linear_v_dim);
  const std::size_t head_count = static_cast<std::size_t>(dims.linear_num_v_heads);
  const std::size_t key_dim = static_cast<std::size_t>(dims.linear_head_k_dim);
  const std::size_t value_dim = static_cast<std::size_t>(dims.linear_head_v_dim);
  const std::size_t q_width = static_cast<std::size_t>(dims.linear_q_dim);
  const std::size_t projection_width = conv_channels + value_width + 2 * head_count;
  const std::size_t expected_state = head_count * value_dim * key_dim;
  if (dims.linear_num_k_heads != dims.linear_num_v_heads || key_dim != value_dim ||
      state.recurrent_state.size() != expected_state ||
      state.conv_state.size() !=
        static_cast<std::size_t>(dims.linear_kernel - 1) * conv_channels) {
    error_message = "Batched CPU DeltaNet state dimensions are unsupported or inconsistent.";
    return false;
  }

  std::vector<float> projected;
  if (!matmul_2d_q8_batch(
        layer.linear.in_proj_all_cpu, input, batch_size, projected, error_message)) {
    return false;
  }
  if (projected.size() != batch_size * projection_width) {
    error_message = "Batched linear-attention projection output size mismatch.";
    return false;
  }

  std::vector<float> gated(batch_size * value_width);
  std::vector<float> conv_batch(batch_size * conv_channels);
  std::vector<float> q(q_width);
  std::vector<float> k(q_width);
  std::vector<float> v(value_width);
  std::vector<float> alpha(head_count);
  std::vector<float> beta(head_count);
  std::vector<float> q_batch(batch_size * q_width);
  std::vector<float> k_batch(batch_size * q_width);
  std::vector<float> v_batch(batch_size * value_width);
  std::vector<float> alpha_batch(batch_size * head_count);
  std::vector<float> beta_batch(batch_size * head_count);
  std::vector<float> core_batch(batch_size * value_width);
  const float q_scale = 1.0F / std::sqrt(static_cast<float>(dims.linear_head_k_dim));
  CpuQ8Runtime * const runtime = layer.linear.out_proj.q8_0_runtime;

  CausalConvBatchCpuJob conv_job{
    state.conv_state.data(),
    projected.data(),
    layer.linear.conv1d.data.data(),
    conv_batch.data(),
    batch_size,
    projection_width,
    conv_channels,
    dims.linear_kernel,
  };
  if (runtime != nullptr && runtime->executor != nullptr) {
    const cpu::CpuExecutorStatus status = runtime->executor->parallel_for_rows(
      conv_channels, run_causal_conv_batch_cpu_channels, &conv_job);
    if (status != cpu::CpuExecutorStatus::ok) {
      error_message = std::string("Batched causal-conv CPU executor failed: ") +
        cpu::cpu_executor_status_name(status) + ".";
      return false;
    }
  } else {
    run_causal_conv_batch_cpu_channels(&conv_job, 0, conv_channels);
  }
  cpu::silu_f32(
    conv_batch.data(), conv_batch.data(), conv_batch.size(),
    layer.linear.out_proj.q8_0_backend);

  for (std::size_t token = 0; token < batch_size; ++token) {
    const float * projection = projected.data() + token * projection_width;
    const float * mixed_qkv = projection;
    const float * z = mixed_qkv + conv_channels;
    const float * b = z + value_width;
    const float * a = b + head_count;

    for (std::size_t head = 0; head < head_count; ++head) {
      beta[head] = sigmoidf_stable(b[head]);
      const float pre_gate = softplusf_stable(a[head] + layer.linear.dt_bias.data[head]);
      alpha[head] = std::exp(pre_gate * layer.linear.ssm_a[head]);
    }

    const float * conv_out = conv_batch.data() + token * conv_channels;
    std::memcpy(q.data(), conv_out, q_width * sizeof(float));
    std::memcpy(k.data(), conv_out + q_width, q_width * sizeof(float));
    std::memcpy(v.data(), conv_out + 2 * q_width, value_width * sizeof(float));
    l2_norm_per_head(
      q, dims.linear_num_k_heads, dims.linear_head_k_dim, 1.0e-6F, q_scale);
    l2_norm_per_head(k, dims.linear_num_k_heads, dims.linear_head_k_dim);
    for (std::size_t head = 0; head < head_count; ++head) {
      const std::size_t head_token = head * batch_size + token;
      std::memcpy(
        q_batch.data() + head_token * key_dim,
        q.data() + head * key_dim,
        key_dim * sizeof(float));
      std::memcpy(
        k_batch.data() + head_token * key_dim,
        k.data() + head * key_dim,
        key_dim * sizeof(float));
      std::memcpy(
        v_batch.data() + head_token * value_dim,
        v.data() + head * value_dim,
        value_dim * sizeof(float));
      alpha_batch[head_token] = alpha[head];
      beta_batch[head_token] = beta[head];
    }
  }

  if (runtime != nullptr && runtime->executor != nullptr) {
    GatedDeltaNetBatchCpuJob job{
      state.recurrent_state.data(),
      q_batch.data(),
      k_batch.data(),
      v_batch.data(),
      alpha_batch.data(),
      beta_batch.data(),
      core_batch.data(),
      batch_size,
      head_count,
      key_dim,
      value_dim,
      layer.linear.out_proj.q8_0_backend,
    };
    const cpu::CpuExecutorStatus status = runtime->executor->parallel_for_rows(
      head_count, run_gated_delta_net_batch_cpu_rows, &job);
    if (status != cpu::CpuExecutorStatus::ok) {
      error_message = std::string("Batched DeltaNet CPU executor failed: ") +
        cpu::cpu_executor_status_name(status) + ".";
      return false;
    }
  } else {
    cpu::gated_delta_net_update_batch_rows(
      state.recurrent_state.data(),
      q_batch.data(),
      k_batch.data(),
      v_batch.data(),
      alpha_batch.data(),
      beta_batch.data(),
      core_batch.data(),
      batch_size,
      head_count,
      key_dim,
      value_dim,
      0,
      head_count,
      layer.linear.out_proj.q8_0_backend);
  }

  for (std::size_t token = 0; token < batch_size; ++token) {
    const float * projection = projected.data() + token * projection_width;
    const float * z = projection + conv_channels;
    const float * core_out = core_batch.data() + token * value_width;
    float * gated_token = gated.data() + token * value_width;
    cpu::rms_norm_f32(
      core_out, layer.linear.norm.data.data(), gated_token,
      head_count, value_dim, dims.rms_eps, 0.0F,
      layer.linear.out_proj.q8_0_backend);
    cpu::silu_mul_f32(
      z, gated_token, gated_token, value_width,
      layer.linear.out_proj.q8_0_backend);
  }

  return matmul_2d_q8_batch(
    layer.linear.out_proj, gated, batch_size, output, error_message);
}

bool run_full_attention_batch_cpu_q8(
  const LayerWeights & layer,
  const RuntimeDims & dims,
  FullAttentionState & state,
  const std::vector<float> & input,
  const std::size_t batch_size,
  const int position_start,
  std::vector<float> & output,
  std::string & error_message) {
  if (!layer.full.qkv_proj_cpu.is_q8_0() || !layer.full.o_proj.is_q8_0()) {
    error_message = "Batched CPU full attention requires packed Q8_0 projections.";
    return false;
  }
  const std::size_t query_width = static_cast<std::size_t>(dims.n_heads * dims.head_dim);
  const std::size_t q_full_width = 2 * query_width;
  const std::size_t kv_width = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  const std::size_t projection_width = q_full_width + 2 * kv_width;
  std::vector<float> projected;
  if (!matmul_2d_q8_batch(
        layer.full.qkv_proj_cpu, input, batch_size, projected, error_message)) {
    return false;
  }
  if (projected.size() != batch_size * projection_width) {
    error_message = "Batched full-attention projection output size mismatch.";
    return false;
  }

  std::vector<float> attention(batch_size * query_width);
  std::vector<float> query_batch(batch_size * query_width);
  std::vector<float> gate_batch(batch_size * query_width);
  std::vector<float> q(query_width);
  std::vector<float> gate(query_width);
  std::vector<float> k_flat(kv_width);
  std::vector<float> q_normed;
  std::vector<float> k_normed;
  const int q_span = 2 * dims.head_dim;
  const float attention_scale = 1.0F / std::sqrt(static_cast<float>(dims.head_dim));
  const int rope_half = dims.rope_dim / 2;
  std::vector<float> rope_inverse_frequency(static_cast<std::size_t>(rope_half));
  for (int index = 0; index < rope_half; ++index) {
    rope_inverse_frequency[static_cast<std::size_t>(index)] = std::pow(
      dims.rope_theta,
      -static_cast<float>(2 * index) / static_cast<float>(dims.rope_dim));
  }
  std::vector<float> rope_cosine(batch_size * static_cast<std::size_t>(rope_half));
  std::vector<float> rope_sine(batch_size * static_cast<std::size_t>(rope_half));
  for (std::size_t token = 0; token < batch_size; ++token) {
    const float position = static_cast<float>(position_start + static_cast<int>(token));
    for (int index = 0; index < rope_half; ++index) {
      const float angle =
        position * rope_inverse_frequency[static_cast<std::size_t>(index)];
      rope_cosine[token * static_cast<std::size_t>(rope_half) +
        static_cast<std::size_t>(index)] = std::cos(angle);
      rope_sine[token * static_cast<std::size_t>(rope_half) +
        static_cast<std::size_t>(index)] = std::sin(angle);
    }
  }

  for (std::size_t token = 0; token < batch_size; ++token) {
    const int position = position_start + static_cast<int>(token);
    const float * projection = projected.data() + token * projection_width;
    const float * q_full = projection;
    const float * k_source = q_full + q_full_width;
    const float * v_flat = k_source + kv_width;
    for (int head = 0; head < dims.n_heads; ++head) {
      const std::size_t source = static_cast<std::size_t>(head * q_span);
      const std::size_t destination = static_cast<std::size_t>(head * dims.head_dim);
      std::memcpy(
        q.data() + destination,
        q_full + source,
        static_cast<std::size_t>(dims.head_dim) * sizeof(float));
      std::memcpy(
        gate.data() + destination,
        q_full + source + static_cast<std::size_t>(dims.head_dim),
        static_cast<std::size_t>(dims.head_dim) * sizeof(float));
    }
    std::memcpy(k_flat.data(), k_source, kv_width * sizeof(float));
    rms_norm_per_head_qwen3next(
      q, dims.n_heads, dims.head_dim, layer.full.q_norm, dims.rms_eps, q_normed);
    rms_norm_per_head_qwen3next(
      k_flat, dims.n_kv_heads, dims.head_dim, layer.full.k_norm, dims.rms_eps, k_normed);
    const float * token_cosine =
      rope_cosine.data() + token * static_cast<std::size_t>(rope_half);
    const float * token_sine =
      rope_sine.data() + token * static_cast<std::size_t>(rope_half);
    apply_rope_from_tables_inplace(
      q_normed, dims.n_heads, dims.head_dim, dims.rope_dim, token_cosine, token_sine);
    apply_rope_from_tables_inplace(
      k_normed, dims.n_kv_heads, dims.head_dim, dims.rope_dim, token_cosine, token_sine);

    std::memcpy(
      query_batch.data() + token * query_width,
      q_normed.data(),
      query_width * sizeof(float));
    std::memcpy(
      gate_batch.data() + token * query_width,
      gate.data(),
      query_width * sizeof(float));

    std::memcpy(
      state.k_cache.data() + static_cast<std::size_t>(position) * kv_width,
      k_normed.data(),
      kv_width * sizeof(float));
    std::memcpy(
      state.v_cache.data() + static_cast<std::size_t>(position) * kv_width,
      v_flat,
      kv_width * sizeof(float));
    if (!state.k_cache_f16.empty() && !state.v_cache_f16.empty()) {
      cpu::attention_cache_store_f16(
        k_normed.data(),
        state.k_cache_f16.data() + static_cast<std::size_t>(position) * kv_width,
        kv_width,
        layer.full.o_proj.q8_0_backend);
      cpu::attention_cache_store_f16(
        v_flat,
        state.v_cache_f16.data() + static_cast<std::size_t>(position) * kv_width,
        kv_width,
        layer.full.o_proj.q8_0_backend);
    }
  }

  const std::size_t attention_rows =
    batch_size * static_cast<std::size_t>(dims.n_heads);
  const std::size_t context_stride =
    static_cast<std::size_t>(position_start) + batch_size;
  std::vector<float> scores(attention_rows * context_stride);
  FullAttentionBatchCpuJob job{
    query_batch.data(),
    gate_batch.data(),
    state.k_cache.data(),
    state.v_cache.data(),
    state.k_cache_f16.empty() ? nullptr : state.k_cache_f16.data(),
    state.v_cache_f16.empty() ? nullptr : state.v_cache_f16.data(),
    scores.data(),
    attention.data(),
    context_stride,
    query_width,
    kv_width,
    position_start,
    dims.n_heads,
    dims.n_kv_heads,
    dims.head_dim,
    attention_scale,
    layer.full.o_proj.q8_0_backend,
  };
  CpuQ8Runtime * const runtime = layer.full.o_proj.q8_0_runtime;
  if (runtime != nullptr && runtime->executor != nullptr) {
    const cpu::CpuExecutorStatus status = runtime->executor->parallel_for_rows(
      attention_rows, run_full_attention_batch_cpu_rows, &job);
    if (status != cpu::CpuExecutorStatus::ok) {
      error_message = std::string("Batched full-attention CPU executor failed: ") +
        cpu::cpu_executor_status_name(status) + ".";
      return false;
    }
  } else {
    run_full_attention_batch_cpu_rows(&job, 0, attention_rows);
  }

  return matmul_2d_q8_batch(
    layer.full.o_proj, attention, batch_size, output, error_message);
}

bool run_forward_cpu_q8_batch(
  const ModelWeights & weights,
  const RuntimeDims & dims,
  ModelState & state,
  const std::int32_t * token_ids,
  const std::size_t batch_size,
  const int position_start,
  const bool compute_next_logits,
  std::vector<float> & next_logits,
  DecodeProfilingAccumulator * profiling,
  std::string & error_message) {
  if (weights.cpu_q8_runtime == nullptr || token_ids == nullptr || batch_size == 0) {
    error_message = "Batched CPU Q8_0 forward received an invalid runtime or empty batch.";
    return false;
  }
  const std::size_t hidden = static_cast<std::size_t>(dims.hidden);
  const std::size_t intermediate = static_cast<std::size_t>(dims.intermediate);
  const std::size_t embedding_blocks = hidden / cpu::q8_0_values_per_block;
  if (!weights.embed_tokens.is_q8_0() || (hidden % cpu::q8_0_values_per_block) != 0) {
    error_message = "Batched CPU Q8_0 forward requires Q8_0 embeddings aligned to 32 values.";
    return false;
  }
  if (profiling != nullptr) {
    profiling->forward_pass_tokens += static_cast<int>(batch_size);
  }

  const auto embedding_start = std::chrono::steady_clock::now();
  std::vector<float> x(batch_size * hidden);
  for (std::size_t token = 0; token < batch_size; ++token) {
    const int token_id = token_ids[token];
    if (token_id < 0 || token_id >= dims.vocab_size) {
      error_message = "Batched CPU prefill token id is outside the vocabulary.";
      return false;
    }
    cpu::q8_0_dequantize(
      weights.embed_tokens.q8_0_blocks.data() +
        static_cast<std::size_t>(token_id) * embedding_blocks,
      x.data() + token * hidden,
      embedding_blocks,
      weights.embed_tokens.q8_0_backend);
  }
  if (profiling != nullptr) {
    profiling->embedding_ms += elapsed_ms(embedding_start);
  }

  std::vector<float> normed;
  std::vector<float> attention;
  std::vector<float> residual(batch_size * hidden);
  std::vector<float> post_norm;
  std::vector<float> gate_up;
  std::vector<float> mlp_hidden(batch_size * intermediate);
  std::vector<float> mlp_output;
  int full_index = 0;
  int linear_index = 0;

  for (int layer_index = 0; layer_index < dims.n_layers; ++layer_index) {
    const LayerWeights & layer = weights.layers[static_cast<std::size_t>(layer_index)];
    rms_norm_qwen3next_batch(
      x, batch_size, hidden, layer.input_layernorm, dims.rms_eps, normed);
    const auto attention_start = std::chrono::steady_clock::now();
    bool attention_ok = false;
    if (layer.is_linear) {
      attention_ok = run_linear_attention_batch_cpu_q8(
        layer,
        dims,
        state.linear_states[static_cast<std::size_t>(linear_index++)],
        normed,
        batch_size,
        attention,
        error_message);
    } else {
      attention_ok = run_full_attention_batch_cpu_q8(
        layer,
        dims,
        state.full_states[static_cast<std::size_t>(full_index++)],
        normed,
        batch_size,
        position_start,
        attention,
        error_message);
    }
    if (!attention_ok) {
      return false;
    }
    if (profiling != nullptr) {
      profiling->attention_ms += elapsed_ms(attention_start);
    }

    const auto mlp_start = std::chrono::steady_clock::now();
    cpu::add_f32(x.data(), attention.data(), residual.data(), residual.size());
    rms_norm_qwen3next_batch(
      residual, batch_size, hidden, layer.post_attention_layernorm, dims.rms_eps, post_norm);
    if (!layer.mlp_gate_up_cpu.is_q8_0() ||
        !matmul_2d_q8_batch(
          layer.mlp_gate_up_cpu, post_norm, batch_size, gate_up, error_message)) {
      if (error_message.empty()) {
        error_message = "Batched CPU MLP requires packed Q8_0 gate/up weights.";
      }
      return false;
    }
    for (std::size_t token = 0; token < batch_size; ++token) {
      const float * gate = gate_up.data() + token * 2 * intermediate;
      cpu::silu_mul_f32(
        gate,
        gate + intermediate,
        mlp_hidden.data() + token * intermediate,
        intermediate,
        layer.mlp_down.q8_0_backend);
    }
    if (!matmul_2d_q8_batch(
          layer.mlp_down, mlp_hidden, batch_size, mlp_output, error_message)) {
      return false;
    }
    cpu::add_f32(
      residual.data(), mlp_output.data(), x.data(), x.size());
    if (profiling != nullptr) {
      profiling->mlp_ms += elapsed_ms(mlp_start);
    }
  }

  if (!compute_next_logits) {
    return true;
  }
  const auto logits_start = std::chrono::steady_clock::now();
  const float * last_hidden = x.data() + (batch_size - 1) * hidden;
  std::vector<float> final_input(last_hidden, last_hidden + hidden);
  std::vector<float> final_hidden;
  rms_norm_qwen3next(final_input, weights.final_norm, dims.rms_eps, final_hidden);
  const bool ok = compute_next_logits_from_embedding(
    weights.embed_tokens, final_hidden, false, next_logits, error_message);
  if (profiling != nullptr) {
    profiling->logits_ms += elapsed_ms(logits_start);
  }
  return ok;
}

} // namespace
