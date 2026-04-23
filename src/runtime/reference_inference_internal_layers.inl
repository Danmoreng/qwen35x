bool run_linear_attention_step_cuda_device(
  const LayerWeights & layer,
  const RuntimeDims & dims,
  const std::size_t layer_slot,
  LinearAttentionState & state,
  const cuda::CudaDeviceBufferF32 & x_in,
  CudaForwardWorkspace & workspace,
  cuda::CudaDeviceBufferF32 & out_hidden,
  std::string & error_message) {
  if (!state.has_device_state || !layer.linear.has_device_params || !layer.linear.has_device_in_proj_all ||
      !layer.linear.in_proj_all_device.data || !layer.linear.conv1d.has_device_matrix ||
      !layer.linear.out_proj.has_device_matrix) {
    error_message = "Linear-attention CUDA device path is not fully initialized.";
    return false;
  }

  const std::size_t mixed_count = static_cast<std::size_t>(dims.linear_conv_channels);
  const std::size_t z_count = static_cast<std::size_t>(dims.linear_v_dim);
  const std::size_t b_count = static_cast<std::size_t>(dims.linear_num_v_heads);
  const std::size_t a_count = static_cast<std::size_t>(dims.linear_num_v_heads);
  const std::size_t packed_count = mixed_count + z_count + b_count + a_count;
  if (workspace.projection_out.count < packed_count) {
    error_message = "CUDA workspace projection buffer is too small for packed linear projections.";
    return false;
  }

  const cuda::CudaDeviceBufferF32 mixed = buffer_slice_f32(workspace.projection_out, 0, mixed_count);
  const cuda::CudaDeviceBufferF32 z = buffer_slice_f32(workspace.projection_out, mixed_count, z_count);
  const cuda::CudaDeviceBufferF32 b = buffer_slice_f32(workspace.projection_out, mixed_count + z_count, b_count);
  const cuda::CudaDeviceBufferF32 a = buffer_slice_f32(workspace.projection_out, mixed_count + z_count + b_count, a_count);
  if (mixed.data == nullptr || z.data == nullptr || b.data == nullptr || a.data == nullptr) {
    error_message = "Failed to create packed linear projection slices.";
    return false;
  }

  const auto run_linear_direct = [&](std::string & run_error) -> bool {
    return cuda::run_matvec_f32_device(layer.linear.in_proj_all_device, x_in, workspace.projection_out, run_error) &&
           cuda::run_linear_attention_decode(
             mixed,
             z,
             b,
             a,
             layer.linear.conv1d.device_matrix,
             layer.linear.norm_device,
             layer.linear.dt_bias_device,
             layer.linear.ssm_a_device,
             dims.linear_kernel,
             dims.linear_num_k_heads,
             dims.linear_num_v_heads,
             dims.linear_head_k_dim,
             dims.linear_head_v_dim,
             dims.rms_eps,
             state.conv_state_device,
             state.recurrent_state_device,
             workspace.linear_conv_out,
             workspace.linear_gated_norm,
             run_error) &&
           cuda::run_matvec_f32_device(layer.linear.out_proj.device_matrix, workspace.linear_gated_norm, out_hidden, run_error);
  };

  bool linear_done = false;
  if (layer_slot < workspace.linear_attention_graphs.size()) {
    if (workspace.linear_attention_graphs[layer_slot].ready) {
      linear_done = cuda::launch_captured_graph(workspace.linear_attention_graphs[layer_slot], error_message);
      if (!linear_done) {
        workspace.linear_attention_graph_disabled[layer_slot] = true;
        cuda::free_captured_graph(workspace.linear_attention_graphs[layer_slot]);
      }
    }

    if (!linear_done && !workspace.linear_attention_graph_disabled[layer_slot] &&
        workspace.linear_attention_graph_warmup_done[layer_slot]) {
      std::string capture_error;
      const bool capture_started = cuda::begin_stream_capture(capture_error);
      bool capture_ok = false;
      if (capture_started) {
        std::string captured_run_error;
        const bool captured_run_ok = run_linear_direct(captured_run_error);
        std::string capture_end_error;
        const bool capture_end_ok = cuda::end_stream_capture(
          workspace.linear_attention_graphs[layer_slot],
          capture_end_error);
        if (captured_run_ok && capture_end_ok) {
          capture_ok = cuda::launch_captured_graph(workspace.linear_attention_graphs[layer_slot], capture_error);
        }
      }
      if (!capture_ok) {
        workspace.linear_attention_graph_disabled[layer_slot] = true;
        cuda::free_captured_graph(workspace.linear_attention_graphs[layer_slot]);
      } else {
        linear_done = true;
      }
    }

    if (!linear_done && !workspace.linear_attention_graph_disabled[layer_slot] &&
        !workspace.linear_attention_graph_warmup_done[layer_slot]) {
      std::string warmup_error;
      if (!run_linear_direct(warmup_error)) {
        error_message = warmup_error;
        return false;
      }
      workspace.linear_attention_graph_warmup_done[layer_slot] = true;
      linear_done = true;
    }
  }

  if (linear_done) {
    return true;
  }
  return run_linear_direct(error_message);
}

bool run_full_attention_step_cuda_device(
  const LayerWeights & layer,
  const RuntimeDims & dims,
  FullAttentionState & state,
  const cuda::CudaDeviceBufferF32 & x_in,
  const int position,
  CudaForwardWorkspace & workspace,
  cuda::CudaDeviceBufferF32 & out_hidden,
  std::string & error_message) {
  const bool has_packed_qkv = layer.full.has_device_qkv_proj && layer.full.qkv_proj_device.data != nullptr;
  const bool has_legacy_qkv =
    layer.full.has_device_kv_proj && layer.full.q_proj.has_device_matrix && layer.full.kv_proj_device.data != nullptr;
  if (!state.has_device_state || !layer.full.has_device_norm || !layer.full.o_proj.has_device_matrix ||
      (!has_packed_qkv && !has_legacy_qkv)) {
    error_message = "Full-attention CUDA device path is not fully initialized.";
    return false;
  }

  const std::size_t q_packed_count = static_cast<std::size_t>(dims.n_heads * dims.head_dim * 2);
  const std::size_t kv_count = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  const std::size_t kv_packed_count = kv_count * 2;
  const std::size_t qkv_packed_count = q_packed_count + kv_packed_count;
  const std::size_t cache_offset = static_cast<std::size_t>(position) * kv_count;
  const std::size_t required_projection_count = has_packed_qkv ? qkv_packed_count : kv_packed_count;
  if ((!has_packed_qkv && workspace.logits.count < q_packed_count) || workspace.projection_out.count < required_projection_count ||
      workspace.full_q.count < static_cast<std::size_t>(dims.n_heads * dims.head_dim) ||
      workspace.full_gate.count < static_cast<std::size_t>(dims.n_heads * dims.head_dim) ||
      workspace.full_attn.count < static_cast<std::size_t>(dims.n_heads * dims.head_dim)) {
    error_message = "CUDA forward workspace is too small for full-attention device step.";
    return false;
  }

  const cuda::CudaDeviceBufferF32 q_gate_packed = has_packed_qkv
    ? buffer_slice_f32(workspace.projection_out, 0, q_packed_count)
    : workspace.logits;
  const std::size_t kv_offset = has_packed_qkv ? q_packed_count : 0;
  const cuda::CudaDeviceBufferF32 k_raw = buffer_slice_f32(workspace.projection_out, kv_offset, kv_count);
  const cuda::CudaDeviceBufferF32 v_raw = buffer_slice_f32(workspace.projection_out, kv_offset + kv_count, kv_count);
  if (q_gate_packed.data == nullptr || k_raw.data == nullptr || v_raw.data == nullptr) {
    error_message = "Failed to create packed KV projection slices.";
    return false;
  }

  const auto run_qkv_projections = [&](std::string & run_error) -> bool {
    if (has_packed_qkv) {
      return cuda::run_matvec_f32_device(layer.full.qkv_proj_device, x_in, workspace.projection_out, run_error);
    }
    return cuda::run_matvec_f32_device(layer.full.q_proj.device_matrix, x_in, workspace.logits, run_error) &&
           cuda::run_matvec_f32_device(layer.full.kv_proj_device, x_in, workspace.projection_out, run_error);
  };

  return run_qkv_projections(error_message) &&
         cuda::run_prepare_full_attention_qkv_f32(
           q_gate_packed,
           k_raw,
           v_raw,
           layer.full.q_norm_device,
           layer.full.k_norm_device,
           dims.n_heads,
           dims.n_kv_heads,
           dims.head_dim,
           dims.rope_dim,
           position,
           dims.rope_theta,
           dims.rms_eps,
           workspace.full_q,
           workspace.full_gate,
           state.k_cache_device,
           cache_offset,
           state.v_cache_device,
           cache_offset,
           error_message) &&
         cuda::run_full_attention_decode_gqa(
           workspace.full_q,
           workspace.full_gate,
           state.k_cache_device,
           state.v_cache_device,
           dims.n_heads,
           dims.n_kv_heads,
           dims.head_dim,
           position + 1,
           workspace.full_attn,
           workspace.full_scores,
           error_message) &&
         cuda::run_matvec_f32_device(layer.full.o_proj.device_matrix, workspace.full_attn, out_hidden, error_message);
}

bool run_linear_attention_step(
  const LayerWeights & layer,
  const RuntimeDims & dims,
  LinearAttentionState & state,
  const std::vector<float> & x,
  std::vector<float> & out,
  const bool use_cuda,
  CudaForwardWorkspace * cuda_workspace,
  std::string & error_message) {
  std::vector<float> mixed_qkv;
  std::vector<float> z_vec;
  std::vector<float> b_vec;
  std::vector<float> a_vec;
  const bool use_cuda_linear_kernel =
    use_cuda && cuda_workspace != nullptr && cuda_workspace->has_device_buffers && state.has_device_state &&
    layer.linear.has_device_params && layer.linear.in_proj_qkv.has_device_matrix && layer.linear.in_proj_z.has_device_matrix &&
    layer.linear.in_proj_b.has_device_matrix && layer.linear.in_proj_a.has_device_matrix && layer.linear.conv1d.has_device_matrix &&
    layer.linear.out_proj.has_device_matrix;
  if (use_cuda_linear_kernel) {
    if (!cuda::upload_to_buffer_f32(x.data(), x.size(), cuda_workspace->hidden_in, 0, error_message) ||
        !cuda::run_matvec_f32_device(layer.linear.in_proj_qkv.device_matrix, cuda_workspace->hidden_in, cuda_workspace->linear_mixed_qkv, error_message) ||
        !cuda::run_matvec_f32_device(layer.linear.in_proj_z.device_matrix, cuda_workspace->hidden_in, cuda_workspace->linear_z, error_message) ||
        !cuda::run_matvec_f32_device(layer.linear.in_proj_b.device_matrix, cuda_workspace->hidden_in, cuda_workspace->linear_b, error_message) ||
        !cuda::run_matvec_f32_device(layer.linear.in_proj_a.device_matrix, cuda_workspace->hidden_in, cuda_workspace->linear_a, error_message) ||
        !cuda::run_linear_attention_decode(
          cuda_workspace->linear_mixed_qkv,
          cuda_workspace->linear_z,
          cuda_workspace->linear_b,
          cuda_workspace->linear_a,
          layer.linear.conv1d.device_matrix,
          layer.linear.norm_device,
          layer.linear.dt_bias_device,
          layer.linear.ssm_a_device,
          dims.linear_kernel,
          dims.linear_num_k_heads,
          dims.linear_num_v_heads,
          dims.linear_head_k_dim,
          dims.linear_head_v_dim,
          dims.rms_eps,
          state.conv_state_device,
          state.recurrent_state_device,
          cuda_workspace->linear_conv_out,
          cuda_workspace->linear_gated_norm,
          error_message) ||
        !cuda::run_matvec_f32_device(layer.linear.out_proj.device_matrix, cuda_workspace->linear_gated_norm, cuda_workspace->hidden_out, error_message) ||
        !cuda::download_from_buffer_f32(
          cuda_workspace->hidden_out,
          static_cast<std::size_t>(dims.hidden),
          0,
          out,
          error_message)) {
      return false;
    }
    return true;
  }

  const bool use_cuda_projection_batch =
    use_cuda && cuda_workspace != nullptr && cuda_workspace->has_device_buffers;
  if (use_cuda_projection_batch) {
    if (!cuda::upload_to_buffer_f32(x.data(), x.size(), cuda_workspace->hidden_in, 0, error_message) ||
        !project_from_uploaded_hidden_cuda(
          layer.linear.in_proj_qkv,
          *cuda_workspace,
          static_cast<std::size_t>(dims.linear_conv_channels),
          mixed_qkv,
          error_message) ||
        !project_from_uploaded_hidden_cuda(
          layer.linear.in_proj_z,
          *cuda_workspace,
          static_cast<std::size_t>(dims.linear_v_dim),
          z_vec,
          error_message) ||
        !project_from_uploaded_hidden_cuda(
          layer.linear.in_proj_b,
          *cuda_workspace,
          static_cast<std::size_t>(dims.linear_num_v_heads),
          b_vec,
          error_message) ||
        !project_from_uploaded_hidden_cuda(
          layer.linear.in_proj_a,
          *cuda_workspace,
          static_cast<std::size_t>(dims.linear_num_v_heads),
          a_vec,
          error_message)) {
      return false;
    }
  } else {
    if (!matvec_2d(layer.linear.in_proj_qkv, x, mixed_qkv, use_cuda, error_message) ||
        !matvec_2d(layer.linear.in_proj_z, x, z_vec, use_cuda, error_message) ||
        !matvec_2d(layer.linear.in_proj_b, x, b_vec, use_cuda, error_message) ||
        !matvec_2d(layer.linear.in_proj_a, x, a_vec, use_cuda, error_message)) {
      return false;
    }
  }

  std::vector<float> beta(static_cast<std::size_t>(dims.linear_num_v_heads), 0.0f);
  std::vector<float> alpha(static_cast<std::size_t>(dims.linear_num_v_heads), 0.0f);
  for (int h = 0; h < dims.linear_num_v_heads; ++h) {
    beta[static_cast<std::size_t>(h)] = sigmoidf_stable(b_vec[static_cast<std::size_t>(h)]);
    const float pre_gate = softplusf_stable(a_vec[static_cast<std::size_t>(h)] + layer.linear.dt_bias.data[static_cast<std::size_t>(h)]);
    alpha[static_cast<std::size_t>(h)] = std::exp(pre_gate * layer.linear.ssm_a[static_cast<std::size_t>(h)]);
  }

  const int conv_hist = dims.linear_kernel - 1;
  std::vector<float> conv_window(static_cast<std::size_t>(dims.linear_kernel * dims.linear_conv_channels));
  if (conv_hist > 0) {
    std::memcpy(
      conv_window.data(),
      state.conv_state.data(),
      static_cast<std::size_t>(conv_hist * dims.linear_conv_channels) * sizeof(float));
  }
  std::memcpy(
    conv_window.data() + static_cast<std::size_t>(conv_hist * dims.linear_conv_channels),
    mixed_qkv.data(),
    static_cast<std::size_t>(dims.linear_conv_channels) * sizeof(float));

  if (conv_hist > 0) {
    std::memcpy(
      state.conv_state.data(),
      conv_window.data() + static_cast<std::size_t>(dims.linear_conv_channels),
      static_cast<std::size_t>(conv_hist * dims.linear_conv_channels) * sizeof(float));
  }

  std::vector<float> conv_out(static_cast<std::size_t>(dims.linear_conv_channels), 0.0f);
  for (int c = 0; c < dims.linear_conv_channels; ++c) {
    float s = 0.0f;
    const float * w = layer.linear.conv1d.data.data() +
                      static_cast<std::size_t>(c) * static_cast<std::size_t>(dims.linear_kernel);
    for (int k = 0; k < dims.linear_kernel; ++k) {
      const std::size_t idx = static_cast<std::size_t>(k * dims.linear_conv_channels + c);
      s += conv_window[idx] * w[k];
    }
    conv_out[static_cast<std::size_t>(c)] = siluf(s);
  }

  std::vector<float> q(conv_out.begin(), conv_out.begin() + dims.linear_q_dim);
  std::vector<float> k(conv_out.begin() + dims.linear_q_dim, conv_out.begin() + 2 * dims.linear_q_dim);
  std::vector<float> v(conv_out.begin() + 2 * dims.linear_q_dim, conv_out.end());
  l2_norm_per_head(q, dims.linear_num_k_heads, dims.linear_head_k_dim);
  l2_norm_per_head(k, dims.linear_num_k_heads, dims.linear_head_k_dim);

  const float q_scale = 1.0f / std::sqrt(static_cast<float>(dims.linear_head_k_dim));
  for (float & qv : q) {
    qv *= q_scale;
  }

  std::vector<float> core_out(static_cast<std::size_t>(dims.linear_v_dim), 0.0f);
  for (int h = 0; h < dims.linear_num_v_heads; ++h) {
    const std::size_t q_base = static_cast<std::size_t>(h * dims.linear_head_k_dim);
    const std::size_t v_base = static_cast<std::size_t>(h * dims.linear_head_v_dim);
    const std::size_t s_base =
      static_cast<std::size_t>(h) * static_cast<std::size_t>(dims.linear_head_v_dim * dims.linear_head_v_dim);
    float * s = state.recurrent_state.data() + s_base;

    for (int i = 0; i < dims.linear_head_v_dim; ++i) {
      for (int j = 0; j < dims.linear_head_v_dim; ++j) {
        s[static_cast<std::size_t>(i * dims.linear_head_v_dim + j)] *= alpha[static_cast<std::size_t>(h)];
      }
    }

    std::vector<float> sk(static_cast<std::size_t>(dims.linear_head_v_dim), 0.0f);
    for (int i = 0; i < dims.linear_head_v_dim; ++i) {
      float sum = 0.0f;
      const float * s_row = s + static_cast<std::size_t>(i * dims.linear_head_v_dim);
      for (int j = 0; j < dims.linear_head_v_dim; ++j) {
        sum += s_row[j] * k[q_base + static_cast<std::size_t>(j)];
      }
      sk[static_cast<std::size_t>(i)] = sum;
    }

    for (int i = 0; i < dims.linear_head_v_dim; ++i) {
      const float delta = (v[v_base + static_cast<std::size_t>(i)] - sk[static_cast<std::size_t>(i)]) *
                          beta[static_cast<std::size_t>(h)];
      float * s_row = s + static_cast<std::size_t>(i * dims.linear_head_v_dim);
      for (int j = 0; j < dims.linear_head_v_dim; ++j) {
        s_row[j] += delta * k[q_base + static_cast<std::size_t>(j)];
      }
    }

    for (int i = 0; i < dims.linear_head_v_dim; ++i) {
      float sum = 0.0f;
      const float * s_row = s + static_cast<std::size_t>(i * dims.linear_head_v_dim);
      for (int j = 0; j < dims.linear_head_v_dim; ++j) {
        sum += s_row[j] * q[q_base + static_cast<std::size_t>(j)];
      }
      core_out[v_base + static_cast<std::size_t>(i)] = sum;
    }
  }

  std::vector<float> gated_norm(static_cast<std::size_t>(dims.linear_v_dim), 0.0f);
  for (int h = 0; h < dims.linear_num_v_heads; ++h) {
    const std::size_t base = static_cast<std::size_t>(h * dims.linear_head_v_dim);
    float sq_sum = 0.0f;
    for (int d = 0; d < dims.linear_head_v_dim; ++d) {
      const float cv = core_out[base + static_cast<std::size_t>(d)];
      sq_sum += cv * cv;
    }
    const float inv = 1.0f / std::sqrt(sq_sum / static_cast<float>(dims.linear_head_v_dim) + dims.rms_eps);
    for (int d = 0; d < dims.linear_head_v_dim; ++d) {
      const std::size_t idx = base + static_cast<std::size_t>(d);
      gated_norm[idx] = core_out[idx] * inv * layer.linear.norm.data[static_cast<std::size_t>(d)] * siluf(z_vec[idx]);
    }
  }

  if (!matvec_2d(layer.linear.out_proj, gated_norm, out, use_cuda, error_message)) {
    return false;
  }
  return true;
}

bool run_full_attention_step(
  const LayerWeights & layer,
  const RuntimeDims & dims,
  FullAttentionState & state,
  const std::vector<float> & x,
  const int position,
  std::vector<float> & out,
  const bool use_cuda,
  CudaForwardWorkspace * cuda_workspace,
  std::string & error_message) {
  std::vector<float> q_full;
  std::vector<float> k_flat;
  std::vector<float> v_flat;
  const bool use_cuda_projection_batch =
    use_cuda && cuda_workspace != nullptr && cuda_workspace->has_device_buffers;
  const std::size_t full_q_out = static_cast<std::size_t>(dims.n_heads * dims.head_dim * 2);
  const std::size_t full_kv_out = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  if (use_cuda_projection_batch) {
    if (!cuda::upload_to_buffer_f32(x.data(), x.size(), cuda_workspace->hidden_in, 0, error_message) ||
        !project_from_uploaded_hidden_cuda(layer.full.q_proj, *cuda_workspace, full_q_out, q_full, error_message) ||
        !project_from_uploaded_hidden_cuda(layer.full.k_proj, *cuda_workspace, full_kv_out, k_flat, error_message) ||
        !project_from_uploaded_hidden_cuda(layer.full.v_proj, *cuda_workspace, full_kv_out, v_flat, error_message)) {
      return false;
    }
  } else {
    if (!matvec_2d(layer.full.q_proj, x, q_full, use_cuda, error_message) ||
        !matvec_2d(layer.full.k_proj, x, k_flat, use_cuda, error_message) ||
        !matvec_2d(layer.full.v_proj, x, v_flat, use_cuda, error_message)) {
      return false;
    }
  }

  const int q_span = dims.head_dim * 2;
  std::vector<float> q(static_cast<std::size_t>(dims.n_heads * dims.head_dim));
  std::vector<float> gate(static_cast<std::size_t>(dims.n_heads * dims.head_dim));
  for (int h = 0; h < dims.n_heads; ++h) {
    const std::size_t src = static_cast<std::size_t>(h * q_span);
    const std::size_t dst = static_cast<std::size_t>(h * dims.head_dim);
    std::memcpy(q.data() + dst, q_full.data() + src, static_cast<std::size_t>(dims.head_dim) * sizeof(float));
    std::memcpy(gate.data() + dst, q_full.data() + src + static_cast<std::size_t>(dims.head_dim),
                static_cast<std::size_t>(dims.head_dim) * sizeof(float));
  }

  std::vector<float> q_normed;
  std::vector<float> k_normed;
  rms_norm_per_head_qwen3next(q, dims.n_heads, dims.head_dim, layer.full.q_norm, dims.rms_eps, q_normed);
  rms_norm_per_head_qwen3next(k_flat, dims.n_kv_heads, dims.head_dim, layer.full.k_norm, dims.rms_eps, k_normed);

  apply_rope_inplace(q_normed, dims.n_heads, dims.head_dim, dims.rope_dim, position, dims.rope_theta);
  apply_rope_inplace(k_normed, dims.n_kv_heads, dims.head_dim, dims.rope_dim, position, dims.rope_theta);

  const std::size_t token_stride = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  std::memcpy(
    state.k_cache.data() + static_cast<std::size_t>(position) * token_stride,
    k_normed.data(),
    token_stride * sizeof(float));
  std::memcpy(
    state.v_cache.data() + static_cast<std::size_t>(position) * token_stride,
    v_flat.data(),
    token_stride * sizeof(float));

  const bool use_cuda_full_kernel =
    use_cuda && state.has_device_state && cuda_workspace != nullptr && cuda_workspace->has_device_buffers &&
    layer.full.o_proj.has_device_matrix;
  if (use_cuda_full_kernel) {
    const std::size_t offset = static_cast<std::size_t>(position) * token_stride;
    const std::size_t full_q_count = static_cast<std::size_t>(dims.n_heads * dims.head_dim);
    if (!cuda::upload_to_buffer_f32(k_normed.data(), token_stride, state.k_cache_device, offset, error_message) ||
        !cuda::upload_to_buffer_f32(v_flat.data(), token_stride, state.v_cache_device, offset, error_message) ||
        !cuda::upload_to_buffer_f32(q_normed.data(), full_q_count, cuda_workspace->full_q, 0, error_message) ||
        !cuda::upload_to_buffer_f32(gate.data(), full_q_count, cuda_workspace->full_gate, 0, error_message) ||
        !cuda::run_full_attention_decode_gqa(
          cuda_workspace->full_q,
          cuda_workspace->full_gate,
          state.k_cache_device,
          state.v_cache_device,
          dims.n_heads,
          dims.n_kv_heads,
          dims.head_dim,
          position + 1,
          cuda_workspace->full_attn,
          cuda_workspace->full_scores,
          error_message) ||
        !cuda::run_matvec_f32_device(layer.full.o_proj.device_matrix, cuda_workspace->full_attn, cuda_workspace->hidden_out, error_message) ||
        !cuda::download_from_buffer_f32(
          cuda_workspace->hidden_out,
          static_cast<std::size_t>(dims.hidden),
          0,
          out,
          error_message)) {
      return false;
    }
    return true;
  }

  const int n_rep = dims.n_heads / dims.n_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));
  const int seq_len = position + 1;
  std::vector<float> attn_cat(static_cast<std::size_t>(dims.n_heads * dims.head_dim), 0.0f);
  std::vector<float> scores(static_cast<std::size_t>(seq_len), 0.0f);

  for (int h = 0; h < dims.n_heads; ++h) {
    const int kvh = h / n_rep;
    const std::size_t q_base = static_cast<std::size_t>(h * dims.head_dim);

    float max_score = -std::numeric_limits<float>::infinity();
    for (int t = 0; t < seq_len; ++t) {
      const float * k_ptr = state.k_cache.data() +
                           (static_cast<std::size_t>(t) * static_cast<std::size_t>(dims.n_kv_heads) +
                            static_cast<std::size_t>(kvh)) *
                             static_cast<std::size_t>(dims.head_dim);
      float dot = 0.0f;
      for (int d = 0; d < dims.head_dim; ++d) {
        dot += q_normed[q_base + static_cast<std::size_t>(d)] * k_ptr[d];
      }
      dot *= scale;
      scores[static_cast<std::size_t>(t)] = dot;
      if (dot > max_score) {
        max_score = dot;
      }
    }

    float denom = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      const float ev = std::exp(scores[static_cast<std::size_t>(t)] - max_score);
      scores[static_cast<std::size_t>(t)] = ev;
      denom += ev;
    }
    if (denom <= 0.0f) {
      denom = 1.0f;
    }

    for (int d = 0; d < dims.head_dim; ++d) {
      float acc = 0.0f;
      for (int t = 0; t < seq_len; ++t) {
        const float p = scores[static_cast<std::size_t>(t)] / denom;
        const float * v_ptr = state.v_cache.data() +
                             (static_cast<std::size_t>(t) * static_cast<std::size_t>(dims.n_kv_heads) +
                              static_cast<std::size_t>(kvh)) *
                               static_cast<std::size_t>(dims.head_dim);
        acc += p * v_ptr[d];
      }
      attn_cat[q_base + static_cast<std::size_t>(d)] = acc * sigmoidf_stable(gate[q_base + static_cast<std::size_t>(d)]);
    }
  }

  if (!matvec_2d(layer.full.o_proj, attn_cat, out, use_cuda, error_message)) {
    return false;
  }
  return true;
}

bool compute_next_logits_from_embedding(
  const TensorData & embed,
  const std::vector<float> & hidden,
  const bool use_cuda,
  std::vector<float> & out_logits,
  std::string & error_message) {
  const int vocab = static_cast<int>(embed.shape[0]);
  const int dim = static_cast<int>(embed.shape[1]);

  if (use_cuda) {
    if (!embed.has_device_matrix || embed.device_matrix.data == nullptr) {
      error_message = "CUDA logits requested but embedding matrix is not uploaded.";
      return false;
    }
    if (!cuda::run_matvec_f32(embed.device_matrix, hidden, out_logits, error_message)) {
      return false;
    }
  } else {
    out_logits.assign(static_cast<std::size_t>(vocab), 0.0f);
    for (int token = 0; token < vocab; ++token) {
      const float * row = embed.data.data() + static_cast<std::size_t>(token) * static_cast<std::size_t>(dim);
      float logit = 0.0f;
      for (int d = 0; d < dim; ++d) {
        logit += row[d] * hidden[static_cast<std::size_t>(d)];
      }
      out_logits[static_cast<std::size_t>(token)] = logit;
    }
  }

  return true;
}

int argmax_index(const std::vector<float> & values) {
  int best_idx = 0;
  float best_value = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    if (values[static_cast<std::size_t>(i)] > best_value) {
      best_value = values[static_cast<std::size_t>(i)];
      best_idx = i;
    }
  }
  return best_idx;
}

void apply_repetition_penalty_inplace(
  std::vector<float> & logits,
  const std::vector<int> & token_counts,
  const float repetition_penalty) {
  if (repetition_penalty <= 1.0f) {
    return;
  }
  const std::size_t n = std::min(logits.size(), token_counts.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (token_counts[i] <= 0) {
      continue;
    }
    if (logits[i] > 0.0f) {
      logits[i] /= repetition_penalty;
    } else {
      logits[i] *= repetition_penalty;
    }
  }
}

bool sample_token_from_logits(
  const std::vector<float> & raw_logits,
  const SamplingOptions & sampling,
  const std::vector<int> & token_counts,
  std::mt19937 & rng,
  int & out_token,
  std::string & error_message) {
  if (raw_logits.empty()) {
    error_message = "Sampling logits are empty.";
    return false;
  }
  if (sampling.temperature < 0.0f) {
    error_message = "temperature must be >= 0.";
    return false;
  }
  if (sampling.top_p <= 0.0f || sampling.top_p > 1.0f) {
    error_message = "top_p must be in (0, 1].";
    return false;
  }
  if (sampling.top_k < 0) {
    error_message = "top_k must be >= 0.";
    return false;
  }
  if (sampling.repetition_penalty < 1.0f) {
    error_message = "repeat_penalty must be >= 1.0.";
    return false;
  }

  std::vector<float> logits = raw_logits;
  apply_repetition_penalty_inplace(logits, token_counts, sampling.repetition_penalty);

  if (sampling.temperature <= 1.0e-6f) {
    out_token = argmax_index(logits);
    return true;
  }

  const float inv_temp = 1.0f / sampling.temperature;
  for (float & v : logits) {
    v *= inv_temp;
  }

  std::vector<int> candidate_ids(logits.size());
  std::iota(candidate_ids.begin(), candidate_ids.end(), 0);
  if (sampling.top_k > 0 && sampling.top_k < static_cast<int>(candidate_ids.size())) {
    const std::size_t keep = static_cast<std::size_t>(sampling.top_k);
    std::nth_element(
      candidate_ids.begin(),
      candidate_ids.begin() + static_cast<std::ptrdiff_t>(keep),
      candidate_ids.end(),
      [&](const int a, const int b) {
        return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
      });
    candidate_ids.resize(keep);
  }

  std::sort(candidate_ids.begin(), candidate_ids.end(), [&](const int a, const int b) {
    return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
  });

  if (candidate_ids.empty()) {
    error_message = "Sampling candidates are empty after top_k filtering.";
    return false;
  }

  float max_logit = -std::numeric_limits<float>::infinity();
  for (const int id : candidate_ids) {
    max_logit = std::max(max_logit, logits[static_cast<std::size_t>(id)]);
  }

  std::vector<float> probs(candidate_ids.size(), 0.0f);
  float denom = 0.0f;
  for (std::size_t i = 0; i < candidate_ids.size(); ++i) {
    const float p = std::exp(logits[static_cast<std::size_t>(candidate_ids[i])] - max_logit);
    probs[i] = p;
    denom += p;
  }
  if (!(denom > 0.0f)) {
    out_token = candidate_ids.front();
    return true;
  }
  for (float & p : probs) {
    p /= denom;
  }

  std::size_t nucleus_keep = probs.size();
  if (sampling.top_p < 1.0f) {
    float cumulative = 0.0f;
    nucleus_keep = 0;
    for (; nucleus_keep < probs.size(); ++nucleus_keep) {
      cumulative += probs[nucleus_keep];
      if (cumulative >= sampling.top_p) {
        ++nucleus_keep;
        break;
      }
    }
    nucleus_keep = std::max<std::size_t>(1, std::min(nucleus_keep, probs.size()));
  }

  candidate_ids.resize(nucleus_keep);
  probs.resize(nucleus_keep);
  float renorm = 0.0f;
  for (const float p : probs) {
    renorm += p;
  }
  if (!(renorm > 0.0f)) {
    out_token = candidate_ids.front();
    return true;
  }
  for (float & p : probs) {
    p /= renorm;
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float r = dist(rng);
  float cumulative = 0.0f;
  for (std::size_t i = 0; i < probs.size(); ++i) {
    cumulative += probs[i];
    if (r <= cumulative || i + 1 == probs.size()) {
      out_token = candidate_ids[i];
      return true;
    }
  }

  out_token = candidate_ids.back();
  return true;
}

bool generated_ends_with_sequence(
  const std::vector<std::int32_t> & generated_tokens,
  const std::vector<std::int32_t> & stop_sequence) {
  if (stop_sequence.empty() || stop_sequence.size() > generated_tokens.size()) {
    return false;
  }
  const std::size_t start = generated_tokens.size() - stop_sequence.size();
  for (std::size_t i = 0; i < stop_sequence.size(); ++i) {
    if (generated_tokens[start + i] != stop_sequence[i]) {
      return false;
    }
  }
  return true;
}

bool project_from_uploaded_hidden_cuda(
  const TensorData & projection,
  CudaForwardWorkspace & workspace,
  const std::size_t output_count,
  std::vector<float> & out_values,
  std::string & error_message) {
  if (!projection.has_device_matrix || projection.device_matrix.data == nullptr) {
    error_message = "CUDA projection requested but device matrix is unavailable.";
    return false;
  }
  if (!workspace.has_device_buffers || workspace.hidden_in.data == nullptr || workspace.projection_out.data == nullptr) {
    error_message = "CUDA forward workspace is not initialized.";
    return false;
  }
  if (workspace.projection_out.count < output_count) {
    error_message = "CUDA projection workspace is smaller than requested output.";
    return false;
  }
  if (!cuda::run_matvec_f32_device(
        projection.device_matrix,
        workspace.hidden_in,
        workspace.projection_out,
        error_message) ||
      !cuda::download_from_buffer_f32(workspace.projection_out, output_count, 0, out_values, error_message)) {
    return false;
  }
  return true;
}

bool run_mlp_block_cuda_hybrid(
  const LayerWeights & layer,
  const std::vector<float> & post_norm,
  CudaForwardWorkspace & workspace,
  std::vector<float> & mlp_out,
  std::string & error_message) {
  if (!workspace.has_device_buffers) {
    error_message = "CUDA forward workspace is not initialized.";
    return false;
  }
  if (!layer.mlp_gate.has_device_matrix || !layer.mlp_up.has_device_matrix || !layer.mlp_down.has_device_matrix) {
    error_message = "CUDA MLP path requested but one or more MLP weights are not uploaded.";
    return false;
  }

  const std::size_t hidden_count = post_norm.size();
  const std::size_t intermediate_count = static_cast<std::size_t>(layer.mlp_gate.shape[0]);
  const std::size_t out_hidden_count = static_cast<std::size_t>(layer.mlp_down.shape[0]);
  if (workspace.hidden_in.count < hidden_count || workspace.hidden_out.count < out_hidden_count ||
      workspace.intermediate_a.count < intermediate_count || workspace.intermediate_b.count < intermediate_count ||
      workspace.intermediate_hidden.count < intermediate_count) {
    error_message = "CUDA forward workspace buffer sizes do not match model dimensions.";
    return false;
  }

  if (!cuda::upload_to_buffer_f32(post_norm.data(), hidden_count, workspace.hidden_in, 0, error_message) ||
      !cuda::run_matvec_f32_device(layer.mlp_gate.device_matrix, workspace.hidden_in, workspace.intermediate_a, error_message) ||
      !cuda::run_matvec_f32_device(layer.mlp_up.device_matrix, workspace.hidden_in, workspace.intermediate_b, error_message) ||
      !cuda::run_silu_mul_f32(
        workspace.intermediate_a,
        workspace.intermediate_b,
        intermediate_count,
        workspace.intermediate_hidden,
        error_message) ||
      !cuda::run_matvec_f32_device(layer.mlp_down.device_matrix, workspace.intermediate_hidden, workspace.hidden_out, error_message) ||
      !cuda::download_from_buffer_f32(workspace.hidden_out, out_hidden_count, 0, mlp_out, error_message)) {
    return false;
  }

  return true;
}

