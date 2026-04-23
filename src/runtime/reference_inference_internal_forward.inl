bool run_forward_single_token_cuda_device_core(
  const ModelWeights & weights,
  const RuntimeDims & dims,
  ModelState & state,
  const int position,
  const bool use_cuda_gpu_sampling,
  const bool compute_next_logits,
  const bool profile_cuda_sync,
  CudaForwardWorkspace & workspace,
  std::vector<float> & next_logits,
  DecodeProfilingAccumulator * profiling,
  std::string & error_message) {
  const std::size_t hidden_count = static_cast<std::size_t>(dims.hidden);
  const std::size_t intermediate_count = static_cast<std::size_t>(dims.intermediate);

  int full_idx = 0;
  int linear_idx = 0;
  for (int il = 0; il < dims.n_layers; ++il) {
    const LayerWeights & layer = weights.layers[static_cast<std::size_t>(il)];
    if (!layer.has_device_norms) {
      error_message = "CUDA forward device path requires uploaded layer norm weights.";
      return false;
    }

    if (!cuda::run_rms_norm_f32(
          workspace.hidden_in,
          layer.input_layernorm_device,
          hidden_count,
          dims.rms_eps,
          workspace.full_q,
          error_message)) {
      return false;
    }

    if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
      return false;
    }
    const auto attention_start = std::chrono::steady_clock::now();
    if (layer.is_linear) {
      if (!run_linear_attention_step_cuda_device(
            layer,
            dims,
            static_cast<std::size_t>(il),
            state.linear_states[static_cast<std::size_t>(linear_idx)],
            workspace.full_q,
            workspace,
            workspace.hidden_out,
            error_message)) {
        return false;
      }
      ++linear_idx;
    } else {
      if (!run_full_attention_step_cuda_device(
            layer,
            dims,
            state.full_states[static_cast<std::size_t>(full_idx)],
            workspace.full_q,
            position,
            workspace,
            workspace.hidden_out,
            error_message)) {
        return false;
      }
      ++full_idx;
    }
    if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
      return false;
    }
    if (profiling != nullptr) {
      profiling->attention_ms += elapsed_ms(attention_start);
    }

    if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
      return false;
    }
    const auto mlp_start = std::chrono::steady_clock::now();
    if (!cuda::run_add_f32(workspace.hidden_in, workspace.hidden_out, hidden_count, workspace.full_attn, error_message) ||
        !cuda::run_rms_norm_f32(
          workspace.full_attn,
          layer.post_attention_layernorm_device,
          hidden_count,
          dims.rms_eps,
          workspace.full_q,
          error_message)) {
      return false;
    }

    const auto run_mlp_direct = [&](std::string & run_error) -> bool {
      bool mlp_ok = false;
      if (layer.has_device_mlp_gate_up && layer.mlp_gate_up_device.data != nullptr) {
        const std::size_t packed_count = intermediate_count * 2;
        if (workspace.projection_out.count < packed_count) {
          run_error = "CUDA workspace projection buffer is too small for packed MLP projections.";
          return false;
        }
        const cuda::CudaDeviceBufferF32 gate = buffer_slice_f32(workspace.projection_out, 0, intermediate_count);
        const cuda::CudaDeviceBufferF32 up = buffer_slice_f32(workspace.projection_out, intermediate_count, intermediate_count);
        if (gate.data == nullptr || up.data == nullptr) {
          run_error = "Failed to create packed MLP projection slices.";
          return false;
        }
        mlp_ok = cuda::run_matvec_f32_device(layer.mlp_gate_up_device, workspace.full_q, workspace.projection_out, run_error) &&
                 cuda::run_silu_mul_f32(gate, up, intermediate_count, workspace.intermediate_hidden, run_error);
      } else {
        mlp_ok = cuda::run_matvec_f32_device(layer.mlp_gate.device_matrix, workspace.full_q, workspace.intermediate_a, run_error) &&
                 cuda::run_matvec_f32_device(layer.mlp_up.device_matrix, workspace.full_q, workspace.intermediate_b, run_error) &&
                 cuda::run_silu_mul_f32(
                   workspace.intermediate_a,
                   workspace.intermediate_b,
                   intermediate_count,
                   workspace.intermediate_hidden,
                   run_error);
      }
      return mlp_ok &&
             cuda::run_matvec_f32_device(layer.mlp_down.device_matrix, workspace.intermediate_hidden, workspace.hidden_out, run_error) &&
             cuda::run_add_f32(workspace.full_attn, workspace.hidden_out, hidden_count, workspace.hidden_in, run_error);
    };

    bool mlp_done = false;
    const std::size_t layer_slot = static_cast<std::size_t>(il);
    if (layer_slot < workspace.mlp_graphs.size()) {
      if (workspace.mlp_graphs[layer_slot].ready) {
        mlp_done = cuda::launch_captured_graph(workspace.mlp_graphs[layer_slot], error_message);
        if (!mlp_done) {
          workspace.mlp_graph_disabled[layer_slot] = true;
          cuda::free_captured_graph(workspace.mlp_graphs[layer_slot]);
        }
      }

      if (!mlp_done && !workspace.mlp_graph_disabled[layer_slot] && workspace.mlp_graph_warmup_done[layer_slot]) {
        std::string capture_error;
        const bool capture_started = cuda::begin_stream_capture(capture_error);
        bool capture_ok = false;
        if (capture_started) {
          std::string captured_run_error;
          const bool captured_run_ok = run_mlp_direct(captured_run_error);
          std::string capture_end_error;
          const bool capture_end_ok = cuda::end_stream_capture(workspace.mlp_graphs[layer_slot], capture_end_error);
          if (captured_run_ok && capture_end_ok) {
            capture_ok = cuda::launch_captured_graph(workspace.mlp_graphs[layer_slot], capture_error);
          }
        }
        if (!capture_ok) {
          workspace.mlp_graph_disabled[layer_slot] = true;
          cuda::free_captured_graph(workspace.mlp_graphs[layer_slot]);
        } else {
          mlp_done = true;
        }
      }

      if (!mlp_done && !workspace.mlp_graph_disabled[layer_slot] && !workspace.mlp_graph_warmup_done[layer_slot]) {
        std::string warmup_error;
        if (!run_mlp_direct(warmup_error)) {
          error_message = warmup_error;
          return false;
        }
        workspace.mlp_graph_warmup_done[layer_slot] = true;
        mlp_done = true;
      }
    }

    if (!mlp_done) {
      std::string direct_error;
      if (!run_mlp_direct(direct_error)) {
        error_message = direct_error;
        return false;
      }
    }

    if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
      return false;
    }
    if (profiling != nullptr) {
      profiling->mlp_ms += elapsed_ms(mlp_start);
    }
  }

  if (!compute_next_logits) {
    return true;
  }

  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  const auto logits_start = std::chrono::steady_clock::now();
  bool ok = false;
  if (!cuda::run_rms_norm_f32(
        workspace.hidden_in,
        weights.final_norm_device,
        hidden_count,
        dims.rms_eps,
        workspace.hidden_out,
        error_message)) {
    return false;
  }

  if (use_cuda_gpu_sampling) {
    ok = cuda::run_matvec_f32_device(weights.embed_tokens.device_matrix, workspace.hidden_out, workspace.logits, error_message);
  } else {
    std::vector<float> final_hidden;
    if (!cuda::download_from_buffer_f32(workspace.hidden_out, hidden_count, 0, final_hidden, error_message)) {
      return false;
    }
    ok = compute_next_logits_from_embedding(weights.embed_tokens, final_hidden, false, next_logits, error_message);
  }
  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  if (profiling != nullptr) {
    profiling->logits_ms += elapsed_ms(logits_start);
  }
  return ok;
}

bool run_forward_single_token_cuda_device(
  const ModelWeights & weights,
  const RuntimeDims & dims,
  ModelState & state,
  const int token_id,
  const int position,
  const bool use_cuda_gpu_sampling,
  const bool compute_next_logits,
  const bool profile_cuda_sync,
  CudaForwardWorkspace & workspace,
  std::vector<float> & next_logits,
  DecodeProfilingAccumulator * profiling,
  std::string & error_message) {
  if (!workspace.has_device_buffers || !workspace.has_gpu_sampling_buffers || !weights.has_device_final_norm ||
      !weights.embed_tokens.has_device_matrix || weights.embed_tokens.device_matrix.data == nullptr) {
    error_message = "CUDA forward device path is not fully initialized.";
    return false;
  }
  if (token_id < 0 || token_id >= dims.vocab_size) {
    error_message = "Token id out of vocabulary range.";
    return false;
  }

  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  const auto embedding_start = std::chrono::steady_clock::now();
  if (!cuda::gather_matrix_row_f32(weights.embed_tokens.device_matrix, token_id, workspace.hidden_in, error_message)) {
    return false;
  }
  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  if (profiling != nullptr) {
    profiling->embedding_ms += elapsed_ms(embedding_start);
  }

  return run_forward_single_token_cuda_device_core(
    weights,
    dims,
    state,
    position,
    use_cuda_gpu_sampling,
    compute_next_logits,
    profile_cuda_sync,
    workspace,
    next_logits,
    profiling,
    error_message);
}

bool run_forward_single_token_cuda_device_from_token_buffer(
  const ModelWeights & weights,
  const RuntimeDims & dims,
  ModelState & state,
  const cuda::CudaDeviceBufferF32 & token_id_device,
  const int position,
  const bool use_cuda_gpu_sampling,
  const bool compute_next_logits,
  const bool profile_cuda_sync,
  CudaForwardWorkspace & workspace,
  std::vector<float> & next_logits,
  DecodeProfilingAccumulator * profiling,
  std::string & error_message) {
  if (!workspace.has_device_buffers || !workspace.has_gpu_sampling_buffers || !weights.has_device_final_norm ||
      !weights.embed_tokens.has_device_matrix || weights.embed_tokens.device_matrix.data == nullptr) {
    error_message = "CUDA forward device path is not fully initialized.";
    return false;
  }

  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  const auto embedding_start = std::chrono::steady_clock::now();
  if (!cuda::gather_matrix_row_f32_from_token_f32(
        weights.embed_tokens.device_matrix,
        token_id_device,
        workspace.hidden_in,
        error_message)) {
    return false;
  }
  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  if (profiling != nullptr) {
    profiling->embedding_ms += elapsed_ms(embedding_start);
  }

  return run_forward_single_token_cuda_device_core(
    weights,
    dims,
    state,
    position,
    use_cuda_gpu_sampling,
    compute_next_logits,
    profile_cuda_sync,
    workspace,
    next_logits,
    profiling,
    error_message);
}

bool run_forward_single_token(
  const ModelWeights & weights,
  const RuntimeDims & dims,
  ModelState & state,
  const int token_id,
  const int position,
  const bool use_cuda,
  const bool profile_cuda_sync,
  std::vector<float> & next_logits,
  const bool use_cuda_gpu_sampling,
  const bool compute_next_logits,
  CudaForwardWorkspace * cuda_workspace,
  DecodeProfilingAccumulator * profiling,
  std::string & error_message) {
  if (token_id < 0 || token_id >= dims.vocab_size) {
    error_message = "Token id out of vocabulary range.";
    return false;
  }

  if (profiling != nullptr) {
    ++profiling->forward_pass_tokens;
  }

  const bool use_cuda_device_forward =
    use_cuda && use_cuda_gpu_sampling && cuda_workspace != nullptr && cuda_workspace->has_device_buffers &&
    cuda_workspace->has_gpu_sampling_buffers;
  if (use_cuda_device_forward) {
    return run_forward_single_token_cuda_device(
      weights,
      dims,
      state,
      token_id,
      position,
      use_cuda_gpu_sampling,
      compute_next_logits,
      profile_cuda_sync,
      *cuda_workspace,
      next_logits,
      profiling,
      error_message);
  }

  const auto embedding_start = std::chrono::steady_clock::now();
  std::vector<float> x(static_cast<std::size_t>(dims.hidden), 0.0f);
  const float * emb_row = weights.embed_tokens.data.data() +
                         static_cast<std::size_t>(token_id) * static_cast<std::size_t>(dims.hidden);
  std::memcpy(x.data(), emb_row, static_cast<std::size_t>(dims.hidden) * sizeof(float));
  if (profiling != nullptr) {
    profiling->embedding_ms += elapsed_ms(embedding_start);
  }

  int full_idx = 0;
  int linear_idx = 0;
  std::vector<float> normed;
  std::vector<float> attn_out;
  std::vector<float> residual;
  std::vector<float> post_norm;
  std::vector<float> mlp_gate;
  std::vector<float> mlp_up;
  std::vector<float> mlp_hidden;
  std::vector<float> mlp_out;

  for (int il = 0; il < dims.n_layers; ++il) {
    const LayerWeights & layer = weights.layers[static_cast<std::size_t>(il)];
    rms_norm_qwen3next(x, layer.input_layernorm, dims.rms_eps, normed);

    const auto attention_start = std::chrono::steady_clock::now();
    if (layer.is_linear) {
      if (!run_linear_attention_step(
            layer,
            dims,
            state.linear_states[static_cast<std::size_t>(linear_idx)],
            normed,
            attn_out,
            use_cuda,
            cuda_workspace,
            error_message)) {
        return false;
      }
      ++linear_idx;
    } else {
      if (!run_full_attention_step(
            layer,
            dims,
            state.full_states[static_cast<std::size_t>(full_idx)],
            normed,
            position,
            attn_out,
            use_cuda,
            cuda_workspace,
            error_message)) {
        return false;
      }
      ++full_idx;
    }
    if (profiling != nullptr) {
      profiling->attention_ms += elapsed_ms(attention_start);
    }

    const auto mlp_start = std::chrono::steady_clock::now();
    residual.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      residual[i] = x[i] + attn_out[i];
    }

    rms_norm_qwen3next(residual, layer.post_attention_layernorm, dims.rms_eps, post_norm);
    if (use_cuda && cuda_workspace != nullptr && cuda_workspace->has_device_buffers) {
      if (!run_mlp_block_cuda_hybrid(layer, post_norm, *cuda_workspace, mlp_out, error_message)) {
        return false;
      }
    } else {
      if (!matvec_2d(layer.mlp_gate, post_norm, mlp_gate, use_cuda, error_message) ||
          !matvec_2d(layer.mlp_up, post_norm, mlp_up, use_cuda, error_message)) {
        return false;
      }
      mlp_hidden.resize(mlp_gate.size());
      for (std::size_t i = 0; i < mlp_gate.size(); ++i) {
        mlp_hidden[i] = siluf(mlp_gate[i]) * mlp_up[i];
      }
      if (!matvec_2d(layer.mlp_down, mlp_hidden, mlp_out, use_cuda, error_message)) {
        return false;
      }
    }

    x.resize(residual.size());
    for (std::size_t i = 0; i < residual.size(); ++i) {
      x[i] = residual[i] + mlp_out[i];
    }
    if (profiling != nullptr) {
      profiling->mlp_ms += elapsed_ms(mlp_start);
    }
  }

  if (!compute_next_logits) {
    return true;
  }

  const auto logits_start = std::chrono::steady_clock::now();
  std::vector<float> final_hidden;
  rms_norm_qwen3next(x, weights.final_norm, dims.rms_eps, final_hidden);
  bool ok = false;
  if (use_cuda_gpu_sampling && use_cuda && cuda_workspace != nullptr && cuda_workspace->has_gpu_sampling_buffers) {
    if (!weights.embed_tokens.has_device_matrix || weights.embed_tokens.device_matrix.data == nullptr) {
      error_message = "CUDA GPU sampling path requested but embedding matrix is not uploaded.";
      ok = false;
    } else {
      ok = cuda::upload_to_buffer_f32(final_hidden.data(), final_hidden.size(), cuda_workspace->hidden_in, 0, error_message) &&
           cuda::run_matvec_f32_device(
             weights.embed_tokens.device_matrix,
             cuda_workspace->hidden_in,
             cuda_workspace->logits,
             error_message);
    }
  } else {
    ok = compute_next_logits_from_embedding(weights.embed_tokens, final_hidden, use_cuda, next_logits, error_message);
  }
  if (profiling != nullptr) {
    profiling->logits_ms += elapsed_ms(logits_start);
  }
  return ok;
}

} // namespace
