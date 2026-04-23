bool init_runtime_decode_backend(
  RuntimeDecodeBackend & backend,
  const ReferenceInferenceOptions & options,
  std::string & error_message) {
  backend = RuntimeDecodeBackend{};
  backend.kind = RuntimeDecodeBackendKind::runtime_default;
  if (options.use_cuda && options.gpu_decode_backend == GpuDecodeBackend::luce) {
    backend.kind = RuntimeDecodeBackendKind::luce;
    if (options.sampling.temperature > 0.0f) {
      error_message = "Luce decode backend currently supports only greedy decode (temperature <= 0).";
      return false;
    }
    if (options.gpu_decode_blocks < 0) {
      error_message = "gpu_decode_blocks must be >= 0.";
      return false;
    }
    luce::LuceDecodeBackendConfig config;
    config.model_dir = options.model_dir;
    config.max_context = options.max_context;
    config.decode_blocks = options.gpu_decode_blocks;
    if (!backend.luce_backend.initialize(config, error_message)) {
      return false;
    }
  }
  backend.initialized = true;
  return true;
}

bool reset_runtime_decode_backend(
  RuntimeDecodeBackend & backend,
  std::string & error_message) {
  if (!backend.initialized) {
    error_message = "Decode backend reset requested before init.";
    return false;
  }
  switch (backend.kind) {
    case RuntimeDecodeBackendKind::runtime_default:
      return true;
    case RuntimeDecodeBackendKind::luce:
      return backend.luce_backend.reset(error_message);
    default:
      error_message = "Unsupported runtime decode backend kind.";
      return false;
  }
}

void release_runtime_decode_backend(RuntimeDecodeBackend & backend) {
  backend = RuntimeDecodeBackend{};
}

bool decode_step_with_runtime_backend(
  RuntimeDecodeBackend & backend,
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
  if (!backend.initialized) {
    error_message = "Decode backend decode_step requested before init.";
    return false;
  }

  switch (backend.kind) {
    case RuntimeDecodeBackendKind::runtime_default:
      return run_forward_single_token(
        weights,
        dims,
        state,
        token_id,
        position,
        use_cuda,
        profile_cuda_sync,
        next_logits,
        use_cuda_gpu_sampling,
        compute_next_logits,
        cuda_workspace,
        profiling,
        error_message);
    case RuntimeDecodeBackendKind::luce: {
      int next_token = 0;
      if (!backend.luce_backend.run_decode_step(token_id, position, next_token, error_message)) {
        return false;
      }
      if (compute_next_logits) {
        if (next_token < 0 || next_token >= dims.vocab_size) {
          error_message = "Luce decode backend produced an out-of-range token id.";
          return false;
        }
        next_logits.assign(static_cast<std::size_t>(dims.vocab_size), 0.0f);
        next_logits[static_cast<std::size_t>(next_token)] = 1.0f;
      } else {
        next_logits.clear();
      }
      if (profiling != nullptr) {
        ++profiling->forward_pass_tokens;
      }
      return true;
    }
    default:
      error_message = "Unsupported runtime decode backend kind.";
      return false;
  }
}

bool decode_step_with_runtime_backend_from_device_token(
  RuntimeDecodeBackend & backend,
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
  if (!backend.initialized) {
    error_message = "Decode backend decode_step requested before init.";
    return false;
  }

  switch (backend.kind) {
    case RuntimeDecodeBackendKind::runtime_default:
      return run_forward_single_token_cuda_device_from_token_buffer(
        weights,
        dims,
        state,
        token_id_device,
        position,
        use_cuda_gpu_sampling,
        compute_next_logits,
        profile_cuda_sync,
        workspace,
        next_logits,
        profiling,
        error_message);
    case RuntimeDecodeBackendKind::luce:
      error_message = "Luce decode backend does not support device-token decode path.";
      return false;
    default:
      error_message = "Unsupported runtime decode backend kind.";
      return false;
  }
}
