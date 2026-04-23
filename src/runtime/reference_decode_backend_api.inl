bool init_runtime_decode_backend(
  RuntimeDecodeBackend & backend,
  std::string & error_message) {
  backend = RuntimeDecodeBackend{};
  backend.kind = RuntimeDecodeBackendKind::runtime_default;
  backend.initialized = true;
  (void)error_message;
  return true;
}

bool reset_runtime_decode_backend(
  RuntimeDecodeBackend & backend,
  std::string & error_message) {
  if (!backend.initialized) {
    error_message = "Decode backend reset requested before init.";
    return false;
  }
  return true;
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
    default:
      error_message = "Unsupported runtime decode backend kind.";
      return false;
  }
}
