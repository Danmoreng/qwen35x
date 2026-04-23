bool sample_token_from_logits_f32_device_to_buffer(
  const CudaDeviceBufferF32 & logits,
  const CudaDeviceBufferF32 & seen_token_mask,
  const int vocab_size,
  const float temperature,
  const float top_p,
  const int top_k,
  const float repetition_penalty,
  const float random_u01,
  const CudaDeviceBufferF32 & out_token,
  std::string & error_message) {
  if (logits.data == nullptr || seen_token_mask.data == nullptr) {
    error_message = "CUDA sampling requires initialized device buffers.";
    return false;
  }
  if (vocab_size <= 0 || logits.count < static_cast<std::size_t>(vocab_size) ||
      seen_token_mask.count < static_cast<std::size_t>(vocab_size)) {
    error_message = "CUDA sampling buffer size is out of bounds.";
    return false;
  }
  if (top_p <= 0.0f || top_p > 1.0f) {
    error_message = "CUDA sampling requires top_p in (0, 1].";
    return false;
  }
  if (repetition_penalty <= 0.0f) {
    error_message = "CUDA sampling requires repetition_penalty > 0.";
    return false;
  }
  if (temperature < 0.0f) {
    error_message = "CUDA sampling requires temperature >= 0.";
    return false;
  }
  if (temperature > 0.0f) {
    if (top_k <= 0 || top_k > kGpuSamplingMaxTopK) {
      error_message = "CUDA sampling requires top_k in [1, 64] when temperature > 0.";
      return false;
    }
  }
  if (out_token.data == nullptr || out_token.count < 1) {
    error_message = "CUDA sampling output token buffer is not initialized.";
    return false;
  }
  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA sampling stream is not initialized.";
    return false;
  }

  if (!ensure_sampling_workspace(vocab_size, error_message)) {
    return false;
  }

  const int block_count = std::max(1, (vocab_size + kSamplingArgmaxChunkSize - 1) / kSamplingArgmaxChunkSize);

  if (temperature <= 0.0f) {
    argmax_blocks_kernel<<<block_count, kSamplingArgmaxBlockSize, 0, stream>>>(
      static_cast<const float *>(logits.data),
      static_cast<const float *>(seen_token_mask.data),
      vocab_size,
      repetition_penalty,
      g_session.sampling_block_values,
      g_session.sampling_block_indices);
    if (!check_cuda(cudaGetLastError(), "argmax_blocks_kernel", error_message)) {
      return false;
    }
    finalize_argmax_token_kernel<<<1, 1, 0, stream>>>(
      g_session.sampling_block_values,
      g_session.sampling_block_indices,
      block_count,
      static_cast<float *>(seen_token_mask.data),
      vocab_size,
      static_cast<float *>(out_token.data));
    if (!check_cuda(cudaGetLastError(), "finalize_argmax_token_kernel", error_message)) {
      return false;
    }
  } else {
    topk_blocks_kernel<<<block_count, 1, 0, stream>>>(
      static_cast<const float *>(logits.data),
      static_cast<const float *>(seen_token_mask.data),
      vocab_size,
      temperature,
      repetition_penalty,
      top_k,
      g_session.sampling_block_topk_values,
      g_session.sampling_block_topk_indices);
    if (!check_cuda(cudaGetLastError(), "topk_blocks_kernel", error_message)) {
      return false;
    }
    sample_token_from_block_topk_kernel<<<1, 1, 0, stream>>>(
      g_session.sampling_block_topk_values,
      g_session.sampling_block_topk_indices,
      block_count,
      vocab_size,
      top_p,
      top_k,
      random_u01,
      static_cast<float *>(seen_token_mask.data),
      static_cast<float *>(out_token.data));
    if (!check_cuda(cudaGetLastError(), "sample_token_from_block_topk_kernel", error_message)) {
      return false;
    }
  }

  return true;
}

bool sample_token_from_logits_f32_device(
  const CudaDeviceBufferF32 & logits,
  const CudaDeviceBufferF32 & seen_token_mask,
  const int vocab_size,
  const float temperature,
  const float top_p,
  const int top_k,
  const float repetition_penalty,
  const float random_u01,
  const CudaDeviceBufferF32 & topk_values_scratch,
  const CudaDeviceBufferF32 & topk_indices_scratch,
  int & out_token,
  std::string & error_message) {
  (void)topk_values_scratch;
  (void)topk_indices_scratch;

  if (g_session.workspace_output == nullptr || g_session.workspace_output_count < 1) {
    error_message = "CUDA sampling output workspace is not initialized.";
    return false;
  }
  CudaDeviceBufferF32 sampled_token_device;
  sampled_token_device.data = g_session.workspace_output;
  sampled_token_device.count = 1;

  if (!sample_token_from_logits_f32_device_to_buffer(
        logits,
        seen_token_mask,
        vocab_size,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        random_u01,
        sampled_token_device,
        error_message)) {
    return false;
  }

  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA sampling stream is not initialized.";
    return false;
  }

  float token_value = -1.0f;
  if (!tracked_memcpy_async(
        &token_value,
        sampled_token_device.data,
        sizeof(float),
        cudaMemcpyDeviceToHost,
        stream,
        "cudaMemcpyAsync(sampled_token)",
        error_message)) {
    return false;
  }
  if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(sampled_token)", error_message)) {
    return false;
  }

  out_token = static_cast<int>(token_value);
  if (out_token < 0 || out_token >= vocab_size) {
    error_message = "CUDA sampling failed to produce a valid token id.";
    return false;
  }
  return true;
}

bool begin_stream_capture(std::string & error_message) {
  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA stream capture requires an active inference session stream.";
    return false;
  }
  return check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture", error_message);
}

bool end_stream_capture(
  CudaCapturedGraph & out_graph,
  std::string & error_message) {
  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA stream capture requires an active inference session stream.";
    return false;
  }

  cudaGraph_t captured_graph = nullptr;
  if (!check_cuda(cudaStreamEndCapture(stream, &captured_graph), "cudaStreamEndCapture", error_message)) {
    return false;
  }
  if (captured_graph == nullptr) {
    error_message = "CUDA stream capture returned a null graph.";
    return false;
  }

  cudaGraphExec_t graph_exec = nullptr;
  if (!check_cuda(
        cudaGraphInstantiate(&graph_exec, captured_graph, nullptr, nullptr, 0),
        "cudaGraphInstantiate",
        error_message)) {
    cudaGraphDestroy(captured_graph);
    return false;
  }

  free_captured_graph(out_graph);
  out_graph.graph = captured_graph;
  out_graph.exec = graph_exec;
  out_graph.ready = true;
  return true;
}

bool launch_captured_graph(
  const CudaCapturedGraph & graph,
  std::string & error_message) {
  if (!graph.ready || graph.exec == nullptr) {
    error_message = "CUDA captured graph is not initialized.";
    return false;
  }

  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA stream capture requires an active inference session stream.";
    return false;
  }
  return check_cuda(
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph.exec), stream),
    "cudaGraphLaunch",
    error_message);
}

bool synchronize_stream(std::string & error_message) {
  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA stream is not initialized.";
    return false;
  }
  return check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", error_message);
}

void free_captured_graph(CudaCapturedGraph & graph) {
  if (graph.exec != nullptr) {
    cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph.exec));
    graph.exec = nullptr;
  }
  if (graph.graph != nullptr) {
    cudaGraphDestroy(static_cast<cudaGraph_t>(graph.graph));
    graph.graph = nullptr;
  }
  graph.ready = false;
}

void reset_transfer_stats() {
  g_transfer_stats = CudaTransferStats{};
}

void get_transfer_stats(CudaTransferStats & out_stats) {
  out_stats = g_transfer_stats;
}

