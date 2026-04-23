bool begin_inference_session(
  const std::size_t max_input_count,
  const std::size_t max_output_count,
  std::string & error_message) {
  if (max_input_count == 0 || max_output_count == 0) {
    error_message = "Inference session requires non-zero max input/output counts.";
    return false;
  }
  return ensure_session_workspace(max_input_count, max_output_count, error_message);
}

void end_inference_session() {
  if (!g_session.active) {
    return;
  }
  release_session_storage();
  g_session.active = false;
}

bool upload_matrix_f32(
  const std::vector<float> & host_data,
  const int rows,
  const int cols,
  CudaDeviceMatrixF32 & out_matrix,
  std::string & error_message) {
  if (rows <= 0 || cols <= 0) {
    error_message = "Invalid matrix dimensions for CUDA upload.";
    return false;
  }

  const std::size_t expected_count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (host_data.size() != expected_count) {
    error_message = "Host matrix size does not match rows * cols.";
    return false;
  }

  void * device_ptr = nullptr;
  if (!check_cuda(cudaMalloc(&device_ptr, expected_count * sizeof(float)), "cudaMalloc(matrix)", error_message)) {
    return false;
  }
  if (!tracked_memcpy(
        device_ptr,
        host_data.data(),
        expected_count * sizeof(float),
        cudaMemcpyHostToDevice,
        "cudaMemcpy(matrix)",
        error_message)) {
    cudaFree(device_ptr);
    return false;
  }

  out_matrix.data = device_ptr;
  out_matrix.data_bf16 = nullptr;
  out_matrix.rows = rows;
  out_matrix.cols = cols;
  return true;
}

bool upload_matrix_bf16_shadow_from_f32(
  const std::vector<float> & host_data,
  const int rows,
  const int cols,
  CudaDeviceMatrixF32 & matrix,
  std::string & error_message) {
  if (rows <= 0 || cols <= 0) {
    error_message = "Invalid matrix dimensions for CUDA BF16 shadow upload.";
    return false;
  }
  const std::size_t expected_count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (host_data.size() != expected_count) {
    error_message = "Host matrix size does not match rows * cols for BF16 upload.";
    return false;
  }
  if (matrix.rows != rows || matrix.cols != cols) {
    error_message = "BF16 shadow upload dimensions do not match matrix metadata.";
    return false;
  }

  std::vector<std::uint16_t> bf16_values(expected_count);
  for (std::size_t i = 0; i < expected_count; ++i) {
    bf16_values[i] = float_to_bf16_bits(host_data[i]);
  }

  void * bf16_ptr = nullptr;
  if (!check_cuda(cudaMalloc(&bf16_ptr, expected_count * sizeof(std::uint16_t)), "cudaMalloc(matrix_bf16)", error_message)) {
    return false;
  }
  if (!tracked_memcpy(
        bf16_ptr,
        bf16_values.data(),
        expected_count * sizeof(std::uint16_t),
        cudaMemcpyHostToDevice,
        "cudaMemcpy(matrix_bf16)",
        error_message)) {
    cudaFree(bf16_ptr);
    return false;
  }

  if (matrix.data_bf16 != nullptr) {
    cudaFree(matrix.data_bf16);
  }
  matrix.data_bf16 = bf16_ptr;
  return true;
}

void free_matrix_f32(CudaDeviceMatrixF32 & matrix) {
  if (matrix.data != nullptr) {
    cudaFree(matrix.data);
    matrix.data = nullptr;
  }
  if (matrix.data_bf16 != nullptr) {
    cudaFree(matrix.data_bf16);
    matrix.data_bf16 = nullptr;
  }
  matrix.rows = 0;
  matrix.cols = 0;
}

bool run_matvec_f32(
  const CudaDeviceMatrixF32 & matrix,
  const std::vector<float> & input,
  std::vector<float> & output,
  std::string & error_message) {
  if (matrix.data == nullptr || matrix.rows <= 0 || matrix.cols <= 0) {
    error_message = "CUDA matrix is not initialized.";
    return false;
  }
  if (input.size() != static_cast<std::size_t>(matrix.cols)) {
    error_message = "Input size does not match matrix columns.";
    return false;
  }

  if (!ensure_session_workspace(input.size(), static_cast<std::size_t>(matrix.rows), error_message)) {
    return false;
  }
  cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA inference session stream is not initialized.";
    return false;
  }

  if (!tracked_memcpy_async(
        g_session.workspace_input,
        input.data(),
        input.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream,
        "cudaMemcpyAsync(input)",
        error_message)) {
    return false;
  }

  const int block_size = 128;
  const int grid_size = (matrix.rows + block_size - 1) / block_size;
  f32_matvec_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(matrix.data),
    g_session.workspace_input,
    g_session.workspace_output,
    matrix.rows,
    matrix.cols);
  if (!check_cuda(cudaGetLastError(), "f32_matvec_kernel", error_message)) {
    return false;
  }

  output.resize(static_cast<std::size_t>(matrix.rows));
  if (!tracked_memcpy_async(
        output.data(),
        g_session.workspace_output,
        output.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream,
        "cudaMemcpyAsync(output)",
        error_message)) {
    return false;
  }
  if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(matvec)", error_message)) {
    return false;
  }

  return true;
}

bool run_matvec_f32_device(
  const CudaDeviceMatrixF32 & matrix,
  const CudaDeviceBufferF32 & input,
  CudaDeviceBufferF32 & output,
  std::string & error_message) {
  if (matrix.data == nullptr || matrix.rows <= 0 || matrix.cols <= 0) {
    error_message = "CUDA matrix is not initialized.";
    return false;
  }
  if (input.data == nullptr || input.count < static_cast<std::size_t>(matrix.cols)) {
    error_message = "CUDA device input buffer is invalid for matvec.";
    return false;
  }
  if (output.data == nullptr || output.count < static_cast<std::size_t>(matrix.rows)) {
    error_message = "CUDA device output buffer is invalid for matvec.";
    return false;
  }

  if (g_session.prefer_bf16_matvec && matrix.data_bf16 != nullptr) {
    std::string bf16_error;
    if (run_matvec_bf16_device_cublaslt(matrix, input, output, bf16_error)) {
      return true;
    }
  }

  std::string cublas_error;
  if (run_matvec_f32_device_cublaslt(matrix, input, output, cublas_error)) {
    return true;
  }

  const int block_size = 128;
  const int grid_size = (matrix.rows + block_size - 1) / block_size;
  const cudaStream_t stream = active_stream();
  f32_matvec_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(matrix.data),
    static_cast<const float *>(input.data),
    static_cast<float *>(output.data),
    matrix.rows,
    matrix.cols);
  if (check_cuda(cudaGetLastError(), "f32_matvec_kernel(device)", error_message)) {
    return true;
  }
  if (!cublas_error.empty()) {
    error_message += " (fallback reason: " + cublas_error + ")";
  }
  return false;
}

bool run_matmul_f32_device(
  const CudaDeviceMatrixF32 & matrix,
  const CudaDeviceBufferF32 & input,
  const int input_cols,
  CudaDeviceBufferF32 & output,
  std::string & error_message) {
  if (matrix.data == nullptr || matrix.rows <= 0 || matrix.cols <= 0) {
    error_message = "CUDA matrix is not initialized.";
    return false;
  }
  if (input_cols <= 0) {
    error_message = "CUDA matmul requires input_cols > 0.";
    return false;
  }

  const std::size_t input_required =
    static_cast<std::size_t>(matrix.cols) * static_cast<std::size_t>(input_cols);
  const std::size_t output_required =
    static_cast<std::size_t>(matrix.rows) * static_cast<std::size_t>(input_cols);
  if (input.data == nullptr || input.count < input_required) {
    error_message = "CUDA matmul input buffer is invalid.";
    return false;
  }
  if (output.data == nullptr || output.count < output_required) {
    error_message = "CUDA matmul output buffer is invalid.";
    return false;
  }
  if (g_session.cublas_lt == nullptr) {
    error_message = "cuBLASLt handle is not initialized.";
    return false;
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  auto cleanup = [&]() {
    if (pref != nullptr) {
      cublasLtMatmulPreferenceDestroy(pref);
      pref = nullptr;
    }
    if (c_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(c_desc);
      c_desc = nullptr;
    }
    if (b_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(b_desc);
      b_desc = nullptr;
    }
    if (a_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(a_desc);
      a_desc = nullptr;
    }
    if (op_desc != nullptr) {
      cublasLtMatmulDescDestroy(op_desc);
      op_desc = nullptr;
    }
  };

  const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  const cudaDataType_t scale_type = CUDA_R_32F;
  const cudaDataType_t data_type = CUDA_R_32F;
  cublasOperation_t trans = CUBLAS_OP_N;
  if (!check_cublas(cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type), "cublasLtMatmulDescCreate", error_message) ||
      !check_cublas(
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)),
        "cublasLtMatmulDescSetAttribute(TRANSA)",
        error_message) ||
      !check_cublas(
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)),
        "cublasLtMatmulDescSetAttribute(TRANSB)",
        error_message)) {
    cleanup();
    return false;
  }

  if (!check_cublas(
        cublasLtMatrixLayoutCreate(
          &a_desc,
          data_type,
          static_cast<std::uint64_t>(matrix.rows),
          static_cast<std::uint64_t>(matrix.cols),
          static_cast<std::int64_t>(matrix.cols)),
        "cublasLtMatrixLayoutCreate(A)",
        error_message) ||
      !check_cublas(
        cublasLtMatrixLayoutCreate(
          &b_desc,
          data_type,
          static_cast<std::uint64_t>(matrix.cols),
          static_cast<std::uint64_t>(input_cols),
          static_cast<std::int64_t>(input_cols)),
        "cublasLtMatrixLayoutCreate(B)",
        error_message) ||
      !check_cublas(
        cublasLtMatrixLayoutCreate(
          &c_desc,
          data_type,
          static_cast<std::uint64_t>(matrix.rows),
          static_cast<std::uint64_t>(input_cols),
          static_cast<std::int64_t>(input_cols)),
        "cublasLtMatrixLayoutCreate(C)",
        error_message)) {
    cleanup();
    return false;
  }

  const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  if (!check_cublas(
        cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
        "cublasLtMatrixLayoutSetAttribute(A_ORDER)",
        error_message) ||
      !check_cublas(
        cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
        "cublasLtMatrixLayoutSetAttribute(B_ORDER)",
        error_message) ||
      !check_cublas(
        cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
        "cublasLtMatrixLayoutSetAttribute(C_ORDER)",
        error_message)) {
    cleanup();
    return false;
  }

  if (!check_cublas(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate", error_message)) {
    cleanup();
    return false;
  }
  std::size_t workspace_bytes = g_session.cublas_lt_workspace_bytes;
  if (!check_cublas(
        cublasLtMatmulPreferenceSetAttribute(
          pref,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_bytes,
          sizeof(workspace_bytes)),
        "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE_BYTES)",
        error_message)) {
    cleanup();
    return false;
  }

  cublasLtMatmulHeuristicResult_t heuristic_result{};
  int returned_results = 0;
  if (!check_cublas(
        cublasLtMatmulAlgoGetHeuristic(
          g_session.cublas_lt,
          op_desc,
          a_desc,
          b_desc,
          c_desc,
          c_desc,
          pref,
          1,
          &heuristic_result,
          &returned_results),
        "cublasLtMatmulAlgoGetHeuristic",
        error_message)) {
    cleanup();
    return false;
  }
  if (returned_results <= 0) {
    error_message = "cublasLtMatmulAlgoGetHeuristic did not return a valid algorithm for matmul.";
    cleanup();
    return false;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const bool ok = check_cublas(
    cublasLtMatmul(
      g_session.cublas_lt,
      op_desc,
      &alpha,
      matrix.data,
      a_desc,
      input.data,
      b_desc,
      &beta,
      output.data,
      c_desc,
      output.data,
      c_desc,
      &heuristic_result.algo,
      g_session.cublas_lt_workspace,
      g_session.cublas_lt_workspace_bytes,
      active_stream()),
    "cublasLtMatmul(matmul)",
    error_message);
  cleanup();
  return ok;
}

void set_prefer_bf16_matvec(const bool enabled) {
  g_session.prefer_bf16_matvec = enabled;
}

bool gather_matrix_row_f32(
  const CudaDeviceMatrixF32 & matrix,
  const int row_index,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (matrix.data == nullptr || matrix.rows <= 0 || matrix.cols <= 0) {
    error_message = "CUDA matrix is not initialized.";
    return false;
  }
  if (row_index < 0 || row_index >= matrix.rows) {
    error_message = "CUDA matrix row index is out of range.";
    return false;
  }
  if (out.data == nullptr || out.count < static_cast<std::size_t>(matrix.cols)) {
    error_message = "CUDA output buffer is invalid for gather_matrix_row_f32.";
    return false;
  }

  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA inference session stream is not initialized.";
    return false;
  }

  const int block_size = 256;
  const int grid_size = (matrix.cols + block_size - 1) / block_size;
  if (g_session.prefer_bf16_matvec && matrix.data_bf16 != nullptr) {
    gather_matrix_row_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(matrix.data_bf16),
      matrix.cols,
      row_index,
      static_cast<float *>(out.data));
    return check_cuda(cudaGetLastError(), "gather_matrix_row_bf16_kernel", error_message);
  }

  gather_matrix_row_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(matrix.data),
    matrix.cols,
    row_index,
    static_cast<float *>(out.data));
  return check_cuda(cudaGetLastError(), "gather_matrix_row_kernel", error_message);
}

bool gather_matrix_row_f32_from_token_f32(
  const CudaDeviceMatrixF32 & matrix,
  const CudaDeviceBufferF32 & token_id,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (matrix.data == nullptr || matrix.rows <= 0 || matrix.cols <= 0) {
    error_message = "CUDA matrix is not initialized.";
    return false;
  }
  if (token_id.data == nullptr || token_id.count < 1) {
    error_message = "CUDA token buffer is invalid for gather_matrix_row_f32_from_token_f32.";
    return false;
  }
  if (out.data == nullptr || out.count < static_cast<std::size_t>(matrix.cols)) {
    error_message = "CUDA output buffer is invalid for gather_matrix_row_f32_from_token_f32.";
    return false;
  }

  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA inference session stream is not initialized.";
    return false;
  }

  const int block_size = 256;
  const int grid_size = (matrix.cols + block_size - 1) / block_size;
  if (g_session.prefer_bf16_matvec && matrix.data_bf16 != nullptr) {
    gather_matrix_row_from_token_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(matrix.data_bf16),
      matrix.rows,
      matrix.cols,
      static_cast<const float *>(token_id.data),
      static_cast<float *>(out.data));
    return check_cuda(cudaGetLastError(), "gather_matrix_row_from_token_bf16_kernel", error_message);
  }

  gather_matrix_row_from_token_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(matrix.data),
    matrix.rows,
    matrix.cols,
    static_cast<const float *>(token_id.data),
    static_cast<float *>(out.data));
  return check_cuda(cudaGetLastError(), "gather_matrix_row_from_token_kernel", error_message);
}

bool allocate_buffer_f32(
  const std::size_t count,
  CudaDeviceBufferF32 & out_buffer,
  std::string & error_message) {
  if (count == 0) {
    out_buffer.data = nullptr;
    out_buffer.count = 0;
    return true;
  }

  void * device_ptr = nullptr;
  if (!check_cuda(cudaMalloc(&device_ptr, count * sizeof(float)), "cudaMalloc(buffer)", error_message)) {
    return false;
  }
  if (!check_cuda(cudaMemset(device_ptr, 0, count * sizeof(float)), "cudaMemset(buffer)", error_message)) {
    cudaFree(device_ptr);
    return false;
  }

  out_buffer.data = device_ptr;
  out_buffer.count = count;
  return true;
}

void free_buffer_f32(CudaDeviceBufferF32 & buffer) {
  if (buffer.data != nullptr) {
    cudaFree(buffer.data);
    buffer.data = nullptr;
  }
  buffer.count = 0;
}

bool upload_to_buffer_f32(
  const float * host_data,
  const std::size_t count,
  const CudaDeviceBufferF32 & buffer,
  const std::size_t buffer_offset,
  std::string & error_message) {
  if (count == 0) {
    return true;
  }
  if (buffer.data == nullptr) {
    error_message = "CUDA buffer is not initialized.";
    return false;
  }
  if (host_data == nullptr) {
    error_message = "Host data pointer is null.";
    return false;
  }
  if (buffer_offset > buffer.count || count > (buffer.count - buffer_offset)) {
    error_message = "CUDA buffer upload range is out of bounds.";
    return false;
  }

  float * dst = static_cast<float *>(buffer.data) + buffer_offset;
  const cudaStream_t stream = active_stream();
  if (stream != nullptr) {
    return tracked_memcpy_async(
      dst,
      host_data,
      count * sizeof(float),
      cudaMemcpyHostToDevice,
      stream,
      "cudaMemcpyAsync(buffer_upload)",
      error_message);
  }
  return tracked_memcpy(dst, host_data, count * sizeof(float), cudaMemcpyHostToDevice, "cudaMemcpy(buffer_upload)", error_message);
}

bool download_from_buffer_f32(
  const CudaDeviceBufferF32 & buffer,
  const std::size_t count,
  const std::size_t buffer_offset,
  std::vector<float> & out_data,
  std::string & error_message) {
  if (count == 0) {
    out_data.clear();
    return true;
  }
  if (buffer.data == nullptr) {
    error_message = "CUDA buffer is not initialized.";
    return false;
  }
  if (buffer_offset > buffer.count || count > (buffer.count - buffer_offset)) {
    error_message = "CUDA buffer download range is out of bounds.";
    return false;
  }

  out_data.resize(count);
  const float * src = static_cast<const float *>(buffer.data) + buffer_offset;
  const cudaStream_t stream = active_stream();
  if (stream != nullptr) {
    if (!tracked_memcpy_async(
          out_data.data(),
          src,
          count * sizeof(float),
          cudaMemcpyDeviceToHost,
          stream,
          "cudaMemcpyAsync(buffer_download)",
          error_message)) {
      return false;
    }
    return check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(buffer_download)", error_message);
  }
  return tracked_memcpy(
    out_data.data(),
    src,
    count * sizeof(float),
    cudaMemcpyDeviceToHost,
    "cudaMemcpy(buffer_download)",
    error_message);
}

bool run_silu_mul_f32(
  const CudaDeviceBufferF32 & a,
  const CudaDeviceBufferF32 & b,
  const std::size_t count,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (count == 0) {
    return true;
  }
  if (a.data == nullptr || b.data == nullptr || out.data == nullptr) {
    error_message = "CUDA silu_mul requires initialized buffers.";
    return false;
  }
  if (a.count < count || b.count < count || out.count < count) {
    error_message = "CUDA silu_mul buffer range is out of bounds.";
    return false;
  }

  const int block_size = 256;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size));
  const cudaStream_t stream = active_stream();
  silu_mul_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(a.data),
    static_cast<const float *>(b.data),
    static_cast<float *>(out.data),
    count);
  return check_cuda(cudaGetLastError(), "silu_mul_kernel", error_message);
}

bool run_add_f32(
  const CudaDeviceBufferF32 & a,
  const CudaDeviceBufferF32 & b,
  const std::size_t count,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (count == 0) {
    return true;
  }
  if (a.data == nullptr || b.data == nullptr || out.data == nullptr) {
    error_message = "CUDA add requires initialized buffers.";
    return false;
  }
  if (a.count < count || b.count < count || out.count < count) {
    error_message = "CUDA add buffer range is out of bounds.";
    return false;
  }
  const int block_size = 256;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size));
  const cudaStream_t stream = active_stream();
  add_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(a.data),
    static_cast<const float *>(b.data),
    static_cast<float *>(out.data),
    count);
  return check_cuda(cudaGetLastError(), "add_kernel", error_message);
}

bool run_rms_norm_f32(
  const CudaDeviceBufferF32 & input,
  const CudaDeviceBufferF32 & weight,
  const std::size_t count,
  const float eps,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (count == 0) {
    return true;
  }
  if (input.data == nullptr || weight.data == nullptr || out.data == nullptr) {
    error_message = "CUDA rms_norm requires initialized buffers.";
    return false;
  }
  if (input.count < count || weight.count < count || out.count < count) {
    error_message = "CUDA rms_norm buffer range is out of bounds.";
    return false;
  }
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    error_message = "CUDA rms_norm count exceeds kernel limits.";
    return false;
  }

  const cudaStream_t stream = active_stream();
  rms_norm_kernel<<<1, 256, 0, stream>>>(
    static_cast<const float *>(input.data),
    static_cast<const float *>(weight.data),
    static_cast<float *>(out.data),
    static_cast<int>(count),
    eps);
  return check_cuda(cudaGetLastError(), "rms_norm_kernel", error_message);
}

bool run_split_q_gate_f32(
  const CudaDeviceBufferF32 & q_gate_packed,
  const int n_heads,
  const int head_dim,
  CudaDeviceBufferF32 & out_q,
  CudaDeviceBufferF32 & out_gate,
  std::string & error_message) {
  if (n_heads <= 0 || head_dim <= 0) {
    error_message = "Invalid dimensions for split_q_gate.";
    return false;
  }
  const std::size_t packed_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim) * 2;
  const std::size_t q_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  if (q_gate_packed.data == nullptr || out_q.data == nullptr || out_gate.data == nullptr) {
    error_message = "CUDA split_q_gate requires initialized buffers.";
    return false;
  }
  if (q_gate_packed.count < packed_count || out_q.count < q_count || out_gate.count < q_count) {
    error_message = "CUDA split_q_gate buffer range is out of bounds.";
    return false;
  }

  int block_size = 1;
  while (block_size < head_dim && block_size < 256) {
    block_size <<= 1;
  }
  if (block_size < 32) {
    block_size = 32;
  }
  if (block_size > 256) {
    block_size = 256;
  }
  const cudaStream_t stream = active_stream();
  split_q_gate_kernel<<<n_heads, block_size, 0, stream>>>(
    static_cast<const float *>(q_gate_packed.data),
    static_cast<float *>(out_q.data),
    static_cast<float *>(out_gate.data),
    n_heads,
    head_dim);
  return check_cuda(cudaGetLastError(), "split_q_gate_kernel", error_message);
}

bool run_rms_norm_per_head_f32(
  const CudaDeviceBufferF32 & input,
  const CudaDeviceBufferF32 & weight,
  const int n_heads,
  const int head_dim,
  const float eps,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (n_heads <= 0 || head_dim <= 0) {
    error_message = "Invalid dimensions for rms_norm_per_head.";
    return false;
  }
  const std::size_t count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  if (input.data == nullptr || weight.data == nullptr || out.data == nullptr) {
    error_message = "CUDA rms_norm_per_head requires initialized buffers.";
    return false;
  }
  if (input.count < count || weight.count < static_cast<std::size_t>(head_dim) || out.count < count) {
    error_message = "CUDA rms_norm_per_head buffer range is out of bounds.";
    return false;
  }

  int block_size = 1;
  while (block_size < head_dim && block_size < 256) {
    block_size <<= 1;
  }
  if (block_size < 32) {
    block_size = 32;
  }
  if (block_size > 256) {
    block_size = 256;
  }
  const cudaStream_t stream = active_stream();
  rms_norm_per_head_kernel<<<n_heads, block_size, 0, stream>>>(
    static_cast<const float *>(input.data),
    static_cast<const float *>(weight.data),
    static_cast<float *>(out.data),
    n_heads,
    head_dim,
    eps);
  return check_cuda(cudaGetLastError(), "rms_norm_per_head_kernel", error_message);
}

bool run_apply_rope_inplace_f32(
  const CudaDeviceBufferF32 & values,
  const int n_heads,
  const int head_dim,
  const int rope_dim,
  const int position,
  const float rope_theta,
  std::string & error_message) {
  if (n_heads <= 0 || head_dim <= 0 || rope_dim <= 0) {
    error_message = "Invalid dimensions for apply_rope.";
    return false;
  }
  const int applied_rope_dim = std::min(rope_dim, head_dim);
  const std::size_t count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  if (values.data == nullptr || values.count < count) {
    error_message = "CUDA apply_rope buffer range is out of bounds.";
    return false;
  }
  const int pair_count = applied_rope_dim / 2;
  if (pair_count <= 0) {
    return true;
  }
  const int block_size = 128;
  const int grid_y = (pair_count + block_size - 1) / block_size;
  const dim3 grid(n_heads, grid_y, 1);
  const cudaStream_t stream = active_stream();
  rope_inplace_kernel<<<grid, block_size, 0, stream>>>(
    static_cast<float *>(values.data),
    n_heads,
    head_dim,
    applied_rope_dim,
    position,
    rope_theta);
  return check_cuda(cudaGetLastError(), "rope_inplace_kernel", error_message);
}

