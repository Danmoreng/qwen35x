bool run_prepare_full_attention_qkv_f32(
  const CudaDeviceBufferF32 & q_gate_packed,
  const CudaDeviceBufferF32 & k_raw,
  const CudaDeviceBufferF32 & v_raw,
  const CudaDeviceBufferF32 & q_norm_weight,
  const CudaDeviceBufferF32 & k_norm_weight,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int rope_dim,
  const int position,
  const float rope_theta,
  const float eps,
  CudaDeviceBufferF32 & out_q,
  CudaDeviceBufferF32 & out_gate,
  CudaDeviceBufferF32 & k_cache,
  const std::size_t k_cache_offset,
  CudaDeviceBufferF32 & v_cache,
  const std::size_t v_cache_offset,
  std::string & error_message) {
  if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
    error_message = "Invalid dimensions for full-attention QKV prepare.";
    return false;
  }
  if (n_heads % n_kv_heads != 0) {
    error_message = "n_heads must be divisible by n_kv_heads for GQA.";
    return false;
  }
  const int applied_rope_dim = std::min(rope_dim, head_dim);
  const std::size_t q_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  const std::size_t q_packed_count = q_count * 2;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv_heads) * static_cast<std::size_t>(head_dim);
  if (q_gate_packed.data == nullptr || k_raw.data == nullptr || v_raw.data == nullptr || q_norm_weight.data == nullptr ||
      k_norm_weight.data == nullptr || out_q.data == nullptr || out_gate.data == nullptr || k_cache.data == nullptr ||
      v_cache.data == nullptr) {
    error_message = "CUDA full-attention QKV prepare requires initialized buffers.";
    return false;
  }
  if (q_gate_packed.count < q_packed_count || k_raw.count < kv_count || v_raw.count < kv_count || out_q.count < q_count ||
      out_gate.count < q_count || q_norm_weight.count < static_cast<std::size_t>(head_dim) ||
      k_norm_weight.count < static_cast<std::size_t>(head_dim)) {
    error_message = "CUDA full-attention QKV prepare buffer range is out of bounds.";
    return false;
  }
  if (k_cache_offset > k_cache.count || kv_count > (k_cache.count - k_cache_offset) ||
      v_cache_offset > v_cache.count || kv_count > (v_cache.count - v_cache_offset)) {
    error_message = "CUDA full-attention QKV prepare cache range is out of bounds.";
    return false;
  }

  constexpr int block_size = 256;
  const cudaStream_t stream = active_stream();
  prepare_full_attention_qkv_kernel<<<n_heads, block_size, 0, stream>>>(
    static_cast<const float *>(q_gate_packed.data),
    static_cast<const float *>(k_raw.data),
    static_cast<const float *>(v_raw.data),
    static_cast<const float *>(q_norm_weight.data),
    static_cast<const float *>(k_norm_weight.data),
    static_cast<float *>(out_q.data),
    static_cast<float *>(out_gate.data),
    static_cast<float *>(k_cache.data),
    static_cast<float *>(v_cache.data),
    n_heads,
    n_kv_heads,
    head_dim,
    applied_rope_dim,
    position,
    rope_theta,
    eps,
    static_cast<unsigned long long>(k_cache_offset),
    static_cast<unsigned long long>(v_cache_offset));
  return check_cuda(cudaGetLastError(), "prepare_full_attention_qkv_kernel", error_message);
}

bool copy_buffer_f32(
  const CudaDeviceBufferF32 & src,
  const std::size_t count,
  const std::size_t src_offset,
  const CudaDeviceBufferF32 & dst,
  const std::size_t dst_offset,
  std::string & error_message) {
  if (count == 0) {
    return true;
  }
  if (src.data == nullptr || dst.data == nullptr) {
    error_message = "CUDA copy_buffer requires initialized buffers.";
    return false;
  }
  if (src_offset > src.count || count > (src.count - src_offset) || dst_offset > dst.count || count > (dst.count - dst_offset)) {
    error_message = "CUDA copy_buffer range is out of bounds.";
    return false;
  }
  const float * src_ptr = static_cast<const float *>(src.data) + src_offset;
  float * dst_ptr = static_cast<float *>(dst.data) + dst_offset;
  const cudaStream_t stream = active_stream();
  if (stream != nullptr) {
    return tracked_memcpy_async(
      dst_ptr,
      src_ptr,
      count * sizeof(float),
      cudaMemcpyDeviceToDevice,
      stream,
      "cudaMemcpyAsync(buffer_copy_d2d)",
      error_message);
  }
  return tracked_memcpy(
    dst_ptr,
    src_ptr,
    count * sizeof(float),
    cudaMemcpyDeviceToDevice,
    "cudaMemcpy(buffer_copy_d2d)",
    error_message);
}

bool run_full_attention_decode_gqa(
  const CudaDeviceBufferF32 & q,
  const CudaDeviceBufferF32 & gate,
  const CudaDeviceBufferF32 & k_cache,
  const CudaDeviceBufferF32 & v_cache,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int seq_len,
  CudaDeviceBufferF32 & out,
  CudaDeviceBufferF32 & scratch_scores,
  std::string & error_message) {
  if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || seq_len <= 0) {
    error_message = "Invalid dimensions for full attention decode.";
    return false;
  }
  if ((n_heads % n_kv_heads) != 0) {
    error_message = "Invalid GQA ratio for full attention decode.";
    return false;
  }

  const std::size_t q_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv_heads) * static_cast<std::size_t>(head_dim);
  const std::size_t kv_count = static_cast<std::size_t>(seq_len) * kv_stride;
  if (q.data == nullptr || gate.data == nullptr || k_cache.data == nullptr || v_cache.data == nullptr ||
      out.data == nullptr) {
    error_message = "One or more CUDA buffers are not initialized for full attention decode.";
    return false;
  }
  if (q.count < q_count || gate.count < q_count || out.count < q_count || k_cache.count < kv_count ||
      v_cache.count < kv_count) {
    error_message = "CUDA buffer range is out of bounds for full attention decode.";
    return false;
  }
  (void)scratch_scores;

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
  full_attention_decode_gqa_kernel_shared<<<n_heads, block_size, 0, stream>>>(
    static_cast<const float *>(q.data),
    static_cast<const float *>(gate.data),
    static_cast<const float *>(k_cache.data),
    static_cast<const float *>(v_cache.data),
    static_cast<float *>(out.data),
    n_heads,
    n_kv_heads,
    head_dim,
    seq_len);
  return check_cuda(cudaGetLastError(), "full_attention_decode_gqa_kernel_shared", error_message);
}

bool run_linear_attention_decode(
  const CudaDeviceBufferF32 & mixed_qkv,
  const CudaDeviceBufferF32 & z,
  const CudaDeviceBufferF32 & b,
  const CudaDeviceBufferF32 & a,
  const CudaDeviceMatrixF32 & conv1d,
  const CudaDeviceBufferF32 & norm,
  const CudaDeviceBufferF32 & dt_bias,
  const CudaDeviceBufferF32 & ssm_a,
  const int linear_kernel,
  const int linear_num_k_heads,
  const int linear_num_v_heads,
  const int linear_head_k_dim,
  const int linear_head_v_dim,
  const float rms_eps,
  CudaDeviceBufferF32 & conv_state,
  CudaDeviceBufferF32 & recurrent_state,
  CudaDeviceBufferF32 & scratch_conv_out,
  CudaDeviceBufferF32 & out_gated_norm,
  std::string & error_message) {
  if (linear_kernel <= 1 || linear_num_k_heads <= 0 || linear_num_v_heads <= 0 || linear_head_k_dim <= 0 ||
      linear_head_v_dim <= 0) {
    error_message = "Invalid dimensions for linear attention decode.";
    return false;
  }
  if (linear_head_k_dim != linear_head_v_dim) {
    error_message = "Linear attention decode expects head_k_dim == head_v_dim.";
    return false;
  }

  const int linear_q_dim = linear_num_k_heads * linear_head_k_dim;
  const int linear_v_dim = linear_num_v_heads * linear_head_v_dim;
  const int linear_conv_channels = linear_q_dim * 2 + linear_v_dim;
  const int conv_hist = linear_kernel - 1;

  if (mixed_qkv.data == nullptr || z.data == nullptr || b.data == nullptr || a.data == nullptr ||
      conv1d.data == nullptr || norm.data == nullptr || dt_bias.data == nullptr || ssm_a.data == nullptr ||
      conv_state.data == nullptr || recurrent_state.data == nullptr || scratch_conv_out.data == nullptr ||
      out_gated_norm.data == nullptr) {
    error_message = "One or more CUDA buffers are not initialized for linear attention decode.";
    return false;
  }
  if (mixed_qkv.count < static_cast<std::size_t>(linear_conv_channels) || z.count < static_cast<std::size_t>(linear_v_dim) ||
      b.count < static_cast<std::size_t>(linear_num_v_heads) || a.count < static_cast<std::size_t>(linear_num_v_heads) ||
      norm.count < static_cast<std::size_t>(linear_head_v_dim) || dt_bias.count < static_cast<std::size_t>(linear_num_v_heads) ||
      ssm_a.count < static_cast<std::size_t>(linear_num_v_heads) ||
      conv_state.count < static_cast<std::size_t>(conv_hist * linear_conv_channels) ||
      recurrent_state.count <
        static_cast<std::size_t>(linear_num_v_heads) * static_cast<std::size_t>(linear_head_v_dim) *
          static_cast<std::size_t>(linear_head_v_dim) ||
      scratch_conv_out.count < static_cast<std::size_t>(linear_conv_channels) ||
      out_gated_norm.count < static_cast<std::size_t>(linear_v_dim)) {
    error_message = "CUDA buffer range is out of bounds for linear attention decode.";
    return false;
  }
  if (conv1d.rows != linear_conv_channels || conv1d.cols != linear_kernel) {
    error_message = "conv1d device matrix dimensions do not match linear attention dimensions.";
    return false;
  }

  const cudaStream_t stream = active_stream();
  const int conv_block = 256;
  const int conv_grid = (linear_conv_channels + conv_block - 1) / conv_block;
  linear_conv_update_kernel<<<conv_grid, conv_block, 0, stream>>>(
    static_cast<const float *>(mixed_qkv.data),
    static_cast<const float *>(conv1d.data),
    static_cast<float *>(conv_state.data),
    static_cast<float *>(scratch_conv_out.data),
    linear_conv_channels,
    linear_kernel);
  if (!check_cuda(cudaGetLastError(), "linear_conv_update_kernel", error_message)) {
    return false;
  }

  int block_size = 1;
  while (block_size < linear_head_v_dim && block_size < 256) {
    block_size <<= 1;
  }
  if (block_size < 32) {
    block_size = 32;
  }
  if (block_size > 256) {
    block_size = 256;
  }
  const std::size_t shared_bytes = static_cast<std::size_t>(2 * linear_head_v_dim) * sizeof(float);
  linear_recurrent_decode_kernel<<<linear_num_v_heads, block_size, shared_bytes, stream>>>(
    static_cast<const float *>(scratch_conv_out.data),
    static_cast<const float *>(z.data),
    static_cast<const float *>(b.data),
    static_cast<const float *>(a.data),
    static_cast<const float *>(norm.data),
    static_cast<const float *>(dt_bias.data),
    static_cast<const float *>(ssm_a.data),
    static_cast<float *>(recurrent_state.data),
    static_cast<float *>(out_gated_norm.data),
    linear_num_k_heads,
    linear_num_v_heads,
    linear_head_k_dim,
    linear_head_v_dim,
    rms_eps);
  return check_cuda(cudaGetLastError(), "linear_recurrent_decode_kernel", error_message);
}

