bool run_matvec_f32_device_cublaslt(
  const CudaDeviceMatrixF32 & matrix,
  const CudaDeviceBufferF32 & input,
  CudaDeviceBufferF32 & output,
  std::string & error_message) {
  if (g_session.cublas_lt == nullptr) {
    error_message = "cuBLASLt handle is not initialized.";
    return false;
  }

  MatvecPlan * plan = find_matvec_plan(matrix.rows, matrix.cols, false);
  if (plan == nullptr) {
    MatvecPlan new_plan;
    if (!initialize_matvec_plan(new_plan, matrix.rows, matrix.cols, false, error_message)) {
      return false;
    }
    g_session.matvec_plans.push_back(std::move(new_plan));
    plan = &g_session.matvec_plans.back();
  }
  if (!plan->has_algo) {
    error_message = "cuBLASLt matvec plan has no selected algorithm.";
    return false;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  return check_cublas(
    cublasLtMatmul(
      g_session.cublas_lt,
      plan->op_desc,
      &alpha,
      matrix.data,
      plan->a_desc,
      input.data,
      plan->b_desc,
      &beta,
      output.data,
      plan->c_desc,
      output.data,
      plan->c_desc,
      &plan->algo,
      g_session.cublas_lt_workspace,
      g_session.cublas_lt_workspace_bytes,
      active_stream()),
    "cublasLtMatmul(matvec)",
    error_message);
}

bool run_matvec_bf16_device_cublaslt(
  const CudaDeviceMatrixF32 & matrix,
  const CudaDeviceBufferF32 & input,
  CudaDeviceBufferF32 & output,
  std::string & error_message) {
  if (g_session.cublas_lt == nullptr) {
    error_message = "cuBLASLt handle is not initialized.";
    return false;
  }
  if (matrix.data_bf16 == nullptr) {
    error_message = "BF16 shadow matrix is not initialized.";
    return false;
  }
  if (!ensure_session_workspace_input_bf16(static_cast<std::size_t>(matrix.cols), error_message)) {
    return false;
  }

  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA inference session stream is not initialized.";
    return false;
  }

  const int convert_block = 256;
  const int convert_grid = (matrix.cols + convert_block - 1) / convert_block;
  f32_to_bf16_kernel<<<convert_grid, convert_block, 0, stream>>>(
    static_cast<const float *>(input.data),
    g_session.workspace_input_bf16,
    matrix.cols);
  if (!check_cuda(cudaGetLastError(), "f32_to_bf16_kernel", error_message)) {
    return false;
  }

  MatvecPlan * plan = find_matvec_plan(matrix.rows, matrix.cols, true);
  if (plan == nullptr) {
    MatvecPlan new_plan;
    if (!initialize_matvec_plan(new_plan, matrix.rows, matrix.cols, true, error_message)) {
      return false;
    }
    g_session.matvec_plans.push_back(std::move(new_plan));
    plan = &g_session.matvec_plans.back();
  }
  if (!plan->has_algo) {
    error_message = "cuBLASLt BF16 matvec plan has no selected algorithm.";
    return false;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  return check_cublas(
    cublasLtMatmul(
      g_session.cublas_lt,
      plan->op_desc,
      &alpha,
      matrix.data_bf16,
      plan->a_desc,
      g_session.workspace_input_bf16,
      plan->b_desc,
      &beta,
      output.data,
      plan->c_desc,
      output.data,
      plan->c_desc,
      &plan->algo,
      g_session.cublas_lt_workspace,
      g_session.cublas_lt_workspace_bytes,
      stream),
    "cublasLtMatmul(matvec_bf16)",
    error_message);
}

__global__ void f32_matvec_kernel(
  const float * weights,
  const float * input,
  float * output,
  int rows,
  int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float * row_ptr = weights + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  float sum = 0.0f;
  for (int c = 0; c < cols; ++c) {
    sum += row_ptr[c] * input[c];
  }
  output[row] = sum;
}

__global__ void f32_to_bf16_kernel(
  const float * input,
  __nv_bfloat16 * output,
  const int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  output[idx] = __float2bfloat16(input[idx]);
}

__global__ void gather_matrix_row_kernel(
  const float * matrix,
  const int cols,
  const int row_index,
  float * out_row) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  const std::size_t src_offset =
    static_cast<std::size_t>(row_index) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col);
  out_row[col] = matrix[src_offset];
}

__global__ void gather_matrix_row_bf16_kernel(
  const __nv_bfloat16 * matrix,
  const int cols,
  const int row_index,
  float * out_row) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  const std::size_t src_offset =
    static_cast<std::size_t>(row_index) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col);
  out_row[col] = __bfloat162float(matrix[src_offset]);
}

__global__ void gather_matrix_row_from_token_kernel(
  const float * matrix,
  const int rows,
  const int cols,
  const float * token_id,
  float * out_row) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }

  int row_index = static_cast<int>(token_id[0]);
  if (row_index < 0) {
    row_index = 0;
  } else if (row_index >= rows) {
    row_index = rows - 1;
  }

  const std::size_t src_offset =
    static_cast<std::size_t>(row_index) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col);
  out_row[col] = matrix[src_offset];
}

__global__ void gather_matrix_row_from_token_bf16_kernel(
  const __nv_bfloat16 * matrix,
  const int rows,
  const int cols,
  const float * token_id,
  float * out_row) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }

  int row_index = static_cast<int>(token_id[0]);
  if (row_index < 0) {
    row_index = 0;
  } else if (row_index >= rows) {
    row_index = rows - 1;
  }

  const std::size_t src_offset =
    static_cast<std::size_t>(row_index) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col);
  out_row[col] = __bfloat162float(matrix[src_offset]);
}

__global__ void silu_mul_kernel(
  const float * a,
  const float * b,
  float * out,
  const std::size_t count) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                          static_cast<std::size_t>(threadIdx.x);
  if (idx >= count) {
    return;
  }
  const float av = a[idx];
  const float sig = 1.0f / (1.0f + expf(-av));
  out[idx] = (av * sig) * b[idx];
}

__global__ void add_kernel(
  const float * a,
  const float * b,
  float * out,
  const std::size_t count) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                          static_cast<std::size_t>(threadIdx.x);
  if (idx >= count) {
    return;
  }
  out[idx] = a[idx] + b[idx];
}

__global__ void rms_norm_kernel(
  const float * input,
  const float * weight,
  float * out,
  const int count,
  const float eps) {
  __shared__ float reduction[256];
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (int i = tid; i < count; i += blockDim.x) {
    const float v = input[i];
    sum += v * v;
  }
  reduction[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }

  __shared__ float inv;
  if (tid == 0) {
    inv = rsqrtf((reduction[0] / static_cast<float>(count)) + eps);
  }
  __syncthreads();

  for (int i = tid; i < count; i += blockDim.x) {
    out[i] = input[i] * inv * (1.0f + weight[i]);
  }
}

__global__ void split_q_gate_kernel(
  const float * packed,
  float * out_q,
  float * out_gate,
  const int n_heads,
  const int head_dim) {
  const int h = blockIdx.x;
  const int d = threadIdx.x;
  if (h >= n_heads || d >= head_dim) {
    return;
  }
  const int q_span = head_dim * 2;
  const std::size_t src = static_cast<std::size_t>(h) * static_cast<std::size_t>(q_span) + static_cast<std::size_t>(d);
  const std::size_t dst = static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim) + static_cast<std::size_t>(d);
  out_q[dst] = packed[src];
  out_gate[dst] = packed[src + static_cast<std::size_t>(head_dim)];
}

__global__ void rms_norm_per_head_kernel(
  const float * input,
  const float * weight,
  float * out,
  const int n_heads,
  const int head_dim,
  const float eps) {
  __shared__ float reduction[256];
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  if (head >= n_heads) {
    return;
  }

  const float * in_ptr = input + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  float * out_ptr = out + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);

  float sum = 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const float v = in_ptr[d];
    sum += v * v;
  }
  reduction[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }

  __shared__ float inv;
  if (tid == 0) {
    inv = rsqrtf((reduction[0] / static_cast<float>(head_dim)) + eps);
  }
  __syncthreads();

  for (int d = tid; d < head_dim; d += blockDim.x) {
    out_ptr[d] = in_ptr[d] * inv * (1.0f + weight[d]);
  }
}

__global__ void rope_inplace_kernel(
  float * values,
  const int n_heads,
  const int head_dim,
  const int rope_dim,
  const int position,
  const float rope_theta) {
  const int head = blockIdx.x;
  const int pair_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (head >= n_heads) {
    return;
  }
  const int half = rope_dim / 2;
  if (pair_idx >= half) {
    return;
  }

  const std::size_t base = static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  float * ptr = values + base;
  const int i0 = pair_idx;
  const int i1 = pair_idx + half;
  if (i1 >= head_dim) {
    return;
  }
  const float x0 = ptr[i0];
  const float x1 = ptr[i1];
  const float inv_freq = expf(
    -logf(rope_theta) * (static_cast<float>(2 * pair_idx) / static_cast<float>(rope_dim)));
  const float angle = static_cast<float>(position) * inv_freq;
  const float c = cosf(angle);
  const float s = sinf(angle);
  ptr[i0] = x0 * c - x1 * s;
  ptr[i1] = x1 * c + x0 * s;
}

__global__ void prepare_full_attention_qkv_kernel(
  const float * q_gate_packed,
  const float * k_raw,
  const float * v_raw,
  const float * q_norm_weight,
  const float * k_norm_weight,
  float * out_q,
  float * out_gate,
  float * k_cache,
  float * v_cache,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int applied_rope_dim,
  const int position,
  const float rope_theta,
  const float eps,
  const unsigned long long k_cache_offset,
  const unsigned long long v_cache_offset) {
  __shared__ float reduction[256];
  __shared__ float inv_norm;

  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  if (head >= n_heads) {
    return;
  }

  const int pair_count = applied_rope_dim / 2;
  const std::size_t q_head_base = static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  const std::size_t packed_base = q_head_base * 2;

  // Split Q/G, apply per-head RMSNorm on Q, then RoPE on Q.
  float q_sq_sum = 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const float qv = q_gate_packed[packed_base + static_cast<std::size_t>(d)];
    q_sq_sum += qv * qv;
  }
  reduction[tid] = q_sq_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    inv_norm = rsqrtf((reduction[0] / static_cast<float>(head_dim)) + eps);
  }
  __syncthreads();

  for (int d = tid; d < head_dim; d += blockDim.x) {
    const std::size_t idx = q_head_base + static_cast<std::size_t>(d);
    const float qv = q_gate_packed[packed_base + static_cast<std::size_t>(d)];
    const float gv = q_gate_packed[packed_base + static_cast<std::size_t>(head_dim) + static_cast<std::size_t>(d)];
    out_q[idx] = qv * inv_norm * (1.0f + q_norm_weight[d]);
    out_gate[idx] = gv;
  }
  __syncthreads();

  for (int pair_idx = tid; pair_idx < pair_count; pair_idx += blockDim.x) {
    const int i0 = pair_idx;
    const int i1 = pair_idx + pair_count;
    if (i1 >= head_dim) {
      continue;
    }
    const float inv_freq = expf(
      -logf(rope_theta) * (static_cast<float>(2 * pair_idx) / static_cast<float>(applied_rope_dim)));
    const float angle = static_cast<float>(position) * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const std::size_t q0_idx = q_head_base + static_cast<std::size_t>(i0);
    const std::size_t q1_idx = q_head_base + static_cast<std::size_t>(i1);
    const float x0 = out_q[q0_idx];
    const float x1 = out_q[q1_idx];
    out_q[q0_idx] = x0 * c - x1 * s;
    out_q[q1_idx] = x1 * c + x0 * s;
  }
  __syncthreads();

  // For KV heads, apply RMSNorm+RoPE on K and write K/V directly into caches.
  if (head >= n_kv_heads) {
    return;
  }

  const std::size_t kv_head_base = static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);

  float k_sq_sum = 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const float kv = k_raw[kv_head_base + static_cast<std::size_t>(d)];
    k_sq_sum += kv * kv;
  }
  reduction[tid] = k_sq_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    inv_norm = rsqrtf((reduction[0] / static_cast<float>(head_dim)) + eps);
  }
  __syncthreads();

  const std::size_t k_cache_base = static_cast<std::size_t>(k_cache_offset) + kv_head_base;
  const std::size_t v_cache_base = static_cast<std::size_t>(v_cache_offset) + kv_head_base;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const std::size_t local_idx = kv_head_base + static_cast<std::size_t>(d);
    k_cache[k_cache_base + static_cast<std::size_t>(d)] =
      k_raw[local_idx] * inv_norm * (1.0f + k_norm_weight[d]);
    v_cache[v_cache_base + static_cast<std::size_t>(d)] = v_raw[local_idx];
  }
  __syncthreads();

  for (int pair_idx = tid; pair_idx < pair_count; pair_idx += blockDim.x) {
    const int i0 = pair_idx;
    const int i1 = pair_idx + pair_count;
    if (i1 >= head_dim) {
      continue;
    }
    const float inv_freq = expf(
      -logf(rope_theta) * (static_cast<float>(2 * pair_idx) / static_cast<float>(applied_rope_dim)));
    const float angle = static_cast<float>(position) * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const std::size_t k0_idx = k_cache_base + static_cast<std::size_t>(i0);
    const std::size_t k1_idx = k_cache_base + static_cast<std::size_t>(i1);
    const float x0 = k_cache[k0_idx];
    const float x1 = k_cache[k1_idx];
    k_cache[k0_idx] = x0 * c - x1 * s;
    k_cache[k1_idx] = x1 * c + x0 * s;
  }
}

__device__ __forceinline__ float adjust_sampling_logit(
  const float raw,
  const float seen_mask,
  const float repetition_penalty,
  const float temperature) {
  float value = raw;
  if (repetition_penalty > 1.0f && seen_mask > 0.0f) {
    if (value > 0.0f) {
      value /= repetition_penalty;
    } else {
      value *= repetition_penalty;
    }
  }
  if (temperature > 0.0f) {
    value /= temperature;
  }
  return value;
}

__global__ void argmax_blocks_kernel(
  const float * logits,
  const float * seen_token_mask,
  const int vocab_size,
  const float repetition_penalty,
  float * block_values,
  int * block_indices) {
  __shared__ float sh_values[kSamplingArgmaxBlockSize];
  __shared__ int sh_indices[kSamplingArgmaxBlockSize];

  const int tid = threadIdx.x;
  const int block_start = blockIdx.x * kSamplingArgmaxChunkSize;
  const int block_end = min(vocab_size, block_start + kSamplingArgmaxChunkSize);

  float best_value = -1.0e30f;
  int best_index = -1;
  for (int idx = block_start + tid; idx < block_end; idx += blockDim.x) {
    const float value = adjust_sampling_logit(logits[idx], seen_token_mask[idx], repetition_penalty, 0.0f);
    if (value > best_value || (value == best_value && (best_index < 0 || idx < best_index))) {
      best_value = value;
      best_index = idx;
    }
  }

  sh_values[tid] = best_value;
  sh_indices[tid] = best_index;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_value = sh_values[tid + stride];
      const int other_index = sh_indices[tid + stride];
      if (other_value > sh_values[tid] ||
          (other_value == sh_values[tid] && (sh_indices[tid] < 0 || (other_index >= 0 && other_index < sh_indices[tid])))) {
        sh_values[tid] = other_value;
        sh_indices[tid] = other_index;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_values[blockIdx.x] = sh_values[0];
    block_indices[blockIdx.x] = sh_indices[0];
  }
}

__global__ void finalize_argmax_token_kernel(
  const float * block_values,
  const int * block_indices,
  const int block_count,
  float * seen_token_mask,
  const int vocab_size,
  float * out_token) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  float best_value = -1.0e30f;
  int best_index = -1;
  for (int i = 0; i < block_count; ++i) {
    const float value = block_values[i];
    const int idx = block_indices[i];
    if (idx < 0) {
      continue;
    }
    if (value > best_value || (value == best_value && (best_index < 0 || idx < best_index))) {
      best_value = value;
      best_index = idx;
    }
  }

  if (best_index < 0 || best_index >= vocab_size) {
    out_token[0] = -1.0f;
    return;
  }
  seen_token_mask[best_index] = 1.0f;
  out_token[0] = static_cast<float>(best_index);
}

__global__ void topk_blocks_kernel(
  const float * logits,
  const float * seen_token_mask,
  const int vocab_size,
  const float temperature,
  const float repetition_penalty,
  const int top_k,
  float * block_top_values,
  int * block_top_indices) {
  if (threadIdx.x != 0) {
    return;
  }

  const int block_start = blockIdx.x * kSamplingArgmaxChunkSize;
  const int block_end = min(vocab_size, block_start + kSamplingArgmaxChunkSize);
  const int base = blockIdx.x * kGpuSamplingMaxTopK;

  float local_values[kGpuSamplingMaxTopK];
  int local_indices[kGpuSamplingMaxTopK];
  for (int i = 0; i < top_k; ++i) {
    local_values[i] = -1.0e30f;
    local_indices[i] = -1;
  }

  for (int idx = block_start; idx < block_end; ++idx) {
    const float value =
      adjust_sampling_logit(logits[idx], seen_token_mask[idx], repetition_penalty, temperature);
    int min_slot = 0;
    float min_value = local_values[0];
    for (int i = 1; i < top_k; ++i) {
      if (local_values[i] < min_value) {
        min_value = local_values[i];
        min_slot = i;
      }
    }
    if (value > min_value) {
      local_values[min_slot] = value;
      local_indices[min_slot] = idx;
    }
  }

  for (int i = 0; i + 1 < top_k; ++i) {
    int best = i;
    for (int j = i + 1; j < top_k; ++j) {
      if (local_values[j] > local_values[best]) {
        best = j;
      }
    }
    if (best != i) {
      const float tmp_v = local_values[i];
      local_values[i] = local_values[best];
      local_values[best] = tmp_v;
      const int tmp_i = local_indices[i];
      local_indices[i] = local_indices[best];
      local_indices[best] = tmp_i;
    }
  }

  for (int i = 0; i < top_k; ++i) {
    block_top_values[base + i] = local_values[i];
    block_top_indices[base + i] = local_indices[i];
  }
}

__global__ void sample_token_from_block_topk_kernel(
  const float * block_top_values,
  const int * block_top_indices,
  const int block_count,
  const int vocab_size,
  const float top_p,
  const int top_k,
  const float random_u01,
  float * seen_token_mask,
  float * out_token) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  float global_values[kGpuSamplingMaxTopK];
  int global_indices[kGpuSamplingMaxTopK];
  for (int i = 0; i < top_k; ++i) {
    global_values[i] = -1.0e30f;
    global_indices[i] = -1;
  }

  const int candidate_count = block_count * top_k;
  for (int c = 0; c < candidate_count; ++c) {
    const int idx = block_top_indices[c];
    if (idx < 0 || idx >= vocab_size) {
      continue;
    }
    const float value = block_top_values[c];
    int min_slot = 0;
    float min_value = global_values[0];
    for (int i = 1; i < top_k; ++i) {
      if (global_values[i] < min_value) {
        min_value = global_values[i];
        min_slot = i;
      }
    }
    if (value > min_value) {
      global_values[min_slot] = value;
      global_indices[min_slot] = idx;
    }
  }

  for (int i = 0; i + 1 < top_k; ++i) {
    int best = i;
    for (int j = i + 1; j < top_k; ++j) {
      if (global_values[j] > global_values[best]) {
        best = j;
      }
    }
    if (best != i) {
      const float tmp_v = global_values[i];
      global_values[i] = global_values[best];
      global_values[best] = tmp_v;
      const int tmp_i = global_indices[i];
      global_indices[i] = global_indices[best];
      global_indices[best] = tmp_i;
    }
  }

  int valid_count = 0;
  float max_logit = -1.0e30f;
  for (int i = 0; i < top_k; ++i) {
    if (global_indices[i] < 0) {
      continue;
    }
    ++valid_count;
    if (global_values[i] > max_logit) {
      max_logit = global_values[i];
    }
  }
  if (valid_count <= 0) {
    out_token[0] = -1.0f;
    return;
  }

  float probs[kGpuSamplingMaxTopK];
  float prob_sum = 0.0f;
  for (int i = 0; i < valid_count; ++i) {
    probs[i] = expf(global_values[i] - max_logit);
    prob_sum += probs[i];
  }
  if (prob_sum <= 0.0f) {
    const int fallback = global_indices[0];
    if (fallback < 0 || fallback >= vocab_size) {
      out_token[0] = -1.0f;
      return;
    }
    seen_token_mask[fallback] = 1.0f;
    out_token[0] = static_cast<float>(fallback);
    return;
  }
  for (int i = 0; i < valid_count; ++i) {
    probs[i] /= prob_sum;
  }

  int keep = valid_count;
  if (top_p < 1.0f) {
    float cumulative = 0.0f;
    keep = 0;
    for (int i = 0; i < valid_count; ++i) {
      cumulative += probs[i];
      ++keep;
      if (cumulative >= top_p && keep >= 1) {
        break;
      }
    }
    if (keep <= 0) {
      keep = 1;
    }
  }

  float keep_sum = 0.0f;
  for (int i = 0; i < keep; ++i) {
    keep_sum += probs[i];
  }
  if (keep_sum <= 0.0f) {
    const int fallback = global_indices[0];
    if (fallback < 0 || fallback >= vocab_size) {
      out_token[0] = -1.0f;
      return;
    }
    seen_token_mask[fallback] = 1.0f;
    out_token[0] = static_cast<float>(fallback);
    return;
  }

  float clamped_u = random_u01;
  if (clamped_u < 0.0f) {
    clamped_u = 0.0f;
  }
  if (clamped_u >= 1.0f) {
    clamped_u = 0.99999994f;
  }
  const float threshold = clamped_u * keep_sum;
  float cumulative = 0.0f;
  int sampled = global_indices[keep - 1];
  for (int i = 0; i < keep; ++i) {
    cumulative += probs[i];
    if (threshold <= cumulative) {
      sampled = global_indices[i];
      break;
    }
  }

  if (sampled < 0 || sampled >= vocab_size) {
    out_token[0] = -1.0f;
    return;
  }
  seen_token_mask[sampled] = 1.0f;
  out_token[0] = static_cast<float>(sampled);
}

__global__ void full_attention_decode_gqa_kernel_shared(
  const float * q,
  const float * gate,
  const float * k_cache,
  const float * v_cache,
  float * out,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int seq_len) {
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  if (head >= n_heads || seq_len <= 0) {
    return;
  }

  const int n_rep = n_heads / n_kv_heads;
  const int kvh = head / n_rep;
  const int kv_stride = n_kv_heads * head_dim;

  const float * q_ptr = q + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  const float * gate_ptr = gate + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  float * out_ptr = out + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);

  __shared__ float reduction[256];
  __shared__ float sh_alpha;
  __shared__ float sh_beta;
  __shared__ float sh_denom;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const float q_value = (tid < head_dim) ? q_ptr[tid] : 0.0f;
  float acc = 0.0f;
  float running_max = -1.0e30f;
  float running_denom = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    const float * k_ptr = k_cache +
                          (static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_stride)) +
                          (static_cast<std::size_t>(kvh) * static_cast<std::size_t>(head_dim));
    const float * v_ptr = v_cache +
                          (static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_stride)) +
                          (static_cast<std::size_t>(kvh) * static_cast<std::size_t>(head_dim));

    float partial = (tid < head_dim) ? (q_value * k_ptr[tid]) : 0.0f;
    reduction[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float score = reduction[0] * scale;
      const float next_max = fmaxf(running_max, score);
      const float alpha = (running_max <= -1.0e29f) ? 0.0f : expf(running_max - next_max);
      const float beta = expf(score - next_max);
      running_denom = running_denom * alpha + beta;
      running_max = next_max;
      sh_alpha = alpha;
      sh_beta = beta;
      sh_denom = running_denom;
    }
    __syncthreads();
    if (tid < head_dim) {
      acc = acc * sh_alpha + sh_beta * v_ptr[tid];
    }
    __syncthreads();
  }

  if (tid < head_dim) {
    const float denom = (sh_denom > 0.0f) ? sh_denom : 1.0f;
    const float g = gate_ptr[tid];
    const float sig = 1.0f / (1.0f + expf(-g));
    out_ptr[tid] = (acc / denom) * sig;
  }
}

__device__ float sigmoidf_device(const float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ float softplusf_device(const float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return expf(x);
  }
  return log1pf(expf(x));
}

__global__ void linear_conv_update_kernel(
  const float * mixed_qkv,
  const float * conv_weights,
  float * conv_state,
  float * conv_out,
  const int channels,
  const int kernel_size) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }

  const int history = kernel_size - 1;
  const float * w_ptr = conv_weights + static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel_size);
  float sum = 0.0f;
  for (int k = 0; k < history; ++k) {
    sum += conv_state[static_cast<std::size_t>(k) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)] * w_ptr[k];
  }
  sum += mixed_qkv[c] * w_ptr[history];
  conv_out[c] = sum * sigmoidf_device(sum);

  for (int r = 0; r + 1 < history; ++r) {
    conv_state[static_cast<std::size_t>(r) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)] =
      conv_state[static_cast<std::size_t>(r + 1) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)];
  }
  if (history > 0) {
    conv_state[static_cast<std::size_t>(history - 1) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)] =
      mixed_qkv[c];
  }
}

__global__ void linear_recurrent_decode_kernel(
  const float * conv_out,
  const float * z,
  const float * b,
  const float * a,
  const float * norm,
  const float * dt_bias,
  const float * ssm_a,
  float * recurrent_state,
  float * out_gated_norm,
  const int linear_num_k_heads,
  const int linear_num_v_heads,
  const int head_k_dim,
  const int head_v_dim,
  const float rms_eps) {
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  if (h >= linear_num_v_heads) {
    return;
  }

  const int q_dim = linear_num_k_heads * head_k_dim;
  const int v_dim = linear_num_v_heads * head_v_dim;
  const int q_base = h * head_k_dim;
  const int v_base = h * head_v_dim;
  if (q_base + head_k_dim > q_dim || v_base + head_v_dim > v_dim) {
    return;
  }

  float * s = recurrent_state +
              static_cast<std::size_t>(h) * static_cast<std::size_t>(head_v_dim) * static_cast<std::size_t>(head_v_dim);
  const float * q_ptr = conv_out + q_base;
  const float * k_ptr = conv_out + q_dim + q_base;
  const float * v_ptr = conv_out + 2 * q_dim + v_base;
  const float * z_ptr = z + v_base;
  float * out_ptr = out_gated_norm + v_base;

  __shared__ float sh_inv_q;
  __shared__ float sh_inv_k;
  __shared__ float sh_alpha;
  __shared__ float sh_beta;
  __shared__ float sh_inv_norm;
  extern __shared__ float shared[];
  float * sh_sk = shared;
  float * sh_core = shared + head_v_dim;

  if (tid == 0) {
    float sq_q = 0.0f;
    float sq_k = 0.0f;
    for (int d = 0; d < head_k_dim; ++d) {
      sq_q += q_ptr[d] * q_ptr[d];
      sq_k += k_ptr[d] * k_ptr[d];
    }
    sh_inv_q = rsqrtf(sq_q + 1.0e-6f) * rsqrtf(static_cast<float>(head_k_dim));
    sh_inv_k = rsqrtf(sq_k + 1.0e-6f);
    sh_beta = sigmoidf_device(b[h]);
    const float pre_gate = softplusf_device(a[h] + dt_bias[h]);
    sh_alpha = expf(pre_gate * ssm_a[h]);
  }
  __syncthreads();

  const int state_count = head_v_dim * head_v_dim;
  for (int idx = tid; idx < state_count; idx += blockDim.x) {
    s[idx] *= sh_alpha;
  }
  __syncthreads();

  for (int i = tid; i < head_v_dim; i += blockDim.x) {
    float sk = 0.0f;
    const float * row = s + static_cast<std::size_t>(i) * static_cast<std::size_t>(head_v_dim);
    for (int j = 0; j < head_v_dim; ++j) {
      sk += row[j] * (k_ptr[j] * sh_inv_k);
    }
    sh_sk[i] = sk;
  }
  __syncthreads();

  for (int i = tid; i < head_v_dim; i += blockDim.x) {
    const float delta = (v_ptr[i] - sh_sk[i]) * sh_beta;
    float * row = s + static_cast<std::size_t>(i) * static_cast<std::size_t>(head_v_dim);
    for (int j = 0; j < head_v_dim; ++j) {
      row[j] += delta * (k_ptr[j] * sh_inv_k);
    }
  }
  __syncthreads();

  for (int i = tid; i < head_v_dim; i += blockDim.x) {
    float core = 0.0f;
    const float * row = s + static_cast<std::size_t>(i) * static_cast<std::size_t>(head_v_dim);
    for (int j = 0; j < head_v_dim; ++j) {
      core += row[j] * (q_ptr[j] * sh_inv_q);
    }
    sh_core[i] = core;
  }
  __syncthreads();

  if (tid == 0) {
    float sq_sum = 0.0f;
    for (int i = 0; i < head_v_dim; ++i) {
      sq_sum += sh_core[i] * sh_core[i];
    }
    sh_inv_norm = rsqrtf(sq_sum / static_cast<float>(head_v_dim) + rms_eps);
  }
  __syncthreads();

  for (int i = tid; i < head_v_dim; i += blockDim.x) {
    const float zv = z_ptr[i];
    out_ptr[i] = sh_core[i] * sh_inv_norm * norm[i] * (zv * sigmoidf_device(zv));
  }
}

} // namespace
