#include "qwen35x/runtime/cuda_inference.h"

#if QWEN35X_HAS_CUDA

#include <cuda_runtime.h>

#include <cmath>

namespace qwen35x::cuda {

namespace {

struct InferenceSessionState {
  float * workspace_input = nullptr;
  std::size_t workspace_input_count = 0;
  float * workspace_output = nullptr;
  std::size_t workspace_output_count = 0;
  cudaStream_t stream = nullptr;
  bool active = false;
};

thread_local InferenceSessionState g_session;
CudaTransferStats g_transfer_stats;

bool check_cuda(cudaError_t status, const char * step, std::string & error_message) {
  if (status == cudaSuccess) {
    return true;
  }
  error_message = std::string(step) + " failed: " + cudaGetErrorString(status);
  return false;
}

bool record_copy(const std::size_t bytes, const cudaMemcpyKind kind) {
  ++g_transfer_stats.copy_calls;
  switch (kind) {
    case cudaMemcpyHostToDevice:
      g_transfer_stats.host_to_device_bytes += static_cast<std::uint64_t>(bytes);
      return true;
    case cudaMemcpyDeviceToHost:
      g_transfer_stats.device_to_host_bytes += static_cast<std::uint64_t>(bytes);
      return true;
    default:
      g_transfer_stats.other_bytes += static_cast<std::uint64_t>(bytes);
      return true;
  }
}

bool tracked_memcpy(
  void * dst,
  const void * src,
  const std::size_t bytes,
  const cudaMemcpyKind kind,
  const char * step,
  std::string & error_message) {
  if (!check_cuda(cudaMemcpy(dst, src, bytes, kind), step, error_message)) {
    return false;
  }
  return record_copy(bytes, kind);
}

bool tracked_memcpy_async(
  void * dst,
  const void * src,
  const std::size_t bytes,
  const cudaMemcpyKind kind,
  cudaStream_t stream,
  const char * step,
  std::string & error_message) {
  if (!check_cuda(cudaMemcpyAsync(dst, src, bytes, kind, stream), step, error_message)) {
    return false;
  }
  return record_copy(bytes, kind);
}

void release_session_storage() {
  if (g_session.workspace_input != nullptr) {
    cudaFree(g_session.workspace_input);
    g_session.workspace_input = nullptr;
  }
  if (g_session.workspace_output != nullptr) {
    cudaFree(g_session.workspace_output);
    g_session.workspace_output = nullptr;
  }
  g_session.workspace_input_count = 0;
  g_session.workspace_output_count = 0;
  if (g_session.stream != nullptr) {
    cudaStreamDestroy(g_session.stream);
    g_session.stream = nullptr;
  }
}

bool ensure_session_workspace(
  const std::size_t input_count,
  const std::size_t output_count,
  std::string & error_message) {
  if (!g_session.active) {
    if (!check_cuda(cudaStreamCreateWithFlags(&g_session.stream, cudaStreamNonBlocking), "cudaStreamCreate", error_message)) {
      return false;
    }
    g_session.active = true;
  }

  if (input_count > g_session.workspace_input_count) {
    if (g_session.workspace_input != nullptr) {
      cudaFree(g_session.workspace_input);
      g_session.workspace_input = nullptr;
      g_session.workspace_input_count = 0;
    }
    if (!check_cuda(cudaMalloc(&g_session.workspace_input, input_count * sizeof(float)), "cudaMalloc(workspace_input)", error_message)) {
      return false;
    }
    g_session.workspace_input_count = input_count;
  }

  if (output_count > g_session.workspace_output_count) {
    if (g_session.workspace_output != nullptr) {
      cudaFree(g_session.workspace_output);
      g_session.workspace_output = nullptr;
      g_session.workspace_output_count = 0;
    }
    if (!check_cuda(cudaMalloc(&g_session.workspace_output, output_count * sizeof(float)), "cudaMalloc(workspace_output)", error_message)) {
      return false;
    }
    g_session.workspace_output_count = output_count;
  }

  return true;
}

cudaStream_t active_stream() {
  if (g_session.active && g_session.stream != nullptr) {
    return g_session.stream;
  }
  return nullptr;
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

__global__ void full_attention_decode_gqa_kernel(
  const float * q,
  const float * gate,
  const float * k_cache,
  const float * v_cache,
  float * out,
  float * scratch_scores,
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
  float * scores_ptr = scratch_scores + static_cast<std::size_t>(head) * static_cast<std::size_t>(seq_len);

  __shared__ float reduction[256];
  const float scale = rsqrtf(static_cast<float>(head_dim));

  for (int t = 0; t < seq_len; ++t) {
    const float * k_ptr = k_cache +
                          (static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_stride)) +
                          (static_cast<std::size_t>(kvh) * static_cast<std::size_t>(head_dim));

    float partial = 0.0f;
    if (tid < head_dim) {
      partial = q_ptr[tid] * k_ptr[tid];
    }
    reduction[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      scores_ptr[t] = reduction[0] * scale;
    }
    __syncthreads();
  }

  if (tid == 0) {
    float max_score = -1.0e30f;
    for (int t = 0; t < seq_len; ++t) {
      max_score = fmaxf(max_score, scores_ptr[t]);
    }
    float denom = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      const float ev = expf(scores_ptr[t] - max_score);
      scores_ptr[t] = ev;
      denom += ev;
    }
    if (denom <= 0.0f) {
      denom = 1.0f;
    }
    const float inv_denom = 1.0f / denom;
    for (int t = 0; t < seq_len; ++t) {
      scores_ptr[t] *= inv_denom;
    }
  }
  __syncthreads();

  if (tid < head_dim) {
    float acc = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      const float * v_ptr = v_cache +
                            (static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_stride)) +
                            (static_cast<std::size_t>(kvh) * static_cast<std::size_t>(head_dim));
      acc += scores_ptr[t] * v_ptr[tid];
    }
    const float g = gate_ptr[tid];
    const float sig = 1.0f / (1.0f + expf(-g));
    out_ptr[tid] = acc * sig;
  }
}

} // namespace

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
  out_matrix.rows = rows;
  out_matrix.cols = cols;
  return true;
}

void free_matrix_f32(CudaDeviceMatrixF32 & matrix) {
  if (matrix.data != nullptr) {
    cudaFree(matrix.data);
    matrix.data = nullptr;
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

  const int block_size = 128;
  const int grid_size = (matrix.rows + block_size - 1) / block_size;
  const cudaStream_t stream = active_stream();
  f32_matvec_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(matrix.data),
    static_cast<const float *>(input.data),
    static_cast<float *>(output.data),
    matrix.rows,
    matrix.cols);
  return check_cuda(cudaGetLastError(), "f32_matvec_kernel(device)", error_message);
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
  const std::size_t scores_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(seq_len);
  if (q.data == nullptr || gate.data == nullptr || k_cache.data == nullptr || v_cache.data == nullptr ||
      out.data == nullptr || scratch_scores.data == nullptr) {
    error_message = "One or more CUDA buffers are not initialized for full attention decode.";
    return false;
  }
  if (q.count < q_count || gate.count < q_count || out.count < q_count || k_cache.count < kv_count ||
      v_cache.count < kv_count || scratch_scores.count < scores_count) {
    error_message = "CUDA buffer range is out of bounds for full attention decode.";
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
  full_attention_decode_gqa_kernel<<<n_heads, block_size, 0, stream>>>(
    static_cast<const float *>(q.data),
    static_cast<const float *>(gate.data),
    static_cast<const float *>(k_cache.data),
    static_cast<const float *>(v_cache.data),
    static_cast<float *>(out.data),
    static_cast<float *>(scratch_scores.data),
    n_heads,
    n_kv_heads,
    head_dim,
    seq_len);
  return check_cuda(cudaGetLastError(), "full_attention_decode_gqa_kernel", error_message);
}

void reset_transfer_stats() {
  g_transfer_stats = CudaTransferStats{};
}

void get_transfer_stats(CudaTransferStats & out_stats) {
  out_stats = g_transfer_stats;
}

} // namespace qwen35x::cuda

#endif
