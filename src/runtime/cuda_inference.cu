#include "qwen35x/runtime/cuda_inference.h"

#if QWEN35X_HAS_CUDA

#include <cuda_runtime.h>

namespace qwen35x::cuda {

namespace {

bool check_cuda(cudaError_t status, const char * step, std::string & error_message) {
  if (status == cudaSuccess) {
    return true;
  }
  error_message = std::string(step) + " failed: " + cudaGetErrorString(status);
  return false;
}

CudaTransferStats g_transfer_stats;

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

  ++g_transfer_stats.copy_calls;
  switch (kind) {
    case cudaMemcpyHostToDevice:
      g_transfer_stats.host_to_device_bytes += static_cast<std::uint64_t>(bytes);
      break;
    case cudaMemcpyDeviceToHost:
      g_transfer_stats.device_to_host_bytes += static_cast<std::uint64_t>(bytes);
      break;
    default:
      g_transfer_stats.other_bytes += static_cast<std::uint64_t>(bytes);
      break;
  }
  return true;
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

float * g_workspace_input = nullptr;
std::size_t g_workspace_input_count = 0;
float * g_workspace_output = nullptr;
std::size_t g_workspace_output_count = 0;

bool ensure_workspace(const std::size_t input_count, const std::size_t output_count, std::string & error_message) {
  if (input_count > g_workspace_input_count) {
    if (g_workspace_input != nullptr) {
      cudaFree(g_workspace_input);
      g_workspace_input = nullptr;
      g_workspace_input_count = 0;
    }
    if (!check_cuda(cudaMalloc(&g_workspace_input, input_count * sizeof(float)), "cudaMalloc(workspace_input)", error_message)) {
      return false;
    }
    g_workspace_input_count = input_count;
  }

  if (output_count > g_workspace_output_count) {
    if (g_workspace_output != nullptr) {
      cudaFree(g_workspace_output);
      g_workspace_output = nullptr;
      g_workspace_output_count = 0;
    }
    if (!check_cuda(cudaMalloc(&g_workspace_output, output_count * sizeof(float)), "cudaMalloc(workspace_output)", error_message)) {
      return false;
    }
    g_workspace_output_count = output_count;
  }

  return true;
}

} // namespace

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

  if (!ensure_workspace(input.size(), static_cast<std::size_t>(matrix.rows), error_message)) {
    return false;
  }

  if (!tracked_memcpy(
        g_workspace_input,
        input.data(),
        input.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        "cudaMemcpy(input)",
        error_message)) {
    return false;
  }

  const int block_size = 128;
  const int grid_size = (matrix.rows + block_size - 1) / block_size;
  f32_matvec_kernel<<<grid_size, block_size>>>(
    static_cast<const float *>(matrix.data), g_workspace_input, g_workspace_output, matrix.rows, matrix.cols);

  if (!check_cuda(cudaGetLastError(), "f32_matvec_kernel", error_message) ||
      !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(matvec)", error_message)) {
    return false;
  }

  output.resize(static_cast<std::size_t>(matrix.rows));
  if (!tracked_memcpy(
        output.data(),
        g_workspace_output,
        output.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        "cudaMemcpy(output)",
        error_message)) {
    return false;
  }

  return true;
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
  if (!tracked_memcpy(dst, host_data, count * sizeof(float), cudaMemcpyHostToDevice, "cudaMemcpy(buffer)", error_message)) {
    return false;
  }
  return true;
}

void reset_transfer_stats() {
  g_transfer_stats = CudaTransferStats{};
}

void get_transfer_stats(CudaTransferStats & out_stats) {
  out_stats = g_transfer_stats;
}

} // namespace qwen35x::cuda

#endif
