#include "qwen35x/runtime/cuda_bench.h"

#if QWEN35X_HAS_CUDA

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace qwen35x::cuda {

namespace {

__global__ void bf16_matvec_kernel(
  const __nv_bfloat16 * weights,
  const float * input,
  float * output,
  int rows,
  int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const __nv_bfloat16 * row_ptr = weights + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  float sum = 0.0f;
  for (int c = 0; c < cols; ++c) {
    sum += __bfloat162float(row_ptr[c]) * input[c];
  }
  output[row] = sum;
}

bool check_cuda(cudaError_t status, const char * step, std::string & error_message) {
  if (status == cudaSuccess) {
    return true;
  }
  error_message = std::string(step) + " failed: " + cudaGetErrorString(status);
  return false;
}

} // namespace

bool run_bf16_matvec_benchmark(
  const std::vector<std::uint16_t> & weights,
  int rows,
  int cols,
  int warmup_iterations,
  int benchmark_iterations,
  double & avg_iteration_ms,
  std::string & error_message) {
  if (rows <= 0 || cols <= 0) {
    error_message = "Invalid matrix dimensions for BF16 benchmark.";
    return false;
  }
  if (benchmark_iterations <= 0) {
    error_message = "benchmark_iterations must be > 0.";
    return false;
  }

  const std::size_t expected_values = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (weights.size() != expected_values) {
    error_message = "Weight size does not match matrix dimensions.";
    return false;
  }

  __nv_bfloat16 * d_weights = nullptr;
  float * d_input = nullptr;
  float * d_output = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;

  const std::size_t weights_bytes = expected_values * sizeof(__nv_bfloat16);
  const std::size_t input_bytes = static_cast<std::size_t>(cols) * sizeof(float);
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * sizeof(float);

  if (!check_cuda(cudaMalloc(&d_weights, weights_bytes), "cudaMalloc(weights)", error_message) ||
      !check_cuda(cudaMalloc(&d_input, input_bytes), "cudaMalloc(input)", error_message) ||
      !check_cuda(cudaMalloc(&d_output, output_bytes), "cudaMalloc(output)", error_message)) {
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  if (!check_cuda(cudaMemcpy(d_weights, weights.data(), weights_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(weights)", error_message)) {
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  std::vector<float> input(static_cast<std::size_t>(cols), 1.0f / static_cast<float>(cols));
  if (!check_cuda(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(input)", error_message)) {
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  if (!check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)", error_message) ||
      !check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)", error_message)) {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  const int block_size = 128;
  const int grid_size = (rows + block_size - 1) / block_size;

  for (int i = 0; i < warmup_iterations; ++i) {
    bf16_matvec_kernel<<<grid_size, block_size>>>(d_weights, d_input, d_output, rows, cols);
  }
  if (!check_cuda(cudaGetLastError(), "bf16_matvec_kernel(warmup)", error_message) ||
      !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)", error_message)) {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  if (!check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)", error_message)) {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  for (int i = 0; i < benchmark_iterations; ++i) {
    bf16_matvec_kernel<<<grid_size, block_size>>>(d_weights, d_input, d_output, rows, cols);
  }

  if (!check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)", error_message) ||
      !check_cuda(cudaGetLastError(), "bf16_matvec_kernel(benchmark)", error_message) ||
      !check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)", error_message)) {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  float total_ms = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&total_ms, start_event, stop_event), "cudaEventElapsedTime", error_message)) {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  avg_iteration_ms = static_cast<double>(total_ms) / static_cast<double>(benchmark_iterations);

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  cudaFree(d_weights);
  cudaFree(d_input);
  cudaFree(d_output);
  return true;
}

} // namespace qwen35x::cuda

#endif

