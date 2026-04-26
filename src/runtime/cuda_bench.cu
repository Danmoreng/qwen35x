#include "qwen35x/runtime/cuda_bench.h"

#if QWEN35X_HAS_CUDA

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

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

__device__ __forceinline__ float decode_nvfp4_e2m1_device(const std::uint8_t nibble) {
  constexpr float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float value = levels[nibble & 0x7u];
  return (nibble & 0x8u) ? -value : value;
}

__device__ __forceinline__ float decode_e4m3_device(const std::uint8_t bits) {
  if (bits == 0) {
    return 0.0f;
  }
  const int sign = (bits >> 7) & 1;
  const int exponent = (bits >> 3) & 0xf;
  const int mantissa = bits & 0x7;
  float value = 0.0f;
  if (exponent == 0) {
    value = ldexpf(static_cast<float>(mantissa) / 8.0f, -6);
  } else {
    value = ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
  }
  return sign ? -value : value;
}

__global__ void nvfp4_matvec_check_kernel(
  const std::uint8_t * packed_weights,
  const std::uint8_t * weight_scales,
  const float * input,
  float input_scale,
  float weight_scale_2,
  float * output,
  int rows,
  int cols,
  int sample_rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= sample_rows || row >= rows) {
    return;
  }

  float sum = 0.0f;
  const int packed_cols = cols / 2;
  const int scale_cols = cols / 16;
  const std::uint8_t * row_packed = packed_weights + static_cast<std::size_t>(row) * packed_cols;
  const std::uint8_t * row_scales = weight_scales + static_cast<std::size_t>(row) * scale_cols;
  for (int c = 0; c < cols; ++c) {
    const std::uint8_t packed = row_packed[c / 2];
    const std::uint8_t nibble = (c & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
    const float scale = decode_e4m3_device(row_scales[c / 16]) * weight_scale_2;
    const float weight = decode_nvfp4_e2m1_device(nibble) * scale;
    sum += weight * (input[c] * input_scale);
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

float decode_nvfp4_e2m1_host(const std::uint8_t nibble) {
  constexpr float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float value = levels[nibble & 0x7u];
  return (nibble & 0x8u) ? -value : value;
}

float decode_e4m3_host(const std::uint8_t bits) {
  if (bits == 0) {
    return 0.0f;
  }
  const int sign = (bits >> 7) & 1;
  const int exponent = (bits >> 3) & 0xf;
  const int mantissa = bits & 0x7;
  float value = 0.0f;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
  } else {
    value = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
  }
  return sign ? -value : value;
}

bool run_nvfp4_matvec_check(
  const std::vector<std::uint8_t> & packed_weights,
  const std::vector<std::uint8_t> & weight_scales_e4m3,
  float input_scale,
  float weight_scale_2,
  int rows,
  int cols,
  int sample_rows,
  double & max_abs_error,
  std::string & error_message) {
  if (rows <= 0 || cols <= 0 || (cols % 16) != 0) {
    error_message = "Invalid matrix dimensions for NVFP4 check.";
    return false;
  }
  sample_rows = std::max(1, std::min(sample_rows, rows));
  const int packed_cols = cols / 2;
  const int scale_cols = cols / 16;
  if (packed_weights.size() != static_cast<std::size_t>(rows) * packed_cols ||
      weight_scales_e4m3.size() != static_cast<std::size_t>(rows) * scale_cols) {
    error_message = "NVFP4 packed weight or scale size does not match matrix dimensions.";
    return false;
  }

  std::vector<float> input(static_cast<std::size_t>(cols));
  for (int i = 0; i < cols; ++i) {
    input[static_cast<std::size_t>(i)] = (static_cast<float>((i % 17) - 8) / 17.0f);
  }
  std::vector<float> expected(static_cast<std::size_t>(sample_rows), 0.0f);
  for (int row = 0; row < sample_rows; ++row) {
    for (int c = 0; c < cols; ++c) {
      const std::uint8_t packed = packed_weights[static_cast<std::size_t>(row) * packed_cols + c / 2];
      const std::uint8_t nibble = (c & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
      const float scale = decode_e4m3_host(weight_scales_e4m3[static_cast<std::size_t>(row) * scale_cols + c / 16]) * weight_scale_2;
      expected[static_cast<std::size_t>(row)] += decode_nvfp4_e2m1_host(nibble) * scale * input[static_cast<std::size_t>(c)];
    }
  }

  std::uint8_t * d_weights = nullptr;
  std::uint8_t * d_scales = nullptr;
  float * d_input = nullptr;
  float * d_output = nullptr;
  const std::size_t weights_bytes = packed_weights.size();
  const std::size_t scales_bytes = weight_scales_e4m3.size();
  const std::size_t input_bytes = input.size() * sizeof(float);
  const std::size_t output_bytes = static_cast<std::size_t>(sample_rows) * sizeof(float);
  if (!check_cuda(cudaMalloc(&d_weights, weights_bytes), "cudaMalloc(nvfp4 weights)", error_message) ||
      !check_cuda(cudaMalloc(&d_scales, scales_bytes), "cudaMalloc(nvfp4 scales)", error_message) ||
      !check_cuda(cudaMalloc(&d_input, input_bytes), "cudaMalloc(nvfp4 input)", error_message) ||
      !check_cuda(cudaMalloc(&d_output, output_bytes), "cudaMalloc(nvfp4 output)", error_message)) {
    cudaFree(d_weights);
    cudaFree(d_scales);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }
  if (!check_cuda(cudaMemcpy(d_weights, packed_weights.data(), weights_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(nvfp4 weights)", error_message) ||
      !check_cuda(cudaMemcpy(d_scales, weight_scales_e4m3.data(), scales_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(nvfp4 scales)", error_message) ||
      !check_cuda(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(nvfp4 input)", error_message)) {
    cudaFree(d_weights);
    cudaFree(d_scales);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  const int block_size = 128;
  const int grid_size = (sample_rows + block_size - 1) / block_size;
  nvfp4_matvec_check_kernel<<<grid_size, block_size>>>(
    d_weights, d_scales, d_input, 1.0f, weight_scale_2, d_output, rows, cols, sample_rows);
  std::vector<float> actual(static_cast<std::size_t>(sample_rows), 0.0f);
  if (!check_cuda(cudaGetLastError(), "nvfp4_matvec_check_kernel", error_message) ||
      !check_cuda(cudaMemcpy(actual.data(), d_output, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(nvfp4 output)", error_message)) {
    cudaFree(d_weights);
    cudaFree(d_scales);
    cudaFree(d_input);
    cudaFree(d_output);
    return false;
  }

  max_abs_error = 0.0;
  for (int i = 0; i < sample_rows; ++i) {
    max_abs_error = std::max(max_abs_error, static_cast<double>(std::fabs(actual[static_cast<std::size_t>(i)] - expected[static_cast<std::size_t>(i)])));
  }

  cudaFree(d_weights);
  cudaFree(d_scales);
  cudaFree(d_input);
  cudaFree(d_output);
  return true;
}

} // namespace qwen35x::cuda

#endif
