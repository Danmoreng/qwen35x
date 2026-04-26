#include "qwen35x/runtime/cuda_bench.h"

#if QWEN35X_HAS_CUDA

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>

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

__device__ __forceinline__ std::uint8_t encode_ue4m3_scale_device(float value) {
  if (!(value > 0.0f)) {
    return 0;
  }
  int best_bits = 0;
  float best_error = 3.402823466e+38f;
  for (int exponent = 0; exponent < 16; ++exponent) {
    for (int mantissa = 0; mantissa < 8; ++mantissa) {
      float decoded = 0.0f;
      if (exponent == 0) {
        decoded = ldexpf(static_cast<float>(mantissa) / 8.0f, -6);
      } else {
        decoded = ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
      }
      const float error = fabsf(decoded - value);
      if (error < best_error) {
        best_error = error;
        best_bits = (exponent << 3) | mantissa;
      }
    }
  }
  return static_cast<std::uint8_t>(best_bits);
}

__device__ __forceinline__ std::uint8_t encode_nvfp4_e2m1_device(float value) {
  constexpr float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const bool negative = value < 0.0f;
  const float abs_value = fabsf(value);
  int best = 0;
  float best_error = 3.402823466e+38f;
  for (int i = 0; i < 8; ++i) {
    const float error = fabsf(levels[i] - abs_value);
    if (error < best_error) {
      best_error = error;
      best = i;
    }
  }
  return static_cast<std::uint8_t>((negative ? 0x8u : 0u) | static_cast<std::uint8_t>(best));
}

__global__ void nvfp4_quantize_single_row_for_cublaslt_kernel(
  const float * input,
  std::uint8_t * packed_input,
  std::uint8_t * tiled_scales,
  int cols) {
  const int group = blockIdx.x;
  const int lane = threadIdx.x;
  if (lane >= 16 || group * 16 + lane >= cols) {
    return;
  }

  float local_abs = fabsf(input[group * 16 + lane]);
  for (int offset = 8; offset > 0; offset >>= 1) {
    local_abs = fmaxf(local_abs, __shfl_down_sync(0xffffu, local_abs, offset));
  }
  const float max_abs = __shfl_sync(0xffffu, local_abs, 0);
  const float scale = fmaxf(max_abs / 6.0f, 1.0e-8f);

  if (lane == 0) {
    const int scale_col = group;
    const int col_block = scale_col / 4;
    const int col_in_group = scale_col % 4;
    tiled_scales[col_block * 512 + col_in_group] = encode_ue4m3_scale_device(scale);
  }

  if ((lane & 1) == 0) {
    const int col0 = group * 16 + lane;
    const int col1 = col0 + 1;
    const std::uint8_t nibble0 = encode_nvfp4_e2m1_device(input[col0] / scale);
    const std::uint8_t nibble1 = col1 < cols ? encode_nvfp4_e2m1_device(input[col1] / scale) : 0;
    packed_input[col0 / 2] = static_cast<std::uint8_t>((nibble0 << 4u) | nibble1);
  }
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

bool check_cublas(cublasStatus_t status, const char * step, std::string & error_message) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return true;
  }
  error_message = std::string(step) + " failed with cuBLAS status " + std::to_string(static_cast<int>(status)) + ".";
  return false;
}

std::uint8_t encode_ue4m3_scale(float value) {
  if (!(value > 0.0f)) {
    return 0;
  }
  int best_bits = 0;
  float best_error = std::numeric_limits<float>::infinity();
  for (int exponent = 0; exponent < 16; ++exponent) {
    for (int mantissa = 0; mantissa < 8; ++mantissa) {
      float decoded = 0.0f;
      if (exponent == 0) {
        decoded = std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
      } else {
        decoded = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
      }
      const float error = std::fabs(decoded - value);
      if (error < best_error) {
        best_error = error;
        best_bits = (exponent << 3) | mantissa;
      }
    }
  }
  return static_cast<std::uint8_t>(best_bits);
}

std::uint8_t encode_nvfp4_e2m1(float value) {
  constexpr float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const bool negative = value < 0.0f;
  const float abs_value = std::fabs(value);
  int best = 0;
  float best_error = std::numeric_limits<float>::infinity();
  for (int i = 0; i < 8; ++i) {
    const float error = std::fabs(levels[i] - abs_value);
    if (error < best_error) {
      best_error = error;
      best = i;
    }
  }
  return static_cast<std::uint8_t>((negative ? 0x8u : 0u) | static_cast<std::uint8_t>(best));
}

int round_up_to_multiple(const int value, const int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

std::vector<std::uint8_t> swizzle_nvfp4_block_scale(
  const std::vector<std::uint8_t> & scale,
  int rows,
  int cols) {
  const int padded_rows = round_up_to_multiple(rows, 128);
  const int padded_cols = round_up_to_multiple(cols, 4);
  std::vector<std::uint8_t> padded(static_cast<std::size_t>(padded_rows) * padded_cols, 0);
  for (int row = 0; row < rows; ++row) {
    std::copy_n(
      scale.data() + static_cast<std::size_t>(row) * cols,
      cols,
      padded.data() + static_cast<std::size_t>(row) * padded_cols);
  }

  std::vector<std::uint8_t> swizzled(padded.size(), 0);
  std::size_t out = 0;
  for (int row_block = 0; row_block < padded_rows / 128; ++row_block) {
    for (int col_block = 0; col_block < padded_cols / 4; ++col_block) {
      for (int row_in_group = 0; row_in_group < 32; ++row_in_group) {
        for (int row_group = 0; row_group < 4; ++row_group) {
          for (int col_in_group = 0; col_in_group < 4; ++col_in_group) {
            const int row = row_block * 128 + row_group * 32 + row_in_group;
            const int col = col_block * 4 + col_in_group;
            swizzled[out++] = padded[static_cast<std::size_t>(row) * padded_cols + col];
          }
        }
      }
    }
  }
  return swizzled;
}

std::vector<std::uint8_t> swap_fp4_nibbles(const std::uint8_t * data, const std::size_t bytes) {
  std::vector<std::uint8_t> swapped(bytes, 0);
  for (std::size_t i = 0; i < bytes; ++i) {
    swapped[i] = static_cast<std::uint8_t>((data[i] >> 4u) | (data[i] << 4u));
  }
  return swapped;
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

bool run_nvfp4_cublaslt_probe(
  const std::vector<std::uint8_t> & packed_weights,
  const std::vector<std::uint8_t> & weight_scales_e4m3,
  float weight_scale_2,
  int rows,
  int cols,
  int sample_rows,
  double & max_abs_error,
  double & elapsed_ms,
  double & max_abs_expected,
  double & max_abs_actual,
  std::string & error_message) {
  if (rows <= 0 || cols <= 0 || (cols % 16) != 0 || (cols % 2) != 0) {
    error_message = "Invalid matrix dimensions for NVFP4 cuBLASLt probe.";
    return false;
  }
  const int used_rows = std::min(rows, std::max(1, sample_rows));
  const int packed_cols = cols / 2;
  const int scale_cols = cols / 16;
  if (packed_weights.size() < static_cast<std::size_t>(used_rows) * packed_cols ||
      weight_scales_e4m3.size() < static_cast<std::size_t>(used_rows) * scale_cols) {
    error_message = "NVFP4 tensor buffers are smaller than the requested probe shape.";
    return false;
  }

  std::vector<float> input(static_cast<std::size_t>(cols));
  for (int col = 0; col < cols; ++col) {
    input[static_cast<std::size_t>(col)] = static_cast<float>((col % 17) - 8) / 17.0f;
  }
  std::vector<std::uint8_t> host_packed_input(static_cast<std::size_t>(cols / 2), 0);
  std::vector<std::uint8_t> input_scales_linear(static_cast<std::size_t>(scale_cols), 0);
  for (int group = 0; group < scale_cols; ++group) {
    float max_abs = 0.0f;
    for (int i = 0; i < 16; ++i) {
      max_abs = std::max(max_abs, std::fabs(input[static_cast<std::size_t>(group * 16 + i)]));
    }
    const float scale = std::max(max_abs / 6.0f, 1.0e-8f);
    input_scales_linear[static_cast<std::size_t>(group)] = encode_ue4m3_scale(scale);
    for (int i = 0; i < 16; ++i) {
      const int col = group * 16 + i;
      const std::uint8_t nibble = encode_nvfp4_e2m1(input[static_cast<std::size_t>(col)] / scale);
      auto & packed = host_packed_input[static_cast<std::size_t>(col / 2)];
      packed = (col & 1) == 0 ? static_cast<std::uint8_t>((packed & 0xf0u) | nibble)
                              : static_cast<std::uint8_t>((packed & 0x0fu) | (nibble << 4u));
    }
  }
  const auto input_scales = swizzle_nvfp4_block_scale(input_scales_linear, 1, scale_cols);
  const auto cublaslt_host_packed_input = swap_fp4_nibbles(host_packed_input.data(), host_packed_input.size());

  std::vector<float> expected(static_cast<std::size_t>(used_rows), 0.0f);
  std::vector<std::uint8_t> weight_scales_ue4m3_linear(static_cast<std::size_t>(used_rows) * scale_cols, 0);
  for (int row = 0; row < used_rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const auto packed = packed_weights[static_cast<std::size_t>(row * packed_cols + col / 2)];
      const auto nibble = (col & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
      const float scale = decode_e4m3_host(weight_scales_e4m3[static_cast<std::size_t>(row * scale_cols + col / 16)]) * weight_scale_2;
      expected[static_cast<std::size_t>(row)] += decode_nvfp4_e2m1_host(nibble) * scale * input[static_cast<std::size_t>(col)];
    }
    for (int group = 0; group < scale_cols; ++group) {
      const float scale = decode_e4m3_host(weight_scales_e4m3[static_cast<std::size_t>(row * scale_cols + group)]);
      weight_scales_ue4m3_linear[static_cast<std::size_t>(row * scale_cols + group)] = encode_ue4m3_scale(scale);
    }
  }
  const auto weight_scales_ue4m3 = swizzle_nvfp4_block_scale(weight_scales_ue4m3_linear, used_rows, scale_cols);
  const auto cublaslt_packed_weights = swap_fp4_nibbles(packed_weights.data(), static_cast<std::size_t>(used_rows) * packed_cols);

  float * d_input_f32 = nullptr;
  std::uint8_t * d_input = nullptr;
  std::uint8_t * d_input_scales = nullptr;
  std::uint8_t * d_weights = nullptr;
  std::uint8_t * d_weight_scales = nullptr;
  float * d_output = nullptr;
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  auto cleanup = [&]() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
    if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
    if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
    if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    if (handle) cublasLtDestroy(handle);
    cudaFree(d_input_f32);
    cudaFree(d_input);
    cudaFree(d_input_scales);
    cudaFree(d_weights);
    cudaFree(d_weight_scales);
    cudaFree(d_output);
  };

  const std::size_t input_f32_bytes = input.size() * sizeof(float);
  const std::size_t input_bytes = static_cast<std::size_t>(cols / 2);
  const std::size_t weight_bytes = static_cast<std::size_t>(used_rows) * packed_cols;
  const std::size_t output_bytes = static_cast<std::size_t>(used_rows) * sizeof(float);
  if (!check_cuda(cudaMalloc(&d_input_f32, input_f32_bytes), "cudaMalloc(fp4 input f32)", error_message) ||
      !check_cuda(cudaMalloc(&d_input, input_bytes), "cudaMalloc(fp4 input)", error_message) ||
      !check_cuda(cudaMalloc(&d_input_scales, input_scales.size()), "cudaMalloc(fp4 input scales)", error_message) ||
      !check_cuda(cudaMalloc(&d_weights, weight_bytes), "cudaMalloc(fp4 weights)", error_message) ||
      !check_cuda(cudaMalloc(&d_weight_scales, weight_scales_ue4m3.size()), "cudaMalloc(fp4 weight scales)", error_message) ||
      !check_cuda(cudaMalloc(&d_output, output_bytes), "cudaMalloc(fp4 output)", error_message)) {
    cleanup();
    return false;
  }
  if (!check_cuda(cudaMemcpy(d_input_f32, input.data(), input_f32_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(fp4 input f32)", error_message) ||
      !check_cuda(cudaMemset(d_input, 0, input_bytes), "cudaMemset(fp4 input)", error_message) ||
      !check_cuda(cudaMemset(d_input_scales, 0, input_scales.size()), "cudaMemset(fp4 input scales)", error_message) ||
      !check_cuda(cudaMemcpy(d_weights, cublaslt_packed_weights.data(), weight_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(fp4 weights)", error_message) ||
      !check_cuda(cudaMemcpy(d_weight_scales, weight_scales_ue4m3.data(), weight_scales_ue4m3.size(), cudaMemcpyHostToDevice), "cudaMemcpy(fp4 weight scales)", error_message) ||
      !check_cublas(cublasLtCreate(&handle), "cublasLtCreate", error_message)) {
    cleanup();
    return false;
  }
  nvfp4_quantize_single_row_for_cublaslt_kernel<<<scale_cols, 16>>>(d_input_f32, d_input, d_input_scales, cols);
  if (!check_cuda(cudaGetLastError(), "nvfp4_quantize_single_row_for_cublaslt_kernel", error_message)) {
    cleanup();
    return false;
  }
  std::vector<std::uint8_t> gpu_packed_input(input_bytes, 0);
  std::vector<std::uint8_t> gpu_input_scales(input_scales.size(), 0);
  if (!check_cuda(cudaMemcpy(gpu_packed_input.data(), d_input, input_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H fp4 quantized input)", error_message) ||
      !check_cuda(cudaMemcpy(gpu_input_scales.data(), d_input_scales, input_scales.size(), cudaMemcpyDeviceToHost), "cudaMemcpy(D2H fp4 input scales)", error_message)) {
    cleanup();
    return false;
  }
  if (gpu_packed_input != cublaslt_host_packed_input || gpu_input_scales != input_scales) {
    std::size_t packed_mismatch = gpu_packed_input.size();
    for (std::size_t i = 0; i < gpu_packed_input.size(); ++i) {
      if (gpu_packed_input[i] != cublaslt_host_packed_input[i]) {
        packed_mismatch = i;
        break;
      }
    }
    std::size_t scale_mismatch = gpu_input_scales.size();
    for (std::size_t i = 0; i < gpu_input_scales.size(); ++i) {
      if (gpu_input_scales[i] != input_scales[i]) {
        scale_mismatch = i;
        break;
      }
    }
    error_message =
      "GPU FP4 activation quantization does not match host reference: packed_mismatch=" +
      (packed_mismatch == gpu_packed_input.size() ? std::string("none") : std::to_string(packed_mismatch)) +
      " scale_mismatch=" +
      (scale_mismatch == gpu_input_scales.size() ? std::string("none") : std::to_string(scale_mismatch)) +
      (scale_mismatch == gpu_input_scales.size()
         ? std::string()
         : (" host_scale=" + std::to_string(static_cast<int>(input_scales[scale_mismatch])) +
            " gpu_scale=" + std::to_string(static_cast<int>(gpu_input_scales[scale_mismatch]))));
    cleanup();
    return false;
  }

  const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  const cudaDataType_t scale_type = CUDA_R_32F;
  const cublasOperation_t transa = CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_T;
  const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  const cublasLtMatmulMatrixScale_t vec16_scale = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  if (!check_cublas(cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type), "cublasLtMatmulDescCreate(fp4)", error_message) ||
      !check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)), "cublasLtMatmulDescSetAttribute(TRANSA fp4)", error_message) ||
      !check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)), "cublasLtMatmulDescSetAttribute(TRANSB fp4)", error_message) ||
      !check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16_scale, sizeof(vec16_scale)), "cublasLtMatmulDescSetAttribute(A_SCALE_MODE fp4)", error_message) ||
      !check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16_scale, sizeof(vec16_scale)), "cublasLtMatmulDescSetAttribute(B_SCALE_MODE fp4)", error_message) ||
      !check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_input_scales, sizeof(d_input_scales)), "cublasLtMatmulDescSetAttribute(A_SCALE_POINTER fp4)", error_message) ||
      !check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_weight_scales, sizeof(d_weight_scales)), "cublasLtMatmulDescSetAttribute(B_SCALE_POINTER fp4)", error_message) ||
      !check_cublas(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, 1, cols, cols), "cublasLtMatrixLayoutCreate(A fp4)", error_message) ||
      !check_cublas(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, used_rows, cols, cols), "cublasLtMatrixLayoutCreate(B fp4)", error_message) ||
      !check_cublas(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, 1, used_rows, used_rows), "cublasLtMatrixLayoutCreate(C fp4)", error_message) ||
      !check_cublas(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)), "cublasLtMatrixLayoutSetAttribute(A_ORDER fp4)", error_message) ||
      !check_cublas(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)), "cublasLtMatrixLayoutSetAttribute(B_ORDER fp4)", error_message) ||
      !check_cublas(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)), "cublasLtMatrixLayoutSetAttribute(C_ORDER fp4)", error_message) ||
      !check_cublas(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate(fp4)", error_message)) {
    cleanup();
    return false;
  }

  std::size_t workspace_bytes = 0;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  if (!check_cublas(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)), "cublasLtMatmulPreferenceSetAttribute(fp4 workspace)", error_message) ||
      !check_cublas(cublasLtMatmulAlgoGetHeuristic(handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heuristic, &returned_results), "cublasLtMatmulAlgoGetHeuristic(fp4)", error_message)) {
    cleanup();
    return false;
  }
  if (returned_results <= 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
    error_message = "cuBLASLt returned no valid FP4 block-scale matmul algorithm.";
    cleanup();
    return false;
  }

  const float alpha = weight_scale_2;
  const float beta = 0.0f;
  if (!check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(fp4 start)", error_message) ||
      !check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(fp4 stop)", error_message) ||
      !check_cuda(cudaEventRecord(start_event), "cudaEventRecord(fp4 start)", error_message) ||
      !check_cublas(cublasLtMatmul(handle, op_desc, &alpha, d_input, a_desc, d_weights, b_desc, &beta, d_output, c_desc, d_output, c_desc, &heuristic.algo, nullptr, 0, nullptr), "cublasLtMatmul(fp4)", error_message) ||
      !check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(fp4 stop)", error_message) ||
      !check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(fp4 stop)", error_message)) {
    cleanup();
    return false;
  }
  float elapsed = 0.0f;
  std::vector<float> actual(static_cast<std::size_t>(used_rows), 0.0f);
  if (!check_cuda(cudaEventElapsedTime(&elapsed, start_event, stop_event), "cudaEventElapsedTime(fp4)", error_message) ||
      !check_cuda(cudaMemcpy(actual.data(), d_output, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(fp4 output)", error_message)) {
    cleanup();
    return false;
  }
  max_abs_error = 0.0;
  max_abs_expected = 0.0;
  max_abs_actual = 0.0;
  for (int row = 0; row < used_rows; ++row) {
    max_abs_expected = std::max(max_abs_expected, static_cast<double>(std::fabs(expected[static_cast<std::size_t>(row)])));
    max_abs_actual = std::max(max_abs_actual, static_cast<double>(std::fabs(actual[static_cast<std::size_t>(row)])));
    max_abs_error = std::max(max_abs_error, static_cast<double>(std::fabs(actual[static_cast<std::size_t>(row)] - expected[static_cast<std::size_t>(row)])));
  }
  elapsed_ms = static_cast<double>(elapsed);
  cleanup();
  return true;
}

} // namespace qwen35x::cuda

#endif
