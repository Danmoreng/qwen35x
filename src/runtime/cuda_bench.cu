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

__global__ void silu_multiply_kernel(const float * gate, const float * up, float * output, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  const float gate_value = gate[idx];
  output[idx] = (gate_value / (1.0f + expf(-gate_value))) * up[idx];
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

__global__ void nvfp4_projection_row_parallel_kernel(
  const std::uint8_t * packed_weights,
  const std::uint8_t * weight_scales,
  const float * input,
  float input_scale,
  float weight_scale_2,
  float * output,
  int rows,
  int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int packed_cols = cols / 2;
  const int scale_cols = cols / 16;
  const std::uint8_t * row_packed = packed_weights + static_cast<std::size_t>(row) * packed_cols;
  const std::uint8_t * row_scales = weight_scales + static_cast<std::size_t>(row) * scale_cols;

  float partial = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const std::uint8_t packed = row_packed[c / 2];
    const std::uint8_t nibble = (c & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
    const float scale = decode_e4m3_device(row_scales[c / 16]) * weight_scale_2;
    const float weight = decode_nvfp4_e2m1_device(nibble) * scale;
    partial += weight * (input[c] * input_scale);
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffffu, partial, offset);
  }

  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = partial;
  }
  __syncthreads();

  if (warp == 0) {
    const int warp_count = blockDim.x / 32;
    float sum = lane < warp_count ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
      output[row] = sum;
    }
  }
}

__global__ void nvfp4_projection_warp_rows_kernel(
  const std::uint8_t * packed_weights,
  const std::uint8_t * weight_scales,
  const float * input,
  float input_scale,
  float weight_scale_2,
  float * output,
  int rows,
  int cols) {
  const int warp_id_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + warp_id_in_block;
  if (row >= rows) {
    return;
  }

  const int packed_cols = cols / 2;
  const int scale_cols = cols / 16;
  const std::uint8_t * row_packed = packed_weights + static_cast<std::size_t>(row) * packed_cols;
  const std::uint8_t * row_scales = weight_scales + static_cast<std::size_t>(row) * scale_cols;

  float partial = 0.0f;
  for (int c = lane; c < cols; c += 32) {
    const std::uint8_t packed = row_packed[c / 2];
    const std::uint8_t nibble = (c & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
    const float scale = decode_e4m3_device(row_scales[c / 16]) * weight_scale_2;
    const float weight = decode_nvfp4_e2m1_device(nibble) * scale;
    partial += weight * (input[c] * input_scale);
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffffu, partial, offset);
  }
  if (lane == 0) {
    output[row] = partial;
  }
}

__global__ void nvfp4_projection_scale_group_kernel(
  const std::uint8_t * packed_weights,
  const std::uint8_t * weight_scales,
  const float * input,
  float input_scale,
  float weight_scale_2,
  float * output,
  int rows,
  int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int packed_cols = cols / 2;
  const int scale_cols = cols / 16;
  const std::uint8_t * row_packed = packed_weights + static_cast<std::size_t>(row) * packed_cols;
  const std::uint8_t * row_scales = weight_scales + static_cast<std::size_t>(row) * scale_cols;

  float partial = 0.0f;
  for (int group = threadIdx.x; group < scale_cols; group += blockDim.x) {
    const float scale = decode_e4m3_device(row_scales[group]) * weight_scale_2;
    const int col_base = group * 16;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int c = col_base + i;
      const std::uint8_t packed = row_packed[c / 2];
      const std::uint8_t nibble = (c & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
      const float weight = decode_nvfp4_e2m1_device(nibble) * scale;
      partial += weight * (input[c] * input_scale);
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffffu, partial, offset);
  }

  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = partial;
  }
  __syncthreads();

  if (warp == 0) {
    const int warp_count = blockDim.x / 32;
    float sum = lane < warp_count ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
      output[row] = sum;
    }
  }
}

__global__ void nvfp4_mma_sync_mxf4nvf4_tile_probe_kernel(
  const std::uint32_t * a_fragments,
  const std::uint32_t * b_fragments,
  const std::uint32_t * a_scales,
  const std::uint32_t * b_scales,
  float * output) {
#if defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200)
  const int lane = threadIdx.x & 31;
  const std::uint32_t a0 = a_fragments[lane * 4 + 0];
  const std::uint32_t a1 = a_fragments[lane * 4 + 1];
  const std::uint32_t a2 = a_fragments[lane * 4 + 2];
  const std::uint32_t a3 = a_fragments[lane * 4 + 3];
  const std::uint32_t b0 = b_fragments[lane * 2 + 0];
  const std::uint32_t b1 = b_fragments[lane * 2 + 1];
  float c0 = 0.0f;
  float c1 = 1.0f;
  float c2 = 2.0f;
  float c3 = 3.0f;
  float d0 = 0.0f;
  float d1 = 0.0f;
  float d2 = 0.0f;
  float d3 = 0.0f;
  const std::uint32_t scale_a = a_scales[lane];
  const std::uint32_t scale_b = b_scales[lane];
  std::uint16_t bid = 0;
  std::uint16_t tid = 0;
  asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%10, %11, %12, %13}, "
    "%14, {%16, %17}, "
    "%15, {%16, %17};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(scale_a), "r"(scale_b), "h"(bid), "h"(tid));
  if (output != nullptr) {
    output[lane * 4 + 0] = d0;
    output[lane * 4 + 1] = d1;
    output[lane * 4 + 2] = d2;
    output[lane * 4 + 3] = d3;
  }
#else
  if (threadIdx.x == 0 && output != nullptr) {
    output[0] = -1.0f;
  }
#endif
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

bool query_device_compute_capability(int & major, int & minor, std::string & error_message) {
  int device = 0;
  cudaDeviceProp props{};
  if (!check_cuda(cudaGetDevice(&device), "cudaGetDevice", error_message) ||
      !check_cuda(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties", error_message)) {
    return false;
  }
  major = props.major;
  minor = props.minor;
  return true;
}

bool device_supports_sm120_mma_mxf4nvf4(const int major, const int minor) {
  return (major == 12 && (minor == 0 || minor == 1));
}

bool run_nvfp4_blackwell_fp4_projection_benchmark(
  const std::vector<std::uint8_t> & packed_weights,
  const std::vector<std::uint8_t> & weight_scales_e4m3,
  float input_scale,
  float weight_scale_2,
  int rows,
  int cols,
  int warmup_iterations,
  int benchmark_iterations,
  double & avg_iteration_ms,
  double & max_abs_error,
  std::string & error_message) {
  int major = 0;
  int minor = 0;
  if (!query_device_compute_capability(major, minor, error_message)) {
    return false;
  }
  if (!device_supports_sm120_mma_mxf4nvf4(major, minor)) {
    error_message =
      "blackwell-fp4 currently targets the SM120 mma.sync.aligned kind::mxf4nvf4.block_scale path; current device is sm_" +
      std::to_string(major) + std::to_string(minor) +
      ". Keep using --nvfp4-projection-kernel scale-group on unsupported devices.";
    return false;
  }

  if (rows < 8 || cols < 64 || (cols % 16) != 0 || (cols % 2) != 0) {
    error_message = "blackwell-fp4 tile probe requires at least 8 rows and 64 columns.";
    return false;
  }
  const int packed_cols = cols / 2;
  const int scale_cols = cols / 16;
  if (packed_weights.size() < static_cast<std::size_t>(rows) * packed_cols ||
      weight_scales_e4m3.size() < static_cast<std::size_t>(rows) * scale_cols) {
    error_message = "NVFP4 packed weight or scale size does not match matrix dimensions.";
    return false;
  }

  std::vector<std::uint32_t> host_a_fragments(32 * 4, 0);
  std::vector<std::uint32_t> host_b_fragments(32 * 2, 0);
  std::vector<std::uint32_t> host_a_scales(32, 0);
  std::vector<std::uint32_t> host_b_scales(32, 0);
  auto set_packed_nibble = [](std::uint32_t & word, const int nibble_index, const std::uint32_t nibble) {
    const int shift = nibble_index * 4;
    word = static_cast<std::uint32_t>((word & ~(0xfu << shift)) | ((nibble & 0xfu) << shift));
  };
  auto encode_scale = [](const float value) -> std::uint8_t {
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
  };
  auto encode_e2m1 = [](const float value) -> std::uint8_t {
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
  };
  auto decode_e2m1 = [](const std::uint8_t nibble) -> float {
    constexpr float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = levels[nibble & 0x7u];
    return (nibble & 0x8u) ? -value : value;
  };
  auto decode_e4m3 = [](const std::uint8_t bits) -> float {
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
  };

  std::vector<float> input(static_cast<std::size_t>(64), 0.0f);
  std::vector<std::uint8_t> input_nibbles(64, 0);
  std::vector<std::uint8_t> input_scales(4, 0);
  for (int col = 0; col < 64; ++col) {
    input[static_cast<std::size_t>(col)] = static_cast<float>((col % 17) - 8) / 17.0f;
  }
  for (int group = 0; group < 4; ++group) {
    float max_abs = 0.0f;
    for (int i = 0; i < 16; ++i) {
      max_abs = std::max(max_abs, std::fabs(input[static_cast<std::size_t>(group * 16 + i)] * input_scale));
    }
    const float scale = std::max(max_abs / 6.0f, 1.0e-8f);
    input_scales[static_cast<std::size_t>(group)] = encode_scale(scale);
    for (int i = 0; i < 16; ++i) {
      const int col = group * 16 + i;
      input_nibbles[static_cast<std::size_t>(col)] =
        encode_e2m1((input[static_cast<std::size_t>(col)] * input_scale) / scale);
    }
  }
  const std::uint32_t packed_input_scales =
    static_cast<std::uint32_t>(input_scales[0]) |
    (static_cast<std::uint32_t>(input_scales[1]) << 8u) |
    (static_cast<std::uint32_t>(input_scales[2]) << 16u) |
    (static_cast<std::uint32_t>(input_scales[3]) << 24u);
  std::fill(host_a_scales.begin(), host_a_scales.end(), packed_input_scales);

  for (int lane = 0; lane < 32; ++lane) {
    const int group_id = lane >> 2;
    const int thread_id_in_group = lane & 3;
    for (int i = 0; i < 32; ++i) {
      const int col = thread_id_in_group * 8 + (i & 7) + (i >= 16 ? 32 : 0);
      set_packed_nibble(host_a_fragments[static_cast<std::size_t>(lane) * 4 + i / 8], i & 7, input_nibbles[static_cast<std::size_t>(col)]);
    }
    for (int i = 0; i < 16; ++i) {
      const int k = thread_id_in_group * 8 + (i & 7) + (i >= 8 ? 32 : 0);
      const int col = group_id;
      const std::uint8_t packed = packed_weights[static_cast<std::size_t>(col) * packed_cols + k / 2];
      const std::uint8_t nibble = (k & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
      set_packed_nibble(host_b_fragments[static_cast<std::size_t>(lane) * 2 + i / 8], i & 7, nibble);
    }

    std::uint32_t packed_b_scales = 0;
    for (int group = 0; group < 4; ++group) {
      const auto scale_bits = static_cast<std::uint32_t>(weight_scales_e4m3[static_cast<std::size_t>(group_id) * scale_cols + group]);
      packed_b_scales |= (scale_bits << (group * 8u));
    }
    host_b_scales[static_cast<std::size_t>(lane)] = packed_b_scales;
  }

  std::uint32_t * device_a_fragments = nullptr;
  std::uint32_t * device_b_fragments = nullptr;
  std::uint32_t * device_a_scales = nullptr;
  std::uint32_t * device_b_scales = nullptr;
  float * device_output = nullptr;
  if (!check_cuda(cudaMalloc(&device_a_fragments, host_a_fragments.size() * sizeof(std::uint32_t)), "cudaMalloc SM120 FP4 A fragments", error_message) ||
      !check_cuda(cudaMalloc(&device_b_fragments, host_b_fragments.size() * sizeof(std::uint32_t)), "cudaMalloc SM120 FP4 B fragments", error_message) ||
      !check_cuda(cudaMalloc(&device_a_scales, host_a_scales.size() * sizeof(std::uint32_t)), "cudaMalloc SM120 FP4 A scales", error_message) ||
      !check_cuda(cudaMalloc(&device_b_scales, host_b_scales.size() * sizeof(std::uint32_t)), "cudaMalloc SM120 FP4 B scales", error_message) ||
      !check_cuda(cudaMalloc(&device_output, 32 * 4 * sizeof(float)), "cudaMalloc synthetic SM120 FP4 probe output", error_message)) {
    cudaFree(device_a_fragments);
    cudaFree(device_b_fragments);
    cudaFree(device_a_scales);
    cudaFree(device_b_scales);
    cudaFree(device_output);
    return false;
  }
  if (!check_cuda(cudaMemcpy(device_a_fragments, host_a_fragments.data(), host_a_fragments.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "cudaMemcpy SM120 FP4 A fragments", error_message) ||
      !check_cuda(cudaMemcpy(device_b_fragments, host_b_fragments.data(), host_b_fragments.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "cudaMemcpy SM120 FP4 B fragments", error_message) ||
      !check_cuda(cudaMemcpy(device_a_scales, host_a_scales.data(), host_a_scales.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "cudaMemcpy SM120 FP4 A scales", error_message) ||
      !check_cuda(cudaMemcpy(device_b_scales, host_b_scales.data(), host_b_scales.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "cudaMemcpy SM120 FP4 B scales", error_message)) {
    cudaFree(device_a_fragments);
    cudaFree(device_b_fragments);
    cudaFree(device_a_scales);
    cudaFree(device_b_scales);
    cudaFree(device_output);
    return false;
  }

  auto cleanup = [&]() {
    cudaFree(device_a_fragments);
    cudaFree(device_b_fragments);
    cudaFree(device_a_scales);
    cudaFree(device_b_scales);
    if (device_output != nullptr) {
      cudaFree(device_output);
    }
  };

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!check_cuda(cudaEventCreate(&start), "cudaEventCreate start", error_message) ||
      !check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop", error_message)) {
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    cleanup();
    return false;
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    nvfp4_mma_sync_mxf4nvf4_tile_probe_kernel<<<1, 32>>>(
      device_a_fragments, device_b_fragments, device_a_scales, device_b_scales, device_output);
  }
  if (!check_cuda(cudaGetLastError(), "launch synthetic SM120 FP4 warmup probe", error_message) ||
      !check_cuda(cudaDeviceSynchronize(), "synchronize synthetic SM120 FP4 warmup probe", error_message)) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cleanup();
    return false;
  }

  if (!check_cuda(cudaEventRecord(start), "cudaEventRecord start", error_message)) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cleanup();
    return false;
  }
  for (int i = 0; i < benchmark_iterations; ++i) {
    nvfp4_mma_sync_mxf4nvf4_tile_probe_kernel<<<1, 32>>>(
      device_a_fragments, device_b_fragments, device_a_scales, device_b_scales, device_output);
  }
  if (!check_cuda(cudaGetLastError(), "launch synthetic SM120 FP4 benchmark probe", error_message) ||
      !check_cuda(cudaEventRecord(stop), "cudaEventRecord stop", error_message) ||
      !check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop", error_message)) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cleanup();
    return false;
  }

  float elapsed_ms = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime", error_message)) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cleanup();
    return false;
  }

  std::vector<float> host_output(32 * 4, 0.0f);
  if (!check_cuda(cudaMemcpy(host_output.data(), device_output, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy synthetic SM120 FP4 probe output", error_message)) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cleanup();
    return false;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cleanup();

  if (host_output[0] < 0.0f) {
    error_message =
      "blackwell-fp4 was built without the SM120a architecture-specific MMA path. Rebuild with -CudaArchitectures 120a or 120f.";
    return false;
  }

  avg_iteration_ms = static_cast<double>(elapsed_ms) / static_cast<double>(benchmark_iterations);
  max_abs_error = 0.0;
  for (int lane = 0; lane < 32; ++lane) {
    const int group_id = lane >> 2;
    const int thread_id_in_group = lane & 3;
    for (int i = 0; i < 4; ++i) {
      const int row = i < 2 ? group_id : group_id + 8;
      const int col = thread_id_in_group * 2 + (i & 1);
      double expected = static_cast<double>(i);
      (void)row;
      for (int k = 0; k < 64; ++k) {
        const std::uint8_t packed = packed_weights[static_cast<std::size_t>(col) * packed_cols + k / 2];
        const std::uint8_t nibble = (k & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
        const float scale = decode_e4m3(weight_scales_e4m3[static_cast<std::size_t>(col) * scale_cols + k / 16]) * weight_scale_2;
        expected += static_cast<double>(decode_e2m1(nibble) * scale * (input[static_cast<std::size_t>(k)] * input_scale));
      }
      const double actual = static_cast<double>(host_output[static_cast<std::size_t>(lane) * 4 + i]);
      max_abs_error = std::max(max_abs_error, std::fabs(actual - expected));
    }
  }
  if (max_abs_error > 0.25) {
    error_message =
      "synthetic SM120 mxf4nvf4 MMA tile probe produced unexpected accumulator value " + std::to_string(host_output[0]) +
      " for the real model-tile check.";
    return false;
  }
  return true;
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

bool run_cublaslt_fp4_projection(
  const std::uint8_t * d_input,
  const std::uint8_t * d_input_scales,
  const std::uint8_t * d_weights,
  const std::uint8_t * d_weight_scales,
  float weight_scale_2,
  int rows,
  int cols,
  float * d_output,
  double * elapsed_ms,
  std::string & error_message) {
  struct Plan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};
    int rows = 0;
    int cols = 0;
    bool initialized = false;
  };
  static thread_local Plan plan;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  auto cleanup_events = [&]() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  };

  if (!plan.initialized || plan.rows != rows || plan.cols != cols) {
    if (plan.pref) cublasLtMatmulPreferenceDestroy(plan.pref);
    if (plan.c_desc) cublasLtMatrixLayoutDestroy(plan.c_desc);
    if (plan.b_desc) cublasLtMatrixLayoutDestroy(plan.b_desc);
    if (plan.a_desc) cublasLtMatrixLayoutDestroy(plan.a_desc);
    if (plan.op_desc) cublasLtMatmulDescDestroy(plan.op_desc);
    if (plan.handle) cublasLtDestroy(plan.handle);
    plan = Plan{};

    const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    const cudaDataType_t scale_type = CUDA_R_32F;
    const cublasOperation_t transa = CUBLAS_OP_N;
    const cublasOperation_t transb = CUBLAS_OP_T;
    const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    const cublasLtMatmulMatrixScale_t vec16_scale = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    if (!check_cublas(cublasLtCreate(&plan.handle), "cublasLtCreate", error_message) ||
        !check_cublas(cublasLtMatmulDescCreate(&plan.op_desc, compute_type, scale_type), "cublasLtMatmulDescCreate(fp4)", error_message) ||
        !check_cublas(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)), "cublasLtMatmulDescSetAttribute(TRANSA fp4)", error_message) ||
        !check_cublas(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)), "cublasLtMatmulDescSetAttribute(TRANSB fp4)", error_message) ||
        !check_cublas(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16_scale, sizeof(vec16_scale)), "cublasLtMatmulDescSetAttribute(A_SCALE_MODE fp4)", error_message) ||
        !check_cublas(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16_scale, sizeof(vec16_scale)), "cublasLtMatmulDescSetAttribute(B_SCALE_MODE fp4)", error_message) ||
        !check_cublas(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_input_scales, sizeof(d_input_scales)), "cublasLtMatmulDescSetAttribute(A_SCALE_POINTER fp4)", error_message) ||
        !check_cublas(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_weight_scales, sizeof(d_weight_scales)), "cublasLtMatmulDescSetAttribute(B_SCALE_POINTER fp4)", error_message) ||
        !check_cublas(cublasLtMatrixLayoutCreate(&plan.a_desc, CUDA_R_4F_E2M1, 1, cols, cols), "cublasLtMatrixLayoutCreate(A fp4)", error_message) ||
        !check_cublas(cublasLtMatrixLayoutCreate(&plan.b_desc, CUDA_R_4F_E2M1, rows, cols, cols), "cublasLtMatrixLayoutCreate(B fp4)", error_message) ||
        !check_cublas(cublasLtMatrixLayoutCreate(&plan.c_desc, CUDA_R_32F, 1, rows, rows), "cublasLtMatrixLayoutCreate(C fp4)", error_message) ||
        !check_cublas(cublasLtMatrixLayoutSetAttribute(plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)), "cublasLtMatrixLayoutSetAttribute(A_ORDER fp4)", error_message) ||
        !check_cublas(cublasLtMatrixLayoutSetAttribute(plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)), "cublasLtMatrixLayoutSetAttribute(B_ORDER fp4)", error_message) ||
        !check_cublas(cublasLtMatrixLayoutSetAttribute(plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)), "cublasLtMatrixLayoutSetAttribute(C_ORDER fp4)", error_message) ||
        !check_cublas(cublasLtMatmulPreferenceCreate(&plan.pref), "cublasLtMatmulPreferenceCreate(fp4)", error_message)) {
      return false;
    }

    std::size_t workspace_bytes = 0;
    int returned_results = 0;
    if (!check_cublas(cublasLtMatmulPreferenceSetAttribute(plan.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)), "cublasLtMatmulPreferenceSetAttribute(fp4 workspace)", error_message) ||
        !check_cublas(cublasLtMatmulAlgoGetHeuristic(plan.handle, plan.op_desc, plan.a_desc, plan.b_desc, plan.c_desc, plan.c_desc, plan.pref, 1, &plan.heuristic, &returned_results), "cublasLtMatmulAlgoGetHeuristic(fp4)", error_message)) {
      return false;
    }
    if (returned_results <= 0 || plan.heuristic.state != CUBLAS_STATUS_SUCCESS) {
      error_message = "cuBLASLt returned no valid FP4 block-scale matmul algorithm.";
      return false;
    }
    plan.rows = rows;
    plan.cols = cols;
    plan.initialized = true;
  }

  if (!check_cublas(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_input_scales, sizeof(d_input_scales)), "cublasLtMatmulDescSetAttribute(A_SCALE_POINTER fp4)", error_message) ||
      !check_cublas(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_weight_scales, sizeof(d_weight_scales)), "cublasLtMatmulDescSetAttribute(B_SCALE_POINTER fp4)", error_message)) {
    return false;
  }
  const float alpha = weight_scale_2;
  const float beta = 0.0f;
  if (elapsed_ms != nullptr &&
      (!check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(fp4 start)", error_message) ||
       !check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(fp4 stop)", error_message) ||
       !check_cuda(cudaEventRecord(start_event), "cudaEventRecord(fp4 start)", error_message))) {
    cleanup_events();
    return false;
  }
  if (!check_cublas(cublasLtMatmul(plan.handle, plan.op_desc, &alpha, d_input, plan.a_desc, d_weights, plan.b_desc, &beta, d_output, plan.c_desc, d_output, plan.c_desc, &plan.heuristic.algo, nullptr, 0, nullptr), "cublasLtMatmul(fp4)", error_message)) {
    cleanup_events();
    return false;
  }
  if (elapsed_ms != nullptr) {
    float elapsed = 0.0f;
    if (!check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(fp4 stop)", error_message) ||
        !check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(fp4 stop)", error_message) ||
        !check_cuda(cudaEventElapsedTime(&elapsed, start_event, stop_event), "cudaEventElapsedTime(fp4)", error_message)) {
      cleanup_events();
      return false;
    }
    *elapsed_ms = static_cast<double>(elapsed);
  }
  cleanup_events();
  return true;
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

bool run_nvfp4_custom_projection_benchmark(
  const std::vector<std::uint8_t> & packed_weights,
  const std::vector<std::uint8_t> & weight_scales_e4m3,
  float input_scale,
  float weight_scale_2,
  int rows,
  int cols,
  int kernel_variant,
  int warmup_iterations,
  int benchmark_iterations,
  double & avg_iteration_ms,
  double & max_abs_error,
  std::string & error_message) {
  if (rows <= 0 || cols <= 0 || (cols % 16) != 0 || (cols % 2) != 0) {
    error_message = "Invalid matrix dimensions for custom NVFP4 projection benchmark.";
    return false;
  }
  if (warmup_iterations < 0 || benchmark_iterations <= 0) {
    error_message = "warmup_iterations must be >= 0 and benchmark_iterations must be > 0.";
    return false;
  }
  if (kernel_variant == 3) {
    return run_nvfp4_blackwell_fp4_projection_benchmark(
      packed_weights,
      weight_scales_e4m3,
      input_scale,
      weight_scale_2,
      rows,
      cols,
      warmup_iterations,
      benchmark_iterations,
      avg_iteration_ms,
      max_abs_error,
      error_message);
  }
  const int packed_cols = cols / 2;
  const int scale_cols = cols / 16;
  if (packed_weights.size() != static_cast<std::size_t>(rows) * packed_cols ||
      weight_scales_e4m3.size() != static_cast<std::size_t>(rows) * scale_cols) {
    error_message = "NVFP4 packed weight or scale size does not match matrix dimensions.";
    return false;
  }

  std::vector<float> input(static_cast<std::size_t>(cols));
  for (int i = 0; i < cols; ++i) {
    input[static_cast<std::size_t>(i)] = static_cast<float>((i % 17) - 8) / 17.0f;
  }

  const int check_rows = std::min(rows, 64);
  std::vector<float> expected(static_cast<std::size_t>(check_rows), 0.0f);
  for (int row = 0; row < check_rows; ++row) {
    for (int c = 0; c < cols; ++c) {
      const std::uint8_t packed = packed_weights[static_cast<std::size_t>(row) * packed_cols + c / 2];
      const std::uint8_t nibble = (c & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
      const float scale = decode_e4m3_host(weight_scales_e4m3[static_cast<std::size_t>(row) * scale_cols + c / 16]) * weight_scale_2;
      expected[static_cast<std::size_t>(row)] +=
        decode_nvfp4_e2m1_host(nibble) * scale * (input[static_cast<std::size_t>(c)] * input_scale);
    }
  }

  std::uint8_t * d_weights = nullptr;
  std::uint8_t * d_scales = nullptr;
  float * d_input = nullptr;
  float * d_output = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  auto cleanup = [&]() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_weights);
    cudaFree(d_scales);
    cudaFree(d_input);
    cudaFree(d_output);
  };

  const std::size_t weights_bytes = packed_weights.size();
  const std::size_t scales_bytes = weight_scales_e4m3.size();
  const std::size_t input_bytes = input.size() * sizeof(float);
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * sizeof(float);
  if (!check_cuda(cudaMalloc(&d_weights, weights_bytes), "cudaMalloc(custom nvfp4 weights)", error_message) ||
      !check_cuda(cudaMalloc(&d_scales, scales_bytes), "cudaMalloc(custom nvfp4 scales)", error_message) ||
      !check_cuda(cudaMalloc(&d_input, input_bytes), "cudaMalloc(custom nvfp4 input)", error_message) ||
      !check_cuda(cudaMalloc(&d_output, output_bytes), "cudaMalloc(custom nvfp4 output)", error_message)) {
    cleanup();
    return false;
  }
  if (!check_cuda(cudaMemcpy(d_weights, packed_weights.data(), weights_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(custom nvfp4 weights)", error_message) ||
      !check_cuda(cudaMemcpy(d_scales, weight_scales_e4m3.data(), scales_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(custom nvfp4 scales)", error_message) ||
      !check_cuda(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(custom nvfp4 input)", error_message)) {
    cleanup();
    return false;
  }

  constexpr int block_size = 256;
  auto launch_kernel = [&]() {
    if (kernel_variant == 1) {
      constexpr int warps_per_block = block_size / 32;
      const int grid_size = (rows + warps_per_block - 1) / warps_per_block;
      nvfp4_projection_warp_rows_kernel<<<grid_size, block_size>>>(
        d_weights, d_scales, d_input, input_scale, weight_scale_2, d_output, rows, cols);
    } else if (kernel_variant == 2) {
      nvfp4_projection_scale_group_kernel<<<rows, block_size>>>(
        d_weights, d_scales, d_input, input_scale, weight_scale_2, d_output, rows, cols);
    } else {
      nvfp4_projection_row_parallel_kernel<<<rows, block_size>>>(
        d_weights, d_scales, d_input, input_scale, weight_scale_2, d_output, rows, cols);
    }
  };
  for (int i = 0; i < warmup_iterations; ++i) {
    launch_kernel();
  }
  if (!check_cuda(cudaGetLastError(), "custom nvfp4 projection kernel(warmup)", error_message) ||
      !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(custom nvfp4 warmup)", error_message) ||
      !check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(custom nvfp4 start)", error_message) ||
      !check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(custom nvfp4 stop)", error_message) ||
      !check_cuda(cudaEventRecord(start_event), "cudaEventRecord(custom nvfp4 start)", error_message)) {
    cleanup();
    return false;
  }
  for (int i = 0; i < benchmark_iterations; ++i) {
    launch_kernel();
  }
  float total_ms = 0.0f;
  if (!check_cuda(cudaGetLastError(), "custom nvfp4 projection kernel", error_message) ||
      !check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(custom nvfp4 stop)", error_message) ||
      !check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(custom nvfp4 stop)", error_message) ||
      !check_cuda(cudaEventElapsedTime(&total_ms, start_event, stop_event), "cudaEventElapsedTime(custom nvfp4)", error_message)) {
    cleanup();
    return false;
  }
  avg_iteration_ms = static_cast<double>(total_ms) / static_cast<double>(benchmark_iterations);

  std::vector<float> actual(static_cast<std::size_t>(check_rows), 0.0f);
  if (!check_cuda(cudaMemcpy(actual.data(), d_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(custom nvfp4 output)", error_message)) {
    cleanup();
    return false;
  }
  max_abs_error = 0.0;
  for (int row = 0; row < check_rows; ++row) {
    max_abs_error = std::max(
      max_abs_error,
      static_cast<double>(std::fabs(actual[static_cast<std::size_t>(row)] - expected[static_cast<std::size_t>(row)])));
  }

  cleanup();
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
  auto cleanup = [&]() {
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
      !check_cuda(cudaMemcpy(d_weight_scales, weight_scales_ue4m3.data(), weight_scales_ue4m3.size(), cudaMemcpyHostToDevice), "cudaMemcpy(fp4 weight scales)", error_message)) {
    cleanup();
    return false;
  }
  if (!run_nvfp4_cublaslt_projection_device(
        d_input_f32,
        d_weights,
        d_weight_scales,
        weight_scale_2,
        used_rows,
        cols,
        d_input,
        d_input_scales,
        d_output,
        &elapsed_ms,
        error_message)) {
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

  std::vector<float> actual(static_cast<std::size_t>(used_rows), 0.0f);
  if (!check_cuda(cudaMemcpy(actual.data(), d_output, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(fp4 output)", error_message)) {
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
  cleanup();
  return true;
}

bool run_nvfp4_cublaslt_projection_device(
  const float * input_f32,
  const std::uint8_t * packed_weights_cublaslt,
  const std::uint8_t * weight_scales_tiled,
  float weight_scale_2,
  int rows,
  int cols,
  std::uint8_t * activation_scratch,
  std::uint8_t * activation_scale_scratch,
  float * output_f32,
  double * elapsed_ms,
  std::string & error_message) {
  if (input_f32 == nullptr || packed_weights_cublaslt == nullptr || weight_scales_tiled == nullptr ||
      activation_scratch == nullptr || activation_scale_scratch == nullptr || output_f32 == nullptr) {
    error_message = "NVFP4 cuBLASLt projection received a null device buffer.";
    return false;
  }
  if (rows <= 0 || cols <= 0 || (cols % 16) != 0 || (cols % 2) != 0) {
    error_message = "Invalid matrix dimensions for NVFP4 cuBLASLt projection.";
    return false;
  }

  const int scale_cols = cols / 16;
  const std::size_t activation_bytes = static_cast<std::size_t>(cols / 2);
  const std::size_t activation_scale_bytes =
    static_cast<std::size_t>(round_up_to_multiple(1, 128)) * round_up_to_multiple(scale_cols, 4);
  if (!check_cuda(cudaMemset(activation_scratch, 0, activation_bytes), "cudaMemset(fp4 projection activation)", error_message) ||
      !check_cuda(cudaMemset(activation_scale_scratch, 0, activation_scale_bytes), "cudaMemset(fp4 projection activation scales)", error_message)) {
    return false;
  }

  nvfp4_quantize_single_row_for_cublaslt_kernel<<<scale_cols, 16>>>(input_f32, activation_scratch, activation_scale_scratch, cols);
  if (!check_cuda(cudaGetLastError(), "nvfp4_quantize_single_row_for_cublaslt_kernel", error_message)) {
    return false;
  }

  double local_elapsed_ms = 0.0;
  if (!run_cublaslt_fp4_projection(
        activation_scratch,
        activation_scale_scratch,
        packed_weights_cublaslt,
        weight_scales_tiled,
        weight_scale_2,
        rows,
        cols,
        output_f32,
        elapsed_ms == nullptr ? nullptr : &local_elapsed_ms,
        error_message)) {
    return false;
  }
  if (elapsed_ms != nullptr) {
    *elapsed_ms = local_elapsed_ms;
  }
  return true;
}

bool run_nvfp4_cublaslt_gate_up_silu_device(
  const float * input_f32,
  const std::uint8_t * gate_packed_weights_cublaslt,
  const std::uint8_t * gate_weight_scales_tiled,
  float gate_weight_scale_2,
  const std::uint8_t * up_packed_weights_cublaslt,
  const std::uint8_t * up_weight_scales_tiled,
  float up_weight_scale_2,
  int rows,
  int cols,
  std::uint8_t * activation_scratch,
  std::uint8_t * activation_scale_scratch,
  float * gate_output_f32,
  float * up_output_f32,
  double * elapsed_ms,
  std::string & error_message) {
  if (gate_output_f32 == nullptr || up_output_f32 == nullptr) {
    error_message = "NVFP4 gate/up projection received a null output buffer.";
    return false;
  }
  if (input_f32 == nullptr || gate_packed_weights_cublaslt == nullptr || gate_weight_scales_tiled == nullptr ||
      up_packed_weights_cublaslt == nullptr || up_weight_scales_tiled == nullptr ||
      activation_scratch == nullptr || activation_scale_scratch == nullptr) {
    error_message = "NVFP4 gate/up projection received a null device buffer.";
    return false;
  }
  if (rows <= 0 || cols <= 0 || (cols % 16) != 0 || (cols % 2) != 0) {
    error_message = "Invalid matrix dimensions for NVFP4 gate/up projection.";
    return false;
  }

  const int scale_cols = cols / 16;
  const std::size_t activation_bytes = static_cast<std::size_t>(cols / 2);
  const std::size_t activation_scale_bytes =
    static_cast<std::size_t>(round_up_to_multiple(1, 128)) * round_up_to_multiple(scale_cols, 4);
  if (!check_cuda(cudaMemset(activation_scratch, 0, activation_bytes), "cudaMemset(fp4 gate/up activation)", error_message) ||
      !check_cuda(cudaMemset(activation_scale_scratch, 0, activation_scale_bytes), "cudaMemset(fp4 gate/up activation scales)", error_message)) {
    return false;
  }

  nvfp4_quantize_single_row_for_cublaslt_kernel<<<scale_cols, 16>>>(input_f32, activation_scratch, activation_scale_scratch, cols);
  if (!check_cuda(cudaGetLastError(), "nvfp4_quantize_single_row_for_cublaslt_kernel(gate/up)", error_message)) {
    return false;
  }

  double gate_ms = 0.0;
  double up_ms = 0.0;
  if (!run_cublaslt_fp4_projection(
        activation_scratch,
        activation_scale_scratch,
        gate_packed_weights_cublaslt,
        gate_weight_scales_tiled,
        gate_weight_scale_2,
        rows,
        cols,
        gate_output_f32,
        &gate_ms,
        error_message) ||
      !run_cublaslt_fp4_projection(
        activation_scratch,
        activation_scale_scratch,
        up_packed_weights_cublaslt,
        up_weight_scales_tiled,
        up_weight_scale_2,
        rows,
        cols,
        up_output_f32,
        &up_ms,
        error_message)) {
    return false;
  }

  constexpr int block_size = 256;
  const int grid_size = (rows + block_size - 1) / block_size;
  silu_multiply_kernel<<<grid_size, block_size>>>(gate_output_f32, up_output_f32, gate_output_f32, rows);
  if (!check_cuda(cudaGetLastError(), "silu_multiply_kernel", error_message)) {
    return false;
  }
  if (elapsed_ms != nullptr) {
    *elapsed_ms = gate_ms + up_ms;
  }
  return true;
}

bool run_nvfp4_cublaslt_gate_up_benchmark(
  const std::vector<std::uint8_t> & gate_packed_weights,
  const std::vector<std::uint8_t> & gate_weight_scales_e4m3,
  float gate_weight_scale_2,
  const std::vector<std::uint8_t> & up_packed_weights,
  const std::vector<std::uint8_t> & up_weight_scales_e4m3,
  float up_weight_scale_2,
  int rows,
  int cols,
  int warmup_iterations,
  int benchmark_iterations,
  double & avg_iteration_ms,
  std::string & error_message) {
  if (rows <= 0 || cols <= 0 || (cols % 16) != 0 || (cols % 2) != 0) {
    error_message = "Invalid matrix dimensions for NVFP4 gate/up benchmark.";
    return false;
  }
  if (warmup_iterations < 0 || benchmark_iterations <= 0) {
    error_message = "warmup_iterations must be >= 0 and benchmark_iterations must be > 0.";
    return false;
  }
  const int packed_cols = cols / 2;
  const int scale_cols = cols / 16;
  const std::size_t weight_bytes = static_cast<std::size_t>(rows) * packed_cols;
  const std::size_t scale_bytes = static_cast<std::size_t>(rows) * scale_cols;
  if (gate_packed_weights.size() != weight_bytes || up_packed_weights.size() != weight_bytes ||
      gate_weight_scales_e4m3.size() != scale_bytes || up_weight_scales_e4m3.size() != scale_bytes) {
    error_message = "NVFP4 gate/up benchmark tensor sizes do not match dimensions.";
    return false;
  }

  std::vector<float> input(static_cast<std::size_t>(cols));
  for (int col = 0; col < cols; ++col) {
    input[static_cast<std::size_t>(col)] = static_cast<float>((col % 17) - 8) / 17.0f;
  }
  std::vector<std::uint8_t> gate_scales_linear(static_cast<std::size_t>(rows) * scale_cols, 0);
  std::vector<std::uint8_t> up_scales_linear(static_cast<std::size_t>(rows) * scale_cols, 0);
  for (int row = 0; row < rows; ++row) {
    for (int group = 0; group < scale_cols; ++group) {
      gate_scales_linear[static_cast<std::size_t>(row) * scale_cols + group] =
        encode_ue4m3_scale(decode_e4m3_host(gate_weight_scales_e4m3[static_cast<std::size_t>(row) * scale_cols + group]));
      up_scales_linear[static_cast<std::size_t>(row) * scale_cols + group] =
        encode_ue4m3_scale(decode_e4m3_host(up_weight_scales_e4m3[static_cast<std::size_t>(row) * scale_cols + group]));
    }
  }
  const auto gate_scales = swizzle_nvfp4_block_scale(gate_scales_linear, rows, scale_cols);
  const auto up_scales = swizzle_nvfp4_block_scale(up_scales_linear, rows, scale_cols);
  const auto gate_weights = swap_fp4_nibbles(gate_packed_weights.data(), gate_packed_weights.size());
  const auto up_weights = swap_fp4_nibbles(up_packed_weights.data(), up_packed_weights.size());

  float * d_input = nullptr;
  std::uint8_t * d_activation = nullptr;
  std::uint8_t * d_activation_scales = nullptr;
  std::uint8_t * d_gate_weights = nullptr;
  std::uint8_t * d_gate_scales = nullptr;
  std::uint8_t * d_up_weights = nullptr;
  std::uint8_t * d_up_scales = nullptr;
  float * d_gate_output = nullptr;
  float * d_up_output = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  auto cleanup = [&]() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_input);
    cudaFree(d_activation);
    cudaFree(d_activation_scales);
    cudaFree(d_gate_weights);
    cudaFree(d_gate_scales);
    cudaFree(d_up_weights);
    cudaFree(d_up_scales);
    cudaFree(d_gate_output);
    cudaFree(d_up_output);
  };

  const std::size_t input_bytes = input.size() * sizeof(float);
  const std::size_t activation_bytes = static_cast<std::size_t>(cols / 2);
  const std::size_t activation_scale_bytes = static_cast<std::size_t>(round_up_to_multiple(1, 128)) * round_up_to_multiple(scale_cols, 4);
  const std::size_t output_bytes = static_cast<std::size_t>(rows) * sizeof(float);
  if (!check_cuda(cudaMalloc(&d_input, input_bytes), "cudaMalloc(gate/up input)", error_message) ||
      !check_cuda(cudaMalloc(&d_activation, activation_bytes), "cudaMalloc(gate/up activation)", error_message) ||
      !check_cuda(cudaMalloc(&d_activation_scales, activation_scale_bytes), "cudaMalloc(gate/up activation scales)", error_message) ||
      !check_cuda(cudaMalloc(&d_gate_weights, gate_weights.size()), "cudaMalloc(gate weights)", error_message) ||
      !check_cuda(cudaMalloc(&d_gate_scales, gate_scales.size()), "cudaMalloc(gate scales)", error_message) ||
      !check_cuda(cudaMalloc(&d_up_weights, up_weights.size()), "cudaMalloc(up weights)", error_message) ||
      !check_cuda(cudaMalloc(&d_up_scales, up_scales.size()), "cudaMalloc(up scales)", error_message) ||
      !check_cuda(cudaMalloc(&d_gate_output, output_bytes), "cudaMalloc(gate output)", error_message) ||
      !check_cuda(cudaMalloc(&d_up_output, output_bytes), "cudaMalloc(up output)", error_message)) {
    cleanup();
    return false;
  }
  if (!check_cuda(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(gate/up input)", error_message) ||
      !check_cuda(cudaMemcpy(d_gate_weights, gate_weights.data(), gate_weights.size(), cudaMemcpyHostToDevice), "cudaMemcpy(gate weights)", error_message) ||
      !check_cuda(cudaMemcpy(d_gate_scales, gate_scales.data(), gate_scales.size(), cudaMemcpyHostToDevice), "cudaMemcpy(gate scales)", error_message) ||
      !check_cuda(cudaMemcpy(d_up_weights, up_weights.data(), up_weights.size(), cudaMemcpyHostToDevice), "cudaMemcpy(up weights)", error_message) ||
      !check_cuda(cudaMemcpy(d_up_scales, up_scales.data(), up_scales.size(), cudaMemcpyHostToDevice), "cudaMemcpy(up scales)", error_message)) {
    cleanup();
    return false;
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    if (!run_nvfp4_cublaslt_gate_up_silu_device(
          d_input, d_gate_weights, d_gate_scales, gate_weight_scale_2, d_up_weights, d_up_scales, up_weight_scale_2,
          rows, cols, d_activation, d_activation_scales, d_gate_output, d_up_output, nullptr, error_message)) {
      cleanup();
      return false;
    }
  }
  if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(gate/up warmup)", error_message) ||
      !check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(gate/up start)", error_message) ||
      !check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(gate/up stop)", error_message) ||
      !check_cuda(cudaEventRecord(start_event), "cudaEventRecord(gate/up start)", error_message)) {
    cleanup();
    return false;
  }
  for (int i = 0; i < benchmark_iterations; ++i) {
    if (!run_nvfp4_cublaslt_gate_up_silu_device(
          d_input, d_gate_weights, d_gate_scales, gate_weight_scale_2, d_up_weights, d_up_scales, up_weight_scale_2,
          rows, cols, d_activation, d_activation_scales, d_gate_output, d_up_output, nullptr, error_message)) {
      cleanup();
      return false;
    }
  }
  float total_ms = 0.0f;
  if (!check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(gate/up stop)", error_message) ||
      !check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(gate/up stop)", error_message) ||
      !check_cuda(cudaEventElapsedTime(&total_ms, start_event, stop_event), "cudaEventElapsedTime(gate/up)", error_message)) {
    cleanup();
    return false;
  }
  avg_iteration_ms = static_cast<double>(total_ms) / static_cast<double>(benchmark_iterations);
  cleanup();
  return true;
}

} // namespace qwen35x::cuda

#endif
