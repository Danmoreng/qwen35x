#include "qwen35x/cpu/q8_0.h"

#include "q8_0_internal.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

#if QWEN35X_Q8_0_HAS_AVX2_TU && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace qwen35x::cpu {

namespace detail {

std::uint16_t float_to_half(const float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t magnitude = bits & 0x7fffffffU;

  if (magnitude >= 0x7f800000U) {
    if (magnitude == 0x7f800000U) {
      return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    const std::uint16_t payload = static_cast<std::uint16_t>((magnitude >> 13U) & 0x03ffU);
    return static_cast<std::uint16_t>(sign | 0x7c00U | payload | (payload == 0));
  }

  // Values above the largest finite binary16 number round to infinity.
  if (magnitude > 0x477fefffU) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }

  if (magnitude < 0x38800000U) {
    // Everything below half the smallest subnormal rounds to signed zero.
    if (magnitude < 0x33000000U) {
      return static_cast<std::uint16_t>(sign);
    }

    const std::uint32_t exponent = magnitude >> 23U;
    const std::uint32_t mantissa = (magnitude & 0x007fffffU) | 0x00800000U;
    const std::uint32_t shift = 126U - exponent;
    std::uint32_t half_mantissa = mantissa >> shift;
    const std::uint32_t remainder_mask = (std::uint32_t{1} << shift) - 1U;
    const std::uint32_t remainder = mantissa & remainder_mask;
    const std::uint32_t halfway = std::uint32_t{1} << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0U)) {
      ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }

  const std::uint32_t rounded = magnitude + 0x00000fffU + ((magnitude >> 13U) & 1U);
  return static_cast<std::uint16_t>(sign | ((rounded - 0x38000000U) >> 13U));
}

float half_to_float(const std::uint16_t value) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  const std::uint32_t exponent = (value >> 10U) & 0x1fU;
  std::uint32_t mantissa = value & 0x03ffU;
  std::uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      std::uint32_t normalized_exponent = 113U;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        --normalized_exponent;
      }
      mantissa &= 0x03ffU;
      bits = sign | (normalized_exponent << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  }

  return std::bit_cast<float>(bits);
}

void q8_0_quantize_scalar(
  const float * input,
  Q8_0Block * output,
  const std::size_t block_count) noexcept {
  for (std::size_t block = 0; block < block_count; ++block) {
    const float * x = input + block * q8_0_values_per_block;
    float absolute_max = 0.0F;
    for (std::size_t index = 0; index < q8_0_values_per_block; ++index) {
      absolute_max = std::max(absolute_max, std::fabs(x[index]));
    }

    const float scale = absolute_max / 127.0F;
    const float inverse_scale = scale == 0.0F ? 0.0F : 1.0F / scale;
    output[block].d = float_to_half(scale);
    for (std::size_t index = 0; index < q8_0_values_per_block; ++index) {
      const float rounded = std::round(x[index] * inverse_scale);
      output[block].qs[index] = static_cast<std::int8_t>(
        std::clamp(rounded, -127.0F, 127.0F));
    }
  }
}

void q8_0_dequantize_scalar(
  const Q8_0Block * input,
  float * output,
  const std::size_t block_count) noexcept {
  for (std::size_t block = 0; block < block_count; ++block) {
    const float scale = half_to_float(input[block].d);
    for (std::size_t index = 0; index < q8_0_values_per_block; ++index) {
      output[block * q8_0_values_per_block + index] =
        scale * static_cast<float>(input[block].qs[index]);
    }
  }
}

float q8_0_dot_scalar(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  const std::size_t block_count) noexcept {
  float result = 0.0F;
  for (std::size_t block = 0; block < block_count; ++block) {
    std::int32_t integer_dot = 0;
    for (std::size_t index = 0; index < q8_0_values_per_block; ++index) {
      integer_dot += static_cast<std::int32_t>(lhs[block].qs[index]) *
        static_cast<std::int32_t>(rhs[block].qs[index]);
    }
    const float scale = half_to_float(lhs[block].d) * half_to_float(rhs[block].d);
    result += scale * static_cast<float>(integer_dot);
  }
  return result;
}

void q8_0_matvec_scalar(
  const Q8_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  if (blocks_per_row == 0) {
    std::fill_n(output, row_count, 0.0F);
    return;
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    output[row] = q8_0_dot_scalar(
      matrix + row * blocks_per_row,
      vector,
      blocks_per_row);
  }
}

void q8_0_matmul_scalar(
  const Q8_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const std::size_t output_row_stride,
  const float * vector_scales,
  const float * matrix_scales) noexcept {
  for (std::size_t row = 0; row < row_count; ++row) {
    const Q8_0Block * matrix_row = matrix + row * blocks_per_row;
    for (std::size_t vector_index = 0; vector_index < vector_count; ++vector_index) {
      float result = 0.0F;
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const std::size_t vector_offset = vector_index * blocks_per_row + block;
        const Q8_0Block & vector_block = vectors[vector_offset];
        std::int32_t integer_dot = 0;
        for (std::size_t index = 0; index < q8_0_values_per_block; ++index) {
          integer_dot += static_cast<std::int32_t>(matrix_row[block].qs[index]) *
            static_cast<std::int32_t>(vector_block.qs[index]);
        }
        const float matrix_scale = matrix_scales != nullptr
          ? matrix_scales[row * blocks_per_row + block]
          : half_to_float(matrix_row[block].d);
        const float vector_scale = vector_scales != nullptr
          ? vector_scales[vector_offset]
          : half_to_float(vector_block.d);
        result += matrix_scale * vector_scale * static_cast<float>(integer_dot);
      }
      output[vector_index * output_row_stride + row] = result;
    }
  }
}

} // namespace detail

namespace {

bool avx2_runtime_available() noexcept {
#if !QWEN35X_Q8_0_HAS_AVX2_TU
  return false;
#elif defined(_MSC_VER)
  int registers[4] = {};
  __cpuid(registers, 1);
  constexpr int osxsave_bit = 1 << 27;
  constexpr int avx_bit = 1 << 28;
  constexpr int f16c_bit = 1 << 29;
  constexpr int fma_bit = 1 << 12;
  if ((registers[2] & (osxsave_bit | avx_bit | f16c_bit | fma_bit)) !=
      (osxsave_bit | avx_bit | f16c_bit | fma_bit)) {
    return false;
  }
  if ((_xgetbv(0) & 0x6U) != 0x6U) {
    return false;
  }
  __cpuidex(registers, 7, 0);
  return (registers[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") &&
    __builtin_cpu_supports("fma") &&
    __builtin_cpu_supports("f16c");
#else
  return false;
#endif
}

} // namespace

bool q8_0_backend_available(const Q8_0Backend backend) noexcept {
  switch (backend) {
    case Q8_0Backend::auto_select:
    case Q8_0Backend::scalar:
      return true;
    case Q8_0Backend::avx2: {
      static const bool available = avx2_runtime_available();
      return available;
    }
  }
  return false;
}

Q8_0Backend q8_0_resolve_backend(const Q8_0Backend requested) noexcept {
  if (requested == Q8_0Backend::scalar) {
    return Q8_0Backend::scalar;
  }
  return q8_0_backend_available(Q8_0Backend::avx2)
    ? Q8_0Backend::avx2
    : Q8_0Backend::scalar;
}

const char * q8_0_backend_name(const Q8_0Backend backend) noexcept {
  switch (backend) {
    case Q8_0Backend::auto_select:
      return "auto";
    case Q8_0Backend::scalar:
      return "scalar";
    case Q8_0Backend::avx2:
      return "avx2+fma+f16c";
  }
  return "unknown";
}

void q8_0_quantize(
  const float * input,
  Q8_0Block * output,
  const std::size_t block_count,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::q8_0_quantize_avx2(input, output, block_count);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::q8_0_quantize_scalar(input, output, block_count);
}

void q8_0_dequantize(
  const Q8_0Block * input,
  float * output,
  const std::size_t block_count,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::q8_0_dequantize_avx2(input, output, block_count);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::q8_0_dequantize_scalar(input, output, block_count);
}

void q8_0_scales_to_f32(
  const Q8_0Block * input,
  float * output,
  const std::size_t block_count) noexcept {
  for (std::size_t block = 0; block < block_count; ++block) {
    output[block] = detail::half_to_float(input[block].d);
  }
}

float q8_0_dot(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  const std::size_t block_count,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    return detail::q8_0_dot_avx2(lhs, rhs, block_count);
  }
#else
  static_cast<void>(backend);
#endif
  return detail::q8_0_dot_scalar(lhs, rhs, block_count);
}

void q8_0_matvec(
  const Q8_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::q8_0_matvec_avx2(matrix, vector, output, row_count, blocks_per_row);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::q8_0_matvec_scalar(matrix, vector, output, row_count, blocks_per_row);
}

void q8_0_matmul(
  const Q8_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const std::size_t output_row_stride,
  const Q8_0Backend backend,
  const float * vector_scales,
  const float * matrix_scales) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::q8_0_matmul_avx2(
      matrix,
      vectors,
      output,
      row_count,
      vector_count,
      blocks_per_row,
      output_row_stride,
      vector_scales,
      matrix_scales);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::q8_0_matmul_scalar(
    matrix,
    vectors,
    output,
    row_count,
    vector_count,
    blocks_per_row,
    output_row_stride,
    vector_scales,
    matrix_scales);
}

} // namespace qwen35x::cpu
