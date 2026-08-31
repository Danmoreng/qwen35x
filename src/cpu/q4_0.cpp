#include "qwen35x/cpu/q4_0.h"

#include "q4_0_internal.h"
#include "q8_0_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu {
namespace detail {

void q4_0_dequantize_scalar(
  const Q4_0Block * input,
  float * output,
  const std::size_t block_count) noexcept {
  for (std::size_t block = 0; block < block_count; ++block) {
    const float scale = half_to_float(input[block].d);
    float * values = output + block * q4_0_values_per_block;
    for (std::size_t index = 0; index < q4_0_values_per_block / 2; ++index) {
      const std::uint8_t packed = input[block].qs[index];
      values[index] = scale * static_cast<float>(static_cast<int>(packed & 0x0fU) - 8);
      values[index + 16] = scale * static_cast<float>(static_cast<int>(packed >> 4U) - 8);
    }
  }
}

float q4_0_dot_q8_0_scalar(
  const Q4_0Block * weights,
  const Q8_0Block * activations,
  const std::size_t block_count) noexcept {
  float result = 0.0F;
  for (std::size_t block = 0; block < block_count; ++block) {
    std::int32_t integer_dot = 0;
    for (std::size_t index = 0; index < 16; ++index) {
      const std::uint8_t packed = weights[block].qs[index];
      integer_dot += (static_cast<int>(packed & 0x0fU) - 8) *
        static_cast<int>(activations[block].qs[index]);
      integer_dot += (static_cast<int>(packed >> 4U) - 8) *
        static_cast<int>(activations[block].qs[index + 16]);
    }
    result += half_to_float(weights[block].d) * half_to_float(activations[block].d) *
      static_cast<float>(integer_dot);
  }
  return result;
}

void q4_0_matvec_q8_0_scalar(
  const Q4_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  if (blocks_per_row == 0) {
    std::fill_n(output, row_count, 0.0F);
    return;
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    output[row] = q4_0_dot_q8_0_scalar(
      matrix + row * blocks_per_row, vector, blocks_per_row);
  }
}

void q4_0_matmul_q8_0_scalar(
  const Q4_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const std::size_t output_row_stride,
  const float * vector_scales,
  const float * matrix_scales) noexcept {
  for (std::size_t row = 0; row < row_count; ++row) {
    const Q4_0Block * matrix_row = matrix + row * blocks_per_row;
    for (std::size_t vector_index = 0; vector_index < vector_count; ++vector_index) {
      float result = 0.0F;
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const std::size_t vector_offset = vector_index * blocks_per_row + block;
        std::int32_t integer_dot = 0;
        for (std::size_t index = 0; index < 16; ++index) {
          const std::uint8_t packed = matrix_row[block].qs[index];
          integer_dot += (static_cast<int>(packed & 0x0fU) - 8) *
            static_cast<int>(vectors[vector_offset].qs[index]);
          integer_dot += (static_cast<int>(packed >> 4U) - 8) *
            static_cast<int>(vectors[vector_offset].qs[index + 16]);
        }
        const float matrix_scale = matrix_scales != nullptr
          ? matrix_scales[row * blocks_per_row + block]
          : half_to_float(matrix_row[block].d);
        const float vector_scale = vector_scales != nullptr
          ? vector_scales[vector_offset]
          : half_to_float(vectors[vector_offset].d);
        result += matrix_scale * vector_scale * static_cast<float>(integer_dot);
      }
      output[vector_index * output_row_stride + row] = result;
    }
  }
}

void q4_0_packed_matmul_q8_0_scalar(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX4 * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const std::size_t output_row_stride) noexcept {
  const std::size_t row_tiles = row_count / q4_0_packed_rows;
  const std::size_t vector_tiles = vector_count / q8_0_packed_vectors;
  for (std::size_t vector_tile = 0; vector_tile < vector_tiles; ++vector_tile) {
    for (std::size_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
      float accumulators[q8_0_packed_vectors][q4_0_packed_rows]{};
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const Q4_0BlockX8 & weights = matrix[row_tile * blocks_per_row + block];
        const Q8_0BlockX4 & activations = vectors[vector_tile * blocks_per_row + block];
        for (std::size_t token = 0; token < q8_0_packed_vectors; ++token) {
          const float activation_scale = half_to_float(activations.d[token]);
          for (std::size_t row = 0; row < q4_0_packed_rows; ++row) {
            std::int32_t integer_dot = 0;
            for (std::size_t index = 0; index < 16; ++index) {
              const std::size_t chunk = index / 8;
              const std::size_t within_chunk = index % 8;
              const std::uint8_t packed = static_cast<std::uint8_t>(
                weights.qs[chunk * 64 + row * 8 + within_chunk] ^ 0x88U);
              integer_dot += (static_cast<int>(packed & 0x0fU) - 8) *
                static_cast<int>(activations.qs[chunk * 32 + token * 8 + within_chunk]);
              integer_dot += (static_cast<int>(packed >> 4U) - 8) *
                static_cast<int>(activations.qs[(chunk + 2) * 32 + token * 8 + within_chunk]);
            }
            accumulators[token][row] +=
              half_to_float(weights.d[row]) * activation_scale *
              static_cast<float>(integer_dot);
          }
        }
      }
      for (std::size_t token = 0; token < q8_0_packed_vectors; ++token) {
        for (std::size_t row = 0; row < q4_0_packed_rows; ++row) {
          output[(vector_tile * 4 + token) * output_row_stride + row_tile * 8 + row] =
            accumulators[token][row];
        }
      }
    }
  }
}

void q8_0_quantize_vectors_4_scalar(
  const float * input,
  Q8_0BlockX4 * packed,
  const std::size_t vector_count,
  const std::size_t blocks_per_vector) noexcept {
  const std::size_t values_per_vector = blocks_per_vector * q8_0_values_per_block;
  for (std::size_t vector_tile = 0; vector_tile < vector_count / 4; ++vector_tile) {
    for (std::size_t block = 0; block < blocks_per_vector; ++block) {
      Q8_0BlockX4 & destination = packed[vector_tile * blocks_per_vector + block];
      for (std::size_t token = 0; token < 4; ++token) {
        const float * source = input +
          (vector_tile * 4 + token) * values_per_vector + block * 32;
        float absolute_max = 0.0F;
        for (std::size_t index = 0; index < 32; ++index) {
          absolute_max = std::max(absolute_max, std::fabs(source[index]));
        }
        const float scale = absolute_max / 127.0F;
        const float inverse_scale = scale == 0.0F ? 0.0F : 1.0F / scale;
        destination.d[token] = float_to_half(scale);
        std::int32_t sum = 0;
        for (std::size_t index = 0; index < 32; ++index) {
          const float rounded = std::round(source[index] * inverse_scale);
          const auto quantized = static_cast<std::int8_t>(
            std::clamp(rounded, -127.0F, 127.0F));
          destination.qs[(index / 8) * 32 + token * 8 + index % 8] = quantized;
          sum += quantized;
        }
        destination.sums[token] = static_cast<std::int16_t>(sum);
      }
    }
  }
}

void q4_0_packed_matvec_q8_0_scalar(
  const Q4_0BlockX8 * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  for (std::size_t row_tile = 0; row_tile < row_count / 8; ++row_tile) {
    float accumulators[8]{};
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      const Q4_0BlockX8 & weights = matrix[row_tile * blocks_per_row + block];
      for (std::size_t row = 0; row < 8; ++row) {
        std::int32_t integer_dot = 0;
        for (std::size_t index = 0; index < 16; ++index) {
          const std::size_t chunk = index / 8;
          const std::size_t within_chunk = index % 8;
          const std::uint8_t packed = static_cast<std::uint8_t>(
            weights.qs[chunk * 64 + row * 8 + within_chunk] ^ 0x88U);
          integer_dot += (static_cast<int>(packed & 0x0fU) - 8) *
            static_cast<int>(vector[block].qs[index]);
          integer_dot += (static_cast<int>(packed >> 4U) - 8) *
            static_cast<int>(vector[block].qs[index + 16]);
        }
        accumulators[row] += half_to_float(weights.d[row]) *
          half_to_float(vector[block].d) * static_cast<float>(integer_dot);
      }
    }
    std::copy_n(accumulators, 8, output + row_tile * 8);
  }
}

} // namespace detail

void q4_0_pack_rows_8(
  const Q4_0Block * canonical,
  Q4_0BlockX8 * packed,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  for (std::size_t row_tile = 0; row_tile < row_count / q4_0_packed_rows; ++row_tile) {
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      Q4_0BlockX8 & destination = packed[row_tile * blocks_per_row + block];
      for (std::size_t row = 0; row < q4_0_packed_rows; ++row) {
        const Q4_0Block & source =
          canonical[(row_tile * q4_0_packed_rows + row) * blocks_per_row + block];
        destination.d[row] = source.d;
        for (std::size_t chunk = 0; chunk < 2; ++chunk) {
          for (std::size_t index = 0; index < 8; ++index) {
            destination.qs[chunk * 64 + row * 8 + index] =
              static_cast<std::uint8_t>(source.qs[chunk * 8 + index] ^ 0x88U);
          }
        }
      }
    }
  }
}

void q8_0_pack_vectors_4(
  const Q8_0Block * canonical,
  Q8_0BlockX4 * packed,
  const std::size_t vector_count,
  const std::size_t blocks_per_vector) noexcept {
  for (std::size_t vector_tile = 0;
       vector_tile < vector_count / q8_0_packed_vectors;
       ++vector_tile) {
    for (std::size_t block = 0; block < blocks_per_vector; ++block) {
      Q8_0BlockX4 & destination = packed[vector_tile * blocks_per_vector + block];
      for (std::size_t token = 0; token < q8_0_packed_vectors; ++token) {
        const Q8_0Block & source =
          canonical[(vector_tile * q8_0_packed_vectors + token) * blocks_per_vector + block];
        destination.d[token] = source.d;
        std::int32_t sum = 0;
        for (std::size_t chunk = 0; chunk < 4; ++chunk) {
          std::copy_n(
            source.qs + chunk * 8,
            8,
            destination.qs + chunk * 32 + token * 8);
          for (std::size_t index = 0; index < 8; ++index) {
            sum += source.qs[chunk * 8 + index];
          }
        }
        destination.sums[token] = static_cast<std::int16_t>(sum);
      }
    }
  }
}

void q8_0_quantize_vectors_4(
  const float * input,
  Q8_0BlockX4 * packed,
  const std::size_t vector_count,
  const std::size_t blocks_per_vector,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::q8_0_quantize_vectors_4_avx2(
      input, packed, vector_count, blocks_per_vector);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::q8_0_quantize_vectors_4_scalar(
    input, packed, vector_count, blocks_per_vector);
}

void q4_0_packed_dequantize_row(
  const Q4_0BlockX8 * matrix,
  const std::size_t row_index,
  float * output,
  const std::size_t blocks_per_row) noexcept {
  const std::size_t row_tile = row_index / 8;
  const std::size_t row = row_index % 8;
  for (std::size_t block = 0; block < blocks_per_row; ++block) {
    const Q4_0BlockX8 & source = matrix[row_tile * blocks_per_row + block];
    const float scale = detail::half_to_float(source.d[row]);
    float * destination = output + block * 32;
    for (std::size_t index = 0; index < 16; ++index) {
      const std::size_t chunk = index / 8;
      const std::size_t within_chunk = index % 8;
      const std::uint8_t packed = static_cast<std::uint8_t>(
        source.qs[chunk * 64 + row * 8 + within_chunk] ^ 0x88U);
      destination[index] = scale *
        static_cast<float>(static_cast<int>(packed & 0x0fU) - 8);
      destination[index + 16] = scale *
        static_cast<float>(static_cast<int>(packed >> 4U) - 8);
    }
  }
}

void q4_0_dequantize(
  const Q4_0Block * input,
  float * output,
  const std::size_t block_count,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::q4_0_dequantize_avx2(input, output, block_count);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::q4_0_dequantize_scalar(input, output, block_count);
}

void q4_0_scales_to_f32(
  const Q4_0Block * input,
  float * output,
  const std::size_t block_count) noexcept {
  for (std::size_t block = 0; block < block_count; ++block) {
    output[block] = detail::half_to_float(input[block].d);
  }
}

float q4_0_dot_q8_0(
  const Q4_0Block * weights,
  const Q8_0Block * activations,
  const std::size_t block_count,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    return detail::q4_0_dot_q8_0_avx2(weights, activations, block_count);
  }
#else
  static_cast<void>(backend);
#endif
  return detail::q4_0_dot_q8_0_scalar(weights, activations, block_count);
}

void q4_0_matvec_q8_0(
  const Q4_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::q4_0_matvec_q8_0_avx2(matrix, vector, output, row_count, blocks_per_row);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::q4_0_matvec_q8_0_scalar(matrix, vector, output, row_count, blocks_per_row);
}

void q4_0_matmul_q8_0(
  const Q4_0Block * matrix,
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
    detail::q4_0_matmul_q8_0_avx2(
      matrix, vectors, output, row_count, vector_count, blocks_per_row,
      output_row_stride, vector_scales, matrix_scales);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::q4_0_matmul_q8_0_scalar(
    matrix, vectors, output, row_count, vector_count, blocks_per_row,
    output_row_stride, vector_scales, matrix_scales);
}

void q4_0_packed_matmul_q8_0(
  const Q4_0BlockX8 * matrix,
  const Q8_0BlockX4 * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const std::size_t output_row_stride,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::q4_0_packed_matmul_q8_0_avx2(
      matrix, vectors, output, row_count, vector_count, blocks_per_row,
      output_row_stride);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::q4_0_packed_matmul_q8_0_scalar(
    matrix, vectors, output, row_count, vector_count, blocks_per_row,
    output_row_stride);
}

void q4_0_packed_matvec_q8_0(
  const Q4_0BlockX8 * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::q4_0_packed_matvec_q8_0_avx2(
      matrix, vector, output, row_count, blocks_per_row);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::q4_0_packed_matvec_q8_0_scalar(
    matrix, vector, output, row_count, blocks_per_row);
}

} // namespace qwen35x::cpu
