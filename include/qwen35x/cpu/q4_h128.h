#pragma once

#include "qwen35x/cpu/q4_0.h"

#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu {

inline constexpr std::size_t q4_h128_transform_size = 128;
inline constexpr std::size_t q4_h128_q4_blocks_per_transform =
  q4_h128_transform_size / q4_0_values_per_block;
inline constexpr std::uint64_t q4_h128_default_sign_seed =
  UINT64_C(0x5148333548313238);

// Applies R = H_128 D / sqrt(128), where D is a deterministic randomized-sign
// diagonal. transform_block_index is the block index within an activation
// vector and therefore deliberately does not depend on a tensor name: all
// projections sharing an activation must use the same basis.
void q4_h128_transform_block(
  const float * input,
  float * output,
  std::size_t transform_block_index,
  std::uint64_t sign_seed = q4_h128_default_sign_seed,
  Q8_0Backend backend = Q8_0Backend::scalar) noexcept;

void q4_h128_transform_block_inplace(
  float * values,
  std::size_t transform_block_index,
  std::uint64_t sign_seed = q4_h128_default_sign_seed,
  Q8_0Backend backend = Q8_0Backend::scalar) noexcept;

// rows are row-major and columns must be a non-zero multiple of 128. The
// transform block index restarts at zero for each row.
[[nodiscard]] bool q4_h128_transform_rows(
  const float * input,
  float * output,
  std::size_t row_count,
  std::size_t column_count,
  std::uint64_t sign_seed = q4_h128_default_sign_seed,
  Q8_0Backend backend = Q8_0Backend::scalar) noexcept;

// Quantizes already-transformed values into the engine's signed Q4 nibble
// layout with one binary16 symmetric scale per 32 values.
void q4_h128_quantize_transformed(
  const float * input,
  Q4_0Block * output,
  std::size_t q4_block_count) noexcept;

// Direct BF16/F32 conversion primitive for an eligible projection matrix.
[[nodiscard]] bool q4_h128_quantize_matrix(
  const float * input,
  Q4_0Block * output,
  std::size_t row_count,
  std::size_t column_count,
  std::uint64_t sign_seed = q4_h128_default_sign_seed) noexcept;

// Transforms one activation vector and quantizes the result to the existing
// Q8_0 dot-product layout. transformed_scratch holds column_count floats.
[[nodiscard]] bool q4_h128_prepare_activation(
  const float * input,
  float * transformed_scratch,
  Q8_0Block * output,
  std::size_t column_count,
  std::uint64_t sign_seed = q4_h128_default_sign_seed,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

} // namespace qwen35x::cpu
