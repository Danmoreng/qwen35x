#pragma once

#include "qwen35x/cpu/q8_0.h"

#include <cstddef>

namespace qwen35x::cpu {

// Applies RMS normalization to row_count rows. The width-element weight is
// shared by all rows and weight_offset supports Qwen's (1 + weight) form.
void rms_norm_f32(
  const float * input,
  const float * weight,
  float * output,
  std::size_t row_count,
  std::size_t width,
  float eps,
  float weight_offset,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

void silu_f32(
  const float * input,
  float * output,
  std::size_t count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Computes output[i] = silu(gate[i]) * up[i]. The AVX2 implementation uses
// the same bounded exp approximation as the optimized attention kernels.
void silu_mul_f32(
  const float * gate,
  const float * up,
  float * output,
  std::size_t count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

} // namespace qwen35x::cpu
