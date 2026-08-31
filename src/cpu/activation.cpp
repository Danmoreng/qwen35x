#include "qwen35x/cpu/activation.h"

#include <cmath>
#include <cstddef>

#ifndef QWEN35X_Q8_0_HAS_AVX2_TU
#define QWEN35X_Q8_0_HAS_AVX2_TU 0
#endif

namespace qwen35x::cpu {

namespace detail {

#if QWEN35X_Q8_0_HAS_AVX2_TU
void silu_f32_avx2(
  const float * input,
  float * output,
  std::size_t count) noexcept;

void silu_mul_f32_avx2(
  const float * gate,
  const float * up,
  float * output,
  std::size_t count) noexcept;
#endif

} // namespace detail

void silu_f32(
  const float * input,
  float * output,
  const std::size_t count,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::silu_f32_avx2(input, output, count);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  for (std::size_t index = 0; index < count; ++index) {
    const float value = input[index];
    const float exponential = std::exp(-std::fabs(value));
    const float sigmoid = value >= 0.0F
      ? 1.0F / (1.0F + exponential)
      : exponential / (1.0F + exponential);
    output[index] = value * sigmoid;
  }
}

void silu_mul_f32(
  const float * gate,
  const float * up,
  float * output,
  const std::size_t count,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::silu_mul_f32_avx2(gate, up, output, count);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  for (std::size_t index = 0; index < count; ++index) {
    const float value = gate[index];
    const float exponential = std::exp(-std::fabs(value));
    const float sigmoid = value >= 0.0F
      ? 1.0F / (1.0F + exponential)
      : exponential / (1.0F + exponential);
    output[index] = value * sigmoid * up[index];
  }
}

} // namespace qwen35x::cpu
