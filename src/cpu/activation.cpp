#include "qwen35x/cpu/activation.h"

#include <cmath>
#include <cstddef>

#ifndef QWEN35X_Q8_0_HAS_AVX2_TU
#define QWEN35X_Q8_0_HAS_AVX2_TU 0
#endif

namespace qwen35x::cpu {

namespace detail {

#if QWEN35X_Q8_0_HAS_AVX2_TU
void rms_norm_f32_avx2(
  const float * input,
  const float * weight,
  float * output,
  std::size_t row_count,
  std::size_t width,
  float eps,
  float weight_offset) noexcept;

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

void rms_norm_f32(
  const float * input,
  const float * weight,
  float * output,
  const std::size_t row_count,
  const std::size_t width,
  const float eps,
  const float weight_offset,
  const Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_resolve_backend(backend) == Q8_0Backend::avx2) {
    detail::rms_norm_f32_avx2(
      input, weight, output, row_count, width, eps, weight_offset);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  for (std::size_t row = 0; row < row_count; ++row) {
    const float * x = input + row * width;
    float * y = output + row * width;
    float squared_sum = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      squared_sum += x[column] * x[column];
    }
    const float inverse = 1.0F /
      std::sqrt(squared_sum / static_cast<float>(width) + eps);
    for (std::size_t column = 0; column < width; ++column) {
      y[column] = x[column] * inverse * (weight[column] + weight_offset);
    }
  }
}

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
