#include "qwen35x/cpu/activation.h"

#include <immintrin.h>

#include <cmath>
#include <cstddef>

namespace qwen35x::cpu::detail {

namespace {

[[nodiscard]] __m256 exp_f32_avx2(__m256 value) noexcept {
  value = _mm256_min_ps(value, _mm256_set1_ps(88.3762626647949F));
  value = _mm256_max_ps(value, _mm256_set1_ps(-88.3762626647949F));
  __m256 exponent = _mm256_fmadd_ps(
    value, _mm256_set1_ps(1.44269504088896341F), _mm256_set1_ps(0.5F));
  exponent = _mm256_floor_ps(exponent);
  value = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(0.693359375F), value);
  value = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(-2.12194440e-4F), value);
  const __m256 squared = _mm256_mul_ps(value, value);
  __m256 polynomial = _mm256_set1_ps(1.9875691500e-4F);
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(1.3981999507e-3F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(8.3334519073e-3F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(4.1665795894e-2F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(1.6666665459e-1F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(5.0000001201e-1F));
  polynomial = _mm256_fmadd_ps(polynomial, squared, value);
  polynomial = _mm256_add_ps(polynomial, _mm256_set1_ps(1.0F));
  __m256i integer_exponent = _mm256_cvttps_epi32(exponent);
  integer_exponent = _mm256_add_epi32(integer_exponent, _mm256_set1_epi32(127));
  integer_exponent = _mm256_slli_epi32(integer_exponent, 23);
  return _mm256_mul_ps(polynomial, _mm256_castsi256_ps(integer_exponent));
}

} // namespace

void silu_mul_f32_avx2(
  const float * gate,
  const float * up,
  float * output,
  const std::size_t count) noexcept {
  const __m256 one = _mm256_set1_ps(1.0F);
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 value = _mm256_loadu_ps(gate + index);
    const __m256 sigmoid = _mm256_div_ps(
      one, _mm256_add_ps(one, exp_f32_avx2(_mm256_sub_ps(_mm256_setzero_ps(), value))));
    _mm256_storeu_ps(
      output + index,
      _mm256_mul_ps(_mm256_mul_ps(value, sigmoid), _mm256_loadu_ps(up + index)));
  }
  for (; index < count; ++index) {
    const float value = gate[index];
    const float sigmoid = value >= 0.0F
      ? 1.0F / (1.0F + std::exp(-value))
      : std::exp(value) / (1.0F + std::exp(value));
    output[index] = value * sigmoid * up[index];
  }
}

} // namespace qwen35x::cpu::detail
