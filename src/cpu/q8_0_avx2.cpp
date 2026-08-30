#include "q8_0_internal.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu::detail {

namespace {

[[nodiscard]] float horizontal_sum_f32(const __m256 value) noexcept {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

[[nodiscard]] __m256i dot_block_i8x8(
  const std::int8_t * lhs,
  const std::int8_t * rhs) noexcept {
  const __m256i lhs_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lhs));
  const __m256i rhs_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rhs));

  // Q8_0 never emits -128, so abs(lhs) fits in unsigned bytes and every
  // adjacent pair product fits in signed int16. This maps signed*signed i8
  // onto AVX2's unsigned*signed maddubs instruction without saturation.
  const __m256i lhs_absolute = _mm256_abs_epi8(lhs_bytes);
  const __m256i rhs_with_lhs_sign = _mm256_sign_epi8(rhs_bytes, lhs_bytes);
  const __m256i pair_products = _mm256_maddubs_epi16(lhs_absolute, rhs_with_lhs_sign);
  return _mm256_madd_epi16(pair_products, _mm256_set1_epi16(1));
}

[[nodiscard]] __m256 quantize_eight(
  const float * values,
  const __m256 inverse_scale) noexcept {
  const __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(values), inverse_scale);
  const __m256 sign = _mm256_and_ps(scaled, _mm256_set1_ps(-0.0F));
  const __m256 half_away_from_zero = _mm256_or_ps(_mm256_set1_ps(0.5F), sign);
  return _mm256_add_ps(scaled, half_away_from_zero);
}

[[nodiscard]] float dot_avx2_impl(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  const std::size_t block_count) noexcept {
  // Keep eight independent FP32 accumulation lanes across all Q8 blocks and
  // reduce only once per row. Reducing each 32-value block separately creates
  // a long scalar dependency chain and leaves most of AVX2 idle.
  __m256 accumulator = _mm256_setzero_ps();
  for (std::size_t block = 0; block < block_count; ++block) {
    const __m256 integer_dot = _mm256_cvtepi32_ps(dot_block_i8x8(lhs[block].qs, rhs[block].qs));
    const float lhs_scale = _cvtsh_ss(lhs[block].d);
    const float rhs_scale = _cvtsh_ss(rhs[block].d);
    accumulator = _mm256_fmadd_ps(
      integer_dot,
      _mm256_set1_ps(lhs_scale * rhs_scale),
      accumulator);
  }
  return horizontal_sum_f32(accumulator);
}

} // namespace

void q8_0_quantize_avx2(
  const float * input,
  Q8_0Block * output,
  const std::size_t block_count) noexcept {
  const __m256 sign_bit = _mm256_set1_ps(-0.0F);
  const __m256i pack_order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

  for (std::size_t block = 0; block < block_count; ++block) {
    const float * x = input + block * q8_0_values_per_block;
    const __m256 x0 = _mm256_loadu_ps(x);
    const __m256 x1 = _mm256_loadu_ps(x + 8);
    const __m256 x2 = _mm256_loadu_ps(x + 16);
    const __m256 x3 = _mm256_loadu_ps(x + 24);

    __m256 absolute_max = _mm256_andnot_ps(sign_bit, x0);
    absolute_max = _mm256_max_ps(absolute_max, _mm256_andnot_ps(sign_bit, x1));
    absolute_max = _mm256_max_ps(absolute_max, _mm256_andnot_ps(sign_bit, x2));
    absolute_max = _mm256_max_ps(absolute_max, _mm256_andnot_ps(sign_bit, x3));
    __m128 max4 = _mm_max_ps(
      _mm256_castps256_ps128(absolute_max),
      _mm256_extractf128_ps(absolute_max, 1));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_shuffle_ps(max4, max4, 1));
    const float max_scalar = _mm_cvtss_f32(max4);
    const float scale = max_scalar / 127.0F;
    const float inverse_scale_scalar = scale == 0.0F ? 0.0F : 1.0F / scale;
    output[block].d = float_to_half(scale);
    const __m256 inverse_scale = _mm256_set1_ps(inverse_scale_scalar);

    __m256i i0 = _mm256_cvttps_epi32(quantize_eight(x, inverse_scale));
    __m256i i1 = _mm256_cvttps_epi32(quantize_eight(x + 8, inverse_scale));
    __m256i i2 = _mm256_cvttps_epi32(quantize_eight(x + 16, inverse_scale));
    __m256i i3 = _mm256_cvttps_epi32(quantize_eight(x + 24, inverse_scale));
    const __m256i minimum = _mm256_set1_epi32(-127);
    const __m256i maximum = _mm256_set1_epi32(127);
    i0 = _mm256_min_epi32(_mm256_max_epi32(i0, minimum), maximum);
    i1 = _mm256_min_epi32(_mm256_max_epi32(i1, minimum), maximum);
    i2 = _mm256_min_epi32(_mm256_max_epi32(i2, minimum), maximum);
    i3 = _mm256_min_epi32(_mm256_max_epi32(i3, minimum), maximum);
    i0 = _mm256_packs_epi32(i0, i1);
    i2 = _mm256_packs_epi32(i2, i3);
    i0 = _mm256_packs_epi16(i0, i2);
    i0 = _mm256_permutevar8x32_epi32(i0, pack_order);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output[block].qs), i0);
  }
}

void q8_0_dequantize_avx2(
  const Q8_0Block * input,
  float * output,
  const std::size_t block_count) noexcept {
  for (std::size_t block = 0; block < block_count; ++block) {
    const __m256 scale = _mm256_set1_ps(_cvtsh_ss(input[block].d));
    float * y = output + block * q8_0_values_per_block;
    for (std::size_t group = 0; group < 4; ++group) {
      const __m128i bytes = _mm_loadl_epi64(
        reinterpret_cast<const __m128i *>(input[block].qs + group * 8));
      const __m256i integers = _mm256_cvtepi8_epi32(bytes);
      const __m256 values = _mm256_mul_ps(_mm256_cvtepi32_ps(integers), scale);
      _mm256_storeu_ps(y + group * 8, values);
    }
  }
}

float q8_0_dot_avx2(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  const std::size_t block_count) noexcept {
  return dot_avx2_impl(lhs, rhs, block_count);
}

void q8_0_matvec_avx2(
  const Q8_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row) noexcept {
  if (blocks_per_row == 0) {
    for (std::size_t row = 0; row < row_count; ++row) {
      output[row] = 0.0F;
    }
    return;
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    output[row] = dot_avx2_impl(
      matrix + row * blocks_per_row,
      vector,
      blocks_per_row);
  }
}

} // namespace qwen35x::cpu::detail
