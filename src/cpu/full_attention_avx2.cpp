#include "qwen35x/cpu/full_attention.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

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

// Cephes-derived exp approximation used by common SIMD math libraries. The
// range reduction keeps softmax inputs in a compact interval and the degree-5
// polynomial is accurate enough for FP32 attention while avoiding one libm
// call per score.
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

[[nodiscard]] float dot_f32_avx2(
  const float * lhs,
  const float * rhs,
  const int count) noexcept {
  __m256 accum0 = _mm256_setzero_ps();
  __m256 accum1 = _mm256_setzero_ps();
  __m256 accum2 = _mm256_setzero_ps();
  __m256 accum3 = _mm256_setzero_ps();
  int index = 0;
  for (; index + 32 <= count; index += 32) {
    accum0 = _mm256_fmadd_ps(
      _mm256_loadu_ps(lhs + index), _mm256_loadu_ps(rhs + index), accum0);
    accum1 = _mm256_fmadd_ps(
      _mm256_loadu_ps(lhs + index + 8), _mm256_loadu_ps(rhs + index + 8), accum1);
    accum2 = _mm256_fmadd_ps(
      _mm256_loadu_ps(lhs + index + 16), _mm256_loadu_ps(rhs + index + 16), accum2);
    accum3 = _mm256_fmadd_ps(
      _mm256_loadu_ps(lhs + index + 24), _mm256_loadu_ps(rhs + index + 24), accum3);
  }
  accum0 = _mm256_add_ps(accum0, accum1);
  accum2 = _mm256_add_ps(accum2, accum3);
  float result = horizontal_sum_f32(_mm256_add_ps(accum0, accum2));
  for (; index < count; ++index) {
    result += lhs[index] * rhs[index];
  }
  return result;
}

[[nodiscard]] float dot_f16_avx2(
  const float * lhs,
  const std::uint16_t * rhs,
  const int count) noexcept {
  __m256 accum0 = _mm256_setzero_ps();
  __m256 accum1 = _mm256_setzero_ps();
  int index = 0;
  for (; index + 16 <= count; index += 16) {
    const __m256 rhs0 = _mm256_cvtph_ps(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + index)));
    const __m256 rhs1 = _mm256_cvtph_ps(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + index + 8)));
    accum0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index), rhs0, accum0);
    accum1 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + index + 8), rhs1, accum1);
  }
  float result = horizontal_sum_f32(_mm256_add_ps(accum0, accum1));
  for (; index < count; ++index) {
    result += lhs[index] * _cvtsh_ss(rhs[index]);
  }
  return result;
}

void weighted_sum_f32_avx2(
  const float * v_cache,
  const std::uint16_t * v_cache_f16,
  const float * probabilities,
  float * output,
  const std::size_t kv_width,
  const std::size_t head_offset,
  const int sequence_length,
  const int head_dim) noexcept {
  int column_base = 0;
  for (; column_base + 64 <= head_dim; column_base += 64) {
    __m256 accum0 = _mm256_setzero_ps();
    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    __m256 accum3 = _mm256_setzero_ps();
    __m256 accum4 = _mm256_setzero_ps();
    __m256 accum5 = _mm256_setzero_ps();
    __m256 accum6 = _mm256_setzero_ps();
    __m256 accum7 = _mm256_setzero_ps();
    for (int context = 0; context < sequence_length; ++context) {
      const __m256 probability = _mm256_set1_ps(
        probabilities[static_cast<std::size_t>(context)]);
      const std::size_t offset = static_cast<std::size_t>(context) * kv_width +
        head_offset + static_cast<std::size_t>(column_base);
      if (v_cache_f16 != nullptr) {
        const std::uint16_t * value = v_cache_f16 + offset;
        accum0 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value))), accum0);
        accum1 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 8))), accum1);
        accum2 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 16))), accum2);
        accum3 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 24))), accum3);
        accum4 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 32))), accum4);
        accum5 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 40))), accum5);
        accum6 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 48))), accum6);
        accum7 = _mm256_fmadd_ps(probability, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(value + 56))), accum7);
      } else {
        const float * value = v_cache + offset;
        accum0 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value), accum0);
        accum1 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 8), accum1);
        accum2 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 16), accum2);
        accum3 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 24), accum3);
        accum4 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 32), accum4);
        accum5 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 40), accum5);
        accum6 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 48), accum6);
        accum7 = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + 56), accum7);
      }
    }
    float * destination = output + column_base;
    _mm256_storeu_ps(destination, accum0);
    _mm256_storeu_ps(destination + 8, accum1);
    _mm256_storeu_ps(destination + 16, accum2);
    _mm256_storeu_ps(destination + 24, accum3);
    _mm256_storeu_ps(destination + 32, accum4);
    _mm256_storeu_ps(destination + 40, accum5);
    _mm256_storeu_ps(destination + 48, accum6);
    _mm256_storeu_ps(destination + 56, accum7);
  }
  for (; column_base + 8 <= head_dim; column_base += 8) {
    __m256 accumulator = _mm256_setzero_ps();
    for (int context = 0; context < sequence_length; ++context) {
      const __m256 probability = _mm256_set1_ps(
        probabilities[static_cast<std::size_t>(context)]);
      const std::size_t offset = static_cast<std::size_t>(context) * kv_width +
        head_offset + static_cast<std::size_t>(column_base);
      const __m256 value = v_cache_f16 != nullptr
        ? _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(v_cache_f16 + offset)))
        : _mm256_loadu_ps(v_cache + offset);
      accumulator = _mm256_fmadd_ps(probability, value, accumulator);
    }
    _mm256_storeu_ps(output + column_base, accumulator);
  }
  for (; column_base < head_dim; ++column_base) {
    float accumulator = 0.0F;
    for (int context = 0; context < sequence_length; ++context) {
      const std::size_t offset = static_cast<std::size_t>(context) * kv_width + head_offset;
      const float value = v_cache_f16 != nullptr
        ? _cvtsh_ss(v_cache_f16[offset + static_cast<std::size_t>(column_base)])
        : v_cache[offset + static_cast<std::size_t>(column_base)];
      accumulator += probabilities[static_cast<std::size_t>(context)] * value;
    }
    output[column_base] = accumulator;
  }
}

[[nodiscard]] float stable_sigmoid(const float value) noexcept {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-value));
  }
  const float exp_value = std::exp(value);
  return exp_value / (1.0F + exp_value);
}

} // namespace

void attention_cache_store_f16_avx2(
  const float * input,
  std::uint16_t * output,
  const std::size_t count) noexcept {
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    _mm_storeu_si128(
      reinterpret_cast<__m128i *>(output + index),
      _mm256_cvtps_ph(_mm256_loadu_ps(input + index), _MM_FROUND_TO_NEAREST_INT));
  }
  for (; index < count; ++index) {
    output[index] = static_cast<std::uint16_t>(_cvtss_sh(input[index], _MM_FROUND_TO_NEAREST_INT));
  }
}

void causal_attention_batch_rows_avx2(
  const float * queries,
  const float * gates,
  const float * k_cache,
  const float * v_cache,
  const std::uint16_t * k_cache_f16,
  const std::uint16_t * v_cache_f16,
  float * scores,
  float * output,
  const std::size_t context_stride,
  const std::size_t query_width,
  const std::size_t kv_width,
  const int position_start,
  const int head_count,
  const int kv_head_count,
  const int head_dim,
  const float attention_scale,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  const int heads_per_kv = head_count / kv_head_count;
  for (std::size_t row = row_begin; row < row_end; ++row) {
    const std::size_t token = row / static_cast<std::size_t>(head_count);
    const int head = static_cast<int>(row % static_cast<std::size_t>(head_count));
    const int kv_head = head / heads_per_kv;
    const int sequence_length = position_start + static_cast<int>(token) + 1;
    const std::size_t head_offset =
      static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
    const std::size_t kv_head_offset =
      static_cast<std::size_t>(kv_head) * static_cast<std::size_t>(head_dim);
    const float * query = queries + token * query_width + head_offset;
    const float * gate = gates + token * query_width + head_offset;
    float * score_row = scores + row * context_stride;
    float max_score = -std::numeric_limits<float>::infinity();
    for (int context = 0; context < sequence_length; ++context) {
      const std::size_t cache_offset =
        static_cast<std::size_t>(context) * kv_width + kv_head_offset;
      const float score = (k_cache_f16 != nullptr
        ? dot_f16_avx2(query, k_cache_f16 + cache_offset, head_dim)
        : dot_f32_avx2(query, k_cache + cache_offset, head_dim)) * attention_scale;
      score_row[static_cast<std::size_t>(context)] = score;
      max_score = std::max(max_score, score);
    }
    __m256 denominator_vector = _mm256_setzero_ps();
    const __m256 maximum = _mm256_set1_ps(max_score);
    int context = 0;
    for (; context + 8 <= sequence_length; context += 8) {
      const __m256 probability = exp_f32_avx2(
        _mm256_sub_ps(_mm256_loadu_ps(score_row + context), maximum));
      _mm256_storeu_ps(score_row + context, probability);
      denominator_vector = _mm256_add_ps(denominator_vector, probability);
    }
    float denominator = horizontal_sum_f32(denominator_vector);
    for (; context < sequence_length; ++context) {
      const float probability = std::exp(
        score_row[static_cast<std::size_t>(context)] - max_score);
      score_row[static_cast<std::size_t>(context)] = probability;
      denominator += probability;
    }
    const float inverse_denominator = 1.0F / (denominator > 0.0F ? denominator : 1.0F);
    const __m256 inverse_denominator_vector = _mm256_set1_ps(inverse_denominator);
    context = 0;
    for (; context + 8 <= sequence_length; context += 8) {
      _mm256_storeu_ps(
        score_row + context,
        _mm256_mul_ps(
          _mm256_loadu_ps(score_row + context), inverse_denominator_vector));
    }
    for (; context < sequence_length; ++context) {
      score_row[static_cast<std::size_t>(context)] *= inverse_denominator;
    }
    float * output_head = output + token * query_width + head_offset;
    weighted_sum_f32_avx2(
      v_cache, v_cache_f16, score_row, output_head, kv_width, kv_head_offset,
      sequence_length, head_dim);
    int gate_column = 0;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0F);
    const __m256 sign_mask = _mm256_set1_ps(-0.0F);
    for (; gate_column + 8 <= head_dim; gate_column += 8) {
      const __m256 gate_value = _mm256_loadu_ps(gate + gate_column);
      const __m256 absolute = _mm256_andnot_ps(sign_mask, gate_value);
      const __m256 exponential = exp_f32_avx2(_mm256_sub_ps(zero, absolute));
      const __m256 denominator = _mm256_add_ps(one, exponential);
      const __m256 positive = _mm256_div_ps(one, denominator);
      const __m256 negative = _mm256_mul_ps(exponential, positive);
      const __m256 sigmoid = _mm256_blendv_ps(
        negative, positive, _mm256_cmp_ps(gate_value, zero, _CMP_GE_OQ));
      _mm256_storeu_ps(
        output_head + gate_column,
        _mm256_mul_ps(_mm256_loadu_ps(output_head + gate_column), sigmoid));
    }
    for (; gate_column < head_dim; ++gate_column) {
      output_head[static_cast<std::size_t>(gate_column)] *= stable_sigmoid(gate[gate_column]);
    }
  }
}

} // namespace qwen35x::cpu::detail
