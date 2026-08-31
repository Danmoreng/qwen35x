#include "q4_h128_internal.h"

#include "qwen35x/cpu/q4_h128.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu::detail {

void q4_h128_transform_block_avx2(
  const float * input,
  float * output,
  const std::size_t transform_block_index,
  const std::uint64_t sign_seed) noexcept {
  for (std::size_t vector = 0; vector < 16; ++vector) {
    const std::size_t value_base = vector * 8;
    const std::size_t word = value_base / 64;
    const std::size_t bit_base = value_base % 64;
    const std::uint64_t signs = q4_h128_sign_word(
      transform_block_index, word, sign_seed) >> bit_base;
    const __m256i sign_mask = _mm256_slli_epi32(
      _mm256_set_epi32(
        static_cast<int>((signs >> 7U) & 1U),
        static_cast<int>((signs >> 6U) & 1U),
        static_cast<int>((signs >> 5U) & 1U),
        static_cast<int>((signs >> 4U) & 1U),
        static_cast<int>((signs >> 3U) & 1U),
        static_cast<int>((signs >> 2U) & 1U),
        static_cast<int>((signs >> 1U) & 1U),
        static_cast<int>(signs & 1U)),
      31);
    const __m256 values = _mm256_loadu_ps(input + value_base);
    _mm256_storeu_ps(
      output + value_base,
      _mm256_xor_ps(values, _mm256_castsi256_ps(sign_mask)));
  }

  // The first three stages operate within eight-float vectors. Keeping their
  // scalar butterfly order preserves exact parity with the portable ABI.
  for (std::size_t stride = 1; stride < 8; stride *= 2) {
    for (std::size_t base = 0; base < q4_h128_transform_size; base += 2 * stride) {
      for (std::size_t lane = 0; lane < stride; ++lane) {
        const float lhs = output[base + lane];
        const float rhs = output[base + stride + lane];
        output[base + lane] = lhs + rhs;
        output[base + stride + lane] = lhs - rhs;
      }
    }
  }

  for (std::size_t stride = 8; stride < q4_h128_transform_size; stride *= 2) {
    for (std::size_t base = 0; base < q4_h128_transform_size; base += 2 * stride) {
      for (std::size_t lane = 0; lane < stride; lane += 8) {
        const __m256 lhs = _mm256_loadu_ps(output + base + lane);
        const __m256 rhs = _mm256_loadu_ps(output + base + stride + lane);
        _mm256_storeu_ps(output + base + lane, _mm256_add_ps(lhs, rhs));
        _mm256_storeu_ps(output + base + stride + lane, _mm256_sub_ps(lhs, rhs));
      }
    }
  }
  const __m256 scale = _mm256_set1_ps(0.08838834764831844055F);
  for (std::size_t index = 0; index < q4_h128_transform_size; index += 8) {
    _mm256_storeu_ps(
      output + index,
      _mm256_mul_ps(_mm256_loadu_ps(output + index), scale));
  }
}

} // namespace qwen35x::cpu::detail
