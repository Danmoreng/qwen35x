#include "qwen35x/cpu/delta_net.h"

#include <immintrin.h>

#include <cstddef>

namespace qwen35x::cpu::detail {

namespace {

[[nodiscard]] float horizontal_sum_f32(const __m256 value) noexcept {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
  return _mm_cvtss_f32(sum);
}

} // namespace

void gated_delta_net_update_rows_avx2(
  float * state,
  const float * q,
  const float * k,
  const float * v,
  const float * alpha,
  const float * beta,
  float * output,
  const std::size_t key_dim,
  const std::size_t value_dim,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  for (std::size_t flat_row = row_begin; flat_row < row_end; ++flat_row) {
    const std::size_t head = flat_row / value_dim;
    const std::size_t value_row = flat_row - head * value_dim;
    float * state_row = state + flat_row * key_dim;
    const float * q_head = q + head * key_dim;
    const float * k_head = k + head * key_dim;
    const __m256 decay = _mm256_set1_ps(alpha[head]);

    __m256 state_dot_k_vector = _mm256_setzero_ps();
    std::size_t column = 0;
    for (; column + 8 <= key_dim; column += 8) {
      const __m256 decayed = _mm256_mul_ps(_mm256_loadu_ps(state_row + column), decay);
      _mm256_storeu_ps(state_row + column, decayed);
      state_dot_k_vector = _mm256_fmadd_ps(
        decayed, _mm256_loadu_ps(k_head + column), state_dot_k_vector);
    }
    float state_dot_k = horizontal_sum_f32(state_dot_k_vector);
    for (; column < key_dim; ++column) {
      const float decayed = state_row[column] * alpha[head];
      state_row[column] = decayed;
      state_dot_k += decayed * k_head[column];
    }

    const float delta_scalar =
      (v[head * value_dim + value_row] - state_dot_k) * beta[head];
    const __m256 delta = _mm256_set1_ps(delta_scalar);
    __m256 state_dot_q_vector = _mm256_setzero_ps();
    column = 0;
    for (; column + 8 <= key_dim; column += 8) {
      const __m256 updated = _mm256_fmadd_ps(
        delta, _mm256_loadu_ps(k_head + column), _mm256_loadu_ps(state_row + column));
      _mm256_storeu_ps(state_row + column, updated);
      state_dot_q_vector = _mm256_fmadd_ps(
        updated, _mm256_loadu_ps(q_head + column), state_dot_q_vector);
    }
    float state_dot_q = horizontal_sum_f32(state_dot_q_vector);
    for (; column < key_dim; ++column) {
      const float updated = state_row[column] + delta_scalar * k_head[column];
      state_row[column] = updated;
      state_dot_q += updated * q_head[column];
    }
    output[head * value_dim + value_row] = state_dot_q;
  }
}

void gated_delta_net_update_batch_rows_avx2(
  float * state,
  const float * q,
  const float * k,
  const float * v,
  const float * alpha,
  const float * beta,
  float * output,
  const std::size_t batch_size,
  const std::size_t head_count,
  const std::size_t key_dim,
  const std::size_t value_dim,
  const std::size_t head_begin,
  const std::size_t head_end) noexcept {
  const std::size_t state_rows = head_count * value_dim;
  for (std::size_t head = head_begin; head < head_end; ++head) {
    for (std::size_t token = 0; token < batch_size; ++token) {
      const std::size_t head_token = head * batch_size + token;
      const float * q_head = q + head_token * key_dim;
      const float * k_head = k + head_token * key_dim;
      const __m256 decay = _mm256_set1_ps(alpha[head_token]);
      for (std::size_t value_row = 0; value_row < value_dim; ++value_row) {
        const std::size_t flat_row = head * value_dim + value_row;
        float * state_row = state + flat_row * key_dim;
        __m256 state_dot_k_vector = _mm256_setzero_ps();
        std::size_t column = 0;
        for (; column + 8 <= key_dim; column += 8) {
          const __m256 decayed = _mm256_mul_ps(_mm256_loadu_ps(state_row + column), decay);
          _mm256_storeu_ps(state_row + column, decayed);
          state_dot_k_vector = _mm256_fmadd_ps(
            decayed, _mm256_loadu_ps(k_head + column), state_dot_k_vector);
        }
        float state_dot_k = horizontal_sum_f32(state_dot_k_vector);
        for (; column < key_dim; ++column) {
          const float decayed = state_row[column] * alpha[head_token];
          state_row[column] = decayed;
          state_dot_k += decayed * k_head[column];
        }
        const float delta_scalar =
          (v[head_token * value_dim + value_row] - state_dot_k) * beta[head_token];
        const __m256 delta = _mm256_set1_ps(delta_scalar);
        __m256 state_dot_q_vector = _mm256_setzero_ps();
        column = 0;
        for (; column + 8 <= key_dim; column += 8) {
          const __m256 updated = _mm256_fmadd_ps(
            delta, _mm256_loadu_ps(k_head + column), _mm256_loadu_ps(state_row + column));
          _mm256_storeu_ps(state_row + column, updated);
          state_dot_q_vector = _mm256_fmadd_ps(
            updated, _mm256_loadu_ps(q_head + column), state_dot_q_vector);
        }
        float state_dot_q = horizontal_sum_f32(state_dot_q_vector);
        for (; column < key_dim; ++column) {
          const float updated = state_row[column] + delta_scalar * k_head[column];
          state_row[column] = updated;
          state_dot_q += updated * q_head[column];
        }
        output[token * state_rows + flat_row] = state_dot_q;
      }
    }
  }
}

} // namespace qwen35x::cpu::detail
