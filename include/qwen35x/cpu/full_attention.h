#pragma once

#include "qwen35x/cpu/q8_0.h"

#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu {

void attention_cache_store_f16(
  const float * input,
  std::uint16_t * output,
  std::size_t count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// Computes a range of flattened [token, query_head] rows for causal GQA.
// Queries, gates, and output are token-major. K/V cache entries are
// [context, kv_head, head_dim]. Scores is caller-owned scratch with one
// context_stride row per flattened query row, which makes disjoint row ranges
// safe to execute concurrently.
void causal_attention_batch_rows(
  const float * queries,
  const float * gates,
  const float * k_cache,
  const float * v_cache,
  const std::uint16_t * k_cache_f16,
  const std::uint16_t * v_cache_f16,
  float * scores,
  float * output,
  std::size_t context_stride,
  std::size_t query_width,
  std::size_t kv_width,
  int position_start,
  int head_count,
  int kv_head_count,
  int head_dim,
  float attention_scale,
  std::size_t row_begin,
  std::size_t row_end,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

} // namespace qwen35x::cpu
