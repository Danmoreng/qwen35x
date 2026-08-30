#pragma once

#include "qwen35x/cpu/q8_0.h"

#include <cstddef>

namespace qwen35x::cpu {

// Updates a contiguous range of value rows in a decode-time gated DeltaNet.
// State layout is [head_count, value_dim, key_dim]. q/k are
// [head_count, key_dim], while v/output are [head_count, value_dim].
// Rows are flattened as head * value_dim + value_row, which makes disjoint
// ranges safe to execute concurrently.
void gated_delta_net_update_rows(
  float * state,
  const float * q,
  const float * k,
  const float * v,
  const float * alpha,
  const float * beta,
  float * output,
  std::size_t head_count,
  std::size_t key_dim,
  std::size_t value_dim,
  std::size_t row_begin,
  std::size_t row_end,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

} // namespace qwen35x::cpu
