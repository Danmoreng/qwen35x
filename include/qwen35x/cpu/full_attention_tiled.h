#pragma once
#include "qwen35x/cpu/q8_0.h"
#include <cstddef>
#include <cstdint>

namespace qwen35x::cpu {
// One synchronous task owns one scratch region; no global converted KV mirror.
struct TiledAttention {
  const float *queries{}, *gates{}, *keys{}, *values{};
  const std::uint16_t *keys_f16{}, *values_f16{};
  float *output{};
  std::size_t tokens{}, position{}, query_stride{}, kv_stride{};
  int heads{}, kv_heads{}, dimension{};
  float scale{};
  int query_tile = 8, kv_tile = 64;
  bool share_gqa = false;
};
struct AttentionTileTimes {
  double pack_ms{}, qk_ms{}, softmax_ms{}, pv_ms{};
};
// Finite inputs with finite QK scores, D=256, divisible GQA head counts;
// query tiles 4/8/16, KV tiles 32/64/128, shared GQA groups at most four.
// Caller provides valid strides/cache capacity through position+tokens and
// scratch_floats elements. False rejects unsupported arguments/ISA untouched.
// Dispatch is independent of VNNI.
const char *tiled_attention_kernel(Q8_0Backend backend) noexcept;
std::size_t tiled_attention_scratch_floats(const TiledAttention &a) noexcept;
std::size_t tiled_attention_tasks(const TiledAttention &a) noexcept;
bool causal_attention_tiled(const TiledAttention &a, std::size_t task, float *scratch,
                            Q8_0Backend backend, AttentionTileTimes *times = nullptr) noexcept;
} // namespace qwen35x::cpu
