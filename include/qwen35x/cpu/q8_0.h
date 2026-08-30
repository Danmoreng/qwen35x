#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace qwen35x::cpu {

inline constexpr std::size_t q8_0_values_per_block = 32;

// Binary-compatible with GGML's block_q8_0: one IEEE-754 binary16 scale
// followed by 32 signed quants. In particular, there is no tail padding.
struct Q8_0Block {
  std::uint16_t d;
  std::int8_t qs[q8_0_values_per_block];
};

static_assert(std::is_standard_layout_v<Q8_0Block>);
static_assert(offsetof(Q8_0Block, d) == 0);
static_assert(offsetof(Q8_0Block, qs) == sizeof(std::uint16_t));
static_assert(sizeof(Q8_0Block) == 34);

enum class Q8_0Backend : std::uint8_t {
  auto_select = 0,
  scalar = 1,
  avx2 = 2,
};

// Explicit AVX2 requests remain safe on unsupported CPUs and resolve to the
// scalar implementation. Call q8_0_backend_available() when a hard requirement
// (for example, an ISA-specific test) must be enforced by the caller.
[[nodiscard]] bool q8_0_backend_available(Q8_0Backend backend) noexcept;
[[nodiscard]] Q8_0Backend q8_0_resolve_backend(Q8_0Backend requested) noexcept;
[[nodiscard]] const char * q8_0_backend_name(Q8_0Backend backend) noexcept;

// Each operation works on complete 32-value Q8_0 blocks. Source and destination
// ranges must not overlap. A block_count of zero is valid.
void q8_0_quantize(
  const float * input,
  Q8_0Block * output,
  std::size_t block_count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

void q8_0_dequantize(
  const Q8_0Block * input,
  float * output,
  std::size_t block_count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

[[nodiscard]] float q8_0_dot(
  const Q8_0Block * lhs,
  const Q8_0Block * rhs,
  std::size_t block_count,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

// matrix is row-major with blocks_per_row consecutive Q8_0 blocks per row.
// vector contains blocks_per_row blocks and output contains row_count floats.
void q8_0_matvec(
  const Q8_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  std::size_t row_count,
  std::size_t blocks_per_row,
  Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

} // namespace qwen35x::cpu
