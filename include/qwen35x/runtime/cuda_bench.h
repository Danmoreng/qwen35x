#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwen35x::cuda {

bool run_bf16_matvec_benchmark(
  const std::vector<std::uint16_t> & weights,
  int rows,
  int cols,
  int warmup_iterations,
  int benchmark_iterations,
  double & avg_iteration_ms,
  std::string & error_message);

} // namespace qwen35x::cuda

