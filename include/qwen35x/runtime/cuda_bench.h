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

bool run_nvfp4_matvec_check(
  const std::vector<std::uint8_t> & packed_weights,
  const std::vector<std::uint8_t> & weight_scales_e4m3,
  float input_scale,
  float weight_scale_2,
  int rows,
  int cols,
  int sample_rows,
  double & max_abs_error,
  std::string & error_message);

bool run_nvfp4_cublaslt_probe(
  const std::vector<std::uint8_t> & packed_weights,
  const std::vector<std::uint8_t> & weight_scales_e4m3,
  float weight_scale_2,
  int rows,
  int cols,
  int sample_rows,
  double & max_abs_error,
  double & elapsed_ms,
  double & max_abs_expected,
  double & max_abs_actual,
  std::string & error_message);

} // namespace qwen35x::cuda
