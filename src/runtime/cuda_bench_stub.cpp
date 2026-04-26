#include "qwen35x/runtime/cuda_bench.h"

#if !QWEN35X_HAS_CUDA

namespace qwen35x::cuda {

bool run_bf16_matvec_benchmark(
  const std::vector<std::uint16_t> &,
  int,
  int,
  int,
  int,
  double &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_nvfp4_matvec_check(
  const std::vector<std::uint8_t> &,
  const std::vector<std::uint8_t> &,
  float,
  float,
  int,
  int,
  int,
  double &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_nvfp4_cublaslt_probe(
  const std::vector<std::uint8_t> &,
  const std::vector<std::uint8_t> &,
  float,
  int,
  int,
  int,
  double &,
  double &,
  double &,
  double &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

} // namespace qwen35x::cuda

#endif
