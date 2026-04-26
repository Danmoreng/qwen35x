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

bool run_nvfp4_cublaslt_projection_device(
  const float *,
  const std::uint8_t *,
  const std::uint8_t *,
  float,
  int,
  int,
  std::uint8_t *,
  std::uint8_t *,
  float *,
  double *,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_nvfp4_sm120_projection_device(
  const float *,
  const std::uint32_t *,
  const std::uint32_t *,
  float,
  float,
  int,
  int,
  int,
  int,
  std::uint32_t *,
  std::uint32_t *,
  float *,
  double *,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_nvfp4_sm120_gate_up_silu_device(
  const float *,
  const std::uint32_t *,
  const std::uint32_t *,
  float,
  float,
  const std::uint32_t *,
  const std::uint32_t *,
  float,
  float,
  int,
  int,
  int,
  int,
  std::uint32_t *,
  std::uint32_t *,
  float *,
  float *,
  double *,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_nvfp4_sm120_mlp_device(
  const float *,
  const std::uint32_t *,
  const std::uint32_t *,
  float,
  float,
  const std::uint32_t *,
  const std::uint32_t *,
  float,
  float,
  const std::uint32_t *,
  const std::uint32_t *,
  float,
  float,
  int,
  int,
  int,
  int,
  int,
  int,
  int,
  int,
  std::uint32_t *,
  std::uint32_t *,
  float *,
  float *,
  float *,
  double *,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_nvfp4_sm120_mlp_residual_device(
  const float *,
  const std::uint32_t *,
  const std::uint32_t *,
  float,
  float,
  const std::uint32_t *,
  const std::uint32_t *,
  float,
  float,
  const std::uint32_t *,
  const std::uint32_t *,
  float,
  float,
  int,
  int,
  int,
  int,
  int,
  int,
  int,
  int,
  std::uint32_t *,
  std::uint32_t *,
  float *,
  float *,
  float *,
  const void *,
  void *,
  double *,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_nvfp4_custom_projection_benchmark(
  const std::vector<std::uint8_t> &,
  const std::vector<std::uint8_t> &,
  float,
  float,
  int,
  int,
  int,
  int,
  double &,
  double &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_nvfp4_cublaslt_gate_up_silu_device(
  const float *,
  const std::uint8_t *,
  const std::uint8_t *,
  float,
  const std::uint8_t *,
  const std::uint8_t *,
  float,
  int,
  int,
  std::uint8_t *,
  std::uint8_t *,
  float *,
  float *,
  double *,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_nvfp4_cublaslt_gate_up_benchmark(
  const std::vector<std::uint8_t> &,
  const std::vector<std::uint8_t> &,
  float,
  const std::vector<std::uint8_t> &,
  const std::vector<std::uint8_t> &,
  float,
  int,
  int,
  int,
  int,
  double &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

} // namespace qwen35x::cuda

#endif
