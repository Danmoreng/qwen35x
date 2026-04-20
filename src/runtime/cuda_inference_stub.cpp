#include "qwen35x/runtime/cuda_inference.h"

#if !QWEN35X_HAS_CUDA

namespace qwen35x::cuda {

bool upload_matrix_f32(
  const std::vector<float> &,
  int,
  int,
  CudaDeviceMatrixF32 &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

void free_matrix_f32(CudaDeviceMatrixF32 &) {
}

bool run_matvec_f32(
  const CudaDeviceMatrixF32 &,
  const std::vector<float> &,
  std::vector<float> &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool allocate_buffer_f32(
  std::size_t,
  CudaDeviceBufferF32 &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

void free_buffer_f32(CudaDeviceBufferF32 &) {
}

bool upload_to_buffer_f32(
  const float *,
  std::size_t,
  const CudaDeviceBufferF32 &,
  std::size_t,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

void reset_transfer_stats() {
}

void get_transfer_stats(CudaTransferStats & out_stats) {
  out_stats = CudaTransferStats{};
}

} // namespace qwen35x::cuda

#endif
