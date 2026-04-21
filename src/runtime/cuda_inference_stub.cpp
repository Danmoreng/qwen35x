#include "qwen35x/runtime/cuda_inference.h"

#if !QWEN35X_HAS_CUDA

namespace qwen35x::cuda {

bool begin_inference_session(
  std::size_t,
  std::size_t,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

void end_inference_session() {
}

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

bool run_matvec_f32_device(
  const CudaDeviceMatrixF32 &,
  const CudaDeviceBufferF32 &,
  CudaDeviceBufferF32 &,
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

bool download_from_buffer_f32(
  const CudaDeviceBufferF32 &,
  std::size_t,
  std::size_t,
  std::vector<float> &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_silu_mul_f32(
  const CudaDeviceBufferF32 &,
  const CudaDeviceBufferF32 &,
  std::size_t,
  CudaDeviceBufferF32 &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_full_attention_decode_gqa(
  const CudaDeviceBufferF32 &,
  const CudaDeviceBufferF32 &,
  const CudaDeviceBufferF32 &,
  const CudaDeviceBufferF32 &,
  int,
  int,
  int,
  int,
  CudaDeviceBufferF32 &,
  CudaDeviceBufferF32 &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool run_linear_attention_decode(
  const CudaDeviceBufferF32 &,
  const CudaDeviceBufferF32 &,
  const CudaDeviceBufferF32 &,
  const CudaDeviceBufferF32 &,
  const CudaDeviceMatrixF32 &,
  const CudaDeviceBufferF32 &,
  const CudaDeviceBufferF32 &,
  const CudaDeviceBufferF32 &,
  int,
  int,
  int,
  int,
  int,
  float,
  CudaDeviceBufferF32 &,
  CudaDeviceBufferF32 &,
  CudaDeviceBufferF32 &,
  CudaDeviceBufferF32 &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool sample_token_from_logits_f32_device(
  const CudaDeviceBufferF32 &,
  const CudaDeviceBufferF32 &,
  int,
  float,
  float,
  int,
  float,
  float,
  const CudaDeviceBufferF32 &,
  const CudaDeviceBufferF32 &,
  int &,
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
