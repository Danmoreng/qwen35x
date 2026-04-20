#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qwen35x::cuda {

struct CudaTransferStats {
  std::uint64_t host_to_device_bytes = 0;
  std::uint64_t device_to_host_bytes = 0;
  std::uint64_t other_bytes = 0;
  std::uint64_t copy_calls = 0;
};

struct CudaDeviceMatrixF32 {
  void * data = nullptr;
  int rows = 0;
  int cols = 0;
};

struct CudaDeviceBufferF32 {
  void * data = nullptr;
  std::size_t count = 0;
};

bool begin_inference_session(
  std::size_t max_input_count,
  std::size_t max_output_count,
  std::string & error_message);

void end_inference_session();

bool upload_matrix_f32(
  const std::vector<float> & host_data,
  int rows,
  int cols,
  CudaDeviceMatrixF32 & out_matrix,
  std::string & error_message);

void free_matrix_f32(CudaDeviceMatrixF32 & matrix);

bool run_matvec_f32(
  const CudaDeviceMatrixF32 & matrix,
  const std::vector<float> & input,
  std::vector<float> & output,
  std::string & error_message);

bool run_matvec_f32_device(
  const CudaDeviceMatrixF32 & matrix,
  const CudaDeviceBufferF32 & input,
  CudaDeviceBufferF32 & output,
  std::string & error_message);

bool allocate_buffer_f32(
  std::size_t count,
  CudaDeviceBufferF32 & out_buffer,
  std::string & error_message);

void free_buffer_f32(CudaDeviceBufferF32 & buffer);

bool upload_to_buffer_f32(
  const float * host_data,
  std::size_t count,
  const CudaDeviceBufferF32 & buffer,
  std::size_t buffer_offset,
  std::string & error_message);

bool download_from_buffer_f32(
  const CudaDeviceBufferF32 & buffer,
  std::size_t count,
  std::size_t buffer_offset,
  std::vector<float> & out_data,
  std::string & error_message);

bool run_silu_mul_f32(
  const CudaDeviceBufferF32 & a,
  const CudaDeviceBufferF32 & b,
  std::size_t count,
  CudaDeviceBufferF32 & out,
  std::string & error_message);

bool run_full_attention_decode_gqa(
  const CudaDeviceBufferF32 & q,
  const CudaDeviceBufferF32 & gate,
  const CudaDeviceBufferF32 & k_cache,
  const CudaDeviceBufferF32 & v_cache,
  int n_heads,
  int n_kv_heads,
  int head_dim,
  int seq_len,
  CudaDeviceBufferF32 & out,
  CudaDeviceBufferF32 & scratch_scores,
  std::string & error_message);

void reset_transfer_stats();

void get_transfer_stats(CudaTransferStats & out_stats);

} // namespace qwen35x::cuda
