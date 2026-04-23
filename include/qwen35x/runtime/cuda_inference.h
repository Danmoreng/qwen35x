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
  void * data_bf16 = nullptr;
  int rows = 0;
  int cols = 0;
};

struct CudaDeviceBufferF32 {
  void * data = nullptr;
  std::size_t count = 0;
};

struct CudaCapturedGraph {
  void * graph = nullptr;
  void * exec = nullptr;
  bool ready = false;
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

bool upload_matrix_bf16_shadow_from_f32(
  const std::vector<float> & host_data,
  int rows,
  int cols,
  CudaDeviceMatrixF32 & matrix,
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

void set_prefer_bf16_matvec(bool enabled);

bool gather_matrix_row_f32(
  const CudaDeviceMatrixF32 & matrix,
  int row_index,
  CudaDeviceBufferF32 & out,
  std::string & error_message);

bool gather_matrix_row_f32_from_token_f32(
  const CudaDeviceMatrixF32 & matrix,
  const CudaDeviceBufferF32 & token_id,
  CudaDeviceBufferF32 & out,
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

bool run_add_f32(
  const CudaDeviceBufferF32 & a,
  const CudaDeviceBufferF32 & b,
  std::size_t count,
  CudaDeviceBufferF32 & out,
  std::string & error_message);

bool run_rms_norm_f32(
  const CudaDeviceBufferF32 & input,
  const CudaDeviceBufferF32 & weight,
  std::size_t count,
  float eps,
  CudaDeviceBufferF32 & out,
  std::string & error_message);

bool run_split_q_gate_f32(
  const CudaDeviceBufferF32 & q_gate_packed,
  int n_heads,
  int head_dim,
  CudaDeviceBufferF32 & out_q,
  CudaDeviceBufferF32 & out_gate,
  std::string & error_message);

bool run_rms_norm_per_head_f32(
  const CudaDeviceBufferF32 & input,
  const CudaDeviceBufferF32 & weight,
  int n_heads,
  int head_dim,
  float eps,
  CudaDeviceBufferF32 & out,
  std::string & error_message);

bool run_apply_rope_inplace_f32(
  const CudaDeviceBufferF32 & values,
  int n_heads,
  int head_dim,
  int rope_dim,
  int position,
  float rope_theta,
  std::string & error_message);

bool run_prepare_full_attention_qkv_f32(
  const CudaDeviceBufferF32 & q_gate_packed,
  const CudaDeviceBufferF32 & k_raw,
  const CudaDeviceBufferF32 & v_raw,
  const CudaDeviceBufferF32 & q_norm_weight,
  const CudaDeviceBufferF32 & k_norm_weight,
  int n_heads,
  int n_kv_heads,
  int head_dim,
  int rope_dim,
  int position,
  float rope_theta,
  float eps,
  CudaDeviceBufferF32 & out_q,
  CudaDeviceBufferF32 & out_gate,
  CudaDeviceBufferF32 & k_cache,
  std::size_t k_cache_offset,
  CudaDeviceBufferF32 & v_cache,
  std::size_t v_cache_offset,
  std::string & error_message);

bool run_prepare_full_attention_qkv_prefill_chunk_f32(
  const CudaDeviceBufferF32 & q_gate_packed,
  const CudaDeviceBufferF32 & k_raw,
  const CudaDeviceBufferF32 & v_raw,
  const CudaDeviceBufferF32 & q_norm_weight,
  const CudaDeviceBufferF32 & k_norm_weight,
  int token_count,
  int n_heads,
  int n_kv_heads,
  int head_dim,
  int rope_dim,
  int position_base,
  float rope_theta,
  float eps,
  CudaDeviceBufferF32 & out_q,
  CudaDeviceBufferF32 & out_gate,
  CudaDeviceBufferF32 & k_cache,
  std::size_t k_cache_offset_base,
  CudaDeviceBufferF32 & v_cache,
  std::size_t v_cache_offset_base,
  std::string & error_message);

bool copy_buffer_f32(
  const CudaDeviceBufferF32 & src,
  std::size_t count,
  std::size_t src_offset,
  const CudaDeviceBufferF32 & dst,
  std::size_t dst_offset,
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

bool run_full_attention_prefill_gqa_chunk(
  const CudaDeviceBufferF32 & q,
  const CudaDeviceBufferF32 & gate,
  const CudaDeviceBufferF32 & k_cache,
  const CudaDeviceBufferF32 & v_cache,
  int token_count,
  int n_heads,
  int n_kv_heads,
  int head_dim,
  int position_base,
  CudaDeviceBufferF32 & out,
  std::string & error_message);

bool run_linear_attention_decode(
  const CudaDeviceBufferF32 & mixed_qkv,
  const CudaDeviceBufferF32 & z,
  const CudaDeviceBufferF32 & b,
  const CudaDeviceBufferF32 & a,
  const CudaDeviceMatrixF32 & conv1d,
  const CudaDeviceBufferF32 & norm,
  const CudaDeviceBufferF32 & dt_bias,
  const CudaDeviceBufferF32 & ssm_a,
  int linear_kernel,
  int linear_num_k_heads,
  int linear_num_v_heads,
  int linear_head_k_dim,
  int linear_head_v_dim,
  float rms_eps,
  CudaDeviceBufferF32 & conv_state,
  CudaDeviceBufferF32 & recurrent_state,
  CudaDeviceBufferF32 & scratch_conv_out,
  CudaDeviceBufferF32 & out_gated_norm,
  std::string & error_message);

bool run_linear_attention_prefill_chunk(
  const CudaDeviceBufferF32 & mixed_qkv,
  const CudaDeviceBufferF32 & z,
  const CudaDeviceBufferF32 & b,
  const CudaDeviceBufferF32 & a,
  const CudaDeviceMatrixF32 & conv1d,
  const CudaDeviceBufferF32 & norm,
  const CudaDeviceBufferF32 & dt_bias,
  const CudaDeviceBufferF32 & ssm_a,
  int token_count,
  int linear_kernel,
  int linear_num_k_heads,
  int linear_num_v_heads,
  int linear_head_k_dim,
  int linear_head_v_dim,
  float rms_eps,
  CudaDeviceBufferF32 & conv_state,
  CudaDeviceBufferF32 & recurrent_state,
  CudaDeviceBufferF32 & scratch_conv_out,
  CudaDeviceBufferF32 & out_gated_norm,
  std::string & error_message);

bool sample_token_from_logits_f32_device(
  const CudaDeviceBufferF32 & logits,
  const CudaDeviceBufferF32 & seen_token_mask,
  int vocab_size,
  float temperature,
  float top_p,
  int top_k,
  float repetition_penalty,
  float random_u01,
  const CudaDeviceBufferF32 & topk_values_scratch,
  const CudaDeviceBufferF32 & topk_indices_scratch,
  int & out_token,
  std::string & error_message);

bool sample_token_from_logits_f32_device_to_buffer(
  const CudaDeviceBufferF32 & logits,
  const CudaDeviceBufferF32 & seen_token_mask,
  int vocab_size,
  float temperature,
  float top_p,
  int top_k,
  float repetition_penalty,
  float random_u01,
  const CudaDeviceBufferF32 & out_token,
  std::string & error_message);

bool synchronize_stream(std::string & error_message);

bool begin_stream_capture(std::string & error_message);

bool end_stream_capture(
  CudaCapturedGraph & out_graph,
  std::string & error_message);

bool launch_captured_graph(
  const CudaCapturedGraph & graph,
  std::string & error_message);

void free_captured_graph(CudaCapturedGraph & graph);

void reset_transfer_stats();

void get_transfer_stats(CudaTransferStats & out_stats);

} // namespace qwen35x::cuda
