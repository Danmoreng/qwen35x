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

bool run_nvfp4_custom_projection_benchmark(
  const std::vector<std::uint8_t> & packed_weights,
  const std::vector<std::uint8_t> & weight_scales_e4m3,
  float input_scale,
  float weight_scale_2,
  int rows,
  int cols,
  int kernel_variant,
  int warmup_iterations,
  int benchmark_iterations,
  double & avg_iteration_ms,
  double & max_abs_error,
  std::string & error_message);

bool run_nvfp4_cublaslt_projection_device(
  const float * input_f32,
  const std::uint8_t * packed_weights_cublaslt,
  const std::uint8_t * weight_scales_tiled,
  float weight_scale_2,
  int rows,
  int cols,
  std::uint8_t * activation_scratch,
  std::uint8_t * activation_scale_scratch,
  float * output_f32,
  double * elapsed_ms,
  std::string & error_message);

bool run_nvfp4_sm120_projection_device(
  const float * input_f32,
  const std::uint32_t * packed_weight_fragments,
  const std::uint32_t * weight_scale_fragments,
  float input_scale,
  float weight_scale_2,
  int rows,
  int cols,
  int row_tiles,
  int k_blocks,
  std::uint32_t * activation_fragment_scratch,
  std::uint32_t * activation_scale_scratch,
  float * output_f32,
  double * elapsed_ms,
  std::string & error_message);

bool run_nvfp4_cublaslt_gate_up_silu_device(
  const float * input_f32,
  const std::uint8_t * gate_packed_weights_cublaslt,
  const std::uint8_t * gate_weight_scales_tiled,
  float gate_weight_scale_2,
  const std::uint8_t * up_packed_weights_cublaslt,
  const std::uint8_t * up_weight_scales_tiled,
  float up_weight_scale_2,
  int rows,
  int cols,
  std::uint8_t * activation_scratch,
  std::uint8_t * activation_scale_scratch,
  float * gate_output_f32,
  float * up_output_f32,
  double * elapsed_ms,
  std::string & error_message);

bool run_nvfp4_sm120_gate_up_silu_device(
  const float * input_f32,
  const std::uint32_t * gate_packed_weight_fragments,
  const std::uint32_t * gate_weight_scale_fragments,
  float gate_input_scale,
  float gate_weight_scale_2,
  const std::uint32_t * up_packed_weight_fragments,
  const std::uint32_t * up_weight_scale_fragments,
  float up_input_scale,
  float up_weight_scale_2,
  int rows,
  int cols,
  int row_tiles,
  int k_blocks,
  std::uint32_t * activation_fragment_scratch,
  std::uint32_t * activation_scale_scratch,
  float * gate_output_f32,
  float * up_output_f32,
  double * elapsed_ms,
  std::string & error_message);

bool run_nvfp4_sm120_mlp_device(
  const float * input_f32,
  const std::uint32_t * gate_packed_weight_fragments,
  const std::uint32_t * gate_weight_scale_fragments,
  float gate_input_scale,
  float gate_weight_scale_2,
  const std::uint32_t * up_packed_weight_fragments,
  const std::uint32_t * up_weight_scale_fragments,
  float up_input_scale,
  float up_weight_scale_2,
  const std::uint32_t * down_packed_weight_fragments,
  const std::uint32_t * down_weight_scale_fragments,
  float down_input_scale,
  float down_weight_scale_2,
  int intermediate_rows,
  int hidden_cols,
  int gate_up_row_tiles,
  int gate_up_k_blocks,
  int down_rows,
  int down_cols,
  int down_row_tiles,
  int down_k_blocks,
  std::uint32_t * activation_fragment_scratch,
  std::uint32_t * activation_scale_scratch,
  float * gate_silu_output_f32,
  float * up_output_f32,
  float * down_output_f32,
  double * elapsed_ms,
  std::string & error_message);

bool run_nvfp4_cublaslt_gate_up_benchmark(
  const std::vector<std::uint8_t> & gate_packed_weights,
  const std::vector<std::uint8_t> & gate_weight_scales_e4m3,
  float gate_weight_scale_2,
  const std::vector<std::uint8_t> & up_packed_weights,
  const std::vector<std::uint8_t> & up_weight_scales_e4m3,
  float up_weight_scale_2,
  int rows,
  int cols,
  int warmup_iterations,
  int benchmark_iterations,
  double & avg_iteration_ms,
  std::string & error_message);

} // namespace qwen35x::cuda
