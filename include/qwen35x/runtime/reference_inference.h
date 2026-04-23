#pragma once

#include "qwen35x/common/model_profile.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qwen35x {

struct SamplingOptions {
  float temperature = 0.7f;
  float top_p = 0.8f;
  int top_k = 20;
  float repetition_penalty = 1.05f;
  std::int64_t seed = -1;
};

enum class GpuDecodeBackend {
  runtime_default = 0,
  luce = 1
};

struct ReferenceInferenceOptions {
  std::string model_dir;
  std::vector<std::int32_t> prompt_tokens;
  int max_new_tokens = 1;
  int max_context = 4096;
  bool use_cuda = false;
  bool use_cuda_matvec_bf16 = false;
  GpuDecodeBackend gpu_decode_backend = GpuDecodeBackend::runtime_default;
  int gpu_decode_blocks = 0;
  bool profile_cuda_sync = false;
  SamplingOptions sampling;
  std::vector<std::int32_t> stop_token_ids;
  std::vector<std::vector<std::int32_t>> stop_token_sequences;
};

struct ReferenceTimingBreakdown {
  double embedding_ms = 0.0;
  double attention_ms = 0.0;
  double mlp_ms = 0.0;
  double logits_ms = 0.0;
  double sampling_ms = 0.0;
  double stop_checks_ms = 0.0;
};

struct ReferenceTransferBreakdown {
  std::uint64_t host_to_device_bytes = 0;
  std::uint64_t device_to_host_bytes = 0;
  std::uint64_t other_bytes = 0;
  std::uint64_t copy_calls = 0;
};

struct ReferenceInferenceResult {
  std::vector<std::int32_t> generated_tokens;
  double load_time_ms = 0.0;
  double prefill_time_ms = 0.0;
  double prefill_tokens_per_second = 0.0;
  double decode_time_ms = 0.0;
  double tokens_per_second = 0.0;
  int forward_pass_tokens = 0;
  double host_to_device_bytes_per_forward_token = 0.0;
  double device_to_host_bytes_per_forward_token = 0.0;
  ReferenceTimingBreakdown timing_breakdown;
  ReferenceTransferBreakdown transfer_breakdown;
};

bool parse_token_list_csv(
  const std::string & csv,
  std::vector<std::int32_t> & out_tokens,
  std::string & error_message);

bool run_reference_qwen35_inference(
  const ModelProfile & profile,
  const ReferenceInferenceOptions & options,
  ReferenceInferenceResult & result,
  std::string & error_message);

} // namespace qwen35x
