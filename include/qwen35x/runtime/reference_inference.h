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

struct ReferenceInferenceOptions {
  std::string model_dir;
  std::vector<std::int32_t> prompt_tokens;
  int max_new_tokens = 1;
  int max_context = 4096;
  bool use_cuda = false;
  SamplingOptions sampling;
  std::vector<std::int32_t> stop_token_ids;
  std::vector<std::vector<std::int32_t>> stop_token_sequences;
};

struct ReferenceInferenceResult {
  std::vector<std::int32_t> generated_tokens;
  double load_time_ms = 0.0;
  double decode_time_ms = 0.0;
  double tokens_per_second = 0.0;
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
