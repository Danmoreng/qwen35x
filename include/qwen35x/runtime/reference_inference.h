#pragma once

#include "qwen35x/common/model_profile.h"
#include "qwen35x/cpu/q8_0.h"
#include "qwen35x/runtime/qwen35x_cuda_backend.h"
#include "qwen35x/runtime/qwen35x_profile.h"

#include <cstddef>
#include <cstdint>
#include <memory>
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
  qwen35x_cuda = 1
};

enum class Qwen35xPrefillMode {
  replay = 0,
  batched = 1
};

using ReferenceLogitsCallback = bool (*)(
  void * context,
  std::size_t output_index,
  std::int32_t target_token,
  const float * logits,
  std::size_t logit_count,
  std::string & error_message);

// Reusable in-memory CPU state after an invariant prompt prefix. Keep one
// instance alive across inference calls that share the same model and prefix.
// A cache handle is not thread-safe and must not be accessed concurrently.
struct ReferenceInferenceOptions;
struct ReferenceInferenceResult;

class ReferenceCpuPrefixCache {
public:
  ReferenceCpuPrefixCache() = default;
  ReferenceCpuPrefixCache(const ReferenceCpuPrefixCache &) = delete;
  ReferenceCpuPrefixCache & operator=(const ReferenceCpuPrefixCache &) = delete;
  ReferenceCpuPrefixCache(ReferenceCpuPrefixCache &&) noexcept = default;
  ReferenceCpuPrefixCache & operator=(ReferenceCpuPrefixCache &&) noexcept = default;

  void clear() noexcept { implementation_.reset(); }
  [[nodiscard]] bool empty() const noexcept { return implementation_ == nullptr; }

private:
  std::shared_ptr<void> implementation_;

  friend bool run_reference_qwen35_inference(
    const ModelProfile &,
    const ReferenceInferenceOptions &,
    ReferenceInferenceResult &,
    std::string &);
};

// Persistent CPU model weights and executor for repeated inference requests.
// Concurrent callers are safe, but serialize on one session; use one session
// per worker when parallel request execution is required.
class ReferenceCpuModelSession {
public:
  ReferenceCpuModelSession();
  ReferenceCpuModelSession(const ReferenceCpuModelSession &) = delete;
  ReferenceCpuModelSession & operator=(const ReferenceCpuModelSession &) = delete;
  ReferenceCpuModelSession(ReferenceCpuModelSession &&) noexcept = default;
  ReferenceCpuModelSession & operator=(ReferenceCpuModelSession &&) noexcept = default;

  void clear();
  [[nodiscard]] bool empty() const;

private:
  std::shared_ptr<void> implementation_;

  friend bool prepare_reference_cpu_model_session(
    const ModelProfile &,
    const ReferenceInferenceOptions &,
    ReferenceCpuModelSession &,
    std::string &);
  friend bool run_reference_qwen35_inference(
    const ModelProfile &,
    const ReferenceInferenceOptions &,
    ReferenceInferenceResult &,
    std::string &);
};

struct ReferenceInferenceOptions {
  std::string model_dir;
  std::string cpu_gguf_path;
  std::vector<std::int32_t> prompt_tokens;
  int max_new_tokens = 1;
  int max_context = 4096;
  bool use_cuda = false;
  bool use_cuda_matvec_bf16 = false;
  GpuDecodeBackend gpu_decode_backend = GpuDecodeBackend::runtime_default;
  Qwen35xPrefillMode qwen35x_prefill_mode = Qwen35xPrefillMode::batched;
  cuda_backend::Qwen35xWeightPrecision qwen35x_weight_precision = cuda_backend::Qwen35xWeightPrecision::bf16;
  cuda_backend::Qwen35xCachePrecision qwen35x_cache_precision = cuda_backend::Qwen35xCachePrecision::bf16;
  int gpu_decode_blocks = 0;
  bool profile_cuda_sync = false;
  bool profile_qwen35x = false;
  bool prefill_only = false;
  int cpu_threads = 0;
  cpu::Q8_0Backend cpu_q8_backend = cpu::Q8_0Backend::auto_select;
  ReferenceCpuModelSession * cpu_model_session = nullptr;
  ReferenceCpuPrefixCache * cpu_prefix_cache = nullptr;
  std::size_t cpu_prefix_token_count = 0;
  std::vector<std::int32_t> forced_output_tokens;
  ReferenceLogitsCallback logits_callback = nullptr;
  void * logits_callback_context = nullptr;
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
  bool cpu_model_session_hit = false;
  int cached_prefix_tokens = 0;
  double prefix_cache_restore_time_ms = 0.0;
  std::size_t prefix_cache_bytes = 0;
  double host_to_device_bytes_per_forward_token = 0.0;
  double device_to_host_bytes_per_forward_token = 0.0;
  ReferenceTimingBreakdown timing_breakdown;
  ReferenceTransferBreakdown transfer_breakdown;
  cuda_backend::Qwen35xRuntimeProfile qwen35x_profile;
};

bool parse_token_list_csv(
  const std::string & csv,
  std::vector<std::int32_t> & out_tokens,
  std::string & error_message);

bool prepare_reference_cpu_model_session(
  const ModelProfile & profile,
  const ReferenceInferenceOptions & options,
  ReferenceCpuModelSession & session,
  std::string & error_message);

bool run_reference_qwen35_inference(
  const ModelProfile & profile,
  const ReferenceInferenceOptions & options,
  ReferenceInferenceResult & result,
  std::string & error_message);

} // namespace qwen35x
