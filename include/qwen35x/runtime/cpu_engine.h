#pragma once

#include "qwen35x/runtime/reference_inference.h"
#include <span>

namespace qwen35x {

// Immutable CPU weights shared by engines. Loading does not create a thread
// pool.
class CpuModel final {
public:
  static std::shared_ptr<const CpuModel>
  load(const ModelProfile &profile, const ReferenceInferenceOptions &options,
       std::string &error);
  ~CpuModel();

private:
  struct Impl;
  explicit CpuModel(std::shared_ptr<const Impl> impl);
  std::shared_ptr<const Impl> impl_;
  friend class CpuEngine;
};

using CpuRequestId = std::uint64_t;
enum class CpuRequestStatus { prefill, decode, complete, cancelled, failed };
struct CpuEngineConfig {
  int threads = 8;
  std::size_t max_resident_requests = 32;
  std::size_t max_context = 8192;
  std::size_t prefill_chunk_size = 0;
  // Shared vocabulary-tile reduction for all-greedy batches. False preserves
  // the full-logits reference. General sampling/callbacks always use logits.
  bool batch_greedy = true;
  // Dispatch private state rows jointly when B >= executor thread count.
  bool parallel_state_batches = true;
};
struct CpuRequestSpec {
  std::vector<std::int32_t> prompt_tokens;
  std::vector<std::int32_t> forced_output_tokens;
  std::size_t max_new_tokens = 128;
  SamplingOptions sampling;
  std::vector<std::int32_t> stop_token_ids;
  std::vector<std::vector<std::int32_t>> stop_token_sequences;
  // Invoked synchronously, with raw logits before penalties/sampling. It must
  // not reenter this engine. The callback context must outlive the request.
  ReferenceLogitsCallback logits_callback = nullptr;
  void *logits_callback_context = nullptr;
};
struct CpuRequestResult {
  CpuRequestStatus status = CpuRequestStatus::prefill;
  std::vector<std::int32_t> output_tokens;
  std::size_t committed_tokens = 0;
  std::size_t prefill_tokens = 0, decode_forwards = 0;
  // Forward-call time only; aggregate serving timing must also include sampling
  // and scheduling.
  double prefill_ms = 0, decode_ms = 0;
  std::string error;
};

struct CpuEngineStats {
  std::uint64_t batch_ticks = 0, batch_rows = 0;
  std::uint64_t projection_batches = 0, lm_head_batches = 0;
  std::uint64_t greedy_lm_head_batches = 0;
  std::uint64_t state_batches = 0;
};

// One owner calls these methods serially. Each engine owns one executor and
// scratch workspace; independent engines can share CpuModel read-only.
// Request IDs are monotonic within an engine and are never reused.
class CpuEngine final {
public:
  static std::unique_ptr<CpuEngine>
  create(std::shared_ptr<const CpuModel> model, const CpuEngineConfig &config,
         std::string &error);
  ~CpuEngine();
  CpuEngine(const CpuEngine &) = delete;
  CpuEngine &operator=(const CpuEngine &) = delete;
  CpuRequestId submit(CpuRequestSpec spec, std::string &error);
  // One bounded prefill chunk or one decode forward; no background work.
  bool advance(CpuRequestId id, std::string &error);
  // One decode row per distinct active request, in the supplied row order.
  // Prompts must be initialized first. Empty batches are a no-op; B=1 keeps
  // the existing fast path. Validation errors leave every request untouched.
  bool advance_batch(std::span<const CpuRequestId> ids, std::string &error);
  CpuEngineStats stats() const;
  bool cancel(CpuRequestId id);
  bool release(CpuRequestId id);
  bool result(CpuRequestId id, CpuRequestResult &out) const;

private:
  struct Impl;
  explicit CpuEngine(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace qwen35x
