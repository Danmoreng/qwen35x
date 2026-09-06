#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include <cstring>
#include <iostream>
using namespace qwen35x;
namespace {
bool capture(void *ctx, std::size_t, std::int32_t, const float *v,
             std::size_t n, std::string &) {
  auto &out = *static_cast<std::vector<float> *>(ctx);
  out.insert(out.end(), v, v + n);
  return true;
}
bool equal(const std::vector<float> &a, const std::vector<float> &b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: cpu_engine_model_test <HF directory> <Q4 artifact>\n";
    return 2;
  }
  std::string error;
  auto profile = ProfileLoader::load_from_hf_directory(argv[1], error);
  if (!profile) {
    std::cerr << error;
    return 1;
  }
  ReferenceInferenceOptions options;
  options.model_dir = argv[1];
  options.cpu_q4_h128_path = argv[2];
  options.cpu_threads = 8;
  options.max_context = 512;
  options.max_new_tokens = 4;
  options.sampling.seed = 123;
  auto model = CpuModel::load(*profile, options, error);
  if (!model) {
    std::cerr << error;
    return 1;
  }
  auto engine = CpuEngine::create(
      model, {.threads = 8, .max_resident_requests = 3, .max_context = 512},
      error);
  if (!engine) {
    std::cerr << error;
    return 1;
  }
  for (bool forced : {true, false}) {
    CpuRequestId ids[2];
    std::vector<float> reference_logits[2], actual_logits[2];
    std::vector<std::int32_t> reference_tokens[2];
    for (int r = 0; r < 2; ++r) {
      options.prompt_tokens.assign(r ? 257 : 65, r + 1);
      options.forced_output_tokens =
          forced ? std::vector<std::int32_t>{19, 13, 198, 18 + r}
                 : std::vector<std::int32_t>{};
      options.logits_callback = capture;
      options.logits_callback_context = &reference_logits[r];
      ReferenceInferenceResult result;
      if (!run_reference_qwen35_inference(*profile, options, result, error)) {
        std::cerr << error;
        return 1;
      }
      reference_tokens[r] = result.generated_tokens;
      CpuRequestSpec spec;
      spec.prompt_tokens = options.prompt_tokens;
      spec.forced_output_tokens = options.forced_output_tokens;
      spec.max_new_tokens = 4;
      spec.sampling = options.sampling;
      spec.logits_callback = capture;
      spec.logits_callback_context = &actual_logits[r];
      ids[r] = engine->submit(std::move(spec), error);
      if (!ids[r]) {
        std::cerr << error;
        return 1;
      }
    }
    bool done = false;
    for (int tick = 0; tick < 32 && !done; ++tick) {
      done = true;
      for (int r = 1; r >= 0; --r) {
        CpuRequestResult state;
        engine->result(ids[r], state);
        if (state.status != CpuRequestStatus::complete) {
          done = false;
          if (!engine->advance(ids[r], error)) {
            std::cerr << error;
            return 1;
          }
        }
      }
    }
    if (!done)
      return 3;
    for (int r = 0; r < 2; ++r) {
      CpuRequestResult state;
      engine->result(ids[r], state);
      if (!equal(reference_logits[r], actual_logits[r]) ||
          state.output_tokens != reference_tokens[r] ||
          state.committed_tokens != (r ? 257U : 65U) + 3 ||
          state.decode_forwards != 3) {
        std::cerr << "Incremental/interleaved mismatch: " << r
                  << " forced=" << forced << '\n';
        return 4;
      }
      engine->release(ids[r]);
    }
  }
  CpuRequestSpec spec;
  spec.prompt_tokens = {1};
  spec.max_new_tokens = 1;
  spec.forced_output_tokens = {19};
  spec.stop_token_ids = {19};
  auto a = engine->submit(spec, error), b = engine->submit(spec, error),
       c = engine->submit(spec, error);
  if (!a || !b || !c || engine->submit(spec, error) != 0)
    return 5;
  engine->cancel(a);
  engine->advance(a, error);
  CpuRequestResult state;
  engine->result(a, state);
  if (state.status != CpuRequestStatus::cancelled)
    return 6;
  engine->release(a);
  auto d = engine->submit(spec, error);
  if (d <= c)
    return 7;
  if (!engine->advance(b, error))
    return 8;
  engine->result(b, state);
  if (state.status != CpuRequestStatus::complete ||
      !state.output_tokens.empty() || state.committed_tokens != 1)
    return 9;
  auto second = CpuEngine::create(
      model, {.threads = 2, .max_resident_requests = 1, .max_context = 512},
      error);
  auto e = second->submit(spec, error);
  if (!e || !second->advance(e, error))
    return 10;
  std::cout << "Incremental requests, shared immutable model, interleaved "
               "states/RNG, cancellation and limits passed\n";
}
