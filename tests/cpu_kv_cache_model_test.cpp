// Optional full-model regression: pass the HF directory and Q4_H128 artifact.
// Checks cached/uncached logits and precision changes on one persistent session.
#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/reference_inference.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {
bool collect_logits(void * context, std::size_t, std::int32_t,
                    const float * values, std::size_t count, std::string &) {
  auto & output = *static_cast<std::vector<float> *>(context);
  output.insert(output.end(), values, values + count);
  return true;
}
}

int main(int argc, char ** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "Usage: qwen35x_cpu_kv_cache_model_test <HF directory> <Q4 H128 artifact> [rows|tiled|auto]\n";
    return 2;
  }
  std::string error;
  const auto profile = qwen35x::ProfileLoader::load_from_hf_directory(argv[1], error);
  if (!profile) {
    std::cerr << error << '\n';
    return 1;
  }
  qwen35x::ReferenceCpuModelSession session;
  qwen35x::ReferenceCpuPrefixCache cache;
  qwen35x::ReferenceInferenceOptions options;
  options.model_dir = argv[1];
  options.cpu_q4_h128_path = argv[2];
  if(argc==4) options.cpu_attention=argv[3];
  options.cpu_attention_gqa=true;
  options.profile_cpu_prefill=true;
  options.cpu_threads = 12;
  options.cpu_model_session = &session;
  options.cpu_q8_backend = qwen35x::cpu::Q8_0Backend::avx512_vnni;
  options.prompt_tokens.assign(257, 1);
  options.forced_output_tokens = {19, 13, 198};
  options.max_new_tokens = 3;
  options.max_context = 512;
  options.sampling.temperature = 0;
  options.logits_callback = collect_logits;
  bool have_snapshot = false;
  bool snapshot_f16 = false;
  // The second visit to FP16 must invalidate the FP32 prefix snapshot too.
  for (const bool fp32 : {false, true, false}) {
    options.cpu_kv_cache_f32 = fp32;
    const bool expected_f16 = !fp32 &&
      qwen35x::cpu::q8_0_backend_uses_avx2(options.cpu_q8_backend);
    std::vector<float> reference;
    for (int run = 0; run < 3; ++run) {
      options.cpu_prefix_cache = run == 0 ? nullptr : &cache;
      options.cpu_prefix_token_count = run == 0 ? 0 : 256;
      std::vector<float> logits;
      options.logits_callback_context = &logits;
      qwen35x::ReferenceInferenceResult result;
      if (!qwen35x::run_reference_qwen35_inference(*profile, options, result, error)) {
        std::cerr << error << '\n';
        return 1;
      }
      bool actual_prefill=false;
      for(const auto &stage:result.cpu_prefill_stages) {
        if(stage.kind=="full" && stage.tokens>1) actual_prefill=true;
      }
      if(run==0 && !actual_prefill) {std::cerr<<"Test did not execute batched prefill\n";return 1;}
      const int expected_prefix = run == 2 ||
        (run == 1 && have_snapshot && snapshot_f16 == expected_f16) ? 256 : 0;
      if (result.cpu_kv_cache_f16 != expected_f16 ||
          result.cached_prefix_tokens != expected_prefix ||
          (run != 0 && !result.cpu_model_session_hit)) {
        std::cerr << "Wrong precision, session reuse or prefix-cache invalidation\n";
        return 1;
      }
      if (run == 0) {
        reference = logits;
      } else if (reference.size() != logits.size() ||
                 std::memcmp(reference.data(), logits.data(), logits.size() * sizeof(float)) != 0) {
        std::cerr << "Prefix caching changed full-vocabulary logits\n";
        return 1;
      }
    }
    have_snapshot = true;
    snapshot_f16 = expected_f16;
  }
  // The same session keeps weights, but a different arithmetic policy must
  // invalidate its prefix state, then permit replay of the new snapshot.
  options.cpu_attention=options.cpu_attention=="rows" ? "auto" : "rows";
  for(int run=0;run<2;++run) {
    std::vector<float> logits;
    options.logits_callback_context=&logits;
    qwen35x::ReferenceInferenceResult result;
    if(!qwen35x::run_reference_qwen35_inference(*profile,options,result,error) ||
       result.cached_prefix_tokens!=(run==0 ? 0 : 256) || !result.cpu_model_session_hit) {
      std::cerr<<"Attention policy switch did not invalidate prefix state: "<<error<<'\n';return 1;
    }
  }
  std::cout << "FP16/FP32 cache replay and precision-switch tests passed\n";
  return 0;
}
