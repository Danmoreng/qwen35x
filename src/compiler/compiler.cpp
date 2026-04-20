#include "qwen35x/compiler/compiler.h"

namespace qwen35x {

CompilationPlan ModelCompiler::create_plan(const ModelProfile & profile, const RuntimeTarget & target) const {
  const std::string dense_dtype = target.sm_version >= 120 ? "bf16" : "fp16";
  const std::string layout = target.sm_version >= 120 ? "packed_blackwell_v1" : "packed_generic_v1";

  CompilationPlan plan;
  plan.packed_tensors = {
    {"attn_qkv", dense_dtype, layout},
    {"attn_out", dense_dtype, layout},
    {"mlp_up", dense_dtype, layout},
    {"mlp_down", dense_dtype, layout},
    {"conv_kernel", dense_dtype, layout},
    {"recurrent_proj", dense_dtype, layout},
  };

  plan.decode_ops.push_back("linear_attention_decode");
  plan.decode_ops.push_back("full_attention_decode_gqa");
  plan.decode_ops.push_back("sampler_top_p");

  if (profile.family == "qwen3.5") {
    plan.decode_ops.push_back("qwen35_family_fastpath_guard");
  }

  return plan;
}

} // namespace qwen35x

