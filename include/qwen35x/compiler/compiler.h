#pragma once

#include "qwen35x/common/model_profile.h"
#include "qwen35x/common/runtime_target.h"

#include <optional>
#include <string>
#include <vector>

namespace qwen35x {

struct PackedTensorSpec {
  std::string name;
  std::string dtype;
  std::string layout;
};

struct CompilationPlan {
  std::vector<PackedTensorSpec> packed_tensors;
  std::vector<std::string> decode_ops;
};

class ProfileLoader {
public:
  static std::optional<ModelProfile> load_from_json(const std::string & path, std::string & error_message);
  static std::optional<ModelProfile> load_from_hf_directory(const std::string & model_dir, std::string & error_message);
};

class ModelCompiler {
public:
  CompilationPlan create_plan(const ModelProfile & profile, const RuntimeTarget & target) const;
};

} // namespace qwen35x
