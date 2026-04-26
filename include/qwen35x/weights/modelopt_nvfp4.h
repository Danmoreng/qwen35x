#pragma once

#include "qwen35x/common/model_profile.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qwen35x {

struct ModelOptNvfp4ValidationOptions {
  std::string model_dir;
};

struct ModelOptNvfp4TensorReport {
  std::string base_name;
  std::vector<std::int64_t> source_shape;
  std::vector<std::int64_t> packed_shape;
  std::vector<std::int64_t> scale_shape;
};

struct ModelOptNvfp4ValidationResult {
  int group_size = 0;
  int quantized_tensors = 0;
  int packed_u8_tensors = 0;
  int fp8_scale_tensors = 0;
  int f32_input_scale_tensors = 0;
  int f32_weight_scale2_tensors = 0;
  std::vector<ModelOptNvfp4TensorReport> tensors;
};

bool validate_modelopt_nvfp4_checkpoint(
  const ModelProfile & profile,
  const ModelOptNvfp4ValidationOptions & options,
  ModelOptNvfp4ValidationResult & result,
  std::string & error_message);

} // namespace qwen35x
