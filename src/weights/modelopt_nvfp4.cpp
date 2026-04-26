#include "qwen35x/weights/modelopt_nvfp4.h"

#include "qwen35x/weights/safetensors.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>

namespace qwen35x {

namespace {

std::optional<std::string> read_text_file(const std::filesystem::path & path, std::string & error_message) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error_message = "Could not open file: " + path.string();
    return std::nullopt;
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

bool contains_text(const std::string & text, const std::string & needle) {
  return text.find(needle) != std::string::npos;
}

std::vector<std::string> expected_quantized_module_bases(const ModelProfile & profile) {
  std::vector<std::string> names;
  for (int layer = 0; layer < profile.text.num_hidden_layers; ++layer) {
    const std::string base = "model.language_model.layers." + std::to_string(layer) + ".";
    names.push_back(base + "mlp.gate_proj");
    names.push_back(base + "mlp.up_proj");
    names.push_back(base + "mlp.down_proj");

    const bool full_attention =
      layer < static_cast<int>(profile.fingerprint.attention_schedule.size()) &&
      profile.fingerprint.attention_schedule[static_cast<std::size_t>(layer)] == AttentionBlock::full;
    if (full_attention) {
      names.push_back(base + "self_attn.q_proj");
      names.push_back(base + "self_attn.k_proj");
      names.push_back(base + "self_attn.v_proj");
      names.push_back(base + "self_attn.o_proj");
    } else {
      names.push_back(base + "linear_attn.in_proj_qkv");
      names.push_back(base + "linear_attn.in_proj_z");
      names.push_back(base + "linear_attn.in_proj_b");
      names.push_back(base + "linear_attn.in_proj_a");
      names.push_back(base + "linear_attn.out_proj");
    }
  }
  return names;
}

bool same_shape(const std::vector<std::int64_t> & a, const std::vector<std::int64_t> & b) {
  return a == b;
}

std::string shape_string(const std::vector<std::int64_t> & shape) {
  if (shape.empty()) {
    return "[]";
  }
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << shape[i];
  }
  out << "]";
  return out.str();
}

bool load_info(
  const std::string & model_dir,
  const std::string & tensor_name,
  SafetensorTensorInfo & info,
  std::string & error_message) {
  std::string file;
  if (!SafetensorLoader::resolve_tensor_file(model_dir, tensor_name, file, error_message)) {
    return false;
  }
  return SafetensorLoader::load_tensor_info(file, tensor_name, info, error_message);
}

bool validate_tensor_family(
  const std::string & model_dir,
  const std::string & base_name,
  const int group_size,
  ModelOptNvfp4TensorReport & report,
  std::string & error_message) {
  SafetensorTensorInfo weight;
  SafetensorTensorInfo scale;
  SafetensorTensorInfo input_scale;
  SafetensorTensorInfo weight_scale_2;
  if (!load_info(model_dir, base_name + ".weight", weight, error_message) ||
      !load_info(model_dir, base_name + ".weight_scale", scale, error_message) ||
      !load_info(model_dir, base_name + ".input_scale", input_scale, error_message) ||
      !load_info(model_dir, base_name + ".weight_scale_2", weight_scale_2, error_message)) {
    return false;
  }

  if (weight.dtype != "U8") {
    error_message = "Tensor '" + weight.name + "' has dtype " + weight.dtype + " (expected U8).";
    return false;
  }
  if (scale.dtype != "F8_E4M3") {
    error_message = "Tensor '" + scale.name + "' has dtype " + scale.dtype + " (expected F8_E4M3).";
    return false;
  }
  if (input_scale.dtype != "F32" || !input_scale.shape.empty()) {
    error_message = "Tensor '" + input_scale.name + "' must be scalar F32.";
    return false;
  }
  if (weight_scale_2.dtype != "F32" || !weight_scale_2.shape.empty()) {
    error_message = "Tensor '" + weight_scale_2.name + "' must be scalar F32.";
    return false;
  }
  if (weight.shape.size() != 2 || scale.shape.size() != 2) {
    error_message = "NVFP4 tensors for '" + base_name + "' must be 2D.";
    return false;
  }
  if ((weight.shape[1] * 2) % group_size != 0) {
    error_message = "Packed tensor '" + weight.name + "' has an invalid column count for group_size.";
    return false;
  }

  const std::vector<std::int64_t> source_shape = {weight.shape[0], weight.shape[1] * 2};
  const std::vector<std::int64_t> expected_scale_shape = {source_shape[0], source_shape[1] / group_size};
  if (!same_shape(scale.shape, expected_scale_shape)) {
    error_message = "Tensor '" + scale.name + "' shape " + shape_string(scale.shape) +
                    " does not match expected " + shape_string(expected_scale_shape) + ".";
    return false;
  }

  report.base_name = base_name;
  report.source_shape = source_shape;
  report.packed_shape = weight.shape;
  report.scale_shape = scale.shape;
  return true;
}

} // namespace

bool validate_modelopt_nvfp4_checkpoint(
  const ModelProfile & profile,
  const ModelOptNvfp4ValidationOptions & options,
  ModelOptNvfp4ValidationResult & result,
  std::string & error_message) {
  if (options.model_dir.empty()) {
    error_message = "model_dir is required for ModelOpt NVFP4 validation.";
    return false;
  }

  const std::filesystem::path root(options.model_dir);
  const auto quant_config = read_text_file(root / "hf_quant_config.json", error_message);
  if (!quant_config) {
    return false;
  }
  if (!contains_text(*quant_config, "\"quant_algo\"") || !contains_text(*quant_config, "\"NVFP4\"")) {
    error_message = "hf_quant_config.json does not declare quant_algo=NVFP4.";
    return false;
  }
  if (!contains_text(*quant_config, "\"group_size\"") || !contains_text(*quant_config, "16")) {
    error_message = "hf_quant_config.json does not declare group_size=16.";
    return false;
  }
  if (!contains_text(*quant_config, "\"kv_cache_quant_algo\"") || !contains_text(*quant_config, "null")) {
    error_message = "hf_quant_config.json must leave kv_cache_quant_algo null for the current plan.";
    return false;
  }

  result = {};
  result.group_size = 16;
  const auto bases = expected_quantized_module_bases(profile);
  result.tensors.reserve(bases.size());
  for (const auto & base : bases) {
    ModelOptNvfp4TensorReport report;
    if (!validate_tensor_family(options.model_dir, base, result.group_size, report, error_message)) {
      return false;
    }
    result.tensors.push_back(std::move(report));
  }

  result.quantized_tensors = static_cast<int>(result.tensors.size());
  result.packed_u8_tensors = result.quantized_tensors;
  result.fp8_scale_tensors = result.quantized_tensors;
  result.f32_input_scale_tensors = result.quantized_tensors;
  result.f32_weight_scale2_tensors = result.quantized_tensors;
  return true;
}

} // namespace qwen35x
