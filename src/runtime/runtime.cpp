#include "qwen35x/runtime/runtime.h"
#include "qwen35x/runtime/cuda_bench.h"
#include "qwen35x/weights/safetensors.h"

#include <climits>
#include <cstdint>
#include <filesystem>
#include <fstream>

namespace qwen35x {

bool run_bf16_tensor_benchmark(
  const Bf16TensorBenchOptions & options,
  Bf16TensorBenchResult & result,
  std::string & error_message) {
  if (options.model_dir.empty()) {
    error_message = "model_dir is required for BF16 tensor benchmark.";
    return false;
  }
  if (options.tensor_name.empty()) {
    error_message = "tensor_name is required for BF16 tensor benchmark.";
    return false;
  }
  if (options.warmup_iterations < 0 || options.benchmark_iterations <= 0) {
    error_message = "warmup_iterations must be >= 0 and benchmark_iterations must be > 0.";
    return false;
  }

  std::string tensor_file;
  if (!SafetensorLoader::resolve_tensor_file(options.model_dir, options.tensor_name, tensor_file, error_message)) {
    return false;
  }

  SafetensorTensorInfo info;
  if (!SafetensorLoader::load_tensor_info(tensor_file, options.tensor_name, info, error_message)) {
    return false;
  }
  if (info.dtype != "BF16") {
    error_message = "Tensor '" + options.tensor_name + "' is " + info.dtype + " (expected BF16).";
    return false;
  }
  if (info.shape.size() != 2) {
    error_message = "Tensor '" + options.tensor_name + "' must be 2D for current benchmark path.";
    return false;
  }

  const std::int64_t rows64 = info.shape[0];
  const std::int64_t cols64 = info.shape[1];
  if (rows64 <= 0 || cols64 <= 0 || rows64 > static_cast<std::int64_t>(INT32_MAX) ||
      cols64 > static_cast<std::int64_t>(INT32_MAX)) {
    error_message = "Tensor shape is out of supported range for benchmark.";
    return false;
  }

  std::vector<std::uint16_t> weights;
  if (!SafetensorLoader::read_bf16_tensor(tensor_file, info, weights, error_message)) {
    return false;
  }

  const int rows = static_cast<int>(rows64);
  const int cols = static_cast<int>(cols64);
  double avg_ms = 0.0;
  if (!cuda::run_bf16_matvec_benchmark(
        weights,
        rows,
        cols,
        options.warmup_iterations,
        options.benchmark_iterations,
        avg_ms,
        error_message)) {
    return false;
  }

  const double iterations_per_second = 1000.0 / avg_ms;
  const double flops_per_iteration = 2.0 * static_cast<double>(rows) * static_cast<double>(cols);
  const double gflops = (flops_per_iteration * iterations_per_second) / 1.0e9;

  result.tensor_file = std::filesystem::path(tensor_file).string();
  result.tensor_name = options.tensor_name;
  result.dtype = info.dtype;
  result.shape = info.shape;
  result.avg_iteration_ms = avg_ms;
  result.matvec_per_second = iterations_per_second;
  result.gflops = gflops;
  return true;
}

namespace {

bool read_raw_tensor_bytes(
  const std::string & safetensors_file,
  const SafetensorTensorInfo & info,
  std::vector<std::uint8_t> & out_data,
  std::string & error_message) {
  const std::uint64_t byte_count = info.data_end - info.data_start;
  out_data.resize(static_cast<std::size_t>(byte_count));
  std::ifstream in(safetensors_file, std::ios::binary);
  if (!in) {
    error_message = "Could not open safetensors shard: " + safetensors_file;
    return false;
  }
  in.seekg(static_cast<std::streamoff>(info.data_start), std::ios::beg);
  if (!in) {
    error_message = "Failed to seek tensor '" + info.name + "' in shard.";
    return false;
  }
  in.read(reinterpret_cast<char *>(out_data.data()), static_cast<std::streamsize>(out_data.size()));
  if (!in) {
    error_message = "Failed to read tensor '" + info.name + "' from shard.";
    return false;
  }
  return true;
}

bool read_raw_tensor_from_model(
  const std::string & model_dir,
  const std::string & tensor_name,
  const std::string & expected_dtype,
  SafetensorTensorInfo & info,
  std::vector<std::uint8_t> & data,
  std::string & error_message) {
  std::string tensor_file;
  if (!SafetensorLoader::resolve_tensor_file(model_dir, tensor_name, tensor_file, error_message) ||
      !SafetensorLoader::load_tensor_info(tensor_file, tensor_name, info, error_message)) {
    return false;
  }
  if (info.dtype != expected_dtype) {
    error_message = "Tensor '" + tensor_name + "' has dtype " + info.dtype + " (expected " + expected_dtype + ").";
    return false;
  }
  return read_raw_tensor_bytes(tensor_file, info, data, error_message);
}

} // namespace

bool run_nvfp4_tensor_check(
  const Nvfp4TensorCheckOptions & options,
  Nvfp4TensorCheckResult & result,
  std::string & error_message) {
  if (options.model_dir.empty()) {
    error_message = "model_dir is required for NVFP4 tensor check.";
    return false;
  }
  if (options.tensor_base_name.empty()) {
    error_message = "tensor_base_name is required for NVFP4 tensor check.";
    return false;
  }
  if (options.sample_rows <= 0) {
    error_message = "sample_rows must be > 0.";
    return false;
  }

  SafetensorTensorInfo weight_info;
  SafetensorTensorInfo scale_info;
  std::vector<std::uint8_t> packed_weights;
  std::vector<std::uint8_t> weight_scales;
  if (!read_raw_tensor_from_model(options.model_dir, options.tensor_base_name + ".weight", "U8", weight_info, packed_weights, error_message) ||
      !read_raw_tensor_from_model(options.model_dir, options.tensor_base_name + ".weight_scale", "F8_E4M3", scale_info, weight_scales, error_message)) {
    return false;
  }
  SafetensorTensorF32 input_scale_tensor;
  SafetensorTensorF32 weight_scale2_tensor;
  if (!SafetensorLoader::read_tensor_f32(options.model_dir, options.tensor_base_name + ".input_scale", input_scale_tensor, error_message) ||
      !SafetensorLoader::read_tensor_f32(options.model_dir, options.tensor_base_name + ".weight_scale_2", weight_scale2_tensor, error_message)) {
    return false;
  }
  if (weight_info.shape.size() != 2 || scale_info.shape.size() != 2) {
    error_message = "NVFP4 packed weight and scale tensors must be 2D.";
    return false;
  }
  if (input_scale_tensor.data.size() != 1 || weight_scale2_tensor.data.size() != 1 ||
      !input_scale_tensor.shape.empty() || !weight_scale2_tensor.shape.empty()) {
    error_message = "NVFP4 input_scale and weight_scale_2 tensors must be scalar F32.";
    return false;
  }

  const int rows = static_cast<int>(weight_info.shape[0]);
  const int cols = static_cast<int>(weight_info.shape[1] * 2);
  const std::vector<std::int64_t> expected_scale_shape{weight_info.shape[0], cols / 16};
  if (scale_info.shape != expected_scale_shape) {
    error_message = "NVFP4 weight_scale shape does not match packed weight shape.";
    return false;
  }

  double max_abs_error = 0.0;
  if (!cuda::run_nvfp4_matvec_check(
        packed_weights,
        weight_scales,
        input_scale_tensor.data[0],
        weight_scale2_tensor.data[0],
        rows,
        cols,
        options.sample_rows,
        max_abs_error,
        error_message)) {
    return false;
  }

  result.tensor_base_name = options.tensor_base_name;
  result.packed_shape = weight_info.shape;
  result.scale_shape = scale_info.shape;
  result.source_shape = {weight_info.shape[0], weight_info.shape[1] * 2};
  result.max_abs_error = max_abs_error;
  return true;
}

bool EngineRuntime::initialize(const ModelProfile & profile, const RuntimeTarget & target, std::string & error_message) {
  if (profile.family != "qwen3.5") {
    error_message = "Only qwen3.5 family is supported by this scaffold fast path.";
    return false;
  }

  if (profile.fingerprint.num_key_value_heads <= 0 ||
      (profile.fingerprint.num_attention_heads % profile.fingerprint.num_key_value_heads) != 0) {
    error_message = "Invalid profile: expected GQA-compatible q/kv head ratio.";
    return false;
  }

  profile_ = profile;
  target_ = target;

  ModelCompiler compiler;
  plan_ = compiler.create_plan(profile_, target_);

  const std::string layout = target_.sm_version >= 120 ? "packed_blackwell_v1" : "packed_generic_v1";
  const std::string dtype = target_.sm_version >= 120 ? "bf16" : "fp16";

  if (target_.cuda_enabled) {
    registry_.register_kernel(
      {"linear_attention_decode", "decode", dtype, layout, target_.sm_version},
      "qwen35x_linear_decode_stub");
    registry_.register_kernel(
      {"full_attention_decode_gqa", "decode", dtype, layout, target_.sm_version},
      "qwen35x_full_decode_gqa_stub");
    registry_.register_kernel(
      {"sampler_top_p", "decode", "fp32", "scalar", target_.sm_version},
      "qwen35x_sampler_top_p_ref");
  } else {
    registry_.register_kernel(
      {"linear_attention_decode", "decode", "fp32", "reference", 0},
      "qwen35x_linear_decode_ref");
    registry_.register_kernel(
      {"full_attention_decode_gqa", "decode", "fp32", "reference", 0},
      "qwen35x_full_decode_gqa_ref");
    registry_.register_kernel(
      {"sampler_top_p", "decode", "fp32", "scalar", 0},
      "qwen35x_sampler_top_p_ref");
  }

  return true;
}

void EngineRuntime::print_dispatch_table(std::ostream & os) const {
  os << "dispatch table:\n";
  for (const auto & entry : registry_.table()) {
    const auto & key = entry.first;
    os << "  op=" << key.op
       << " mode=" << key.mode
       << " dtype=" << key.dtype
       << " layout=" << key.layout
       << " sm=" << key.sm
       << " -> " << entry.second << "\n";
  }

  os << "packed tensor plan:\n";
  for (const auto & tensor : plan_.packed_tensors) {
    os << "  " << tensor.name << " dtype=" << tensor.dtype << " layout=" << tensor.layout << "\n";
  }
}

} // namespace qwen35x
