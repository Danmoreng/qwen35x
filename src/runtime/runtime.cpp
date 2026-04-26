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

bool run_nvfp4_cublaslt_probe(
  const Nvfp4TensorCheckOptions & options,
  Nvfp4CublasLtProbeResult & result,
  std::string & error_message) {
  if (options.model_dir.empty()) {
    error_message = "model_dir is required for NVFP4 cuBLASLt probe.";
    return false;
  }
  if (options.tensor_base_name.empty()) {
    error_message = "tensor_base_name is required for NVFP4 cuBLASLt probe.";
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
  SafetensorTensorF32 weight_scale2_tensor;
  if (!SafetensorLoader::read_tensor_f32(options.model_dir, options.tensor_base_name + ".weight_scale_2", weight_scale2_tensor, error_message)) {
    return false;
  }
  if (weight_info.shape.size() != 2 || scale_info.shape.size() != 2 || weight_scale2_tensor.data.size() != 1) {
    error_message = "Invalid NVFP4 tensor family for cuBLASLt probe.";
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
  double elapsed_ms = 0.0;
  double max_abs_expected = 0.0;
  double max_abs_actual = 0.0;
  if (!cuda::run_nvfp4_cublaslt_probe(
        packed_weights,
        weight_scales,
        weight_scale2_tensor.data[0],
        rows,
        cols,
        options.sample_rows,
        max_abs_error,
        elapsed_ms,
        max_abs_expected,
        max_abs_actual,
        error_message)) {
    return false;
  }

  result.tensor_base_name = options.tensor_base_name;
  result.packed_shape = weight_info.shape;
  result.scale_shape = scale_info.shape;
  result.source_shape = {weight_info.shape[0], weight_info.shape[1] * 2};
  result.max_abs_error = max_abs_error;
  result.elapsed_ms = elapsed_ms;
  result.max_abs_expected = max_abs_expected;
  result.max_abs_actual = max_abs_actual;
  return true;
}

bool run_nvfp4_gate_up_benchmark(
  const Nvfp4GateUpBenchOptions & options,
  Nvfp4GateUpBenchResult & result,
  std::string & error_message) {
  if (options.model_dir.empty()) {
    error_message = "model_dir is required for NVFP4 gate/up benchmark.";
    return false;
  }
  if (options.gate_tensor_base_name.empty() || options.up_tensor_base_name.empty()) {
    error_message = "gate and up tensor base names are required for NVFP4 gate/up benchmark.";
    return false;
  }
  if (options.warmup_iterations < 0 || options.benchmark_iterations <= 0) {
    error_message = "warmup_iterations must be >= 0 and benchmark_iterations must be > 0.";
    return false;
  }

  SafetensorTensorInfo gate_weight_info;
  SafetensorTensorInfo gate_scale_info;
  SafetensorTensorInfo up_weight_info;
  SafetensorTensorInfo up_scale_info;
  std::vector<std::uint8_t> gate_packed_weights;
  std::vector<std::uint8_t> gate_weight_scales;
  std::vector<std::uint8_t> up_packed_weights;
  std::vector<std::uint8_t> up_weight_scales;
  if (!read_raw_tensor_from_model(options.model_dir, options.gate_tensor_base_name + ".weight", "U8", gate_weight_info, gate_packed_weights, error_message) ||
      !read_raw_tensor_from_model(options.model_dir, options.gate_tensor_base_name + ".weight_scale", "F8_E4M3", gate_scale_info, gate_weight_scales, error_message) ||
      !read_raw_tensor_from_model(options.model_dir, options.up_tensor_base_name + ".weight", "U8", up_weight_info, up_packed_weights, error_message) ||
      !read_raw_tensor_from_model(options.model_dir, options.up_tensor_base_name + ".weight_scale", "F8_E4M3", up_scale_info, up_weight_scales, error_message)) {
    return false;
  }

  SafetensorTensorF32 gate_weight_scale2_tensor;
  SafetensorTensorF32 up_weight_scale2_tensor;
  if (!SafetensorLoader::read_tensor_f32(options.model_dir, options.gate_tensor_base_name + ".weight_scale_2", gate_weight_scale2_tensor, error_message) ||
      !SafetensorLoader::read_tensor_f32(options.model_dir, options.up_tensor_base_name + ".weight_scale_2", up_weight_scale2_tensor, error_message)) {
    return false;
  }
  if (gate_weight_info.shape.size() != 2 || gate_scale_info.shape.size() != 2 ||
      up_weight_info.shape.size() != 2 || up_scale_info.shape.size() != 2 ||
      gate_weight_scale2_tensor.data.size() != 1 || up_weight_scale2_tensor.data.size() != 1) {
    error_message = "Invalid NVFP4 gate/up tensor family for benchmark.";
    return false;
  }
  if (gate_weight_info.shape != up_weight_info.shape) {
    error_message = "Gate and up packed weight shapes must match for benchmark.";
    return false;
  }

  const int rows = static_cast<int>(gate_weight_info.shape[0]);
  const int cols = static_cast<int>(gate_weight_info.shape[1] * 2);
  const std::vector<std::int64_t> expected_scale_shape{gate_weight_info.shape[0], cols / 16};
  if (gate_scale_info.shape != expected_scale_shape || up_scale_info.shape != expected_scale_shape) {
    error_message = "NVFP4 gate/up weight_scale shape does not match packed weight shape.";
    return false;
  }

  double avg_ms = 0.0;
  if (!cuda::run_nvfp4_cublaslt_gate_up_benchmark(
        gate_packed_weights,
        gate_weight_scales,
        gate_weight_scale2_tensor.data[0],
        up_packed_weights,
        up_weight_scales,
        up_weight_scale2_tensor.data[0],
        rows,
        cols,
        options.warmup_iterations,
        options.benchmark_iterations,
        avg_ms,
        error_message)) {
    return false;
  }

  result.gate_tensor_base_name = options.gate_tensor_base_name;
  result.up_tensor_base_name = options.up_tensor_base_name;
  result.source_shape = {gate_weight_info.shape[0], gate_weight_info.shape[1] * 2};
  result.avg_iteration_ms = avg_ms;
  result.iterations_per_second = 1000.0 / avg_ms;
  return true;
}

bool run_nvfp4_projection_benchmark(
  const Nvfp4ProjectionBenchOptions & options,
  Nvfp4ProjectionBenchResult & result,
  std::string & error_message) {
  if (options.model_dir.empty()) {
    error_message = "model_dir is required for custom NVFP4 projection benchmark.";
    return false;
  }
  if (options.tensor_base_name.empty()) {
    error_message = "tensor_base_name is required for custom NVFP4 projection benchmark.";
    return false;
  }
  if (options.warmup_iterations < 0 || options.benchmark_iterations <= 0) {
    error_message = "warmup_iterations must be >= 0 and benchmark_iterations must be > 0.";
    return false;
  }
  int kernel_variant = 0;
  if (options.kernel == "row") {
    kernel_variant = 0;
  } else if (options.kernel == "warp") {
    kernel_variant = 1;
  } else if (options.kernel == "scale-group") {
    kernel_variant = 2;
  } else if (options.kernel == "blackwell-fp4") {
    kernel_variant = 3;
  } else {
    error_message = "unknown custom NVFP4 projection kernel: " + options.kernel + " (expected row|warp|scale-group|blackwell-fp4).";
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
  if (weight_info.shape.size() != 2 || scale_info.shape.size() != 2 ||
      input_scale_tensor.data.size() != 1 || weight_scale2_tensor.data.size() != 1) {
    error_message = "Invalid NVFP4 tensor family for custom projection benchmark.";
    return false;
  }

  const int rows = static_cast<int>(weight_info.shape[0]);
  const int cols = static_cast<int>(weight_info.shape[1] * 2);
  const std::vector<std::int64_t> expected_scale_shape{weight_info.shape[0], cols / 16};
  if (scale_info.shape != expected_scale_shape) {
    error_message = "NVFP4 weight_scale shape does not match packed weight shape.";
    return false;
  }

  double avg_ms = 0.0;
  double max_abs_error = 0.0;
  if (!cuda::run_nvfp4_custom_projection_benchmark(
        packed_weights,
        weight_scales,
        input_scale_tensor.data[0],
        weight_scale2_tensor.data[0],
        rows,
        cols,
        kernel_variant,
        options.warmup_iterations,
        options.benchmark_iterations,
        avg_ms,
        max_abs_error,
        error_message)) {
    return false;
  }

  result.tensor_base_name = options.tensor_base_name;
  result.kernel = options.kernel;
  if (options.kernel == "blackwell-fp4") {
    result.backend_note = "native SM120 mma.sync mxf4nvf4 block-scale path";
  } else if (options.kernel == "scale-group") {
    result.backend_note = "CUDA-core fallback; one thread processes one 16-value NVFP4 scale group";
  }
  result.packed_shape = weight_info.shape;
  result.scale_shape = scale_info.shape;
  result.source_shape = {weight_info.shape[0], weight_info.shape[1] * 2};
  result.avg_iteration_ms = avg_ms;
  result.iterations_per_second = 1000.0 / avg_ms;
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
