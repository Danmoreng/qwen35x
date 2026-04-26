#include "qwen35x/runtime/qwen35x_cuda_backend.h"

#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cuda_bench.h"
#include "qwen35x/weights/safetensors.h"
#include "qwen35x/weights/modelopt_nvfp4.h"

#if QWEN35X_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>

namespace qwen35x::cuda_backend {

const char * to_string(const Qwen35xWeightPrecision precision) {
  switch (precision) {
    case Qwen35xWeightPrecision::bf16:
      return "bf16";
    case Qwen35xWeightPrecision::nvfp4:
      return "nvfp4";
    default:
      return "unknown";
  }
}

const char * to_string(const Qwen35xCachePrecision precision) {
  switch (precision) {
    case Qwen35xCachePrecision::bf16:
      return "bf16";
    case Qwen35xCachePrecision::quantized:
      return "quantized";
    default:
      return "unknown";
  }
}

bool validate_descriptor(const Qwen35xModelDescriptor & descriptor, std::string & error_message) {
  if (descriptor.num_layers <= 0 ||
      descriptor.hidden_size <= 0 ||
      descriptor.intermediate_size <= 0 ||
      descriptor.vocab_size <= 0 ||
      descriptor.fa_num_q_heads <= 0 ||
      descriptor.fa_num_kv_heads <= 0 ||
      descriptor.fa_head_dim <= 0 ||
      descriptor.fa_rot_dim <= 0 ||
      descriptor.dn_num_heads <= 0 ||
      descriptor.dn_gate_heads <= 0 ||
      descriptor.dn_key_dim <= 0 ||
      descriptor.dn_value_head_dim <= 0 ||
      descriptor.dn_value_dim <= 0 ||
      descriptor.dn_conv_kernel <= 0) {
    error_message = "invalid Qwen3.5 CUDA model descriptor: dimensions must be positive.";
    return false;
  }
  if (static_cast<int>(descriptor.layer_type.size()) != descriptor.num_layers) {
    error_message =
      "invalid Qwen3.5 CUDA model descriptor: layer schedule length " +
      std::to_string(descriptor.layer_type.size()) +
      " does not match layer count " + std::to_string(descriptor.num_layers) + ".";
    return false;
  }
  if ((descriptor.fa_head_dim % 2) != 0 || descriptor.fa_rot_dim > descriptor.fa_head_dim) {
    error_message = "invalid Qwen3.5 CUDA model descriptor: RoPE dimensions are incompatible with head_dim.";
    return false;
  }
  if (descriptor.fa_num_q_heads % descriptor.fa_num_kv_heads != 0) {
    error_message = "invalid Qwen3.5 CUDA model descriptor: attention heads must be divisible by KV heads.";
    return false;
  }
  if (descriptor.dn_gate_heads % descriptor.dn_num_heads != 0) {
    error_message = "invalid Qwen3.5 CUDA model descriptor: DeltaNet value heads must be divisible by key heads.";
    return false;
  }
  if (descriptor.dn_value_dim != descriptor.dn_value_head_dim * (descriptor.dn_gate_heads / descriptor.dn_num_heads)) {
    error_message = "invalid Qwen3.5 CUDA model descriptor: grouped DeltaNet value dimension is inconsistent.";
    return false;
  }
  return true;
}

bool build_model_descriptor(
  const qwen35x::ModelProfile & profile,
  Qwen35xModelDescriptor & descriptor,
  std::string & error_message) {
  descriptor = Qwen35xModelDescriptor{};
  descriptor.family = profile.family;
  descriptor.variant = profile.variant;
  descriptor.num_layers = profile.text.num_hidden_layers;
  descriptor.hidden_size = profile.text.hidden_size;
  descriptor.intermediate_size = profile.text.intermediate_size;
  descriptor.vocab_size = profile.text.vocab_size;
  descriptor.fa_num_q_heads = profile.text.num_attention_heads;
  descriptor.fa_num_kv_heads = profile.text.num_key_value_heads;
  descriptor.fa_head_dim = profile.text.head_dim;
  descriptor.fa_rot_dim = static_cast<int>(
    std::lround(static_cast<float>(profile.text.head_dim) * profile.text.partial_rotary_factor));
  descriptor.rope_theta = profile.text.rope_theta;
  descriptor.dn_num_heads = profile.text.linear_num_key_heads;
  descriptor.dn_gate_heads = profile.text.linear_num_value_heads;
  descriptor.dn_key_dim = profile.text.linear_key_head_dim;
  descriptor.dn_value_head_dim = profile.text.linear_value_head_dim;
  if (descriptor.dn_num_heads > 0 && descriptor.dn_gate_heads > 0) {
    descriptor.dn_value_dim = descriptor.dn_value_head_dim * (descriptor.dn_gate_heads / descriptor.dn_num_heads);
  }
  descriptor.dn_conv_kernel = profile.text.linear_conv_kernel_dim;

  if (static_cast<int>(profile.fingerprint.attention_schedule.size()) != descriptor.num_layers) {
    error_message =
      "failed to build Qwen3.5 CUDA model descriptor: attention schedule length " +
      std::to_string(profile.fingerprint.attention_schedule.size()) +
      " does not match num_hidden_layers " + std::to_string(descriptor.num_layers) + ".";
    return false;
  }

  descriptor.layer_type.reserve(profile.fingerprint.attention_schedule.size());
  for (const auto block : profile.fingerprint.attention_schedule) {
    descriptor.layer_type.push_back(block == qwen35x::AttentionBlock::full ? 1 : 0);
  }

  std::string validation_error;
  if (!validate_descriptor(descriptor, validation_error)) {
    error_message = "failed to build Qwen3.5 CUDA model descriptor: " + validation_error;
    return false;
  }
  return true;
}

int descriptor_full_layer_count(const Qwen35xModelDescriptor & descriptor) {
  return static_cast<int>(std::count(descriptor.layer_type.begin(), descriptor.layer_type.end(), 1));
}

int descriptor_delta_layer_count(const Qwen35xModelDescriptor & descriptor) {
  return descriptor.num_layers - descriptor_full_layer_count(descriptor);
}

int descriptor_fa_gqa_ratio(const Qwen35xModelDescriptor & descriptor) {
  return descriptor.fa_num_q_heads / descriptor.fa_num_kv_heads;
}

int descriptor_fa_q_size(const Qwen35xModelDescriptor & descriptor) {
  return descriptor.fa_num_q_heads * descriptor.fa_head_dim;
}

int descriptor_fa_qproj_size(const Qwen35xModelDescriptor & descriptor) {
  return descriptor_fa_q_size(descriptor) * 2;
}

int descriptor_fa_kv_size(const Qwen35xModelDescriptor & descriptor) {
  return descriptor.fa_num_kv_heads * descriptor.fa_head_dim;
}

int descriptor_dn_qk_size(const Qwen35xModelDescriptor & descriptor) {
  return descriptor.dn_num_heads * descriptor.dn_key_dim;
}

int descriptor_dn_v_size(const Qwen35xModelDescriptor & descriptor) {
  return descriptor.dn_num_heads * descriptor.dn_value_dim;
}

int descriptor_dn_conv_channels(const Qwen35xModelDescriptor & descriptor) {
  return descriptor_dn_qk_size(descriptor) * 2 + descriptor_dn_v_size(descriptor);
}

#if QWEN35X_HAS_CUDA

namespace {

constexpr int kMaxDecodeBlocks = 1024;

constexpr int kLayerType0p8b[] = {
  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1
};
constexpr int kLayerType4b[] = {
  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1
};

struct PackedLayerWeights {
  int layer_type = 0;
  int pad[3] = {0, 0, 0};
  void * ptrs[14] = {};
};

struct PackedNvfp4Weight {
  void * packed_weight = nullptr;
  void * weight_scale = nullptr;
  float * weight_scale_2 = nullptr;
  void * tc_packed_weight = nullptr;
  void * tc_weight_scale = nullptr;
  float * tc_alpha = nullptr;
  void * sm120_packed_weight_fragments = nullptr;
  void * sm120_weight_scale_fragments = nullptr;
  int output_size = 0;
  int input_size = 0;
  int padded_output_size = 0;
  int padded_scale_cols = 0;
  int weight_padding_cols = 0;
  int sm120_row_tiles = 0;
  int sm120_k_blocks = 0;
};

struct PackedLayerNvfp4Weights {
  int layer_type = 0;
  int pad[3] = {0, 0, 0};
  PackedNvfp4Weight ptrs[14] = {};
};

#define QWEN35X_LAUNCH_DECODE_PARAMS \
  int input_token_id, \
  int * output_token_id, \
  const void * embed_weight, \
  const PackedLayerWeights * layer_weights, \
  const PackedLayerNvfp4Weights * layer_nvfp4_weights, \
  const void * final_norm_weight, \
  const void * lm_head_weight, \
  void * fa_k_cache, \
  void * fa_v_cache, \
  void * dn_states, \
  void * conv_bufs, \
  void * hidden_buffer, \
  void * g_activations, \
  void * g_residual, \
  void * g_qkv_scratch, \
  void * g_kv_scratch, \
  void * g_attn_out, \
  void * g_attn_partials, \
  void * g_mlp_inter, \
  void * g_z_scratch, \
  void * g_beta_scratch, \
  void * g_alpha_scratch, \
  void * g_normalized, \
  unsigned int * barrier_counter, \
  unsigned int * barrier_generation, \
  float * block_max_vals, \
  int * block_max_idxs, \
  unsigned int * lm_sync_counter, \
  float * seen_token_mask, \
  float repetition_penalty, \
  int position, \
  int max_seq_len, \
  Qwen35xDecodeProfile * profile, \
  cudaStream_t stream

#define QWEN35X_LAUNCH_PREFILL_PARAMS \
  const int * token_ids, \
  int seq_len, \
  int * output_token, \
  const void * embed_weight, \
  const PackedLayerWeights * layers, \
  const void * final_norm_w, \
  const void * lm_head_w, \
  void * fa_k_cache, \
  void * fa_v_cache, \
  void * dn_states, \
  void * conv_bufs, \
  void * hidden, \
  void * residual, \
  void * normalized, \
  void * proj_buf, \
  void * proj_buf2, \
  void * attn_buf, \
  void * mlp_buf, \
  void * dn_out_buf, \
  float * dn_qkv_f32, \
  float * dn_out_f32, \
  void * beta_buf, \
  void * alpha_buf, \
  void * final_normed, \
  void * hidden_bf16_out, \
  void * lm_bmv, \
  void * lm_bmi, \
  float * seen_token_mask, \
  float repetition_penalty, \
  const float * rope_cos, \
  const float * rope_sin, \
  int max_seq_len, \
  int mlp_chunk_tokens, \
  int attention_query_tokens, \
  int compute_logits, \
  Qwen35xPrefillProfile * profile, \
  cudaStream_t stream

using LaunchDecodeFn = void (*)(QWEN35X_LAUNCH_DECODE_PARAMS);
using LaunchPrefillFn = void (*)(QWEN35X_LAUNCH_PREFILL_PARAMS);

extern "C" void launch_decode_0p8b(QWEN35X_LAUNCH_DECODE_PARAMS);
extern "C" void launch_decode_4b(QWEN35X_LAUNCH_DECODE_PARAMS);
extern "C" void launch_prefill_bf16_0p8b(QWEN35X_LAUNCH_PREFILL_PARAMS);
extern "C" void launch_prefill_bf16_4b(QWEN35X_LAUNCH_PREFILL_PARAMS);
#undef QWEN35X_LAUNCH_DECODE_PARAMS
#undef QWEN35X_LAUNCH_PREFILL_PARAMS
extern "C" void set_decode_blocks_override_0p8b(int blocks);
extern "C" void set_decode_blocks_override_4b(int blocks);
extern "C" int query_max_safe_decode_blocks_0p8b();
extern "C" int query_max_safe_decode_blocks_4b();

struct VariantDescriptor {
  const char * name;
  int num_layers;
  int hidden_size;
  int intermediate_size;
  int vocab_size;
  int fa_num_q_heads;
  int fa_num_kv_heads;
  int fa_head_dim;
  int fa_rot_dim;
  int dn_num_heads;
  int dn_gate_heads;
  int dn_key_dim;
  int dn_value_dim;
  int dn_conv_kernel;
  const int * layer_type;
  int default_attention_query_tokens;
  LaunchDecodeFn launch_decode;
  LaunchPrefillFn launch_prefill;
  void (*set_decode_blocks_override)(int);
  int (*query_max_safe_decode_blocks)();

  int fa_gqa_ratio() const { return fa_num_q_heads / fa_num_kv_heads; }
  int fa_q_size() const { return fa_num_q_heads * fa_head_dim; }
  int fa_qproj_size() const { return fa_q_size() * 2; }
  int fa_kv_size() const { return fa_num_kv_heads * fa_head_dim; }
  int dn_qk_size() const { return dn_num_heads * dn_key_dim; }
  int dn_v_size() const { return dn_num_heads * dn_value_dim; }
  int dn_conv_channels() const { return dn_qk_size() * 2 + dn_v_size(); }
};

const VariantDescriptor kVariant0p8b{
  "0.8b", 24, 1024, 3584, 248320, 8, 2, 256, 64, 16, 16, 128, 128, 4, kLayerType0p8b, 3584,
  launch_decode_0p8b, launch_prefill_bf16_0p8b, set_decode_blocks_override_0p8b, query_max_safe_decode_blocks_0p8b
};
const VariantDescriptor kVariant4b{
  "4b", 32, 2560, 9216, 248320, 16, 4, 256, 64, 16, 32, 128, 256, 4, kLayerType4b, 64,
  launch_decode_4b, launch_prefill_bf16_4b, set_decode_blocks_override_4b, query_max_safe_decode_blocks_4b
};
const VariantDescriptor * const kVariants[] = {&kVariant0p8b, &kVariant4b};

struct DeviceArena {
  std::vector<void *> allocations;

  ~DeviceArena() {
    for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
      if (*it != nullptr) {
        cudaFree(*it);
      }
    }
  }

  bool alloc_bytes(std::size_t bytes, void *& out, std::string & error_message) {
    out = nullptr;
    const auto status = cudaMalloc(&out, bytes);
    if (status != cudaSuccess) {
      error_message = "cudaMalloc failed for " + std::to_string(bytes) + " bytes: " + cudaGetErrorString(status);
      return false;
    }
    allocations.push_back(out);
    return true;
  }
};

struct BackendState {
  void * embed_weight = nullptr;
  void * final_norm_weight = nullptr;
  void * lm_head_weight = nullptr;
  PackedLayerWeights * layer_weights = nullptr;
  PackedLayerNvfp4Weights * layer_nvfp4_weights = nullptr;

  void * fa_k_cache = nullptr;
  void * fa_v_cache = nullptr;
  void * dn_states = nullptr;
  void * conv_bufs = nullptr;

  void * hidden_buffer = nullptr;
  void * g_activations = nullptr;
  void * g_residual = nullptr;
  void * g_qkv_scratch = nullptr;
  void * g_kv_scratch = nullptr;
  void * g_attn_out = nullptr;
  void * g_attn_partials = nullptr;
  void * g_mlp_inter = nullptr;
  void * g_z_scratch = nullptr;
  void * g_beta_scratch = nullptr;
  void * g_alpha_scratch = nullptr;
  void * g_normalized = nullptr;
  unsigned int * barrier_counter = nullptr;
  unsigned int * barrier_generation = nullptr;
  float * block_max_vals = nullptr;
  int * block_max_idxs = nullptr;
  unsigned int * lm_sync_counter = nullptr;
  float * seen_token_mask = nullptr;
  int * output_token = nullptr;

  int * token_ids = nullptr;

  void * pf_hidden = nullptr;
  void * pf_residual = nullptr;
  void * pf_normalized = nullptr;
  void * pf_proj_buf = nullptr;
  void * pf_proj_buf2 = nullptr;
  void * pf_attn_buf = nullptr;
  void * pf_mlp_buf = nullptr;
  void * pf_dn_out_buf = nullptr;
  float * pf_dn_qkv_f32 = nullptr;
  float * pf_dn_out_f32 = nullptr;
  float * pf_beta_buf = nullptr;
  float * pf_alpha_buf = nullptr;
  void * pf_final_normed = nullptr;
  void * pf_hidden_bf16_out = nullptr;
  float * pf_lm_bmv = nullptr;
  int * pf_lm_bmi = nullptr;
  float * pf_rope_cos = nullptr;
  float * pf_rope_sin = nullptr;
  float * fp4_projection_input_f32 = nullptr;
  std::uint8_t * fp4_activation = nullptr;
  std::uint8_t * fp4_activation_scales = nullptr;
  float * fp4_projection_output_f32 = nullptr;
  int prefill_mlp_chunk_tokens = 0;
  int prefill_attention_query_tokens = 0;
};

struct Nvfp4TensorDevice {
  void * packed_weight = nullptr;
  void * weight_scale = nullptr;
  float * input_scale = nullptr;
  float * weight_scale_2 = nullptr;
  void * tc_packed_weight = nullptr;
  void * tc_weight_scale = nullptr;
  float * tc_alpha = nullptr;
  void * sm120_packed_weight_fragments = nullptr;
  void * sm120_weight_scale_fragments = nullptr;
  int output_size = 0;
  int input_size = 0;
  int padded_output_size = 0;
  int padded_scale_cols = 0;
  int weight_padding_cols = 0;
  int sm120_row_tiles = 0;
  int sm120_k_blocks = 0;
};

struct Nvfp4BackendState {
  std::vector<Nvfp4TensorDevice> tensors;
};

bool check_cuda(cudaError_t status, const std::string & label, std::string & error_message) {
  if (status == cudaSuccess) {
    return true;
  }
  error_message = label + " failed: " + cudaGetErrorString(status);
  return false;
}

double elapsed_ms_since(const std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

int get_prefill_mlp_chunk_tokens(const int max_context) {
  constexpr int kDefaultChunkTokens = 4096;
  int chunk_tokens = std::min(max_context, kDefaultChunkTokens);
  if (const char *env = std::getenv("QWEN35X_PREFILL_MLP_CHUNK_TOKENS")) {
    try {
      const int requested = std::stoi(env);
      if (requested > 0) {
        chunk_tokens = std::min(max_context, requested);
      }
    } catch (...) {
      // Ignore invalid tuning overrides and keep the conservative default.
    }
  }
  return std::max(1, chunk_tokens);
}

int get_prefill_attention_query_tokens(const VariantDescriptor & variant, const int max_context) {
  int query_tokens = std::min(max_context, variant.default_attention_query_tokens);
  if (const char *env = std::getenv("QWEN35X_PREFILL_ATTENTION_QUERY_TOKENS")) {
    try {
      const int requested = std::stoi(env);
      if (requested > 0) {
        query_tokens = std::min(max_context, requested);
      }
    } catch (...) {
      // Ignore invalid tuning overrides and keep the variant default.
    }
  }
  return std::max(1, query_tokens);
}

bool load_bf16_tensor_to_device(
  const std::string & model_dir,
  const std::vector<std::string> & tensor_names,
  const std::vector<std::vector<std::int64_t>> & accepted_shapes,
  DeviceArena & arena,
  void *& out_device_ptr,
  std::string & out_tensor_name,
  std::string & error_message) {
  auto float_to_bf16 = [](float value) -> std::uint16_t {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t lsb = (bits >> 16U) & 1U;
    const std::uint32_t rounding_bias = 0x7FFFU + lsb;
    bits += rounding_bias;
    return static_cast<std::uint16_t>(bits >> 16U);
  };
  auto decode_nvfp4_e2m1 = [](std::uint8_t nibble) -> float {
    constexpr float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = levels[nibble & 0x7u];
    return (nibble & 0x8u) ? -value : value;
  };
  auto decode_e4m3 = [](std::uint8_t bits) -> float {
    if (bits == 0) {
      return 0.0f;
    }
    const int sign = (bits >> 7) & 1;
    const int exponent = (bits >> 3) & 0xf;
    const int mantissa = bits & 0x7;
    float value = 0.0f;
    if (exponent == 0) {
      value = std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
    } else {
      value = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
    }
    return sign ? -value : value;
  };
  auto read_raw = [](const std::string & file, const qwen35x::SafetensorTensorInfo & tensor_info, std::vector<std::uint8_t> & out, std::string & local_error) -> bool {
    const std::uint64_t byte_count = tensor_info.data_end - tensor_info.data_start;
    out.resize(static_cast<std::size_t>(byte_count));
    std::ifstream in(file, std::ios::binary);
    if (!in) {
      local_error = "Could not open safetensors shard: " + file;
      return false;
    }
    in.seekg(static_cast<std::streamoff>(tensor_info.data_start), std::ios::beg);
    if (!in) {
      local_error = "Failed to seek tensor '" + tensor_info.name + "' in shard.";
      return false;
    }
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!in) {
      local_error = "Failed to read tensor '" + tensor_info.name + "' from shard.";
      return false;
    }
    return true;
  };

  std::string last_error;
  for (const auto & name : tensor_names) {
    std::string tensor_file;
    qwen35x::SafetensorTensorInfo info;
    std::vector<std::uint16_t> host_data;
    std::string local_error;

    if (!qwen35x::SafetensorLoader::resolve_tensor_file(model_dir, name, tensor_file, local_error)) {
      last_error = local_error;
      continue;
    }
    if (!qwen35x::SafetensorLoader::load_tensor_info(tensor_file, name, info, local_error)) {
      last_error = local_error;
      continue;
    }
    if (info.dtype == "U8" && !accepted_shapes.empty()) {
      bool matched_quantized_shape = false;
      std::vector<std::int64_t> source_shape;
      for (const auto & accepted_shape : accepted_shapes) {
        if (accepted_shape.size() == 2 &&
            info.shape.size() == 2 &&
            info.shape[0] == accepted_shape[0] &&
            info.shape[1] * 2 == accepted_shape[1] &&
            (accepted_shape[1] % 16) == 0) {
          matched_quantized_shape = true;
          source_shape = accepted_shape;
          break;
        }
      }
      if (!matched_quantized_shape) {
        last_error = "Tensor '" + name + "' has U8 NVFP4 shape that does not match accepted source shapes.";
        continue;
      }

      std::vector<std::uint8_t> packed;
      if (!read_raw(tensor_file, info, packed, local_error)) {
        last_error = local_error;
        continue;
      }

      const std::string base_name = name.size() > 7 && name.substr(name.size() - 7) == ".weight"
        ? name.substr(0, name.size() - 7)
        : name;
      std::string scale_file;
      qwen35x::SafetensorTensorInfo scale_info;
      if (!qwen35x::SafetensorLoader::resolve_tensor_file(model_dir, base_name + ".weight_scale", scale_file, local_error) ||
          !qwen35x::SafetensorLoader::load_tensor_info(scale_file, base_name + ".weight_scale", scale_info, local_error)) {
        last_error = local_error;
        continue;
      }
      const std::vector<std::int64_t> expected_scale_shape{source_shape[0], source_shape[1] / 16};
      if (scale_info.dtype != "F8_E4M3" || scale_info.shape != expected_scale_shape) {
        last_error = "Tensor '" + scale_info.name + "' shape or dtype mismatch for NVFP4 dequantization.";
        continue;
      }
      std::vector<std::uint8_t> scales;
      if (!read_raw(scale_file, scale_info, scales, local_error)) {
        last_error = local_error;
        continue;
      }

      qwen35x::SafetensorTensorF32 input_scale_tensor;
      qwen35x::SafetensorTensorF32 weight_scale2_tensor;
      if (!qwen35x::SafetensorLoader::read_tensor_f32(model_dir, base_name + ".input_scale", input_scale_tensor, local_error) ||
          !qwen35x::SafetensorLoader::read_tensor_f32(model_dir, base_name + ".weight_scale_2", weight_scale2_tensor, local_error) ||
          input_scale_tensor.data.size() != 1 ||
          weight_scale2_tensor.data.size() != 1) {
        last_error = local_error.empty() ? ("Invalid NVFP4 scalar scales for tensor '" + name + "'.") : local_error;
        continue;
      }

      const std::int64_t rows = source_shape[0];
      const std::int64_t cols = source_shape[1];
      const std::int64_t packed_cols = cols / 2;
      const std::int64_t scale_cols = cols / 16;
      host_data.resize(static_cast<std::size_t>(rows * cols));
      for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t col = 0; col < cols; ++col) {
          const std::uint8_t packed_byte = packed[static_cast<std::size_t>(row * packed_cols + col / 2)];
          const std::uint8_t nibble = (col & 1) == 0 ? (packed_byte & 0x0fu) : (packed_byte >> 4u);
          const float scale = decode_e4m3(scales[static_cast<std::size_t>(row * scale_cols + col / 16)]) * weight_scale2_tensor.data[0];
          const float value = decode_nvfp4_e2m1(nibble) * scale;
          host_data[static_cast<std::size_t>(row * cols + col)] = float_to_bf16(value);
        }
      }
    } else if (!accepted_shapes.empty() &&
        std::find(accepted_shapes.begin(), accepted_shapes.end(), info.shape) == accepted_shapes.end()) {
      auto shape_to_string = [](const std::vector<std::int64_t> & shape) {
        std::string out = "[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
          if (i != 0) {
            out += ",";
          }
          out += std::to_string(shape[i]);
        }
        out += "]";
        return out;
      };
      std::string expected;
      for (std::size_t i = 0; i < accepted_shapes.size(); ++i) {
        if (i != 0) {
          expected += " or ";
        }
        expected += shape_to_string(accepted_shapes[i]);
      }
      error_message =
        "Tensor '" + name + "' shape mismatch: expected " + expected +
        " actual " + shape_to_string(info.shape) + ".";
      return false;
    }

    if (!host_data.empty()) {
      // Already populated by the ModelOpt NVFP4 dequantization path above.
    } else if (info.dtype == "BF16") {
      if (!qwen35x::SafetensorLoader::read_bf16_tensor(tensor_file, info, host_data, local_error)) {
        last_error = local_error;
        continue;
      }
    } else {
      qwen35x::SafetensorTensorF32 as_f32;
      if (!qwen35x::SafetensorLoader::read_tensor_f32(model_dir, name, as_f32, local_error)) {
        last_error = local_error;
        continue;
      }
      host_data.resize(as_f32.data.size());
      for (std::size_t i = 0; i < as_f32.data.size(); ++i) {
        host_data[i] = float_to_bf16(as_f32.data[i]);
      }
    }

    void * device_ptr = nullptr;
    if (!arena.alloc_bytes(host_data.size() * sizeof(std::uint16_t), device_ptr, local_error)) {
      error_message = local_error;
      return false;
    }
    if (!check_cuda(
          cudaMemcpy(device_ptr, host_data.data(), host_data.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
          "cudaMemcpy(H2D BF16 " + name + ")",
          local_error)) {
      error_message = local_error;
      return false;
    }

    out_device_ptr = device_ptr;
    out_tensor_name = name;
    return true;
  }

  if (!last_error.empty()) {
    error_message = last_error;
  } else {
    error_message = "None of the candidate tensors were found.";
  }
  return false;
}

bool read_raw_tensor_bytes(
  const std::string & safetensors_file,
  const qwen35x::SafetensorTensorInfo & info,
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

bool load_tensor_info_from_model_dir(
  const std::string & model_dir,
  const std::string & tensor_name,
  std::string & tensor_file,
  qwen35x::SafetensorTensorInfo & info,
  std::string & error_message) {
  if (!qwen35x::SafetensorLoader::resolve_tensor_file(model_dir, tensor_name, tensor_file, error_message)) {
    return false;
  }
  return qwen35x::SafetensorLoader::load_tensor_info(tensor_file, tensor_name, info, error_message);
}

int round_up_to_multiple(const int value, const int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

bool upload_bytes_to_device(
  const std::vector<std::uint8_t> & host_data,
  const std::string & label,
  DeviceArena & arena,
  void *& out_device_ptr,
  std::string & error_message) {
  if (!arena.alloc_bytes(host_data.size(), out_device_ptr, error_message)) {
    return false;
  }
  return check_cuda(
    cudaMemcpy(out_device_ptr, host_data.data(), host_data.size(), cudaMemcpyHostToDevice),
    "cudaMemcpy(H2D " + label + ")",
    error_message);
}

bool upload_u32_to_device(
  const std::vector<std::uint32_t> & host_data,
  const std::string & label,
  DeviceArena & arena,
  void *& out_device_ptr,
  std::string & error_message) {
  const std::size_t bytes = host_data.size() * sizeof(std::uint32_t);
  if (!arena.alloc_bytes(bytes, out_device_ptr, error_message)) {
    return false;
  }
  return check_cuda(
    cudaMemcpy(out_device_ptr, host_data.data(), bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(H2D " + label + ")",
    error_message);
}

bool upload_scalar_to_device(
  const float value,
  const std::string & label,
  DeviceArena & arena,
  float *& out_device_ptr,
  std::string & error_message) {
  if (!arena.alloc_bytes(sizeof(float), reinterpret_cast<void *&>(out_device_ptr), error_message)) {
    return false;
  }
  return check_cuda(
    cudaMemcpy(out_device_ptr, &value, sizeof(float), cudaMemcpyHostToDevice),
    "cudaMemcpy(H2D " + label + ")",
    error_message);
}

bool read_raw_tensor_from_model_dir(
  const std::string & model_dir,
  const std::string & tensor_name,
  const std::string & expected_dtype,
  const std::vector<std::int64_t> & expected_shape,
  std::vector<std::uint8_t> & out_host_data,
  std::string & error_message) {
  std::string tensor_file;
  qwen35x::SafetensorTensorInfo info;
  if (!load_tensor_info_from_model_dir(model_dir, tensor_name, tensor_file, info, error_message)) {
    return false;
  }
  if (info.dtype != expected_dtype) {
    error_message = "Tensor '" + tensor_name + "' has dtype " + info.dtype + " (expected " + expected_dtype + ").";
    return false;
  }
  if (info.shape != expected_shape) {
    auto shape_to_string = [](const std::vector<std::int64_t> & shape) {
      std::string out = "[";
      for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
          out += ",";
        }
        out += std::to_string(shape[i]);
      }
      out += "]";
      return out;
    };
    error_message =
      "Tensor '" + tensor_name + "' shape mismatch: expected " + shape_to_string(expected_shape) +
      " actual " + shape_to_string(info.shape) + ".";
    return false;
  }
  return read_raw_tensor_bytes(tensor_file, info, out_host_data, error_message);
}

bool read_scalar_f32_from_model_dir(
  const std::string & model_dir,
  const std::string & tensor_name,
  float & out_value,
  std::string & error_message) {
  qwen35x::SafetensorTensorF32 tensor;
  if (!qwen35x::SafetensorLoader::read_tensor_f32(model_dir, tensor_name, tensor, error_message)) {
    return false;
  }
  if (tensor.dtype != "F32" || !tensor.shape.empty() || tensor.data.size() != 1) {
    error_message = "Tensor '" + tensor_name + "' must be scalar F32.";
    return false;
  }
  out_value = tensor.data[0];
  return true;
}

std::vector<std::uint8_t> pad_nvfp4_weight_for_tensor_core(
  const std::vector<std::uint8_t> & packed_weight,
  const int rows,
  const int cols,
  int & padded_rows,
  int & padded_cols,
  int & weight_padding_cols) {
  padded_rows = round_up_to_multiple(rows, 32);
  padded_cols = round_up_to_multiple(cols, 32);
  const int source_packed_cols = cols / 2;
  const int padded_packed_cols = padded_cols / 2;
  weight_padding_cols = padded_packed_cols - source_packed_cols;
  std::vector<std::uint8_t> padded(static_cast<std::size_t>(padded_rows) * padded_packed_cols, 0);
  for (int row = 0; row < rows; ++row) {
    auto * dst_row = padded.data() + static_cast<std::size_t>(row) * padded_packed_cols;
    const auto * src_row = packed_weight.data() + static_cast<std::size_t>(row) * source_packed_cols;
    for (int col = 0; col < source_packed_cols; ++col) {
      dst_row[col] = static_cast<std::uint8_t>((src_row[col] >> 4u) | (src_row[col] << 4u));
    }
  }
  return padded;
}

std::vector<std::uint8_t> swizzle_nvfp4_block_scale_for_tensor_core(
  const std::vector<std::uint8_t> & weight_scale,
  const int rows,
  const int scale_cols,
  int & padded_rows,
  int & padded_scale_cols) {
  padded_rows = round_up_to_multiple(rows, 128);
  padded_scale_cols = round_up_to_multiple(scale_cols, 4);
  std::vector<std::uint8_t> padded(static_cast<std::size_t>(padded_rows) * padded_scale_cols, 0);
  for (int row = 0; row < rows; ++row) {
    std::memcpy(
      padded.data() + static_cast<std::size_t>(row) * padded_scale_cols,
      weight_scale.data() + static_cast<std::size_t>(row) * scale_cols,
      static_cast<std::size_t>(scale_cols));
  }

  std::vector<std::uint8_t> swizzled(padded.size(), 0);
  std::size_t out_idx = 0;
  for (int row_block = 0; row_block < padded_rows / 128; ++row_block) {
    for (int col_block = 0; col_block < padded_scale_cols / 4; ++col_block) {
      for (int row_in_group = 0; row_in_group < 32; ++row_in_group) {
        for (int row_group = 0; row_group < 4; ++row_group) {
          for (int col_in_group = 0; col_in_group < 4; ++col_in_group) {
            const int row = row_block * 128 + row_group * 32 + row_in_group;
            const int col = col_block * 4 + col_in_group;
            swizzled[out_idx++] = padded[static_cast<std::size_t>(row) * padded_scale_cols + col];
          }
        }
      }
    }
  }
  return swizzled;
}

void set_packed_u4(std::uint32_t & word, const int nibble_index, const std::uint8_t nibble) {
  const int shift = nibble_index * 4;
  word = static_cast<std::uint32_t>((word & ~(0xfu << shift)) | ((static_cast<std::uint32_t>(nibble) & 0xfu) << shift));
}

std::vector<std::uint32_t> pack_nvfp4_weight_for_sm120_mma(
  const std::vector<std::uint8_t> & packed_weight,
  const int rows,
  const int cols,
  int & row_tiles,
  int & k_blocks) {
  row_tiles = (rows + 7) / 8;
  k_blocks = cols / 64;
  const int packed_cols = cols / 2;
  std::vector<std::uint32_t> fragments(static_cast<std::size_t>(row_tiles) * k_blocks * 32 * 2, 0);
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    for (int kb = 0; kb < k_blocks; ++kb) {
      for (int lane = 0; lane < 32; ++lane) {
        const int group_id = lane >> 2;
        const int thread_id_in_group = lane & 3;
        const int out_row = row_tile * 8 + group_id;
        for (int i = 0; i < 16; ++i) {
          const int k = kb * 64 + thread_id_in_group * 8 + (i & 7) + (i >= 8 ? 32 : 0);
          std::uint8_t nibble = 0;
          if (out_row < rows) {
            const std::uint8_t packed = packed_weight[static_cast<std::size_t>(out_row) * packed_cols + k / 2];
            nibble = (k & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
          }
          set_packed_u4(
            fragments[((static_cast<std::size_t>(row_tile) * k_blocks + kb) * 32 + lane) * 2 + i / 8],
            i & 7,
            nibble);
        }
      }
    }
  }
  return fragments;
}

std::vector<std::uint32_t> pack_nvfp4_weight_scales_for_sm120_mma(
  const std::vector<std::uint8_t> & weight_scale,
  const int rows,
  const int scale_cols,
  const int row_tiles,
  const int k_blocks) {
  std::vector<std::uint32_t> fragments(static_cast<std::size_t>(row_tiles) * k_blocks * 32, 0);
  for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
    for (int kb = 0; kb < k_blocks; ++kb) {
      for (int lane = 0; lane < 32; ++lane) {
        const int group_id = lane >> 2;
        const int out_row = row_tile * 8 + group_id;
        std::uint32_t packed_scales = 0;
        if (out_row < rows) {
          for (int group = 0; group < 4; ++group) {
            packed_scales |= static_cast<std::uint32_t>(
              weight_scale[static_cast<std::size_t>(out_row) * scale_cols + kb * 4 + group]) << (group * 8u);
          }
        }
        fragments[(static_cast<std::size_t>(row_tile) * k_blocks + kb) * 32 + lane] = packed_scales;
      }
    }
  }
  return fragments;
}

bool load_raw_tensor_to_device(
  const std::string & model_dir,
  const std::string & tensor_name,
  const std::string & expected_dtype,
  const std::vector<std::int64_t> & expected_shape,
  DeviceArena & arena,
  void *& out_device_ptr,
  std::string & error_message) {
  std::vector<std::uint8_t> host_data;
  if (!read_raw_tensor_from_model_dir(model_dir, tensor_name, expected_dtype, expected_shape, host_data, error_message)) {
    return false;
  }
  return upload_bytes_to_device(host_data, "raw " + tensor_name, arena, out_device_ptr, error_message);
}

bool load_scalar_f32_to_device(
  const std::string & model_dir,
  const std::string & tensor_name,
  DeviceArena & arena,
  float *& out_device_ptr,
  std::string & error_message) {
  float value = 0.0f;
  if (!read_scalar_f32_from_model_dir(model_dir, tensor_name, value, error_message)) {
    return false;
  }
  return upload_scalar_to_device(value, "scalar " + tensor_name, arena, out_device_ptr, error_message);
}

bool alloc_and_zero(DeviceArena & arena, std::size_t bytes, void *& out_ptr, const std::string & label, std::string & error_message) {
  if (!arena.alloc_bytes(bytes, out_ptr, error_message)) {
    return false;
  }
  return check_cuda(cudaMemset(out_ptr, 0, bytes), "cudaMemset(" + label + ")", error_message);
}

std::size_t nvfp4_tiled_scale_bytes(const int rows, const int cols) {
  return static_cast<std::size_t>(round_up_to_multiple(rows, 128)) * round_up_to_multiple(cols / 16, 4);
}

bool resolve_model_descriptor(
  const Qwen35xCudaBackendConfig & config,
  Qwen35xModelDescriptor & descriptor,
  std::string & error_message);

const VariantDescriptor * select_variant_for_descriptor(
  const Qwen35xModelDescriptor & descriptor,
  std::string & error_message) {
  if (!validate_descriptor(descriptor, error_message)) {
    return nullptr;
  }
  for (const auto * variant : kVariants) {
    const bool dimensions_match =
      descriptor.num_layers == variant->num_layers &&
      descriptor.hidden_size == variant->hidden_size &&
      descriptor.intermediate_size == variant->intermediate_size &&
      descriptor.vocab_size == variant->vocab_size &&
      descriptor.fa_num_q_heads == variant->fa_num_q_heads &&
      descriptor.fa_num_kv_heads == variant->fa_num_kv_heads &&
      descriptor.fa_head_dim == variant->fa_head_dim &&
      descriptor.fa_rot_dim == variant->fa_rot_dim &&
      descriptor.dn_num_heads == variant->dn_num_heads &&
      descriptor.dn_gate_heads == variant->dn_gate_heads &&
      descriptor.dn_key_dim == variant->dn_key_dim &&
      descriptor.dn_value_dim == variant->dn_value_dim &&
      descriptor.dn_conv_kernel == variant->dn_conv_kernel;
    if (!dimensions_match) {
      continue;
    }

    if (static_cast<int>(descriptor.layer_type.size()) != variant->num_layers) {
      continue;
    }
    bool schedule_matches = true;
    for (int i = 0; i < variant->num_layers; ++i) {
      if (descriptor.layer_type[static_cast<std::size_t>(i)] != variant->layer_type[i]) {
        schedule_matches = false;
        break;
      }
    }
    if (schedule_matches) {
      return variant;
    }
  }

  error_message =
    "unsupported Qwen3.5 CUDA model variant: descriptor variant=" + descriptor.variant +
    " layers=" + std::to_string(descriptor.num_layers) +
    " hidden=" + std::to_string(descriptor.hidden_size) +
    " intermediate=" + std::to_string(descriptor.intermediate_size) +
    " vocab=" + std::to_string(descriptor.vocab_size) +
    " q_heads=" + std::to_string(descriptor.fa_num_q_heads) +
    " kv_heads=" + std::to_string(descriptor.fa_num_kv_heads) +
    " head_dim=" + std::to_string(descriptor.fa_head_dim) +
    " rope_dim=" + std::to_string(descriptor.fa_rot_dim) +
    " linear_key_heads=" + std::to_string(descriptor.dn_num_heads) +
    " linear_value_heads=" + std::to_string(descriptor.dn_gate_heads) +
    " linear_value_head_dim=" + std::to_string(descriptor.dn_value_head_dim) +
    " grouped_linear_value_dim=" + std::to_string(descriptor.dn_value_dim) +
    ". Supported CUDA variants: 0.8b, 4b.";
  return nullptr;
}

bool resolve_model_descriptor(
  const Qwen35xCudaBackendConfig & config,
  Qwen35xModelDescriptor & descriptor,
  std::string & error_message) {
  if (config.model_descriptor.has_value()) {
    descriptor = *config.model_descriptor;
    return validate_descriptor(descriptor, error_message);
  }

  std::string profile_error;
  const auto profile = qwen35x::ProfileLoader::load_from_hf_directory(config.model_dir, profile_error);
  if (!profile) {
    error_message = "failed to load model profile for CUDA descriptor construction: " + profile_error;
    return false;
  }
  return build_model_descriptor(*profile, descriptor, error_message);
}

bool validate_descriptor_matches_variant(
  const Qwen35xModelDescriptor & descriptor,
  const VariantDescriptor & variant,
  std::string & error_message) {
  std::string select_error;
  const VariantDescriptor * selected = select_variant_for_descriptor(descriptor, select_error);
  if (selected == &variant) {
    return true;
  }
  error_message = "Qwen35x CUDA descriptor/compiled variant mismatch: " + select_error;
  return false;
}

bool initialize_backend_state(
  const Qwen35xModelDescriptor & descriptor,
  const VariantDescriptor & variant,
  const Qwen35xCudaBackendConfig & config,
  DeviceArena & arena,
  BackendState & state,
  std::string & error_message) {
  if (!validate_descriptor_matches_variant(descriptor, variant, error_message)) {
    return false;
  }
  const auto shape = [](std::initializer_list<std::int64_t> dims) {
    return std::vector<std::int64_t>(dims);
  };
  const auto one_shape = [&](std::initializer_list<std::int64_t> dims) {
    return std::vector<std::vector<std::int64_t>>{shape(dims)};
  };
  const auto conv_shapes = [&](const std::int64_t channels, const std::int64_t kernel) {
    return std::vector<std::vector<std::int64_t>>{shape({channels, kernel}), shape({channels, 1, kernel})};
  };
  const std::int64_t hidden = descriptor.hidden_size;
  const std::int64_t intermediate = descriptor.intermediate_size;
  const std::int64_t vocab = descriptor.vocab_size;
  const std::int64_t full_q_out = descriptor_fa_q_size(descriptor);
  const std::int64_t full_qproj_out = descriptor_fa_qproj_size(descriptor);
  const std::int64_t full_kv_out = descriptor_fa_kv_size(descriptor);
  const std::int64_t linear_conv_channels = descriptor_dn_conv_channels(descriptor);
  const std::int64_t linear_v_dim = descriptor_dn_v_size(descriptor);
  {
    std::string used_name;
    if (!load_bf16_tensor_to_device(
          config.model_dir,
          {"model.language_model.embed_tokens.weight", "model.embed_tokens.weight"},
          one_shape({vocab, hidden}),
          arena,
          state.embed_weight,
          used_name,
          error_message)) {
      return false;
    }
  }

  {
    std::string used_name;
    if (!load_bf16_tensor_to_device(
          config.model_dir,
          {"model.language_model.norm.weight", "model.norm.weight"},
          one_shape({hidden}),
          arena,
          state.final_norm_weight,
          used_name,
          error_message)) {
      return false;
    }
  }

  {
    std::string used_name;
    if (!load_bf16_tensor_to_device(
          config.model_dir,
          {"lm_head.weight", "model.language_model.embed_tokens.weight", "model.embed_tokens.weight"},
          one_shape({vocab, hidden}),
          arena,
          state.lm_head_weight,
          used_name,
          error_message)) {
      return false;
    }
  }

  std::vector<PackedLayerWeights> host_layers(static_cast<std::size_t>(descriptor.num_layers));
  for (int layer_idx = 0; layer_idx < descriptor.num_layers; ++layer_idx) {
    auto & layer = host_layers[static_cast<std::size_t>(layer_idx)];
    layer.layer_type = descriptor.layer_type[static_cast<std::size_t>(layer_idx)];
    const std::string base = "model.language_model.layers." + std::to_string(layer_idx) + ".";

    auto load_ptr = [&](int ptr_idx, const std::vector<std::string> & suffixes, const std::vector<std::vector<std::int64_t>> & accepted_shapes) -> bool {
      std::vector<std::string> names;
      names.reserve(suffixes.size());
      for (const auto & suffix : suffixes) {
        names.push_back(base + suffix);
      }
      std::string used_name;
      void * tensor_ptr = nullptr;
      if (!load_bf16_tensor_to_device(config.model_dir, names, accepted_shapes, arena, tensor_ptr, used_name, error_message)) {
        error_message = "layer " + std::to_string(layer_idx) + " ptr[" + std::to_string(ptr_idx) + "]: " + error_message;
        return false;
      }
      layer.ptrs[ptr_idx] = tensor_ptr;
      return true;
    };

    if (layer.layer_type == 0) {
      if (!load_ptr(0, {"input_layernorm.weight"}, one_shape({hidden})) ||
          !load_ptr(1, {"linear_attn.in_proj_qkv.weight"}, one_shape({linear_conv_channels, hidden})) ||
          !load_ptr(2, {"linear_attn.in_proj_z.weight"}, one_shape({linear_v_dim, hidden})) ||
          !load_ptr(3, {"linear_attn.in_proj_b.weight"}, one_shape({descriptor.dn_gate_heads, hidden})) ||
          !load_ptr(4, {"linear_attn.in_proj_a.weight"}, one_shape({descriptor.dn_gate_heads, hidden})) ||
          !load_ptr(5, {"linear_attn.conv1d.weight"}, conv_shapes(linear_conv_channels, descriptor.dn_conv_kernel)) ||
          !load_ptr(6, {"linear_attn.A_log"}, one_shape({descriptor.dn_gate_heads})) ||
          !load_ptr(7, {"linear_attn.dt_bias"}, one_shape({descriptor.dn_gate_heads})) ||
          !load_ptr(8, {"linear_attn.norm.weight"}, one_shape({descriptor.dn_value_head_dim})) ||
          !load_ptr(9, {"linear_attn.out_proj.weight"}, one_shape({hidden, linear_v_dim})) ||
          !load_ptr(10, {"post_attention_layernorm.weight"}, one_shape({hidden})) ||
          !load_ptr(11, {"mlp.gate_proj.weight"}, one_shape({intermediate, hidden})) ||
          !load_ptr(12, {"mlp.up_proj.weight"}, one_shape({intermediate, hidden})) ||
          !load_ptr(13, {"mlp.down_proj.weight"}, one_shape({hidden, intermediate}))) {
        return false;
      }
    } else {
      if (!load_ptr(0, {"input_layernorm.weight"}, one_shape({hidden})) ||
          !load_ptr(1, {"self_attn.q_proj.weight"}, one_shape({full_qproj_out, hidden})) ||
          !load_ptr(2, {"self_attn.k_proj.weight"}, one_shape({full_kv_out, hidden})) ||
          !load_ptr(3, {"self_attn.v_proj.weight"}, one_shape({full_kv_out, hidden})) ||
          !load_ptr(4, {"self_attn.q_norm.weight"}, one_shape({descriptor.fa_head_dim})) ||
          !load_ptr(5, {"self_attn.k_norm.weight"}, one_shape({descriptor.fa_head_dim})) ||
          !load_ptr(6, {"self_attn.o_proj.weight"}, one_shape({hidden, full_q_out})) ||
          !load_ptr(7, {"post_attention_layernorm.weight"}, one_shape({hidden})) ||
          !load_ptr(8, {"mlp.gate_proj.weight"}, one_shape({intermediate, hidden})) ||
          !load_ptr(9, {"mlp.up_proj.weight"}, one_shape({intermediate, hidden})) ||
          !load_ptr(10, {"mlp.down_proj.weight"}, one_shape({hidden, intermediate}))) {
        return false;
      }
    }
  }

  if (!arena.alloc_bytes(host_layers.size() * sizeof(PackedLayerWeights), reinterpret_cast<void *&>(state.layer_weights), error_message)) {
    return false;
  }
  if (!check_cuda(
        cudaMemcpy(
          state.layer_weights,
          host_layers.data(),
          host_layers.size() * sizeof(PackedLayerWeights),
          cudaMemcpyHostToDevice),
        "cudaMemcpy(H2D layer_weights)",
        error_message)) {
    return false;
  }

  const int n_full_layers = descriptor_full_layer_count(descriptor);
  const int n_delta_layers = descriptor_delta_layer_count(descriptor);

  const std::size_t bf16_bytes = sizeof(std::uint16_t);
  const std::size_t f32_bytes = sizeof(float);
  const int max_seq_len = config.max_context;

  if (!alloc_and_zero(
        arena,
        static_cast<std::size_t>(n_full_layers) * descriptor.fa_num_kv_heads * max_seq_len * descriptor.fa_head_dim * bf16_bytes,
        state.fa_k_cache,
        "fa_k_cache",
        error_message) ||
      !alloc_and_zero(
        arena,
        static_cast<std::size_t>(n_full_layers) * descriptor.fa_num_kv_heads * max_seq_len * descriptor.fa_head_dim * bf16_bytes,
        state.fa_v_cache,
        "fa_v_cache",
        error_message) ||
      !alloc_and_zero(
        arena,
        static_cast<std::size_t>(n_delta_layers) * descriptor.dn_num_heads * descriptor.dn_key_dim * descriptor.dn_value_dim * f32_bytes,
        state.dn_states,
        "dn_states",
        error_message) ||
      !alloc_and_zero(
        arena,
        static_cast<std::size_t>(n_delta_layers) * descriptor_dn_conv_channels(descriptor) * descriptor.dn_conv_kernel * f32_bytes,
        state.conv_bufs,
        "conv_bufs",
        error_message)) {
    return false;
  }

  if (!arena.alloc_bytes(descriptor.hidden_size * bf16_bytes, state.hidden_buffer, error_message) ||
      !arena.alloc_bytes(std::max({descriptor_fa_qproj_size(descriptor), descriptor_dn_conv_channels(descriptor), (descriptor.hidden_size * 8 + descriptor.intermediate_size)}) * f32_bytes, state.g_activations, error_message) ||
      !arena.alloc_bytes(descriptor.hidden_size * bf16_bytes, state.g_residual, error_message) ||
      !arena.alloc_bytes(std::max(descriptor_fa_qproj_size(descriptor), descriptor_dn_conv_channels(descriptor)) * f32_bytes, state.g_qkv_scratch, error_message) ||
      !arena.alloc_bytes((descriptor_fa_kv_size(descriptor) * 2) * f32_bytes, state.g_kv_scratch, error_message) ||
      !arena.alloc_bytes(std::max(descriptor_fa_q_size(descriptor), descriptor_dn_v_size(descriptor)) * f32_bytes, state.g_attn_out, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(kMaxDecodeBlocks) * descriptor_fa_gqa_ratio(descriptor) * (descriptor.fa_head_dim + 2) * f32_bytes, state.g_attn_partials, error_message) ||
      !arena.alloc_bytes(descriptor.intermediate_size * f32_bytes, state.g_mlp_inter, error_message) ||
      !arena.alloc_bytes(descriptor_dn_v_size(descriptor) * f32_bytes, state.g_z_scratch, error_message) ||
      !arena.alloc_bytes(descriptor.dn_gate_heads * f32_bytes, state.g_beta_scratch, error_message) ||
      !arena.alloc_bytes(descriptor.dn_gate_heads * f32_bytes, state.g_alpha_scratch, error_message) ||
      !arena.alloc_bytes(descriptor.hidden_size * f32_bytes, state.g_normalized, error_message) ||
      !arena.alloc_bytes(sizeof(unsigned int), reinterpret_cast<void *&>(state.barrier_counter), error_message) ||
      !arena.alloc_bytes(sizeof(unsigned int), reinterpret_cast<void *&>(state.barrier_generation), error_message) ||
      !arena.alloc_bytes(1024 * sizeof(float), reinterpret_cast<void *&>(state.block_max_vals), error_message) ||
      !arena.alloc_bytes(1024 * sizeof(int), reinterpret_cast<void *&>(state.block_max_idxs), error_message) ||
      !arena.alloc_bytes(sizeof(unsigned int), reinterpret_cast<void *&>(state.lm_sync_counter), error_message) ||
      !alloc_and_zero(
        arena,
        static_cast<std::size_t>(descriptor.vocab_size) * f32_bytes,
        reinterpret_cast<void *&>(state.seen_token_mask),
        "seen_token_mask",
        error_message) ||
      !arena.alloc_bytes(sizeof(int), reinterpret_cast<void *&>(state.output_token), error_message)) {
    return false;
  }

  if (!arena.alloc_bytes(static_cast<std::size_t>(config.max_context) * sizeof(int), reinterpret_cast<void *&>(state.token_ids), error_message)) {
    return false;
  }

  const int max_fp4_input = std::max(descriptor.hidden_size, descriptor.intermediate_size);
  const int max_fp4_output = std::max({
    descriptor.intermediate_size,
    descriptor.hidden_size,
    descriptor_fa_qproj_size(descriptor),
    descriptor_dn_conv_channels(descriptor),
    descriptor_dn_v_size(descriptor),
    descriptor.dn_gate_heads
  });
  if (!arena.alloc_bytes(static_cast<std::size_t>(max_fp4_input) * f32_bytes, reinterpret_cast<void *&>(state.fp4_projection_input_f32), error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(max_fp4_input / 2), reinterpret_cast<void *&>(state.fp4_activation), error_message) ||
      !arena.alloc_bytes(nvfp4_tiled_scale_bytes(1, max_fp4_input), reinterpret_cast<void *&>(state.fp4_activation_scales), error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(max_fp4_output) * f32_bytes, reinterpret_cast<void *&>(state.fp4_projection_output_f32), error_message)) {
    return false;
  }

  const std::size_t rope_elems = static_cast<std::size_t>(config.max_context) * descriptor.fa_rot_dim;
  if (!arena.alloc_bytes(rope_elems * sizeof(float), reinterpret_cast<void *&>(state.pf_rope_cos), error_message) ||
      !arena.alloc_bytes(rope_elems * sizeof(float), reinterpret_cast<void *&>(state.pf_rope_sin), error_message)) {
    return false;
  }
  std::vector<float> host_rope_cos(rope_elems);
  std::vector<float> host_rope_sin(rope_elems);
  for (int pos = 0; pos < config.max_context; ++pos) {
    for (int i = 0; i < descriptor.fa_rot_dim; ++i) {
      const float exponent = static_cast<float>(2 * (i % (descriptor.fa_rot_dim / 2))) / static_cast<float>(descriptor.fa_rot_dim);
      const float freq = static_cast<float>(pos) / std::pow(descriptor.rope_theta, exponent);
      host_rope_cos[static_cast<std::size_t>(pos) * descriptor.fa_rot_dim + i] = std::cos(freq);
      host_rope_sin[static_cast<std::size_t>(pos) * descriptor.fa_rot_dim + i] = std::sin(freq);
    }
  }
  if (!check_cuda(
        cudaMemcpy(state.pf_rope_cos, host_rope_cos.data(), rope_elems * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(H2D rope_cos)",
        error_message) ||
      !check_cuda(
        cudaMemcpy(state.pf_rope_sin, host_rope_sin.data(), rope_elems * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(H2D rope_sin)",
        error_message)) {
    return false;
  }

  const int prefill_s = config.max_context;
  const int prefill_mlp_chunk_tokens = get_prefill_mlp_chunk_tokens(config.max_context);
  const int prefill_attention_query_tokens = get_prefill_attention_query_tokens(variant, config.max_context);
  const std::size_t full_projection_elems =
    static_cast<std::size_t>(prefill_s) * static_cast<std::size_t>(std::max(descriptor_dn_conv_channels(descriptor), descriptor_fa_qproj_size(descriptor)));
  const std::size_t chunked_mlp_elems =
    static_cast<std::size_t>(prefill_mlp_chunk_tokens) * static_cast<std::size_t>(descriptor.intermediate_size);
  const std::size_t attention_prob_elems =
    static_cast<std::size_t>(prefill_attention_query_tokens) * static_cast<std::size_t>(prefill_s);
  const std::size_t proj_buf_elems = std::max(full_projection_elems, chunked_mlp_elems);
  const std::size_t proj_buf2_elems =
    std::max(
      static_cast<std::size_t>(prefill_s) * static_cast<std::size_t>(std::max(descriptor_dn_v_size(descriptor), descriptor_fa_kv_size(descriptor))),
      chunked_mlp_elems);
  const std::size_t mlp_buf_elems = std::max(chunked_mlp_elems, attention_prob_elems);
  const std::size_t attention_score_elems =
    static_cast<std::size_t>(prefill_attention_query_tokens) * static_cast<std::size_t>(prefill_s);
  const std::size_t attention_accum_elems =
    static_cast<std::size_t>(prefill_attention_query_tokens) * static_cast<std::size_t>(descriptor.fa_head_dim);
  const std::size_t dn_qkv_f32_elems =
    std::max(
      static_cast<std::size_t>(prefill_mlp_chunk_tokens) * static_cast<std::size_t>(descriptor_dn_conv_channels(descriptor)),
      attention_score_elems);
  const std::size_t dn_out_f32_elems =
    std::max(
      static_cast<std::size_t>(prefill_mlp_chunk_tokens) * static_cast<std::size_t>(descriptor_dn_v_size(descriptor)),
      attention_accum_elems);

  if (!arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * descriptor.hidden_size * bf16_bytes, state.pf_hidden, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * descriptor.hidden_size * bf16_bytes, state.pf_residual, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * descriptor.hidden_size * bf16_bytes, state.pf_normalized, error_message) ||
      !arena.alloc_bytes(proj_buf_elems * bf16_bytes, state.pf_proj_buf, error_message) ||
      !arena.alloc_bytes(proj_buf2_elems * bf16_bytes, state.pf_proj_buf2, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * std::max(descriptor_fa_q_size(descriptor), descriptor_fa_kv_size(descriptor)) * bf16_bytes, state.pf_attn_buf, error_message) ||
      !arena.alloc_bytes(mlp_buf_elems * bf16_bytes, state.pf_mlp_buf, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * descriptor_dn_v_size(descriptor) * bf16_bytes, state.pf_dn_out_buf, error_message) ||
      !arena.alloc_bytes(dn_qkv_f32_elems * sizeof(float), reinterpret_cast<void *&>(state.pf_dn_qkv_f32), error_message) ||
      !arena.alloc_bytes(dn_out_f32_elems * sizeof(float), reinterpret_cast<void *&>(state.pf_dn_out_f32), error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_mlp_chunk_tokens) * descriptor.dn_gate_heads * sizeof(float), reinterpret_cast<void *&>(state.pf_beta_buf), error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_mlp_chunk_tokens) * descriptor.dn_gate_heads * sizeof(float), reinterpret_cast<void *&>(state.pf_alpha_buf), error_message) ||
      !arena.alloc_bytes(descriptor.hidden_size * f32_bytes, state.pf_final_normed, error_message) ||
      !arena.alloc_bytes(descriptor.hidden_size * bf16_bytes, state.pf_hidden_bf16_out, error_message) ||
      !arena.alloc_bytes(1024 * sizeof(float), reinterpret_cast<void *&>(state.pf_lm_bmv), error_message) ||
      !arena.alloc_bytes(1024 * sizeof(int), reinterpret_cast<void *&>(state.pf_lm_bmi), error_message)) {
    return false;
  }
  state.prefill_mlp_chunk_tokens = prefill_mlp_chunk_tokens;
  state.prefill_attention_query_tokens = prefill_attention_query_tokens;

  return true;
}

std::vector<std::pair<std::string, std::vector<std::int64_t>>> expected_nvfp4_module_shapes(const Qwen35xModelDescriptor & descriptor) {
  std::vector<std::pair<std::string, std::vector<std::int64_t>>> modules;
  const std::int64_t hidden = descriptor.hidden_size;
  const std::int64_t intermediate = descriptor.intermediate_size;
  const std::int64_t full_q_out = descriptor_fa_q_size(descriptor);
  const std::int64_t full_qproj_out = descriptor_fa_qproj_size(descriptor);
  const std::int64_t full_kv_out = descriptor_fa_kv_size(descriptor);
  const std::int64_t linear_conv_channels = descriptor_dn_conv_channels(descriptor);
  const std::int64_t linear_v_dim = descriptor_dn_v_size(descriptor);

  for (int layer_idx = 0; layer_idx < descriptor.num_layers; ++layer_idx) {
    const std::string base = "model.language_model.layers." + std::to_string(layer_idx) + ".";
    modules.push_back({base + "mlp.gate_proj", {intermediate, hidden}});
    modules.push_back({base + "mlp.up_proj", {intermediate, hidden}});
    modules.push_back({base + "mlp.down_proj", {hidden, intermediate}});

    const int layer_type = descriptor.layer_type[static_cast<std::size_t>(layer_idx)];
    if (layer_type == 0) {
      modules.push_back({base + "linear_attn.in_proj_qkv", {linear_conv_channels, hidden}});
      modules.push_back({base + "linear_attn.in_proj_z", {linear_v_dim, hidden}});
      modules.push_back({base + "linear_attn.in_proj_b", {descriptor.dn_gate_heads, hidden}});
      modules.push_back({base + "linear_attn.in_proj_a", {descriptor.dn_gate_heads, hidden}});
      modules.push_back({base + "linear_attn.out_proj", {hidden, linear_v_dim}});
    } else {
      modules.push_back({base + "self_attn.q_proj", {full_qproj_out, hidden}});
      modules.push_back({base + "self_attn.k_proj", {full_kv_out, hidden}});
      modules.push_back({base + "self_attn.v_proj", {full_kv_out, hidden}});
      modules.push_back({base + "self_attn.o_proj", {hidden, full_q_out}});
    }
  }
  return modules;
}

bool load_modelopt_nvfp4_state(
  const Qwen35xModelDescriptor & descriptor,
  const Qwen35xCudaBackendConfig & config,
  DeviceArena & arena,
  Nvfp4BackendState & state,
  std::string & error_message) {
  std::string profile_error;
  const auto profile = qwen35x::ProfileLoader::load_from_hf_directory(config.model_dir, profile_error);
  if (!profile) {
    error_message = "failed to load model profile for ModelOpt NVFP4 validation: " + profile_error;
    return false;
  }
  qwen35x::ModelOptNvfp4ValidationOptions validation_options;
  validation_options.model_dir = config.model_dir;
  qwen35x::ModelOptNvfp4ValidationResult validation_result;
  if (!qwen35x::validate_modelopt_nvfp4_checkpoint(*profile, validation_options, validation_result, error_message)) {
    return false;
  }

  constexpr std::int64_t group_size = 16;
  const auto modules = expected_nvfp4_module_shapes(descriptor);
  state.tensors.clear();
  state.tensors.reserve(modules.size());
  for (const auto & module : modules) {
    const std::string & base_name = module.first;
    const auto & source_shape = module.second;
    if (source_shape.size() != 2 || (source_shape[1] % group_size) != 0) {
      error_message = "Invalid source shape for ModelOpt NVFP4 module '" + base_name + "'.";
      return false;
    }

    const std::vector<std::int64_t> packed_shape{source_shape[0], source_shape[1] / 2};
    const std::vector<std::int64_t> scale_shape{source_shape[0], source_shape[1] / group_size};
    Nvfp4TensorDevice device_tensor;
    std::vector<std::uint8_t> packed_weight;
    std::vector<std::uint8_t> weight_scale;
    float input_scale = 0.0f;
    float weight_scale_2 = 0.0f;
    if (!read_raw_tensor_from_model_dir(config.model_dir, base_name + ".weight", "U8", packed_shape, packed_weight, error_message) ||
        !read_raw_tensor_from_model_dir(config.model_dir, base_name + ".weight_scale", "F8_E4M3", scale_shape, weight_scale, error_message) ||
        !read_scalar_f32_from_model_dir(config.model_dir, base_name + ".input_scale", input_scale, error_message) ||
        !read_scalar_f32_from_model_dir(config.model_dir, base_name + ".weight_scale_2", weight_scale_2, error_message)) {
      error_message = "ModelOpt NVFP4 load failed for '" + base_name + "': " + error_message;
      return false;
    }

    int padded_weight_rows = 0;
    int padded_weight_cols = 0;
    int weight_padding_cols = 0;
    std::vector<std::uint8_t> tc_packed_weight = pad_nvfp4_weight_for_tensor_core(
      packed_weight,
      static_cast<int>(source_shape[0]),
      static_cast<int>(source_shape[1]),
      padded_weight_rows,
      padded_weight_cols,
      weight_padding_cols);
    (void)padded_weight_cols;

    int padded_scale_rows = 0;
    int padded_scale_cols = 0;
    std::vector<std::uint8_t> tc_weight_scale = swizzle_nvfp4_block_scale_for_tensor_core(
      weight_scale,
      static_cast<int>(source_shape[0]),
      static_cast<int>(source_shape[1] / group_size),
      padded_scale_rows,
      padded_scale_cols);
    (void)padded_scale_rows;

    int sm120_row_tiles = 0;
    int sm120_k_blocks = 0;
    std::vector<std::uint32_t> sm120_packed_weight_fragments = pack_nvfp4_weight_for_sm120_mma(
      packed_weight,
      static_cast<int>(source_shape[0]),
      static_cast<int>(source_shape[1]),
      sm120_row_tiles,
      sm120_k_blocks);
    std::vector<std::uint32_t> sm120_weight_scale_fragments = pack_nvfp4_weight_scales_for_sm120_mma(
      weight_scale,
      static_cast<int>(source_shape[0]),
      static_cast<int>(source_shape[1] / group_size),
      sm120_row_tiles,
      sm120_k_blocks);

    const float alpha = input_scale * weight_scale_2;
    if (!upload_bytes_to_device(packed_weight, "raw " + base_name + ".weight", arena, device_tensor.packed_weight, error_message) ||
        !upload_bytes_to_device(weight_scale, "raw " + base_name + ".weight_scale", arena, device_tensor.weight_scale, error_message) ||
        !upload_scalar_to_device(input_scale, "scalar " + base_name + ".input_scale", arena, device_tensor.input_scale, error_message) ||
        !upload_scalar_to_device(weight_scale_2, "scalar " + base_name + ".weight_scale_2", arena, device_tensor.weight_scale_2, error_message) ||
        !upload_bytes_to_device(tc_packed_weight, "tensor-core " + base_name + ".weight", arena, device_tensor.tc_packed_weight, error_message) ||
        !upload_bytes_to_device(tc_weight_scale, "tensor-core " + base_name + ".weight_scale", arena, device_tensor.tc_weight_scale, error_message) ||
        !upload_scalar_to_device(alpha, "tensor-core " + base_name + ".alpha", arena, device_tensor.tc_alpha, error_message) ||
        !upload_u32_to_device(sm120_packed_weight_fragments, "sm120 " + base_name + ".weight_fragments", arena, device_tensor.sm120_packed_weight_fragments, error_message) ||
        !upload_u32_to_device(sm120_weight_scale_fragments, "sm120 " + base_name + ".weight_scale_fragments", arena, device_tensor.sm120_weight_scale_fragments, error_message)) {
      error_message = "ModelOpt NVFP4 load failed for '" + base_name + "': " + error_message;
      return false;
    }
    device_tensor.output_size = static_cast<int>(source_shape[0]);
    device_tensor.input_size = static_cast<int>(source_shape[1]);
    device_tensor.padded_output_size = padded_weight_rows;
    device_tensor.padded_scale_cols = padded_scale_cols;
    device_tensor.weight_padding_cols = weight_padding_cols;
    device_tensor.sm120_row_tiles = sm120_row_tiles;
    device_tensor.sm120_k_blocks = sm120_k_blocks;
    state.tensors.push_back(device_tensor);
  }

  if (static_cast<int>(state.tensors.size()) != validation_result.quantized_tensors) {
    error_message =
      "ModelOpt NVFP4 loaded tensor count mismatch: loaded " + std::to_string(state.tensors.size()) +
      " validation reported " + std::to_string(validation_result.quantized_tensors) + ".";
    return false;
  }
  return true;
}

bool initialize_nvfp4_layer_weights(
  const Qwen35xModelDescriptor & descriptor,
  DeviceArena & arena,
  const Nvfp4BackendState & nvfp4_state,
  BackendState & state,
  std::string & error_message) {
  std::vector<PackedLayerNvfp4Weights> host_layers(static_cast<std::size_t>(descriptor.num_layers));
  std::size_t tensor_idx = 0;
  auto attach = [&](PackedLayerNvfp4Weights & layer, int ptr_idx) -> bool {
    if (tensor_idx >= nvfp4_state.tensors.size()) {
      error_message = "ModelOpt NVFP4 layer table underflow while assigning ptr[" + std::to_string(ptr_idx) + "].";
      return false;
    }
    const auto & tensor = nvfp4_state.tensors[tensor_idx++];
    layer.ptrs[ptr_idx].packed_weight = tensor.packed_weight;
    layer.ptrs[ptr_idx].weight_scale = tensor.weight_scale;
    layer.ptrs[ptr_idx].weight_scale_2 = tensor.weight_scale_2;
    layer.ptrs[ptr_idx].tc_packed_weight = tensor.tc_packed_weight;
    layer.ptrs[ptr_idx].tc_weight_scale = tensor.tc_weight_scale;
    layer.ptrs[ptr_idx].tc_alpha = tensor.tc_alpha;
    layer.ptrs[ptr_idx].sm120_packed_weight_fragments = tensor.sm120_packed_weight_fragments;
    layer.ptrs[ptr_idx].sm120_weight_scale_fragments = tensor.sm120_weight_scale_fragments;
    layer.ptrs[ptr_idx].output_size = tensor.output_size;
    layer.ptrs[ptr_idx].input_size = tensor.input_size;
    layer.ptrs[ptr_idx].padded_output_size = tensor.padded_output_size;
    layer.ptrs[ptr_idx].padded_scale_cols = tensor.padded_scale_cols;
    layer.ptrs[ptr_idx].weight_padding_cols = tensor.weight_padding_cols;
    layer.ptrs[ptr_idx].sm120_row_tiles = tensor.sm120_row_tiles;
    layer.ptrs[ptr_idx].sm120_k_blocks = tensor.sm120_k_blocks;
    return true;
  };

  for (int layer_idx = 0; layer_idx < descriptor.num_layers; ++layer_idx) {
    auto & layer = host_layers[static_cast<std::size_t>(layer_idx)];
    layer.layer_type = descriptor.layer_type[static_cast<std::size_t>(layer_idx)];
    if (layer.layer_type == 0) {
      if (!attach(layer, 11) || !attach(layer, 12) || !attach(layer, 13) ||
          !attach(layer, 1) || !attach(layer, 2) || !attach(layer, 3) || !attach(layer, 4) ||
          !attach(layer, 9)) {
        error_message = "layer " + std::to_string(layer_idx) + ": " + error_message;
        return false;
      }
    } else {
      if (!attach(layer, 8) || !attach(layer, 9) || !attach(layer, 10) ||
          !attach(layer, 1) || !attach(layer, 2) || !attach(layer, 3) ||
          !attach(layer, 6)) {
        error_message = "layer " + std::to_string(layer_idx) + ": " + error_message;
        return false;
      }
    }
  }

  if (tensor_idx != nvfp4_state.tensors.size()) {
    error_message =
      "ModelOpt NVFP4 layer table overflow: assigned " + std::to_string(tensor_idx) +
      " tensors but loaded " + std::to_string(nvfp4_state.tensors.size()) + ".";
    return false;
  }

  if (!arena.alloc_bytes(
        host_layers.size() * sizeof(PackedLayerNvfp4Weights),
        reinterpret_cast<void *&>(state.layer_nvfp4_weights),
        error_message)) {
    return false;
  }
  return check_cuda(
    cudaMemcpy(
      state.layer_nvfp4_weights,
      host_layers.data(),
      host_layers.size() * sizeof(PackedLayerNvfp4Weights),
      cudaMemcpyHostToDevice),
    "cudaMemcpy(H2D layer_nvfp4_weights)",
    error_message);
}

bool validate_nvfp4_tensor_core_projection(
  const Nvfp4BackendState & nvfp4_state,
  const BackendState & state,
  std::string & error_message) {
  if (nvfp4_state.tensors.empty()) {
    error_message = "No NVFP4 tensors are loaded for tensor-core projection validation.";
    return false;
  }
  const auto & tensor = nvfp4_state.tensors.front();
  if (tensor.tc_packed_weight == nullptr || tensor.tc_weight_scale == nullptr || tensor.weight_scale_2 == nullptr ||
      tensor.input_size <= 0 || tensor.output_size <= 0) {
    error_message = "First NVFP4 tensor is missing tensor-core projection buffers.";
    return false;
  }

  std::vector<float> host_input(static_cast<std::size_t>(tensor.input_size));
  for (int col = 0; col < tensor.input_size; ++col) {
    host_input[static_cast<std::size_t>(col)] = static_cast<float>((col % 17) - 8) / 17.0f;
  }

  float weight_scale_2 = 0.0f;
  double elapsed_ms = 0.0;
  if (!check_cuda(
        cudaMemcpy(state.fp4_projection_input_f32, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(H2D fp4 projection validation input)",
        error_message) ||
      !check_cuda(
        cudaMemcpy(&weight_scale_2, tensor.weight_scale_2, sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy(D2H fp4 projection weight_scale_2)",
        error_message)) {
    return false;
  }

  return qwen35x::cuda::run_nvfp4_cublaslt_projection_device(
    state.fp4_projection_input_f32,
    static_cast<const std::uint8_t *>(tensor.tc_packed_weight),
    static_cast<const std::uint8_t *>(tensor.tc_weight_scale),
    weight_scale_2,
    tensor.output_size,
    tensor.input_size,
    state.fp4_activation,
    state.fp4_activation_scales,
    state.fp4_projection_output_f32,
    &elapsed_ms,
    error_message);
}

bool validate_nvfp4_gate_up_projection(
  const Nvfp4BackendState & nvfp4_state,
  const BackendState & state,
  std::string & error_message) {
  if (nvfp4_state.tensors.size() < 2) {
    error_message = "Need at least gate/up NVFP4 tensors for gate-up projection validation.";
    return false;
  }
  const auto & gate = nvfp4_state.tensors[0];
  const auto & up = nvfp4_state.tensors[1];
  if (gate.input_size != up.input_size || gate.output_size != up.output_size ||
      gate.tc_packed_weight == nullptr || gate.tc_weight_scale == nullptr ||
      up.tc_packed_weight == nullptr || up.tc_weight_scale == nullptr) {
    error_message = "Gate/up NVFP4 tensor-core projection buffers are incompatible.";
    return false;
  }

  std::vector<float> host_input(static_cast<std::size_t>(gate.input_size));
  for (int col = 0; col < gate.input_size; ++col) {
    host_input[static_cast<std::size_t>(col)] = static_cast<float>((col % 17) - 8) / 17.0f;
  }

  float gate_scale_2 = 0.0f;
  float up_scale_2 = 0.0f;
  double elapsed_ms = 0.0;
  if (!check_cuda(
        cudaMemcpy(state.fp4_projection_input_f32, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(H2D fp4 gate-up validation input)",
        error_message) ||
      !check_cuda(
        cudaMemcpy(&gate_scale_2, gate.weight_scale_2, sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy(D2H fp4 gate weight_scale_2)",
        error_message) ||
      !check_cuda(
        cudaMemcpy(&up_scale_2, up.weight_scale_2, sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy(D2H fp4 up weight_scale_2)",
        error_message)) {
    return false;
  }

  return qwen35x::cuda::run_nvfp4_cublaslt_gate_up_silu_device(
    state.fp4_projection_input_f32,
    static_cast<const std::uint8_t *>(gate.tc_packed_weight),
    static_cast<const std::uint8_t *>(gate.tc_weight_scale),
    gate_scale_2,
    static_cast<const std::uint8_t *>(up.tc_packed_weight),
    static_cast<const std::uint8_t *>(up.tc_weight_scale),
    up_scale_2,
    gate.output_size,
    gate.input_size,
    state.fp4_activation,
    state.fp4_activation_scales,
    state.fp4_projection_output_f32,
    static_cast<float *>(state.g_activations),
    &elapsed_ms,
    error_message);
}

bool reset_state(const Qwen35xModelDescriptor & descriptor, const BackendState & state, int max_seq_len, std::string & error_message) {
  const int n_full_layers = descriptor_full_layer_count(descriptor);
  const int n_delta_layers = descriptor_delta_layer_count(descriptor);

  const std::size_t fa_bytes = static_cast<std::size_t>(n_full_layers) * descriptor.fa_num_kv_heads * max_seq_len * descriptor.fa_head_dim * sizeof(std::uint16_t);
  const std::size_t dn_bytes = static_cast<std::size_t>(n_delta_layers) * descriptor.dn_num_heads * descriptor.dn_key_dim * descriptor.dn_value_dim * sizeof(float);
  const std::size_t conv_bytes = static_cast<std::size_t>(n_delta_layers) * descriptor_dn_conv_channels(descriptor) * descriptor.dn_conv_kernel * sizeof(float);
  const std::size_t seen_bytes = static_cast<std::size_t>(descriptor.vocab_size) * sizeof(float);

  return check_cuda(cudaMemset(state.fa_k_cache, 0, fa_bytes), "cudaMemset(fa_k_cache)", error_message) &&
         check_cuda(cudaMemset(state.fa_v_cache, 0, fa_bytes), "cudaMemset(fa_v_cache)", error_message) &&
         check_cuda(cudaMemset(state.dn_states, 0, dn_bytes), "cudaMemset(dn_states)", error_message) &&
         check_cuda(cudaMemset(state.conv_bufs, 0, conv_bytes), "cudaMemset(conv_bufs)", error_message) &&
         check_cuda(cudaMemset(state.seen_token_mask, 0, seen_bytes), "cudaMemset(seen_token_mask)", error_message);
}

void launch_prefill_for_state(
  const VariantDescriptor & variant,
  const BackendState & state,
  const Qwen35xCudaBackendConfig & config,
  int seq_len,
  bool compute_first_token,
  Qwen35xPrefillProfile * profile,
  cudaStream_t stream) {
  variant.launch_prefill(
    state.token_ids,
    seq_len,
    state.output_token,
    state.embed_weight,
    state.layer_weights,
    state.final_norm_weight,
    state.lm_head_weight,
    state.fa_k_cache,
    state.fa_v_cache,
    state.dn_states,
    state.conv_bufs,
    state.pf_hidden,
    state.pf_residual,
    state.pf_normalized,
    state.pf_proj_buf,
    state.pf_proj_buf2,
    state.pf_attn_buf,
    state.pf_mlp_buf,
    state.pf_dn_out_buf,
    state.pf_dn_qkv_f32,
    state.pf_dn_out_f32,
    state.pf_beta_buf,
    state.pf_alpha_buf,
    state.pf_final_normed,
    state.pf_hidden_bf16_out,
    state.pf_lm_bmv,
    state.pf_lm_bmi,
    state.seen_token_mask,
    config.repetition_penalty,
    state.pf_rope_cos,
    state.pf_rope_sin,
    config.max_context,
    state.prefill_mlp_chunk_tokens,
    state.prefill_attention_query_tokens,
    compute_first_token ? 1 : 0,
    profile,
    stream);
}

bool warmup_prefill_backend(
  const Qwen35xModelDescriptor & descriptor,
  const VariantDescriptor & variant,
  BackendState & state,
  const Qwen35xCudaBackendConfig & config,
  std::string & error_message) {
  if (!check_cuda(
        cudaMemset(state.token_ids, 0, static_cast<std::size_t>(config.max_context) * sizeof(int)),
        "cudaMemset(prefill warmup token_ids)",
        error_message)) {
    return false;
  }

  launch_prefill_for_state(variant, state, config, 1, false, nullptr, nullptr);
  if (!check_cuda(cudaGetLastError(), "warmup launch_prefill_bf16", error_message) ||
      !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(prefill warmup)", error_message)) {
    return false;
  }

  return reset_state(descriptor, state, config.max_context, error_message);
}

bool run_prefill_impl(
  const Qwen35xModelDescriptor & descriptor,
  const VariantDescriptor & variant,
  BackendState & state,
  const Qwen35xCudaBackendConfig & config,
  const std::vector<std::int32_t> & tokens,
  int & out_first_token,
  bool compute_first_token,
  Qwen35xPrefillProfile * profile,
  std::string & error_message) {
  const auto host_start = std::chrono::steady_clock::now();
  if (profile != nullptr) {
    *profile = Qwen35xPrefillProfile{};
    profile->enabled = true;
  }

  if (tokens.empty()) {
    error_message = "Prefill tokens are empty.";
    return false;
  }
  if (tokens.size() > static_cast<std::size_t>(config.max_context)) {
    error_message = "Prefill token count exceeds max_context.";
    return false;
  }

  const auto upload_start = std::chrono::steady_clock::now();
  if (!check_cuda(
        cudaMemcpy(
          state.token_ids,
          tokens.data(),
          tokens.size() * sizeof(std::int32_t),
          cudaMemcpyHostToDevice),
        "cudaMemcpy(H2D token_ids)",
        error_message)) {
    return false;
  }
  if (profile != nullptr) {
    profile->token_upload_ms += elapsed_ms_since(upload_start);
  }

  const int seq_len = static_cast<int>(tokens.size());
  launch_prefill_for_state(variant, state, config, seq_len, compute_first_token, profile, nullptr);
  if (!check_cuda(cudaGetLastError(), "launch_prefill_bf16", error_message)) {
    return false;
  }

  if (!compute_first_token) {
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(prefill_only)", error_message)) {
      return false;
    }
    if (profile != nullptr) {
      profile->host_total_ms = elapsed_ms_since(host_start);
    }
    return true;
  }

  const auto handoff_start = std::chrono::steady_clock::now();
  if (!check_cuda(
        cudaMemcpy(state.hidden_buffer, state.pf_hidden_bf16_out, descriptor.hidden_size * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice),
        "cudaMemcpy(D2D hidden handoff)",
        error_message)) {
    return false;
  }
  if (profile != nullptr) {
    profile->hidden_handoff_ms += elapsed_ms_since(handoff_start);
  }

  const auto token_download_start = std::chrono::steady_clock::now();
  if (!check_cuda(
        cudaMemcpy(&out_first_token, state.output_token, sizeof(int), cudaMemcpyDeviceToHost),
        "cudaMemcpy(D2H first_token)",
        error_message)) {
    return false;
  }
  if (profile != nullptr) {
    profile->output_token_download_ms += elapsed_ms_since(token_download_start);
    profile->host_total_ms = elapsed_ms_since(host_start);
  }

  return true;
}

bool run_decode_step_impl(
  const Qwen35xModelDescriptor & descriptor,
  const VariantDescriptor & variant,
  const BackendState & state,
  const Nvfp4BackendState * nvfp4_state,
  int input_token,
  int position,
  int max_seq_len,
  float repetition_penalty,
  Qwen35xDecodeProfile * profile,
  int & out_next_token,
  std::string & error_message) {
  const auto host_start = std::chrono::steady_clock::now();
  if (profile != nullptr) {
    profile->enabled = true;
    profile->steps += 1;
    profile->last_position = position;
  }

  if (position < 0 || position >= max_seq_len) {
    error_message = "Decode position " + std::to_string(position) + " exceeds configured max_context " +
                    std::to_string(max_seq_len) + ".";
    return false;
  }

  if (repetition_penalty > 1.0f && input_token >= 0 && input_token < descriptor.vocab_size) {
    const float seen = 1.0f;
    const auto seen_upload_start = std::chrono::steady_clock::now();
    if (!check_cuda(
          cudaMemcpy(
            state.seen_token_mask + input_token,
            &seen,
            sizeof(float),
            cudaMemcpyHostToDevice),
          "cudaMemcpy(H2D seen_token_mask)",
          error_message)) {
      return false;
    }
    if (profile != nullptr) {
      profile->seen_token_upload_ms += elapsed_ms_since(seen_upload_start);
    }
  }

  variant.launch_decode(
    input_token,
    state.output_token,
    state.embed_weight,
    state.layer_weights,
    state.layer_nvfp4_weights,
    state.final_norm_weight,
    state.lm_head_weight,
    state.fa_k_cache,
    state.fa_v_cache,
    state.dn_states,
    state.conv_bufs,
    state.hidden_buffer,
    state.g_activations,
    state.g_residual,
    state.g_qkv_scratch,
    state.g_kv_scratch,
    state.g_attn_out,
    state.g_attn_partials,
    state.g_mlp_inter,
    state.g_z_scratch,
    state.g_beta_scratch,
    state.g_alpha_scratch,
    state.g_normalized,
    state.barrier_counter,
    state.barrier_generation,
    state.block_max_vals,
    state.block_max_idxs,
    state.lm_sync_counter,
    state.seen_token_mask,
    repetition_penalty,
    position,
    max_seq_len,
    profile,
    nullptr);

  if (!check_cuda(cudaGetLastError(), "launch_decode", error_message)) {
    return false;
  }

  if (std::getenv("QWEN35X_DRY_RUN_FP4_DECODE_PROJECTION") != nullptr &&
      nvfp4_state != nullptr &&
      !nvfp4_state->tensors.empty()) {
    const auto & tensor = nvfp4_state->tensors.front();
    float weight_scale_2 = 0.0f;
    double elapsed_ms = 0.0;
    if (!check_cuda(
          cudaMemcpy(&weight_scale_2, tensor.weight_scale_2, sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy(D2H dry-run fp4 projection weight_scale_2)",
          error_message) ||
        !qwen35x::cuda::run_nvfp4_cublaslt_projection_device(
          static_cast<const float *>(state.g_normalized),
          static_cast<const std::uint8_t *>(tensor.tc_packed_weight),
          static_cast<const std::uint8_t *>(tensor.tc_weight_scale),
          weight_scale_2,
          tensor.output_size,
          tensor.input_size,
          state.fp4_activation,
          state.fp4_activation_scales,
          state.fp4_projection_output_f32,
          &elapsed_ms,
          error_message)) {
      error_message = "dry-run decode FP4 projection failed: " + error_message;
      return false;
    }
  }

  const auto token_download_start = std::chrono::steady_clock::now();
  if (!check_cuda(
        cudaMemcpy(&out_next_token, state.output_token, sizeof(int), cudaMemcpyDeviceToHost),
        "cudaMemcpy(D2H next_token)",
        error_message)) {
    return false;
  }
  if (profile != nullptr) {
    profile->output_token_download_ms += elapsed_ms_since(token_download_start);
    profile->host_total_ms += elapsed_ms_since(host_start);
  }
  return true;
}

} // namespace

struct Qwen35xCudaBackend::Impl {
  Qwen35xCudaBackendConfig config;
  Qwen35xModelDescriptor descriptor;
  const VariantDescriptor * variant = nullptr;
  DeviceArena arena;
  BackendState state;
  Nvfp4BackendState nvfp4_state;
  Qwen35xRuntimeProfile profile;
  bool initialized = false;
};

Qwen35xCudaBackend::Qwen35xCudaBackend() : impl_(std::make_unique<Impl>()) {
}

Qwen35xCudaBackend::~Qwen35xCudaBackend() = default;

Qwen35xCudaBackend::Qwen35xCudaBackend(Qwen35xCudaBackend &&) noexcept = default;
Qwen35xCudaBackend & Qwen35xCudaBackend::operator=(Qwen35xCudaBackend &&) noexcept = default;

bool Qwen35xCudaBackend::initialize(const Qwen35xCudaBackendConfig & config, std::string & error_message) {
  if (config.max_context <= 0) {
    error_message = "max_context must be > 0.";
    return false;
  }
  if (config.decode_blocks < 0) {
    error_message = "decode_blocks must be >= 0.";
    return false;
  }
  if (config.repetition_penalty < 1.0f) {
    error_message = "repetition_penalty must be >= 1.0.";
    return false;
  }
  if (config.cache_precision != Qwen35xCachePrecision::bf16) {
    error_message =
      "Qwen35x CUDA cache precision '" + std::string(to_string(config.cache_precision)) +
      "' is not implemented yet; supported cache_precision=bf16.";
    return false;
  }
  Qwen35xModelDescriptor descriptor;
  if (!resolve_model_descriptor(config, descriptor, error_message)) {
    return false;
  }
  const VariantDescriptor * variant = select_variant_for_descriptor(descriptor, error_message);
  if (variant == nullptr) {
    return false;
  }

  impl_ = std::make_unique<Impl>();
  impl_->config = config;
  impl_->descriptor = descriptor;
  impl_->variant = variant;
  variant->set_decode_blocks_override(config.decode_blocks);

  if (config.weight_precision != Qwen35xWeightPrecision::bf16) {
    if (config.weight_precision != Qwen35xWeightPrecision::nvfp4) {
      error_message =
        "Qwen35x CUDA weight precision '" + std::string(to_string(config.weight_precision)) +
        "' is not implemented yet; supported weight_precision=bf16|nvfp4.";
      impl_ = std::make_unique<Impl>();
      return false;
    }
    if (!load_modelopt_nvfp4_state(impl_->descriptor, config, impl_->arena, impl_->nvfp4_state, error_message)) {
      impl_ = std::make_unique<Impl>();
      return false;
    }
  }

  if (!initialize_backend_state(impl_->descriptor, *variant, config, impl_->arena, impl_->state, error_message)) {
    impl_ = std::make_unique<Impl>();
    return false;
  }
  if (config.weight_precision == Qwen35xWeightPrecision::nvfp4 &&
      !initialize_nvfp4_layer_weights(impl_->descriptor, impl_->arena, impl_->nvfp4_state, impl_->state, error_message)) {
    impl_ = std::make_unique<Impl>();
    return false;
  }
  if (config.weight_precision == Qwen35xWeightPrecision::nvfp4 &&
      std::getenv("QWEN35X_VALIDATE_FP4_PROJECTION") != nullptr &&
      !validate_nvfp4_tensor_core_projection(impl_->nvfp4_state, impl_->state, error_message)) {
    error_message = "NVFP4 tensor-core projection validation failed: " + error_message;
    impl_ = std::make_unique<Impl>();
    return false;
  }
  if (config.weight_precision == Qwen35xWeightPrecision::nvfp4 &&
      std::getenv("QWEN35X_VALIDATE_FP4_GATE_UP") != nullptr &&
      !validate_nvfp4_gate_up_projection(impl_->nvfp4_state, impl_->state, error_message)) {
    error_message = "NVFP4 gate/up projection validation failed: " + error_message;
    impl_ = std::make_unique<Impl>();
    return false;
  }
  if (!warmup_prefill_backend(impl_->descriptor, *variant, impl_->state, config, error_message)) {
    impl_ = std::make_unique<Impl>();
    return false;
  }
  impl_->profile = Qwen35xRuntimeProfile{};
  impl_->profile.enabled = config.profile_enabled;
  impl_->initialized = true;
  return true;
}

bool Qwen35xCudaBackend::reset(std::string & error_message) {
  if (!is_initialized()) {
    error_message = "Qwen35x CUDA backend reset requested before initialize.";
    return false;
  }
  return reset_state(impl_->descriptor, impl_->state, impl_->config.max_context, error_message);
}

bool Qwen35xCudaBackend::run_prefill(
  const std::vector<std::int32_t> & tokens,
  int & out_first_token,
  std::string & error_message) {
  if (!is_initialized()) {
    error_message = "Qwen35x CUDA backend prefill requested before initialize.";
    return false;
  }
  if (static_cast<int>(tokens.size()) > impl_->config.max_context) {
    error_message = "Prefill token count exceeds configured max_context.";
    return false;
  }
  Qwen35xPrefillProfile * profile = impl_->config.profile_enabled ? &impl_->profile.prefill : nullptr;
  const bool ok = run_prefill_impl(impl_->descriptor, *impl_->variant, impl_->state, impl_->config, tokens, out_first_token, true, profile, error_message);
  if (ok && profile != nullptr) {
    impl_->profile.enabled = true;
    impl_->profile.prefill_runs += 1;
  }
  return ok;
}

bool Qwen35xCudaBackend::run_prefill_only(
  const std::vector<std::int32_t> & tokens,
  std::string & error_message) {
  if (!is_initialized()) {
    error_message = "Qwen35x CUDA backend prefill_only requested before initialize.";
    return false;
  }
  if (static_cast<int>(tokens.size()) > impl_->config.max_context) {
    error_message = "Prefill token count exceeds configured max_context.";
    return false;
  }
  int ignored_first_token = 0;
  Qwen35xPrefillProfile * profile = impl_->config.profile_enabled ? &impl_->profile.prefill : nullptr;
  const bool ok = run_prefill_impl(impl_->descriptor, *impl_->variant, impl_->state, impl_->config, tokens, ignored_first_token, false, profile, error_message);
  if (ok && profile != nullptr) {
    impl_->profile.enabled = true;
    impl_->profile.prefill_runs += 1;
  }
  return ok;
}

bool Qwen35xCudaBackend::run_decode_step(
  int input_token,
  int position,
  int & out_next_token,
  std::string & error_message) {
  if (!is_initialized()) {
    error_message = "Qwen35x CUDA backend decode_step requested before initialize.";
    return false;
  }
  return run_decode_step_impl(
    impl_->descriptor,
    *impl_->variant,
    impl_->state,
    impl_->config.weight_precision == Qwen35xWeightPrecision::nvfp4 ? &impl_->nvfp4_state : nullptr,
    input_token,
    position,
    impl_->config.max_context,
    impl_->config.repetition_penalty,
    impl_->config.profile_enabled ? &impl_->profile.decode : nullptr,
    out_next_token,
    error_message);
}

bool Qwen35xCudaBackend::synchronize(std::string & error_message) {
  if (!is_initialized()) {
    error_message = "Qwen35x CUDA backend synchronize requested before initialize.";
    return false;
  }
  return check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", error_message);
}

bool Qwen35xCudaBackend::is_initialized() const {
  return impl_ != nullptr && impl_->initialized;
}

int Qwen35xCudaBackend::max_context() const {
  if (impl_ == nullptr) {
    return 0;
  }
  return impl_->config.max_context;
}

Qwen35xRuntimeProfile Qwen35xCudaBackend::profile() const {
  if (impl_ == nullptr) {
    return Qwen35xRuntimeProfile{};
  }
  return impl_->profile;
}

int query_max_safe_decode_blocks() {
  int result = 0;
  for (const auto * variant : kVariants) {
    result = std::max(result, variant->query_max_safe_decode_blocks());
  }
  return result;
}

void set_decode_blocks_override(int blocks) {
  for (const auto * variant : kVariants) {
    variant->set_decode_blocks_override(blocks);
  }
}

#else

struct Qwen35xCudaBackend::Impl {
  Qwen35xCudaBackendConfig config;
  bool initialized = false;
};

Qwen35xCudaBackend::Qwen35xCudaBackend() : impl_(std::make_unique<Impl>()) {
}

Qwen35xCudaBackend::~Qwen35xCudaBackend() = default;

Qwen35xCudaBackend::Qwen35xCudaBackend(Qwen35xCudaBackend &&) noexcept = default;
Qwen35xCudaBackend & Qwen35xCudaBackend::operator=(Qwen35xCudaBackend &&) noexcept = default;

bool Qwen35xCudaBackend::initialize(const Qwen35xCudaBackendConfig &, std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  impl_->initialized = false;
  return false;
}

bool Qwen35xCudaBackend::reset(std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool Qwen35xCudaBackend::run_prefill(
  const std::vector<std::int32_t> &,
  int &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool Qwen35xCudaBackend::run_prefill_only(
  const std::vector<std::int32_t> &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool Qwen35xCudaBackend::run_decode_step(
  int,
  int,
  int &,
  std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool Qwen35xCudaBackend::synchronize(std::string & error_message) {
  error_message = "CUDA is not enabled in this build.";
  return false;
}

bool Qwen35xCudaBackend::is_initialized() const {
  return false;
}

int Qwen35xCudaBackend::max_context() const {
  return 0;
}

Qwen35xRuntimeProfile Qwen35xCudaBackend::profile() const {
  return Qwen35xRuntimeProfile{};
}

int query_max_safe_decode_blocks() {
  return 0;
}

void set_decode_blocks_override(int) {
}

#endif

} // namespace qwen35x::cuda_backend
