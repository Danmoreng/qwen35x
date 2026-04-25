#include "qwen35x/runtime/qwen35x_cuda_backend.h"

#include "qwen35x/compiler/compiler.h"
#include "qwen35x/weights/safetensors.h"

#if QWEN35X_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <utility>

extern "C" void set_decode_blocks_override(int blocks);
extern "C" int query_max_safe_decode_blocks();

namespace qwen35x::cuda_backend {

#if QWEN35X_HAS_CUDA

namespace {

constexpr int kMaxDecodeBlocks = 1024;

#if defined(QWEN35X_CUDA_VARIANT_4b)
constexpr const char * kCompiledVariant = "4b";
constexpr int kNumLayers = 32;
constexpr int kHiddenSize = 2560;
constexpr int kIntermediateSize = 9216;
constexpr int kVocabSize = 248320;
constexpr int kFaNumQHeads = 16;
constexpr int kFaNumKvHeads = 4;
constexpr int kFaHeadDim = 256;
constexpr int kFaRotDim = 64;
constexpr int kDnNumHeads = 16;
constexpr int kDnGateHeads = 32;
constexpr int kDnKeyDim = 128;
constexpr int kDnValueDim = 256;
constexpr int kDnConvKernel = 4;
constexpr int kLayerType[kNumLayers] = {
  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1
};
#else
constexpr const char * kCompiledVariant = "0.8b";
constexpr int kNumLayers = 24;
constexpr int kHiddenSize = 1024;
constexpr int kIntermediateSize = 3584;
constexpr int kVocabSize = 248320;
constexpr int kFaNumQHeads = 8;
constexpr int kFaNumKvHeads = 2;
constexpr int kFaHeadDim = 256;
constexpr int kFaRotDim = 64;
constexpr int kDnNumHeads = 16;
constexpr int kDnGateHeads = 16;
constexpr int kDnKeyDim = 128;
constexpr int kDnValueDim = 128;
constexpr int kDnConvKernel = 4;
constexpr int kLayerType[kNumLayers] = {
  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1
};
#endif

constexpr int kFaGqaRatio = kFaNumQHeads / kFaNumKvHeads;
constexpr int kFaQSize = kFaNumQHeads * kFaHeadDim;
constexpr int kFaQprojSize = kFaQSize * 2;
constexpr int kFaKvSize = kFaNumKvHeads * kFaHeadDim;
constexpr int kDnQkSize = kDnNumHeads * kDnKeyDim;
constexpr int kDnVSize = kDnNumHeads * kDnValueDim;
constexpr int kDnConvChannels = kDnQkSize * 2 + kDnVSize;

struct PackedLayerWeights {
  int layer_type = 0;
  int pad[3] = {0, 0, 0};
  void * ptrs[14] = {};
};

extern "C" void launch_decode(
  int input_token_id,
  int * output_token_id,
  const void * embed_weight,
  const PackedLayerWeights * layer_weights,
  const void * final_norm_weight,
  const void * lm_head_weight,
  void * fa_k_cache,
  void * fa_v_cache,
  void * dn_states,
  void * conv_bufs,
  void * hidden_buffer,
  void * g_activations,
  void * g_residual,
  void * g_qkv_scratch,
  void * g_kv_scratch,
  void * g_attn_out,
  void * g_attn_partials,
  void * g_mlp_inter,
  void * g_z_scratch,
  void * g_beta_scratch,
  void * g_alpha_scratch,
  void * g_normalized,
  unsigned int * barrier_counter,
  unsigned int * barrier_generation,
  float * block_max_vals,
  int * block_max_idxs,
  unsigned int * lm_sync_counter,
  float * seen_token_mask,
  float repetition_penalty,
  int position,
  int max_seq_len,
  Qwen35xDecodeProfile * profile,
  cudaStream_t stream);

extern "C" void launch_prefill_bf16(
  const int * token_ids,
  int seq_len,
  int * output_token,
  const void * embed_weight,
  const PackedLayerWeights * layers,
  const void * final_norm_w,
  const void * lm_head_w,
  void * fa_k_cache,
  void * fa_v_cache,
  void * dn_states,
  void * conv_bufs,
  void * hidden,
  void * residual,
  void * normalized,
  void * proj_buf,
  void * proj_buf2,
  void * attn_buf,
  void * mlp_buf,
  void * dn_out_buf,
  float * dn_qkv_f32,
  float * dn_out_f32,
  void * beta_buf,
  void * alpha_buf,
  void * final_normed,
  void * hidden_bf16_out,
  void * lm_bmv,
  void * lm_bmi,
  float * seen_token_mask,
  float repetition_penalty,
  const float * rope_cos,
  const float * rope_sin,
  int max_seq_len,
  int compute_logits,
  Qwen35xPrefillProfile * profile,
  cudaStream_t stream);

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

bool load_bf16_tensor_to_device(
  const std::string & model_dir,
  const std::vector<std::string> & tensor_names,
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

    if (info.dtype == "BF16") {
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

bool alloc_and_zero(DeviceArena & arena, std::size_t bytes, void *& out_ptr, const std::string & label, std::string & error_message) {
  if (!arena.alloc_bytes(bytes, out_ptr, error_message)) {
    return false;
  }
  return check_cuda(cudaMemset(out_ptr, 0, bytes), "cudaMemset(" + label + ")", error_message);
}

bool validate_compiled_variant(const Qwen35xCudaBackendConfig & config, std::string & error_message) {
  std::string profile_error;
  const auto profile = qwen35x::ProfileLoader::load_from_hf_directory(config.model_dir, profile_error);
  if (!profile) {
    error_message = "failed to load model profile for CUDA variant validation: " + profile_error;
    return false;
  }

  const bool dimensions_match =
    profile->text.num_hidden_layers == kNumLayers &&
    profile->text.hidden_size == kHiddenSize &&
    profile->text.intermediate_size == kIntermediateSize &&
    profile->text.vocab_size == kVocabSize &&
    profile->text.num_attention_heads == kFaNumQHeads &&
    profile->text.num_key_value_heads == kFaNumKvHeads &&
    profile->text.head_dim == kFaHeadDim &&
    profile->text.linear_num_key_heads == kDnNumHeads &&
    profile->text.linear_num_value_heads == kDnGateHeads &&
    profile->text.linear_key_head_dim == kDnKeyDim &&
    profile->text.linear_conv_kernel_dim == kDnConvKernel;

  if (!dimensions_match) {
    error_message =
      "Qwen35x CUDA backend was compiled for qwen3.5-" + std::string(kCompiledVariant) +
      ", but model profile is variant=" + profile->variant +
      " layers=" + std::to_string(profile->text.num_hidden_layers) +
      " hidden=" + std::to_string(profile->text.hidden_size) +
      " intermediate=" + std::to_string(profile->text.intermediate_size) +
      " q_heads=" + std::to_string(profile->text.num_attention_heads) +
      " kv_heads=" + std::to_string(profile->text.num_key_value_heads) +
      " linear_key_heads=" + std::to_string(profile->text.linear_num_key_heads) +
      " linear_value_heads=" + std::to_string(profile->text.linear_num_value_heads) + ".";
    return false;
  }

  if (static_cast<int>(profile->fingerprint.attention_schedule.size()) != kNumLayers) {
    error_message = "model attention schedule length does not match compiled CUDA variant.";
    return false;
  }
  for (int i = 0; i < kNumLayers; ++i) {
    const int expected = profile->fingerprint.attention_schedule[static_cast<std::size_t>(i)] == qwen35x::AttentionBlock::full ? 1 : 0;
    if (expected != kLayerType[i]) {
      error_message = "model attention schedule differs from compiled CUDA variant at layer " + std::to_string(i) + ".";
      return false;
    }
  }

  return true;
}

bool initialize_backend_state(
  const Qwen35xCudaBackendConfig & config,
  DeviceArena & arena,
  BackendState & state,
  std::string & error_message) {
  {
    std::string used_name;
    if (!load_bf16_tensor_to_device(
          config.model_dir,
          {"model.language_model.embed_tokens.weight", "model.embed_tokens.weight"},
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
          arena,
          state.lm_head_weight,
          used_name,
          error_message)) {
      return false;
    }
  }

  std::vector<PackedLayerWeights> host_layers(static_cast<std::size_t>(kNumLayers));
  for (int layer_idx = 0; layer_idx < kNumLayers; ++layer_idx) {
    auto & layer = host_layers[static_cast<std::size_t>(layer_idx)];
    layer.layer_type = kLayerType[layer_idx];
    const std::string base = "model.language_model.layers." + std::to_string(layer_idx) + ".";

    auto load_ptr = [&](int ptr_idx, const std::vector<std::string> & suffixes) -> bool {
      std::vector<std::string> names;
      names.reserve(suffixes.size());
      for (const auto & suffix : suffixes) {
        names.push_back(base + suffix);
      }
      std::string used_name;
      void * tensor_ptr = nullptr;
      if (!load_bf16_tensor_to_device(config.model_dir, names, arena, tensor_ptr, used_name, error_message)) {
        error_message = "layer " + std::to_string(layer_idx) + " ptr[" + std::to_string(ptr_idx) + "]: " + error_message;
        return false;
      }
      layer.ptrs[ptr_idx] = tensor_ptr;
      return true;
    };

    if (layer.layer_type == 0) {
      if (!load_ptr(0, {"input_layernorm.weight"}) ||
          !load_ptr(1, {"linear_attn.in_proj_qkv.weight"}) ||
          !load_ptr(2, {"linear_attn.in_proj_z.weight"}) ||
          !load_ptr(3, {"linear_attn.in_proj_b.weight"}) ||
          !load_ptr(4, {"linear_attn.in_proj_a.weight"}) ||
          !load_ptr(5, {"linear_attn.conv1d.weight"}) ||
          !load_ptr(6, {"linear_attn.A_log"}) ||
          !load_ptr(7, {"linear_attn.dt_bias"}) ||
          !load_ptr(8, {"linear_attn.norm.weight"}) ||
          !load_ptr(9, {"linear_attn.out_proj.weight"}) ||
          !load_ptr(10, {"post_attention_layernorm.weight"}) ||
          !load_ptr(11, {"mlp.gate_proj.weight"}) ||
          !load_ptr(12, {"mlp.up_proj.weight"}) ||
          !load_ptr(13, {"mlp.down_proj.weight"})) {
        return false;
      }
    } else {
      if (!load_ptr(0, {"input_layernorm.weight"}) ||
          !load_ptr(1, {"self_attn.q_proj.weight"}) ||
          !load_ptr(2, {"self_attn.k_proj.weight"}) ||
          !load_ptr(3, {"self_attn.v_proj.weight"}) ||
          !load_ptr(4, {"self_attn.q_norm.weight"}) ||
          !load_ptr(5, {"self_attn.k_norm.weight"}) ||
          !load_ptr(6, {"self_attn.o_proj.weight"}) ||
          !load_ptr(7, {"post_attention_layernorm.weight"}) ||
          !load_ptr(8, {"mlp.gate_proj.weight"}) ||
          !load_ptr(9, {"mlp.up_proj.weight"}) ||
          !load_ptr(10, {"mlp.down_proj.weight"})) {
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

  int n_full_layers = 0;
  int n_delta_layers = 0;
  for (const int layer_type : kLayerType) {
    if (layer_type == 0) {
      ++n_delta_layers;
    } else {
      ++n_full_layers;
    }
  }

  const std::size_t bf16_bytes = sizeof(std::uint16_t);
  const std::size_t f32_bytes = sizeof(float);
  const int max_seq_len = config.max_context;

  if (!alloc_and_zero(
        arena,
        static_cast<std::size_t>(n_full_layers) * kFaNumKvHeads * max_seq_len * kFaHeadDim * bf16_bytes,
        state.fa_k_cache,
        "fa_k_cache",
        error_message) ||
      !alloc_and_zero(
        arena,
        static_cast<std::size_t>(n_full_layers) * kFaNumKvHeads * max_seq_len * kFaHeadDim * bf16_bytes,
        state.fa_v_cache,
        "fa_v_cache",
        error_message) ||
      !alloc_and_zero(
        arena,
        static_cast<std::size_t>(n_delta_layers) * kDnNumHeads * kDnKeyDim * kDnValueDim * f32_bytes,
        state.dn_states,
        "dn_states",
        error_message) ||
      !alloc_and_zero(
        arena,
        static_cast<std::size_t>(n_delta_layers) * kDnConvChannels * kDnConvKernel * f32_bytes,
        state.conv_bufs,
        "conv_bufs",
        error_message)) {
    return false;
  }

  if (!arena.alloc_bytes(kHiddenSize * bf16_bytes, state.hidden_buffer, error_message) ||
      !arena.alloc_bytes(std::max({kFaQprojSize, kDnConvChannels, (kHiddenSize * 8 + kIntermediateSize)}) * f32_bytes, state.g_activations, error_message) ||
      !arena.alloc_bytes(kHiddenSize * bf16_bytes, state.g_residual, error_message) ||
      !arena.alloc_bytes(std::max(kFaQprojSize, kDnConvChannels) * f32_bytes, state.g_qkv_scratch, error_message) ||
      !arena.alloc_bytes((kFaKvSize * 2) * f32_bytes, state.g_kv_scratch, error_message) ||
      !arena.alloc_bytes(std::max(kFaQSize, kDnVSize) * f32_bytes, state.g_attn_out, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(kMaxDecodeBlocks) * kFaGqaRatio * (kFaHeadDim + 2) * f32_bytes, state.g_attn_partials, error_message) ||
      !arena.alloc_bytes(kIntermediateSize * f32_bytes, state.g_mlp_inter, error_message) ||
      !arena.alloc_bytes(kDnVSize * f32_bytes, state.g_z_scratch, error_message) ||
      !arena.alloc_bytes(kDnGateHeads * f32_bytes, state.g_beta_scratch, error_message) ||
      !arena.alloc_bytes(kDnGateHeads * f32_bytes, state.g_alpha_scratch, error_message) ||
      !arena.alloc_bytes(kHiddenSize * f32_bytes, state.g_normalized, error_message) ||
      !arena.alloc_bytes(sizeof(unsigned int), reinterpret_cast<void *&>(state.barrier_counter), error_message) ||
      !arena.alloc_bytes(sizeof(unsigned int), reinterpret_cast<void *&>(state.barrier_generation), error_message) ||
      !arena.alloc_bytes(1024 * sizeof(float), reinterpret_cast<void *&>(state.block_max_vals), error_message) ||
      !arena.alloc_bytes(1024 * sizeof(int), reinterpret_cast<void *&>(state.block_max_idxs), error_message) ||
      !arena.alloc_bytes(sizeof(unsigned int), reinterpret_cast<void *&>(state.lm_sync_counter), error_message) ||
      !alloc_and_zero(
        arena,
        static_cast<std::size_t>(kVocabSize) * f32_bytes,
        reinterpret_cast<void *&>(state.seen_token_mask),
        "seen_token_mask",
        error_message) ||
      !arena.alloc_bytes(sizeof(int), reinterpret_cast<void *&>(state.output_token), error_message)) {
    return false;
  }

  if (!arena.alloc_bytes(static_cast<std::size_t>(config.max_context) * sizeof(int), reinterpret_cast<void *&>(state.token_ids), error_message)) {
    return false;
  }

  const std::size_t rope_elems = static_cast<std::size_t>(config.max_context) * kFaRotDim;
  if (!arena.alloc_bytes(rope_elems * sizeof(float), reinterpret_cast<void *&>(state.pf_rope_cos), error_message) ||
      !arena.alloc_bytes(rope_elems * sizeof(float), reinterpret_cast<void *&>(state.pf_rope_sin), error_message)) {
    return false;
  }
  std::vector<float> host_rope_cos(rope_elems);
  std::vector<float> host_rope_sin(rope_elems);
  for (int pos = 0; pos < config.max_context; ++pos) {
    for (int i = 0; i < kFaRotDim; ++i) {
      const float exponent = static_cast<float>(2 * (i % (kFaRotDim / 2))) / static_cast<float>(kFaRotDim);
      const float freq = static_cast<float>(pos) / std::pow(10000000.0f, exponent);
      host_rope_cos[static_cast<std::size_t>(pos) * kFaRotDim + i] = std::cos(freq);
      host_rope_sin[static_cast<std::size_t>(pos) * kFaRotDim + i] = std::sin(freq);
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
  const int mx = std::max({kDnConvChannels, kFaQprojSize, kIntermediateSize});

  if (!arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kHiddenSize * bf16_bytes, state.pf_hidden, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kHiddenSize * bf16_bytes, state.pf_residual, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kHiddenSize * bf16_bytes, state.pf_normalized, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * mx * bf16_bytes, state.pf_proj_buf, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * mx * bf16_bytes, state.pf_proj_buf2, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * std::max(kFaQSize, kFaKvSize) * bf16_bytes, state.pf_attn_buf, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kIntermediateSize * bf16_bytes, state.pf_mlp_buf, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kDnVSize * bf16_bytes, state.pf_dn_out_buf, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kDnConvChannels * sizeof(float), reinterpret_cast<void *&>(state.pf_dn_qkv_f32), error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kDnVSize * sizeof(float), reinterpret_cast<void *&>(state.pf_dn_out_f32), error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kDnGateHeads * sizeof(float), reinterpret_cast<void *&>(state.pf_beta_buf), error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kDnGateHeads * sizeof(float), reinterpret_cast<void *&>(state.pf_alpha_buf), error_message) ||
      !arena.alloc_bytes(kHiddenSize * f32_bytes, state.pf_final_normed, error_message) ||
      !arena.alloc_bytes(kHiddenSize * bf16_bytes, state.pf_hidden_bf16_out, error_message) ||
      !arena.alloc_bytes(1024 * sizeof(float), reinterpret_cast<void *&>(state.pf_lm_bmv), error_message) ||
      !arena.alloc_bytes(1024 * sizeof(int), reinterpret_cast<void *&>(state.pf_lm_bmi), error_message)) {
    return false;
  }

  return true;
}

bool reset_state(const BackendState & state, int max_seq_len, std::string & error_message) {
  int n_full_layers = 0;
  int n_delta_layers = 0;
  for (const int layer_type : kLayerType) {
    if (layer_type == 0) {
      ++n_delta_layers;
    } else {
      ++n_full_layers;
    }
  }

  const std::size_t fa_bytes = static_cast<std::size_t>(n_full_layers) * kFaNumKvHeads * max_seq_len * kFaHeadDim * sizeof(std::uint16_t);
  const std::size_t dn_bytes = static_cast<std::size_t>(n_delta_layers) * kDnNumHeads * kDnKeyDim * kDnValueDim * sizeof(float);
  const std::size_t conv_bytes = static_cast<std::size_t>(n_delta_layers) * kDnConvChannels * kDnConvKernel * sizeof(float);
  const std::size_t seen_bytes = static_cast<std::size_t>(kVocabSize) * sizeof(float);

  return check_cuda(cudaMemset(state.fa_k_cache, 0, fa_bytes), "cudaMemset(fa_k_cache)", error_message) &&
         check_cuda(cudaMemset(state.fa_v_cache, 0, fa_bytes), "cudaMemset(fa_v_cache)", error_message) &&
         check_cuda(cudaMemset(state.dn_states, 0, dn_bytes), "cudaMemset(dn_states)", error_message) &&
         check_cuda(cudaMemset(state.conv_bufs, 0, conv_bytes), "cudaMemset(conv_bufs)", error_message) &&
         check_cuda(cudaMemset(state.seen_token_mask, 0, seen_bytes), "cudaMemset(seen_token_mask)", error_message);
}

void launch_prefill_for_state(
  const BackendState & state,
  const Qwen35xCudaBackendConfig & config,
  int seq_len,
  bool compute_first_token,
  Qwen35xPrefillProfile * profile,
  cudaStream_t stream) {
  launch_prefill_bf16(
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
    compute_first_token ? 1 : 0,
    profile,
    stream);
}

bool warmup_prefill_backend(BackendState & state, const Qwen35xCudaBackendConfig & config, std::string & error_message) {
  if (!check_cuda(
        cudaMemset(state.token_ids, 0, static_cast<std::size_t>(config.max_context) * sizeof(int)),
        "cudaMemset(prefill warmup token_ids)",
        error_message)) {
    return false;
  }

  launch_prefill_for_state(state, config, 1, false, nullptr, nullptr);
  if (!check_cuda(cudaGetLastError(), "warmup launch_prefill_bf16", error_message) ||
      !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(prefill warmup)", error_message)) {
    return false;
  }

  return reset_state(state, config.max_context, error_message);
}

bool run_prefill_impl(
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
  launch_prefill_for_state(state, config, seq_len, compute_first_token, profile, nullptr);
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
        cudaMemcpy(state.hidden_buffer, state.pf_hidden_bf16_out, kHiddenSize * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice),
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
  const BackendState & state,
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

  if (repetition_penalty > 1.0f && input_token >= 0 && input_token < kVocabSize) {
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

  launch_decode(
    input_token,
    state.output_token,
    state.embed_weight,
    state.layer_weights,
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
  DeviceArena arena;
  BackendState state;
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
  if (!validate_compiled_variant(config, error_message)) {
    return false;
  }

  impl_ = std::make_unique<Impl>();
  impl_->config = config;
  ::set_decode_blocks_override(config.decode_blocks);

  if (!initialize_backend_state(config, impl_->arena, impl_->state, error_message)) {
    impl_ = std::make_unique<Impl>();
    return false;
  }
  if (!warmup_prefill_backend(impl_->state, config, error_message)) {
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
  return reset_state(impl_->state, impl_->config.max_context, error_message);
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
  const bool ok = run_prefill_impl(impl_->state, impl_->config, tokens, out_first_token, true, profile, error_message);
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
  const bool ok = run_prefill_impl(impl_->state, impl_->config, tokens, ignored_first_token, false, profile, error_message);
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
    impl_->state,
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
  return ::query_max_safe_decode_blocks();
}

void set_decode_blocks_override(int blocks) {
  ::set_decode_blocks_override(blocks);
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
