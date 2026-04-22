#include "qwen35x/tokenizer/tokenizer.h"
#include "qwen35x/weights/safetensors.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kNumLayers = 24;
constexpr int kHiddenSize = 1024;
constexpr int kIntermediateSize = 3584;
constexpr int kVocabSize = 248320;
constexpr int kMaxSeqLen = 2048;
constexpr int kFaNumQHeads = 8;
constexpr int kFaNumKvHeads = 2;
constexpr int kFaHeadDim = 256;
constexpr int kFaQSize = kFaNumQHeads * kFaHeadDim;
constexpr int kFaQprojSize = kFaQSize * 2;
constexpr int kFaKvSize = kFaNumKvHeads * kFaHeadDim;
constexpr int kDnNumHeads = 16;
constexpr int kDnKeyDim = 128;
constexpr int kDnValueDim = 128;
constexpr int kDnQkSize = kDnNumHeads * kDnKeyDim;
constexpr int kDnVSize = kDnNumHeads * kDnValueDim;
constexpr int kDnConvChannels = kDnQkSize * 2 + kDnVSize;
constexpr int kDnConvKernel = 4;
constexpr int kLayerType[kNumLayers] = {
  0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1
};

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
  int position,
  int max_seq_len,
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
  void * beta_buf,
  void * alpha_buf,
  void * final_normed,
  void * hidden_bf16_out,
  void * lm_bmv,
  void * lm_bmi,
  cudaStream_t stream);

extern "C" void set_decode_blocks_override(int blocks);
extern "C" int query_max_safe_decode_blocks();

struct BenchmarkOptions {
  std::string model_dir = "models/qwen3.5-0.8b";
  std::string prompt_text = "Hello";
  std::string long_prompt_text =
    "Explain in great detail the history of artificial intelligence, machine learning, deep learning, and neural networks. ";
  int max_new_tokens = 128;
  int max_context = 256;
  int warmup_runs = 1;
  int runs = 3;
  int decode_blocks = 0;
  bool query_decode_blocks_only = false;
  std::string profile_json_path;
};

struct BenchmarkResult {
  double load_time_ms = 0.0;
  double avg_pp_time_ms = 0.0;
  double avg_pp_tokens_per_second = 0.0;
  double avg_tg_time_ms = 0.0;
  double avg_tg_tokens_per_second = 0.0;
  int prompt_tokens = 0;
  int prefill_prompt_tokens = 0;
  int generated_tokens = 0;
  std::vector<std::int32_t> output_tokens;
};

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

struct MegakernelState {
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
  float * pf_beta_buf = nullptr;
  float * pf_alpha_buf = nullptr;
  void * pf_final_normed = nullptr;
  void * pf_hidden_bf16_out = nullptr;
  float * pf_lm_bmv = nullptr;
  int * pf_lm_bmi = nullptr;
};

bool check_cuda(cudaError_t status, const std::string & label, std::string & error_message) {
  if (status == cudaSuccess) {
    return true;
  }
  error_message = label + " failed: " + cudaGetErrorString(status);
  return false;
}

std::string json_escape(const std::string & input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (const char ch : input) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '\"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(ch);
        break;
    }
  }
  return out;
}

bool write_profile_json(
  const std::string & output_path,
  const BenchmarkOptions & options,
  const BenchmarkResult & result,
  std::string & error_message) {
  std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    error_message = "Failed to open profile JSON path: " + output_path;
    return false;
  }

  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"backend\": \"luce-megakernel-cuda\",\n";
  out << "  \"prompt_tokens\": " << result.prompt_tokens << ",\n";
  out << "  \"prefill_prompt_tokens\": " << result.prefill_prompt_tokens << ",\n";
  out << "  \"generated_tokens\": " << result.generated_tokens << ",\n";
  out << "  \"load_time_ms\": " << result.load_time_ms << ",\n";
  out << "  \"decode_time_ms\": " << result.avg_tg_time_ms << ",\n";
  out << "  \"tokens_per_second\": " << result.avg_tg_tokens_per_second << ",\n";
  out << "  \"pp_time_ms\": " << result.avg_pp_time_ms << ",\n";
  out << "  \"pp_tokens_per_second\": " << result.avg_pp_tokens_per_second << ",\n";
  out << "  \"max_new_tokens\": " << options.max_new_tokens << ",\n";
  out << "  \"max_context\": " << options.max_context << ",\n";
  out << "  \"warmup_runs\": " << options.warmup_runs << ",\n";
  out << "  \"runs\": " << options.runs << ",\n";
  out << "  \"decode_blocks\": " << options.decode_blocks << ",\n";
  out << "  \"output_token_ids\": [";
  for (std::size_t i = 0; i < result.output_tokens.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << result.output_tokens[i];
  }
  out << "],\n";
  out << "  \"prompt_text\": \"" << json_escape(options.prompt_text) << "\"\n";
  out << "}\n";

  if (!out.good()) {
    error_message = "Failed to write profile JSON: " + output_path;
    return false;
  }
  return true;
}

bool parse_args(int argc, char ** argv, BenchmarkOptions & options, std::string & error_message) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--hf-model-dir" && i + 1 < argc) {
      options.model_dir = argv[++i];
    } else if (arg == "--prompt-text" && i + 1 < argc) {
      options.prompt_text = argv[++i];
    } else if (arg == "--long-prompt-text" && i + 1 < argc) {
      options.long_prompt_text = argv[++i];
    } else if (arg == "--max-new-tokens" && i + 1 < argc) {
      options.max_new_tokens = std::stoi(argv[++i]);
    } else if (arg == "--max-context" && i + 1 < argc) {
      options.max_context = std::stoi(argv[++i]);
    } else if (arg == "--warmup-runs" && i + 1 < argc) {
      options.warmup_runs = std::stoi(argv[++i]);
    } else if (arg == "--runs" && i + 1 < argc) {
      options.runs = std::stoi(argv[++i]);
    } else if (arg == "--decode-blocks" && i + 1 < argc) {
      options.decode_blocks = std::stoi(argv[++i]);
    } else if (arg == "--query-decode-blocks") {
      options.query_decode_blocks_only = true;
    } else if (arg == "--profile-json" && i + 1 < argc) {
      options.profile_json_path = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: qwen35x_lucebench [--hf-model-dir <path>] [--prompt-text <text>]\n";
      std::cout << "                       [--long-prompt-text <text>] [--max-new-tokens <n>] [--max-context <n>]\n";
      std::cout << "                       [--warmup-runs <n>] [--runs <n>] [--decode-blocks <n>] [--profile-json <path>]\n";
      std::cout << "                       [--query-decode-blocks]\n";
      return false;
    } else {
      error_message = "Unknown argument: " + arg;
      return false;
    }
  }

  if (options.max_new_tokens <= 0) {
    error_message = "max-new-tokens must be > 0";
    return false;
  }
  if (options.max_context <= 0 || options.max_context > kMaxSeqLen) {
    error_message = "max-context must be in [1, " + std::to_string(kMaxSeqLen) + "]";
    return false;
  }
  if (options.runs <= 0) {
    error_message = "runs must be > 0";
    return false;
  }
  if (options.warmup_runs < 0) {
    error_message = "warmup-runs must be >= 0";
    return false;
  }
  if (options.prompt_text.empty()) {
    error_message = "prompt-text must not be empty";
    return false;
  }
  if (options.decode_blocks < 0) {
    error_message = "decode-blocks must be >= 0";
    return false;
  }
  return true;
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

bool initialize_megakernel_state(
  const BenchmarkOptions & options,
  DeviceArena & arena,
  MegakernelState & state,
  std::string & error_message) {
  {
    std::string used_name;
    if (!load_bf16_tensor_to_device(
          options.model_dir,
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
          options.model_dir,
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
          options.model_dir,
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
      if (!load_bf16_tensor_to_device(options.model_dir, names, arena, tensor_ptr, used_name, error_message)) {
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

  if (!alloc_and_zero(
        arena,
        static_cast<std::size_t>(n_full_layers) * kFaNumKvHeads * kMaxSeqLen * kFaHeadDim * bf16_bytes,
        state.fa_k_cache,
        "fa_k_cache",
        error_message) ||
      !alloc_and_zero(
        arena,
        static_cast<std::size_t>(n_full_layers) * kFaNumKvHeads * kMaxSeqLen * kFaHeadDim * bf16_bytes,
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
      !arena.alloc_bytes(kIntermediateSize * f32_bytes, state.g_mlp_inter, error_message) ||
      !arena.alloc_bytes(kDnVSize * f32_bytes, state.g_z_scratch, error_message) ||
      !arena.alloc_bytes(kDnNumHeads * f32_bytes, state.g_beta_scratch, error_message) ||
      !arena.alloc_bytes(kDnNumHeads * f32_bytes, state.g_alpha_scratch, error_message) ||
      !arena.alloc_bytes(kHiddenSize * f32_bytes, state.g_normalized, error_message) ||
      !arena.alloc_bytes(sizeof(unsigned int), reinterpret_cast<void *&>(state.barrier_counter), error_message) ||
      !arena.alloc_bytes(sizeof(unsigned int), reinterpret_cast<void *&>(state.barrier_generation), error_message) ||
      !arena.alloc_bytes(1024 * sizeof(float), reinterpret_cast<void *&>(state.block_max_vals), error_message) ||
      !arena.alloc_bytes(1024 * sizeof(int), reinterpret_cast<void *&>(state.block_max_idxs), error_message) ||
      !arena.alloc_bytes(sizeof(unsigned int), reinterpret_cast<void *&>(state.lm_sync_counter), error_message) ||
      !arena.alloc_bytes(sizeof(int), reinterpret_cast<void *&>(state.output_token), error_message)) {
    return false;
  }

  if (!arena.alloc_bytes(static_cast<std::size_t>(options.max_context) * sizeof(int), reinterpret_cast<void *&>(state.token_ids), error_message)) {
    return false;
  }

  const int prefill_s = options.max_context;
  const int mx = std::max({kDnConvChannels, kFaQprojSize, kIntermediateSize});

  if (!arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kHiddenSize * bf16_bytes, state.pf_hidden, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kHiddenSize * bf16_bytes, state.pf_residual, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kHiddenSize * bf16_bytes, state.pf_normalized, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * mx * bf16_bytes, state.pf_proj_buf, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * mx * bf16_bytes, state.pf_proj_buf2, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * std::max(kFaQSize, kFaKvSize) * bf16_bytes, state.pf_attn_buf, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kIntermediateSize * bf16_bytes, state.pf_mlp_buf, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kDnVSize * bf16_bytes, state.pf_dn_out_buf, error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kDnNumHeads * sizeof(float), reinterpret_cast<void *&>(state.pf_beta_buf), error_message) ||
      !arena.alloc_bytes(static_cast<std::size_t>(prefill_s) * kDnNumHeads * sizeof(float), reinterpret_cast<void *&>(state.pf_alpha_buf), error_message) ||
      !arena.alloc_bytes(kHiddenSize * bf16_bytes, state.pf_final_normed, error_message) ||
      !arena.alloc_bytes(kHiddenSize * bf16_bytes, state.pf_hidden_bf16_out, error_message) ||
      !arena.alloc_bytes(1024 * sizeof(float), reinterpret_cast<void *&>(state.pf_lm_bmv), error_message) ||
      !arena.alloc_bytes(1024 * sizeof(int), reinterpret_cast<void *&>(state.pf_lm_bmi), error_message)) {
    return false;
  }

  return true;
}

bool reset_state(const MegakernelState & state, std::string & error_message) {
  int n_full_layers = 0;
  int n_delta_layers = 0;
  for (const int layer_type : kLayerType) {
    if (layer_type == 0) {
      ++n_delta_layers;
    } else {
      ++n_full_layers;
    }
  }

  const std::size_t fa_bytes = static_cast<std::size_t>(n_full_layers) * kFaNumKvHeads * kMaxSeqLen * kFaHeadDim * sizeof(std::uint16_t);
  const std::size_t dn_bytes = static_cast<std::size_t>(n_delta_layers) * kDnNumHeads * kDnKeyDim * kDnValueDim * sizeof(float);
  const std::size_t conv_bytes = static_cast<std::size_t>(n_delta_layers) * kDnConvChannels * kDnConvKernel * sizeof(float);

  return check_cuda(cudaMemset(state.fa_k_cache, 0, fa_bytes), "cudaMemset(fa_k_cache)", error_message) &&
         check_cuda(cudaMemset(state.fa_v_cache, 0, fa_bytes), "cudaMemset(fa_v_cache)", error_message) &&
         check_cuda(cudaMemset(state.dn_states, 0, dn_bytes), "cudaMemset(dn_states)", error_message) &&
         check_cuda(cudaMemset(state.conv_bufs, 0, conv_bytes), "cudaMemset(conv_bufs)", error_message);
}

bool run_prefill(
  const MegakernelState & state,
  const std::vector<std::int32_t> & tokens,
  int & out_first_token,
  std::string & error_message) {
  if (tokens.empty()) {
    error_message = "Prefill tokens are empty.";
    return false;
  }

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

  launch_prefill_bf16(
    state.token_ids,
    static_cast<int>(tokens.size()),
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
    state.pf_beta_buf,
    state.pf_alpha_buf,
    state.pf_final_normed,
    state.pf_hidden_bf16_out,
    state.pf_lm_bmv,
    state.pf_lm_bmi,
    nullptr);

  if (!check_cuda(cudaGetLastError(), "launch_prefill_bf16", error_message)) {
    return false;
  }

  if (!check_cuda(
        cudaMemcpy(state.hidden_buffer, state.pf_hidden_bf16_out, kHiddenSize * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice),
        "cudaMemcpy(D2D hidden handoff)",
        error_message)) {
    return false;
  }

  if (!check_cuda(
        cudaMemcpy(&out_first_token, state.output_token, sizeof(int), cudaMemcpyDeviceToHost),
        "cudaMemcpy(D2H first_token)",
        error_message)) {
    return false;
  }

  return true;
}

bool run_decode_step(
  const MegakernelState & state,
  int input_token,
  int position,
  int & out_next_token,
  std::string & error_message) {
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
    position,
    kMaxSeqLen,
    nullptr);

  if (!check_cuda(cudaGetLastError(), "launch_decode", error_message)) {
    return false;
  }

  if (!check_cuda(
        cudaMemcpy(&out_next_token, state.output_token, sizeof(int), cudaMemcpyDeviceToHost),
        "cudaMemcpy(D2H next_token)",
        error_message)) {
    return false;
  }
  return true;
}

bool repeat_to_target_tokens(
  const std::vector<std::int32_t> & base_tokens,
  int target_count,
  std::vector<std::int32_t> & out_tokens,
  std::string & error_message) {
  if (base_tokens.empty()) {
    error_message = "Long prompt tokenization returned zero tokens.";
    return false;
  }
  out_tokens.clear();
  out_tokens.reserve(static_cast<std::size_t>(target_count));
  while (static_cast<int>(out_tokens.size()) < target_count) {
    const int remaining = target_count - static_cast<int>(out_tokens.size());
    const int chunk = std::min(remaining, static_cast<int>(base_tokens.size()));
    out_tokens.insert(out_tokens.end(), base_tokens.begin(), base_tokens.begin() + chunk);
  }
  return true;
}

} // namespace

int main(int argc, char ** argv) {
  BenchmarkOptions options;
  std::string error_message;
  const auto parse_ok = parse_args(argc, argv, options, error_message);
  if (!parse_ok) {
    if (!error_message.empty()) {
      std::cerr << "argument parse failed: " << error_message << "\n";
      return 2;
    }
    return 0;
  }

  int device_count = 0;
  if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error_message)) {
    std::cerr << "cuda init failed: " << error_message << "\n";
    return 3;
  }
  if (device_count <= 0) {
    std::cerr << "cuda init failed: no CUDA devices found.\n";
    return 3;
  }

  const int max_safe_decode_blocks = query_max_safe_decode_blocks();
  if (options.query_decode_blocks_only) {
    std::cout << "max_safe_decode_blocks: " << max_safe_decode_blocks << "\n";
    return 0;
  }
  if (options.decode_blocks > 0) {
    set_decode_blocks_override(options.decode_blocks);
  }

  auto load_start = std::chrono::steady_clock::now();

  qwen35x::QwenTokenizer tokenizer;
  if (!qwen35x::QwenTokenizer::load_from_hf_directory(options.model_dir, tokenizer, error_message)) {
    std::cerr << "tokenizer load failed: " << error_message << "\n";
    return 4;
  }

  std::vector<std::int32_t> prompt_tokens;
  if (!tokenizer.encode(options.prompt_text, prompt_tokens, error_message)) {
    std::cerr << "prompt tokenize failed: " << error_message << "\n";
    return 4;
  }
  if (prompt_tokens.empty()) {
    std::cerr << "prompt tokenize failed: zero tokens.\n";
    return 4;
  }
  if (static_cast<int>(prompt_tokens.size()) > options.max_context) {
    prompt_tokens.resize(static_cast<std::size_t>(options.max_context));
  }

  std::vector<std::int32_t> long_prompt_seed;
  if (!tokenizer.encode(options.long_prompt_text, long_prompt_seed, error_message)) {
    std::cerr << "long prompt tokenize failed: " << error_message << "\n";
    return 4;
  }

  std::vector<std::int32_t> prefill_tokens;
  if (!repeat_to_target_tokens(long_prompt_seed, options.max_context, prefill_tokens, error_message)) {
    std::cerr << "long prompt preparation failed: " << error_message << "\n";
    return 4;
  }

  DeviceArena arena;
  MegakernelState state;
  if (!initialize_megakernel_state(options, arena, state, error_message)) {
    std::cerr << "megakernel init failed: " << error_message << "\n";
    return 5;
  }

  const auto load_end = std::chrono::steady_clock::now();
  const double load_time_ms =
    std::chrono::duration<double, std::milli>(load_end - load_start).count();

  std::optional<std::int32_t> eos_token = tokenizer.token_to_id("<|endoftext|>");

  for (int warm = 0; warm < options.warmup_runs; ++warm) {
    if (!reset_state(state, error_message)) {
      std::cerr << "warmup reset failed: " << error_message << "\n";
      return 6;
    }
    int first_token = 0;
    if (!run_prefill(state, prefill_tokens, first_token, error_message)) {
      std::cerr << "warmup prefill failed: " << error_message << "\n";
      return 6;
    }
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup pp)", error_message)) {
      std::cerr << "warmup sync failed: " << error_message << "\n";
      return 6;
    }

    if (!reset_state(state, error_message)) {
      std::cerr << "warmup reset failed: " << error_message << "\n";
      return 6;
    }
    if (!run_prefill(state, prompt_tokens, first_token, error_message)) {
      std::cerr << "warmup prefill failed: " << error_message << "\n";
      return 6;
    }
    int token = first_token;
    int position = static_cast<int>(prompt_tokens.size());
    for (int step = 0; step < options.max_new_tokens; ++step) {
      int next = 0;
      if (!run_decode_step(state, token, position, next, error_message)) {
        std::cerr << "warmup decode failed: " << error_message << "\n";
        return 6;
      }
      token = next;
      ++position;
      if (eos_token && token == *eos_token) {
        break;
      }
    }
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup tg)", error_message)) {
      std::cerr << "warmup sync failed: " << error_message << "\n";
      return 6;
    }
  }

  std::vector<double> pp_tps_runs;
  std::vector<double> pp_ms_runs;
  std::vector<double> tg_tps_runs;
  std::vector<double> tg_ms_runs;
  std::vector<std::int32_t> last_tokens;

  for (int run = 0; run < options.runs; ++run) {
    if (!reset_state(state, error_message)) {
      std::cerr << "run reset failed: " << error_message << "\n";
      return 7;
    }

    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before pp)", error_message)) {
      std::cerr << "run sync failed: " << error_message << "\n";
      return 7;
    }
    const auto pp_t0 = std::chrono::steady_clock::now();
    int pp_first = 0;
    if (!run_prefill(state, prefill_tokens, pp_first, error_message)) {
      std::cerr << "run prefill failed: " << error_message << "\n";
      return 7;
    }
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after pp)", error_message)) {
      std::cerr << "run sync failed: " << error_message << "\n";
      return 7;
    }
    const auto pp_t1 = std::chrono::steady_clock::now();
    const double pp_ms = std::chrono::duration<double, std::milli>(pp_t1 - pp_t0).count();
    const double pp_tps = static_cast<double>(prefill_tokens.size()) / (pp_ms / 1000.0);
    pp_ms_runs.push_back(pp_ms);
    pp_tps_runs.push_back(pp_tps);

    if (!reset_state(state, error_message)) {
      std::cerr << "run reset failed: " << error_message << "\n";
      return 7;
    }
    int first = 0;
    if (!run_prefill(state, prompt_tokens, first, error_message)) {
      std::cerr << "run prefill failed: " << error_message << "\n";
      return 7;
    }

    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before tg)", error_message)) {
      std::cerr << "run sync failed: " << error_message << "\n";
      return 7;
    }

    last_tokens.clear();
    int token = first;
    int position = static_cast<int>(prompt_tokens.size());
    const auto tg_t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < options.max_new_tokens; ++step) {
      int next = 0;
      if (!run_decode_step(state, token, position, next, error_message)) {
        std::cerr << "run decode failed: " << error_message << "\n";
        return 7;
      }
      token = next;
      ++position;
      if (eos_token && token == *eos_token) {
        break;
      }
      last_tokens.push_back(token);
    }
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after tg)", error_message)) {
      std::cerr << "run sync failed: " << error_message << "\n";
      return 7;
    }
    const auto tg_t1 = std::chrono::steady_clock::now();
    const double tg_ms = std::chrono::duration<double, std::milli>(tg_t1 - tg_t0).count();
    const double tg_tps = last_tokens.empty() ? 0.0 : static_cast<double>(last_tokens.size()) / (tg_ms / 1000.0);
    tg_ms_runs.push_back(tg_ms);
    tg_tps_runs.push_back(tg_tps);

    std::cout << "run " << (run + 1) << "/" << options.runs
              << " pp" << prefill_tokens.size() << "=" << std::fixed << std::setprecision(2) << pp_tps
              << " tok/s tg" << last_tokens.size() << "=" << tg_tps << " tok/s\n";
  }

  auto avg = [](const std::vector<double> & values) -> double {
    if (values.empty()) {
      return 0.0;
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
  };

  BenchmarkResult result;
  result.load_time_ms = load_time_ms;
  result.avg_pp_time_ms = avg(pp_ms_runs);
  result.avg_pp_tokens_per_second = avg(pp_tps_runs);
  result.avg_tg_time_ms = avg(tg_ms_runs);
  result.avg_tg_tokens_per_second = avg(tg_tps_runs);
  result.prompt_tokens = static_cast<int>(prompt_tokens.size());
  result.prefill_prompt_tokens = static_cast<int>(prefill_tokens.size());
  result.generated_tokens = static_cast<int>(last_tokens.size());
  result.output_tokens = last_tokens;

  std::cout << "luce megakernel benchmark\n";
  std::cout << "  model_dir: " << options.model_dir << "\n";
  std::cout << "  max_safe_decode_blocks: " << max_safe_decode_blocks << "\n";
  std::cout << "  decode_blocks: " << ((options.decode_blocks > 0) ? options.decode_blocks : max_safe_decode_blocks) << "\n";
  std::cout << "  prompt_tokens: " << result.prompt_tokens << "\n";
  std::cout << "  prefill_tokens: " << result.prefill_prompt_tokens << "\n";
  std::cout << "  generated_tokens: " << result.generated_tokens << "\n";
  std::cout << "  load_time_ms: " << std::fixed << std::setprecision(2) << result.load_time_ms << "\n";
  std::cout << "  pp_tokens_per_second: " << std::fixed << std::setprecision(2) << result.avg_pp_tokens_per_second << "\n";
  std::cout << "  tg_tokens_per_second: " << std::fixed << std::setprecision(2) << result.avg_tg_tokens_per_second << "\n";

  if (!last_tokens.empty()) {
    std::string decoded;
    if (tokenizer.decode(last_tokens, decoded, error_message)) {
      std::cout << "  output_text: " << decoded << "\n";
    } else {
      std::cerr << "token decode warning: " << error_message << "\n";
    }
  }

  if (!options.profile_json_path.empty()) {
    if (!write_profile_json(options.profile_json_path, options, result, error_message)) {
      std::cerr << "profile json write failed: " << error_message << "\n";
      return 8;
    }
    std::cout << "  profile_json: " << options.profile_json_path << "\n";
  }

  return 0;
}
