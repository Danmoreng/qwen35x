#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/reference_inference.h"
#include "qwen35x/runtime/runtime.h"
#include "qwen35x/tokenizer/tokenizer.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

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

const char * gpu_decode_backend_name(const qwen35x::GpuDecodeBackend backend) {
  switch (backend) {
    case qwen35x::GpuDecodeBackend::runtime_default:
      return "runtime_default";
    case qwen35x::GpuDecodeBackend::luce:
      return "luce";
    default:
      return "unknown";
  }
}

const char * luce_prefill_mode_name(const qwen35x::LucePrefillMode mode) {
  switch (mode) {
    case qwen35x::LucePrefillMode::replay:
      return "replay";
    case qwen35x::LucePrefillMode::batched:
      return "batched";
    default:
      return "unknown";
  }
}

const char * luce_layer_type_name(const int layer_type) {
  return layer_type == 0 ? "deltanet" : "full_attention";
}

void write_luce_layer_profile_json(std::ostream & out, const qwen35x::luce::LuceLayerProfile & layer) {
  out << "      {\n";
  out << "        \"layer_index\": " << layer.layer_index << ",\n";
  out << "        \"layer_type\": \"" << luce_layer_type_name(layer.layer_type) << "\",\n";
  out << "        \"total_ms\": " << layer.total_ms << ",\n";
  out << "        \"rms_norm_ms\": " << layer.rms_norm_ms << ",\n";
  out << "        \"qkv_projection_ms\": " << layer.qkv_projection_ms << ",\n";
  out << "        \"kv_projection_ms\": " << layer.kv_projection_ms << ",\n";
  out << "        \"z_projection_ms\": " << layer.z_projection_ms << ",\n";
  out << "        \"beta_alpha_projection_ms\": " << layer.beta_alpha_projection_ms << ",\n";
  out << "        \"conv_ms\": " << layer.conv_ms << ",\n";
  out << "        \"gate_ms\": " << layer.gate_ms << ",\n";
  out << "        \"recurrence_ms\": " << layer.recurrence_ms << ",\n";
  out << "        \"post_norm_gate_ms\": " << layer.post_norm_gate_ms << ",\n";
  out << "        \"qk_norm_rope_ms\": " << layer.qk_norm_rope_ms << ",\n";
  out << "        \"attention_ms\": " << layer.attention_ms << ",\n";
  out << "        \"out_projection_ms\": " << layer.out_projection_ms << ",\n";
  out << "        \"residual_ms\": " << layer.residual_ms << ",\n";
  out << "        \"mlp_norm_ms\": " << layer.mlp_norm_ms << ",\n";
  out << "        \"mlp_projection_ms\": " << layer.mlp_projection_ms << ",\n";
  out << "        \"mlp_activation_ms\": " << layer.mlp_activation_ms << ",\n";
  out << "        \"mlp_down_projection_ms\": " << layer.mlp_down_projection_ms << ",\n";
  out << "        \"mlp_residual_ms\": " << layer.mlp_residual_ms << "\n";
  out << "      }";
}

void write_luce_profile_json(std::ostream & out, const qwen35x::luce::LuceRuntimeProfile & profile) {
  out << "  \"luce_profile\": {\n";
  out << "    \"enabled\": " << (profile.enabled ? "true" : "false") << ",\n";
  out << "    \"prefill_runs\": " << profile.prefill_runs << ",\n";
  out << "    \"prefill\": {\n";
  out << "      \"enabled\": " << (profile.prefill.enabled ? "true" : "false") << ",\n";
  out << "      \"seq_len\": " << profile.prefill.seq_len << ",\n";
  out << "      \"compute_logits\": " << (profile.prefill.compute_logits ? "true" : "false") << ",\n";
  out << "      \"host_total_ms\": " << profile.prefill.host_total_ms << ",\n";
  out << "      \"gpu_total_ms\": " << profile.prefill.gpu_total_ms << ",\n";
  out << "      \"token_upload_ms\": " << profile.prefill.token_upload_ms << ",\n";
  out << "      \"embed_ms\": " << profile.prefill.embed_ms << ",\n";
  out << "      \"mark_seen_ms\": " << profile.prefill.mark_seen_ms << ",\n";
  out << "      \"final_norm_ms\": " << profile.prefill.final_norm_ms << ",\n";
  out << "      \"lm_head_ms\": " << profile.prefill.lm_head_ms << ",\n";
  out << "      \"lm_reduce_ms\": " << profile.prefill.lm_reduce_ms << ",\n";
  out << "      \"hidden_handoff_ms\": " << profile.prefill.hidden_handoff_ms << ",\n";
  out << "      \"output_token_download_ms\": " << profile.prefill.output_token_download_ms << ",\n";
  out << "      \"layers\": [\n";
  const int layer_count = std::min(profile.prefill.layer_count, qwen35x::luce::kLuceProfileMaxLayers);
  for (int i = 0; i < layer_count; ++i) {
    write_luce_layer_profile_json(out, profile.prefill.layers[i]);
    out << (i + 1 < layer_count ? ",\n" : "\n");
  }
  out << "      ]\n";
  out << "    },\n";
  out << "    \"decode\": {\n";
  out << "      \"enabled\": " << (profile.decode.enabled ? "true" : "false") << ",\n";
  out << "      \"steps\": " << profile.decode.steps << ",\n";
  out << "      \"last_position\": " << profile.decode.last_position << ",\n";
  out << "      \"host_total_ms\": " << profile.decode.host_total_ms << ",\n";
  out << "      \"seen_token_upload_ms\": " << profile.decode.seen_token_upload_ms << ",\n";
  out << "      \"launch_total_ms\": " << profile.decode.launch_total_ms << ",\n";
  out << "      \"decode_kernel_ms\": " << profile.decode.decode_kernel_ms << ",\n";
  out << "      \"lm_head_ms\": " << profile.decode.lm_head_ms << ",\n";
  out << "      \"output_token_download_ms\": " << profile.decode.output_token_download_ms << "\n";
  out << "    }\n";
  out << "  }";
}

bool write_profile_json(
  const std::string & output_path,
  const qwen35x::ReferenceInferenceOptions & options,
  const qwen35x::ReferenceInferenceResult & result,
  const std::string & generated_text,
  const std::string & backend,
  const std::string & decode_backend,
  std::string & error_message) {
  std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    error_message = "Failed to open profile JSON path: " + output_path;
    return false;
  }

  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"backend\": \"" << json_escape(backend) << "\",\n";
  out << "  \"decode_backend\": \"" << json_escape(decode_backend) << "\",\n";
  out << "  \"luce_prefill_mode\": \"" << json_escape(luce_prefill_mode_name(options.luce_prefill_mode)) << "\",\n";
  out << "  \"prefill_only\": " << (options.prefill_only ? "true" : "false") << ",\n";
  out << "  \"prompt_tokens\": " << options.prompt_tokens.size() << ",\n";
  out << "  \"prompt_token_ids\": [";
  for (std::size_t i = 0; i < options.prompt_tokens.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << options.prompt_tokens[i];
  }
  out << "],\n";
  out << "  \"generated_tokens\": " << result.generated_tokens.size() << ",\n";
  out << "  \"forward_pass_tokens\": " << result.forward_pass_tokens << ",\n";
  out << "  \"load_time_ms\": " << result.load_time_ms << ",\n";
  out << "  \"prefill_time_ms\": " << result.prefill_time_ms << ",\n";
  out << "  \"prefill_tokens_per_second\": " << result.prefill_tokens_per_second << ",\n";
  out << "  \"decode_time_ms\": " << result.decode_time_ms << ",\n";
  out << "  \"tokens_per_second\": " << result.tokens_per_second << ",\n";
  out << "  \"sampling\": {\n";
  out << "    \"temperature\": " << options.sampling.temperature << ",\n";
  out << "    \"top_p\": " << options.sampling.top_p << ",\n";
  out << "    \"top_k\": " << options.sampling.top_k << ",\n";
  out << "    \"repeat_penalty\": " << options.sampling.repetition_penalty << ",\n";
  out << "    \"seed\": " << options.sampling.seed << "\n";
  out << "  },\n";
  out << "  \"cuda\": {\n";
  out << "    \"matvec_bf16\": " << (options.use_cuda_matvec_bf16 ? "true" : "false") << ",\n";
  out << "    \"profile_sync\": " << (options.profile_cuda_sync ? "true" : "false") << "\n";
  out << "  },\n";
  out << "  \"stages_ms\": {\n";
  out << "    \"embedding\": " << result.timing_breakdown.embedding_ms << ",\n";
  out << "    \"attention\": " << result.timing_breakdown.attention_ms << ",\n";
  out << "    \"mlp\": " << result.timing_breakdown.mlp_ms << ",\n";
  out << "    \"logits\": " << result.timing_breakdown.logits_ms << ",\n";
  out << "    \"sampling\": " << result.timing_breakdown.sampling_ms << ",\n";
  out << "    \"stop_checks\": " << result.timing_breakdown.stop_checks_ms << "\n";
  out << "  },\n";
  out << "  \"cuda_transfers\": {\n";
  out << "    \"host_to_device_bytes\": " << result.transfer_breakdown.host_to_device_bytes << ",\n";
  out << "    \"device_to_host_bytes\": " << result.transfer_breakdown.device_to_host_bytes << ",\n";
  out << "    \"other_bytes\": " << result.transfer_breakdown.other_bytes << ",\n";
  out << "    \"copy_calls\": " << result.transfer_breakdown.copy_calls << ",\n";
  out << "    \"host_to_device_bytes_per_forward_token\": " << result.host_to_device_bytes_per_forward_token << ",\n";
  out << "    \"device_to_host_bytes_per_forward_token\": " << result.device_to_host_bytes_per_forward_token << "\n";
  out << "  },\n";
  if (result.luce_profile.enabled) {
    write_luce_profile_json(out, result.luce_profile);
    out << ",\n";
  }
  out << "  \"output_token_ids\": [";
  for (std::size_t i = 0; i < result.generated_tokens.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << result.generated_tokens[i];
  }
  out << "],\n";
  out << "  \"generated_text\": \"" << json_escape(generated_text) << "\"\n";
  out << "}\n";
  if (!out.good()) {
    error_message = "Failed to write profile JSON: " + output_path;
    return false;
  }

  return true;
}

bool read_text_file(const std::string & path, std::string & out_text, std::string & error_message) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    error_message = "Failed to open prompt file: " + path;
    return false;
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  out_text = buffer.str();
  if (!in.good() && !in.eof()) {
    error_message = "Failed to read prompt file: " + path;
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char ** argv) {
  std::string profile_path = "configs/qwen3_5_0_8b.profile.json";
  std::string hf_model_dir;
  std::string prompt_tokens_csv;
  std::string prompt_text;
  std::string prompt_file_path;
  std::string chat_user_text;
  std::string stop_tokens_csv;
  std::vector<std::string> stop_texts;
  bool stop_on_im_end = false;
  std::string profile_json_path;
  qwen35x::RuntimeTarget target;
  bool bench_bf16 = false;
  bool infer_reference = false;
  bool infer_gpu = false;
  bool gpu_decode_backend_explicit = false;
  qwen35x::Bf16TensorBenchOptions bench_options;
  qwen35x::ReferenceInferenceOptions infer_options;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--profile" && i + 1 < argc) {
      profile_path = argv[++i];
    } else if (arg == "--hf-model-dir" && i + 1 < argc) {
      hf_model_dir = argv[++i];
      bench_options.model_dir = hf_model_dir;
    } else if (arg == "--bench-bf16") {
      bench_bf16 = true;
    } else if (arg == "--infer-reference") {
      infer_reference = true;
    } else if (arg == "--infer-gpu") {
      infer_reference = true;
      infer_gpu = true;
      infer_options.use_cuda = true;
      infer_options.use_cuda_matvec_bf16 = true;
    } else if (arg == "--bench-tensor" && i + 1 < argc) {
      bench_options.tensor_name = argv[++i];
    } else if (arg == "--bench-warmup" && i + 1 < argc) {
      bench_options.warmup_iterations = std::stoi(argv[++i]);
    } else if (arg == "--bench-iters" && i + 1 < argc) {
      bench_options.benchmark_iterations = std::stoi(argv[++i]);
    } else if (arg == "--prompt-tokens" && i + 1 < argc) {
      prompt_tokens_csv = argv[++i];
    } else if (arg == "--prompt-text" && i + 1 < argc) {
      prompt_text = argv[++i];
    } else if (arg == "--prompt-file" && i + 1 < argc) {
      prompt_file_path = argv[++i];
    } else if (arg == "--chat-user" && i + 1 < argc) {
      chat_user_text = argv[++i];
    } else if (arg == "--max-new-tokens" && i + 1 < argc) {
      infer_options.max_new_tokens = std::stoi(argv[++i]);
    } else if (arg == "--max-context" && i + 1 < argc) {
      infer_options.max_context = std::stoi(argv[++i]);
    } else if (arg == "--temperature" && i + 1 < argc) {
      infer_options.sampling.temperature = std::stof(argv[++i]);
    } else if (arg == "--top-p" && i + 1 < argc) {
      infer_options.sampling.top_p = std::stof(argv[++i]);
    } else if (arg == "--top-k" && i + 1 < argc) {
      infer_options.sampling.top_k = std::stoi(argv[++i]);
    } else if (arg == "--repeat-penalty" && i + 1 < argc) {
      infer_options.sampling.repetition_penalty = std::stof(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      infer_options.sampling.seed = std::stoll(argv[++i]);
    } else if (arg == "--gpu-bf16") {
      infer_options.use_cuda_matvec_bf16 = true;
    } else if (arg == "--gpu-f32-matvec") {
      infer_options.use_cuda_matvec_bf16 = false;
    } else if (arg == "--gpu-decode-backend" && i + 1 < argc) {
      const std::string backend = argv[++i];
      gpu_decode_backend_explicit = true;
      if (backend == "default") {
        infer_options.gpu_decode_backend = qwen35x::GpuDecodeBackend::runtime_default;
      } else if (backend == "luce") {
        infer_options.gpu_decode_backend = qwen35x::GpuDecodeBackend::luce;
      } else {
        std::cerr << "unknown --gpu-decode-backend value: " << backend << " (expected: default|luce)\n";
        return 11;
      }
    } else if (arg == "--gpu-decode-blocks" && i + 1 < argc) {
      infer_options.gpu_decode_blocks = std::stoi(argv[++i]);
    } else if (arg == "--luce-prefill-mode" && i + 1 < argc) {
      const std::string mode = argv[++i];
      if (mode == "replay") {
        infer_options.luce_prefill_mode = qwen35x::LucePrefillMode::replay;
      } else if (mode == "batched") {
        infer_options.luce_prefill_mode = qwen35x::LucePrefillMode::batched;
      } else {
        std::cerr << "unknown --luce-prefill-mode value: " << mode << " (expected: replay|batched)\n";
        return 11;
      }
    } else if (arg == "--profile-sync") {
      infer_options.profile_cuda_sync = true;
    } else if (arg == "--luce-profile") {
      infer_options.profile_luce = true;
    } else if (arg == "--prefill-only") {
      infer_options.prefill_only = true;
    } else if (arg == "--stop-token" && i + 1 < argc) {
      stop_tokens_csv = argv[++i];
    } else if (arg == "--stop-text" && i + 1 < argc) {
      stop_texts.push_back(argv[++i]);
    } else if (arg == "--stop-on-im-end") {
      stop_on_im_end = true;
    } else if (arg == "--profile-json" && i + 1 < argc) {
      profile_json_path = argv[++i];
    } else if (arg == "--sm" && i + 1 < argc) {
      target.sm_version = std::stoi(argv[++i]);
    } else if (arg == "--cpu") {
      target.cuda_enabled = false;
      target.sm_version = 0;
      target.gpu_name = "cpu";
      target.vram_mb = 0;
    } else if (arg == "--help") {
      std::cout << "usage: qwen35x [--profile <json>] [--hf-model-dir <path>] [--sm <int>] [--cpu]\n";
      std::cout << "       qwen35x --bench-bf16 --hf-model-dir <path> [--bench-tensor <name>] [--bench-warmup <n>] [--bench-iters <n>]\n";
      std::cout << "       qwen35x --infer-reference --hf-model-dir <path> (--prompt-tokens <csv> | --prompt-text <text> | --prompt-file <path> | --chat-user <text>) [--max-new-tokens <n>] [--max-context <n>]\n";
      std::cout << "       qwen35x --infer-gpu --hf-model-dir <path> (--prompt-tokens <csv> | --prompt-text <text> | --prompt-file <path> | --chat-user <text>) [--max-new-tokens <n>] [--max-context <n>]\n";
      std::cout << "               [--temperature <float>] [--top-p <float>] [--top-k <int>] [--repeat-penalty <float>] [--seed <int64>]\n";
      std::cout << "               [--gpu-bf16|--gpu-f32-matvec] [--gpu-decode-backend <default|luce>] [--gpu-decode-blocks <n>] [--luce-prefill-mode <replay|batched>] [--profile-sync] [--luce-profile] [--prefill-only]\n";
      std::cout << "               [--stop-token <csv>] [--stop-text <text>] [--stop-on-im-end] [--profile-json <path>]\n";
      return 0;
    }
  }

  if (infer_gpu && !gpu_decode_backend_explicit) {
    infer_options.gpu_decode_backend = qwen35x::GpuDecodeBackend::luce;
  }

  if (bench_bf16) {
    if (bench_options.model_dir.empty()) {
      bench_options.model_dir = "models/qwen3.5-0.8b";
    }

    std::string error_message;
    qwen35x::Bf16TensorBenchResult bench_result;
    if (!qwen35x::run_bf16_tensor_benchmark(bench_options, bench_result, error_message)) {
      std::cerr << "bf16 benchmark failed: " << error_message << "\n";
      return 10;
    }

    std::cout << "bf16 tensor benchmark\n";
    std::cout << "  tensor: " << bench_result.tensor_name << "\n";
    std::cout << "  file: " << bench_result.tensor_file << "\n";
    std::cout << "  dtype: " << bench_result.dtype << "\n";
    std::cout << "  shape: ";
    for (std::size_t i = 0; i < bench_result.shape.size(); ++i) {
      if (i > 0) {
        std::cout << "x";
      }
      std::cout << bench_result.shape[i];
    }
    std::cout << "\n";
    std::cout << "  avg iteration: " << bench_result.avg_iteration_ms << " ms\n";
    std::cout << "  matvec/s: " << bench_result.matvec_per_second << "\n";
    std::cout << "  effective gflops: " << bench_result.gflops << "\n";
    return 0;
  }

  if (infer_reference) {
    if (hf_model_dir.empty()) {
      hf_model_dir = "models/qwen3.5-0.8b";
    }
    infer_options.model_dir = hf_model_dir;

    std::string error_message;
    qwen35x::QwenTokenizer tokenizer;
    bool has_tokenizer = false;
    const bool need_tokenizer =
      !prompt_text.empty() || !prompt_file_path.empty() || !chat_user_text.empty() || !stop_texts.empty() || stop_on_im_end;
    if (need_tokenizer) {
      if (!qwen35x::QwenTokenizer::load_from_hf_directory(hf_model_dir, tokenizer, error_message)) {
        std::cerr << "tokenizer load failed: " << error_message << "\n";
        return 11;
      }
      has_tokenizer = true;
    }

    if (!stop_tokens_csv.empty()) {
      if (!qwen35x::parse_token_list_csv(stop_tokens_csv, infer_options.stop_token_ids, error_message)) {
        std::cerr << "stop-token parse failed: " << error_message << "\n";
        return 11;
      }
    }

    for (const auto & stop_text : stop_texts) {
      std::vector<std::int32_t> stop_sequence;
      if (!has_tokenizer) {
        std::cerr << "stop-text parse failed: tokenizer is not loaded.\n";
        return 11;
      }
      if (!tokenizer.encode(stop_text, stop_sequence, error_message)) {
        std::cerr << "stop-text parse failed: " << error_message << "\n";
        return 11;
      }
      if (stop_sequence.empty()) {
        std::cerr << "stop-text parse failed: empty stop-text tokenization.\n";
        return 11;
      }
      infer_options.stop_token_sequences.push_back(std::move(stop_sequence));
    }

    if (stop_on_im_end) {
      std::int32_t im_end_id = 248046;
      if (has_tokenizer) {
        const auto token_id = tokenizer.token_to_id("<|im_end|>");
        if (token_id) {
          im_end_id = *token_id;
        }
      }
      infer_options.stop_token_ids.push_back(im_end_id);
    }

    const int prompt_modes =
      (!prompt_tokens_csv.empty() ? 1 : 0) + (!prompt_text.empty() ? 1 : 0) + (!prompt_file_path.empty() ? 1 : 0) +
      (!chat_user_text.empty() ? 1 : 0);
    if (prompt_modes > 1) {
      std::cerr << "prompt parse failed: provide only one of --prompt-tokens, --prompt-text, --prompt-file, or --chat-user.\n";
      return 11;
    }
    if (!prompt_file_path.empty()) {
      if (!read_text_file(prompt_file_path, prompt_text, error_message)) {
        std::cerr << "prompt file read failed: " << error_message << "\n";
        return 11;
      }
      if (prompt_text.empty()) {
        std::cerr << "prompt file read failed: prompt file is empty.\n";
        return 11;
      }
    }
    if (!chat_user_text.empty()) {
      prompt_text = "<|im_start|>user\n" + chat_user_text + "<|im_end|>\n<|im_start|>assistant\n";
    }

    if (!prompt_text.empty()) {
      if (!prompt_tokens_csv.empty()) {
        std::cerr << "prompt parse failed: prompt mode conflict.\n";
        return 11;
      }
      if (!tokenizer.encode(prompt_text, infer_options.prompt_tokens, error_message)) {
        std::cerr << "prompt tokenize failed: " << error_message << "\n";
        return 11;
      }
      if (infer_options.prompt_tokens.empty()) {
        std::cerr << "prompt tokenize failed: prompt text produced zero tokens.\n";
        return 11;
      }
    } else {
      if (!qwen35x::parse_token_list_csv(prompt_tokens_csv, infer_options.prompt_tokens, error_message)) {
        std::cerr << "prompt parse failed: " << error_message << "\n";
        return 11;
      }
    }

    const auto profile = qwen35x::ProfileLoader::load_from_hf_directory(hf_model_dir, error_message);
    if (!profile) {
      std::cerr << "profile load failed: " << error_message << "\n";
      return 12;
    }

    qwen35x::ReferenceInferenceResult infer_result;
    if (!qwen35x::run_reference_qwen35_inference(*profile, infer_options, infer_result, error_message)) {
      std::cerr << "reference inference failed: " << error_message << "\n";
      return 13;
    }

    std::cout << "reference inference\n";
    std::cout << "  backend: " << (infer_options.use_cuda ? "cuda-hybrid" : "cpu-reference") << "\n";
    std::cout << "  decode_backend: " << gpu_decode_backend_name(infer_options.gpu_decode_backend) << "\n";
    std::cout << "  luce_prefill_mode: " << luce_prefill_mode_name(infer_options.luce_prefill_mode) << "\n";
    std::cout << "  prefill_only: " << (infer_options.prefill_only ? "on" : "off") << "\n";
    std::cout << "  prompt_tokens: " << infer_options.prompt_tokens.size() << "\n";
    std::cout << "  generated_tokens: " << infer_result.generated_tokens.size() << "\n";
    std::cout << "  load_time_ms: " << infer_result.load_time_ms << "\n";
    std::cout << "  prefill_time_ms: " << infer_result.prefill_time_ms << "\n";
    std::cout << "  prefill_tokens_per_second: " << infer_result.prefill_tokens_per_second << "\n";
    std::cout << "  decode_time_ms: " << infer_result.decode_time_ms << "\n";
    std::cout << "  tokens_per_second: " << infer_result.tokens_per_second << "\n";
    std::cout << "  stage_ms: embedding=" << infer_result.timing_breakdown.embedding_ms
              << " attention=" << infer_result.timing_breakdown.attention_ms
              << " mlp=" << infer_result.timing_breakdown.mlp_ms
              << " logits=" << infer_result.timing_breakdown.logits_ms
              << " sampling=" << infer_result.timing_breakdown.sampling_ms
              << " stop_checks=" << infer_result.timing_breakdown.stop_checks_ms << "\n";
    std::cout << "  cuda_transfers: h2d_bytes=" << infer_result.transfer_breakdown.host_to_device_bytes
              << " d2h_bytes=" << infer_result.transfer_breakdown.device_to_host_bytes
              << " copy_calls=" << infer_result.transfer_breakdown.copy_calls
              << " h2d_per_forward_token=" << infer_result.host_to_device_bytes_per_forward_token
              << " d2h_per_forward_token=" << infer_result.device_to_host_bytes_per_forward_token << "\n";
    std::cout << "  sampling: temperature=" << infer_options.sampling.temperature
              << " top_p=" << infer_options.sampling.top_p
              << " top_k=" << infer_options.sampling.top_k
              << " repeat_penalty=" << infer_options.sampling.repetition_penalty
              << " seed=" << infer_options.sampling.seed << "\n";
    std::cout << "  cuda: matvec_bf16=" << (infer_options.use_cuda_matvec_bf16 ? "on" : "off")
              << " profile_sync=" << (infer_options.profile_cuda_sync ? "on" : "off") << "\n";
    std::cout << "  stop: token_ids=" << infer_options.stop_token_ids.size()
              << " token_sequences=" << infer_options.stop_token_sequences.size() << "\n";
    std::cout << "  output_token_ids:";
    for (const auto token : infer_result.generated_tokens) {
      std::cout << " " << token;
    }
    std::cout << "\n";

    std::string generated_text;
    if (has_tokenizer) {
      if (tokenizer.decode(infer_result.generated_tokens, generated_text, error_message)) {
        std::cout << "  generated_text: " << generated_text << "\n";
      } else {
        std::cerr << "token decode warning: " << error_message << "\n";
      }
    }
    if (!profile_json_path.empty()) {
      if (!write_profile_json(
            profile_json_path,
            infer_options,
            infer_result,
            generated_text,
            infer_options.use_cuda ? "cuda-hybrid" : "cpu-reference",
            gpu_decode_backend_name(infer_options.gpu_decode_backend),
            error_message)) {
        std::cerr << "profile json write failed: " << error_message << "\n";
        return 14;
      }
      std::cout << "  profile_json: " << profile_json_path << "\n";
    }
    return 0;
  }

  std::string error_message;
  const auto profile = hf_model_dir.empty()
                         ? qwen35x::ProfileLoader::load_from_json(profile_path, error_message)
                         : qwen35x::ProfileLoader::load_from_hf_directory(hf_model_dir, error_message);
  if (!profile) {
    std::cerr << "profile load failed: " << error_message << "\n";
    return 1;
  }

  std::cout << "model: " << profile->model_id << "\n";
  std::cout << "family: " << profile->family << " variant: " << profile->variant << "\n";
  std::cout << "fingerprint: " << qwen35x::fingerprint_summary(profile->fingerprint) << "\n";
  std::cout << "weights: shards=" << profile->weights.shard_files.size()
            << " total_bytes=" << profile->weights.total_size_bytes << "\n";

  qwen35x::EngineRuntime runtime;
  if (!runtime.initialize(*profile, target, error_message)) {
    std::cerr << "runtime init failed: " << error_message << "\n";
    return 2;
  }

  std::cout << "target: " << target.gpu_name
            << " sm=" << target.sm_version
            << " cuda=" << (target.cuda_enabled ? "on" : "off")
            << " vram_mb=" << target.vram_mb << "\n";
  runtime.print_dispatch_table(std::cout);

  std::cout << "\nnext: implement packer + real decode/prefill kernels.\n";
  return 0;
}
