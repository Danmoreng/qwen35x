#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/reference_inference.h"
#include "qwen35x/runtime/runtime.h"
#include "qwen35x/tokenizer/tokenizer.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
  std::string profile_path = "configs/qwen3_5_0_8b.profile.json";
  std::string hf_model_dir;
  std::string prompt_tokens_csv;
  std::string prompt_text;
  std::string chat_user_text;
  std::string stop_tokens_csv;
  std::vector<std::string> stop_texts;
  bool stop_on_im_end = false;
  qwen35x::RuntimeTarget target;
  bool bench_bf16 = false;
  bool infer_reference = false;
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
      infer_options.use_cuda = true;
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
    } else if (arg == "--stop-token" && i + 1 < argc) {
      stop_tokens_csv = argv[++i];
    } else if (arg == "--stop-text" && i + 1 < argc) {
      stop_texts.push_back(argv[++i]);
    } else if (arg == "--stop-on-im-end") {
      stop_on_im_end = true;
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
      std::cout << "       qwen35x --infer-reference --hf-model-dir <path> (--prompt-tokens <csv> | --prompt-text <text> | --chat-user <text>) [--max-new-tokens <n>] [--max-context <n>]\n";
      std::cout << "       qwen35x --infer-gpu --hf-model-dir <path> (--prompt-tokens <csv> | --prompt-text <text> | --chat-user <text>) [--max-new-tokens <n>] [--max-context <n>]\n";
      std::cout << "               [--temperature <float>] [--top-p <float>] [--top-k <int>] [--repeat-penalty <float>] [--seed <int64>]\n";
      std::cout << "               [--stop-token <csv>] [--stop-text <text>] [--stop-on-im-end]\n";
      return 0;
    }
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
      !prompt_text.empty() || !chat_user_text.empty() || !stop_texts.empty() || stop_on_im_end;
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

    const int prompt_modes = (!prompt_tokens_csv.empty() ? 1 : 0) + (!prompt_text.empty() ? 1 : 0) + (!chat_user_text.empty() ? 1 : 0);
    if (prompt_modes > 1) {
      std::cerr << "prompt parse failed: provide only one of --prompt-tokens, --prompt-text, or --chat-user.\n";
      return 11;
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
    std::cout << "  prompt_tokens: " << infer_options.prompt_tokens.size() << "\n";
    std::cout << "  generated_tokens: " << infer_result.generated_tokens.size() << "\n";
    std::cout << "  load_time_ms: " << infer_result.load_time_ms << "\n";
    std::cout << "  decode_time_ms: " << infer_result.decode_time_ms << "\n";
    std::cout << "  tokens_per_second: " << infer_result.tokens_per_second << "\n";
    std::cout << "  sampling: temperature=" << infer_options.sampling.temperature
              << " top_p=" << infer_options.sampling.top_p
              << " top_k=" << infer_options.sampling.top_k
              << " repeat_penalty=" << infer_options.sampling.repetition_penalty
              << " seed=" << infer_options.sampling.seed << "\n";
    std::cout << "  stop: token_ids=" << infer_options.stop_token_ids.size()
              << " token_sequences=" << infer_options.stop_token_sequences.size() << "\n";
    std::cout << "  output_token_ids:";
    for (const auto token : infer_result.generated_tokens) {
      std::cout << " " << token;
    }
    std::cout << "\n";

    if (has_tokenizer) {
      std::string generated_text;
      if (tokenizer.decode(infer_result.generated_tokens, generated_text, error_message)) {
        std::cout << "  generated_text: " << generated_text << "\n";
      } else {
        std::cerr << "token decode warning: " << error_message << "\n";
      }
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
