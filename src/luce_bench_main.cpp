#include "qwen35x/runtime/luce_decode_backend.h"
#include "qwen35x/tokenizer/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

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
  if (options.max_context <= 0) {
    error_message = "max-context must be > 0";
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

double mean(const std::vector<double> & values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

} // namespace

int main(int argc, char ** argv) {
  BenchmarkOptions options;
  std::string error_message;
  const bool parse_ok = parse_args(argc, argv, options, error_message);
  if (!parse_ok) {
    if (!error_message.empty()) {
      std::cerr << "argument parse failed: " << error_message << "\n";
      return 2;
    }
    return 0;
  }

  const int max_safe_decode_blocks = qwen35x::luce::query_max_safe_decode_blocks();
  if (options.query_decode_blocks_only) {
    std::cout << "max_safe_decode_blocks: " << max_safe_decode_blocks << "\n";
    return 0;
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

  qwen35x::luce::LuceDecodeBackend backend;
  qwen35x::luce::LuceDecodeBackendConfig backend_config;
  backend_config.model_dir = options.model_dir;
  backend_config.max_context = options.max_context;
  backend_config.decode_blocks = options.decode_blocks;
  if (!backend.initialize(backend_config, error_message)) {
    std::cerr << "luce backend init failed: " << error_message << "\n";
    return 5;
  }

  const auto load_end = std::chrono::steady_clock::now();
  const double load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

  std::optional<std::int32_t> eos_token = tokenizer.token_to_id("<|endoftext|>");

  for (int warm = 0; warm < options.warmup_runs; ++warm) {
    if (!backend.reset(error_message)) {
      std::cerr << "warmup reset failed: " << error_message << "\n";
      return 6;
    }
    int first_token = 0;
    if (!backend.run_prefill(prefill_tokens, first_token, error_message)) {
      std::cerr << "warmup prefill failed: " << error_message << "\n";
      return 6;
    }
    if (!backend.synchronize(error_message)) {
      std::cerr << "warmup sync failed: " << error_message << "\n";
      return 6;
    }

    if (!backend.reset(error_message)) {
      std::cerr << "warmup reset failed: " << error_message << "\n";
      return 6;
    }
    if (!backend.run_prefill(prompt_tokens, first_token, error_message)) {
      std::cerr << "warmup prefill failed: " << error_message << "\n";
      return 6;
    }
    int token = first_token;
    int position = static_cast<int>(prompt_tokens.size());
    for (int step = 0; step < options.max_new_tokens; ++step) {
      int next = 0;
      if (!backend.run_decode_step(token, position, next, error_message)) {
        std::cerr << "warmup decode failed: " << error_message << "\n";
        return 6;
      }
      token = next;
      ++position;
      if (eos_token && token == *eos_token) {
        break;
      }
    }
    if (!backend.synchronize(error_message)) {
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
    if (!backend.reset(error_message)) {
      std::cerr << "run reset failed: " << error_message << "\n";
      return 7;
    }

    if (!backend.synchronize(error_message)) {
      std::cerr << "run sync failed: " << error_message << "\n";
      return 7;
    }
    const auto pp_t0 = std::chrono::steady_clock::now();
    int pp_first = 0;
    if (!backend.run_prefill(prefill_tokens, pp_first, error_message)) {
      std::cerr << "run prefill failed: " << error_message << "\n";
      return 7;
    }
    if (!backend.synchronize(error_message)) {
      std::cerr << "run sync failed: " << error_message << "\n";
      return 7;
    }
    const auto pp_t1 = std::chrono::steady_clock::now();
    const double pp_ms = std::chrono::duration<double, std::milli>(pp_t1 - pp_t0).count();
    const double pp_tps = static_cast<double>(prefill_tokens.size()) / (pp_ms / 1000.0);
    pp_ms_runs.push_back(pp_ms);
    pp_tps_runs.push_back(pp_tps);

    if (!backend.reset(error_message)) {
      std::cerr << "run reset failed: " << error_message << "\n";
      return 7;
    }
    int first = 0;
    if (!backend.run_prefill(prompt_tokens, first, error_message)) {
      std::cerr << "run prefill failed: " << error_message << "\n";
      return 7;
    }

    if (!backend.synchronize(error_message)) {
      std::cerr << "run sync failed: " << error_message << "\n";
      return 7;
    }
    last_tokens.clear();
    int token = first;
    int position = static_cast<int>(prompt_tokens.size());
    const auto tg_t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < options.max_new_tokens; ++step) {
      int next = 0;
      if (!backend.run_decode_step(token, position, next, error_message)) {
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
    if (!backend.synchronize(error_message)) {
      std::cerr << "run sync failed: " << error_message << "\n";
      return 7;
    }
    const auto tg_t1 = std::chrono::steady_clock::now();

    const double tg_ms = std::chrono::duration<double, std::milli>(tg_t1 - tg_t0).count();
    const double tg_tps = last_tokens.empty() ? 0.0 : static_cast<double>(last_tokens.size()) / (tg_ms / 1000.0);
    tg_ms_runs.push_back(tg_ms);
    tg_tps_runs.push_back(tg_tps);

    std::cout << "run " << (run + 1) << "/" << options.runs
              << " prefill_tps=" << pp_tps
              << " decode_tps=" << tg_tps
              << " generated=" << last_tokens.size() << "\n";
  }

  BenchmarkResult result;
  result.load_time_ms = load_time_ms;
  result.avg_pp_time_ms = mean(pp_ms_runs);
  result.avg_pp_tokens_per_second = mean(pp_tps_runs);
  result.avg_tg_time_ms = mean(tg_ms_runs);
  result.avg_tg_tokens_per_second = mean(tg_tps_runs);
  result.prompt_tokens = static_cast<int>(prompt_tokens.size());
  result.prefill_prompt_tokens = static_cast<int>(prefill_tokens.size());
  result.generated_tokens = static_cast<int>(last_tokens.size());
  result.output_tokens = last_tokens;

  std::string out_text;
  if (!last_tokens.empty()) {
    tokenizer.decode(last_tokens, out_text, error_message);
  }

  const int effective_decode_blocks = options.decode_blocks > 0 ? options.decode_blocks : max_safe_decode_blocks;
  std::cout << "luce megakernel benchmark\n";
  std::cout << "  model_dir: " << options.model_dir << "\n";
  std::cout << "  prompt_tokens: " << result.prompt_tokens << "\n";
  std::cout << "  prefill_prompt_tokens: " << result.prefill_prompt_tokens << "\n";
  std::cout << "  generated_tokens: " << result.generated_tokens << "\n";
  std::cout << "  load_time_ms: " << result.load_time_ms << "\n";
  std::cout << "  avg_pp_time_ms: " << result.avg_pp_time_ms << "\n";
  std::cout << "  avg_pp_tokens_per_second: " << result.avg_pp_tokens_per_second << "\n";
  std::cout << "  avg_tg_time_ms: " << result.avg_tg_time_ms << "\n";
  std::cout << "  avg_tg_tokens_per_second: " << result.avg_tg_tokens_per_second << "\n";
  std::cout << "  decode_blocks: " << effective_decode_blocks << "\n";
  std::cout << "  max_safe_decode_blocks: " << max_safe_decode_blocks << "\n";
  std::cout << "  output_token_ids:";
  for (const auto token : result.output_tokens) {
    std::cout << " " << token;
  }
  std::cout << "\n";
  if (!out_text.empty()) {
    std::cout << "  output_text: " << out_text << "\n";
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
