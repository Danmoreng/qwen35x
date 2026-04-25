#pragma once

#include "qwen35x/runtime/qwen35x_profile.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qwen35x::cuda_backend {

struct Qwen35xCudaBackendConfig {
  std::string model_dir = "models/qwen3.5-0.8b";
  int max_context = 256;
  int decode_blocks = 0;
  float repetition_penalty = 1.0f;
  bool profile_enabled = false;
};

class Qwen35xCudaBackend {
public:
  Qwen35xCudaBackend();
  ~Qwen35xCudaBackend();

  Qwen35xCudaBackend(const Qwen35xCudaBackend &) = delete;
  Qwen35xCudaBackend & operator=(const Qwen35xCudaBackend &) = delete;
  Qwen35xCudaBackend(Qwen35xCudaBackend &&) noexcept;
  Qwen35xCudaBackend & operator=(Qwen35xCudaBackend &&) noexcept;

  bool initialize(const Qwen35xCudaBackendConfig & config, std::string & error_message);
  bool reset(std::string & error_message);
  bool run_prefill(
    const std::vector<std::int32_t> & tokens,
    int & out_first_token,
    std::string & error_message);
  bool run_prefill_only(
    const std::vector<std::int32_t> & tokens,
    std::string & error_message);
  bool run_decode_step(
    int input_token,
    int position,
    int & out_next_token,
    std::string & error_message);
  bool synchronize(std::string & error_message);

  bool is_initialized() const;
  int max_context() const;
  Qwen35xRuntimeProfile profile() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

int query_max_safe_decode_blocks();
void set_decode_blocks_override(int blocks);

} // namespace qwen35x::cuda_backend
