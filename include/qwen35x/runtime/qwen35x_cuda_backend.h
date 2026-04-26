#pragma once

#include "qwen35x/common/model_profile.h"
#include "qwen35x/runtime/qwen35x_profile.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace qwen35x::cuda_backend {

enum class Qwen35xWeightPrecision {
  bf16 = 0,
  nvfp4 = 1,
};

enum class Qwen35xCachePrecision {
  bf16 = 0,
  quantized = 1,
};

struct Qwen35xModelDescriptor {
  std::string family;
  std::string variant;
  int num_layers = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int vocab_size = 0;
  int fa_num_q_heads = 0;
  int fa_num_kv_heads = 0;
  int fa_head_dim = 0;
  int fa_rot_dim = 0;
  float rope_theta = 0.0f;
  int dn_num_heads = 0;
  int dn_gate_heads = 0;
  int dn_key_dim = 0;
  int dn_value_head_dim = 0;
  int dn_value_dim = 0;
  int dn_conv_kernel = 0;
  std::vector<int> layer_type;
};

struct Qwen35xCudaBackendConfig {
  std::string model_dir = "models/qwen3.5-0.8b";
  std::optional<Qwen35xModelDescriptor> model_descriptor;
  int max_context = 256;
  int decode_blocks = 0;
  float repetition_penalty = 1.0f;
  Qwen35xWeightPrecision weight_precision = Qwen35xWeightPrecision::bf16;
  Qwen35xCachePrecision cache_precision = Qwen35xCachePrecision::bf16;
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
const char * to_string(Qwen35xWeightPrecision precision);
const char * to_string(Qwen35xCachePrecision precision);
bool build_model_descriptor(
  const qwen35x::ModelProfile & profile,
  Qwen35xModelDescriptor & descriptor,
  std::string & error_message);

} // namespace qwen35x::cuda_backend
