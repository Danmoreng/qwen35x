#include "qwen35x/runtime/reference_inference.h"

#include "qwen35x/runtime/cuda_inference.h"
#include "qwen35x/weights/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_set>

namespace qwen35x {

namespace {

struct TensorData {
  std::vector<std::int64_t> shape;
  std::vector<float> data;
  cuda::CudaDeviceMatrixF32 device_matrix;
  bool has_device_matrix = false;
};

struct FullAttentionWeights {
  TensorData q_proj;
  TensorData k_proj;
  TensorData v_proj;
  TensorData o_proj;
  TensorData q_norm;
  TensorData k_norm;
  cuda::CudaDeviceMatrixF32 qkv_proj_device;
  bool has_device_qkv_proj = false;
  cuda::CudaDeviceMatrixF32 kv_proj_device;
  bool has_device_kv_proj = false;
  cuda::CudaDeviceBufferF32 q_norm_device;
  cuda::CudaDeviceBufferF32 k_norm_device;
  bool has_device_norm = false;
};

struct LinearAttentionWeights {
  TensorData in_proj_qkv;
  TensorData in_proj_z;
  TensorData in_proj_b;
  TensorData in_proj_a;
  TensorData conv1d;
  TensorData out_proj;
  TensorData norm;
  TensorData a_log;
  TensorData dt_bias;
  std::vector<float> ssm_a;
  cuda::CudaDeviceMatrixF32 in_proj_all_device;
  bool has_device_in_proj_all = false;
  cuda::CudaDeviceBufferF32 norm_device;
  cuda::CudaDeviceBufferF32 dt_bias_device;
  cuda::CudaDeviceBufferF32 ssm_a_device;
  bool has_device_params = false;
};

struct LayerWeights {
  TensorData input_layernorm;
  TensorData post_attention_layernorm;
  TensorData mlp_gate;
  TensorData mlp_up;
  TensorData mlp_down;
  cuda::CudaDeviceMatrixF32 mlp_gate_up_device;
  bool has_device_mlp_gate_up = false;
  cuda::CudaDeviceBufferF32 input_layernorm_device;
  cuda::CudaDeviceBufferF32 post_attention_layernorm_device;
  bool has_device_norms = false;
  bool is_linear = false;
  FullAttentionWeights full;
  LinearAttentionWeights linear;
};

struct ModelWeights {
  TensorData embed_tokens;
  TensorData final_norm;
  cuda::CudaDeviceBufferF32 final_norm_device;
  bool has_device_final_norm = false;
  std::vector<LayerWeights> layers;
};

struct RuntimeDims {
  int n_layers = 0;
  int hidden = 0;
  int intermediate = 0;
  int vocab_size = 0;

  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  int rope_dim = 0;
  float rope_theta = 10000000.0f;
  float rms_eps = 1.0e-6f;

  int linear_kernel = 0;
  int linear_num_k_heads = 0;
  int linear_num_v_heads = 0;
  int linear_head_k_dim = 0;
  int linear_head_v_dim = 0;
  int linear_q_dim = 0;
  int linear_v_dim = 0;
  int linear_conv_channels = 0;
};

struct FullAttentionState {
  std::vector<float> k_cache;
  std::vector<float> v_cache;
  cuda::CudaDeviceBufferF32 k_cache_device;
  cuda::CudaDeviceBufferF32 v_cache_device;
  cuda::CudaDeviceBufferBF16 k_cache_bf16_device;
  cuda::CudaDeviceBufferBF16 v_cache_bf16_device;
  bool has_device_state = false;
  bool has_device_state_bf16 = false;
};

struct LinearAttentionState {
  std::vector<float> conv_state;
  std::vector<float> recurrent_state;
  cuda::CudaDeviceBufferF32 conv_state_device;
  cuda::CudaDeviceBufferF32 recurrent_state_device;
  bool has_device_state = false;
};

struct ModelState {
  std::vector<FullAttentionState> full_states;
  std::vector<LinearAttentionState> linear_states;
};

struct CudaForwardWorkspace {
  cuda::CudaDeviceBufferF32 hidden_in;
  cuda::CudaDeviceBufferBF16 hidden_in_bf16;
  cuda::CudaDeviceBufferF32 intermediate_a;
  cuda::CudaDeviceBufferBF16 intermediate_a_bf16;
  cuda::CudaDeviceBufferF32 intermediate_b;
  cuda::CudaDeviceBufferBF16 intermediate_b_bf16;
  cuda::CudaDeviceBufferF32 intermediate_hidden;
  cuda::CudaDeviceBufferBF16 intermediate_hidden_bf16;
  cuda::CudaDeviceBufferF32 hidden_out;
  cuda::CudaDeviceBufferBF16 hidden_out_bf16;
  cuda::CudaDeviceBufferF32 projection_out;
  cuda::CudaDeviceBufferBF16 projection_out_bf16;
  cuda::CudaDeviceBufferF32 full_q;
  cuda::CudaDeviceBufferBF16 full_q_bf16;
  cuda::CudaDeviceBufferF32 full_gate;
  cuda::CudaDeviceBufferF32 full_attn;
  cuda::CudaDeviceBufferBF16 full_attn_bf16;
  cuda::CudaDeviceBufferF32 full_scores;
  cuda::CudaDeviceBufferF32 linear_mixed_qkv;
  cuda::CudaDeviceBufferF32 linear_z;
  cuda::CudaDeviceBufferF32 linear_b;
  cuda::CudaDeviceBufferF32 linear_a;
  cuda::CudaDeviceBufferF32 linear_conv_out;
  cuda::CudaDeviceBufferF32 linear_gated_norm;
  cuda::CudaDeviceBufferF32 logits;
  cuda::CudaDeviceBufferF32 seen_token_mask;
  cuda::CudaDeviceBufferF32 topk_values_scratch;
  cuda::CudaDeviceBufferF32 topk_indices_scratch;
  std::vector<cuda::CudaCapturedGraph> mlp_graphs;
  std::vector<bool> mlp_graph_warmup_done;
  std::vector<bool> mlp_graph_disabled;
  std::vector<cuda::CudaCapturedGraph> linear_attention_graphs;
  std::vector<bool> linear_attention_graph_warmup_done;
  std::vector<bool> linear_attention_graph_disabled;
  bool has_gpu_sampling_buffers = false;
  bool has_device_buffers = false;
};

struct DecodeProfilingAccumulator {
  double embedding_ms = 0.0;
  double attention_ms = 0.0;
  double mlp_ms = 0.0;
  double logits_ms = 0.0;
  double sampling_ms = 0.0;
  double stop_checks_ms = 0.0;
  int forward_pass_tokens = 0;
};

double elapsed_ms(const std::chrono::steady_clock::time_point start_time) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
}

constexpr int kCudaSamplingMaxTopK = 64;

bool maybe_sync_cuda_for_stage_timing(
  const bool use_cuda,
  const bool profile_cuda_sync,
  std::string & error_message) {
  if (!use_cuda || !profile_cuda_sync) {
    return true;
  }
  return cuda::synchronize_stream(error_message);
}

bool project_from_uploaded_hidden_cuda(
  const TensorData & projection,
  CudaForwardWorkspace & workspace,
  const std::size_t output_count,
  std::vector<float> & out_values,
  std::string & error_message);

float sigmoidf_stable(const float x) {
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

float siluf(const float x) {
  return x * sigmoidf_stable(x);
}

float softplusf_stable(const float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return std::exp(x);
  }
  return std::log1p(std::exp(x));
}

bool tensor_is_2d(const TensorData & t, std::int64_t rows, std::int64_t cols) {
  return t.shape.size() == 2 && t.shape[0] == rows && t.shape[1] == cols;
}

bool tensor_is_1d(const TensorData & t, std::int64_t size) {
  return t.shape.size() == 1 && t.shape[0] == size;
}

bool tensor_is_conv1d(const TensorData & t, std::int64_t channels, std::int64_t kernel) {
  if (t.shape.size() == 2) {
    return t.shape[0] == channels && t.shape[1] == kernel;
  }
  if (t.shape.size() == 3) {
    return t.shape[0] == channels && t.shape[1] == 1 && t.shape[2] == kernel;
  }
  return false;
}

bool load_tensor_f32(
  const std::string & model_dir,
  const std::string & tensor_name,
  TensorData & out,
  std::string & error_message) {
  SafetensorTensorF32 tensor;
  if (!SafetensorLoader::read_tensor_f32(model_dir, tensor_name, tensor, error_message)) {
    return false;
  }
  out.shape = std::move(tensor.shape);
  out.data = std::move(tensor.data);
  return true;
}

bool load_tensor_checked_2d(
  const std::string & model_dir,
  const std::string & tensor_name,
  std::int64_t rows,
  std::int64_t cols,
  TensorData & out,
  std::string & error_message) {
  if (!load_tensor_f32(model_dir, tensor_name, out, error_message)) {
    return false;
  }
  if (!tensor_is_2d(out, rows, cols)) {
    error_message = "Tensor '" + tensor_name + "' shape mismatch.";
    return false;
  }
  return true;
}

bool load_tensor_checked_1d(
  const std::string & model_dir,
  const std::string & tensor_name,
  std::int64_t size,
  TensorData & out,
  std::string & error_message) {
  if (!load_tensor_f32(model_dir, tensor_name, out, error_message)) {
    return false;
  }
  if (!tensor_is_1d(out, size)) {
    error_message = "Tensor '" + tensor_name + "' shape mismatch.";
    return false;
  }
  return true;
}

bool load_tensor_checked_conv1d(
  const std::string & model_dir,
  const std::string & tensor_name,
  std::int64_t channels,
  std::int64_t kernel,
  TensorData & out,
  std::string & error_message) {
  if (!load_tensor_f32(model_dir, tensor_name, out, error_message)) {
    return false;
  }
  if (!tensor_is_conv1d(out, channels, kernel)) {
    error_message = "Tensor '" + tensor_name + "' shape mismatch.";
    return false;
  }
  if (out.shape.size() == 3) {
    std::vector<float> flattened(static_cast<std::size_t>(channels * kernel));
    for (std::int64_t c = 0; c < channels; ++c) {
      for (std::int64_t k = 0; k < kernel; ++k) {
        const std::size_t src_idx = static_cast<std::size_t>((c * kernel) + k);
        flattened[src_idx] = out.data[src_idx];
      }
    }
    out.shape = {channels, kernel};
    out.data = std::move(flattened);
  }
  return true;
}

bool matvec_2d(
  const TensorData & w,
  const std::vector<float> & x,
  std::vector<float> & out,
  const bool use_cuda,
  std::string & error_message) {
  const int rows = static_cast<int>(w.shape[0]);
  const int cols = static_cast<int>(w.shape[1]);
  if (static_cast<int>(x.size()) != cols) {
    error_message = "matvec input size mismatch.";
    return false;
  }

  if (use_cuda) {
    if (!w.has_device_matrix || w.device_matrix.data == nullptr) {
      error_message = "CUDA matvec requested but device matrix is not uploaded.";
      return false;
    }
    return cuda::run_matvec_f32(w.device_matrix, x, out, error_message);
  }

  out.assign(static_cast<std::size_t>(rows), 0.0f);
  for (int r = 0; r < rows; ++r) {
    const float * row_ptr = w.data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      sum += row_ptr[c] * x[static_cast<std::size_t>(c)];
    }
    out[static_cast<std::size_t>(r)] = sum;
  }
  return true;
}

void rms_norm_qwen3next(
  const std::vector<float> & x,
  const TensorData & weight,
  const float eps,
  std::vector<float> & out) {
  float sq_sum = 0.0f;
  for (const float v : x) {
    sq_sum += v * v;
  }
  const float inv = 1.0f / std::sqrt(sq_sum / static_cast<float>(x.size()) + eps);
  out.resize(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] * inv * (1.0f + weight.data[i]);
  }
}

void rms_norm_per_head_qwen3next(
  const std::vector<float> & x,
  const int num_heads,
  const int head_dim,
  const TensorData & weight,
  const float eps,
  std::vector<float> & out) {
  out.resize(x.size());
  for (int h = 0; h < num_heads; ++h) {
    const std::size_t base = static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim);
    float sq_sum = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      const float v = x[base + static_cast<std::size_t>(d)];
      sq_sum += v * v;
    }
    const float inv = 1.0f / std::sqrt(sq_sum / static_cast<float>(head_dim) + eps);
    for (int d = 0; d < head_dim; ++d) {
      out[base + static_cast<std::size_t>(d)] = x[base + static_cast<std::size_t>(d)] * inv *
                                                (1.0f + weight.data[static_cast<std::size_t>(d)]);
    }
  }
}

void l2_norm_per_head(
  std::vector<float> & x,
  const int num_heads,
  const int head_dim,
  const float eps = 1.0e-6f) {
  for (int h = 0; h < num_heads; ++h) {
    const std::size_t base = static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim);
    float sq_sum = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      const float v = x[base + static_cast<std::size_t>(d)];
      sq_sum += v * v;
    }
    const float inv = 1.0f / std::sqrt(sq_sum + eps);
    for (int d = 0; d < head_dim; ++d) {
      x[base + static_cast<std::size_t>(d)] *= inv;
    }
  }
}

void apply_rope_inplace(
  std::vector<float> & x,
  const int num_heads,
  const int head_dim,
  const int rope_dim,
  const int position,
  const float rope_theta) {
  if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) {
    return;
  }
  const int half = rope_dim / 2;
  for (int h = 0; h < num_heads; ++h) {
    const std::size_t base = static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim);
    for (int i = 0; i < half; ++i) {
      const float inv_freq = std::pow(rope_theta, -static_cast<float>(2 * i) / static_cast<float>(rope_dim));
      const float angle = static_cast<float>(position) * inv_freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);

      const std::size_t i0 = base + static_cast<std::size_t>(i);
      const std::size_t i1 = base + static_cast<std::size_t>(i + half);

      const float x0 = x[i0];
      const float x1 = x[i1];
      x[i0] = x0 * c - x1 * s;
      x[i1] = x1 * c + x0 * s;
    }
  }
}

bool build_runtime_dims(const ModelProfile & profile, RuntimeDims & dims, std::string & error_message) {
  dims.n_layers = profile.text.num_hidden_layers;
  dims.hidden = profile.text.hidden_size;
  dims.intermediate = profile.text.intermediate_size;
  dims.vocab_size = profile.text.vocab_size;

  dims.n_heads = profile.text.num_attention_heads;
  dims.n_kv_heads = profile.text.num_key_value_heads;
  dims.head_dim = profile.text.head_dim > 0 ? profile.text.head_dim : (dims.hidden / std::max(1, dims.n_heads));
  dims.rope_theta = profile.text.rope_theta;
  dims.rms_eps = profile.text.rms_norm_eps;

  dims.rope_dim = static_cast<int>(static_cast<float>(dims.head_dim) * profile.text.partial_rotary_factor);
  if ((dims.rope_dim % 2) != 0) {
    dims.rope_dim -= 1;
  }
  if (dims.rope_dim < 0) {
    dims.rope_dim = 0;
  }

  dims.linear_kernel = profile.text.linear_conv_kernel_dim;
  dims.linear_num_k_heads = profile.text.linear_num_key_heads;
  dims.linear_num_v_heads = profile.text.linear_num_value_heads;
  dims.linear_head_k_dim = profile.text.linear_key_head_dim;
  dims.linear_head_v_dim = profile.text.linear_value_head_dim;
  dims.linear_q_dim = dims.linear_num_k_heads * dims.linear_head_k_dim;
  dims.linear_v_dim = dims.linear_num_v_heads * dims.linear_head_v_dim;
  dims.linear_conv_channels = dims.linear_q_dim * 2 + dims.linear_v_dim;

  if (dims.n_layers <= 0 || dims.hidden <= 0 || dims.intermediate <= 0 || dims.vocab_size <= 0 || dims.n_heads <= 0 ||
      dims.n_kv_heads <= 0 || dims.head_dim <= 0 || dims.linear_kernel <= 1 || dims.linear_num_k_heads <= 0 ||
      dims.linear_num_v_heads <= 0 || dims.linear_head_k_dim <= 0 || dims.linear_head_v_dim <= 0) {
    error_message = "Model profile does not contain required runtime dimensions.";
    return false;
  }

  if (dims.n_heads % dims.n_kv_heads != 0) {
    error_message = "Invalid full-attention GQA ratio in profile.";
    return false;
  }
  if (dims.linear_num_v_heads % dims.linear_num_k_heads != 0) {
    error_message = "Invalid linear-attention grouped-head ratio in profile.";
    return false;
  }
  if (dims.linear_head_k_dim != dims.linear_head_v_dim) {
    error_message = "Reference linear path requires linear_key_head_dim == linear_value_head_dim.";
    return false;
  }
  if (static_cast<int>(profile.fingerprint.attention_schedule.size()) != dims.n_layers) {
    error_message = "Attention schedule length does not match num_hidden_layers.";
    return false;
  }

  return true;
}

bool load_model_weights(
  const std::string & model_dir,
  const RuntimeDims & dims,
  const ModelProfile & profile,
  ModelWeights & weights,
  std::string & error_message) {
  if (!load_tensor_checked_2d(
        model_dir, "model.language_model.embed_tokens.weight", dims.vocab_size, dims.hidden, weights.embed_tokens, error_message)) {
    return false;
  }
  if (!load_tensor_checked_1d(model_dir, "model.language_model.norm.weight", dims.hidden, weights.final_norm, error_message)) {
    return false;
  }

  weights.layers.resize(static_cast<std::size_t>(dims.n_layers));
  const int full_q_out = dims.n_heads * dims.head_dim * 2;
  const int full_kv_out = dims.n_kv_heads * dims.head_dim;
  const int full_o_in = dims.n_heads * dims.head_dim;

  for (int il = 0; il < dims.n_layers; ++il) {
    auto & layer = weights.layers[static_cast<std::size_t>(il)];
    const std::string base = "model.language_model.layers." + std::to_string(il) + ".";
    layer.is_linear = profile.fingerprint.attention_schedule[static_cast<std::size_t>(il)] == AttentionBlock::linear;

    if (!load_tensor_checked_1d(model_dir, base + "input_layernorm.weight", dims.hidden, layer.input_layernorm, error_message) ||
        !load_tensor_checked_1d(
          model_dir, base + "post_attention_layernorm.weight", dims.hidden, layer.post_attention_layernorm, error_message) ||
        !load_tensor_checked_2d(model_dir, base + "mlp.gate_proj.weight", dims.intermediate, dims.hidden, layer.mlp_gate, error_message) ||
        !load_tensor_checked_2d(model_dir, base + "mlp.up_proj.weight", dims.intermediate, dims.hidden, layer.mlp_up, error_message) ||
        !load_tensor_checked_2d(model_dir, base + "mlp.down_proj.weight", dims.hidden, dims.intermediate, layer.mlp_down, error_message)) {
      return false;
    }

    if (layer.is_linear) {
      if (!load_tensor_checked_2d(
            model_dir, base + "linear_attn.in_proj_qkv.weight", dims.linear_conv_channels, dims.hidden, layer.linear.in_proj_qkv, error_message) ||
          !load_tensor_checked_2d(
            model_dir, base + "linear_attn.in_proj_z.weight", dims.linear_v_dim, dims.hidden, layer.linear.in_proj_z, error_message) ||
          !load_tensor_checked_2d(
            model_dir, base + "linear_attn.in_proj_b.weight", dims.linear_num_v_heads, dims.hidden, layer.linear.in_proj_b, error_message) ||
          !load_tensor_checked_2d(
            model_dir, base + "linear_attn.in_proj_a.weight", dims.linear_num_v_heads, dims.hidden, layer.linear.in_proj_a, error_message) ||
          !load_tensor_checked_conv1d(
            model_dir, base + "linear_attn.conv1d.weight", dims.linear_conv_channels, dims.linear_kernel, layer.linear.conv1d, error_message) ||
          !load_tensor_checked_2d(
            model_dir, base + "linear_attn.out_proj.weight", dims.hidden, dims.linear_v_dim, layer.linear.out_proj, error_message) ||
          !load_tensor_checked_1d(
            model_dir, base + "linear_attn.norm.weight", dims.linear_head_v_dim, layer.linear.norm, error_message) ||
          !load_tensor_checked_1d(
            model_dir, base + "linear_attn.A_log", dims.linear_num_v_heads, layer.linear.a_log, error_message) ||
          !load_tensor_checked_1d(
            model_dir, base + "linear_attn.dt_bias", dims.linear_num_v_heads, layer.linear.dt_bias, error_message)) {
        return false;
      }

      layer.linear.ssm_a.resize(static_cast<std::size_t>(dims.linear_num_v_heads));
      for (int i = 0; i < dims.linear_num_v_heads; ++i) {
        layer.linear.ssm_a[static_cast<std::size_t>(i)] = -std::exp(layer.linear.a_log.data[static_cast<std::size_t>(i)]);
      }
    } else {
      if (!load_tensor_checked_2d(model_dir, base + "self_attn.q_proj.weight", full_q_out, dims.hidden, layer.full.q_proj, error_message) ||
          !load_tensor_checked_2d(model_dir, base + "self_attn.k_proj.weight", full_kv_out, dims.hidden, layer.full.k_proj, error_message) ||
          !load_tensor_checked_2d(model_dir, base + "self_attn.v_proj.weight", full_kv_out, dims.hidden, layer.full.v_proj, error_message) ||
          !load_tensor_checked_2d(model_dir, base + "self_attn.o_proj.weight", dims.hidden, full_o_in, layer.full.o_proj, error_message) ||
          !load_tensor_checked_1d(model_dir, base + "self_attn.q_norm.weight", dims.head_dim, layer.full.q_norm, error_message) ||
          !load_tensor_checked_1d(model_dir, base + "self_attn.k_norm.weight", dims.head_dim, layer.full.k_norm, error_message)) {
        return false;
      }
    }
  }

  return true;
}

bool upload_tensor_2d_to_cuda(
  TensorData & tensor,
  const std::string & name,
  const bool upload_bf16_shadow,
  std::string & error_message) {
  if (tensor.shape.size() != 2) {
    return true;
  }
  if (tensor.has_device_matrix && tensor.device_matrix.data != nullptr) {
    return true;
  }

  const int rows = static_cast<int>(tensor.shape[0]);
  const int cols = static_cast<int>(tensor.shape[1]);
  if (!cuda::upload_matrix_f32(tensor.data, rows, cols, tensor.device_matrix, error_message)) {
    error_message = "CUDA upload failed for tensor '" + name + "': " + error_message;
    return false;
  }
  if (upload_bf16_shadow &&
      !cuda::upload_matrix_bf16_shadow_from_f32(tensor.data, rows, cols, tensor.device_matrix, error_message)) {
    error_message = "CUDA BF16 shadow upload failed for tensor '" + name + "': " + error_message;
    return false;
  }
  tensor.has_device_matrix = true;
  return true;
}

bool upload_vector_to_cuda(
  const std::vector<float> & host_values,
  cuda::CudaDeviceBufferF32 & out_buffer,
  std::string & error_message) {
  if (!cuda::allocate_buffer_f32(host_values.size(), out_buffer, error_message)) {
    return false;
  }
  if (!host_values.empty() &&
      !cuda::upload_to_buffer_f32(host_values.data(), host_values.size(), out_buffer, 0, error_message)) {
    cuda::free_buffer_f32(out_buffer);
    return false;
  }
  return true;
}

bool upload_row_concat_2d_to_cuda(
  const std::vector<const TensorData *> & parts,
  const std::string & name,
  cuda::CudaDeviceMatrixF32 & out_matrix,
  bool & out_has_matrix,
  const bool upload_bf16_shadow,
  std::string & error_message) {
  if (parts.empty()) {
    error_message = "CUDA packed upload '" + name + "' has no source tensors.";
    return false;
  }

  std::int64_t cols = -1;
  std::int64_t total_rows = 0;
  for (const TensorData * part : parts) {
    if (part == nullptr || part->shape.size() != 2) {
      error_message = "CUDA packed upload '" + name + "' expects 2D source tensors.";
      return false;
    }
    if (cols < 0) {
      cols = part->shape[1];
    } else if (part->shape[1] != cols) {
      error_message = "CUDA packed upload '" + name + "' source tensors have mismatched column counts.";
      return false;
    }
    total_rows += part->shape[0];
  }
  if (cols <= 0 || total_rows <= 0) {
    error_message = "CUDA packed upload '" + name + "' has invalid packed dimensions.";
    return false;
  }

  std::vector<float> packed;
  packed.reserve(static_cast<std::size_t>(total_rows * cols));
  for (const TensorData * part : parts) {
    packed.insert(packed.end(), part->data.begin(), part->data.end());
  }

  if (!cuda::upload_matrix_f32(packed, static_cast<int>(total_rows), static_cast<int>(cols), out_matrix, error_message)) {
    error_message = "CUDA packed upload failed for tensor '" + name + "': " + error_message;
    return false;
  }
  if (upload_bf16_shadow &&
      !cuda::upload_matrix_bf16_shadow_from_f32(
        packed,
        static_cast<int>(total_rows),
        static_cast<int>(cols),
        out_matrix,
        error_message)) {
    error_message = "CUDA packed BF16 shadow upload failed for tensor '" + name + "': " + error_message;
    return false;
  }
  out_has_matrix = true;
  return true;
}

void release_tensor_cuda(TensorData & tensor) {
  if (tensor.has_device_matrix || tensor.device_matrix.data != nullptr) {
    cuda::free_matrix_f32(tensor.device_matrix);
    tensor.has_device_matrix = false;
  }
}

bool upload_model_weights_to_cuda(
  ModelWeights & weights,
  const bool upload_bf16_shadow,
  std::string & error_message) {
  if (!upload_tensor_2d_to_cuda(weights.embed_tokens, "embed_tokens", upload_bf16_shadow, error_message)) {
    return false;
  }
  if (!upload_vector_to_cuda(weights.final_norm.data, weights.final_norm_device, error_message)) {
    return false;
  }
  weights.has_device_final_norm = true;

  for (std::size_t il = 0; il < weights.layers.size(); ++il) {
    LayerWeights & layer = weights.layers[il];
    const std::string prefix = "layer " + std::to_string(il) + " ";

    if (!upload_vector_to_cuda(layer.input_layernorm.data, layer.input_layernorm_device, error_message) ||
        !upload_vector_to_cuda(layer.post_attention_layernorm.data, layer.post_attention_layernorm_device, error_message)) {
      return false;
    }
    layer.has_device_norms = true;

    if (!upload_tensor_2d_to_cuda(layer.mlp_gate, prefix + "mlp_gate", upload_bf16_shadow, error_message) ||
        !upload_tensor_2d_to_cuda(layer.mlp_up, prefix + "mlp_up", upload_bf16_shadow, error_message) ||
        !upload_tensor_2d_to_cuda(layer.mlp_down, prefix + "mlp_down", upload_bf16_shadow, error_message)) {
      return false;
    }
    if (!upload_row_concat_2d_to_cuda(
          {&layer.mlp_gate, &layer.mlp_up},
          prefix + "mlp_gate_up_packed",
          layer.mlp_gate_up_device,
          layer.has_device_mlp_gate_up,
          upload_bf16_shadow,
          error_message)) {
      return false;
    }

    if (layer.is_linear) {
      if (!upload_tensor_2d_to_cuda(layer.linear.in_proj_qkv, prefix + "linear.in_proj_qkv", upload_bf16_shadow, error_message) ||
          !upload_tensor_2d_to_cuda(layer.linear.in_proj_z, prefix + "linear.in_proj_z", upload_bf16_shadow, error_message) ||
          !upload_tensor_2d_to_cuda(layer.linear.in_proj_b, prefix + "linear.in_proj_b", upload_bf16_shadow, error_message) ||
          !upload_tensor_2d_to_cuda(layer.linear.in_proj_a, prefix + "linear.in_proj_a", upload_bf16_shadow, error_message) ||
          !upload_tensor_2d_to_cuda(layer.linear.conv1d, prefix + "linear.conv1d", false, error_message) ||
          !upload_tensor_2d_to_cuda(layer.linear.out_proj, prefix + "linear.out_proj", upload_bf16_shadow, error_message)) {
        return false;
      }
      if (!upload_vector_to_cuda(layer.linear.norm.data, layer.linear.norm_device, error_message) ||
          !upload_vector_to_cuda(layer.linear.dt_bias.data, layer.linear.dt_bias_device, error_message) ||
          !upload_vector_to_cuda(layer.linear.ssm_a, layer.linear.ssm_a_device, error_message)) {
        return false;
      }
      if (!upload_row_concat_2d_to_cuda(
            {&layer.linear.in_proj_qkv, &layer.linear.in_proj_z, &layer.linear.in_proj_b, &layer.linear.in_proj_a},
            prefix + "linear.in_proj_all_packed",
            layer.linear.in_proj_all_device,
            layer.linear.has_device_in_proj_all,
            upload_bf16_shadow,
            error_message)) {
        return false;
      }
      layer.linear.has_device_params = true;
    } else {
      if (!upload_tensor_2d_to_cuda(layer.full.q_proj, prefix + "full.q_proj", upload_bf16_shadow, error_message) ||
          !upload_tensor_2d_to_cuda(layer.full.k_proj, prefix + "full.k_proj", upload_bf16_shadow, error_message) ||
          !upload_tensor_2d_to_cuda(layer.full.v_proj, prefix + "full.v_proj", upload_bf16_shadow, error_message) ||
          !upload_tensor_2d_to_cuda(layer.full.o_proj, prefix + "full.o_proj", upload_bf16_shadow, error_message)) {
        return false;
      }
      if (!upload_vector_to_cuda(layer.full.q_norm.data, layer.full.q_norm_device, error_message) ||
          !upload_vector_to_cuda(layer.full.k_norm.data, layer.full.k_norm_device, error_message)) {
        return false;
      }
      if (!upload_row_concat_2d_to_cuda(
            {&layer.full.q_proj, &layer.full.k_proj, &layer.full.v_proj},
            prefix + "full.qkv_proj_packed",
            layer.full.qkv_proj_device,
            layer.full.has_device_qkv_proj,
            upload_bf16_shadow,
            error_message)) {
        return false;
      }
      if (!upload_row_concat_2d_to_cuda(
            {&layer.full.k_proj, &layer.full.v_proj},
            prefix + "full.kv_proj_packed",
            layer.full.kv_proj_device,
            layer.full.has_device_kv_proj,
            upload_bf16_shadow,
            error_message)) {
        return false;
      }
      layer.full.has_device_norm = true;
    }
  }

  return true;
}

void release_model_weights_cuda(ModelWeights & weights) {
  release_tensor_cuda(weights.embed_tokens);
  cuda::free_buffer_f32(weights.final_norm_device);
  weights.has_device_final_norm = false;
  for (auto & layer : weights.layers) {
    cuda::free_buffer_f32(layer.input_layernorm_device);
    cuda::free_buffer_f32(layer.post_attention_layernorm_device);
    layer.has_device_norms = false;
    release_tensor_cuda(layer.mlp_gate);
    release_tensor_cuda(layer.mlp_up);
    release_tensor_cuda(layer.mlp_down);
    cuda::free_matrix_f32(layer.mlp_gate_up_device);
    layer.has_device_mlp_gate_up = false;
    release_tensor_cuda(layer.full.q_proj);
    release_tensor_cuda(layer.full.k_proj);
    release_tensor_cuda(layer.full.v_proj);
    release_tensor_cuda(layer.full.o_proj);
    cuda::free_matrix_f32(layer.full.qkv_proj_device);
    layer.full.has_device_qkv_proj = false;
    cuda::free_matrix_f32(layer.full.kv_proj_device);
    layer.full.has_device_kv_proj = false;
    cuda::free_buffer_f32(layer.full.q_norm_device);
    cuda::free_buffer_f32(layer.full.k_norm_device);
    layer.full.has_device_norm = false;
    release_tensor_cuda(layer.linear.in_proj_qkv);
    release_tensor_cuda(layer.linear.in_proj_z);
    release_tensor_cuda(layer.linear.in_proj_b);
    release_tensor_cuda(layer.linear.in_proj_a);
    release_tensor_cuda(layer.linear.conv1d);
    release_tensor_cuda(layer.linear.out_proj);
    cuda::free_matrix_f32(layer.linear.in_proj_all_device);
    layer.linear.has_device_in_proj_all = false;
    cuda::free_buffer_f32(layer.linear.norm_device);
    cuda::free_buffer_f32(layer.linear.dt_bias_device);
    cuda::free_buffer_f32(layer.linear.ssm_a_device);
    layer.linear.has_device_params = false;
  }
}

void release_model_state_cuda(ModelState & state) {
  for (auto & fs : state.full_states) {
    cuda::free_buffer_f32(fs.k_cache_device);
    cuda::free_buffer_f32(fs.v_cache_device);
    cuda::free_buffer_bf16(fs.k_cache_bf16_device);
    cuda::free_buffer_bf16(fs.v_cache_bf16_device);
    fs.has_device_state = false;
    fs.has_device_state_bf16 = false;
  }
  for (auto & ls : state.linear_states) {
    cuda::free_buffer_f32(ls.conv_state_device);
    cuda::free_buffer_f32(ls.recurrent_state_device);
    ls.has_device_state = false;
  }
}

bool allocate_forward_workspace_cuda(
  const RuntimeDims & dims,
  const int max_context,
  CudaForwardWorkspace & workspace,
  std::string & error_message) {
  const std::size_t full_q_count = static_cast<std::size_t>(dims.n_heads * dims.head_dim);
  const std::size_t full_q_packed_count = full_q_count * 2;
  const std::size_t full_scores_count = static_cast<std::size_t>(dims.n_heads) * static_cast<std::size_t>(std::max(1, max_context));
  const std::size_t full_kv_combined_count = static_cast<std::size_t>(2 * dims.n_kv_heads * dims.head_dim);
  const std::size_t full_qkv_combined_count = full_q_packed_count + full_kv_combined_count;
  const std::size_t linear_combined_count =
    static_cast<std::size_t>(dims.linear_conv_channels + dims.linear_v_dim + 2 * dims.linear_num_v_heads);
  const std::size_t mlp_gate_up_combined_count = static_cast<std::size_t>(2 * dims.intermediate);
  const std::size_t projection_out_count = std::max<std::size_t>({
    static_cast<std::size_t>(dims.hidden),
    full_q_packed_count,
    static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim),
    full_kv_combined_count,
    full_qkv_combined_count,
    static_cast<std::size_t>(dims.linear_conv_channels),
    static_cast<std::size_t>(dims.linear_v_dim),
    static_cast<std::size_t>(dims.linear_num_v_heads),
    static_cast<std::size_t>(dims.intermediate),
    linear_combined_count,
    mlp_gate_up_combined_count,
    1u
  });
  const std::size_t linear_q_dim = static_cast<std::size_t>(dims.linear_num_k_heads * dims.linear_head_k_dim);
  const std::size_t linear_v_dim = static_cast<std::size_t>(dims.linear_num_v_heads * dims.linear_head_v_dim);
  const std::size_t linear_conv_channels = linear_q_dim * 2 + linear_v_dim;
  if (!cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.hidden), workspace.hidden_in, error_message) ||
      !cuda::allocate_buffer_bf16(static_cast<std::size_t>(dims.hidden), workspace.hidden_in_bf16, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.intermediate), workspace.intermediate_a, error_message) ||
      !cuda::allocate_buffer_bf16(static_cast<std::size_t>(dims.intermediate), workspace.intermediate_a_bf16, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.intermediate), workspace.intermediate_b, error_message) ||
      !cuda::allocate_buffer_bf16(static_cast<std::size_t>(dims.intermediate), workspace.intermediate_b_bf16, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.intermediate), workspace.intermediate_hidden, error_message) ||
      !cuda::allocate_buffer_bf16(static_cast<std::size_t>(dims.intermediate), workspace.intermediate_hidden_bf16, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.hidden), workspace.hidden_out, error_message) ||
      !cuda::allocate_buffer_bf16(static_cast<std::size_t>(dims.hidden), workspace.hidden_out_bf16, error_message) ||
      !cuda::allocate_buffer_f32(projection_out_count, workspace.projection_out, error_message) ||
      !cuda::allocate_buffer_bf16(projection_out_count, workspace.projection_out_bf16, error_message) ||
      !cuda::allocate_buffer_f32(full_q_count, workspace.full_q, error_message) ||
      !cuda::allocate_buffer_bf16(full_q_count, workspace.full_q_bf16, error_message) ||
      !cuda::allocate_buffer_f32(full_q_count, workspace.full_gate, error_message) ||
      !cuda::allocate_buffer_f32(full_q_count, workspace.full_attn, error_message) ||
      !cuda::allocate_buffer_bf16(full_q_count, workspace.full_attn_bf16, error_message) ||
      !cuda::allocate_buffer_f32(full_scores_count, workspace.full_scores, error_message) ||
      !cuda::allocate_buffer_f32(linear_conv_channels, workspace.linear_mixed_qkv, error_message) ||
      !cuda::allocate_buffer_f32(linear_v_dim, workspace.linear_z, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.linear_num_v_heads), workspace.linear_b, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.linear_num_v_heads), workspace.linear_a, error_message) ||
      !cuda::allocate_buffer_f32(linear_conv_channels, workspace.linear_conv_out, error_message) ||
      !cuda::allocate_buffer_f32(linear_v_dim, workspace.linear_gated_norm, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.vocab_size), workspace.logits, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.vocab_size), workspace.seen_token_mask, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(kCudaSamplingMaxTopK), workspace.topk_values_scratch, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(kCudaSamplingMaxTopK), workspace.topk_indices_scratch, error_message)) {
    cuda::free_buffer_f32(workspace.hidden_in);
    cuda::free_buffer_bf16(workspace.hidden_in_bf16);
    cuda::free_buffer_f32(workspace.intermediate_a);
    cuda::free_buffer_bf16(workspace.intermediate_a_bf16);
    cuda::free_buffer_f32(workspace.intermediate_b);
    cuda::free_buffer_bf16(workspace.intermediate_b_bf16);
    cuda::free_buffer_f32(workspace.intermediate_hidden);
    cuda::free_buffer_bf16(workspace.intermediate_hidden_bf16);
    cuda::free_buffer_f32(workspace.hidden_out);
    cuda::free_buffer_bf16(workspace.hidden_out_bf16);
    cuda::free_buffer_f32(workspace.projection_out);
    cuda::free_buffer_bf16(workspace.projection_out_bf16);
    cuda::free_buffer_f32(workspace.full_q);
    cuda::free_buffer_bf16(workspace.full_q_bf16);
    cuda::free_buffer_f32(workspace.full_gate);
    cuda::free_buffer_f32(workspace.full_attn);
    cuda::free_buffer_bf16(workspace.full_attn_bf16);
    cuda::free_buffer_f32(workspace.full_scores);
    cuda::free_buffer_f32(workspace.linear_mixed_qkv);
    cuda::free_buffer_f32(workspace.linear_z);
    cuda::free_buffer_f32(workspace.linear_b);
    cuda::free_buffer_f32(workspace.linear_a);
    cuda::free_buffer_f32(workspace.linear_conv_out);
    cuda::free_buffer_f32(workspace.linear_gated_norm);
    cuda::free_buffer_f32(workspace.logits);
    cuda::free_buffer_f32(workspace.seen_token_mask);
    cuda::free_buffer_f32(workspace.topk_values_scratch);
    cuda::free_buffer_f32(workspace.topk_indices_scratch);
    return false;
  }

  workspace.has_gpu_sampling_buffers = true;
  workspace.has_device_buffers = true;
  workspace.mlp_graphs.clear();
  workspace.mlp_graph_warmup_done.clear();
  workspace.mlp_graph_disabled.clear();
  workspace.linear_attention_graphs.clear();
  workspace.linear_attention_graph_warmup_done.clear();
  workspace.linear_attention_graph_disabled.clear();
  workspace.mlp_graphs.resize(static_cast<std::size_t>(dims.n_layers));
  workspace.mlp_graph_warmup_done.resize(static_cast<std::size_t>(dims.n_layers), false);
  workspace.mlp_graph_disabled.resize(static_cast<std::size_t>(dims.n_layers), false);
  workspace.linear_attention_graphs.resize(static_cast<std::size_t>(dims.n_layers));
  workspace.linear_attention_graph_warmup_done.resize(static_cast<std::size_t>(dims.n_layers), false);
  workspace.linear_attention_graph_disabled.resize(static_cast<std::size_t>(dims.n_layers), false);
  return true;
}

void release_forward_workspace_cuda(CudaForwardWorkspace & workspace) {
  if (!workspace.has_device_buffers) {
    return;
  }
  cuda::free_buffer_f32(workspace.hidden_in);
  cuda::free_buffer_bf16(workspace.hidden_in_bf16);
  cuda::free_buffer_f32(workspace.intermediate_a);
  cuda::free_buffer_bf16(workspace.intermediate_a_bf16);
  cuda::free_buffer_f32(workspace.intermediate_b);
  cuda::free_buffer_bf16(workspace.intermediate_b_bf16);
  cuda::free_buffer_f32(workspace.intermediate_hidden);
  cuda::free_buffer_bf16(workspace.intermediate_hidden_bf16);
  cuda::free_buffer_f32(workspace.hidden_out);
  cuda::free_buffer_bf16(workspace.hidden_out_bf16);
  cuda::free_buffer_f32(workspace.projection_out);
  cuda::free_buffer_bf16(workspace.projection_out_bf16);
  cuda::free_buffer_f32(workspace.full_q);
  cuda::free_buffer_bf16(workspace.full_q_bf16);
  cuda::free_buffer_f32(workspace.full_gate);
  cuda::free_buffer_f32(workspace.full_attn);
  cuda::free_buffer_bf16(workspace.full_attn_bf16);
  cuda::free_buffer_f32(workspace.full_scores);
  cuda::free_buffer_f32(workspace.linear_mixed_qkv);
  cuda::free_buffer_f32(workspace.linear_z);
  cuda::free_buffer_f32(workspace.linear_b);
  cuda::free_buffer_f32(workspace.linear_a);
  cuda::free_buffer_f32(workspace.linear_conv_out);
  cuda::free_buffer_f32(workspace.linear_gated_norm);
  cuda::free_buffer_f32(workspace.logits);
  cuda::free_buffer_f32(workspace.seen_token_mask);
  cuda::free_buffer_f32(workspace.topk_values_scratch);
  cuda::free_buffer_f32(workspace.topk_indices_scratch);
  for (auto & graph : workspace.mlp_graphs) {
    cuda::free_captured_graph(graph);
  }
  for (auto & graph : workspace.linear_attention_graphs) {
    cuda::free_captured_graph(graph);
  }
  workspace.mlp_graphs.clear();
  workspace.mlp_graph_warmup_done.clear();
  workspace.mlp_graph_disabled.clear();
  workspace.linear_attention_graphs.clear();
  workspace.linear_attention_graph_warmup_done.clear();
  workspace.linear_attention_graph_disabled.clear();
  workspace.has_gpu_sampling_buffers = false;
  workspace.has_device_buffers = false;
}

cuda::CudaDeviceBufferF32 buffer_slice_f32(
  const cuda::CudaDeviceBufferF32 & buffer,
  const std::size_t offset,
  const std::size_t count) {
  cuda::CudaDeviceBufferF32 out;
  if (buffer.data == nullptr || offset > buffer.count || count > (buffer.count - offset)) {
    return out;
  }
  out.data = static_cast<float *>(buffer.data) + offset;
  out.count = count;
  return out;
}

cuda::CudaDeviceBufferBF16 buffer_slice_bf16(
  const cuda::CudaDeviceBufferBF16 & buffer,
  const std::size_t offset,
  const std::size_t count) {
  cuda::CudaDeviceBufferBF16 out;
  if (buffer.data == nullptr || offset > buffer.count || count > (buffer.count - offset)) {
    return out;
  }
  out.data = static_cast<std::uint16_t *>(buffer.data) + offset;
  out.count = count;
  return out;
}

bool run_linear_attention_step_cuda_device(
  const LayerWeights & layer,
  const RuntimeDims & dims,
  const std::size_t layer_slot,
  LinearAttentionState & state,
  const cuda::CudaDeviceBufferF32 & x_in,
  CudaForwardWorkspace & workspace,
  cuda::CudaDeviceBufferF32 & out_hidden,
  std::string & error_message) {
  if (!state.has_device_state || !layer.linear.has_device_params || !layer.linear.has_device_in_proj_all ||
      !layer.linear.in_proj_all_device.data || !layer.linear.conv1d.has_device_matrix ||
      !layer.linear.out_proj.has_device_matrix) {
    error_message = "Linear-attention CUDA device path is not fully initialized.";
    return false;
  }

  const std::size_t mixed_count = static_cast<std::size_t>(dims.linear_conv_channels);
  const std::size_t z_count = static_cast<std::size_t>(dims.linear_v_dim);
  const std::size_t b_count = static_cast<std::size_t>(dims.linear_num_v_heads);
  const std::size_t a_count = static_cast<std::size_t>(dims.linear_num_v_heads);
  const std::size_t packed_count = mixed_count + z_count + b_count + a_count;
  if (workspace.projection_out.count < packed_count) {
    error_message = "CUDA workspace projection buffer is too small for packed linear projections.";
    return false;
  }

  const cuda::CudaDeviceBufferF32 mixed = buffer_slice_f32(workspace.projection_out, 0, mixed_count);
  const cuda::CudaDeviceBufferF32 z = buffer_slice_f32(workspace.projection_out, mixed_count, z_count);
  const cuda::CudaDeviceBufferF32 b = buffer_slice_f32(workspace.projection_out, mixed_count + z_count, b_count);
  const cuda::CudaDeviceBufferF32 a = buffer_slice_f32(workspace.projection_out, mixed_count + z_count + b_count, a_count);
  if (mixed.data == nullptr || z.data == nullptr || b.data == nullptr || a.data == nullptr) {
    error_message = "Failed to create packed linear projection slices.";
    return false;
  }

  const auto run_linear_direct = [&](std::string & run_error) -> bool {
    return cuda::run_matvec_f32_device(layer.linear.in_proj_all_device, x_in, workspace.projection_out, run_error) &&
           cuda::run_linear_attention_decode(
             mixed,
             z,
             b,
             a,
             layer.linear.conv1d.device_matrix,
             layer.linear.norm_device,
             layer.linear.dt_bias_device,
             layer.linear.ssm_a_device,
             dims.linear_kernel,
             dims.linear_num_k_heads,
             dims.linear_num_v_heads,
             dims.linear_head_k_dim,
             dims.linear_head_v_dim,
             dims.rms_eps,
             state.conv_state_device,
             state.recurrent_state_device,
             workspace.linear_conv_out,
             workspace.linear_gated_norm,
             run_error) &&
           cuda::run_matvec_f32_device(layer.linear.out_proj.device_matrix, workspace.linear_gated_norm, out_hidden, run_error);
  };

  bool linear_done = false;
  if (layer_slot < workspace.linear_attention_graphs.size()) {
    if (workspace.linear_attention_graphs[layer_slot].ready) {
      linear_done = cuda::launch_captured_graph(workspace.linear_attention_graphs[layer_slot], error_message);
      if (!linear_done) {
        workspace.linear_attention_graph_disabled[layer_slot] = true;
        cuda::free_captured_graph(workspace.linear_attention_graphs[layer_slot]);
      }
    }

    if (!linear_done && !workspace.linear_attention_graph_disabled[layer_slot] &&
        workspace.linear_attention_graph_warmup_done[layer_slot]) {
      std::string capture_error;
      const bool capture_started = cuda::begin_stream_capture(capture_error);
      bool capture_ok = false;
      if (capture_started) {
        std::string captured_run_error;
        const bool captured_run_ok = run_linear_direct(captured_run_error);
        std::string capture_end_error;
        const bool capture_end_ok = cuda::end_stream_capture(
          workspace.linear_attention_graphs[layer_slot],
          capture_end_error);
        if (captured_run_ok && capture_end_ok) {
          capture_ok = cuda::launch_captured_graph(workspace.linear_attention_graphs[layer_slot], capture_error);
        }
      }
      if (!capture_ok) {
        workspace.linear_attention_graph_disabled[layer_slot] = true;
        cuda::free_captured_graph(workspace.linear_attention_graphs[layer_slot]);
      } else {
        linear_done = true;
      }
    }

    if (!linear_done && !workspace.linear_attention_graph_disabled[layer_slot] &&
        !workspace.linear_attention_graph_warmup_done[layer_slot]) {
      std::string warmup_error;
      if (!run_linear_direct(warmup_error)) {
        error_message = warmup_error;
        return false;
      }
      workspace.linear_attention_graph_warmup_done[layer_slot] = true;
      linear_done = true;
    }
  }

  if (linear_done) {
    return true;
  }
  return run_linear_direct(error_message);
}

bool run_full_attention_step_cuda_device(
  const LayerWeights & layer,
  const RuntimeDims & dims,
  FullAttentionState & state,
  const cuda::CudaDeviceBufferF32 & x_in,
  const int position,
  CudaForwardWorkspace & workspace,
  cuda::CudaDeviceBufferF32 & out_hidden,
  std::string & error_message) {
  const bool use_bf16_cache = state.has_device_state_bf16;
  const bool use_f32_cache = state.has_device_state;
  const bool has_packed_qkv = layer.full.has_device_qkv_proj && layer.full.qkv_proj_device.data != nullptr;
  const bool has_legacy_qkv =
    layer.full.has_device_kv_proj && layer.full.q_proj.has_device_matrix && layer.full.kv_proj_device.data != nullptr;
  if ((!use_f32_cache && !use_bf16_cache) || !layer.full.has_device_norm || !layer.full.o_proj.has_device_matrix ||
      (!has_packed_qkv && !has_legacy_qkv)) {
    error_message = "Full-attention CUDA device path is not fully initialized.";
    return false;
  }

  const std::size_t q_packed_count = static_cast<std::size_t>(dims.n_heads * dims.head_dim * 2);
  const std::size_t kv_count = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  const std::size_t kv_packed_count = kv_count * 2;
  const std::size_t qkv_packed_count = q_packed_count + kv_packed_count;
  const std::size_t cache_offset = static_cast<std::size_t>(position) * kv_count;
  const std::size_t required_projection_count = has_packed_qkv ? qkv_packed_count : kv_packed_count;
  const bool bf16_matrix_ready =
    (has_packed_qkv && layer.full.qkv_proj_device.data_bf16 != nullptr) ||
    (has_legacy_qkv && layer.full.kv_proj_device.data_bf16 != nullptr);
  const bool use_bf16_projection =
    use_bf16_cache && bf16_matrix_ready && workspace.projection_out_bf16.data != nullptr &&
    workspace.projection_out_bf16.count >= required_projection_count;

  if ((!has_packed_qkv && workspace.logits.count < q_packed_count) ||
      (!use_bf16_projection && workspace.projection_out.count < required_projection_count) ||
      workspace.full_q.count < static_cast<std::size_t>(dims.n_heads * dims.head_dim) ||
      workspace.full_gate.count < static_cast<std::size_t>(dims.n_heads * dims.head_dim) ||
      workspace.full_attn.count < static_cast<std::size_t>(dims.n_heads * dims.head_dim) ||
      (use_bf16_cache && workspace.full_attn_bf16.count < kv_count)) {
    error_message = "CUDA forward workspace is too small for full-attention device step.";
    return false;
  }

  const std::size_t kv_offset = has_packed_qkv ? q_packed_count : 0;
  const cuda::CudaDeviceBufferF32 q_gate_packed_f32 = has_packed_qkv
    ? buffer_slice_f32(workspace.projection_out, 0, q_packed_count)
    : workspace.logits;
  const cuda::CudaDeviceBufferBF16 q_gate_packed_bf16 = has_packed_qkv
    ? buffer_slice_bf16(workspace.projection_out_bf16, 0, q_packed_count)
    : cuda::CudaDeviceBufferBF16{};
  const cuda::CudaDeviceBufferF32 k_raw_f32 = buffer_slice_f32(workspace.projection_out, kv_offset, kv_count);
  const cuda::CudaDeviceBufferF32 v_raw_f32 = buffer_slice_f32(workspace.projection_out, kv_offset + kv_count, kv_count);
  const cuda::CudaDeviceBufferBF16 k_raw_bf16 = buffer_slice_bf16(workspace.projection_out_bf16, kv_offset, kv_count);
  const cuda::CudaDeviceBufferBF16 v_raw_bf16 = buffer_slice_bf16(workspace.projection_out_bf16, kv_offset + kv_count, kv_count);

  const auto run_qkv_projections = [&](std::string & run_error) -> bool {
    if (has_packed_qkv) {
      if (use_bf16_projection) {
        return cuda::run_matvec_bf16_device_output_bf16(layer.full.qkv_proj_device, x_in, workspace.projection_out_bf16, run_error);
      }
      return cuda::run_matvec_f32_device(layer.full.qkv_proj_device, x_in, workspace.projection_out, run_error);
    }
    if (use_bf16_projection) {
      return cuda::run_matvec_f32_device(layer.full.q_proj.device_matrix, x_in, workspace.logits, run_error) &&
             cuda::run_matvec_bf16_device_output_bf16(layer.full.kv_proj_device, x_in, workspace.projection_out_bf16, run_error);
    }
    return cuda::run_matvec_f32_device(layer.full.q_proj.device_matrix, x_in, workspace.logits, run_error) &&
           cuda::run_matvec_f32_device(layer.full.kv_proj_device, x_in, workspace.projection_out, run_error);
  };

  const auto split_q_gate = [&](std::string & run_error) -> bool {
    if (has_packed_qkv && use_bf16_projection) {
      if (q_gate_packed_bf16.data == nullptr) {
        run_error = "Failed to create packed BF16 Q/G projection slice.";
        return false;
      }
      return cuda::run_split_q_gate_bf16_to_f32(
        q_gate_packed_bf16,
        dims.n_heads,
        dims.head_dim,
        workspace.full_q,
        workspace.full_gate,
        run_error);
    }
    if (q_gate_packed_f32.data == nullptr) {
      run_error = "Failed to create packed Q/G projection slice.";
      return false;
    }
    return cuda::run_split_q_gate_f32(
      q_gate_packed_f32,
      dims.n_heads,
      dims.head_dim,
      workspace.full_q,
      workspace.full_gate,
      run_error);
  };

  const auto normalize_and_cache_kv = [&](std::string & run_error) -> bool {
    if (use_bf16_cache) {
      if (use_bf16_projection) {
        if (k_raw_bf16.data == nullptr || v_raw_bf16.data == nullptr) {
          run_error = "Failed to create packed BF16 KV projection slices.";
          return false;
        }
        return cuda::run_rms_norm_per_head_bf16(
                 k_raw_bf16,
                 layer.full.k_norm_device,
                 dims.n_kv_heads,
                 dims.head_dim,
                 dims.rms_eps,
                 workspace.full_attn_bf16,
                 run_error) &&
               cuda::run_apply_rope_inplace_bf16(
                 workspace.full_attn_bf16,
                 dims.n_kv_heads,
                 dims.head_dim,
                 dims.rope_dim,
                 position,
                 dims.rope_theta,
                 run_error) &&
               cuda::copy_buffer_bf16(
                 workspace.full_attn_bf16,
                 kv_count,
                 0,
                 state.k_cache_bf16_device,
                 cache_offset,
                 run_error) &&
               cuda::copy_buffer_bf16(
                 v_raw_bf16,
                 kv_count,
                 0,
                 state.v_cache_bf16_device,
                 cache_offset,
                 run_error);
      }
      if (k_raw_f32.data == nullptr || v_raw_f32.data == nullptr) {
        run_error = "Failed to create packed KV projection slices.";
        return false;
      }
      return cuda::run_rms_norm_per_head_f32(
               k_raw_f32,
               layer.full.k_norm_device,
               dims.n_kv_heads,
               dims.head_dim,
               dims.rms_eps,
               workspace.full_attn,
               run_error) &&
             cuda::run_apply_rope_inplace_f32(
               workspace.full_attn,
               dims.n_kv_heads,
               dims.head_dim,
               dims.rope_dim,
               position,
               dims.rope_theta,
               run_error) &&
             cuda::copy_buffer_f32_to_bf16(
               workspace.full_attn,
               kv_count,
               0,
               state.k_cache_bf16_device,
               cache_offset,
               run_error) &&
             cuda::copy_buffer_f32_to_bf16(
               v_raw_f32,
               kv_count,
               0,
               state.v_cache_bf16_device,
               cache_offset,
               run_error);
    }

    if (k_raw_f32.data == nullptr || v_raw_f32.data == nullptr) {
      run_error = "Failed to create packed KV projection slices.";
      return false;
    }
    return cuda::run_rms_norm_per_head_f32(
             k_raw_f32,
             layer.full.k_norm_device,
             dims.n_kv_heads,
             dims.head_dim,
             dims.rms_eps,
             workspace.full_attn,
             run_error) &&
           cuda::run_apply_rope_inplace_f32(
             workspace.full_attn,
             dims.n_kv_heads,
             dims.head_dim,
             dims.rope_dim,
             position,
             dims.rope_theta,
             run_error) &&
           cuda::copy_buffer_f32(
             workspace.full_attn,
             kv_count,
             0,
             state.k_cache_device,
             cache_offset,
             run_error) &&
           cuda::copy_buffer_f32(
             v_raw_f32,
             kv_count,
             0,
             state.v_cache_device,
             cache_offset,
             run_error);
  };

  const auto run_attention_decode = [&](std::string & run_error) -> bool {
    if (use_bf16_cache) {
      return cuda::run_full_attention_decode_gqa_bf16_cache(
        workspace.full_q,
        workspace.full_gate,
        state.k_cache_bf16_device,
        state.v_cache_bf16_device,
        dims.n_heads,
        dims.n_kv_heads,
        dims.head_dim,
        position + 1,
        workspace.full_attn,
        workspace.full_scores,
        run_error);
    }
    return cuda::run_full_attention_decode_gqa(
      workspace.full_q,
      workspace.full_gate,
      state.k_cache_device,
      state.v_cache_device,
      dims.n_heads,
      dims.n_kv_heads,
      dims.head_dim,
      position + 1,
      workspace.full_attn,
      workspace.full_scores,
      run_error);
  };

  return run_qkv_projections(error_message) &&
         split_q_gate(error_message) &&
         cuda::run_rms_norm_per_head_f32(
           workspace.full_q,
           layer.full.q_norm_device,
           dims.n_heads,
           dims.head_dim,
           dims.rms_eps,
           workspace.full_q,
           error_message) &&
         cuda::run_apply_rope_inplace_f32(
           workspace.full_q,
           dims.n_heads,
           dims.head_dim,
           dims.rope_dim,
           position,
           dims.rope_theta,
           error_message) &&
         normalize_and_cache_kv(error_message) &&
         run_attention_decode(error_message) &&
         cuda::run_matvec_f32_device(layer.full.o_proj.device_matrix, workspace.full_attn, out_hidden, error_message);
}

bool run_linear_attention_step(
  const LayerWeights & layer,
  const RuntimeDims & dims,
  LinearAttentionState & state,
  const std::vector<float> & x,
  std::vector<float> & out,
  const bool use_cuda,
  CudaForwardWorkspace * cuda_workspace,
  std::string & error_message) {
  std::vector<float> mixed_qkv;
  std::vector<float> z_vec;
  std::vector<float> b_vec;
  std::vector<float> a_vec;
  const bool use_cuda_linear_kernel =
    use_cuda && cuda_workspace != nullptr && cuda_workspace->has_device_buffers && state.has_device_state &&
    layer.linear.has_device_params && layer.linear.in_proj_qkv.has_device_matrix && layer.linear.in_proj_z.has_device_matrix &&
    layer.linear.in_proj_b.has_device_matrix && layer.linear.in_proj_a.has_device_matrix && layer.linear.conv1d.has_device_matrix &&
    layer.linear.out_proj.has_device_matrix;
  if (use_cuda_linear_kernel) {
    if (!cuda::upload_to_buffer_f32(x.data(), x.size(), cuda_workspace->hidden_in, 0, error_message) ||
        !cuda::run_matvec_f32_device(layer.linear.in_proj_qkv.device_matrix, cuda_workspace->hidden_in, cuda_workspace->linear_mixed_qkv, error_message) ||
        !cuda::run_matvec_f32_device(layer.linear.in_proj_z.device_matrix, cuda_workspace->hidden_in, cuda_workspace->linear_z, error_message) ||
        !cuda::run_matvec_f32_device(layer.linear.in_proj_b.device_matrix, cuda_workspace->hidden_in, cuda_workspace->linear_b, error_message) ||
        !cuda::run_matvec_f32_device(layer.linear.in_proj_a.device_matrix, cuda_workspace->hidden_in, cuda_workspace->linear_a, error_message) ||
        !cuda::run_linear_attention_decode(
          cuda_workspace->linear_mixed_qkv,
          cuda_workspace->linear_z,
          cuda_workspace->linear_b,
          cuda_workspace->linear_a,
          layer.linear.conv1d.device_matrix,
          layer.linear.norm_device,
          layer.linear.dt_bias_device,
          layer.linear.ssm_a_device,
          dims.linear_kernel,
          dims.linear_num_k_heads,
          dims.linear_num_v_heads,
          dims.linear_head_k_dim,
          dims.linear_head_v_dim,
          dims.rms_eps,
          state.conv_state_device,
          state.recurrent_state_device,
          cuda_workspace->linear_conv_out,
          cuda_workspace->linear_gated_norm,
          error_message) ||
        !cuda::run_matvec_f32_device(layer.linear.out_proj.device_matrix, cuda_workspace->linear_gated_norm, cuda_workspace->hidden_out, error_message) ||
        !cuda::download_from_buffer_f32(
          cuda_workspace->hidden_out,
          static_cast<std::size_t>(dims.hidden),
          0,
          out,
          error_message)) {
      return false;
    }
    return true;
  }

  const bool use_cuda_projection_batch =
    use_cuda && cuda_workspace != nullptr && cuda_workspace->has_device_buffers;
  if (use_cuda_projection_batch) {
    if (!cuda::upload_to_buffer_f32(x.data(), x.size(), cuda_workspace->hidden_in, 0, error_message) ||
        !project_from_uploaded_hidden_cuda(
          layer.linear.in_proj_qkv,
          *cuda_workspace,
          static_cast<std::size_t>(dims.linear_conv_channels),
          mixed_qkv,
          error_message) ||
        !project_from_uploaded_hidden_cuda(
          layer.linear.in_proj_z,
          *cuda_workspace,
          static_cast<std::size_t>(dims.linear_v_dim),
          z_vec,
          error_message) ||
        !project_from_uploaded_hidden_cuda(
          layer.linear.in_proj_b,
          *cuda_workspace,
          static_cast<std::size_t>(dims.linear_num_v_heads),
          b_vec,
          error_message) ||
        !project_from_uploaded_hidden_cuda(
          layer.linear.in_proj_a,
          *cuda_workspace,
          static_cast<std::size_t>(dims.linear_num_v_heads),
          a_vec,
          error_message)) {
      return false;
    }
  } else {
    if (!matvec_2d(layer.linear.in_proj_qkv, x, mixed_qkv, use_cuda, error_message) ||
        !matvec_2d(layer.linear.in_proj_z, x, z_vec, use_cuda, error_message) ||
        !matvec_2d(layer.linear.in_proj_b, x, b_vec, use_cuda, error_message) ||
        !matvec_2d(layer.linear.in_proj_a, x, a_vec, use_cuda, error_message)) {
      return false;
    }
  }

  std::vector<float> beta(static_cast<std::size_t>(dims.linear_num_v_heads), 0.0f);
  std::vector<float> alpha(static_cast<std::size_t>(dims.linear_num_v_heads), 0.0f);
  for (int h = 0; h < dims.linear_num_v_heads; ++h) {
    beta[static_cast<std::size_t>(h)] = sigmoidf_stable(b_vec[static_cast<std::size_t>(h)]);
    const float pre_gate = softplusf_stable(a_vec[static_cast<std::size_t>(h)] + layer.linear.dt_bias.data[static_cast<std::size_t>(h)]);
    alpha[static_cast<std::size_t>(h)] = std::exp(pre_gate * layer.linear.ssm_a[static_cast<std::size_t>(h)]);
  }

  const int conv_hist = dims.linear_kernel - 1;
  std::vector<float> conv_window(static_cast<std::size_t>(dims.linear_kernel * dims.linear_conv_channels));
  if (conv_hist > 0) {
    std::memcpy(
      conv_window.data(),
      state.conv_state.data(),
      static_cast<std::size_t>(conv_hist * dims.linear_conv_channels) * sizeof(float));
  }
  std::memcpy(
    conv_window.data() + static_cast<std::size_t>(conv_hist * dims.linear_conv_channels),
    mixed_qkv.data(),
    static_cast<std::size_t>(dims.linear_conv_channels) * sizeof(float));

  if (conv_hist > 0) {
    std::memcpy(
      state.conv_state.data(),
      conv_window.data() + static_cast<std::size_t>(dims.linear_conv_channels),
      static_cast<std::size_t>(conv_hist * dims.linear_conv_channels) * sizeof(float));
  }

  std::vector<float> conv_out(static_cast<std::size_t>(dims.linear_conv_channels), 0.0f);
  for (int c = 0; c < dims.linear_conv_channels; ++c) {
    float s = 0.0f;
    const float * w = layer.linear.conv1d.data.data() +
                      static_cast<std::size_t>(c) * static_cast<std::size_t>(dims.linear_kernel);
    for (int k = 0; k < dims.linear_kernel; ++k) {
      const std::size_t idx = static_cast<std::size_t>(k * dims.linear_conv_channels + c);
      s += conv_window[idx] * w[k];
    }
    conv_out[static_cast<std::size_t>(c)] = siluf(s);
  }

  std::vector<float> q(conv_out.begin(), conv_out.begin() + dims.linear_q_dim);
  std::vector<float> k(conv_out.begin() + dims.linear_q_dim, conv_out.begin() + 2 * dims.linear_q_dim);
  std::vector<float> v(conv_out.begin() + 2 * dims.linear_q_dim, conv_out.end());
  l2_norm_per_head(q, dims.linear_num_k_heads, dims.linear_head_k_dim);
  l2_norm_per_head(k, dims.linear_num_k_heads, dims.linear_head_k_dim);

  const float q_scale = 1.0f / std::sqrt(static_cast<float>(dims.linear_head_k_dim));
  for (float & qv : q) {
    qv *= q_scale;
  }

  std::vector<float> core_out(static_cast<std::size_t>(dims.linear_v_dim), 0.0f);
  for (int h = 0; h < dims.linear_num_v_heads; ++h) {
    const std::size_t q_base = static_cast<std::size_t>(h * dims.linear_head_k_dim);
    const std::size_t v_base = static_cast<std::size_t>(h * dims.linear_head_v_dim);
    const std::size_t s_base =
      static_cast<std::size_t>(h) * static_cast<std::size_t>(dims.linear_head_v_dim * dims.linear_head_v_dim);
    float * s = state.recurrent_state.data() + s_base;

    for (int i = 0; i < dims.linear_head_v_dim; ++i) {
      for (int j = 0; j < dims.linear_head_v_dim; ++j) {
        s[static_cast<std::size_t>(i * dims.linear_head_v_dim + j)] *= alpha[static_cast<std::size_t>(h)];
      }
    }

    std::vector<float> sk(static_cast<std::size_t>(dims.linear_head_v_dim), 0.0f);
    for (int i = 0; i < dims.linear_head_v_dim; ++i) {
      float sum = 0.0f;
      const float * s_row = s + static_cast<std::size_t>(i * dims.linear_head_v_dim);
      for (int j = 0; j < dims.linear_head_v_dim; ++j) {
        sum += s_row[j] * k[q_base + static_cast<std::size_t>(j)];
      }
      sk[static_cast<std::size_t>(i)] = sum;
    }

    for (int i = 0; i < dims.linear_head_v_dim; ++i) {
      const float delta = (v[v_base + static_cast<std::size_t>(i)] - sk[static_cast<std::size_t>(i)]) *
                          beta[static_cast<std::size_t>(h)];
      float * s_row = s + static_cast<std::size_t>(i * dims.linear_head_v_dim);
      for (int j = 0; j < dims.linear_head_v_dim; ++j) {
        s_row[j] += delta * k[q_base + static_cast<std::size_t>(j)];
      }
    }

    for (int i = 0; i < dims.linear_head_v_dim; ++i) {
      float sum = 0.0f;
      const float * s_row = s + static_cast<std::size_t>(i * dims.linear_head_v_dim);
      for (int j = 0; j < dims.linear_head_v_dim; ++j) {
        sum += s_row[j] * q[q_base + static_cast<std::size_t>(j)];
      }
      core_out[v_base + static_cast<std::size_t>(i)] = sum;
    }
  }

  std::vector<float> gated_norm(static_cast<std::size_t>(dims.linear_v_dim), 0.0f);
  for (int h = 0; h < dims.linear_num_v_heads; ++h) {
    const std::size_t base = static_cast<std::size_t>(h * dims.linear_head_v_dim);
    float sq_sum = 0.0f;
    for (int d = 0; d < dims.linear_head_v_dim; ++d) {
      const float cv = core_out[base + static_cast<std::size_t>(d)];
      sq_sum += cv * cv;
    }
    const float inv = 1.0f / std::sqrt(sq_sum / static_cast<float>(dims.linear_head_v_dim) + dims.rms_eps);
    for (int d = 0; d < dims.linear_head_v_dim; ++d) {
      const std::size_t idx = base + static_cast<std::size_t>(d);
      gated_norm[idx] = core_out[idx] * inv * layer.linear.norm.data[static_cast<std::size_t>(d)] * siluf(z_vec[idx]);
    }
  }

  if (!matvec_2d(layer.linear.out_proj, gated_norm, out, use_cuda, error_message)) {
    return false;
  }
  return true;
}

bool run_full_attention_step(
  const LayerWeights & layer,
  const RuntimeDims & dims,
  FullAttentionState & state,
  const std::vector<float> & x,
  const int position,
  std::vector<float> & out,
  const bool use_cuda,
  CudaForwardWorkspace * cuda_workspace,
  std::string & error_message) {
  std::vector<float> q_full;
  std::vector<float> k_flat;
  std::vector<float> v_flat;
  const bool use_cuda_projection_batch =
    use_cuda && cuda_workspace != nullptr && cuda_workspace->has_device_buffers;
  const std::size_t full_q_out = static_cast<std::size_t>(dims.n_heads * dims.head_dim * 2);
  const std::size_t full_kv_out = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  if (use_cuda_projection_batch) {
    if (!cuda::upload_to_buffer_f32(x.data(), x.size(), cuda_workspace->hidden_in, 0, error_message) ||
        !project_from_uploaded_hidden_cuda(layer.full.q_proj, *cuda_workspace, full_q_out, q_full, error_message) ||
        !project_from_uploaded_hidden_cuda(layer.full.k_proj, *cuda_workspace, full_kv_out, k_flat, error_message) ||
        !project_from_uploaded_hidden_cuda(layer.full.v_proj, *cuda_workspace, full_kv_out, v_flat, error_message)) {
      return false;
    }
  } else {
    if (!matvec_2d(layer.full.q_proj, x, q_full, use_cuda, error_message) ||
        !matvec_2d(layer.full.k_proj, x, k_flat, use_cuda, error_message) ||
        !matvec_2d(layer.full.v_proj, x, v_flat, use_cuda, error_message)) {
      return false;
    }
  }

  const int q_span = dims.head_dim * 2;
  std::vector<float> q(static_cast<std::size_t>(dims.n_heads * dims.head_dim));
  std::vector<float> gate(static_cast<std::size_t>(dims.n_heads * dims.head_dim));
  for (int h = 0; h < dims.n_heads; ++h) {
    const std::size_t src = static_cast<std::size_t>(h * q_span);
    const std::size_t dst = static_cast<std::size_t>(h * dims.head_dim);
    std::memcpy(q.data() + dst, q_full.data() + src, static_cast<std::size_t>(dims.head_dim) * sizeof(float));
    std::memcpy(gate.data() + dst, q_full.data() + src + static_cast<std::size_t>(dims.head_dim),
                static_cast<std::size_t>(dims.head_dim) * sizeof(float));
  }

  std::vector<float> q_normed;
  std::vector<float> k_normed;
  rms_norm_per_head_qwen3next(q, dims.n_heads, dims.head_dim, layer.full.q_norm, dims.rms_eps, q_normed);
  rms_norm_per_head_qwen3next(k_flat, dims.n_kv_heads, dims.head_dim, layer.full.k_norm, dims.rms_eps, k_normed);

  apply_rope_inplace(q_normed, dims.n_heads, dims.head_dim, dims.rope_dim, position, dims.rope_theta);
  apply_rope_inplace(k_normed, dims.n_kv_heads, dims.head_dim, dims.rope_dim, position, dims.rope_theta);

  const std::size_t token_stride = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  std::memcpy(
    state.k_cache.data() + static_cast<std::size_t>(position) * token_stride,
    k_normed.data(),
    token_stride * sizeof(float));
  std::memcpy(
    state.v_cache.data() + static_cast<std::size_t>(position) * token_stride,
    v_flat.data(),
    token_stride * sizeof(float));

  const bool use_cuda_full_kernel =
    use_cuda && state.has_device_state && cuda_workspace != nullptr && cuda_workspace->has_device_buffers &&
    layer.full.o_proj.has_device_matrix;
  if (use_cuda_full_kernel) {
    const std::size_t offset = static_cast<std::size_t>(position) * token_stride;
    const std::size_t full_q_count = static_cast<std::size_t>(dims.n_heads * dims.head_dim);
    if (!cuda::upload_to_buffer_f32(k_normed.data(), token_stride, state.k_cache_device, offset, error_message) ||
        !cuda::upload_to_buffer_f32(v_flat.data(), token_stride, state.v_cache_device, offset, error_message) ||
        !cuda::upload_to_buffer_f32(q_normed.data(), full_q_count, cuda_workspace->full_q, 0, error_message) ||
        !cuda::upload_to_buffer_f32(gate.data(), full_q_count, cuda_workspace->full_gate, 0, error_message) ||
        !cuda::run_full_attention_decode_gqa(
          cuda_workspace->full_q,
          cuda_workspace->full_gate,
          state.k_cache_device,
          state.v_cache_device,
          dims.n_heads,
          dims.n_kv_heads,
          dims.head_dim,
          position + 1,
          cuda_workspace->full_attn,
          cuda_workspace->full_scores,
          error_message) ||
        !cuda::run_matvec_f32_device(layer.full.o_proj.device_matrix, cuda_workspace->full_attn, cuda_workspace->hidden_out, error_message) ||
        !cuda::download_from_buffer_f32(
          cuda_workspace->hidden_out,
          static_cast<std::size_t>(dims.hidden),
          0,
          out,
          error_message)) {
      return false;
    }
    return true;
  }

  const int n_rep = dims.n_heads / dims.n_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));
  const int seq_len = position + 1;
  std::vector<float> attn_cat(static_cast<std::size_t>(dims.n_heads * dims.head_dim), 0.0f);
  std::vector<float> scores(static_cast<std::size_t>(seq_len), 0.0f);

  for (int h = 0; h < dims.n_heads; ++h) {
    const int kvh = h / n_rep;
    const std::size_t q_base = static_cast<std::size_t>(h * dims.head_dim);

    float max_score = -std::numeric_limits<float>::infinity();
    for (int t = 0; t < seq_len; ++t) {
      const float * k_ptr = state.k_cache.data() +
                           (static_cast<std::size_t>(t) * static_cast<std::size_t>(dims.n_kv_heads) +
                            static_cast<std::size_t>(kvh)) *
                             static_cast<std::size_t>(dims.head_dim);
      float dot = 0.0f;
      for (int d = 0; d < dims.head_dim; ++d) {
        dot += q_normed[q_base + static_cast<std::size_t>(d)] * k_ptr[d];
      }
      dot *= scale;
      scores[static_cast<std::size_t>(t)] = dot;
      if (dot > max_score) {
        max_score = dot;
      }
    }

    float denom = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      const float ev = std::exp(scores[static_cast<std::size_t>(t)] - max_score);
      scores[static_cast<std::size_t>(t)] = ev;
      denom += ev;
    }
    if (denom <= 0.0f) {
      denom = 1.0f;
    }

    for (int d = 0; d < dims.head_dim; ++d) {
      float acc = 0.0f;
      for (int t = 0; t < seq_len; ++t) {
        const float p = scores[static_cast<std::size_t>(t)] / denom;
        const float * v_ptr = state.v_cache.data() +
                             (static_cast<std::size_t>(t) * static_cast<std::size_t>(dims.n_kv_heads) +
                              static_cast<std::size_t>(kvh)) *
                               static_cast<std::size_t>(dims.head_dim);
        acc += p * v_ptr[d];
      }
      attn_cat[q_base + static_cast<std::size_t>(d)] = acc * sigmoidf_stable(gate[q_base + static_cast<std::size_t>(d)]);
    }
  }

  if (!matvec_2d(layer.full.o_proj, attn_cat, out, use_cuda, error_message)) {
    return false;
  }
  return true;
}

bool compute_next_logits_from_embedding(
  const TensorData & embed,
  const std::vector<float> & hidden,
  const bool use_cuda,
  std::vector<float> & out_logits,
  std::string & error_message) {
  const int vocab = static_cast<int>(embed.shape[0]);
  const int dim = static_cast<int>(embed.shape[1]);

  if (use_cuda) {
    if (!embed.has_device_matrix || embed.device_matrix.data == nullptr) {
      error_message = "CUDA logits requested but embedding matrix is not uploaded.";
      return false;
    }
    if (!cuda::run_matvec_f32(embed.device_matrix, hidden, out_logits, error_message)) {
      return false;
    }
  } else {
    out_logits.assign(static_cast<std::size_t>(vocab), 0.0f);
    for (int token = 0; token < vocab; ++token) {
      const float * row = embed.data.data() + static_cast<std::size_t>(token) * static_cast<std::size_t>(dim);
      float logit = 0.0f;
      for (int d = 0; d < dim; ++d) {
        logit += row[d] * hidden[static_cast<std::size_t>(d)];
      }
      out_logits[static_cast<std::size_t>(token)] = logit;
    }
  }

  return true;
}

int argmax_index(const std::vector<float> & values) {
  int best_idx = 0;
  float best_value = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    if (values[static_cast<std::size_t>(i)] > best_value) {
      best_value = values[static_cast<std::size_t>(i)];
      best_idx = i;
    }
  }
  return best_idx;
}

void apply_repetition_penalty_inplace(
  std::vector<float> & logits,
  const std::vector<int> & token_counts,
  const float repetition_penalty) {
  if (repetition_penalty <= 1.0f) {
    return;
  }
  const std::size_t n = std::min(logits.size(), token_counts.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (token_counts[i] <= 0) {
      continue;
    }
    if (logits[i] > 0.0f) {
      logits[i] /= repetition_penalty;
    } else {
      logits[i] *= repetition_penalty;
    }
  }
}

bool sample_token_from_logits(
  const std::vector<float> & raw_logits,
  const SamplingOptions & sampling,
  const std::vector<int> & token_counts,
  std::mt19937 & rng,
  int & out_token,
  std::string & error_message) {
  if (raw_logits.empty()) {
    error_message = "Sampling logits are empty.";
    return false;
  }
  if (sampling.temperature < 0.0f) {
    error_message = "temperature must be >= 0.";
    return false;
  }
  if (sampling.top_p <= 0.0f || sampling.top_p > 1.0f) {
    error_message = "top_p must be in (0, 1].";
    return false;
  }
  if (sampling.top_k < 0) {
    error_message = "top_k must be >= 0.";
    return false;
  }
  if (sampling.repetition_penalty < 1.0f) {
    error_message = "repeat_penalty must be >= 1.0.";
    return false;
  }

  std::vector<float> logits = raw_logits;
  apply_repetition_penalty_inplace(logits, token_counts, sampling.repetition_penalty);

  if (sampling.temperature <= 1.0e-6f) {
    out_token = argmax_index(logits);
    return true;
  }

  const float inv_temp = 1.0f / sampling.temperature;
  for (float & v : logits) {
    v *= inv_temp;
  }

  std::vector<int> candidate_ids(logits.size());
  std::iota(candidate_ids.begin(), candidate_ids.end(), 0);
  if (sampling.top_k > 0 && sampling.top_k < static_cast<int>(candidate_ids.size())) {
    const std::size_t keep = static_cast<std::size_t>(sampling.top_k);
    std::nth_element(
      candidate_ids.begin(),
      candidate_ids.begin() + static_cast<std::ptrdiff_t>(keep),
      candidate_ids.end(),
      [&](const int a, const int b) {
        return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
      });
    candidate_ids.resize(keep);
  }

  std::sort(candidate_ids.begin(), candidate_ids.end(), [&](const int a, const int b) {
    return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
  });

  if (candidate_ids.empty()) {
    error_message = "Sampling candidates are empty after top_k filtering.";
    return false;
  }

  float max_logit = -std::numeric_limits<float>::infinity();
  for (const int id : candidate_ids) {
    max_logit = std::max(max_logit, logits[static_cast<std::size_t>(id)]);
  }

  std::vector<float> probs(candidate_ids.size(), 0.0f);
  float denom = 0.0f;
  for (std::size_t i = 0; i < candidate_ids.size(); ++i) {
    const float p = std::exp(logits[static_cast<std::size_t>(candidate_ids[i])] - max_logit);
    probs[i] = p;
    denom += p;
  }
  if (!(denom > 0.0f)) {
    out_token = candidate_ids.front();
    return true;
  }
  for (float & p : probs) {
    p /= denom;
  }

  std::size_t nucleus_keep = probs.size();
  if (sampling.top_p < 1.0f) {
    float cumulative = 0.0f;
    nucleus_keep = 0;
    for (; nucleus_keep < probs.size(); ++nucleus_keep) {
      cumulative += probs[nucleus_keep];
      if (cumulative >= sampling.top_p) {
        ++nucleus_keep;
        break;
      }
    }
    nucleus_keep = std::max<std::size_t>(1, std::min(nucleus_keep, probs.size()));
  }

  candidate_ids.resize(nucleus_keep);
  probs.resize(nucleus_keep);
  float renorm = 0.0f;
  for (const float p : probs) {
    renorm += p;
  }
  if (!(renorm > 0.0f)) {
    out_token = candidate_ids.front();
    return true;
  }
  for (float & p : probs) {
    p /= renorm;
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float r = dist(rng);
  float cumulative = 0.0f;
  for (std::size_t i = 0; i < probs.size(); ++i) {
    cumulative += probs[i];
    if (r <= cumulative || i + 1 == probs.size()) {
      out_token = candidate_ids[i];
      return true;
    }
  }

  out_token = candidate_ids.back();
  return true;
}

bool generated_ends_with_sequence(
  const std::vector<std::int32_t> & generated_tokens,
  const std::vector<std::int32_t> & stop_sequence) {
  if (stop_sequence.empty() || stop_sequence.size() > generated_tokens.size()) {
    return false;
  }
  const std::size_t start = generated_tokens.size() - stop_sequence.size();
  for (std::size_t i = 0; i < stop_sequence.size(); ++i) {
    if (generated_tokens[start + i] != stop_sequence[i]) {
      return false;
    }
  }
  return true;
}

bool project_from_uploaded_hidden_cuda(
  const TensorData & projection,
  CudaForwardWorkspace & workspace,
  const std::size_t output_count,
  std::vector<float> & out_values,
  std::string & error_message) {
  if (!projection.has_device_matrix || projection.device_matrix.data == nullptr) {
    error_message = "CUDA projection requested but device matrix is unavailable.";
    return false;
  }
  if (!workspace.has_device_buffers || workspace.hidden_in.data == nullptr || workspace.projection_out.data == nullptr) {
    error_message = "CUDA forward workspace is not initialized.";
    return false;
  }
  if (workspace.projection_out.count < output_count) {
    error_message = "CUDA projection workspace is smaller than requested output.";
    return false;
  }
  if (!cuda::run_matvec_f32_device(
        projection.device_matrix,
        workspace.hidden_in,
        workspace.projection_out,
        error_message) ||
      !cuda::download_from_buffer_f32(workspace.projection_out, output_count, 0, out_values, error_message)) {
    return false;
  }
  return true;
}

bool run_mlp_block_cuda_hybrid(
  const LayerWeights & layer,
  const std::vector<float> & post_norm,
  CudaForwardWorkspace & workspace,
  std::vector<float> & mlp_out,
  std::string & error_message) {
  if (!workspace.has_device_buffers) {
    error_message = "CUDA forward workspace is not initialized.";
    return false;
  }
  if (!layer.mlp_gate.has_device_matrix || !layer.mlp_up.has_device_matrix || !layer.mlp_down.has_device_matrix) {
    error_message = "CUDA MLP path requested but one or more MLP weights are not uploaded.";
    return false;
  }

  const std::size_t hidden_count = post_norm.size();
  const std::size_t intermediate_count = static_cast<std::size_t>(layer.mlp_gate.shape[0]);
  const std::size_t out_hidden_count = static_cast<std::size_t>(layer.mlp_down.shape[0]);
  if (workspace.hidden_in.count < hidden_count || workspace.hidden_out.count < out_hidden_count ||
      workspace.intermediate_a.count < intermediate_count || workspace.intermediate_b.count < intermediate_count ||
      workspace.intermediate_hidden.count < intermediate_count) {
    error_message = "CUDA forward workspace buffer sizes do not match model dimensions.";
    return false;
  }

  if (!cuda::upload_to_buffer_f32(post_norm.data(), hidden_count, workspace.hidden_in, 0, error_message) ||
      !cuda::run_matvec_f32_device(layer.mlp_gate.device_matrix, workspace.hidden_in, workspace.intermediate_a, error_message) ||
      !cuda::run_matvec_f32_device(layer.mlp_up.device_matrix, workspace.hidden_in, workspace.intermediate_b, error_message) ||
      !cuda::run_silu_mul_f32(
        workspace.intermediate_a,
        workspace.intermediate_b,
        intermediate_count,
        workspace.intermediate_hidden,
        error_message) ||
      !cuda::run_matvec_f32_device(layer.mlp_down.device_matrix, workspace.intermediate_hidden, workspace.hidden_out, error_message) ||
      !cuda::download_from_buffer_f32(workspace.hidden_out, out_hidden_count, 0, mlp_out, error_message)) {
    return false;
  }

  return true;
}

bool run_forward_single_token_cuda_device_core(
  const ModelWeights & weights,
  const RuntimeDims & dims,
  ModelState & state,
  const int position,
  const bool use_bf16_residual_mlp,
  const bool use_cuda_gpu_sampling,
  const bool profile_cuda_sync,
  CudaForwardWorkspace & workspace,
  std::vector<float> & next_logits,
  DecodeProfilingAccumulator * profiling,
  std::string & error_message) {
  const std::size_t hidden_count = static_cast<std::size_t>(dims.hidden);
  const std::size_t intermediate_count = static_cast<std::size_t>(dims.intermediate);

  int full_idx = 0;
  int linear_idx = 0;
  for (int il = 0; il < dims.n_layers; ++il) {
    const LayerWeights & layer = weights.layers[static_cast<std::size_t>(il)];
    if (!layer.has_device_norms) {
      error_message = "CUDA forward device path requires uploaded layer norm weights.";
      return false;
    }

    if (use_bf16_residual_mlp) {
      if (!cuda::run_rms_norm_bf16_to_f32(
            workspace.hidden_in_bf16,
            layer.input_layernorm_device,
            hidden_count,
            dims.rms_eps,
            workspace.full_q,
            error_message)) {
        return false;
      }
    } else {
      if (!cuda::run_rms_norm_f32(
            workspace.hidden_in,
            layer.input_layernorm_device,
            hidden_count,
            dims.rms_eps,
            workspace.full_q,
            error_message)) {
        return false;
      }
    }

    if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
      return false;
    }
    const auto attention_start = std::chrono::steady_clock::now();
    if (layer.is_linear) {
      if (!run_linear_attention_step_cuda_device(
            layer,
            dims,
            static_cast<std::size_t>(il),
            state.linear_states[static_cast<std::size_t>(linear_idx)],
            workspace.full_q,
            workspace,
            workspace.hidden_out,
            error_message)) {
        return false;
      }
      ++linear_idx;
    } else {
      if (!run_full_attention_step_cuda_device(
            layer,
            dims,
            state.full_states[static_cast<std::size_t>(full_idx)],
            workspace.full_q,
            position,
            workspace,
            workspace.hidden_out,
            error_message)) {
        return false;
      }
      ++full_idx;
    }
    if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
      return false;
    }
    if (profiling != nullptr) {
      profiling->attention_ms += elapsed_ms(attention_start);
    }

    if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
      return false;
    }
    const auto mlp_start = std::chrono::steady_clock::now();
    if (use_bf16_residual_mlp) {
      if (!cuda::copy_buffer_f32_to_bf16(
            workspace.hidden_out,
            hidden_count,
            0,
            workspace.hidden_out_bf16,
            0,
            error_message) ||
          !cuda::run_add_bf16(
            workspace.hidden_in_bf16,
            workspace.hidden_out_bf16,
            hidden_count,
            workspace.full_attn_bf16,
            error_message) ||
          !cuda::run_rms_norm_bf16(
            workspace.full_attn_bf16,
            layer.post_attention_layernorm_device,
            hidden_count,
            dims.rms_eps,
            workspace.full_q_bf16,
            error_message)) {
        return false;
      }
    } else {
      if (!cuda::run_add_f32(workspace.hidden_in, workspace.hidden_out, hidden_count, workspace.full_attn, error_message) ||
          !cuda::run_rms_norm_f32(
            workspace.full_attn,
            layer.post_attention_layernorm_device,
            hidden_count,
            dims.rms_eps,
            workspace.full_q,
            error_message)) {
        return false;
      }
    }

    const auto run_mlp_direct = [&](std::string & run_error) -> bool {
      bool mlp_ok = false;
      if (use_bf16_residual_mlp) {
        if (layer.has_device_mlp_gate_up && layer.mlp_gate_up_device.data != nullptr) {
          const std::size_t packed_count = intermediate_count * 2;
          if (workspace.projection_out_bf16.count < packed_count) {
            run_error = "CUDA BF16 workspace projection buffer is too small for packed MLP projections.";
            return false;
          }
          const cuda::CudaDeviceBufferBF16 gate = buffer_slice_bf16(workspace.projection_out_bf16, 0, intermediate_count);
          const cuda::CudaDeviceBufferBF16 up =
            buffer_slice_bf16(workspace.projection_out_bf16, intermediate_count, intermediate_count);
          if (gate.data == nullptr || up.data == nullptr) {
            run_error = "Failed to create packed BF16 MLP projection slices.";
            return false;
          }
          mlp_ok = cuda::run_matvec_bf16_device_input_bf16_output_bf16(
                     layer.mlp_gate_up_device,
                     workspace.full_q_bf16,
                     workspace.projection_out_bf16,
                     run_error) &&
                   cuda::run_silu_mul_bf16(gate, up, intermediate_count, workspace.intermediate_hidden_bf16, run_error);
        } else {
          mlp_ok = cuda::run_matvec_bf16_device_input_bf16_output_bf16(
                     layer.mlp_gate.device_matrix,
                     workspace.full_q_bf16,
                     workspace.intermediate_a_bf16,
                     run_error) &&
                   cuda::run_matvec_bf16_device_input_bf16_output_bf16(
                     layer.mlp_up.device_matrix,
                     workspace.full_q_bf16,
                     workspace.intermediate_b_bf16,
                     run_error) &&
                   cuda::run_silu_mul_bf16(
                     workspace.intermediate_a_bf16,
                     workspace.intermediate_b_bf16,
                     intermediate_count,
                     workspace.intermediate_hidden_bf16,
                     run_error);
        }
        return mlp_ok &&
               cuda::run_matvec_bf16_device_input_bf16_output_bf16(
                 layer.mlp_down.device_matrix,
                 workspace.intermediate_hidden_bf16,
                 workspace.hidden_out_bf16,
                 run_error) &&
               cuda::run_add_bf16(
                 workspace.full_attn_bf16,
                 workspace.hidden_out_bf16,
                 hidden_count,
                 workspace.hidden_in_bf16,
                 run_error);
      } else {
        if (layer.has_device_mlp_gate_up && layer.mlp_gate_up_device.data != nullptr) {
          const std::size_t packed_count = intermediate_count * 2;
          if (workspace.projection_out.count < packed_count) {
            run_error = "CUDA workspace projection buffer is too small for packed MLP projections.";
            return false;
          }
          const cuda::CudaDeviceBufferF32 gate = buffer_slice_f32(workspace.projection_out, 0, intermediate_count);
          const cuda::CudaDeviceBufferF32 up = buffer_slice_f32(workspace.projection_out, intermediate_count, intermediate_count);
          if (gate.data == nullptr || up.data == nullptr) {
            run_error = "Failed to create packed MLP projection slices.";
            return false;
          }
          mlp_ok = cuda::run_matvec_f32_device(layer.mlp_gate_up_device, workspace.full_q, workspace.projection_out, run_error) &&
                   cuda::run_silu_mul_f32(gate, up, intermediate_count, workspace.intermediate_hidden, run_error);
        } else {
          mlp_ok = cuda::run_matvec_f32_device(layer.mlp_gate.device_matrix, workspace.full_q, workspace.intermediate_a, run_error) &&
                   cuda::run_matvec_f32_device(layer.mlp_up.device_matrix, workspace.full_q, workspace.intermediate_b, run_error) &&
                   cuda::run_silu_mul_f32(
                     workspace.intermediate_a,
                     workspace.intermediate_b,
                     intermediate_count,
                     workspace.intermediate_hidden,
                     run_error);
        }
        return mlp_ok &&
               cuda::run_matvec_f32_device(layer.mlp_down.device_matrix, workspace.intermediate_hidden, workspace.hidden_out, run_error) &&
               cuda::run_add_f32(workspace.full_attn, workspace.hidden_out, hidden_count, workspace.hidden_in, run_error);
      }
    };

    bool mlp_done = false;
    const std::size_t layer_slot = static_cast<std::size_t>(il);
    if (layer_slot < workspace.mlp_graphs.size()) {
      if (workspace.mlp_graphs[layer_slot].ready) {
        mlp_done = cuda::launch_captured_graph(workspace.mlp_graphs[layer_slot], error_message);
        if (!mlp_done) {
          workspace.mlp_graph_disabled[layer_slot] = true;
          cuda::free_captured_graph(workspace.mlp_graphs[layer_slot]);
        }
      }

      if (!mlp_done && !workspace.mlp_graph_disabled[layer_slot] && workspace.mlp_graph_warmup_done[layer_slot]) {
        std::string capture_error;
        const bool capture_started = cuda::begin_stream_capture(capture_error);
        bool capture_ok = false;
        if (capture_started) {
          std::string captured_run_error;
          const bool captured_run_ok = run_mlp_direct(captured_run_error);
          std::string capture_end_error;
          const bool capture_end_ok = cuda::end_stream_capture(workspace.mlp_graphs[layer_slot], capture_end_error);
          if (captured_run_ok && capture_end_ok) {
            capture_ok = cuda::launch_captured_graph(workspace.mlp_graphs[layer_slot], capture_error);
          }
        }
        if (!capture_ok) {
          workspace.mlp_graph_disabled[layer_slot] = true;
          cuda::free_captured_graph(workspace.mlp_graphs[layer_slot]);
        } else {
          mlp_done = true;
        }
      }

      if (!mlp_done && !workspace.mlp_graph_disabled[layer_slot] && !workspace.mlp_graph_warmup_done[layer_slot]) {
        std::string warmup_error;
        if (!run_mlp_direct(warmup_error)) {
          error_message = warmup_error;
          return false;
        }
        workspace.mlp_graph_warmup_done[layer_slot] = true;
        mlp_done = true;
      }
    }

    if (!mlp_done) {
      std::string direct_error;
      if (!run_mlp_direct(direct_error)) {
        error_message = direct_error;
        return false;
      }
    }

    if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
      return false;
    }
    if (profiling != nullptr) {
      profiling->mlp_ms += elapsed_ms(mlp_start);
    }
  }

  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  const auto logits_start = std::chrono::steady_clock::now();
  bool ok = false;
  if (use_bf16_residual_mlp) {
    if (!cuda::run_rms_norm_bf16_to_f32(
          workspace.hidden_in_bf16,
          weights.final_norm_device,
          hidden_count,
          dims.rms_eps,
          workspace.hidden_out,
          error_message)) {
      return false;
    }
  } else {
    if (!cuda::run_rms_norm_f32(
          workspace.hidden_in,
          weights.final_norm_device,
          hidden_count,
          dims.rms_eps,
          workspace.hidden_out,
          error_message)) {
      return false;
    }
  }

  if (use_cuda_gpu_sampling) {
    ok = cuda::run_matvec_f32_device(weights.embed_tokens.device_matrix, workspace.hidden_out, workspace.logits, error_message);
  } else {
    std::vector<float> final_hidden;
    if (!cuda::download_from_buffer_f32(workspace.hidden_out, hidden_count, 0, final_hidden, error_message)) {
      return false;
    }
    ok = compute_next_logits_from_embedding(weights.embed_tokens, final_hidden, false, next_logits, error_message);
  }
  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  if (profiling != nullptr) {
    profiling->logits_ms += elapsed_ms(logits_start);
  }
  return ok;
}

bool run_forward_single_token_cuda_device(
  const ModelWeights & weights,
  const RuntimeDims & dims,
  ModelState & state,
  const int token_id,
  const int position,
  const bool use_bf16_residual_mlp,
  const bool use_cuda_gpu_sampling,
  const bool profile_cuda_sync,
  CudaForwardWorkspace & workspace,
  std::vector<float> & next_logits,
  DecodeProfilingAccumulator * profiling,
  std::string & error_message) {
  if (!workspace.has_device_buffers || !workspace.has_gpu_sampling_buffers || !weights.has_device_final_norm ||
      !weights.embed_tokens.has_device_matrix || weights.embed_tokens.device_matrix.data == nullptr) {
    error_message = "CUDA forward device path is not fully initialized.";
    return false;
  }
  if (token_id < 0 || token_id >= dims.vocab_size) {
    error_message = "Token id out of vocabulary range.";
    return false;
  }
  if (use_bf16_residual_mlp &&
      (workspace.hidden_in_bf16.data == nullptr || workspace.hidden_in_bf16.count < static_cast<std::size_t>(dims.hidden))) {
    error_message = "CUDA BF16 hidden-state workspace is not initialized.";
    return false;
  }

  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  const auto embedding_start = std::chrono::steady_clock::now();
  if (!cuda::gather_matrix_row_f32(weights.embed_tokens.device_matrix, token_id, workspace.hidden_in, error_message)) {
    return false;
  }
  if (use_bf16_residual_mlp &&
      !cuda::copy_buffer_f32_to_bf16(
        workspace.hidden_in,
        static_cast<std::size_t>(dims.hidden),
        0,
        workspace.hidden_in_bf16,
        0,
        error_message)) {
    return false;
  }
  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  if (profiling != nullptr) {
    profiling->embedding_ms += elapsed_ms(embedding_start);
  }

  return run_forward_single_token_cuda_device_core(
    weights,
    dims,
    state,
    position,
    use_bf16_residual_mlp,
    use_cuda_gpu_sampling,
    profile_cuda_sync,
    workspace,
    next_logits,
    profiling,
    error_message);
}

bool run_forward_single_token_cuda_device_from_token_buffer(
  const ModelWeights & weights,
  const RuntimeDims & dims,
  ModelState & state,
  const cuda::CudaDeviceBufferF32 & token_id_device,
  const int position,
  const bool use_bf16_residual_mlp,
  const bool use_cuda_gpu_sampling,
  const bool profile_cuda_sync,
  CudaForwardWorkspace & workspace,
  std::vector<float> & next_logits,
  DecodeProfilingAccumulator * profiling,
  std::string & error_message) {
  if (!workspace.has_device_buffers || !workspace.has_gpu_sampling_buffers || !weights.has_device_final_norm ||
      !weights.embed_tokens.has_device_matrix || weights.embed_tokens.device_matrix.data == nullptr) {
    error_message = "CUDA forward device path is not fully initialized.";
    return false;
  }
  if (use_bf16_residual_mlp &&
      (workspace.hidden_in_bf16.data == nullptr || workspace.hidden_in_bf16.count < static_cast<std::size_t>(dims.hidden))) {
    error_message = "CUDA BF16 hidden-state workspace is not initialized.";
    return false;
  }

  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  const auto embedding_start = std::chrono::steady_clock::now();
  if (!cuda::gather_matrix_row_f32_from_token_f32(
        weights.embed_tokens.device_matrix,
        token_id_device,
        workspace.hidden_in,
        error_message)) {
    return false;
  }
  if (use_bf16_residual_mlp &&
      !cuda::copy_buffer_f32_to_bf16(
        workspace.hidden_in,
        static_cast<std::size_t>(dims.hidden),
        0,
        workspace.hidden_in_bf16,
        0,
        error_message)) {
    return false;
  }
  if (!maybe_sync_cuda_for_stage_timing(true, profile_cuda_sync, error_message)) {
    return false;
  }
  if (profiling != nullptr) {
    profiling->embedding_ms += elapsed_ms(embedding_start);
  }

  return run_forward_single_token_cuda_device_core(
    weights,
    dims,
    state,
    position,
    use_bf16_residual_mlp,
    use_cuda_gpu_sampling,
    profile_cuda_sync,
    workspace,
    next_logits,
    profiling,
    error_message);
}

bool run_forward_single_token(
  const ModelWeights & weights,
  const RuntimeDims & dims,
  ModelState & state,
  const int token_id,
  const int position,
  const bool use_cuda,
  const bool use_bf16_residual_mlp,
  const bool profile_cuda_sync,
  std::vector<float> & next_logits,
  const bool use_cuda_gpu_sampling,
  CudaForwardWorkspace * cuda_workspace,
  DecodeProfilingAccumulator * profiling,
  std::string & error_message) {
  if (token_id < 0 || token_id >= dims.vocab_size) {
    error_message = "Token id out of vocabulary range.";
    return false;
  }

  if (profiling != nullptr) {
    ++profiling->forward_pass_tokens;
  }

  const bool use_cuda_device_forward =
    use_cuda && use_cuda_gpu_sampling && cuda_workspace != nullptr && cuda_workspace->has_device_buffers &&
    cuda_workspace->has_gpu_sampling_buffers;
  if (use_cuda_device_forward) {
    return run_forward_single_token_cuda_device(
      weights,
      dims,
      state,
      token_id,
      position,
      use_bf16_residual_mlp,
      use_cuda_gpu_sampling,
      profile_cuda_sync,
      *cuda_workspace,
      next_logits,
      profiling,
      error_message);
  }

  const auto embedding_start = std::chrono::steady_clock::now();
  std::vector<float> x(static_cast<std::size_t>(dims.hidden), 0.0f);
  const float * emb_row = weights.embed_tokens.data.data() +
                         static_cast<std::size_t>(token_id) * static_cast<std::size_t>(dims.hidden);
  std::memcpy(x.data(), emb_row, static_cast<std::size_t>(dims.hidden) * sizeof(float));
  if (profiling != nullptr) {
    profiling->embedding_ms += elapsed_ms(embedding_start);
  }

  int full_idx = 0;
  int linear_idx = 0;
  std::vector<float> normed;
  std::vector<float> attn_out;
  std::vector<float> residual;
  std::vector<float> post_norm;
  std::vector<float> mlp_gate;
  std::vector<float> mlp_up;
  std::vector<float> mlp_hidden;
  std::vector<float> mlp_out;

  for (int il = 0; il < dims.n_layers; ++il) {
    const LayerWeights & layer = weights.layers[static_cast<std::size_t>(il)];
    rms_norm_qwen3next(x, layer.input_layernorm, dims.rms_eps, normed);

    const auto attention_start = std::chrono::steady_clock::now();
    if (layer.is_linear) {
      if (!run_linear_attention_step(
            layer,
            dims,
            state.linear_states[static_cast<std::size_t>(linear_idx)],
            normed,
            attn_out,
            use_cuda,
            cuda_workspace,
            error_message)) {
        return false;
      }
      ++linear_idx;
    } else {
      if (!run_full_attention_step(
            layer,
            dims,
            state.full_states[static_cast<std::size_t>(full_idx)],
            normed,
            position,
            attn_out,
            use_cuda,
            cuda_workspace,
            error_message)) {
        return false;
      }
      ++full_idx;
    }
    if (profiling != nullptr) {
      profiling->attention_ms += elapsed_ms(attention_start);
    }

    const auto mlp_start = std::chrono::steady_clock::now();
    residual.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      residual[i] = x[i] + attn_out[i];
    }

    rms_norm_qwen3next(residual, layer.post_attention_layernorm, dims.rms_eps, post_norm);
    if (use_cuda && cuda_workspace != nullptr && cuda_workspace->has_device_buffers) {
      if (!run_mlp_block_cuda_hybrid(layer, post_norm, *cuda_workspace, mlp_out, error_message)) {
        return false;
      }
    } else {
      if (!matvec_2d(layer.mlp_gate, post_norm, mlp_gate, use_cuda, error_message) ||
          !matvec_2d(layer.mlp_up, post_norm, mlp_up, use_cuda, error_message)) {
        return false;
      }
      mlp_hidden.resize(mlp_gate.size());
      for (std::size_t i = 0; i < mlp_gate.size(); ++i) {
        mlp_hidden[i] = siluf(mlp_gate[i]) * mlp_up[i];
      }
      if (!matvec_2d(layer.mlp_down, mlp_hidden, mlp_out, use_cuda, error_message)) {
        return false;
      }
    }

    x.resize(residual.size());
    for (std::size_t i = 0; i < residual.size(); ++i) {
      x[i] = residual[i] + mlp_out[i];
    }
    if (profiling != nullptr) {
      profiling->mlp_ms += elapsed_ms(mlp_start);
    }
  }

  const auto logits_start = std::chrono::steady_clock::now();
  std::vector<float> final_hidden;
  rms_norm_qwen3next(x, weights.final_norm, dims.rms_eps, final_hidden);
  bool ok = false;
  if (use_cuda_gpu_sampling && use_cuda && cuda_workspace != nullptr && cuda_workspace->has_gpu_sampling_buffers) {
    if (!weights.embed_tokens.has_device_matrix || weights.embed_tokens.device_matrix.data == nullptr) {
      error_message = "CUDA GPU sampling path requested but embedding matrix is not uploaded.";
      ok = false;
    } else {
      ok = cuda::upload_to_buffer_f32(final_hidden.data(), final_hidden.size(), cuda_workspace->hidden_in, 0, error_message) &&
           cuda::run_matvec_f32_device(
             weights.embed_tokens.device_matrix,
             cuda_workspace->hidden_in,
             cuda_workspace->logits,
             error_message);
    }
  } else {
    ok = compute_next_logits_from_embedding(weights.embed_tokens, final_hidden, use_cuda, next_logits, error_message);
  }
  if (profiling != nullptr) {
    profiling->logits_ms += elapsed_ms(logits_start);
  }
  return ok;
}

} // namespace

bool parse_token_list_csv(
  const std::string & csv,
  std::vector<std::int32_t> & out_tokens,
  std::string & error_message) {
  out_tokens.clear();
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    const auto begin = token.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
      continue;
    }
    const auto end = token.find_last_not_of(" \t\r\n");
    const std::string trimmed = token.substr(begin, end - begin + 1);
    try {
      const long long parsed = std::stoll(trimmed);
      if (parsed < static_cast<long long>(std::numeric_limits<std::int32_t>::min()) ||
          parsed > static_cast<long long>(std::numeric_limits<std::int32_t>::max())) {
        error_message = "Token value is out of int32 range: " + trimmed;
        return false;
      }
      out_tokens.push_back(static_cast<std::int32_t>(parsed));
    } catch (...) {
      error_message = "Invalid token value in CSV list: " + trimmed;
      return false;
    }
  }

  if (out_tokens.empty()) {
    error_message = "Prompt token list is empty.";
    return false;
  }

  return true;
}

bool run_reference_qwen35_inference(
  const ModelProfile & profile,
  const ReferenceInferenceOptions & options,
  ReferenceInferenceResult & result,
  std::string & error_message) {
  result = ReferenceInferenceResult{};

  if (profile.family != "qwen3.5") {
    error_message = "Reference inference path currently supports only qwen3.5 family.";
    return false;
  }
  if (options.model_dir.empty()) {
    error_message = "Reference inference requires --hf-model-dir.";
    return false;
  }
  if (options.prompt_tokens.empty()) {
    error_message = "Reference inference requires non-empty prompt token list.";
    return false;
  }
  if (options.max_new_tokens <= 0) {
    error_message = "max_new_tokens must be > 0.";
    return false;
  }
  if (options.sampling.top_p <= 0.0f || options.sampling.top_p > 1.0f) {
    error_message = "top_p must be in (0, 1].";
    return false;
  }
  if (options.sampling.top_k < 0) {
    error_message = "top_k must be >= 0.";
    return false;
  }
  if (options.sampling.temperature < 0.0f) {
    error_message = "temperature must be >= 0.";
    return false;
  }
  if (options.sampling.repetition_penalty < 1.0f) {
    error_message = "repeat_penalty must be >= 1.0.";
    return false;
  }
  if (options.sampling.seed < -1) {
    error_message = "seed must be >= -1.";
    return false;
  }

#if !QWEN35X_HAS_CUDA
  if (options.use_cuda) {
    error_message = "GPU inference requested but this build has CUDA disabled.";
    return false;
  }
#endif

  RuntimeDims dims;
  if (!build_runtime_dims(profile, dims, error_message)) {
    return false;
  }

  if (options.max_context <= 0) {
    error_message = "max_context must be > 0.";
    return false;
  }
  if (static_cast<int>(options.prompt_tokens.size()) + options.max_new_tokens > options.max_context) {
    error_message = "prompt length + max_new_tokens exceeds max_context.";
    return false;
  }

  const auto load_start = std::chrono::steady_clock::now();
  const bool use_cuda_matvec_bf16 = options.use_cuda && options.use_cuda_matvec_bf16;
  const bool use_cuda_kv_cache_bf16 = use_cuda_matvec_bf16;

  ModelWeights weights;
  if (!load_model_weights(options.model_dir, dims, profile, weights, error_message)) {
    return false;
  }
  if (options.use_cuda && !upload_model_weights_to_cuda(weights, use_cuda_matvec_bf16, error_message)) {
    release_model_weights_cuda(weights);
    return false;
  }

  ModelState state;
  int full_layers = 0;
  int linear_layers = 0;
  for (const auto block : profile.fingerprint.attention_schedule) {
    if (block == AttentionBlock::linear) {
      ++linear_layers;
    } else {
      ++full_layers;
    }
  }

  state.full_states.resize(static_cast<std::size_t>(full_layers));
  for (auto & fs : state.full_states) {
    fs.k_cache.resize(
      static_cast<std::size_t>(options.max_context) * static_cast<std::size_t>(dims.n_kv_heads) *
      static_cast<std::size_t>(dims.head_dim));
    fs.v_cache.resize(
      static_cast<std::size_t>(options.max_context) * static_cast<std::size_t>(dims.n_kv_heads) *
      static_cast<std::size_t>(dims.head_dim));
    if (options.use_cuda) {
      if (!cuda::allocate_buffer_f32(fs.k_cache.size(), fs.k_cache_device, error_message) ||
          !cuda::allocate_buffer_f32(fs.v_cache.size(), fs.v_cache_device, error_message)) {
        release_model_state_cuda(state);
        release_model_weights_cuda(weights);
        return false;
      }
      fs.has_device_state = true;
      if (use_cuda_kv_cache_bf16) {
        if (!cuda::allocate_buffer_bf16(fs.k_cache.size(), fs.k_cache_bf16_device, error_message) ||
            !cuda::allocate_buffer_bf16(fs.v_cache.size(), fs.v_cache_bf16_device, error_message)) {
          release_model_state_cuda(state);
          release_model_weights_cuda(weights);
          return false;
        }
        fs.has_device_state_bf16 = true;
      }
    }
  }

  state.linear_states.resize(static_cast<std::size_t>(linear_layers));
  const int conv_hist = dims.linear_kernel - 1;
  for (auto & ls : state.linear_states) {
    ls.conv_state.resize(static_cast<std::size_t>(conv_hist * dims.linear_conv_channels), 0.0f);
    ls.recurrent_state.resize(
      static_cast<std::size_t>(dims.linear_num_v_heads) * static_cast<std::size_t>(dims.linear_head_v_dim) *
      static_cast<std::size_t>(dims.linear_head_v_dim),
      0.0f);
    if (options.use_cuda) {
      if (!cuda::allocate_buffer_f32(ls.conv_state.size(), ls.conv_state_device, error_message) ||
          !cuda::allocate_buffer_f32(ls.recurrent_state.size(), ls.recurrent_state_device, error_message)) {
        release_model_state_cuda(state);
        release_model_weights_cuda(weights);
        return false;
      }
      ls.has_device_state = true;
    }
  }

  CudaForwardWorkspace cuda_forward_workspace;
  CudaForwardWorkspace * cuda_workspace_ptr = nullptr;
  if (options.use_cuda) {
    const std::size_t max_input_count = std::max<std::size_t>({
      static_cast<std::size_t>(dims.hidden),
      static_cast<std::size_t>(dims.intermediate),
      static_cast<std::size_t>(dims.linear_conv_channels),
      static_cast<std::size_t>(dims.linear_v_dim),
      static_cast<std::size_t>(dims.linear_q_dim),
      static_cast<std::size_t>(dims.n_heads * dims.head_dim),
      static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim),
      1u
    });
    const std::size_t max_output_count = std::max<std::size_t>({
      static_cast<std::size_t>(dims.hidden),
      static_cast<std::size_t>(dims.intermediate),
      static_cast<std::size_t>(dims.vocab_size),
      static_cast<std::size_t>(dims.linear_conv_channels),
      static_cast<std::size_t>(dims.linear_v_dim),
      static_cast<std::size_t>(dims.n_heads * dims.head_dim * 2),
      static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim),
      1u
    });
    if (!cuda::begin_inference_session(max_input_count, max_output_count, error_message)) {
      release_model_state_cuda(state);
      release_model_weights_cuda(weights);
      return false;
    }
    cuda::set_prefer_bf16_matvec(use_cuda_matvec_bf16);
    if (!allocate_forward_workspace_cuda(dims, options.max_context, cuda_forward_workspace, error_message)) {
      cuda::end_inference_session();
      release_model_state_cuda(state);
      release_model_weights_cuda(weights);
      return false;
    }
    cuda_workspace_ptr = &cuda_forward_workspace;
  }

  auto release_cuda_resources = [&]() {
    if (options.use_cuda) {
      release_forward_workspace_cuda(cuda_forward_workspace);
      cuda::set_prefer_bf16_matvec(false);
      cuda::end_inference_session();
      release_model_state_cuda(state);
      release_model_weights_cuda(weights);
    }
  };

  const auto load_end = std::chrono::steady_clock::now();
  result.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

  const auto decode_start = std::chrono::steady_clock::now();
  if (options.use_cuda) {
    cuda::reset_transfer_stats();
  }

  std::mt19937 rng;
  if (options.sampling.seed >= 0) {
    rng.seed(static_cast<std::mt19937::result_type>(options.sampling.seed));
  } else {
    std::random_device random_device;
    rng.seed(random_device());
  }

  std::vector<int> token_counts(static_cast<std::size_t>(dims.vocab_size), 0);
  for (const std::int32_t token : options.prompt_tokens) {
    if (token >= 0 && token < dims.vocab_size) {
      token_counts[static_cast<std::size_t>(token)] += 1;
    }
  }
  std::unordered_set<std::int32_t> stop_token_set;
  stop_token_set.reserve(options.stop_token_ids.size());
  for (const std::int32_t token : options.stop_token_ids) {
    if (token >= 0 && token < dims.vocab_size) {
      stop_token_set.insert(token);
    }
  }

  const bool gpu_sampling_supported_by_config =
    options.sampling.temperature >= 0.0f && options.sampling.top_p > 0.0f && options.sampling.top_p <= 1.0f &&
    options.sampling.repetition_penalty > 0.0f &&
    ((options.sampling.temperature <= 0.0f) ||
     (options.sampling.top_k > 0 && options.sampling.top_k <= kCudaSamplingMaxTopK));
  const bool use_cuda_gpu_sampling =
    options.use_cuda && cuda_workspace_ptr != nullptr && cuda_workspace_ptr->has_gpu_sampling_buffers &&
    gpu_sampling_supported_by_config;
  if (options.use_cuda && !use_cuda_gpu_sampling) {
    error_message = "GPU sampling requires temperature >= 0, top_p in (0,1], repetition_penalty > 0, and top_k in [1, 64] when temperature > 0.";
    release_cuda_resources();
    return false;
  }
  if (use_cuda_gpu_sampling) {
    std::vector<float> seen_mask(static_cast<std::size_t>(dims.vocab_size), 0.0f);
    for (int token = 0; token < dims.vocab_size; ++token) {
      if (token_counts[static_cast<std::size_t>(token)] > 0) {
        seen_mask[static_cast<std::size_t>(token)] = 1.0f;
      }
    }
    if (!cuda::upload_to_buffer_f32(
          seen_mask.data(),
          seen_mask.size(),
          cuda_workspace_ptr->seen_token_mask,
          0,
          error_message)) {
      release_cuda_resources();
      return false;
    }
  }

  cuda::CudaDeviceBufferF32 sampled_token_device;
  cuda::CudaDeviceBufferF32 generated_tokens_device;
  auto release_cuda_decode_buffers = [&]() {
    cuda::free_buffer_f32(sampled_token_device);
    cuda::free_buffer_f32(generated_tokens_device);
  };

  DecodeProfilingAccumulator profiling;
  std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

  int position = 0;
  std::vector<float> predicted_logits;
  for (const std::int32_t prompt_token : options.prompt_tokens) {
    if (!run_forward_single_token(
          weights,
          dims,
          state,
          prompt_token,
          position,
          options.use_cuda,
          use_cuda_matvec_bf16,
          options.profile_cuda_sync,
          predicted_logits,
          use_cuda_gpu_sampling,
          cuda_workspace_ptr,
          &profiling,
          error_message)) {
      release_cuda_resources();
      return false;
    }
    ++position;
  }

  result.generated_tokens.clear();
  result.generated_tokens.reserve(static_cast<std::size_t>(options.max_new_tokens));
  if (use_cuda_gpu_sampling) {
    const bool defer_stop_checks = stop_token_set.empty() && options.stop_token_sequences.empty();
    if (defer_stop_checks) {
      if (options.max_new_tokens > 0) {
        if (!cuda::allocate_buffer_f32(1, sampled_token_device, error_message) ||
            !cuda::allocate_buffer_f32(
              static_cast<std::size_t>(options.max_new_tokens),
              generated_tokens_device,
              error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
      }

      for (int i = 0; i < options.max_new_tokens; ++i) {
        if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        const auto sampling_start = std::chrono::steady_clock::now();
        const float random_u01 = options.sampling.temperature > 0.0f ? uniform01(rng) : 0.0f;
        if (!cuda::sample_token_from_logits_f32_device_to_buffer(
              cuda_workspace_ptr->logits,
              cuda_workspace_ptr->seen_token_mask,
              dims.vocab_size,
              options.sampling.temperature,
              options.sampling.top_p,
              options.sampling.top_k,
              options.sampling.repetition_penalty,
              random_u01,
              sampled_token_device,
              error_message) ||
            !cuda::copy_buffer_f32(
              sampled_token_device,
              1,
              0,
              generated_tokens_device,
              static_cast<std::size_t>(i),
              error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        profiling.sampling_ms += elapsed_ms(sampling_start);

        if (i + 1 < options.max_new_tokens) {
          ++profiling.forward_pass_tokens;
          if (!run_forward_single_token_cuda_device_from_token_buffer(
                weights,
                dims,
                state,
                sampled_token_device,
                position,
                use_cuda_matvec_bf16,
                true,
                options.profile_cuda_sync,
                *cuda_workspace_ptr,
                predicted_logits,
                &profiling,
                error_message)) {
            release_cuda_decode_buffers();
            release_cuda_resources();
            return false;
          }
          ++position;
        }
      }

      std::vector<float> generated_token_values;
      if (options.max_new_tokens > 0 &&
          !cuda::download_from_buffer_f32(
            generated_tokens_device,
            static_cast<std::size_t>(options.max_new_tokens),
            0,
            generated_token_values,
            error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }

      for (int i = 0; i < options.max_new_tokens; ++i) {
        const int current = static_cast<int>(generated_token_values[static_cast<std::size_t>(i)]);
        if (current < 0 || current >= dims.vocab_size) {
          error_message = "CUDA sampling produced an out-of-range token id.";
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        result.generated_tokens.push_back(current);
      }
    } else {
      for (int i = 0; i < options.max_new_tokens; ++i) {
        if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        const auto sampling_start = std::chrono::steady_clock::now();
        int current = 0;
        const float random_u01 = options.sampling.temperature > 0.0f ? uniform01(rng) : 0.0f;
        if (!cuda::sample_token_from_logits_f32_device(
              cuda_workspace_ptr->logits,
              cuda_workspace_ptr->seen_token_mask,
              dims.vocab_size,
              options.sampling.temperature,
              options.sampling.top_p,
              options.sampling.top_k,
              options.sampling.repetition_penalty,
              random_u01,
              cuda_workspace_ptr->topk_values_scratch,
              cuda_workspace_ptr->topk_indices_scratch,
              current,
              error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        profiling.sampling_ms += elapsed_ms(sampling_start);

        result.generated_tokens.push_back(current);

        const auto stop_checks_start = std::chrono::steady_clock::now();
        bool should_stop = false;
        std::size_t trim_count = 0;
        if (stop_token_set.find(current) != stop_token_set.end()) {
          should_stop = true;
          trim_count = std::max<std::size_t>(trim_count, 1);
        }
        for (const auto & stop_sequence : options.stop_token_sequences) {
          if (generated_ends_with_sequence(result.generated_tokens, stop_sequence)) {
            should_stop = true;
            trim_count = std::max(trim_count, stop_sequence.size());
          }
        }
        profiling.stop_checks_ms += elapsed_ms(stop_checks_start);
        if (should_stop) {
          if (trim_count > 0 && trim_count <= result.generated_tokens.size()) {
            result.generated_tokens.resize(result.generated_tokens.size() - trim_count);
          }
          break;
        }

        if (!run_forward_single_token(
              weights,
              dims,
              state,
              current,
              position,
              options.use_cuda,
              use_cuda_matvec_bf16,
              options.profile_cuda_sync,
              predicted_logits,
              true,
              cuda_workspace_ptr,
              &profiling,
              error_message)) {
          release_cuda_decode_buffers();
          release_cuda_resources();
          return false;
        }
        ++position;
      }
    }
  } else {
    for (int i = 0; i < options.max_new_tokens; ++i) {
      if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }
      const auto sampling_start = std::chrono::steady_clock::now();
      int current = 0;
      if (!sample_token_from_logits(
            predicted_logits,
            options.sampling,
            token_counts,
            rng,
            current,
            error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }
      if (!maybe_sync_cuda_for_stage_timing(options.use_cuda, options.profile_cuda_sync, error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }
      profiling.sampling_ms += elapsed_ms(sampling_start);

      result.generated_tokens.push_back(current);
      if (current >= 0 && current < dims.vocab_size) {
        token_counts[static_cast<std::size_t>(current)] += 1;
      }

      const auto stop_checks_start = std::chrono::steady_clock::now();
      bool should_stop = false;
      std::size_t trim_count = 0;
      if (stop_token_set.find(current) != stop_token_set.end()) {
        should_stop = true;
        trim_count = std::max<std::size_t>(trim_count, 1);
      }
      for (const auto & stop_sequence : options.stop_token_sequences) {
        if (generated_ends_with_sequence(result.generated_tokens, stop_sequence)) {
          should_stop = true;
          trim_count = std::max(trim_count, stop_sequence.size());
        }
      }
      profiling.stop_checks_ms += elapsed_ms(stop_checks_start);
      if (should_stop) {
        if (trim_count > 0 && trim_count <= result.generated_tokens.size()) {
          result.generated_tokens.resize(result.generated_tokens.size() - trim_count);
        }
        break;
      }

      if (!run_forward_single_token(
            weights,
            dims,
            state,
            current,
            position,
            options.use_cuda,
            use_cuda_matvec_bf16,
            options.profile_cuda_sync,
            predicted_logits,
            false,
            cuda_workspace_ptr,
            &profiling,
            error_message)) {
        release_cuda_decode_buffers();
        release_cuda_resources();
        return false;
      }
      ++position;
    }
  }

  const auto decode_end = std::chrono::steady_clock::now();
  result.decode_time_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
  result.tokens_per_second =
    (result.decode_time_ms > 0.0)
      ? (static_cast<double>(result.generated_tokens.size()) * 1000.0 / result.decode_time_ms)
      : 0.0;
  result.forward_pass_tokens = profiling.forward_pass_tokens;
  result.timing_breakdown.embedding_ms = profiling.embedding_ms;
  result.timing_breakdown.attention_ms = profiling.attention_ms;
  result.timing_breakdown.mlp_ms = profiling.mlp_ms;
  result.timing_breakdown.logits_ms = profiling.logits_ms;
  result.timing_breakdown.sampling_ms = profiling.sampling_ms;
  result.timing_breakdown.stop_checks_ms = profiling.stop_checks_ms;
  if (options.use_cuda) {
    cuda::CudaTransferStats transfer_stats;
    cuda::get_transfer_stats(transfer_stats);
    result.transfer_breakdown.host_to_device_bytes = transfer_stats.host_to_device_bytes;
    result.transfer_breakdown.device_to_host_bytes = transfer_stats.device_to_host_bytes;
    result.transfer_breakdown.other_bytes = transfer_stats.other_bytes;
    result.transfer_breakdown.copy_calls = transfer_stats.copy_calls;
  }
  if (result.forward_pass_tokens > 0) {
    const double forward_tokens = static_cast<double>(result.forward_pass_tokens);
    result.host_to_device_bytes_per_forward_token =
      static_cast<double>(result.transfer_breakdown.host_to_device_bytes) / forward_tokens;
    result.device_to_host_bytes_per_forward_token =
      static_cast<double>(result.transfer_breakdown.device_to_host_bytes) / forward_tokens;
  }

  release_cuda_decode_buffers();
  release_cuda_resources();
  return true;
}

} // namespace qwen35x
