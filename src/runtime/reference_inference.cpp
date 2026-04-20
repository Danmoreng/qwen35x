#include "qwen35x/runtime/reference_inference.h"

#include "qwen35x/runtime/cuda_inference.h"
#include "qwen35x/weights/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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
};

struct LayerWeights {
  TensorData input_layernorm;
  TensorData post_attention_layernorm;
  TensorData mlp_gate;
  TensorData mlp_up;
  TensorData mlp_down;
  bool is_linear = false;
  FullAttentionWeights full;
  LinearAttentionWeights linear;
};

struct ModelWeights {
  TensorData embed_tokens;
  TensorData final_norm;
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
  bool has_device_state = false;
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

bool upload_tensor_2d_to_cuda(TensorData & tensor, const std::string & name, std::string & error_message) {
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
  tensor.has_device_matrix = true;
  return true;
}

void release_tensor_cuda(TensorData & tensor) {
  if (tensor.has_device_matrix || tensor.device_matrix.data != nullptr) {
    cuda::free_matrix_f32(tensor.device_matrix);
    tensor.has_device_matrix = false;
  }
}

bool upload_model_weights_to_cuda(ModelWeights & weights, std::string & error_message) {
  if (!upload_tensor_2d_to_cuda(weights.embed_tokens, "embed_tokens", error_message)) {
    return false;
  }

  for (std::size_t il = 0; il < weights.layers.size(); ++il) {
    LayerWeights & layer = weights.layers[il];
    const std::string prefix = "layer " + std::to_string(il) + " ";

    if (!upload_tensor_2d_to_cuda(layer.mlp_gate, prefix + "mlp_gate", error_message) ||
        !upload_tensor_2d_to_cuda(layer.mlp_up, prefix + "mlp_up", error_message) ||
        !upload_tensor_2d_to_cuda(layer.mlp_down, prefix + "mlp_down", error_message)) {
      return false;
    }

    if (layer.is_linear) {
      if (!upload_tensor_2d_to_cuda(layer.linear.in_proj_qkv, prefix + "linear.in_proj_qkv", error_message) ||
          !upload_tensor_2d_to_cuda(layer.linear.in_proj_z, prefix + "linear.in_proj_z", error_message) ||
          !upload_tensor_2d_to_cuda(layer.linear.in_proj_b, prefix + "linear.in_proj_b", error_message) ||
          !upload_tensor_2d_to_cuda(layer.linear.in_proj_a, prefix + "linear.in_proj_a", error_message) ||
          !upload_tensor_2d_to_cuda(layer.linear.conv1d, prefix + "linear.conv1d", error_message) ||
          !upload_tensor_2d_to_cuda(layer.linear.out_proj, prefix + "linear.out_proj", error_message)) {
        return false;
      }
    } else {
      if (!upload_tensor_2d_to_cuda(layer.full.q_proj, prefix + "full.q_proj", error_message) ||
          !upload_tensor_2d_to_cuda(layer.full.k_proj, prefix + "full.k_proj", error_message) ||
          !upload_tensor_2d_to_cuda(layer.full.v_proj, prefix + "full.v_proj", error_message) ||
          !upload_tensor_2d_to_cuda(layer.full.o_proj, prefix + "full.o_proj", error_message)) {
        return false;
      }
    }
  }

  return true;
}

void release_model_weights_cuda(ModelWeights & weights) {
  release_tensor_cuda(weights.embed_tokens);
  for (auto & layer : weights.layers) {
    release_tensor_cuda(layer.mlp_gate);
    release_tensor_cuda(layer.mlp_up);
    release_tensor_cuda(layer.mlp_down);
    release_tensor_cuda(layer.full.q_proj);
    release_tensor_cuda(layer.full.k_proj);
    release_tensor_cuda(layer.full.v_proj);
    release_tensor_cuda(layer.full.o_proj);
    release_tensor_cuda(layer.linear.in_proj_qkv);
    release_tensor_cuda(layer.linear.in_proj_z);
    release_tensor_cuda(layer.linear.in_proj_b);
    release_tensor_cuda(layer.linear.in_proj_a);
    release_tensor_cuda(layer.linear.conv1d);
    release_tensor_cuda(layer.linear.out_proj);
  }
}

void release_model_state_cuda(ModelState & state) {
  for (auto & fs : state.full_states) {
    cuda::free_buffer_f32(fs.k_cache_device);
    cuda::free_buffer_f32(fs.v_cache_device);
    fs.has_device_state = false;
  }
  for (auto & ls : state.linear_states) {
    cuda::free_buffer_f32(ls.conv_state_device);
    cuda::free_buffer_f32(ls.recurrent_state_device);
    ls.has_device_state = false;
  }
}

bool sync_full_state_token_to_cuda(
  const FullAttentionState & state,
  const RuntimeDims & dims,
  const int position,
  std::string & error_message) {
  if (!state.has_device_state) {
    return true;
  }
  const std::size_t token_stride = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  const std::size_t offset = static_cast<std::size_t>(position) * token_stride;
  if (!cuda::upload_to_buffer_f32(
        state.k_cache.data() + offset,
        token_stride,
        state.k_cache_device,
        offset,
        error_message)) {
    return false;
  }
  if (!cuda::upload_to_buffer_f32(
        state.v_cache.data() + offset,
        token_stride,
        state.v_cache_device,
        offset,
        error_message)) {
    return false;
  }
  return true;
}

bool sync_linear_state_to_cuda(const LinearAttentionState & state, std::string & error_message) {
  if (!state.has_device_state) {
    return true;
  }
  if (!cuda::upload_to_buffer_f32(
        state.conv_state.data(),
        state.conv_state.size(),
        state.conv_state_device,
        0,
        error_message)) {
    return false;
  }
  if (!cuda::upload_to_buffer_f32(
        state.recurrent_state.data(),
        state.recurrent_state.size(),
        state.recurrent_state_device,
        0,
        error_message)) {
    return false;
  }
  return true;
}

bool run_linear_attention_step(
  const LayerWeights & layer,
  const RuntimeDims & dims,
  LinearAttentionState & state,
  const std::vector<float> & x,
  std::vector<float> & out,
  const bool use_cuda,
  std::string & error_message) {
  std::vector<float> mixed_qkv;
  std::vector<float> z_vec;
  std::vector<float> b_vec;
  std::vector<float> a_vec;
  if (!matvec_2d(layer.linear.in_proj_qkv, x, mixed_qkv, use_cuda, error_message) ||
      !matvec_2d(layer.linear.in_proj_z, x, z_vec, use_cuda, error_message) ||
      !matvec_2d(layer.linear.in_proj_b, x, b_vec, use_cuda, error_message) ||
      !matvec_2d(layer.linear.in_proj_a, x, a_vec, use_cuda, error_message)) {
    return false;
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
  if (use_cuda && !sync_linear_state_to_cuda(state, error_message)) {
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
  std::string & error_message) {
  std::vector<float> q_full;
  std::vector<float> k_flat;
  std::vector<float> v_flat;
  if (!matvec_2d(layer.full.q_proj, x, q_full, use_cuda, error_message) ||
      !matvec_2d(layer.full.k_proj, x, k_flat, use_cuda, error_message) ||
      !matvec_2d(layer.full.v_proj, x, v_flat, use_cuda, error_message)) {
    return false;
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
  if (use_cuda && !sync_full_state_token_to_cuda(state, dims, position, error_message)) {
    return false;
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

bool run_forward_single_token(
  const ModelWeights & weights,
  const RuntimeDims & dims,
  ModelState & state,
  const int token_id,
  const int position,
  const bool use_cuda,
  std::vector<float> & next_logits,
  DecodeProfilingAccumulator * profiling,
  std::string & error_message) {
  if (token_id < 0 || token_id >= dims.vocab_size) {
    error_message = "Token id out of vocabulary range.";
    return false;
  }

  if (profiling != nullptr) {
    ++profiling->forward_pass_tokens;
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
  const bool ok =
    compute_next_logits_from_embedding(weights.embed_tokens, final_hidden, use_cuda, next_logits, error_message);
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

  ModelWeights weights;
  if (!load_model_weights(options.model_dir, dims, profile, weights, error_message)) {
    return false;
  }
  if (options.use_cuda && !upload_model_weights_to_cuda(weights, error_message)) {
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
      if (!sync_linear_state_to_cuda(ls, error_message)) {
        release_model_state_cuda(state);
        release_model_weights_cuda(weights);
        return false;
      }
    }
  }

  auto release_cuda_resources = [&]() {
    if (options.use_cuda) {
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

  DecodeProfilingAccumulator profiling;

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
          predicted_logits,
          &profiling,
          error_message)) {
      release_cuda_resources();
      return false;
    }
    ++position;
  }

  result.generated_tokens.clear();
  result.generated_tokens.reserve(static_cast<std::size_t>(options.max_new_tokens));
  for (int i = 0; i < options.max_new_tokens; ++i) {
    const auto sampling_start = std::chrono::steady_clock::now();
    int current = 0;
    if (!sample_token_from_logits(
          predicted_logits,
          options.sampling,
          token_counts,
          rng,
          current,
          error_message)) {
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
          predicted_logits,
          &profiling,
          error_message)) {
      release_cuda_resources();
      return false;
    }
    ++position;
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

  release_cuda_resources();
  return true;
}

} // namespace qwen35x
