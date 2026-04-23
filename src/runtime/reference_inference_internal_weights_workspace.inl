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

struct CudaForwardWorkspace {
  cuda::CudaDeviceBufferF32 hidden_in;
  cuda::CudaDeviceBufferF32 intermediate_a;
  cuda::CudaDeviceBufferF32 intermediate_b;
  cuda::CudaDeviceBufferF32 intermediate_hidden;
  cuda::CudaDeviceBufferF32 hidden_out;
  cuda::CudaDeviceBufferF32 projection_out;
  cuda::CudaDeviceBufferF32 full_q;
  cuda::CudaDeviceBufferF32 full_gate;
  cuda::CudaDeviceBufferF32 full_attn;
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
    fs.has_device_state = false;
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
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.intermediate), workspace.intermediate_a, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.intermediate), workspace.intermediate_b, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.intermediate), workspace.intermediate_hidden, error_message) ||
      !cuda::allocate_buffer_f32(static_cast<std::size_t>(dims.hidden), workspace.hidden_out, error_message) ||
      !cuda::allocate_buffer_f32(projection_out_count, workspace.projection_out, error_message) ||
      !cuda::allocate_buffer_f32(full_q_count, workspace.full_q, error_message) ||
      !cuda::allocate_buffer_f32(full_q_count, workspace.full_gate, error_message) ||
      !cuda::allocate_buffer_f32(full_q_count, workspace.full_attn, error_message) ||
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
    cuda::free_buffer_f32(workspace.intermediate_a);
    cuda::free_buffer_f32(workspace.intermediate_b);
    cuda::free_buffer_f32(workspace.intermediate_hidden);
    cuda::free_buffer_f32(workspace.hidden_out);
    cuda::free_buffer_f32(workspace.projection_out);
    cuda::free_buffer_f32(workspace.full_q);
    cuda::free_buffer_f32(workspace.full_gate);
    cuda::free_buffer_f32(workspace.full_attn);
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
  cuda::free_buffer_f32(workspace.intermediate_a);
  cuda::free_buffer_f32(workspace.intermediate_b);
  cuda::free_buffer_f32(workspace.intermediate_hidden);
  cuda::free_buffer_f32(workspace.hidden_out);
  cuda::free_buffer_f32(workspace.projection_out);
  cuda::free_buffer_f32(workspace.full_q);
  cuda::free_buffer_f32(workspace.full_gate);
  cuda::free_buffer_f32(workspace.full_attn);
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

