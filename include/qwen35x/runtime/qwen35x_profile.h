#pragma once

namespace qwen35x::cuda_backend {

constexpr int kQwen35xProfileMaxLayers = 32;

struct Qwen35xLayerProfile {
  int layer_index = -1;
  int layer_type = -1; // 0 = DeltaNet, 1 = full attention
  double total_ms = 0.0;
  double rms_norm_ms = 0.0;
  double qkv_projection_ms = 0.0;
  double kv_projection_ms = 0.0;
  double z_projection_ms = 0.0;
  double beta_alpha_projection_ms = 0.0;
  double conv_ms = 0.0;
  double gate_ms = 0.0;
  double recurrence_ms = 0.0;
  double post_norm_gate_ms = 0.0;
  double qk_norm_rope_ms = 0.0;
  double attention_ms = 0.0;
  double attention_qk_ms = 0.0;
  double attention_softmax_ms = 0.0;
  double attention_pv_ms = 0.0;
  double attention_gate_ms = 0.0;
  double out_projection_ms = 0.0;
  double residual_ms = 0.0;
  double mlp_norm_ms = 0.0;
  double mlp_projection_ms = 0.0;
  double mlp_activation_ms = 0.0;
  double mlp_down_projection_ms = 0.0;
  double mlp_residual_ms = 0.0;
};

struct Qwen35xPrefillProfile {
  bool enabled = false;
  int seq_len = 0;
  bool compute_logits = false;
  int layer_count = 0;
  double host_total_ms = 0.0;
  double gpu_total_ms = 0.0;
  double token_upload_ms = 0.0;
  double embed_ms = 0.0;
  double mark_seen_ms = 0.0;
  double final_norm_ms = 0.0;
  double lm_head_ms = 0.0;
  double lm_reduce_ms = 0.0;
  double hidden_handoff_ms = 0.0;
  double output_token_download_ms = 0.0;
  Qwen35xLayerProfile layers[kQwen35xProfileMaxLayers] = {};
};

struct Qwen35xDecodeProfile {
  bool enabled = false;
  int steps = 0;
  int last_position = -1;
  int decode_blocks = 0;
  int max_safe_decode_blocks = 0;
  double host_total_ms = 0.0;
  double seen_token_upload_ms = 0.0;
  double launch_total_ms = 0.0;
  double decode_kernel_ms = 0.0;
  double lm_head_ms = 0.0;
  double output_token_download_ms = 0.0;
};

struct Qwen35xRuntimeProfile {
  bool enabled = false;
  int prefill_runs = 0;
  Qwen35xPrefillProfile prefill;
  Qwen35xDecodeProfile decode;
};

} // namespace qwen35x::cuda_backend
