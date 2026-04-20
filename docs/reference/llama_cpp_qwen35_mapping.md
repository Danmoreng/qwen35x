# Qwen3.5 Handling In llama.cpp (Implementation Mapping)

This note maps `llama.cpp`'s `qwen35` implementation to the HF safetensors we downloaded in this repo.

## Primary references

- `third_party/reference/llama.cpp/src/models/qwen35.cpp`
- `third_party/reference/llama.cpp/src/llama-memory-hybrid.h`
- `third_party/reference/llama.cpp/src/llama-memory-hybrid.cpp`
- `models/qwen3.5-0.8b/config.json`
- `models/qwen3.5-0.8b/model.safetensors.index.json`
- `docs/reference/modular_qwen3_5.py`

## Layer schedule (0.8B text model)

`models/qwen3.5-0.8b/config.json` has 24 layers with fixed pattern:

- 3x `linear_attention`
- 1x `full_attention`
- repeated across all 24 layers (`full_attention_interval = 4`)

This matches the hybrid split in `llama.cpp` where each layer dispatches to:

- `build_layer_attn_linear(...)` for recurrent/linear layers
- `build_layer_attn(...)` for full attention layers

## Top-level forward structure in llama.cpp

From `qwen35.cpp`:

1. Embed tokens
2. For each layer:
 - RMS input norm (`attn_norm`)
 - Run linear or full attention block based on layer type
 - Residual add
 - RMS post-attention norm (`attn_post_norm`)
 - Dense FFN
 - FFN residual add
3. Final RMS norm
4. LM head projection

Important: `llama.cpp` uses hybrid memory (`build_inp_mem_hybrid`) that combines:

- normal attention KV cache for full-attention layers
- recurrent + conv state cache for linear-attention layers

## Tensor mapping: HF safetensors -> llama.cpp layer fields

### Shared

- `tok_embd` -> `model.language_model.embed_tokens.weight`
- `attn_norm` -> `model.language_model.layers.{i}.input_layernorm.weight`
- `attn_post_norm` -> `model.language_model.layers.{i}.post_attention_layernorm.weight`
- FFN:
 - `ffn_gate` -> `...layers.{i}.mlp.gate_proj.weight`
 - `ffn_up` -> `...layers.{i}.mlp.up_proj.weight`
 - `ffn_down` -> `...layers.{i}.mlp.down_proj.weight`
- `output_norm` -> `model.language_model.norm.weight`
- `output` / LM head:
 - tied to embeddings for this config (`tie_word_embeddings = true`)

### Full-attention layers (`layer_types[i] == "full_attention"`)

- `wq` -> `...layers.{i}.self_attn.q_proj.weight`
- `wk` -> `...layers.{i}.self_attn.k_proj.weight`
- `wv` -> `...layers.{i}.self_attn.v_proj.weight`
- `wo` -> `...layers.{i}.self_attn.o_proj.weight`
- `attn_q_norm` -> `...layers.{i}.self_attn.q_norm.weight`
- `attn_k_norm` -> `...layers.{i}.self_attn.k_norm.weight`

### Linear-attention layers (`layer_types[i] == "linear_attention"`)

- `wqkv` -> `...layers.{i}.linear_attn.in_proj_qkv.weight`
- `wqkv_gate` (z-proj) -> `...layers.{i}.linear_attn.in_proj_z.weight`
- `ssm_beta` -> `...layers.{i}.linear_attn.in_proj_b.weight`
- `ssm_alpha` -> `...layers.{i}.linear_attn.in_proj_a.weight`
- `ssm_conv1d` -> `...layers.{i}.linear_attn.conv1d.weight`
- `ssm_norm` -> `...layers.{i}.linear_attn.norm.weight`
- `ssm_out` -> `...layers.{i}.linear_attn.out_proj.weight`
- `ssm_dt` -> `...layers.{i}.linear_attn.dt_bias`
- `ssm_a`:
 - HF stores `A_log` at `...layers.{i}.linear_attn.A_log`
 - llama path uses `ssm_a` directly in `gate = softplus(alpha + dt) * ssm_a`
 - expected equivalent is `ssm_a = -exp(A_log)` during load/pack

## Cache/state implications for our runtime

To match llama behavior we need two cache families:

- Full-attention KV cache (for layers 3, 7, 11, 15, 19, 23 on 0.8B)
- Linear-attention state cache:
 - conv state: last `conv_kernel_size - 1` steps for `(Q, K, V mixed)` channels
 - recurrent state: per layer and sequence, shape equivalent to
   `num_v_heads x head_v_dim x head_v_dim`

## Practical implementation order

1. Parse `layer_types` and build per-layer descriptors from HF config.
2. Build weight-loader mapping exactly as above (including `A_log -> ssm_a` transform).
3. Implement full-attention decode path (single token) with GQA and RMS norms.
4. Implement linear-attention decode path:
 - conv-state update
 - gated delta recurrent update
 - gated RMS norm and output projection
5. Integrate both into one token-generation loop with shared residual/FFN path.

