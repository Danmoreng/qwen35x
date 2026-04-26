#pragma once

#include <cuda_bf16.h>
#include <cstdint>

struct Nvfp4Weight {
    const std::uint8_t *packed_weight;   // [out_dim, in_dim / 2]
    const std::uint8_t *weight_scale;    // [out_dim, in_dim / 16], E4M3 bytes
    const float *weight_scale_2;         // scalar f32
    const std::uint8_t *tc_packed_weight;
    const std::uint8_t *tc_weight_scale;
    const float *tc_alpha;
    const std::uint32_t *sm120_packed_weight_fragments;
    const std::uint32_t *sm120_weight_scale_fragments;
    int output_size;
    int input_size;
    int padded_output_size;
    int padded_scale_cols;
    int weight_padding_cols;
    int sm120_row_tiles;
    int sm120_k_blocks;
};

struct LayerNvfp4Weights {
    int layer_type;
    int _pad[3];
    Nvfp4Weight ptrs[14];
};

// Decode path weight structs - ALL BF16.
struct FullAttnWeights {
    const __nv_bfloat16 *input_layernorm_weight;   // [1024]
    const __nv_bfloat16 *q_proj_weight;            // [4096, 1024]
    const __nv_bfloat16 *k_proj_weight;            // [512, 1024]
    const __nv_bfloat16 *v_proj_weight;            // [512, 1024]
    const __nv_bfloat16 *q_norm_weight;            // [256]
    const __nv_bfloat16 *k_norm_weight;            // [256]
    const __nv_bfloat16 *o_proj_weight;            // [1024, 2048]
    const __nv_bfloat16 *post_attn_layernorm_weight;
    const __nv_bfloat16 *gate_proj_weight;         // [3584, 1024]
    const __nv_bfloat16 *up_proj_weight;           // [3584, 1024]
    const __nv_bfloat16 *down_proj_weight;         // [1024, 3584]
};

struct DeltaNetWeights {
    const __nv_bfloat16 *input_layernorm_weight;
    const __nv_bfloat16 *qkv_proj_weight;          // [6144, 1024]
    const __nv_bfloat16 *z_proj_weight;            // [2048, 1024]
    const __nv_bfloat16 *beta_proj_weight;         // [16, 1024]
    const __nv_bfloat16 *alpha_proj_weight;        // [16, 1024]
    const __nv_bfloat16 *conv1d_weight;            // [6144, 1, 4]
    const __nv_bfloat16 *a_log;                    // [16]
    const __nv_bfloat16 *dt_bias;                  // [16]
    const __nv_bfloat16 *norm_weight;              // [128]
    const __nv_bfloat16 *out_proj_weight;          // [1024, 2048]
    const __nv_bfloat16 *post_attn_layernorm_weight;
    const __nv_bfloat16 *gate_proj_weight;
    const __nv_bfloat16 *up_proj_weight;
    const __nv_bfloat16 *down_proj_weight;
};

struct LayerWeights {
    int layer_type;
    int _pad[3];
    union {
        DeltaNetWeights dn;
        FullAttnWeights fa;
    };
};

// Prefill path uses a compact pointer table because the host loader already
// resolves per-layer tensor layout.
struct PFLayerWeights {
    int layer_type;
    int _pad[3];
    void *ptrs[14];
};
