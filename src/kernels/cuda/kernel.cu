/**
 * Derived from Lucebox megakernel sources, MIT licensed.
 * See LICENSE.Lucebox in this directory.
 *
 * Fused single-kernel decode for Qwen3.5-0.8B (hybrid DeltaNet + Full Attention).
 * ALL BF16: weights bf16, activations bf16, accumulation f32.
 * DeltaNet state: f32 (recurrence needs precision).
 *
 * Optimized for: NVIDIA RTX 3090 (sm_86, 82 SMs)
 * Model:         Qwen/Qwen3.5-0.8B (bf16 weights)
 */

#include "qwen35x/runtime/qwen35x_profile.h"

#include "common.cuh"
#include "decode_sync.cuh"
#include "variant.cuh"
#include "weights.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <type_traits>

// =============================================================================
// Decode variant state
// =============================================================================

static int g_decode_blocks_override = 0;

__device__ __constant__ int LAYER_TYPE[NUM_LAYERS] = {
    QWEN35X_LAYER_TYPE_VALUES
};

#include "decode_layers.cuh"
#include "decode_lm_head.cuh"
#include "decode_kernel.cuh"

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_prefix_mlp_kernel(
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LayerWeights *__restrict__ layer_weights,
    const LayerNvfp4Weights *__restrict__ layer_nvfp4_weights,
    __nv_bfloat16 *__restrict__ fa_k_cache,
    __nv_bfloat16 *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_attn_partials,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    int input_token_id,
    int layer,
    int position,
    int max_seq_len,
    int external_mlp)
{
    int dn_layer_idx = 0;
    int fa_layer_idx = 0;
    for (int i = 0; i < layer; ++i) {
        if (LAYER_TYPE[i] == 0) {
            ++dn_layer_idx;
        } else {
            ++fa_layer_idx;
        }
    }

    int num_blocks = gridDim.x;
    AtomicGridSync grid{barrier_counter, barrier_generation, (unsigned int)num_blocks, 0};
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);

    const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    const __nv_bfloat16 *layer_input = (layer == 0) ? embed_row : hidden_buffer;
    int fa_kv_stride = FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    int dn_state_stride = DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;

    if (LAYER_TYPE[layer] == 0) {
        deltanet_layer(
            grid, layer_weights[layer].dn,
            layer_nvfp4_weights == nullptr ? nullptr : &layer_nvfp4_weights[layer],
            layer_input,
            g_residual, g_activations, g_qkv_scratch, g_z_scratch,
            g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
            dn_states + dn_layer_idx * dn_state_stride,
            conv_bufs, hidden_buffer, dn_layer_idx, shmem_bf16, external_mlp, false);
    } else {
        full_attention_layer(
            grid, layer_weights[layer].fa,
            layer_nvfp4_weights == nullptr ? nullptr : &layer_nvfp4_weights[layer],
            layer_input,
            fa_k_cache + fa_layer_idx * fa_kv_stride,
            fa_v_cache + fa_layer_idx * fa_kv_stride,
            g_residual, g_activations, g_qkv_scratch, g_kv_scratch,
            g_attn_out, g_attn_partials, g_mlp_inter, hidden_buffer,
            position, max_seq_len, shmem_bf16, external_mlp, false);
    }
}

// CUDA Graph decode keeps per-step values in device memory so the captured
// kernel arguments remain stable while token id and position change.
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_graph_layer_kernel(
    const int *__restrict__ decode_control,
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LayerWeights *__restrict__ layer_weights,
    const LayerNvfp4Weights *__restrict__ layer_nvfp4_weights,
    __nv_bfloat16 *__restrict__ fa_k_cache,
    __nv_bfloat16 *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_attn_partials,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    int layer,
    int max_seq_len)
{
    const int input_token_id = decode_control[0];
    const int position = decode_control[1];
    int dn_layer_idx = 0;
    int fa_layer_idx = 0;
    for (int i = 0; i < layer; ++i) {
        if (LAYER_TYPE[i] == 0) {
            ++dn_layer_idx;
        } else {
            ++fa_layer_idx;
        }
    }

    int num_blocks = gridDim.x;
    AtomicGridSync grid{barrier_counter, barrier_generation, (unsigned int)num_blocks, 0};
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);

    const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    const __nv_bfloat16 *layer_input = (layer == 0) ? embed_row : hidden_buffer;
    int fa_kv_stride = FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    int dn_state_stride = DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;

    if (LAYER_TYPE[layer] == 0) {
        deltanet_layer(
            grid, layer_weights[layer].dn,
            layer_nvfp4_weights == nullptr ? nullptr : &layer_nvfp4_weights[layer],
            layer_input,
            g_residual, g_activations, g_qkv_scratch, g_z_scratch,
            g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
            dn_states + dn_layer_idx * dn_state_stride,
            conv_bufs, hidden_buffer, dn_layer_idx, shmem_bf16, 0, false);
    } else {
        full_attention_layer(
            grid, layer_weights[layer].fa,
            layer_nvfp4_weights == nullptr ? nullptr : &layer_nvfp4_weights[layer],
            layer_input,
            fa_k_cache + fa_layer_idx * fa_kv_stride,
            fa_v_cache + fa_layer_idx * fa_kv_stride,
            g_residual, g_activations, g_qkv_scratch, g_kv_scratch,
            g_attn_out, g_attn_partials, g_mlp_inter, hidden_buffer,
            position, max_seq_len, shmem_bf16, 0, false);
    }
}

// Graph decode has a fixed layer schedule. Keep DeltaNet and full-attention
// layers in separate kernels so every thread executes only the relevant path.
// The host passes direct state and cache slots while capturing the graph.
template <bool INPUT_IS_EMBEDDING>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_graph_deltanet_kernel(
    const int *__restrict__ decode_control,
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LayerWeights *__restrict__ layer_weights,
    const LayerNvfp4Weights *__restrict__ layer_nvfp4_weights,
    float *__restrict__ dn_state,
    float *__restrict__ conv_buf,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation)
{
    const int input_token_id = decode_control[0];
    const __nv_bfloat16 *layer_input = hidden_buffer;
    if constexpr (INPUT_IS_EMBEDDING) {
        layer_input = embed_weight + input_token_id * HIDDEN_SIZE;
    }
    AtomicGridSync grid{barrier_counter, barrier_generation, static_cast<unsigned int>(gridDim.x), 0};
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);
    deltanet_layer(
        grid,
        layer_weights->dn,
        layer_nvfp4_weights,
        layer_input,
        g_residual,
        g_activations,
        g_qkv_scratch,
        g_z_scratch,
        g_beta_scratch,
        g_alpha_scratch,
        g_attn_out,
        g_mlp_inter,
        dn_state,
        conv_buf,
        hidden_buffer,
        0,
        shmem_bf16,
        0,
        false);
}

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_graph_full_attention_kernel(
    const int *__restrict__ decode_control,
    const LayerWeights *__restrict__ layer_weights,
    const LayerNvfp4Weights *__restrict__ layer_nvfp4_weights,
    __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_attn_partials,
    float *__restrict__ g_mlp_inter,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    int max_seq_len)
{
    const int position = decode_control[1];
    AtomicGridSync grid{barrier_counter, barrier_generation, static_cast<unsigned int>(gridDim.x), 0};
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);
    full_attention_layer(
        grid,
        layer_weights->fa,
        layer_nvfp4_weights,
        hidden_buffer,
        k_cache,
        v_cache,
        g_residual,
        g_activations,
        g_qkv_scratch,
        g_kv_scratch,
        g_attn_out,
        g_attn_partials,
        g_mlp_inter,
        hidden_buffer,
        position,
        max_seq_len,
        shmem_bf16,
        0,
        false);
}

template <bool INPUT_IS_EMBEDDING>
static __device__ void decode_graph_group4_deltanet_layer(
    AtomicGridSync &grid,
    const int *__restrict__ decode_control,
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LayerWeights *__restrict__ layer_weights,
    const LayerNvfp4Weights *__restrict__ layer_nvfp4_weights,
    int layer,
    int dn_layer_idx,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    __nv_bfloat16 *__restrict__ shmem_bf16)
{
    const __nv_bfloat16 *layer_input = hidden_buffer;
    if constexpr (INPUT_IS_EMBEDDING) {
        layer_input = embed_weight + decode_control[0] * HIDDEN_SIZE;
    }
    const int dn_state_stride = DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;
    deltanet_layer(
        grid,
        layer_weights[layer].dn,
        layer_nvfp4_weights == nullptr ? nullptr : &layer_nvfp4_weights[layer],
        layer_input,
        g_residual,
        g_activations,
        g_qkv_scratch,
        g_z_scratch,
        g_beta_scratch,
        g_alpha_scratch,
        g_attn_out,
        g_mlp_inter,
        dn_states + dn_layer_idx * dn_state_stride,
        conv_bufs,
        hidden_buffer,
        dn_layer_idx,
        shmem_bf16,
        0,
        false);
}

static __device__ void decode_graph_group4_full_attention_layer(
    AtomicGridSync &grid,
    const int *__restrict__ decode_control,
    const LayerWeights *__restrict__ layer_weights,
    const LayerNvfp4Weights *__restrict__ layer_nvfp4_weights,
    int layer,
    int fa_layer_idx,
    __nv_bfloat16 *__restrict__ fa_k_cache,
    __nv_bfloat16 *__restrict__ fa_v_cache,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_attn_partials,
    float *__restrict__ g_mlp_inter,
    int max_seq_len,
    __nv_bfloat16 *__restrict__ shmem_bf16)
{
    const int fa_kv_stride = FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    full_attention_layer(
        grid,
        layer_weights[layer].fa,
        layer_nvfp4_weights == nullptr ? nullptr : &layer_nvfp4_weights[layer],
        hidden_buffer,
        fa_k_cache + fa_layer_idx * fa_kv_stride,
        fa_v_cache + fa_layer_idx * fa_kv_stride,
        g_residual,
        g_activations,
        g_qkv_scratch,
        g_kv_scratch,
        g_attn_out,
        g_attn_partials,
        g_mlp_inter,
        hidden_buffer,
        decode_control[1],
        max_seq_len,
        shmem_bf16,
        0,
        false);
}

static __device__ void decode_graph_group4_final_norm(
    const __nv_bfloat16 *__restrict__ hidden_buffer,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    float *__restrict__ g_activations,
    float *__restrict__ g_normalized)
{
    __shared__ float smem_reduce[NUM_WARPS];
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        const float value = __bfloat162float(hidden_buffer[i]);
        g_activations[i] = value;
        local_sum_sq += value * value;
    }
    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
        smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float sum = lane_id < NUM_WARPS ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / HIDDEN_SIZE + RMS_EPS);
        }
    }
    __syncthreads();
    const float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        const float weight = __bfloat162float(__ldg(final_norm_weight + i));
        g_normalized[i] = g_activations[i] * rstd * (1.0f + weight);
    }
}

template <int GROUP_START>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_graph_group4_kernel(
    const int *__restrict__ decode_control,
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LayerWeights *__restrict__ layer_weights,
    const LayerNvfp4Weights *__restrict__ layer_nvfp4_weights,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    __nv_bfloat16 *__restrict__ fa_k_cache,
    __nv_bfloat16 *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_attn_partials,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ lm_sync_counter,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    int max_seq_len)
{
    constexpr int DN_GROUP_START = (GROUP_START / 4) * 3;
    constexpr int FA_GROUP_INDEX = GROUP_START / 4;
    // Each completed group leaves its counter at zero. Reuse its persistent
    // generation in the next graph replay instead of resetting every slot.
    const unsigned int initial_generation = *barrier_generation;
    AtomicGridSync grid{barrier_counter, barrier_generation, static_cast<unsigned int>(gridDim.x), initial_generation};
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);
    if constexpr (GROUP_START == 0) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *lm_sync_counter = 0;
        }
        decode_graph_group4_deltanet_layer<true>(
            grid, decode_control, embed_weight, layer_weights, layer_nvfp4_weights,
            GROUP_START, DN_GROUP_START, dn_states, conv_bufs, hidden_buffer,
            g_activations, g_residual, g_qkv_scratch, g_attn_out, g_mlp_inter,
            g_z_scratch, g_beta_scratch, g_alpha_scratch, shmem_bf16);
    } else {
        decode_graph_group4_deltanet_layer<false>(
            grid, decode_control, embed_weight, layer_weights, layer_nvfp4_weights,
            GROUP_START, DN_GROUP_START, dn_states, conv_bufs, hidden_buffer,
            g_activations, g_residual, g_qkv_scratch, g_attn_out, g_mlp_inter,
            g_z_scratch, g_beta_scratch, g_alpha_scratch, shmem_bf16);
    }
    decode_graph_group4_deltanet_layer<false>(
        grid, decode_control, embed_weight, layer_weights, layer_nvfp4_weights,
        GROUP_START + 1, DN_GROUP_START + 1, dn_states, conv_bufs, hidden_buffer,
        g_activations, g_residual, g_qkv_scratch, g_attn_out, g_mlp_inter,
        g_z_scratch, g_beta_scratch, g_alpha_scratch, shmem_bf16);
    decode_graph_group4_deltanet_layer<false>(
        grid, decode_control, embed_weight, layer_weights, layer_nvfp4_weights,
        GROUP_START + 2, DN_GROUP_START + 2, dn_states, conv_bufs, hidden_buffer,
        g_activations, g_residual, g_qkv_scratch, g_attn_out, g_mlp_inter,
        g_z_scratch, g_beta_scratch, g_alpha_scratch, shmem_bf16);
    decode_graph_group4_full_attention_layer(
        grid, decode_control, layer_weights, layer_nvfp4_weights,
        GROUP_START + 3, FA_GROUP_INDEX, fa_k_cache, fa_v_cache, hidden_buffer,
        g_activations, g_residual, g_qkv_scratch, g_kv_scratch, g_attn_out,
        g_attn_partials, g_mlp_inter, max_seq_len, shmem_bf16);
    if constexpr (GROUP_START == NUM_LAYERS - 4) {
        if (blockIdx.x == 0) {
            decode_graph_group4_final_norm(
                hidden_buffer, final_norm_weight, g_activations, g_normalized);
        }
    }
}

// The other end of the graph grouping sweep: one specialized monolithic
// decode node followed by the LM-head node. It retains device-resident control
// values and persistent barrier generations, but removes inter-group launches.
template <bool USE_NVFP4>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_graph_megakernel(
    const int *__restrict__ decode_control,
    const __nv_bfloat16 *__restrict__ embed_weight,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const LayerWeights *__restrict__ layer_weights,
    const LayerNvfp4Weights *__restrict__ layer_nvfp4_weights,
    __nv_bfloat16 *__restrict__ fa_k_cache,
    __nv_bfloat16 *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_attn_partials,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    unsigned int *__restrict__ lm_sync_counter,
    float *__restrict__ seen_token_mask,
    float repetition_penalty,
    int max_seq_len)
{
    const unsigned int initial_generation = *barrier_generation;
    AtomicGridSync grid{barrier_counter, barrier_generation, static_cast<unsigned int>(gridDim.x), initial_generation};
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *lm_sync_counter = 0;
        if (repetition_penalty > 1.0f) {
            seen_token_mask[decode_control[0]] = 1.0f;
        }
    }

    const __nv_bfloat16 *embed_row = embed_weight + decode_control[0] * HIDDEN_SIZE;
    const int position = decode_control[1];
    const int fa_kv_stride = FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    const int dn_state_stride = DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;
#pragma unroll
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        const __nv_bfloat16 *layer_input = layer == 0 ? embed_row : hidden_buffer;
#if defined(QWEN35X_VARIANT_4B)
        // The 4B schedule is fixed as three DeltaNet layers followed by one
        // full-attention layer. This is a graph-only specialization; avoid a
        // constant-memory layer-type lookup and dynamic slot counters.
        if ((layer & 3) != 3) {
            constexpr int DNETS_PER_GROUP = 3;
            const int dn_layer_idx = (layer / 4) * DNETS_PER_GROUP + (layer & 3);
            deltanet_layer(
                grid, layer_weights[layer].dn,
                USE_NVFP4 ? &layer_nvfp4_weights[layer] : nullptr,
                layer_input, g_residual, g_activations, g_qkv_scratch, g_z_scratch,
                g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
                dn_states + dn_layer_idx * dn_state_stride, conv_bufs, hidden_buffer,
                dn_layer_idx, shmem_bf16, 0, false);
        } else {
            const int fa_layer_idx = layer / 4;
            full_attention_layer(
                grid, layer_weights[layer].fa,
                USE_NVFP4 ? &layer_nvfp4_weights[layer] : nullptr,
                layer_input, fa_k_cache + fa_layer_idx * fa_kv_stride,
                fa_v_cache + fa_layer_idx * fa_kv_stride, g_residual, g_activations,
                g_qkv_scratch, g_kv_scratch, g_attn_out, g_attn_partials, g_mlp_inter,
                hidden_buffer, position, max_seq_len, shmem_bf16, 0, false);
        }
#else
        int dn_layer_idx = 0;
        int fa_layer_idx = 0;
        for (int previous_layer = 0; previous_layer < layer; ++previous_layer) {
            if (LAYER_TYPE[previous_layer] == 0) ++dn_layer_idx;
            else ++fa_layer_idx;
        }
        if (LAYER_TYPE[layer] == 0) {
            deltanet_layer(
                grid, layer_weights[layer].dn,
                USE_NVFP4 ? &layer_nvfp4_weights[layer] : nullptr,
                layer_input, g_residual, g_activations, g_qkv_scratch, g_z_scratch,
                g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
                dn_states + dn_layer_idx * dn_state_stride, conv_bufs, hidden_buffer,
                dn_layer_idx, shmem_bf16, 0, false);
            ++dn_layer_idx;
        } else {
            full_attention_layer(
                grid, layer_weights[layer].fa,
                USE_NVFP4 ? &layer_nvfp4_weights[layer] : nullptr,
                layer_input, fa_k_cache + fa_layer_idx * fa_kv_stride,
                fa_v_cache + fa_layer_idx * fa_kv_stride, g_residual, g_activations,
                g_qkv_scratch, g_kv_scratch, g_attn_out, g_attn_partials, g_mlp_inter,
                hidden_buffer, position, max_seq_len, shmem_bf16, 0, false);
            ++fa_layer_idx;
        }
#endif
    }

    if (blockIdx.x == 0) {
        decode_graph_group4_final_norm(hidden_buffer, final_norm_weight, g_activations, g_normalized);
    }
}

// CUDA Graph replay needs fresh cooperative-grid barrier state for every
// layer. Reset every slot in one captured node instead of capturing two memset
// nodes per layer plus one for the LM-head reduction.
__global__ void reset_decode_graph_state_kernel(
    unsigned int *__restrict__ barrier_counters,
    unsigned int *__restrict__ barrier_generations,
    unsigned int *__restrict__ lm_sync_counter,
    int layer_count)
{
    const int index = static_cast<int>(threadIdx.x);
    if (index < layer_count) {
        barrier_counters[index] = 0;
        barrier_generations[index] = 0;
    }
    if (index == 0) {
        *lm_sync_counter = 0;
    }
}

__global__ void final_norm_kernel(
    const __nv_bfloat16 *__restrict__ hidden_buffer,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    float *__restrict__ g_activations,
    float *__restrict__ g_normalized)
{
    if (blockIdx.x != 0) {
        return;
    }
    __shared__ float smem_reduce[NUM_WARPS];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = __bfloat162float(hidden_buffer[i]);
        g_activations[i] = v;
        local_sum_sq += v * v;
    }
    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
        smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / HIDDEN_SIZE + RMS_EPS);
        }
    }
    __syncthreads();
    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float wt = __bfloat162float(__ldg(final_norm_weight + i));
        g_normalized[i] = g_activations[i] * rstd * (1.0f + wt);
    }
}

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_mlp_only_kernel(
    const LayerWeights *__restrict__ layer_weights,
    const LayerNvfp4Weights *__restrict__ layer_nvfp4_weights,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_mlp_inter,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    int layer)
{
    int num_blocks = gridDim.x;
    AtomicGridSync grid{barrier_counter, barrier_generation, (unsigned int)num_blocks, 0};
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);
    __nv_bfloat16 *s_act = shmem_bf16;
    const LayerNvfp4Weights *qw = layer_nvfp4_weights == nullptr ? nullptr : &layer_nvfp4_weights[layer];

    if (LAYER_TYPE[layer] == 0) {
        const DeltaNetWeights &w = layer_weights[layer].dn;
        rmsnorm_from_bf16(hidden_buffer, w.post_attn_layernorm_weight, s_act, g_residual);
        if (qw != nullptr && qw->ptrs[11].packed_weight != nullptr && qw->ptrs[12].packed_weight != nullptr) {
            matvec_gate_up_silu_nvfp4(s_act, qw->ptrs[11], qw->ptrs[12],
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        } else {
            matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        }
        grid.sync();
        float *s_mlp = reinterpret_cast<float *>(shmem_bf16);
        for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
        __syncthreads();
        if (qw != nullptr && qw->ptrs[13].packed_weight != nullptr) {
            matvec_down_residual_nvfp4(s_mlp, qw->ptrs[13], g_residual, hidden_buffer,
                                       INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
        } else {
            matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_buffer,
                                       INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
        }
    } else {
        const FullAttnWeights &w = layer_weights[layer].fa;
        rmsnorm_from_bf16(hidden_buffer, w.post_attn_layernorm_weight, s_act, g_residual);
        if (qw != nullptr && qw->ptrs[8].packed_weight != nullptr && qw->ptrs[9].packed_weight != nullptr) {
            matvec_gate_up_silu_nvfp4(s_act, qw->ptrs[8], qw->ptrs[9],
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        } else {
            matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        }
        grid.sync();
        float *s_mlp = reinterpret_cast<float *>(shmem_bf16);
        for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
        __syncthreads();
        if (qw != nullptr && qw->ptrs[10].packed_weight != nullptr) {
            matvec_down_residual_nvfp4(s_mlp, qw->ptrs[10], g_residual, hidden_buffer,
                                       INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
        } else {
            matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_buffer,
                                       INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
        }
    }
    grid.sync();
}

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_mlp_from_activation_kernel(
    const LayerWeights *__restrict__ layer_weights,
    const LayerNvfp4Weights *__restrict__ layer_nvfp4_weights,
    const float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_mlp_inter,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    int layer)
{
    int num_blocks = gridDim.x;
    AtomicGridSync grid{barrier_counter, barrier_generation, (unsigned int)num_blocks, 0};
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *s_act = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);
    const LayerNvfp4Weights *qw = layer_nvfp4_weights == nullptr ? nullptr : &layer_nvfp4_weights[layer];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        s_act[i] = __float2bfloat16(g_activations[i]);
    }
    __syncthreads();

    if (LAYER_TYPE[layer] == 0) {
        const DeltaNetWeights &w = layer_weights[layer].dn;
        if (qw != nullptr && qw->ptrs[11].packed_weight != nullptr && qw->ptrs[12].packed_weight != nullptr) {
            matvec_gate_up_silu_nvfp4(s_act, qw->ptrs[11], qw->ptrs[12],
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        } else {
            matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        }
        grid.sync();
        float *s_mlp = reinterpret_cast<float *>(s_act);
        for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
        __syncthreads();
        if (qw != nullptr && qw->ptrs[13].packed_weight != nullptr) {
            matvec_down_residual_nvfp4(s_mlp, qw->ptrs[13], g_residual, hidden_buffer,
                                       INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
        } else {
            matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_buffer,
                                       INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
        }
    } else {
        const FullAttnWeights &w = layer_weights[layer].fa;
        if (qw != nullptr && qw->ptrs[8].packed_weight != nullptr && qw->ptrs[9].packed_weight != nullptr) {
            matvec_gate_up_silu_nvfp4(s_act, qw->ptrs[8], qw->ptrs[9],
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        } else {
            matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        }
        grid.sync();
        float *s_mlp = reinterpret_cast<float *>(s_act);
        for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
        __syncthreads();
        if (qw != nullptr && qw->ptrs[10].packed_weight != nullptr) {
            matvec_down_residual_nvfp4(s_mlp, qw->ptrs[10], g_residual, hidden_buffer,
                                       INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
        } else {
            matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_buffer,
                                       INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
        }
    }
    grid.sync();
}

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_gate_up_from_activation_kernel(
    const LayerWeights *__restrict__ layer_weights,
    const LayerNvfp4Weights *__restrict__ layer_nvfp4_weights,
    const float *__restrict__ g_activations,
    float *__restrict__ g_mlp_inter,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    int layer)
{
    int num_blocks = gridDim.x;
    AtomicGridSync grid{barrier_counter, barrier_generation, (unsigned int)num_blocks, 0};
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *s_act = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);
    const LayerNvfp4Weights *qw = layer_nvfp4_weights == nullptr ? nullptr : &layer_nvfp4_weights[layer];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        s_act[i] = __float2bfloat16(g_activations[i]);
    }
    __syncthreads();

    if (LAYER_TYPE[layer] == 0) {
        const DeltaNetWeights &w = layer_weights[layer].dn;
        if (qw != nullptr && qw->ptrs[11].packed_weight != nullptr && qw->ptrs[12].packed_weight != nullptr) {
            matvec_gate_up_silu_nvfp4(s_act, qw->ptrs[11], qw->ptrs[12],
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        } else {
            matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        }
    } else {
        const FullAttnWeights &w = layer_weights[layer].fa;
        if (qw != nullptr && qw->ptrs[8].packed_weight != nullptr && qw->ptrs[9].packed_weight != nullptr) {
            matvec_gate_up_silu_nvfp4(s_act, qw->ptrs[8], qw->ptrs[9],
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        } else {
            matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                                      g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
        }
    }
    grid.sync();
}

// =============================================================================
// C entry point
// =============================================================================

extern "C" void launch_decode(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
    const LayerNvfp4Weights *layer_nvfp4_weights,
    const void *final_norm_weight,
    const void *lm_head_weight,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_attn_partials, void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    float *seen_token_mask,
    float repetition_penalty,
    int position, int max_seq_len,
    int use_sm120_mlp,
    qwen35x::cuda_backend::Qwen35xDecodeProfile *profile,
    cudaStream_t stream)
{
    int device_id = 0;
    int sm_count = 0;
    int active_blocks_per_sm = 0;
    int decode_blocks = NUM_BLOCKS;
    int max_safe_blocks = NUM_BLOCKS;

    if (cudaGetDevice(&device_id) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            decode_kernel,
            BLOCK_SIZE,
            0) == cudaSuccess &&
        sm_count > 0 &&
        active_blocks_per_sm > 0) {
        const int resident_blocks = sm_count * active_blocks_per_sm;
        if (resident_blocks > 0) {
            max_safe_blocks = resident_blocks;
        }
    }

    if (g_decode_blocks_override > 0) {
        decode_blocks = g_decode_blocks_override;
    } else {
        decode_blocks = max_safe_blocks;
    }

    if (decode_blocks < MIN_DECODE_BLOCKS) decode_blocks = MIN_DECODE_BLOCKS;
    if (decode_blocks > max_safe_blocks) decode_blocks = max_safe_blocks;
    if (decode_blocks > MAX_DECODE_BLOCKS) decode_blocks = MAX_DECODE_BLOCKS;
    if (profile) {
        profile->decode_blocks = decode_blocks;
        profile->max_safe_decode_blocks = max_safe_blocks;
    }

    cudaEvent_t profile_total_start = nullptr;
    cudaEvent_t profile_decode_end = nullptr;
    cudaEvent_t profile_lm_end = nullptr;
    if (profile) {
        cudaEventCreate(&profile_total_start);
        cudaEventCreate(&profile_decode_end);
        cudaEventCreate(&profile_lm_end);
        cudaEventRecord(profile_total_start, stream);
    }

    cudaMemsetAsync(barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(barrier_generation, 0, sizeof(unsigned int), stream);

    decode_kernel<<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        (const __nv_bfloat16 *)embed_weight,
        (const __nv_bfloat16 *)final_norm_weight,
        (const __nv_bfloat16 *)lm_head_weight,
        layer_weights,
        layer_nvfp4_weights,
        (__nv_bfloat16 *)fa_k_cache, (__nv_bfloat16 *)fa_v_cache,
        (float *)dn_states, (float *)conv_bufs,
        (__nv_bfloat16 *)hidden_buffer,
        (float *)g_activations, (__nv_bfloat16 *)g_residual,
        (float *)g_qkv_scratch, (float *)g_kv_scratch,
        (float *)g_attn_out, (float *)g_attn_partials, (float *)g_mlp_inter,
        (float *)g_z_scratch, (float *)g_beta_scratch,
        (float *)g_alpha_scratch, (float *)g_normalized,
        barrier_counter, barrier_generation,
        input_token_id, position, max_seq_len,
        use_sm120_mlp);

    if (profile) {
        cudaEventRecord(profile_decode_end, stream);
    }

    cudaMemsetAsync(lm_sync_counter, 0, sizeof(unsigned int), stream);

    lm_head_kernel<1><<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        block_max_vals, block_max_idxs,
        output_token_id, lm_sync_counter,
        seen_token_mask, repetition_penalty, nullptr);

    if (profile) {
        cudaEventRecord(profile_lm_end, stream);
        cudaEventSynchronize(profile_lm_end);
        float decode_ms = 0.0f;
        float lm_ms = 0.0f;
        float total_ms = 0.0f;
        if (cudaEventElapsedTime(&decode_ms, profile_total_start, profile_decode_end) == cudaSuccess) {
            profile->decode_kernel_ms += static_cast<double>(decode_ms);
        }
        if (cudaEventElapsedTime(&lm_ms, profile_decode_end, profile_lm_end) == cudaSuccess) {
            profile->lm_head_ms += static_cast<double>(lm_ms);
        }
        if (cudaEventElapsedTime(&total_ms, profile_total_start, profile_lm_end) == cudaSuccess) {
            profile->launch_total_ms += static_cast<double>(total_ms);
        }
        cudaEventDestroy(profile_total_start);
        cudaEventDestroy(profile_decode_end);
        cudaEventDestroy(profile_lm_end);
    }
}

extern "C" void launch_decode_prefix_mlp(
    int input_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
    const LayerNvfp4Weights *layer_nvfp4_weights,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_attn_partials, void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    int layer, int position, int max_seq_len, int requested_decode_blocks, int external_mlp,
    cudaStream_t stream)
{
    int device_id = 0;
    int sm_count = 0;
    int active_blocks_per_sm = 0;
    int decode_blocks = NUM_BLOCKS;
    int max_safe_blocks = NUM_BLOCKS;
    if (cudaGetDevice(&device_id) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            decode_prefix_mlp_kernel,
            BLOCK_SIZE,
            0) == cudaSuccess &&
        sm_count > 0 &&
        active_blocks_per_sm > 0) {
        const int resident_blocks = sm_count * active_blocks_per_sm;
        if (resident_blocks > 0) {
            max_safe_blocks = resident_blocks;
        }
    }
    if (g_decode_blocks_override > 0) {
        decode_blocks = g_decode_blocks_override;
    } else if (requested_decode_blocks > 0) {
        decode_blocks = requested_decode_blocks;
    } else {
        decode_blocks = max_safe_blocks;
    }
    if (decode_blocks < MIN_DECODE_BLOCKS) decode_blocks = MIN_DECODE_BLOCKS;
    if (decode_blocks > max_safe_blocks) decode_blocks = max_safe_blocks;
    if (decode_blocks > MAX_DECODE_BLOCKS) decode_blocks = MAX_DECODE_BLOCKS;

    cudaMemsetAsync(barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(barrier_generation, 0, sizeof(unsigned int), stream);
    decode_prefix_mlp_kernel<<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        (const __nv_bfloat16 *)embed_weight,
        layer_weights,
        layer_nvfp4_weights,
        (__nv_bfloat16 *)fa_k_cache, (__nv_bfloat16 *)fa_v_cache,
        (float *)dn_states, (float *)conv_bufs,
        (__nv_bfloat16 *)hidden_buffer,
        (float *)g_activations, (__nv_bfloat16 *)g_residual,
        (float *)g_qkv_scratch, (float *)g_kv_scratch,
        (float *)g_attn_out, (float *)g_attn_partials, (float *)g_mlp_inter,
        (float *)g_z_scratch, (float *)g_beta_scratch,
        (float *)g_alpha_scratch,
        barrier_counter, barrier_generation,
        input_token_id, layer, position, max_seq_len, external_mlp);
}

extern "C" void launch_decode_graph_layer(
    const int *decode_control,
    const void *embed_weight, const LayerWeights *layer_weights,
    const LayerNvfp4Weights *layer_nvfp4_weights,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_attn_partials, void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    int layer, int max_seq_len, int decode_blocks, cudaStream_t stream)
{
    if (decode_blocks < MIN_DECODE_BLOCKS) decode_blocks = MIN_DECODE_BLOCKS;
    if (decode_blocks > MAX_DECODE_BLOCKS) decode_blocks = MAX_DECODE_BLOCKS;
    decode_graph_layer_kernel<<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        decode_control,
        (const __nv_bfloat16 *)embed_weight,
        layer_weights,
        layer_nvfp4_weights,
        (__nv_bfloat16 *)fa_k_cache, (__nv_bfloat16 *)fa_v_cache,
        (float *)dn_states, (float *)conv_bufs,
        (__nv_bfloat16 *)hidden_buffer,
        (float *)g_activations, (__nv_bfloat16 *)g_residual,
        (float *)g_qkv_scratch, (float *)g_kv_scratch,
        (float *)g_attn_out, (float *)g_attn_partials, (float *)g_mlp_inter,
        (float *)g_z_scratch, (float *)g_beta_scratch, (float *)g_alpha_scratch,
        barrier_counter, barrier_generation,
        layer, max_seq_len);
}

extern "C" void launch_decode_graph_deltanet(
    const int *decode_control,
    const void *embed_weight,
    const LayerWeights *layer_weights,
    const LayerNvfp4Weights *layer_nvfp4_weights,
    const void *final_norm_weight,
    void *dn_state,
    void *conv_buf,
    void *hidden_buffer,
    void *g_activations,
    void *g_residual,
    void *g_qkv_scratch,
    void *g_attn_out,
    void *g_mlp_inter,
    void *g_z_scratch,
    void *g_beta_scratch,
    void *g_alpha_scratch,
    void *g_normalized,
    unsigned int *lm_sync_counter,
    unsigned int *barrier_counter,
    unsigned int *barrier_generation,
    int decode_blocks,
    cudaStream_t stream)
{
    if (decode_blocks < MIN_DECODE_BLOCKS) decode_blocks = MIN_DECODE_BLOCKS;
    if (decode_blocks > MAX_DECODE_BLOCKS) decode_blocks = MAX_DECODE_BLOCKS;
    decode_graph_deltanet_kernel<false><<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        decode_control,
        static_cast<const __nv_bfloat16 *>(embed_weight),
        layer_weights,
        layer_nvfp4_weights,
        static_cast<float *>(dn_state),
        static_cast<float *>(conv_buf),
        static_cast<__nv_bfloat16 *>(hidden_buffer),
        static_cast<float *>(g_activations),
        static_cast<__nv_bfloat16 *>(g_residual),
        static_cast<float *>(g_qkv_scratch),
        static_cast<float *>(g_attn_out),
        static_cast<float *>(g_mlp_inter),
        static_cast<float *>(g_z_scratch),
        static_cast<float *>(g_beta_scratch),
        static_cast<float *>(g_alpha_scratch),
        barrier_counter,
        barrier_generation);
}

extern "C" void launch_decode_graph_deltanet_first(
    const int *decode_control,
    const void *embed_weight,
    const LayerWeights *layer_weights,
    const LayerNvfp4Weights *layer_nvfp4_weights,
    void *dn_state,
    void *conv_buf,
    void *hidden_buffer,
    void *g_activations,
    void *g_residual,
    void *g_qkv_scratch,
    void *g_attn_out,
    void *g_mlp_inter,
    void *g_z_scratch,
    void *g_beta_scratch,
    void *g_alpha_scratch,
    unsigned int *barrier_counter,
    unsigned int *barrier_generation,
    int decode_blocks,
    cudaStream_t stream)
{
    if (decode_blocks < MIN_DECODE_BLOCKS) decode_blocks = MIN_DECODE_BLOCKS;
    if (decode_blocks > MAX_DECODE_BLOCKS) decode_blocks = MAX_DECODE_BLOCKS;
    decode_graph_deltanet_kernel<true><<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        decode_control,
        static_cast<const __nv_bfloat16 *>(embed_weight),
        layer_weights,
        layer_nvfp4_weights,
        static_cast<float *>(dn_state),
        static_cast<float *>(conv_buf),
        static_cast<__nv_bfloat16 *>(hidden_buffer),
        static_cast<float *>(g_activations),
        static_cast<__nv_bfloat16 *>(g_residual),
        static_cast<float *>(g_qkv_scratch),
        static_cast<float *>(g_attn_out),
        static_cast<float *>(g_mlp_inter),
        static_cast<float *>(g_z_scratch),
        static_cast<float *>(g_beta_scratch),
        static_cast<float *>(g_alpha_scratch),
        barrier_counter,
        barrier_generation);
}

extern "C" void launch_decode_graph_full_attention(
    const int *decode_control,
    const LayerWeights *layer_weights,
    const LayerNvfp4Weights *layer_nvfp4_weights,
    void *k_cache,
    void *v_cache,
    void *hidden_buffer,
    void *g_activations,
    void *g_residual,
    void *g_qkv_scratch,
    void *g_kv_scratch,
    void *g_attn_out,
    void *g_attn_partials,
    void *g_mlp_inter,
    unsigned int *barrier_counter,
    unsigned int *barrier_generation,
    int max_seq_len,
    int decode_blocks,
    cudaStream_t stream)
{
    if (decode_blocks < MIN_DECODE_BLOCKS) decode_blocks = MIN_DECODE_BLOCKS;
    if (decode_blocks > MAX_DECODE_BLOCKS) decode_blocks = MAX_DECODE_BLOCKS;
    decode_graph_full_attention_kernel<<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        decode_control,
        layer_weights,
        layer_nvfp4_weights,
        static_cast<__nv_bfloat16 *>(k_cache),
        static_cast<__nv_bfloat16 *>(v_cache),
        static_cast<__nv_bfloat16 *>(hidden_buffer),
        static_cast<float *>(g_activations),
        static_cast<__nv_bfloat16 *>(g_residual),
        static_cast<float *>(g_qkv_scratch),
        static_cast<float *>(g_kv_scratch),
        static_cast<float *>(g_attn_out),
        static_cast<float *>(g_attn_partials),
        static_cast<float *>(g_mlp_inter),
        barrier_counter,
        barrier_generation,
        max_seq_len);
}

extern "C" void launch_decode_graph_group4(
    const int *decode_control,
    const void *embed_weight,
    const LayerWeights *layer_weights,
    const LayerNvfp4Weights *layer_nvfp4_weights,
    const void *final_norm_weight,
    void *fa_k_cache,
    void *fa_v_cache,
    void *dn_states,
    void *conv_bufs,
    void *hidden_buffer,
    void *g_activations,
    void *g_residual,
    void *g_qkv_scratch,
    void *g_kv_scratch,
    void *g_attn_out,
    void *g_attn_partials,
    void *g_mlp_inter,
    void *g_z_scratch,
    void *g_beta_scratch,
    void *g_alpha_scratch,
    void *g_normalized,
    unsigned int *lm_sync_counter,
    unsigned int *barrier_counter,
    unsigned int *barrier_generation,
    int group_start,
    int max_seq_len,
    int decode_blocks,
    cudaStream_t stream)
{
    if (decode_blocks < MIN_DECODE_BLOCKS) decode_blocks = MIN_DECODE_BLOCKS;
    if (decode_blocks > MAX_DECODE_BLOCKS) decode_blocks = MAX_DECODE_BLOCKS;
    const auto launch = [&](auto group_start_tag) {
        constexpr int GROUP_START = decltype(group_start_tag)::value;
        decode_graph_group4_kernel<GROUP_START><<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
            decode_control,
            static_cast<const __nv_bfloat16 *>(embed_weight),
            layer_weights,
            layer_nvfp4_weights,
            static_cast<const __nv_bfloat16 *>(final_norm_weight),
            static_cast<__nv_bfloat16 *>(fa_k_cache),
            static_cast<__nv_bfloat16 *>(fa_v_cache),
            static_cast<float *>(dn_states),
            static_cast<float *>(conv_bufs),
            static_cast<__nv_bfloat16 *>(hidden_buffer),
            static_cast<float *>(g_activations),
            static_cast<__nv_bfloat16 *>(g_residual),
            static_cast<float *>(g_qkv_scratch),
            static_cast<float *>(g_kv_scratch),
            static_cast<float *>(g_attn_out),
            static_cast<float *>(g_attn_partials),
            static_cast<float *>(g_mlp_inter),
            static_cast<float *>(g_z_scratch),
            static_cast<float *>(g_beta_scratch),
            static_cast<float *>(g_alpha_scratch),
            static_cast<float *>(g_normalized),
            lm_sync_counter,
            barrier_counter,
            barrier_generation,
            max_seq_len);
    };
    switch (group_start) {
        case 0: launch(std::integral_constant<int, 0>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        case 12: launch(std::integral_constant<int, 12>{}); break;
        case 16: launch(std::integral_constant<int, 16>{}); break;
        case 20: launch(std::integral_constant<int, 20>{}); break;
        case 24: launch(std::integral_constant<int, 24>{}); break;
        case 28: launch(std::integral_constant<int, 28>{}); break;
        default: break;
    }
}

extern "C" void launch_decode_graph_megakernel(
    const int *decode_control,
    const void *embed_weight,
    const void *final_norm_weight,
    const LayerWeights *layer_weights,
    const LayerNvfp4Weights *layer_nvfp4_weights,
    void *fa_k_cache,
    void *fa_v_cache,
    void *dn_states,
    void *conv_bufs,
    void *hidden_buffer,
    void *g_activations,
    void *g_residual,
    void *g_qkv_scratch,
    void *g_kv_scratch,
    void *g_attn_out,
    void *g_attn_partials,
    void *g_mlp_inter,
    void *g_z_scratch,
    void *g_beta_scratch,
    void *g_alpha_scratch,
    void *g_normalized,
    unsigned int *barrier_counter,
    unsigned int *barrier_generation,
    unsigned int *lm_sync_counter,
    float *seen_token_mask,
    float repetition_penalty,
    int max_seq_len,
    int decode_blocks,
    cudaStream_t stream)
{
    if (decode_blocks < MIN_DECODE_BLOCKS) decode_blocks = MIN_DECODE_BLOCKS;
    if (decode_blocks > MAX_DECODE_BLOCKS) decode_blocks = MAX_DECODE_BLOCKS;
    // CUDA Graph decode is currently BF16-only; instantiate a BF16 path so
    // layer helpers can discard NVFP4 selection branches at compile time.
    decode_graph_megakernel<false><<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        decode_control,
        static_cast<const __nv_bfloat16 *>(embed_weight),
        static_cast<const __nv_bfloat16 *>(final_norm_weight),
        layer_weights, layer_nvfp4_weights,
        static_cast<__nv_bfloat16 *>(fa_k_cache), static_cast<__nv_bfloat16 *>(fa_v_cache),
        static_cast<float *>(dn_states), static_cast<float *>(conv_bufs),
        static_cast<__nv_bfloat16 *>(hidden_buffer), static_cast<float *>(g_activations),
        static_cast<__nv_bfloat16 *>(g_residual), static_cast<float *>(g_qkv_scratch),
        static_cast<float *>(g_kv_scratch), static_cast<float *>(g_attn_out),
        static_cast<float *>(g_attn_partials), static_cast<float *>(g_mlp_inter),
        static_cast<float *>(g_z_scratch), static_cast<float *>(g_beta_scratch),
        static_cast<float *>(g_alpha_scratch), static_cast<float *>(g_normalized),
        barrier_counter, barrier_generation, lm_sync_counter,
        seen_token_mask, repetition_penalty, max_seq_len);
}

extern "C" void launch_decode_graph_reset(
    unsigned int *barrier_counters,
    unsigned int *barrier_generations,
    unsigned int *lm_sync_counter,
    int layer_count,
    cudaStream_t stream)
{
    reset_decode_graph_state_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        barrier_counters, barrier_generations, lm_sync_counter, layer_count);
}

extern "C" void launch_decode_final_lm(
    int *output_token_id,
    const void *final_norm_weight,
    const void *lm_head_weight,
    void *hidden_buffer,
    void *g_activations,
    void *g_normalized,
    float *block_max_vals,
    int *block_max_idxs,
    unsigned int *lm_sync_counter,
    float *seen_token_mask,
    float repetition_penalty,
    cudaStream_t stream)
{
    final_norm_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16 *)hidden_buffer,
        (const __nv_bfloat16 *)final_norm_weight,
        (float *)g_activations,
        (float *)g_normalized);
    cudaMemsetAsync(lm_sync_counter, 0, sizeof(unsigned int), stream);
    lm_head_kernel<1><<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        block_max_vals, block_max_idxs,
        output_token_id, lm_sync_counter,
        seen_token_mask, repetition_penalty, nullptr);
}

extern "C" void launch_decode_graph_final_lm(
    int *output_token_id,
    const void *final_norm_weight,
    const void *lm_head_weight,
    void *hidden_buffer,
    void *g_activations,
    void *g_normalized,
    float *block_max_vals,
    int *block_max_idxs,
    unsigned int *lm_sync_counter,
    float *seen_token_mask,
    float repetition_penalty,
    cudaStream_t stream)
{
    final_norm_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16 *)hidden_buffer,
        (const __nv_bfloat16 *)final_norm_weight,
        (float *)g_activations,
        (float *)g_normalized);
    lm_head_kernel<1><<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        block_max_vals, block_max_idxs,
        output_token_id, lm_sync_counter,
        seen_token_mask, repetition_penalty, nullptr);
}

// The final RMSNorm is fused into the final group kernel for graph/multi-kernel
// decode. This wrapper launches just the vocabulary projection.
constexpr int GRAPH_LM_NUM_BLOCKS = LM_NUM_BLOCKS;
extern "C" void launch_decode_graph_lm_head(
    int *output_token_id,
    const void *lm_head_weight,
    void *g_normalized,
    float *block_max_vals,
    int *block_max_idxs,
    unsigned int *lm_sync_counter,
    float *seen_token_mask,
    float repetition_penalty,
    int *next_decode_control,
    cudaStream_t stream)
{
    lm_head_kernel<4><<<GRAPH_LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        static_cast<const float *>(g_normalized),
        static_cast<const __nv_bfloat16 *>(lm_head_weight),
        block_max_vals, block_max_idxs,
        output_token_id, lm_sync_counter,
        seen_token_mask, repetition_penalty, next_decode_control);
}

extern "C" void launch_decode_mlp_only(
    const LayerWeights *layer_weights,
    const LayerNvfp4Weights *layer_nvfp4_weights,
    void *hidden_buffer,
    void *g_residual,
    void *g_mlp_inter,
    unsigned int *barrier_counter,
    unsigned int *barrier_generation,
    int layer,
    int requested_decode_blocks,
    cudaStream_t stream)
{
    int decode_blocks = NUM_BLOCKS;
    if (g_decode_blocks_override > 0) {
        decode_blocks = g_decode_blocks_override;
    } else if (requested_decode_blocks > 0) {
        decode_blocks = requested_decode_blocks;
    }
    if (decode_blocks < MIN_DECODE_BLOCKS) decode_blocks = MIN_DECODE_BLOCKS;
    if (decode_blocks > MAX_DECODE_BLOCKS) decode_blocks = MAX_DECODE_BLOCKS;
    cudaMemsetAsync(barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(barrier_generation, 0, sizeof(unsigned int), stream);
    decode_mlp_only_kernel<<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        layer_weights,
        layer_nvfp4_weights,
        (__nv_bfloat16 *)hidden_buffer,
        (__nv_bfloat16 *)g_residual,
        (float *)g_mlp_inter,
        barrier_counter,
        barrier_generation,
        layer);
}

extern "C" void launch_decode_mlp_from_activation(
    const LayerWeights *layer_weights,
    const LayerNvfp4Weights *layer_nvfp4_weights,
    const void *g_activations,
    void *hidden_buffer,
    void *g_residual,
    void *g_mlp_inter,
    unsigned int *barrier_counter,
    unsigned int *barrier_generation,
    int layer,
    int decode_blocks,
    cudaStream_t stream)
{
    cudaMemsetAsync(barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(barrier_generation, 0, sizeof(unsigned int), stream);
    if (decode_blocks < MIN_DECODE_BLOCKS) decode_blocks = MIN_DECODE_BLOCKS;
    if (decode_blocks > MAX_DECODE_BLOCKS) decode_blocks = MAX_DECODE_BLOCKS;
    decode_mlp_from_activation_kernel<<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        layer_weights,
        layer_nvfp4_weights,
        (const float *)g_activations,
        (__nv_bfloat16 *)hidden_buffer,
        (__nv_bfloat16 *)g_residual,
        (float *)g_mlp_inter,
        barrier_counter,
        barrier_generation,
        layer);
}

extern "C" void launch_decode_gate_up_from_activation(
    const LayerWeights *layer_weights,
    const LayerNvfp4Weights *layer_nvfp4_weights,
    const void *g_activations,
    void *g_mlp_inter,
    unsigned int *barrier_counter,
    unsigned int *barrier_generation,
    int layer,
    int decode_blocks,
    cudaStream_t stream)
{
    cudaMemsetAsync(barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(barrier_generation, 0, sizeof(unsigned int), stream);
    if (decode_blocks < MIN_DECODE_BLOCKS) decode_blocks = MIN_DECODE_BLOCKS;
    if (decode_blocks > MAX_DECODE_BLOCKS) decode_blocks = MAX_DECODE_BLOCKS;
    decode_gate_up_from_activation_kernel<<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        layer_weights,
        layer_nvfp4_weights,
        (const float *)g_activations,
        (float *)g_mlp_inter,
        barrier_counter,
        barrier_generation,
        layer);
}
extern "C" void set_decode_blocks_override(int blocks) {
    g_decode_blocks_override = blocks;
}

extern "C" int query_max_safe_decode_blocks() {
    int device_id = 0;
    int sm_count = 0;
    int active_blocks_per_sm = 0;
    int max_safe_blocks = NUM_BLOCKS;
    if (cudaGetDevice(&device_id) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            decode_kernel,
            BLOCK_SIZE,
            0) == cudaSuccess &&
        sm_count > 0 &&
        active_blocks_per_sm > 0) {
        const int resident_blocks = sm_count * active_blocks_per_sm;
        if (resident_blocks > 0) {
            max_safe_blocks = resident_blocks;
        }
    }
    if (max_safe_blocks < 1) max_safe_blocks = 1;
    return max_safe_blocks;
}

extern "C" int query_max_safe_graph_decode_blocks() {
    int device_id = 0;
    int sm_count = 0;
    int active_first_group_blocks_per_sm = 0;
    int active_group_blocks_per_sm = 0;
    int active_final_group_blocks_per_sm = 0;
    int max_safe_blocks = NUM_BLOCKS;
    if (cudaGetDevice(&device_id) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_first_group_blocks_per_sm,
            decode_graph_group4_kernel<0>,
            BLOCK_SIZE,
            0) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_group_blocks_per_sm,
            decode_graph_group4_kernel<4>,
            BLOCK_SIZE,
            0) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_final_group_blocks_per_sm,
            decode_graph_group4_kernel<NUM_LAYERS - 4>,
            BLOCK_SIZE,
            0) == cudaSuccess &&
        sm_count > 0 &&
        active_first_group_blocks_per_sm > 0 &&
        active_group_blocks_per_sm > 0 &&
        active_final_group_blocks_per_sm > 0) {
        const int resident_blocks = sm_count * min(
            active_first_group_blocks_per_sm,
            min(active_group_blocks_per_sm, active_final_group_blocks_per_sm));
        if (resident_blocks > 0) {
            max_safe_blocks = resident_blocks;
        }
    }
    if (max_safe_blocks < MIN_DECODE_BLOCKS) max_safe_blocks = MIN_DECODE_BLOCKS;
    if (max_safe_blocks > MAX_DECODE_BLOCKS) max_safe_blocks = MAX_DECODE_BLOCKS;
    return max_safe_blocks;
}
