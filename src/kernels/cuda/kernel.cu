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
    int max_seq_len)
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
            conv_bufs, hidden_buffer, dn_layer_idx, shmem_bf16, true);
    } else {
        full_attention_layer(
            grid, layer_weights[layer].fa,
            layer_nvfp4_weights == nullptr ? nullptr : &layer_nvfp4_weights[layer],
            layer_input,
            fa_k_cache + fa_layer_idx * fa_kv_stride,
            fa_v_cache + fa_layer_idx * fa_kv_stride,
            g_residual, g_activations, g_qkv_scratch, g_kv_scratch,
            g_attn_out, g_attn_partials, g_mlp_inter, hidden_buffer,
            position, max_seq_len, shmem_bf16, true);
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
        input_token_id, position, max_seq_len);

    if (profile) {
        cudaEventRecord(profile_decode_end, stream);
    }

    cudaMemsetAsync(lm_sync_counter, 0, sizeof(unsigned int), stream);

    lm_head_kernel<<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        block_max_vals, block_max_idxs,
        output_token_id, lm_sync_counter,
        seen_token_mask, repetition_penalty);

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
    int layer, int position, int max_seq_len, int requested_decode_blocks,
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
        input_token_id, layer, position, max_seq_len);
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
    lm_head_kernel<<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        block_max_vals, block_max_idxs,
        output_token_id, lm_sync_counter,
        seen_token_mask, repetition_penalty);
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
