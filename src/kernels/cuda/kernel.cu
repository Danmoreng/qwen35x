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

// =============================================================================
// C entry point
// =============================================================================

extern "C" void launch_decode(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
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
