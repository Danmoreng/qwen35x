#pragma once

#include "common.cuh"
#include "variant.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>

// =============================================================================
// LM Head: vocab projection + argmax
// =============================================================================

__global__ void lm_head_kernel(
    const float *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ weight,   // [VOCAB, HIDDEN] bf16
    float *__restrict__ block_max_vals,
    int *__restrict__ block_max_idxs,
    int *__restrict__ output_token,
    unsigned int *__restrict__ sync_counter,
    const float *__restrict__ seen_token_mask,
    float repetition_penalty)
{
    __shared__ float s_hidden[HIDDEN_SIZE];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LM_BLOCK_SIZE) s_hidden[i] = hidden[i];
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = LM_BLOCK_SIZE / WARP_SIZE;
    int rpb = (VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int rs = blockIdx.x * rpb, re = min(rs + rpb, VOCAB_SIZE);

    float local_max = -FLT_MAX; int local_max_idx = -1;
    for (int m = rs + warp_id; m < re; m += num_warps) {
        const __nv_bfloat16 *w_row = weight + m * HIDDEN_SIZE;
        float sum = 0;
#pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
            const __nv_bfloat16 *wp = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
            for (int i = 0; i < 8; i++) sum += __bfloat162float(wp[i]) * s_hidden[k+i];
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0 && repetition_penalty > 1.0f && seen_token_mask[m] > 0.0f) {
            sum = (sum > 0.0f) ? (sum / repetition_penalty) : (sum * repetition_penalty);
        }
        if (lane_id == 0 && sum > local_max) { local_max = sum; local_max_idx = m; }
    }
    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float wm[32]; __shared__ int wi[32];
    if (lane_id == 0) { wm[warp_id] = local_max; wi[warp_id] = local_max_idx; }
    __syncthreads();
    if (warp_id == 0) {
        float mv = (lane_id < num_warps) ? wm[lane_id] : -FLT_MAX;
        int mi = (lane_id < num_warps) ? wi[lane_id] : -1;
        for (int o = WARP_SIZE/2; o > 0; o /= 2) {
            float ov = __shfl_down_sync(0xffffffff, mv, o);
            int oi = __shfl_down_sync(0xffffffff, mi, o);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lane_id == 0) { block_max_vals[blockIdx.x] = mv; block_max_idxs[blockIdx.x] = mi; }
    }
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicAdd(sync_counter, 1); }
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) { volatile unsigned int *vc = (volatile unsigned int *)sync_counter; while (*vc < (unsigned int)gridDim.x) {} __threadfence(); }
        __syncthreads();
        int tid = threadIdx.x; float bv = -FLT_MAX; int bi = -1;
        for (int i = tid; i < gridDim.x; i += LM_BLOCK_SIZE) { float v = block_max_vals[i]; if (v > bv) { bv = v; bi = block_max_idxs[i]; } }
        __shared__ float sv[256]; __shared__ int si[256];
        sv[tid] = bv; si[tid] = bi; __syncthreads();
        for (int s = LM_BLOCK_SIZE/2; s > 0; s >>= 1) { if (tid < s && sv[tid+s] > sv[tid]) { sv[tid] = sv[tid+s]; si[tid] = si[tid+s]; } __syncthreads(); }
        if (tid == 0) *output_token = si[0];
    }
}

