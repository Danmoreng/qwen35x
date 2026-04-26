#pragma once

#include "common.cuh"
#include "decode_sync.cuh"
#include "variant.cuh"
#include "weights.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// =============================================================================
// RMSNorm — reads bf16 input, writes bf16 output
// =============================================================================

__device__ void rmsnorm_redundant(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ s_out,        // shared memory bf16
    __nv_bfloat16 *__restrict__ g_residual)   // global bf16
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = __bfloat162float(__ldg(input + i));
        s_out[i] = __float2bfloat16(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = __bfloat162float(__ldg(weight + i));
        float v = __bfloat162float(s_out[i]);
        s_out[i] = __float2bfloat16(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// RMSNorm from bf16 buffer (for post-attn norm)
__device__ void rmsnorm_from_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ s_out,
    __nv_bfloat16 *__restrict__ g_residual)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = __bfloat162float(input[i]);
        s_out[i] = __float2bfloat16(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = __bfloat162float(__ldg(weight + i));
        float v = __bfloat162float(s_out[i]);
        s_out[i] = __float2bfloat16(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// =============================================================================
// BF16 Matvec: warp-per-row, activations in shared memory (bf16)
// =============================================================================

__device__ void matvec_bf16(
    const __nv_bfloat16 *__restrict__ s_input,  // shared memory bf16 [in_dim]
    const __nv_bfloat16 *__restrict__ weight,   // [out_dim, in_dim] bf16
    float *__restrict__ output,                  // [out_dim] f32 (accumulate in f32)
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const __nv_bfloat16 *w_row = weight + m * in_dim;
            float sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                sum += dot8_bf16(w_u4, s_input + k);
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) output[m] = sum;
        }
    }
}

__device__ __forceinline__ float decode_nvfp4_e2m1(const std::uint8_t nibble)
{
    const std::uint8_t magnitude = nibble & 0x7u;
    float value = 0.0f;
    if (magnitude == 1) value = 0.5f;
    else if (magnitude == 2) value = 1.0f;
    else if (magnitude == 3) value = 1.5f;
    else if (magnitude == 4) value = 2.0f;
    else if (magnitude == 5) value = 3.0f;
    else if (magnitude == 6) value = 4.0f;
    else if (magnitude == 7) value = 6.0f;
    return (nibble & 0x8u) ? -value : value;
}

__device__ __forceinline__ float decode_e4m3_scale(const std::uint8_t bits)
{
    if (bits == 0) return 0.0f;
    const int sign = (bits >> 7) & 1;
    const int exponent = (bits >> 3) & 0xf;
    const int mantissa = bits & 0x7;
    float value = 0.0f;
    if (exponent == 0) {
        value = ldexpf(float(mantissa) * 0.125f, -6);
    } else {
        value = ldexpf(1.0f + float(mantissa) * 0.125f, exponent - 7);
    }
    return sign ? -value : value;
}

static __device__ float nvfp4_row_dot(
    const Nvfp4Weight &weight,
    const __nv_bfloat16 *__restrict__ input,
    int row,
    int in_dim,
    int lane_id)
{
    const std::uint8_t *packed_row = weight.packed_weight + row * (in_dim / 2);
    const std::uint8_t *scale_row = weight.weight_scale + row * (in_dim / 16);
    const float scale2 = __ldg(weight.weight_scale_2);
    float sum = 0.0f;
#pragma unroll 4
    for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int col = k + i;
            const std::uint8_t packed = __ldg(packed_row + col / 2);
            const std::uint8_t nibble = (col & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
            const float scale = decode_e4m3_scale(__ldg(scale_row + col / 16)) * scale2;
            sum += decode_nvfp4_e2m1(nibble) * scale * __bfloat162float(input[col]);
        }
    }
    return sum;
}

static __device__ float nvfp4_row_dot_f32(
    const Nvfp4Weight &weight,
    const float *__restrict__ input,
    int row,
    int in_dim,
    int lane_id)
{
    const std::uint8_t *packed_row = weight.packed_weight + row * (in_dim / 2);
    const std::uint8_t *scale_row = weight.weight_scale + row * (in_dim / 16);
    const float scale2 = __ldg(weight.weight_scale_2);
    float sum = 0.0f;
#pragma unroll 4
    for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int col = k + i;
            const std::uint8_t packed = __ldg(packed_row + col / 2);
            const std::uint8_t nibble = (col & 1) == 0 ? (packed & 0x0fu) : (packed >> 4u);
            const float scale = decode_e4m3_scale(__ldg(scale_row + col / 16)) * scale2;
            sum += decode_nvfp4_e2m1(nibble) * scale * input[col];
        }
    }
    return sum;
}

static __device__ void matvec_nvfp4_bf16(
    const __nv_bfloat16 *__restrict__ s_input,
    const Nvfp4Weight &weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            float sum = nvfp4_row_dot(weight, s_input, m, in_dim, lane_id);
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) output[m] = sum;
        }
    }
}

static __device__ void matvec_select_bf16_or_nvfp4(
    const __nv_bfloat16 *__restrict__ s_input,
    const __nv_bfloat16 *__restrict__ bf16_weight,
    const LayerNvfp4Weights *__restrict__ nvfp4_layer,
    int nvfp4_ptr_idx,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    if (nvfp4_layer != nullptr && nvfp4_layer->ptrs[nvfp4_ptr_idx].packed_weight != nullptr) {
        matvec_nvfp4_bf16(s_input, nvfp4_layer->ptrs[nvfp4_ptr_idx], output, in_dim, out_dim, num_blocks);
    } else {
        matvec_bf16(s_input, bf16_weight, output, in_dim, out_dim, num_blocks);
    }
}

// Fused gate+up+SiLU matvec (bf16 weights, bf16 activations)
__device__ void matvec_gate_up_silu_bf16(
    const __nv_bfloat16 *__restrict__ s_input,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const __nv_bfloat16 *__restrict__ up_weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const __nv_bfloat16 *g_row = gate_weight + m * in_dim;
            const __nv_bfloat16 *u_row = up_weight + m * in_dim;
            float gate_sum = 0.0f, up_sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 g_u4 = load_128bit(reinterpret_cast<const uint4 *>(g_row + k));
                uint4 u_u4 = load_128bit(reinterpret_cast<const uint4 *>(u_row + k));
                gate_sum += dot8_bf16(g_u4, s_input + k);
                up_sum += dot8_bf16(u_u4, s_input + k);
            }
            gate_sum = warp_reduce_sum(gate_sum);
            up_sum = warp_reduce_sum(up_sum);
            if (lane_id == 0)
                output[m] = fast_silu(gate_sum) * up_sum;
        }
    }
}

static __device__ void matvec_gate_up_silu_nvfp4(
    const __nv_bfloat16 *__restrict__ s_input,
    const Nvfp4Weight &gate_weight,
    const Nvfp4Weight &up_weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            float gate_sum = nvfp4_row_dot(gate_weight, s_input, m, in_dim, lane_id);
            float up_sum = nvfp4_row_dot(up_weight, s_input, m, in_dim, lane_id);
            gate_sum = warp_reduce_sum(gate_sum);
            up_sum = warp_reduce_sum(up_sum);
            if (lane_id == 0)
                output[m] = fast_silu(gate_sum) * up_sum;
        }
    }
}

// Down projection + residual → bf16 hidden
__device__ void matvec_down_residual_bf16(
    const float *__restrict__ s_input,           // shared [INTER] f32
    const __nv_bfloat16 *__restrict__ weight,    // [HIDDEN, INTER] bf16
    const __nv_bfloat16 *__restrict__ residual,  // [HIDDEN] bf16
    __nv_bfloat16 *__restrict__ hidden_out,      // [HIDDEN] bf16
    int in_dim, int out_dim, int num_blocks)
{
    // This needs f32 input (MLP intermediate is f32). Convert on the fly.
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const __nv_bfloat16 *w_row = weight + m * in_dim;
            float sum = 0.0f;
            // Weight is bf16, input is f32 — convert input to bf16 on the fly
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const __nv_bfloat16 *w = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += __bfloat162float(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = __float2bfloat16(sum + __bfloat162float(residual[m]));
        }
    }
}

static __device__ void matvec_down_residual_nvfp4(
    const float *__restrict__ s_input,
    const Nvfp4Weight &weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            float sum = nvfp4_row_dot_f32(weight, s_input, m, in_dim, lane_id);
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = __float2bfloat16(sum + __bfloat162float(residual[m]));
        }
    }
}

// O projection + residual → bf16
__device__ void matvec_o_residual_bf16(
    const float *__restrict__ s_input,           // shared [Q_SIZE] f32
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const __nv_bfloat16 *w_row = weight + m * in_dim;
            float sum = 0.0f;
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const __nv_bfloat16 *w = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += __bfloat162float(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = __float2bfloat16(sum + __bfloat162float(residual[m]));
        }
    }
}

static __device__ void matvec_o_residual_nvfp4(
    const float *__restrict__ s_input,
    const Nvfp4Weight &weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            float sum = nvfp4_row_dot_f32(weight, s_input, m, in_dim, lane_id);
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = __float2bfloat16(sum + __bfloat162float(residual[m]));
        }
    }
}

// =============================================================================
// Full Attention layer (bf16)
// =============================================================================

__device__ void full_attention_layer(
    AtomicGridSync &grid,
    const FullAttnWeights &w,
    const LayerNvfp4Weights *__restrict__ qw,
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ g_residual,  // [HIDDEN] bf16
    float *__restrict__ g_activations,        // scratch f32
    float *__restrict__ g_q,                  // [FA_QPROJ_SIZE] f32
    float *__restrict__ g_kv,                 // [FA_KV_SIZE*2] f32
    float *__restrict__ g_attn_out,           // [FA_Q_SIZE] f32
    float *__restrict__ g_attn_partials,      // [gridDim.x * FA_GQA_RATIO * (2 + FA_HEAD_DIM)] f32
    float *__restrict__ g_mlp_inter,          // [INTER] f32
    __nv_bfloat16 *__restrict__ hidden_out,   // [HIDDEN] bf16
    int position, int max_seq_len,
    __nv_bfloat16 *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + QKV projection
    __nv_bfloat16 *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_select_bf16_or_nvfp4(s_norm, w.q_proj_weight, qw, 1, g_q, HIDDEN_SIZE, FA_QPROJ_SIZE, num_blocks);
    matvec_select_bf16_or_nvfp4(s_norm, w.k_proj_weight, qw, 2, g_kv, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    matvec_select_bf16_or_nvfp4(s_norm, w.v_proj_weight, qw, 3, g_kv + FA_KV_SIZE, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    grid.sync();

    // Phase 2: QK norm + partial RoPE + KV cache write
    if (block_id == 0) {
        float *k_buf = g_kv, *v_buf = g_kv + FA_KV_SIZE;
        for (int h = warp_id; h < FA_NUM_KV_HEADS; h += NUM_WARPS) {
            float *kh = k_buf + h * FA_HEAD_DIM, *vh = v_buf + h * FA_HEAD_DIM;
            __nv_bfloat16 *kc = k_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            __nv_bfloat16 *vc = v_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += kh[i]*kh[i];
            ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0);
            for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                float normed = kh[i] * sc * (1.0f + __bfloat162float(__ldg(w.k_norm_weight + i)));
                if (i < FA_ROTARY_DIM) {
                    float fe = float(2*(i%(FA_ROTARY_DIM/2))) / float(FA_ROTARY_DIM);
                    float freq = float(position) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    int p = (i < FA_ROTARY_DIM/2) ? i+FA_ROTARY_DIM/2 : i-FA_ROTARY_DIM/2;
                    float pv = kh[p]*sc*(1.0f+__bfloat162float(__ldg(w.k_norm_weight+p)));
                    float rotated = (i < FA_ROTARY_DIM/2) ? (normed*cv - pv*sv) : (pv*sv + normed*cv);
                    kc[i] = __float2bfloat16(rotated);
                } else { kc[i] = __float2bfloat16(normed); }
                vc[i] = __float2bfloat16(vh[i]);
            }
        }
    }
    // Q norm + RoPE (all blocks)
    {
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        for (int qh = hs; qh < he; qh++) {
            float *qh_ptr = g_q + qh * FA_HEAD_DIM * 2;
            if (warp_id == 0) {
                float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += qh_ptr[i]*qh_ptr[i];
                ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
                sc = __shfl_sync(0xffffffff, sc, 0);
                for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                    float normed = qh_ptr[i]*sc*(1.0f+__bfloat162float(__ldg(w.q_norm_weight+i)));
                    if (i < FA_ROTARY_DIM) {
                        float fe = float(2*(i%(FA_ROTARY_DIM/2))) / float(FA_ROTARY_DIM);
                        float freq = float(position) / powf(FA_ROPE_THETA, fe);
                        float cv = cosf(freq), sv = sinf(freq);
                        int p = (i < FA_ROTARY_DIM/2) ? i+FA_ROTARY_DIM/2 : i-FA_ROTARY_DIM/2;
                        float pv = qh_ptr[p]*sc*(1.0f+__bfloat162float(__ldg(w.q_norm_weight+p)));
                        qh_ptr[i] = (i < FA_ROTARY_DIM/2) ? (normed*cv-pv*sv) : (pv*sv+normed*cv);
                    } else { qh_ptr[i] = normed; }
                }
            }
        }
    }
    grid.sync();

    // Phase 3: Attention decode (online softmax + sigmoid gate)
    {
        int cache_len = position + 1;
        float attn_scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
        __shared__ float s_max_score[FA_GQA_RATIO * NUM_WARPS];
        __shared__ float s_sum_exp[FA_GQA_RATIO * NUM_WARPS];
        float *s_warp_out = reinterpret_cast<float *>(shmem);
        constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;

        const int segments_per_kv = num_blocks / FA_NUM_KV_HEADS;
        if (segments_per_kv >= 2 && g_attn_partials != nullptr) {
            const int partial_slots = num_blocks * FA_GQA_RATIO;
            float *partial_max = g_attn_partials;
            float *partial_sum = partial_max + partial_slots;
            float *partial_out = partial_sum + partial_slots;

            const int active_blocks = segments_per_kv * FA_NUM_KV_HEADS;
            if (block_id < active_blocks) {
                const int kvh = block_id % FA_NUM_KV_HEADS;
                const int segment_idx = block_id / FA_NUM_KV_HEADS;
                const int segment_size = (cache_len + segments_per_kv - 1) / segments_per_kv;
                const int segment_start = segment_idx * segment_size;
                const int segment_end = min(cache_len, segment_start + segment_size);
                const int qh_base = kvh * FA_GQA_RATIO;

                float max_score[FA_GQA_RATIO];
                float sum_exp[FA_GQA_RATIO];
                float out_acc[FA_GQA_RATIO][EPL];
                float q_local[FA_GQA_RATIO][EPL];

#pragma unroll
                for (int local_q = 0; local_q < FA_GQA_RATIO; ++local_q) {
                    float *q_head = g_q + (qh_base + local_q) * FA_HEAD_DIM * 2;
                    max_score[local_q] = -FLT_MAX;
                    sum_exp[local_q] = 0.0f;
#pragma unroll
                    for (int e = 0; e < EPL; e++) {
                        out_acc[local_q][e] = 0.0f;
                        q_local[local_q][e] = q_head[lane_id*EPL+e];
                    }
                }

                for (int pos = segment_start + warp_id; pos < segment_end; pos += NUM_WARPS) {
                    const __nv_bfloat16 *k_pos = k_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                    const __nv_bfloat16 *v_pos = v_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                    float k_local[EPL], v_local[EPL];
#pragma unroll
                    for (int e = 0; e < EPL; e++) {
                        const int d = lane_id*EPL + e;
                        k_local[e] = __bfloat162float(__ldg(k_pos + d));
                        v_local[e] = __bfloat162float(__ldg(v_pos + d));
                    }

                    float score[FA_GQA_RATIO];
#pragma unroll
                    for (int local_q = 0; local_q < FA_GQA_RATIO; ++local_q) {
                        float dot = 0.0f;
#pragma unroll
                        for (int e = 0; e < EPL; e++) dot += q_local[local_q][e] * k_local[e];
                        dot = warp_reduce_sum(dot) * attn_scale;
                        score[local_q] = __shfl_sync(0xffffffff, dot, 0);
                    }

#pragma unroll
                    for (int local_q = 0; local_q < FA_GQA_RATIO; ++local_q) {
                        float old_max = max_score[local_q];
                        max_score[local_q] = fmaxf(max_score[local_q], score[local_q]);
                        float exp_diff = fast_exp(old_max - max_score[local_q]);
                        float wt = fast_exp(score[local_q] - max_score[local_q]);
                        sum_exp[local_q] = sum_exp[local_q] * exp_diff + wt;
#pragma unroll
                        for (int e = 0; e < EPL; e++)
                            out_acc[local_q][e] = out_acc[local_q][e]*exp_diff + wt*v_local[e];
                    }
                }

#pragma unroll
                for (int local_q = 0; local_q < FA_GQA_RATIO; ++local_q) {
                    const int warp_slot = local_q * NUM_WARPS + warp_id;
                    if (lane_id == 0) {
                        s_max_score[warp_slot] = max_score[local_q];
                        s_sum_exp[warp_slot] = sum_exp[local_q];
                    }
#pragma unroll
                    for (int e = 0; e < EPL; e++) {
                        s_warp_out[warp_slot*FA_HEAD_DIM + lane_id*EPL+e] = out_acc[local_q][e];
                    }
                }
                __syncthreads();

                if (warp_id == 0) {
#pragma unroll
                    for (int local_q = 0; local_q < FA_GQA_RATIO; ++local_q) {
                        const int warp_base = local_q * NUM_WARPS;
                        float gm = -FLT_MAX;
                        for (int ww = 0; ww < NUM_WARPS; ww++) {
                            if (s_sum_exp[warp_base + ww] > 0.0f) gm = fmaxf(gm, s_max_score[warp_base + ww]);
                        }
                        float ts = 0.0f; float fo[EPL]; for (int e = 0; e < EPL; e++) fo[e] = 0.0f;
                        for (int ww = 0; ww < NUM_WARPS; ww++) {
                            if (s_sum_exp[warp_base + ww] > 0.0f) {
                                float s = fast_exp(s_max_score[warp_base + ww] - gm);
                                ts += s_sum_exp[warp_base + ww] * s;
#pragma unroll
                                for (int e = 0; e < EPL; e++) {
                                    fo[e] += s_warp_out[(warp_base + ww)*FA_HEAD_DIM+lane_id*EPL+e] * s;
                                }
                            }
                        }
                        const int partial_slot = block_id * FA_GQA_RATIO + local_q;
                        if (lane_id == 0) {
                            partial_max[partial_slot] = gm;
                            partial_sum[partial_slot] = ts;
                        }
#pragma unroll
                        for (int e = 0; e < EPL; e++) {
                            partial_out[partial_slot*FA_HEAD_DIM + lane_id*EPL+e] = fo[e];
                        }
                    }
                }
            }
            grid.sync();

            if (block_id < FA_NUM_Q_HEADS) {
                const int qh = block_id;
                const int kvh = qh / FA_GQA_RATIO;
                const int local_q = qh - kvh * FA_GQA_RATIO;
                float *q_head = g_q + qh * FA_HEAD_DIM * 2;
                float *out_head = g_attn_out + qh * FA_HEAD_DIM;
                if (threadIdx.x == 0) {
                    float gm = -FLT_MAX;
                    for (int seg = 0; seg < segments_per_kv; ++seg) {
                        const int partial_block = seg * FA_NUM_KV_HEADS + kvh;
                        const int partial_slot = partial_block * FA_GQA_RATIO + local_q;
                        if (partial_sum[partial_slot] > 0.0f) {
                            gm = fmaxf(gm, partial_max[partial_slot]);
                        }
                    }
                    float ts = 0.0f;
                    for (int seg = 0; seg < segments_per_kv; ++seg) {
                        const int partial_block = seg * FA_NUM_KV_HEADS + kvh;
                        const int partial_slot = partial_block * FA_GQA_RATIO + local_q;
                        if (partial_sum[partial_slot] > 0.0f) {
                            ts += partial_sum[partial_slot] * fast_exp(partial_max[partial_slot] - gm);
                        }
                    }
                    s_max_score[0] = gm;
                    s_sum_exp[0] = ts;
                }
                __syncthreads();

                const float gm = s_max_score[0];
                const float rcp = (s_sum_exp[0] > 0.0f) ? (1.0f / s_sum_exp[0]) : 0.0f;
                float *gate_ptr = q_head + FA_HEAD_DIM;
                for (int d = threadIdx.x; d < FA_HEAD_DIM; d += BLOCK_SIZE) {
                    float value = 0.0f;
                    for (int seg = 0; seg < segments_per_kv; ++seg) {
                        const int partial_block = seg * FA_NUM_KV_HEADS + kvh;
                        const int partial_slot = partial_block * FA_GQA_RATIO + local_q;
                        if (partial_sum[partial_slot] > 0.0f) {
                            value += partial_out[partial_slot*FA_HEAD_DIM + d] * fast_exp(partial_max[partial_slot] - gm);
                        }
                    }
                    out_head[d] = value * rcp * fast_sigmoid(gate_ptr[d]);
                }
            }
        } else {
            int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
            int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);

            for (int qh = hs; qh < he; qh++) {
                int kvh = qh / FA_GQA_RATIO;
                float *q_head = g_q + qh * FA_HEAD_DIM * 2;
                float *out_head = g_attn_out + qh * FA_HEAD_DIM;
                float max_score = -FLT_MAX, sum_exp = 0.0f;
                float out_acc[EPL], q_local[EPL];
                for (int e = 0; e < EPL; e++) { out_acc[e] = 0.0f; q_local[e] = q_head[lane_id*EPL+e]; }

                for (int pos = warp_id; pos < cache_len; pos += NUM_WARPS) {
                    const __nv_bfloat16 *k_pos = k_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                    const __nv_bfloat16 *v_pos = v_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                    float score = 0.0f;
                    for (int e = 0; e < EPL; e++) score += q_local[e] * __bfloat162float(__ldg(k_pos + lane_id*EPL+e));
                    score = warp_reduce_sum(score) * attn_scale;
                    score = __shfl_sync(0xffffffff, score, 0);
                    float old_max = max_score; max_score = fmaxf(max_score, score);
                    float exp_diff = fast_exp(old_max - max_score);
                    sum_exp = sum_exp * exp_diff + fast_exp(score - max_score);
                    float wt = fast_exp(score - max_score);
                    for (int e = 0; e < EPL; e++)
                        out_acc[e] = out_acc[e]*exp_diff + wt*__bfloat162float(__ldg(v_pos + lane_id*EPL+e));
                }
                if (lane_id == 0) { s_max_score[warp_id] = max_score; s_sum_exp[warp_id] = sum_exp; }
                for (int e = 0; e < EPL; e++) s_warp_out[warp_id*FA_HEAD_DIM + lane_id*EPL+e] = out_acc[e];
                __syncthreads();

                if (warp_id == 0) {
                    float gm = -FLT_MAX; for (int ww = 0; ww < NUM_WARPS; ww++) if (s_sum_exp[ww] > 0.0f) gm = fmaxf(gm, s_max_score[ww]);
                    float ts = 0.0f; float fo[EPL]; for (int e = 0; e < EPL; e++) fo[e] = 0.0f;
                    for (int ww = 0; ww < NUM_WARPS; ww++) {
                        if (s_sum_exp[ww] > 0.0f) {
                            float s = fast_exp(s_max_score[ww]-gm); ts += s_sum_exp[ww]*s;
                            for (int e = 0; e < EPL; e++) fo[e] += s_warp_out[ww*FA_HEAD_DIM+lane_id*EPL+e]*s;
                        }
                    }
                    float *gate_ptr = q_head + FA_HEAD_DIM;
                    float rcp = (ts > 0.0f) ? (1.0f / ts) : 0.0f;
                    for (int e = 0; e < EPL; e++) {
                        int idx = lane_id*EPL+e;
                        out_head[idx] = fo[e]*rcp * fast_sigmoid(gate_ptr[idx]);
                    }
                }
                __syncthreads();
            }
        }
    }
    grid.sync();

    // Phase 4: O projection + residual → bf16
    {
        float *s_attn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < FA_Q_SIZE; i += BLOCK_SIZE) s_attn[i] = g_attn_out[i];
        __syncthreads();
        if (qw != nullptr && qw->ptrs[6].packed_weight != nullptr) {
            matvec_o_residual_nvfp4(s_attn, qw->ptrs[6], g_residual, hidden_out, FA_Q_SIZE, HIDDEN_SIZE, num_blocks);
        } else {
            matvec_o_residual_bf16(s_attn, w.o_proj_weight, g_residual, hidden_out, FA_Q_SIZE, HIDDEN_SIZE, num_blocks);
        }
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    __nv_bfloat16 *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    if (qw != nullptr && qw->ptrs[8].packed_weight != nullptr && qw->ptrs[9].packed_weight != nullptr) {
        matvec_gate_up_silu_nvfp4(s_act, qw->ptrs[8], qw->ptrs[9],
                                  g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    } else {
        matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                                  g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    }
    grid.sync();

    // Load MLP intermediate to shared (f32)
    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();

    if (qw != nullptr && qw->ptrs[10].packed_weight != nullptr) {
        matvec_down_residual_nvfp4(s_mlp, qw->ptrs[10], g_residual, hidden_out,
                                   INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    } else {
        matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                                   INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();
}

// =============================================================================
// DeltaNet layer (bf16) — warp-cooperative state-in-registers recurrence
// =============================================================================

__device__ void deltanet_layer(
    AtomicGridSync &grid,
    const DeltaNetWeights &w,
    const LayerNvfp4Weights *__restrict__ qw,
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_qkv,
    float *__restrict__ g_z,
    float *__restrict__ g_beta,
    float *__restrict__ g_alpha,
    float *__restrict__ g_dn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ dn_state,     // [DN_NUM_HEADS, DN_KEY, DN_VAL] f32
    float *__restrict__ conv_buf,     // [DN_CONV_CH, DN_CONV_K] f32
    __nv_bfloat16 *__restrict__ hidden_out,
    int dn_layer_idx,
    __nv_bfloat16 *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + projections
    __nv_bfloat16 *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_select_bf16_or_nvfp4(s_norm, w.qkv_proj_weight, qw, 1, g_qkv, HIDDEN_SIZE, DN_CONV_CHANNELS, num_blocks);
    matvec_select_bf16_or_nvfp4(s_norm, w.z_proj_weight, qw, 2, g_z, HIDDEN_SIZE, DN_V_SIZE, num_blocks);
    matvec_select_bf16_or_nvfp4(s_norm, w.beta_proj_weight, qw, 3, g_beta, HIDDEN_SIZE, DN_GATE_HEADS, num_blocks);
    matvec_select_bf16_or_nvfp4(s_norm, w.alpha_proj_weight, qw, 4, g_alpha, HIDDEN_SIZE, DN_GATE_HEADS, num_blocks);
    grid.sync();

    if (block_id < DN_GATE_HEADS) {
        int gh = block_id;
        if (threadIdx.x == 0) {
            g_beta[gh] = fast_sigmoid(g_beta[gh]);
            float a_log_val = __bfloat162float(__ldg(w.a_log + gh));
            float dt_b = __bfloat162float(__ldg(w.dt_bias + gh));
            float x = g_alpha[gh] + dt_b;
            float sp = (x > 20.0f) ? x : logf(1.0f + fast_exp(x));
            g_alpha[gh] = fast_exp(-fast_exp(a_log_val) * sp);
        }
    }
    grid.sync();

    // Phase 2+3: Conv1d + recurrence (blocks 0-15 only)
    if (block_id < DN_NUM_HEADS) {
        int h = block_id;
        float *layer_conv = conv_buf + dn_layer_idx * DN_CONV_CHANNELS * DN_CONV_KERNEL;

        // Conv1d + SiLU
        __shared__ float s_q[DN_KEY_DIM], s_k[DN_KEY_DIM], s_v[DN_VALUE_DIM];
        int head_ch[3] = {h*DN_KEY_DIM, DN_QK_SIZE+h*DN_KEY_DIM, 2*DN_QK_SIZE+h*DN_VALUE_DIM};
        for (int region = 0; region < 3; region++) {
            int ch_base = head_ch[region], ch_count = (region < 2) ? DN_KEY_DIM : DN_VALUE_DIM;
            float *dst = (region == 0) ? s_q : (region == 1) ? s_k : s_v;
            for (int c = threadIdx.x; c < ch_count; c += BLOCK_SIZE) {
                int ch = ch_base + c;
                float h0=layer_conv[ch*DN_CONV_KERNEL+1], h1=layer_conv[ch*DN_CONV_KERNEL+2], h2=layer_conv[ch*DN_CONV_KERNEL+3];
                layer_conv[ch*DN_CONV_KERNEL]=h0; layer_conv[ch*DN_CONV_KERNEL+1]=h1;
                layer_conv[ch*DN_CONV_KERNEL+2]=h2; layer_conv[ch*DN_CONV_KERNEL+3]=g_qkv[ch];
                float co = 0;
                for (int t = 0; t < DN_CONV_KERNEL; t++)
                    co += layer_conv[ch*DN_CONV_KERNEL+t] * __bfloat162float(__ldg(w.conv1d_weight + ch*DN_CONV_KERNEL+t));
                dst[c] = fast_silu(co);
            }
        }

        // L2 normalize Q, K
        constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
        if (warp_id == 0) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_q[i]*s_q[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f)*Q_SCALE;
            n = __shfl_sync(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_q[i] *= n;
        }
        if (warp_id == 1) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_k[i]*s_k[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f);
            n = __shfl_sync(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_k[i] *= n;
        }
        __syncthreads();

        // k·q dot
        __shared__ float s_kq;
        if (warp_id == 0) {
            float kq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) kq += s_k[i]*s_q[i];
            kq = warp_reduce_sum(kq); if (lane_id == 0) s_kq = kq;
        }
        __syncthreads();
        float kq = s_kq;

        // Warp-cooperative recurrence (state in global memory — decode is 1 token, fine)
        float *state = dn_state + h * DN_KEY_DIM * DN_VALUE_DIM;
        float *out_head = g_dn_out + h * DN_VALUE_DIM;

        constexpr int J_PER_WARP = DN_VALUE_DIM / NUM_WARPS;
        constexpr int I_PER_LANE = DN_KEY_DIM / WARP_SIZE;

#pragma unroll
        for (int jj = 0; jj < J_PER_WARP; jj++) {
            int j = warp_id * J_PER_WARP + jj;
            float s_regs[I_PER_LANE], stk = 0, sqv = 0;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                float sv = state[j*DN_KEY_DIM+i]; s_regs[ii] = sv;
                stk += sv * s_k[i]; sqv += sv * s_q[i];
            }
            stk = warp_reduce_sum(stk); sqv = warp_reduce_sum(sqv);
            stk = __shfl_sync(0xffffffff,stk,0); sqv = __shfl_sync(0xffffffff,sqv,0);
            int gate_head = h * DN_VALUE_GROUPS + j / DN_VALUE_HEAD_DIM;
            float decay = g_alpha[gate_head], beta = g_beta[gate_head];
            float error_j = (s_v[j] - decay * stk) * beta;
            float o_j = decay * sqv + error_j * kq;
            if (lane_id == 0) out_head[j] = o_j;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                state[j*DN_KEY_DIM+i] = s_regs[ii] * decay + s_k[i] * error_j;
            }
        }

        // Gated RMSNorm
        __syncthreads();
        {
            __shared__ float smem_gnorm[DN_VALUE_GROUPS][NUM_WARPS];
            __shared__ float smem_rstd[DN_VALUE_GROUPS];
            for (int group = 0; group < DN_VALUE_GROUPS; ++group) {
                const int group_base = group * DN_VALUE_HEAD_DIM;
                float sq = 0;
                for (int i = threadIdx.x; i < DN_VALUE_HEAD_DIM; i += BLOCK_SIZE) {
                    float v = out_head[group_base + i];
                    sq += v * v;
                }
                sq = warp_reduce_sum(sq);
                if (lane_id == 0) smem_gnorm[group][warp_id] = sq;
                __syncthreads();
                if (warp_id == 0) {
                    float v = (lane_id < NUM_WARPS) ? smem_gnorm[group][lane_id] : 0;
                    v = warp_reduce_sum(v);
                    if (lane_id == 0) smem_rstd[group] = rsqrtf(v / DN_VALUE_HEAD_DIM + RMS_EPS);
                }
                __syncthreads();
                float rstd = smem_rstd[group];
                for (int i = threadIdx.x; i < DN_VALUE_HEAD_DIM; i += BLOCK_SIZE) {
                    int value_idx = group_base + i;
                    float normed = out_head[value_idx] * rstd * __bfloat162float(__ldg(w.norm_weight + i));
                    float gate = fast_silu(g_z[h * DN_VALUE_DIM + value_idx]);
                    out_head[value_idx] = normed * gate;
                }
                __syncthreads();
            }
        }
    } else {
        // Idle blocks: could prefetch weights
    }
    grid.sync();

    // Phase 4: Out projection + residual → bf16
    {
        float *s_dn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < DN_V_SIZE; i += BLOCK_SIZE) s_dn[i] = g_dn_out[i];
        __syncthreads();
        if (qw != nullptr && qw->ptrs[9].packed_weight != nullptr) {
            matvec_o_residual_nvfp4(s_dn, qw->ptrs[9], g_residual, hidden_out, DN_V_SIZE, HIDDEN_SIZE, num_blocks);
        } else {
            matvec_o_residual_bf16(s_dn, w.out_proj_weight, g_residual, hidden_out, DN_V_SIZE, HIDDEN_SIZE, num_blocks);
        }
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    __nv_bfloat16 *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    if (qw != nullptr && qw->ptrs[11].packed_weight != nullptr && qw->ptrs[12].packed_weight != nullptr) {
        matvec_gate_up_silu_nvfp4(s_act, qw->ptrs[11], qw->ptrs[12],
                                  g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    } else {
        matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                                  g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    }
    grid.sync();

    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();
    if (qw != nullptr && qw->ptrs[13].packed_weight != nullptr) {
        matvec_down_residual_nvfp4(s_mlp, qw->ptrs[13], g_residual, hidden_out,
                                   INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    } else {
        matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                                   INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();
}

