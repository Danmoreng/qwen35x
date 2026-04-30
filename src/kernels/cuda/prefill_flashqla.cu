/**
 * FlashQLA-style DeltaNet prefill kernels.
 *
 * These kernels intentionally use the usual BF16/FP32 path around the
 * recurrence buffers. They do not depend on NVFP4 or SM120-only FP4 assembly.
 */

#include "prefill_flashqla.cuh"

#include "common.cuh"
#include "variant.cuh"

#include <cstddef>
#include <mma.h>

namespace wmma = nvcuda::wmma;

__global__ void __launch_bounds__(32, 2)
pf_deltanet_recurrence_flashqla64(
    const float *qkv_f32, const float *beta_buf, const float *alpha_buf,
    float *state, float *output, int S)
{
    constexpr int CHUNK = 64;
    constexpr int RPL = DN_KEY / 32;

    const int h = blockIdx.x;
    const int col = blockIdx.y;
    const int lid = threadIdx.x;
    if (h >= DN_HEADS || col >= DN_VAL || lid >= 32) return;

    __shared__ float k_shared[CHUNK * DN_KEY];
    __shared__ float chunk_decay_prefix[CHUNK];
    __shared__ float chunk_beta[CHUNK];
    __shared__ float delta[CHUNK];

    float sreg[RPL];
    float *state_col = state + h * DN_KEY * DN_VAL + col * DN_KEY;

#pragma unroll
    for (int r = 0; r < RPL; ++r) {
        sreg[r] = state_col[lid + r * 32];
    }

    const int gate_head = h * DN_VAL_GROUPS + col / DN_VAL_HEAD_DIM;

    for (int chunk_start = 0; chunk_start < S; chunk_start += CHUNK) {
        const int rows = min(CHUNK, S - chunk_start);

        for (int idx = lid; idx < rows * DN_KEY; idx += 32) {
            const int t = idx / DN_KEY;
            const int k = idx - t * DN_KEY;
            k_shared[idx] = qkv_f32[(chunk_start + t) * DN_CONV_CH + DN_QK_SIZE + h * DN_KEY + k];
        }

        if (lid == 0) {
            float prefix = 1.0f;
            for (int t = 0; t < rows; ++t) {
                const int gate_off = (chunk_start + t) * DN_GATE + gate_head;
                prefix *= fmaxf(alpha_buf[gate_off], 1.0e-20f);
                chunk_decay_prefix[t] = prefix;
                chunk_beta[t] = beta_buf[gate_off];
                delta[t] = 0.0f;
            }
        }
        __syncthreads();

        for (int t = 0; t < rows; ++t) {
            const float *kt = k_shared + t * DN_KEY;
            float k_dot_s0 = 0.0f;
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                k_dot_s0 += kt[lid + r * 32] * sreg[r];
            }
            k_dot_s0 = pf_warp_sum(k_dot_s0);

            float correction = 0.0f;
            for (int i = 0; i < t; ++i) {
                const float *ki = k_shared + i * DN_KEY;
                float k_dot_k = 0.0f;
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    const int d = lid + r * 32;
                    k_dot_k += kt[d] * ki[d];
                }
                k_dot_k = pf_warp_sum(k_dot_k);
                if (lid == 0) {
                    const float ratio = chunk_decay_prefix[t] / fmaxf(chunk_decay_prefix[i], 1.0e-20f);
                    correction += ratio * k_dot_k * delta[i];
                }
            }

            if (lid == 0) {
                const float *v = qkv_f32 + (chunk_start + t) * DN_CONV_CH + 2 * DN_QK_SIZE + h * DN_VAL;
                delta[t] = chunk_beta[t] * (v[col] - chunk_decay_prefix[t] * k_dot_s0 - correction);
            }
            __syncthreads();
        }

        for (int t = 0; t < rows; ++t) {
            const float *qt = qkv_f32 + (chunk_start + t) * DN_CONV_CH + h * DN_KEY;
            float q_dot_state = 0.0f;
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                q_dot_state += qt[lid + r * 32] * sreg[r];
            }
            q_dot_state = pf_warp_sum(q_dot_state);

            float out_sum = 0.0f;
            for (int i = 0; i <= t; ++i) {
                const float *ki = k_shared + i * DN_KEY;
                float q_dot_k = 0.0f;
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    const int d = lid + r * 32;
                    q_dot_k += qt[d] * ki[d];
                }
                q_dot_k = pf_warp_sum(q_dot_k);
                if (lid == 0) {
                    const float ratio = (i == t) ? 1.0f : (chunk_decay_prefix[t] / fmaxf(chunk_decay_prefix[i], 1.0e-20f));
                    out_sum += ratio * q_dot_k * delta[i];
                }
            }
            if (lid == 0) {
                output[(chunk_start + t) * DN_V_SIZE + h * DN_VAL + col] =
                    chunk_decay_prefix[t] * q_dot_state + out_sum;
            }
            __syncthreads();
        }

        const float last_prefix = chunk_decay_prefix[rows - 1];
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            const int d = lid + r * 32;
            float new_state = last_prefix * sreg[r];
            for (int i = 0; i < rows; ++i) {
                const float ratio = (i == rows - 1) ? 1.0f : (last_prefix / fmaxf(chunk_decay_prefix[i], 1.0e-20f));
                new_state += ratio * k_shared[i * DN_KEY + d] * delta[i];
            }
            sreg[r] = new_state;
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < RPL; ++r) {
        state_col[lid + r * 32] = sreg[r];
    }
}

__global__ void __launch_bounds__(256, 1)
pf_deltanet_recurrence_flashqla64_tiled(
    const float *qkv_f32, const float *beta_buf, const float *alpha_buf,
    float *state, float *output, int S)
{
    constexpr int CHUNK = 64;
    constexpr int COLS = 8;
    constexpr int RPL = DN_KEY / 32;

    const int h = blockIdx.x;
    const int col_base = blockIdx.y * COLS;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;
    if (h >= DN_HEADS || col_base >= DN_VAL) return;

    extern __shared__ float flashqla_smem[];
    float *k_shared = flashqla_smem;
    float *kk_shared = k_shared + CHUNK * DN_KEY;
    float *qk_shared = kk_shared + CHUNK * CHUNK;
    float *chunk_decay_prefix = qk_shared + CHUNK * CHUNK;
    float *chunk_beta = chunk_decay_prefix + CHUNK * COLS;
    float *delta = chunk_beta + CHUNK * COLS;

    const int col = col_base + warp;
    float sreg[RPL] = {};
    if (col < DN_VAL) {
        float *state_col = state + h * DN_KEY * DN_VAL + col * DN_KEY;
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            sreg[r] = state_col[lane + r * 32];
        }
    }

    for (int chunk_start = 0; chunk_start < S; chunk_start += CHUNK) {
        const int rows = min(CHUNK, S - chunk_start);

        for (int idx = tid; idx < rows * DN_KEY; idx += blockDim.x) {
            const int t = idx / DN_KEY;
            const int k = idx - t * DN_KEY;
            k_shared[idx] = qkv_f32[(chunk_start + t) * DN_CONV_CH + DN_QK_SIZE + h * DN_KEY + k];
        }

        for (int idx = tid; idx < rows * COLS; idx += blockDim.x) {
            const int t = idx / COLS;
            const int c = idx - t * COLS;
            const int vc = col_base + c;
            if (vc < DN_VAL) {
                const int gate_head = h * DN_VAL_GROUPS + vc / DN_VAL_HEAD_DIM;
                chunk_beta[idx] = beta_buf[(chunk_start + t) * DN_GATE + gate_head];
            } else {
                chunk_beta[idx] = 0.0f;
            }
            delta[idx] = 0.0f;
        }
        __syncthreads();

        if (tid < COLS) {
            const int vc = col_base + tid;
            float prefix = 1.0f;
            for (int t = 0; t < rows; ++t) {
                if (vc < DN_VAL) {
                    const int gate_head = h * DN_VAL_GROUPS + vc / DN_VAL_HEAD_DIM;
                    const int gate_off = (chunk_start + t) * DN_GATE + gate_head;
                    prefix *= fmaxf(alpha_buf[gate_off], 1.0e-20f);
                }
                chunk_decay_prefix[t * COLS + tid] = prefix;
            }
        }
        __syncthreads();

        for (int idx = warp; idx < rows * rows; idx += blockDim.x / 32) {
            const int t = idx / rows;
            const int i = idx - t * rows;
            float kk = 0.0f;
            float qk = 0.0f;
            const float *kt = k_shared + t * DN_KEY;
            const float *ki = k_shared + i * DN_KEY;
            const float *qt = qkv_f32 + (chunk_start + t) * DN_CONV_CH + h * DN_KEY;
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                const int d = lane + r * 32;
                const float kid = ki[d];
                kk += kt[d] * kid;
                qk += qt[d] * kid;
            }
            kk = pf_warp_sum(kk);
            qk = pf_warp_sum(qk);
            if (lane == 0) {
                kk_shared[t * CHUNK + i] = kk;
                qk_shared[t * CHUNK + i] = qk;
            }
        }
        __syncthreads();

        if (col < DN_VAL) {
            const int c = warp;
            for (int t = 0; t < rows; ++t) {
                const float *kt = k_shared + t * DN_KEY;
                float k_dot_s0 = 0.0f;
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    k_dot_s0 += kt[lane + r * 32] * sreg[r];
                }
                k_dot_s0 = pf_warp_sum(k_dot_s0);

                if (lane == 0) {
                    float correction = 0.0f;
                    for (int i = 0; i < t; ++i) {
                        const float ratio = chunk_decay_prefix[t * COLS + c] / fmaxf(chunk_decay_prefix[i * COLS + c], 1.0e-20f);
                        correction += ratio * kk_shared[t * CHUNK + i] * delta[i * COLS + c];
                    }
                    const float *v = qkv_f32 + (chunk_start + t) * DN_CONV_CH + 2 * DN_QK_SIZE + h * DN_VAL;
                    delta[t * COLS + c] =
                        chunk_beta[t * COLS + c] * (v[col] - chunk_decay_prefix[t * COLS + c] * k_dot_s0 - correction);
                }
                __syncwarp();
            }

            for (int t = 0; t < rows; ++t) {
                const float *qt = qkv_f32 + (chunk_start + t) * DN_CONV_CH + h * DN_KEY;
                float q_dot_state = 0.0f;
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    q_dot_state += qt[lane + r * 32] * sreg[r];
                }
                q_dot_state = pf_warp_sum(q_dot_state);

                if (lane == 0) {
                    float out_sum = 0.0f;
                    for (int i = 0; i <= t; ++i) {
                        const float ratio = (i == t) ? 1.0f : (chunk_decay_prefix[t * COLS + c] / fmaxf(chunk_decay_prefix[i * COLS + c], 1.0e-20f));
                        out_sum += ratio * qk_shared[t * CHUNK + i] * delta[i * COLS + c];
                    }
                    output[(chunk_start + t) * DN_V_SIZE + h * DN_VAL + col] =
                        chunk_decay_prefix[t * COLS + c] * q_dot_state + out_sum;
                }
                __syncwarp();
            }

            const float last_prefix = chunk_decay_prefix[(rows - 1) * COLS + c];
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                const int d = lane + r * 32;
                float new_state = last_prefix * sreg[r];
                for (int i = 0; i < rows; ++i) {
                    const float ratio = (i == rows - 1) ? 1.0f : (last_prefix / fmaxf(chunk_decay_prefix[i * COLS + c], 1.0e-20f));
                    new_state += ratio * k_shared[i * DN_KEY + d] * delta[i * COLS + c];
                }
                sreg[r] = new_state;
            }
        }
        __syncthreads();
    }

    if (col < DN_VAL) {
        float *state_col = state + h * DN_KEY * DN_VAL + col * DN_KEY;
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            state_col[lane + r * 32] = sreg[r];
        }
    }
}

__global__ void __launch_bounds__(32, 1)
pf_deltanet_flashqla64_cuda_ref(
    const float *qkv_f32, const float *beta_buf, const float *alpha_buf,
    float *state, float *output, int S)
{
    constexpr int CHUNK = 64;
    constexpr int RPL = DN_KEY / 32;

    const int h = blockIdx.x;
    const int col = blockIdx.y;
    const int lid = threadIdx.x;
    if (h >= DN_HEADS || col >= DN_VAL || lid >= 32) return;

    extern __shared__ float smem[];
    float *k_shared = smem;
    float *lower_shared = k_shared + CHUNK * DN_KEY;
    float *a_shared = lower_shared + CHUNK * CHUNK;
    float *g_shared = a_shared + CHUNK * CHUNK;
    float *beta_shared = g_shared + CHUNK;
    float *vnew_shared = beta_shared + CHUNK;

    float sreg[RPL];
    float *state_col = state + h * DN_KEY * DN_VAL + col * DN_KEY;

#pragma unroll
    for (int r = 0; r < RPL; ++r) {
        sreg[r] = state_col[lid + r * 32];
    }

    const int gate_head = h * DN_VAL_GROUPS + col / DN_VAL_HEAD_DIM;

    for (int chunk_start = 0; chunk_start < S; chunk_start += CHUNK) {
        const int rows = min(CHUNK, S - chunk_start);

        for (int idx = lid; idx < rows * DN_KEY; idx += 32) {
            const int t = idx / DN_KEY;
            const int k = idx - t * DN_KEY;
            k_shared[idx] = qkv_f32[(chunk_start + t) * DN_CONV_CH + DN_QK_SIZE + h * DN_KEY + k];
        }

        if (lid == 0) {
            float g = 0.0f;
            for (int t = 0; t < rows; ++t) {
                const int gate_off = (chunk_start + t) * DN_GATE + gate_head;
                g += logf(fmaxf(alpha_buf[gate_off], 1.0e-20f));
                g_shared[t] = g;
                beta_shared[t] = beta_buf[gate_off];
                vnew_shared[t] = 0.0f;
            }
        }
        __syncthreads();

        for (int idx = lid; idx < rows * rows; idx += 32) {
            const int t = idx / rows;
            const int j = idx - t * rows;
            float lower = 0.0f;
            if (j < t) {
                const float *kt = k_shared + t * DN_KEY;
                const float *kj = k_shared + j * DN_KEY;
                float dot = 0.0f;
#pragma unroll
                for (int d = 0; d < DN_KEY; ++d) {
                    dot += kt[d] * kj[d];
                }
                lower = beta_shared[t] * expf(g_shared[t] - g_shared[j]) * dot;
            }
            lower_shared[t * CHUNK + j] = lower;
            a_shared[t * CHUNK + j] = (t == j) ? 1.0f : 0.0f;
        }
        __syncthreads();

        if (lid == 0) {
            for (int t = 1; t < rows; ++t) {
                for (int j = 0; j < t; ++j) {
                    float sum = lower_shared[t * CHUNK + j];
                    for (int m = j + 1; m < t; ++m) {
                        sum += lower_shared[t * CHUNK + m] * a_shared[m * CHUNK + j];
                    }
                    a_shared[t * CHUNK + j] = -sum;
                }
            }
        }
        __syncthreads();

        for (int t = 0; t < rows; ++t) {
            float u = 0.0f;
            float w_dot_state = 0.0f;
            for (int j = 0; j <= t; ++j) {
                const float a = a_shared[t * CHUNK + j];
                const float beta = beta_shared[j];
                const float *kj = k_shared + j * DN_KEY;
                float k_dot_state = 0.0f;
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    k_dot_state += kj[lid + r * 32] * sreg[r];
                }
                k_dot_state = pf_warp_sum(k_dot_state);
                if (lid == 0) {
                    const float *v = qkv_f32 + (chunk_start + j) * DN_CONV_CH + 2 * DN_QK_SIZE + h * DN_VAL;
                    u += a * beta * v[col];
                    w_dot_state += a * beta * expf(g_shared[j]) * k_dot_state;
                }
            }
            if (lid == 0) {
                vnew_shared[t] = u - w_dot_state;
            }
            __syncthreads();
        }

        for (int t = 0; t < rows; ++t) {
            const float *qt = qkv_f32 + (chunk_start + t) * DN_CONV_CH + h * DN_KEY;
            float q_dot_state = 0.0f;
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                q_dot_state += qt[lid + r * 32] * sreg[r];
            }
            q_dot_state = pf_warp_sum(q_dot_state);

            float intra = 0.0f;
            for (int j = 0; j <= t; ++j) {
                const float *kj = k_shared + j * DN_KEY;
                float q_dot_k = 0.0f;
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    const int d = lid + r * 32;
                    q_dot_k += qt[d] * kj[d];
                }
                q_dot_k = pf_warp_sum(q_dot_k);
                if (lid == 0) {
                    intra += expf(g_shared[t] - g_shared[j]) * q_dot_k * vnew_shared[j];
                }
            }

            if (lid == 0) {
                output[(chunk_start + t) * DN_V_SIZE + h * DN_VAL + col] =
                    expf(g_shared[t]) * q_dot_state + intra;
            }
            __syncthreads();
        }

        const float g_last = g_shared[rows - 1];
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            const int d = lid + r * 32;
            float next_state = expf(g_last) * sreg[r];
            for (int t = 0; t < rows; ++t) {
                next_state += k_shared[t * DN_KEY + d] * expf(g_last - g_shared[t]) * vnew_shared[t];
            }
            sreg[r] = next_state;
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < RPL; ++r) {
        state_col[lid + r * 32] = sreg[r];
    }
}

static __global__ void __launch_bounds__(512, 1)
pf_deltanet_flashqla64_tc_prepare(
    const float *qkv_f32, const float *beta_buf, const float *alpha_buf,
    int S, int num_chunks, void *workspace)
{
    constexpr int CHUNK = 64;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int TILES = CHUNK / WMMA_M;

    const int h = blockIdx.x;
    const int group = blockIdx.y;
    const int chunk_id = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    if (h >= DN_HEADS || group >= DN_VAL_GROUPS || chunk_id >= num_chunks) return;

    extern __shared__ float smem[];
    float *kk_shared = smem;
    float *qk_shared = kk_shared + CHUNK * CHUNK;
    float *a_shared = qk_shared + CHUNK * CHUNK;
    float *g_shared = a_shared + CHUNK * CHUNK;
    float *beta_shared = g_shared + CHUNK;
    __nv_bfloat16 *q_bf16 = reinterpret_cast<__nv_bfloat16 *>(beta_shared + CHUNK);
    __nv_bfloat16 *k_bf16 = q_bf16 + CHUNK * DN_KEY;

    const std::size_t head_chunks = static_cast<std::size_t>(DN_GATE) * static_cast<std::size_t>(num_chunks);
    __nv_bfloat16 *a_ws = reinterpret_cast<__nv_bfloat16 *>(workspace);
    __nv_bfloat16 *p_ws = a_ws + head_chunks * CHUNK * CHUNK;
    float *g_ws = reinterpret_cast<float *>(p_ws + head_chunks * CHUNK * CHUNK);
    float *beta_ws = g_ws + head_chunks * CHUNK;
    const std::size_t ws_id =
        (static_cast<std::size_t>(h) * DN_VAL_GROUPS + group) * static_cast<std::size_t>(num_chunks) + chunk_id;
    __nv_bfloat16 *a_out = a_ws + ws_id * CHUNK * CHUNK;
    __nv_bfloat16 *p_out = p_ws + ws_id * CHUNK * CHUNK;
    float *g_out = g_ws + ws_id * CHUNK;
    float *beta_out = beta_ws + ws_id * CHUNK;

    const int chunk_start = chunk_id * CHUNK;
    const int rows = min(CHUNK, S - chunk_start);
    const int gate_head = h * DN_VAL_GROUPS + group;

    for (int idx = tid; idx < CHUNK * DN_KEY; idx += blockDim.x) {
        const int t = idx / DN_KEY;
        const int k = idx - t * DN_KEY;
        float qv = 0.0f;
        float kv = 0.0f;
        if (t < rows) {
            const float *base = qkv_f32 + (chunk_start + t) * DN_CONV_CH;
            qv = base[h * DN_KEY + k];
            kv = base[DN_QK_SIZE + h * DN_KEY + k];
        }
        q_bf16[idx] = __float2bfloat16(qv);
        k_bf16[idx] = __float2bfloat16(kv);
    }

    if (tid == 0) {
        float g = 0.0f;
        for (int t = 0; t < CHUNK; ++t) {
            float gt = 0.0f;
            float bt = 0.0f;
            if (t < rows) {
                const int gate_off = (chunk_start + t) * DN_GATE + gate_head;
                g += logf(fmaxf(alpha_buf[gate_off], 1.0e-20f));
                gt = g;
                bt = beta_buf[gate_off];
            }
            g_shared[t] = gt;
            beta_shared[t] = bt;
            g_out[t] = gt;
            beta_out[t] = bt;
        }
    }
    __syncthreads();

    for (int tile = warp; tile < TILES * TILES; tile += blockDim.x / 32) {
        const int tile_m = tile / TILES;
        const int tile_n = tile - tile_m * TILES;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> k_a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> k_b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> kk_acc;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> qk_acc;
        wmma::fill_fragment(kk_acc, 0.0f);
        wmma::fill_fragment(qk_acc, 0.0f);

        for (int k0 = 0; k0 < DN_KEY; k0 += WMMA_K) {
            const __nv_bfloat16 *q_tile = q_bf16 + (tile_m * WMMA_M) * DN_KEY + k0;
            const __nv_bfloat16 *k_a_tile = k_bf16 + (tile_m * WMMA_M) * DN_KEY + k0;
            const __nv_bfloat16 *k_b_tile = k_bf16 + (tile_n * WMMA_N) * DN_KEY + k0;
            wmma::load_matrix_sync(q_frag, q_tile, DN_KEY);
            wmma::load_matrix_sync(k_a_frag, k_a_tile, DN_KEY);
            wmma::load_matrix_sync(k_b_frag, k_b_tile, DN_KEY);
            wmma::mma_sync(qk_acc, q_frag, k_b_frag, qk_acc);
            wmma::mma_sync(kk_acc, k_a_frag, k_b_frag, kk_acc);
        }

        wmma::store_matrix_sync(kk_shared + (tile_m * WMMA_M) * CHUNK + tile_n * WMMA_N, kk_acc, CHUNK, wmma::mem_row_major);
        wmma::store_matrix_sync(qk_shared + (tile_m * WMMA_M) * CHUNK + tile_n * WMMA_N, qk_acc, CHUNK, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < CHUNK * CHUNK; idx += blockDim.x) {
        const int t = idx / CHUNK;
        const int j = idx - t * CHUNK;
        float lower = 0.0f;
        if (t < rows && j < t) {
            lower = beta_shared[t] * expf(g_shared[t] - g_shared[j]) * kk_shared[t * CHUNK + j];
        }
        kk_shared[t * CHUNK + j] = lower;
        a_shared[t * CHUNK + j] = (j == t) ? 1.0f : 0.0f;
    }
    __syncthreads();

    for (int t = 1; t < rows; ++t) {
        for (int j = tid; j < t; j += blockDim.x) {
            float sum = kk_shared[t * CHUNK + j];
            for (int m = j + 1; m < t; ++m) {
                sum += kk_shared[t * CHUNK + m] * a_shared[m * CHUNK + j];
            }
            a_shared[t * CHUNK + j] = -sum;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < CHUNK * CHUNK; idx += blockDim.x) {
        const int t = idx / CHUNK;
        const int j = idx - t * CHUNK;
        float p = 0.0f;
        if (t < rows && j <= t) {
            p = expf(g_shared[t] - g_shared[j]) * qk_shared[t * CHUNK + j];
        }
        a_out[idx] = __float2bfloat16(a_shared[idx]);
        p_out[idx] = __float2bfloat16(p);
    }
}

__global__ void __launch_bounds__(1024, 1)
pf_deltanet_recurrence_flashqla64_tc_tiled(
    const float *qkv_f32, const float *beta_buf, const float *alpha_buf,
    float *state, float *output, int S, int num_chunks, const void *workspace)
{
    constexpr int CHUNK = 64;
    constexpr int COLS = 32;
    constexpr int RPL = DN_KEY / 32;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int TILES = CHUNK / WMMA_M;

    const int h = blockIdx.x;
    const int col_base = blockIdx.y * COLS;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;
    if (h >= DN_HEADS || col_base >= DN_VAL) return;

    extern __shared__ float smem[];
    float *vnew_shared = smem;
    float *intra_shared = vnew_shared + CHUNK * COLS;
    float *g_shared = intra_shared + CHUNK * COLS;
    float *beta_shared = g_shared + CHUNK;
    __nv_bfloat16 *w_bf16 = reinterpret_cast<__nv_bfloat16 *>(beta_shared + CHUNK);

    const std::size_t head_chunks = static_cast<std::size_t>(DN_GATE) * static_cast<std::size_t>(num_chunks);
    const __nv_bfloat16 *a_ws = reinterpret_cast<const __nv_bfloat16 *>(workspace);
    const __nv_bfloat16 *p_ws = a_ws + head_chunks * CHUNK * CHUNK;
    const float *g_ws = reinterpret_cast<const float *>(p_ws + head_chunks * CHUNK * CHUNK);
    const float *beta_ws = g_ws + head_chunks * CHUNK;

    const int col = col_base + warp;
    float sreg[RPL] = {};
    if (col < DN_VAL) {
        float *state_col = state + h * DN_KEY * DN_VAL + col * DN_KEY;
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            sreg[r] = state_col[lane + r * 32];
        }
    }

    for (int chunk_start = 0; chunk_start < S; chunk_start += CHUNK) {
        const int rows = min(CHUNK, S - chunk_start);
        const int chunk_id = chunk_start / CHUNK;
        const int group = col_base / DN_VAL_HEAD_DIM;
        const std::size_t ws_id =
            (static_cast<std::size_t>(h) * DN_VAL_GROUPS + group) * static_cast<std::size_t>(num_chunks) + chunk_id;
        const __nv_bfloat16 *a_mat = a_ws + ws_id * CHUNK * CHUNK;
        const __nv_bfloat16 *p_mat = p_ws + ws_id * CHUNK * CHUNK;

        for (int idx = tid; idx < CHUNK; idx += blockDim.x) {
            g_shared[idx] = g_ws[ws_id * CHUNK + idx];
            beta_shared[idx] = beta_ws[ws_id * CHUNK + idx];
        }
        __syncthreads();

        if (col < DN_VAL) {
            const int c = warp;
            for (int t = 0; t < rows; ++t) {
                const float *kt = qkv_f32 + (chunk_start + t) * DN_CONV_CH + DN_QK_SIZE + h * DN_KEY;
                float k_dot_state = 0.0f;
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    k_dot_state += kt[lane + r * 32] * sreg[r];
                }
                k_dot_state = pf_warp_sum(k_dot_state);
                if (lane == 0) {
                    const float *v = qkv_f32 + (chunk_start + t) * DN_CONV_CH + 2 * DN_QK_SIZE + h * DN_VAL;
                    const float w = beta_shared[t] * (v[col] - expf(g_shared[t]) * k_dot_state);
                    w_bf16[t * COLS + c] = __float2bfloat16(w);
                }
                __syncwarp();
            }
        }
        for (int idx = tid; idx < CHUNK * COLS; idx += blockDim.x) {
            const int t = idx / COLS;
            const int c = idx - t * COLS;
            if (t >= rows || col_base + c >= DN_VAL) {
                w_bf16[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        for (int tile = warp; tile < TILES * (COLS / WMMA_N); tile += blockDim.x / 32) {
            const int tile_m = tile / (COLS / WMMA_N);
            const int tile_n = tile - tile_m * (COLS / WMMA_N);

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> w_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> vnew_acc;
            wmma::fill_fragment(vnew_acc, 0.0f);

            for (int k0 = 0; k0 < CHUNK; k0 += WMMA_K) {
                const __nv_bfloat16 *a_tile = a_mat + (tile_m * WMMA_M) * CHUNK + k0;
                const __nv_bfloat16 *w_tile = w_bf16 + k0 * COLS + tile_n * WMMA_N;
                wmma::load_matrix_sync(a_frag, a_tile, CHUNK);
                wmma::load_matrix_sync(w_frag, w_tile, COLS);
                wmma::mma_sync(vnew_acc, a_frag, w_frag, vnew_acc);
            }

            wmma::store_matrix_sync(
                vnew_shared + (tile_m * WMMA_M) * COLS + tile_n * WMMA_N,
                vnew_acc,
                COLS,
                wmma::mem_row_major);
        }
        __syncthreads();

        for (int idx = tid; idx < CHUNK * COLS; idx += blockDim.x) {
            const int t = idx / COLS;
            const int c = idx - t * COLS;
            const float vnew = (t < rows && col_base + c < DN_VAL) ? vnew_shared[idx] : 0.0f;
            w_bf16[idx] = __float2bfloat16(vnew);
        }
        __syncthreads();

        for (int tile = warp; tile < TILES * (COLS / WMMA_N); tile += blockDim.x / 32) {
            const int tile_m = tile / (COLS / WMMA_N);
            const int tile_n = tile - tile_m * (COLS / WMMA_N);

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> vnew_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> intra_acc;
            wmma::fill_fragment(intra_acc, 0.0f);

            for (int k0 = 0; k0 < CHUNK; k0 += WMMA_K) {
                const __nv_bfloat16 *p_tile = p_mat + (tile_m * WMMA_M) * CHUNK + k0;
                const __nv_bfloat16 *vnew_tile = w_bf16 + k0 * COLS + tile_n * WMMA_N;
                wmma::load_matrix_sync(p_frag, p_tile, CHUNK);
                wmma::load_matrix_sync(vnew_frag, vnew_tile, COLS);
                wmma::mma_sync(intra_acc, p_frag, vnew_frag, intra_acc);
            }

            wmma::store_matrix_sync(
                intra_shared + (tile_m * WMMA_M) * COLS + tile_n * WMMA_N,
                intra_acc,
                COLS,
                wmma::mem_row_major);
        }
        __syncthreads();

        if (col < DN_VAL) {
            const int c = warp;
            for (int t = 0; t < rows; ++t) {
                const float *qt = qkv_f32 + (chunk_start + t) * DN_CONV_CH + h * DN_KEY;
                float q_dot_state = 0.0f;
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    q_dot_state += qt[lane + r * 32] * sreg[r];
                }
                q_dot_state = pf_warp_sum(q_dot_state);

                if (lane == 0) {
                    output[(chunk_start + t) * DN_V_SIZE + h * DN_VAL + col] =
                        expf(g_shared[t]) * q_dot_state + intra_shared[t * COLS + c];
                }
                __syncwarp();
            }

            const float g_last = g_shared[rows - 1];
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                const int d = lane + r * 32;
                float new_state = expf(g_last) * sreg[r];
                for (int t = 0; t < rows; ++t) {
                    const float kid = qkv_f32[(chunk_start + t) * DN_CONV_CH + DN_QK_SIZE + h * DN_KEY + d];
                    new_state += kid * expf(g_last - g_shared[t]) * vnew_shared[t * COLS + c];
                }
                sreg[r] = new_state;
            }
        }
        __syncthreads();
    }

    if (col < DN_VAL) {
        float *state_col = state + h * DN_KEY * DN_VAL + col * DN_KEY;
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            state_col[lane + r * 32] = sreg[r];
        }
    }
}

static constexpr std::size_t kFlashqla64TiledSharedBytes =
    static_cast<std::size_t>(
        64 * DN_KEY +
        64 * 64 +
        64 * 64 +
        64 * 8 +
        64 * 8 +
        64 * 8) * sizeof(float);

static constexpr std::size_t kFlashqla64CudaRefSharedBytes =
    static_cast<std::size_t>(
        64 * DN_KEY +
        64 * 64 +
        64 * 64 +
        64 +
        64 +
        64) * sizeof(float);

static constexpr std::size_t kFlashqla64TcTiledSharedBytes =
    static_cast<std::size_t>(
        64 * 64 +
        64 * 64 +
        64 * 64 +
        64 +
        64 +
        64 * 32) * sizeof(float) +
    static_cast<std::size_t>(2 * 64 * DN_KEY) * sizeof(__nv_bfloat16);

void launch_pf_deltanet_recurrence_flashqla64(
    const float *qkv_f32,
    const float *beta_buf,
    const float *alpha_buf,
    float *state,
    float *output,
    int S,
    cudaStream_t stream)
{
    pf_deltanet_recurrence_flashqla64<<<dim3(DN_HEADS, DN_VAL), 32, 0, stream>>>(
        qkv_f32, beta_buf, alpha_buf, state, output, S);
}

void launch_pf_deltanet_recurrence_flashqla64_tiled(
    const float *qkv_f32,
    const float *beta_buf,
    const float *alpha_buf,
    float *state,
    float *output,
    int S,
    cudaStream_t stream)
{
    cudaFuncSetAttribute(
        pf_deltanet_recurrence_flashqla64_tiled,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kFlashqla64TiledSharedBytes));
    pf_deltanet_recurrence_flashqla64_tiled<<<dim3(DN_HEADS, (DN_VAL + 7) / 8), 256, kFlashqla64TiledSharedBytes, stream>>>(
        qkv_f32, beta_buf, alpha_buf, state, output, S);
}

void launch_pf_deltanet_flashqla64_cuda_ref(
    const float *qkv_f32,
    const float *beta_buf,
    const float *alpha_buf,
    float *state,
    float *output,
    int S,
    cudaStream_t stream)
{
    cudaFuncSetAttribute(
        pf_deltanet_flashqla64_cuda_ref,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kFlashqla64CudaRefSharedBytes));
    pf_deltanet_flashqla64_cuda_ref<<<dim3(DN_HEADS, DN_VAL), 32, kFlashqla64CudaRefSharedBytes, stream>>>(
        qkv_f32, beta_buf, alpha_buf, state, output, S);
}

void launch_pf_deltanet_recurrence_flashqla64_tc_tiled(
    const float *qkv_f32,
    const float *beta_buf,
    const float *alpha_buf,
    float *state,
    float *output,
    int S,
    void *workspace,
    cudaStream_t stream)
{
    launch_pf_deltanet_flashqla64_tc_prepare(qkv_f32, beta_buf, alpha_buf, S, workspace, stream);
    launch_pf_deltanet_recurrence_flashqla64_tc_consume(
        qkv_f32, beta_buf, alpha_buf, state, output, S, workspace, stream);
}

void launch_pf_deltanet_flashqla64_tc_prepare(
    const float *qkv_f32,
    const float *beta_buf,
    const float *alpha_buf,
    int S,
    void *workspace,
    cudaStream_t stream)
{
    cudaFuncSetAttribute(
        pf_deltanet_flashqla64_tc_prepare,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kFlashqla64TcTiledSharedBytes));
    const int num_chunks = (S + 63) / 64;
    pf_deltanet_flashqla64_tc_prepare<<<dim3(DN_HEADS, DN_VAL_GROUPS, num_chunks), 512, kFlashqla64TcTiledSharedBytes, stream>>>(
        qkv_f32, beta_buf, alpha_buf, S, num_chunks, workspace);
}

void launch_pf_deltanet_recurrence_flashqla64_tc_consume(
    const float *qkv_f32,
    const float *beta_buf,
    const float *alpha_buf,
    float *state,
    float *output,
    int S,
    const void *workspace,
    cudaStream_t stream)
{
    cudaFuncSetAttribute(
        pf_deltanet_recurrence_flashqla64_tc_tiled,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kFlashqla64TcTiledSharedBytes));
    const int num_chunks = (S + 63) / 64;
    pf_deltanet_recurrence_flashqla64_tc_tiled<<<dim3(DN_HEADS, (DN_VAL + 31) / 32), 1024, kFlashqla64TcTiledSharedBytes, stream>>>(
        qkv_f32, beta_buf, alpha_buf, state, output, S, num_chunks, workspace);
}
