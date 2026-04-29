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

__global__ void __launch_bounds__(1024, 1)
pf_deltanet_recurrence_flashqla64_tc_tiled(
    const float *qkv_f32, const float *beta_buf, const float *alpha_buf,
    float *state, float *output, int S)
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
    float *kk_shared = smem;
    float *qk_shared = kk_shared + CHUNK * CHUNK;
    float *chunk_decay_prefix = qk_shared + CHUNK * CHUNK;
    float *chunk_beta = chunk_decay_prefix + CHUNK * COLS;
    float *delta = chunk_beta + CHUNK * COLS;
    __nv_bfloat16 *q_bf16 = reinterpret_cast<__nv_bfloat16 *>(delta + CHUNK * COLS);
    __nv_bfloat16 *k_bf16 = q_bf16 + CHUNK * DN_KEY;

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

            wmma::store_matrix_sync(
                kk_shared + (tile_m * WMMA_M) * CHUNK + tile_n * WMMA_N,
                kk_acc,
                CHUNK,
                wmma::mem_row_major);
            wmma::store_matrix_sync(
                qk_shared + (tile_m * WMMA_M) * CHUNK + tile_n * WMMA_N,
                qk_acc,
                CHUNK,
                wmma::mem_row_major);
        }
        __syncthreads();

        if (col < DN_VAL) {
            const int c = warp;
            for (int t = 0; t < rows; ++t) {
                const float *kt = qkv_f32 + (chunk_start + t) * DN_CONV_CH + DN_QK_SIZE + h * DN_KEY;
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
                    const float kid = qkv_f32[(chunk_start + i) * DN_CONV_CH + DN_QK_SIZE + h * DN_KEY + d];
                    new_state += ratio * kid * delta[i * COLS + c];
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
        64 * 32 +
        64 * 32 +
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
    cudaStream_t stream)
{
    cudaFuncSetAttribute(
        pf_deltanet_recurrence_flashqla64_tc_tiled,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kFlashqla64TcTiledSharedBytes));
    pf_deltanet_recurrence_flashqla64_tc_tiled<<<dim3(DN_HEADS, (DN_VAL + 31) / 32), 1024, kFlashqla64TcTiledSharedBytes, stream>>>(
        qkv_f32, beta_buf, alpha_buf, state, output, S);
}
