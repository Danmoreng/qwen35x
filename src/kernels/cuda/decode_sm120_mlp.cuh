#pragma once

#include "common.cuh"
#include "decode_sync.cuh"
#include "variant.cuh"
#include "weights.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

__device__ __forceinline__ std::uint8_t sm120_encode_ue4m3_scale(float value)
{
    if (!(value > 0.0f)) return 0;
    int best_bits = 0;
    float best_error = 3.402823466e+38f;
    for (int exponent = 0; exponent < 16; ++exponent) {
        for (int mantissa = 0; mantissa < 8; ++mantissa) {
            float decoded = 0.0f;
            if (exponent == 0) {
                decoded = ldexpf(float(mantissa) * 0.125f, -6);
            } else {
                decoded = ldexpf(1.0f + float(mantissa) * 0.125f, exponent - 7);
            }
            const float error = fabsf(decoded - value);
            if (error < best_error) {
                best_error = error;
                best_bits = (exponent << 3) | mantissa;
            }
        }
    }
    return static_cast<std::uint8_t>(best_bits);
}

__device__ __forceinline__ float sm120_decode_e4m3_scale(const std::uint8_t bits)
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

__device__ __forceinline__ std::uint8_t sm120_encode_nvfp4_e2m1(float value)
{
    constexpr float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const bool negative = value < 0.0f;
    const float abs_value = fabsf(value);
    int best = 0;
    float best_error = 3.402823466e+38f;
    for (int i = 0; i < 8; ++i) {
        const float error = fabsf(levels[i] - abs_value);
        if (error < best_error) {
            best_error = error;
            best = i;
        }
    }
    return static_cast<std::uint8_t>((negative ? 0x8u : 0u) | static_cast<std::uint8_t>(best));
}

static __device__ void sm120_pack_activation_grid(
    const __nv_bfloat16 *__restrict__ input_bf16,
    const float *__restrict__ input_f32,
    bool use_f32,
    std::uint32_t *__restrict__ a_fragments,
    std::uint32_t *__restrict__ a_scales,
    int cols,
    int k_blocks,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x & 31;
    const int global_warp = blockIdx.x * NUM_WARPS + warp_id;
    const int total_warps = num_blocks * NUM_WARPS;
    const int lane_in_quad = lane & 3;

    for (int kb = global_warp; kb < k_blocks; kb += total_warps) {
        float group_max[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int group = 0; group < 4; ++group) {
            float local = 0.0f;
            const int group_base = kb * 64 + group * 16;
            for (int i = lane; i < 16; i += 32) {
                const int col = group_base + i;
                if (col < cols) {
                    const float value = use_f32 ? input_f32[col] : __bfloat162float(input_bf16[col]);
                    local = fmaxf(local, fabsf(value));
                }
            }
            if (lane >= 16) {
                local = 0.0f;
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, offset));
            }
            group_max[group] = __shfl_sync(0xffffffffu, local, 0);
        }

        std::uint8_t scale_bytes[4];
        float decoded_scales[4];
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            const float scale = fmaxf(group_max[group] / 6.0f, 1.0e-8f);
            scale_bytes[group] = sm120_encode_ue4m3_scale(scale);
            decoded_scales[group] = fmaxf(sm120_decode_e4m3_scale(scale_bytes[group]), 1.0e-8f);
        }
        a_scales[static_cast<std::size_t>(kb) * 32 + lane] =
            static_cast<std::uint32_t>(scale_bytes[0]) |
            (static_cast<std::uint32_t>(scale_bytes[1]) << 8u) |
            (static_cast<std::uint32_t>(scale_bytes[2]) << 16u) |
            (static_cast<std::uint32_t>(scale_bytes[3]) << 24u);

        std::uint32_t regs[4] = {0, 0, 0, 0};
        for (int i = 0; i < 32; ++i) {
            const int col_offset = lane_in_quad * 8 + (i & 7) + (i >= 16 ? 32 : 0);
            const int group = col_offset / 16;
            const int col = kb * 64 + col_offset;
            const float value = (col < cols) ? (use_f32 ? input_f32[col] : __bfloat162float(input_bf16[col])) : 0.0f;
            const std::uint8_t nibble = sm120_encode_nvfp4_e2m1(value / decoded_scales[group]);
            regs[i / 8] |= static_cast<std::uint32_t>(nibble & 0xfu) << ((i & 7) * 4);
        }
        std::uint32_t *out = a_fragments + (static_cast<std::size_t>(kb) * 32 + lane) * 4;
        out[0] = regs[0];
        out[1] = regs[1];
        out[2] = regs[2];
        out[3] = regs[3];
    }
}

static __device__ void sm120_pack_activation_bf16_grid(
    const __nv_bfloat16 *__restrict__ input,
    std::uint32_t *__restrict__ a_fragments,
    std::uint32_t *__restrict__ a_scales,
    int cols,
    int k_blocks,
    int num_blocks)
{
    sm120_pack_activation_grid(
        input,
        nullptr,
        false,
        a_fragments,
        a_scales,
        cols,
        k_blocks,
        num_blocks);
}

static __device__ void sm120_pack_activation_f32_grid(
    const float *__restrict__ input,
    std::uint32_t *__restrict__ a_fragments,
    std::uint32_t *__restrict__ a_scales,
    int cols,
    int k_blocks,
    int num_blocks)
{
    sm120_pack_activation_grid(
        nullptr,
        input,
        true,
        a_fragments,
        a_scales,
        cols,
        k_blocks,
        num_blocks);
}

static __device__ void sm120_mxf4nvf4_projection_grid(
    const std::uint32_t *__restrict__ a_fragments,
    const std::uint32_t *__restrict__ b_fragments,
    const std::uint32_t *__restrict__ a_scales,
    const std::uint32_t *__restrict__ b_scales,
    float output_alpha,
    float *__restrict__ output,
    int rows,
    int row_tiles,
    int k_blocks,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x & 31;
    const int global_warp = blockIdx.x * NUM_WARPS + warp_id;

#if defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200)
    const int total_warps = num_blocks * NUM_WARPS;
    for (int row_tile = global_warp; row_tile < row_tiles; row_tile += total_warps) {
        float d0 = 0.0f;
        float d1 = 0.0f;
        float d2 = 0.0f;
        float d3 = 0.0f;
        std::uint16_t bid = 0;
        std::uint16_t tid = 0;
        for (int kb = 0; kb < k_blocks; ++kb) {
            const std::uint32_t *a_base = a_fragments + (static_cast<std::size_t>(kb) * 32 + lane) * 4;
            const std::uint32_t *b_base = b_fragments + ((static_cast<std::size_t>(row_tile) * k_blocks + kb) * 32 + lane) * 2;
            const std::uint32_t scale_a = a_scales[static_cast<std::size_t>(kb) * 32 + lane];
            const std::uint32_t scale_b = b_scales[(static_cast<std::size_t>(row_tile) * k_blocks + kb) * 32 + lane];
            float next0 = 0.0f;
            float next1 = 0.0f;
            float next2 = 0.0f;
            float next3 = 0.0f;
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13}, "
                "%14, {%16, %17}, "
                "%15, {%16, %17};\n"
                : "=f"(next0), "=f"(next1), "=f"(next2), "=f"(next3)
                : "r"(a_base[0]), "r"(a_base[1]), "r"(a_base[2]), "r"(a_base[3]),
                  "r"(b_base[0]), "r"(b_base[1]),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(scale_a), "r"(scale_b), "h"(bid), "h"(tid));
            d0 = next0;
            d1 = next1;
            d2 = next2;
            d3 = next3;
        }
        if (lane < 4) {
            const int row0 = row_tile * 8 + lane * 2;
            if (row0 < rows) {
                output[row0] = d0 * output_alpha;
            }
            if (row0 + 1 < rows) {
                output[row0 + 1] = d1 * output_alpha;
            }
        }
    }
#else
    if (global_warp == 0 && lane == 0 && output != nullptr) {
        output[0] = -3.4028234663852886e38f;
    }
#endif
}

static __device__ void matvec_gate_up_silu_nvfp4_sm120(
    AtomicGridSync &grid,
    const __nv_bfloat16 *__restrict__ s_input,
    const Nvfp4Weight &gate_weight,
    const Nvfp4Weight &up_weight,
    float *__restrict__ output,
    float *__restrict__ scratch,
    int in_dim,
    int out_dim,
    int num_blocks)
{
    std::uint32_t *a_fragments = reinterpret_cast<std::uint32_t *>(scratch);
    std::uint32_t *a_scales = a_fragments + gate_weight.sm120_k_blocks * 32 * 4;
    float *up_output = reinterpret_cast<float *>(a_scales + gate_weight.sm120_k_blocks * 32);

    sm120_pack_activation_bf16_grid(s_input, a_fragments, a_scales, in_dim, gate_weight.sm120_k_blocks, num_blocks);
    grid.sync();
    sm120_mxf4nvf4_projection_grid(
        a_fragments, gate_weight.sm120_packed_weight_fragments, a_scales, gate_weight.sm120_weight_scale_fragments,
        __ldg(gate_weight.weight_scale_2), output, out_dim, gate_weight.sm120_row_tiles, gate_weight.sm120_k_blocks, num_blocks);
    sm120_mxf4nvf4_projection_grid(
        a_fragments, up_weight.sm120_packed_weight_fragments, a_scales, up_weight.sm120_weight_scale_fragments,
        __ldg(up_weight.weight_scale_2), up_output, out_dim, up_weight.sm120_row_tiles, up_weight.sm120_k_blocks, num_blocks);
    grid.sync();

    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < out_dim; i += num_blocks * BLOCK_SIZE) {
        output[i] = fast_silu(output[i]) * up_output[i];
    }
}

static __device__ void matvec_down_residual_nvfp4_sm120(
    AtomicGridSync &grid,
    const float *__restrict__ s_input,
    const Nvfp4Weight &weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ hidden_out,
    float *__restrict__ scratch,
    int in_dim,
    int out_dim,
    int num_blocks)
{
    std::uint32_t *a_fragments = reinterpret_cast<std::uint32_t *>(scratch);
    std::uint32_t *a_scales = a_fragments + weight.sm120_k_blocks * 32 * 4;
    float *down_output = reinterpret_cast<float *>(a_scales + weight.sm120_k_blocks * 32);

    sm120_pack_activation_f32_grid(s_input, a_fragments, a_scales, in_dim, weight.sm120_k_blocks, num_blocks);
    grid.sync();
    sm120_mxf4nvf4_projection_grid(
        a_fragments, weight.sm120_packed_weight_fragments, a_scales, weight.sm120_weight_scale_fragments,
        __ldg(weight.weight_scale_2), down_output, out_dim, weight.sm120_row_tiles, weight.sm120_k_blocks, num_blocks);
    grid.sync();

    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < out_dim; i += num_blocks * BLOCK_SIZE) {
        hidden_out[i] = __float2bfloat16(down_output[i] + __bfloat162float(residual[i]));
    }
}
