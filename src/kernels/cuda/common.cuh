#pragma once

#include "variant.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float pf_warp_sum(float v) {
    return warp_reduce_sum(v);
}

__device__ __forceinline__ float pf_warp_max(float v) {
    return warp_reduce_max(v);
}

__device__ __forceinline__ float fast_exp(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.44269504088896340736f));
    return y;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(1.0f + fast_exp(-x)));
    return y;
}

__device__ __forceinline__ float fast_silu(float x) {
    return x * fast_sigmoid(x);
}

__device__ __forceinline__ float pf_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ uint4 load_128bit(const uint4 *ptr) {
    uint4 out;
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
    return out;
}

// BF16 dot product: 8 bf16 weights x 8 bf16 activations -> f32
__device__ __forceinline__ float dot8_bf16(const uint4 &w_u4, const __nv_bfloat16 *act) {
    const __nv_bfloat16 *w = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++)
        sum += __bfloat162float(w[i]) * __bfloat162float(act[i]);
    return sum;
}
