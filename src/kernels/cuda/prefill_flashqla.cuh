#pragma once

#include <cuda_runtime.h>

void launch_pf_deltanet_recurrence_flashqla64(
    const float *qkv_f32,
    const float *beta_buf,
    const float *alpha_buf,
    float *state,
    float *output,
    int S,
    cudaStream_t stream);

void launch_pf_deltanet_recurrence_flashqla64_tiled(
    const float *qkv_f32,
    const float *beta_buf,
    const float *alpha_buf,
    float *state,
    float *output,
    int S,
    cudaStream_t stream);

void launch_pf_deltanet_flashqla64_cuda_ref(
    const float *qkv_f32,
    const float *beta_buf,
    const float *alpha_buf,
    float *state,
    float *output,
    int S,
    cudaStream_t stream);

void launch_pf_deltanet_recurrence_flashqla64_tc_tiled(
    const float *qkv_f32,
    const float *beta_buf,
    const float *alpha_buf,
    float *state,
    float *output,
    int S,
    void *workspace,
    cudaStream_t stream);
