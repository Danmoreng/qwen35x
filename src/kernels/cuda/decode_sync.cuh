#pragma once

#include <cuda_runtime.h>

struct AtomicGridSync {
    unsigned int *counter;
    unsigned int *generation;
    unsigned int nblocks;
    unsigned int local_gen;

    __device__ void sync() {
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int my_gen = local_gen;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
#else
            __threadfence();
#endif
            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1) {
                *counter = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
#else
                __threadfence();
#endif
                atomicAdd(generation, 1);
            } else {
                volatile unsigned int *vgen = (volatile unsigned int *)generation;
                while (*vgen <= my_gen) {}
            }
            local_gen = my_gen + 1;
        }
        __syncthreads();
    }
};
