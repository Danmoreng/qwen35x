#include "qwen35x/runtime/cuda_inference.h"

#if QWEN35X_HAS_CUDA

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace qwen35x::cuda {

#include "cuda_inference_internal_core.inl"
#include "cuda_inference_internal_kernels.inl"
#include "cuda_inference_api_core.inl"
#include "cuda_inference_api_attention.inl"
#include "cuda_inference_api_sampling.inl"

} // namespace qwen35x::cuda

#endif
