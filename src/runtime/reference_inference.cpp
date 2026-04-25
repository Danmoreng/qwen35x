#include "qwen35x/runtime/reference_inference.h"

#include "qwen35x/runtime/cuda_inference.h"
#include "qwen35x/runtime/qwen35x_cuda_backend.h"
#include "qwen35x/weights/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_set>

namespace qwen35x {

#include "reference_inference_internal_weights_workspace.inl"
#include "reference_inference_internal_layers.inl"
#include "reference_inference_internal_forward.inl"
#include "reference_decode_backend_api.inl"
#include "reference_inference_api.inl"

} // namespace qwen35x
