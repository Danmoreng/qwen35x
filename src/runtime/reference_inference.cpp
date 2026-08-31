#include "qwen35x/runtime/reference_inference.h"

#include "qwen35x/cpu/activation.h"

#include "qwen35x/cpu/q4_0.h"
#include "qwen35x/cpu/q4_h128.h"
#include "qwen35x/cpu/q8_0.h"
#include "qwen35x/cpu/executor.h"
#include "qwen35x/cpu/delta_net.h"
#include "qwen35x/cpu/full_attention.h"
#include "qwen35x/runtime/cuda_inference.h"
#include "qwen35x/runtime/qwen35x_cuda_backend.h"
#include "qwen35x/weights/gguf.h"
#include "qwen35x/weights/q4_h128_artifact.h"
#include "qwen35x/weights/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_set>

namespace qwen35x {

#include "reference_inference_internal_weights_workspace.inl"
#include "reference_inference_internal_layers.inl"
#include "reference_inference_internal_forward.inl"
#include "reference_inference_internal_cpu_prefill.inl"
#include "reference_decode_backend_api.inl"
#include "reference_inference_api.inl"

} // namespace qwen35x
