namespace {

constexpr std::size_t kCublasLtWorkspaceBytes = 8u * 1024u * 1024u;
constexpr int kGpuSamplingMaxTopK = 64;
constexpr int kSamplingArgmaxBlockSize = 256;
constexpr int kSamplingArgmaxChunkSize = 2048;

__global__ void f32_to_bf16_kernel(const float * input, __nv_bfloat16 * output, int count);

const char * cublas_status_string(const cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
#ifdef CUBLAS_STATUS_LICENSE_ERROR
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

bool check_cublas(const cublasStatus_t status, const char * step, std::string & error_message) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return true;
  }
  error_message = std::string(step) + " failed: " + cublas_status_string(status);
  return false;
}

struct MatvecPlan {
  int rows = 0;
  int cols = 0;
  bool use_bf16 = false;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  bool has_algo = false;
};

void destroy_matvec_plan(MatvecPlan & plan) {
  if (plan.op_desc != nullptr) {
    cublasLtMatmulDescDestroy(plan.op_desc);
    plan.op_desc = nullptr;
  }
  if (plan.a_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.a_desc);
    plan.a_desc = nullptr;
  }
  if (plan.b_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.b_desc);
    plan.b_desc = nullptr;
  }
  if (plan.c_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(plan.c_desc);
    plan.c_desc = nullptr;
  }
  plan.has_algo = false;
}

struct InferenceSessionState {
  float * workspace_input = nullptr;
  std::size_t workspace_input_count = 0;
  float * workspace_output = nullptr;
  std::size_t workspace_output_count = 0;
  __nv_bfloat16 * workspace_input_bf16 = nullptr;
  std::size_t workspace_input_bf16_count = 0;
  cudaStream_t stream = nullptr;
  cublasLtHandle_t cublas_lt = nullptr;
  void * cublas_lt_workspace = nullptr;
  std::size_t cublas_lt_workspace_bytes = 0;
  float * sampling_block_values = nullptr;
  int * sampling_block_indices = nullptr;
  float * sampling_block_topk_values = nullptr;
  int * sampling_block_topk_indices = nullptr;
  int sampling_block_capacity = 0;
  std::vector<MatvecPlan> matvec_plans;
  bool prefer_bf16_matvec = false;
  bool active = false;
};

thread_local InferenceSessionState g_session;
CudaTransferStats g_transfer_stats;

bool check_cuda(cudaError_t status, const char * step, std::string & error_message) {
  if (status == cudaSuccess) {
    return true;
  }
  error_message = std::string(step) + " failed: " + cudaGetErrorString(status);
  return false;
}

std::uint16_t float_to_bf16_bits(const float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(float) == sizeof(std::uint32_t), "Unexpected float size");
  std::memcpy(&bits, &value, sizeof(float));
  const std::uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  return static_cast<std::uint16_t>(bits >> 16);
}

bool record_copy(const std::size_t bytes, const cudaMemcpyKind kind) {
  ++g_transfer_stats.copy_calls;
  switch (kind) {
    case cudaMemcpyHostToDevice:
      g_transfer_stats.host_to_device_bytes += static_cast<std::uint64_t>(bytes);
      return true;
    case cudaMemcpyDeviceToHost:
      g_transfer_stats.device_to_host_bytes += static_cast<std::uint64_t>(bytes);
      return true;
    default:
      g_transfer_stats.other_bytes += static_cast<std::uint64_t>(bytes);
      return true;
  }
}

bool tracked_memcpy(
  void * dst,
  const void * src,
  const std::size_t bytes,
  const cudaMemcpyKind kind,
  const char * step,
  std::string & error_message) {
  if (!check_cuda(cudaMemcpy(dst, src, bytes, kind), step, error_message)) {
    return false;
  }
  return record_copy(bytes, kind);
}

bool tracked_memcpy_async(
  void * dst,
  const void * src,
  const std::size_t bytes,
  const cudaMemcpyKind kind,
  cudaStream_t stream,
  const char * step,
  std::string & error_message) {
  if (!check_cuda(cudaMemcpyAsync(dst, src, bytes, kind, stream), step, error_message)) {
    return false;
  }
  return record_copy(bytes, kind);
}

void release_session_storage() {
  for (auto & plan : g_session.matvec_plans) {
    destroy_matvec_plan(plan);
  }
  g_session.matvec_plans.clear();
  if (g_session.sampling_block_values != nullptr) {
    cudaFree(g_session.sampling_block_values);
    g_session.sampling_block_values = nullptr;
  }
  if (g_session.sampling_block_indices != nullptr) {
    cudaFree(g_session.sampling_block_indices);
    g_session.sampling_block_indices = nullptr;
  }
  if (g_session.sampling_block_topk_values != nullptr) {
    cudaFree(g_session.sampling_block_topk_values);
    g_session.sampling_block_topk_values = nullptr;
  }
  if (g_session.sampling_block_topk_indices != nullptr) {
    cudaFree(g_session.sampling_block_topk_indices);
    g_session.sampling_block_topk_indices = nullptr;
  }
  g_session.sampling_block_capacity = 0;
  if (g_session.cublas_lt_workspace != nullptr) {
    cudaFree(g_session.cublas_lt_workspace);
    g_session.cublas_lt_workspace = nullptr;
    g_session.cublas_lt_workspace_bytes = 0;
  }
  if (g_session.cublas_lt != nullptr) {
    cublasLtDestroy(g_session.cublas_lt);
    g_session.cublas_lt = nullptr;
  }
  if (g_session.workspace_input != nullptr) {
    cudaFree(g_session.workspace_input);
    g_session.workspace_input = nullptr;
  }
  if (g_session.workspace_input_bf16 != nullptr) {
    cudaFree(g_session.workspace_input_bf16);
    g_session.workspace_input_bf16 = nullptr;
  }
  if (g_session.workspace_output != nullptr) {
    cudaFree(g_session.workspace_output);
    g_session.workspace_output = nullptr;
  }
  g_session.workspace_input_count = 0;
  g_session.workspace_input_bf16_count = 0;
  g_session.workspace_output_count = 0;
  if (g_session.stream != nullptr) {
    cudaStreamDestroy(g_session.stream);
    g_session.stream = nullptr;
  }
}

bool ensure_session_workspace(
  const std::size_t input_count,
  const std::size_t output_count,
  std::string & error_message) {
  if (!g_session.active) {
    if (!check_cuda(cudaStreamCreateWithFlags(&g_session.stream, cudaStreamNonBlocking), "cudaStreamCreate", error_message)) {
      return false;
    }
    if (!check_cublas(cublasLtCreate(&g_session.cublas_lt), "cublasLtCreate", error_message)) {
      release_session_storage();
      return false;
    }
    if (!check_cuda(
          cudaMalloc(&g_session.cublas_lt_workspace, kCublasLtWorkspaceBytes),
          "cudaMalloc(cublasLt_workspace)",
          error_message)) {
      release_session_storage();
      return false;
    }
    g_session.cublas_lt_workspace_bytes = kCublasLtWorkspaceBytes;
    g_session.active = true;
  }

  if (input_count > g_session.workspace_input_count) {
    if (g_session.workspace_input != nullptr) {
      cudaFree(g_session.workspace_input);
      g_session.workspace_input = nullptr;
      g_session.workspace_input_count = 0;
    }
    if (!check_cuda(cudaMalloc(&g_session.workspace_input, input_count * sizeof(float)), "cudaMalloc(workspace_input)", error_message)) {
      return false;
    }
    g_session.workspace_input_count = input_count;
  }

  if (output_count > g_session.workspace_output_count) {
    if (g_session.workspace_output != nullptr) {
      cudaFree(g_session.workspace_output);
      g_session.workspace_output = nullptr;
      g_session.workspace_output_count = 0;
    }
    if (!check_cuda(cudaMalloc(&g_session.workspace_output, output_count * sizeof(float)), "cudaMalloc(workspace_output)", error_message)) {
      return false;
    }
    g_session.workspace_output_count = output_count;
  }

  return true;
}

bool ensure_session_workspace_input_bf16(
  const std::size_t input_count,
  std::string & error_message) {
  if (input_count == 0) {
    return true;
  }
  if (!g_session.active) {
    error_message = "CUDA inference session is not active.";
    return false;
  }
  if (input_count > g_session.workspace_input_bf16_count) {
    if (g_session.workspace_input_bf16 != nullptr) {
      cudaFree(g_session.workspace_input_bf16);
      g_session.workspace_input_bf16 = nullptr;
      g_session.workspace_input_bf16_count = 0;
    }
    if (!check_cuda(
          cudaMalloc(&g_session.workspace_input_bf16, input_count * sizeof(__nv_bfloat16)),
          "cudaMalloc(workspace_input_bf16)",
          error_message)) {
      return false;
    }
    g_session.workspace_input_bf16_count = input_count;
  }
  return true;
}

cudaStream_t active_stream() {
  if (g_session.active && g_session.stream != nullptr) {
    return g_session.stream;
  }
  return nullptr;
}

bool ensure_sampling_workspace(const int vocab_size, std::string & error_message) {
  if (vocab_size <= 0) {
    error_message = "CUDA sampling requires vocab_size > 0.";
    return false;
  }
  const int block_count = std::max(1, (vocab_size + kSamplingArgmaxChunkSize - 1) / kSamplingArgmaxChunkSize);
  if (g_session.sampling_block_capacity < block_count) {
    if (g_session.sampling_block_values != nullptr) {
      cudaFree(g_session.sampling_block_values);
      g_session.sampling_block_values = nullptr;
    }
    if (g_session.sampling_block_indices != nullptr) {
      cudaFree(g_session.sampling_block_indices);
      g_session.sampling_block_indices = nullptr;
    }
    if (!check_cuda(
          cudaMalloc(&g_session.sampling_block_values, static_cast<std::size_t>(block_count) * sizeof(float)),
          "cudaMalloc(sampling_block_values)",
          error_message) ||
        !check_cuda(
          cudaMalloc(&g_session.sampling_block_indices, static_cast<std::size_t>(block_count) * sizeof(int)),
          "cudaMalloc(sampling_block_indices)",
          error_message)) {
      return false;
    }

    if (g_session.sampling_block_topk_values != nullptr) {
      cudaFree(g_session.sampling_block_topk_values);
      g_session.sampling_block_topk_values = nullptr;
    }
    if (g_session.sampling_block_topk_indices != nullptr) {
      cudaFree(g_session.sampling_block_topk_indices);
      g_session.sampling_block_topk_indices = nullptr;
    }
    const std::size_t topk_entries = static_cast<std::size_t>(block_count) * static_cast<std::size_t>(kGpuSamplingMaxTopK);
    if (!check_cuda(
          cudaMalloc(&g_session.sampling_block_topk_values, topk_entries * sizeof(float)),
          "cudaMalloc(sampling_block_topk_values)",
          error_message) ||
        !check_cuda(
          cudaMalloc(&g_session.sampling_block_topk_indices, topk_entries * sizeof(int)),
          "cudaMalloc(sampling_block_topk_indices)",
          error_message)) {
      return false;
    }
    g_session.sampling_block_capacity = block_count;
  }

  return true;
}

MatvecPlan * find_matvec_plan(const int rows, const int cols, const bool use_bf16) {
  for (auto & plan : g_session.matvec_plans) {
    if (plan.rows == rows && plan.cols == cols && plan.use_bf16 == use_bf16) {
      return &plan;
    }
  }
  return nullptr;
}

bool initialize_matvec_plan(
  MatvecPlan & plan,
  const int rows,
  const int cols,
  const bool use_bf16,
  std::string & error_message) {
  plan.rows = rows;
  plan.cols = cols;
  plan.use_bf16 = use_bf16;

  cublasOperation_t trans = CUBLAS_OP_N;
  const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  const cudaDataType_t scale_type = CUDA_R_32F;
  const cudaDataType_t a_type = use_bf16 ? CUDA_R_16BF : CUDA_R_32F;
  const cudaDataType_t b_type = use_bf16 ? CUDA_R_16BF : CUDA_R_32F;
  const cudaDataType_t c_type = CUDA_R_32F;
  if (!check_cublas(
        cublasLtMatmulDescCreate(&plan.op_desc, compute_type, scale_type),
        "cublasLtMatmulDescCreate",
        error_message) ||
      !check_cublas(
        cublasLtMatmulDescSetAttribute(
          plan.op_desc,
          CUBLASLT_MATMUL_DESC_TRANSA,
          &trans,
          sizeof(trans)),
        "cublasLtMatmulDescSetAttribute(TRANSA)",
        error_message) ||
      !check_cublas(
        cublasLtMatmulDescSetAttribute(
          plan.op_desc,
          CUBLASLT_MATMUL_DESC_TRANSB,
          &trans,
          sizeof(trans)),
        "cublasLtMatmulDescSetAttribute(TRANSB)",
        error_message)) {
    destroy_matvec_plan(plan);
    return false;
  }

  if (!check_cublas(
        cublasLtMatrixLayoutCreate(
          &plan.a_desc,
          a_type,
          static_cast<std::uint64_t>(rows),
          static_cast<std::uint64_t>(cols),
          static_cast<std::int64_t>(cols)),
        "cublasLtMatrixLayoutCreate(A)",
        error_message) ||
      !check_cublas(
        cublasLtMatrixLayoutCreate(
          &plan.b_desc,
          b_type,
          static_cast<std::uint64_t>(cols),
          1,
          1),
        "cublasLtMatrixLayoutCreate(B)",
        error_message) ||
      !check_cublas(
        cublasLtMatrixLayoutCreate(
          &plan.c_desc,
          c_type,
          static_cast<std::uint64_t>(rows),
          1,
          1),
        "cublasLtMatrixLayoutCreate(C)",
        error_message)) {
    destroy_matvec_plan(plan);
    return false;
  }

  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  if (!check_cublas(
        cublasLtMatrixLayoutSetAttribute(
          plan.a_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &row_order,
          sizeof(row_order)),
        "cublasLtMatrixLayoutSetAttribute(A_ORDER)",
        error_message) ||
      !check_cublas(
        cublasLtMatrixLayoutSetAttribute(
          plan.b_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &row_order,
          sizeof(row_order)),
        "cublasLtMatrixLayoutSetAttribute(B_ORDER)",
        error_message) ||
      !check_cublas(
        cublasLtMatrixLayoutSetAttribute(
          plan.c_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &row_order,
          sizeof(row_order)),
        "cublasLtMatrixLayoutSetAttribute(C_ORDER)",
        error_message)) {
    destroy_matvec_plan(plan);
    return false;
  }

  cublasLtMatmulPreference_t pref = nullptr;
  if (!check_cublas(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate", error_message)) {
    destroy_matvec_plan(plan);
    return false;
  }
  std::size_t workspace_bytes = g_session.cublas_lt_workspace_bytes;
  const bool pref_ok = check_cublas(
    cublasLtMatmulPreferenceSetAttribute(
      pref,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes,
      sizeof(workspace_bytes)),
    "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE_BYTES)",
    error_message);
  if (!pref_ok) {
    cublasLtMatmulPreferenceDestroy(pref);
    destroy_matvec_plan(plan);
    return false;
  }

  cublasLtMatmulHeuristicResult_t heuristic_result{};
  int returned_results = 0;
  const bool heuristic_ok = check_cublas(
    cublasLtMatmulAlgoGetHeuristic(
      g_session.cublas_lt,
      plan.op_desc,
      plan.a_desc,
      plan.b_desc,
      plan.c_desc,
      plan.c_desc,
      pref,
      1,
      &heuristic_result,
      &returned_results),
    "cublasLtMatmulAlgoGetHeuristic",
    error_message);
  cublasLtMatmulPreferenceDestroy(pref);
  if (!heuristic_ok) {
    destroy_matvec_plan(plan);
    return false;
  }
  if (returned_results <= 0) {
    error_message = "cublasLtMatmulAlgoGetHeuristic did not return a valid algorithm for matvec.";
    destroy_matvec_plan(plan);
    return false;
  }

  plan.algo = heuristic_result.algo;
  plan.has_algo = true;
  return true;
}

