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

bool run_matvec_f32_device_cublaslt(
  const CudaDeviceMatrixF32 & matrix,
  const CudaDeviceBufferF32 & input,
  CudaDeviceBufferF32 & output,
  std::string & error_message) {
  if (g_session.cublas_lt == nullptr) {
    error_message = "cuBLASLt handle is not initialized.";
    return false;
  }

  MatvecPlan * plan = find_matvec_plan(matrix.rows, matrix.cols, false);
  if (plan == nullptr) {
    MatvecPlan new_plan;
    if (!initialize_matvec_plan(new_plan, matrix.rows, matrix.cols, false, error_message)) {
      return false;
    }
    g_session.matvec_plans.push_back(std::move(new_plan));
    plan = &g_session.matvec_plans.back();
  }
  if (!plan->has_algo) {
    error_message = "cuBLASLt matvec plan has no selected algorithm.";
    return false;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  return check_cublas(
    cublasLtMatmul(
      g_session.cublas_lt,
      plan->op_desc,
      &alpha,
      matrix.data,
      plan->a_desc,
      input.data,
      plan->b_desc,
      &beta,
      output.data,
      plan->c_desc,
      output.data,
      plan->c_desc,
      &plan->algo,
      g_session.cublas_lt_workspace,
      g_session.cublas_lt_workspace_bytes,
      active_stream()),
    "cublasLtMatmul(matvec)",
    error_message);
}

bool run_matvec_bf16_device_cublaslt(
  const CudaDeviceMatrixF32 & matrix,
  const CudaDeviceBufferF32 & input,
  CudaDeviceBufferF32 & output,
  std::string & error_message) {
  if (g_session.cublas_lt == nullptr) {
    error_message = "cuBLASLt handle is not initialized.";
    return false;
  }
  if (matrix.data_bf16 == nullptr) {
    error_message = "BF16 shadow matrix is not initialized.";
    return false;
  }
  if (!ensure_session_workspace_input_bf16(static_cast<std::size_t>(matrix.cols), error_message)) {
    return false;
  }

  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA inference session stream is not initialized.";
    return false;
  }

  const int convert_block = 256;
  const int convert_grid = (matrix.cols + convert_block - 1) / convert_block;
  f32_to_bf16_kernel<<<convert_grid, convert_block, 0, stream>>>(
    static_cast<const float *>(input.data),
    g_session.workspace_input_bf16,
    matrix.cols);
  if (!check_cuda(cudaGetLastError(), "f32_to_bf16_kernel", error_message)) {
    return false;
  }

  MatvecPlan * plan = find_matvec_plan(matrix.rows, matrix.cols, true);
  if (plan == nullptr) {
    MatvecPlan new_plan;
    if (!initialize_matvec_plan(new_plan, matrix.rows, matrix.cols, true, error_message)) {
      return false;
    }
    g_session.matvec_plans.push_back(std::move(new_plan));
    plan = &g_session.matvec_plans.back();
  }
  if (!plan->has_algo) {
    error_message = "cuBLASLt BF16 matvec plan has no selected algorithm.";
    return false;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  return check_cublas(
    cublasLtMatmul(
      g_session.cublas_lt,
      plan->op_desc,
      &alpha,
      matrix.data_bf16,
      plan->a_desc,
      g_session.workspace_input_bf16,
      plan->b_desc,
      &beta,
      output.data,
      plan->c_desc,
      output.data,
      plan->c_desc,
      &plan->algo,
      g_session.cublas_lt_workspace,
      g_session.cublas_lt_workspace_bytes,
      stream),
    "cublasLtMatmul(matvec_bf16)",
    error_message);
}

__global__ void f32_matvec_kernel(
  const float * weights,
  const float * input,
  float * output,
  int rows,
  int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float * row_ptr = weights + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  float sum = 0.0f;
  for (int c = 0; c < cols; ++c) {
    sum += row_ptr[c] * input[c];
  }
  output[row] = sum;
}

__global__ void f32_to_bf16_kernel(
  const float * input,
  __nv_bfloat16 * output,
  const int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  output[idx] = __float2bfloat16(input[idx]);
}

__global__ void gather_matrix_row_kernel(
  const float * matrix,
  const int cols,
  const int row_index,
  float * out_row) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  const std::size_t src_offset =
    static_cast<std::size_t>(row_index) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col);
  out_row[col] = matrix[src_offset];
}

__global__ void gather_matrix_row_bf16_kernel(
  const __nv_bfloat16 * matrix,
  const int cols,
  const int row_index,
  float * out_row) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  const std::size_t src_offset =
    static_cast<std::size_t>(row_index) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col);
  out_row[col] = __bfloat162float(matrix[src_offset]);
}

__global__ void gather_matrix_row_from_token_kernel(
  const float * matrix,
  const int rows,
  const int cols,
  const float * token_id,
  float * out_row) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }

  int row_index = static_cast<int>(token_id[0]);
  if (row_index < 0) {
    row_index = 0;
  } else if (row_index >= rows) {
    row_index = rows - 1;
  }

  const std::size_t src_offset =
    static_cast<std::size_t>(row_index) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col);
  out_row[col] = matrix[src_offset];
}

__global__ void gather_matrix_row_from_token_bf16_kernel(
  const __nv_bfloat16 * matrix,
  const int rows,
  const int cols,
  const float * token_id,
  float * out_row) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }

  int row_index = static_cast<int>(token_id[0]);
  if (row_index < 0) {
    row_index = 0;
  } else if (row_index >= rows) {
    row_index = rows - 1;
  }

  const std::size_t src_offset =
    static_cast<std::size_t>(row_index) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col);
  out_row[col] = __bfloat162float(matrix[src_offset]);
}

__global__ void silu_mul_kernel(
  const float * a,
  const float * b,
  float * out,
  const std::size_t count) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                          static_cast<std::size_t>(threadIdx.x);
  if (idx >= count) {
    return;
  }
  const float av = a[idx];
  const float sig = 1.0f / (1.0f + expf(-av));
  out[idx] = (av * sig) * b[idx];
}

__global__ void add_kernel(
  const float * a,
  const float * b,
  float * out,
  const std::size_t count) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                          static_cast<std::size_t>(threadIdx.x);
  if (idx >= count) {
    return;
  }
  out[idx] = a[idx] + b[idx];
}

__global__ void rms_norm_kernel(
  const float * input,
  const float * weight,
  float * out,
  const int count,
  const float eps) {
  __shared__ float reduction[256];
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (int i = tid; i < count; i += blockDim.x) {
    const float v = input[i];
    sum += v * v;
  }
  reduction[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }

  __shared__ float inv;
  if (tid == 0) {
    inv = rsqrtf((reduction[0] / static_cast<float>(count)) + eps);
  }
  __syncthreads();

  for (int i = tid; i < count; i += blockDim.x) {
    out[i] = input[i] * inv * (1.0f + weight[i]);
  }
}

__global__ void split_q_gate_kernel(
  const float * packed,
  float * out_q,
  float * out_gate,
  const int n_heads,
  const int head_dim) {
  const int h = blockIdx.x;
  const int d = threadIdx.x;
  if (h >= n_heads || d >= head_dim) {
    return;
  }
  const int q_span = head_dim * 2;
  const std::size_t src = static_cast<std::size_t>(h) * static_cast<std::size_t>(q_span) + static_cast<std::size_t>(d);
  const std::size_t dst = static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim) + static_cast<std::size_t>(d);
  out_q[dst] = packed[src];
  out_gate[dst] = packed[src + static_cast<std::size_t>(head_dim)];
}

__global__ void rms_norm_per_head_kernel(
  const float * input,
  const float * weight,
  float * out,
  const int n_heads,
  const int head_dim,
  const float eps) {
  __shared__ float reduction[256];
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  if (head >= n_heads) {
    return;
  }

  const float * in_ptr = input + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  float * out_ptr = out + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);

  float sum = 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const float v = in_ptr[d];
    sum += v * v;
  }
  reduction[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }

  __shared__ float inv;
  if (tid == 0) {
    inv = rsqrtf((reduction[0] / static_cast<float>(head_dim)) + eps);
  }
  __syncthreads();

  for (int d = tid; d < head_dim; d += blockDim.x) {
    out_ptr[d] = in_ptr[d] * inv * (1.0f + weight[d]);
  }
}

__global__ void rope_inplace_kernel(
  float * values,
  const int n_heads,
  const int head_dim,
  const int rope_dim,
  const int position,
  const float rope_theta) {
  const int head = blockIdx.x;
  const int pair_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (head >= n_heads) {
    return;
  }
  const int half = rope_dim / 2;
  if (pair_idx >= half) {
    return;
  }

  const std::size_t base = static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  float * ptr = values + base;
  const int i0 = pair_idx;
  const int i1 = pair_idx + half;
  if (i1 >= head_dim) {
    return;
  }
  const float x0 = ptr[i0];
  const float x1 = ptr[i1];
  const float inv_freq = expf(
    -logf(rope_theta) * (static_cast<float>(2 * pair_idx) / static_cast<float>(rope_dim)));
  const float angle = static_cast<float>(position) * inv_freq;
  const float c = cosf(angle);
  const float s = sinf(angle);
  ptr[i0] = x0 * c - x1 * s;
  ptr[i1] = x1 * c + x0 * s;
}

__global__ void prepare_full_attention_qkv_kernel(
  const float * q_gate_packed,
  const float * k_raw,
  const float * v_raw,
  const float * q_norm_weight,
  const float * k_norm_weight,
  float * out_q,
  float * out_gate,
  float * k_cache,
  float * v_cache,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int applied_rope_dim,
  const int position,
  const float rope_theta,
  const float eps,
  const unsigned long long k_cache_offset,
  const unsigned long long v_cache_offset) {
  __shared__ float reduction[256];
  __shared__ float inv_norm;

  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  if (head >= n_heads) {
    return;
  }

  const int pair_count = applied_rope_dim / 2;
  const std::size_t q_head_base = static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  const std::size_t packed_base = q_head_base * 2;

  // Split Q/G, apply per-head RMSNorm on Q, then RoPE on Q.
  float q_sq_sum = 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const float qv = q_gate_packed[packed_base + static_cast<std::size_t>(d)];
    q_sq_sum += qv * qv;
  }
  reduction[tid] = q_sq_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    inv_norm = rsqrtf((reduction[0] / static_cast<float>(head_dim)) + eps);
  }
  __syncthreads();

  for (int d = tid; d < head_dim; d += blockDim.x) {
    const std::size_t idx = q_head_base + static_cast<std::size_t>(d);
    const float qv = q_gate_packed[packed_base + static_cast<std::size_t>(d)];
    const float gv = q_gate_packed[packed_base + static_cast<std::size_t>(head_dim) + static_cast<std::size_t>(d)];
    out_q[idx] = qv * inv_norm * (1.0f + q_norm_weight[d]);
    out_gate[idx] = gv;
  }
  __syncthreads();

  for (int pair_idx = tid; pair_idx < pair_count; pair_idx += blockDim.x) {
    const int i0 = pair_idx;
    const int i1 = pair_idx + pair_count;
    if (i1 >= head_dim) {
      continue;
    }
    const float inv_freq = expf(
      -logf(rope_theta) * (static_cast<float>(2 * pair_idx) / static_cast<float>(applied_rope_dim)));
    const float angle = static_cast<float>(position) * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const std::size_t q0_idx = q_head_base + static_cast<std::size_t>(i0);
    const std::size_t q1_idx = q_head_base + static_cast<std::size_t>(i1);
    const float x0 = out_q[q0_idx];
    const float x1 = out_q[q1_idx];
    out_q[q0_idx] = x0 * c - x1 * s;
    out_q[q1_idx] = x1 * c + x0 * s;
  }
  __syncthreads();

  // For KV heads, apply RMSNorm+RoPE on K and write K/V directly into caches.
  if (head >= n_kv_heads) {
    return;
  }

  const std::size_t kv_head_base = static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);

  float k_sq_sum = 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const float kv = k_raw[kv_head_base + static_cast<std::size_t>(d)];
    k_sq_sum += kv * kv;
  }
  reduction[tid] = k_sq_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    inv_norm = rsqrtf((reduction[0] / static_cast<float>(head_dim)) + eps);
  }
  __syncthreads();

  const std::size_t k_cache_base = static_cast<std::size_t>(k_cache_offset) + kv_head_base;
  const std::size_t v_cache_base = static_cast<std::size_t>(v_cache_offset) + kv_head_base;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const std::size_t local_idx = kv_head_base + static_cast<std::size_t>(d);
    k_cache[k_cache_base + static_cast<std::size_t>(d)] =
      k_raw[local_idx] * inv_norm * (1.0f + k_norm_weight[d]);
    v_cache[v_cache_base + static_cast<std::size_t>(d)] = v_raw[local_idx];
  }
  __syncthreads();

  for (int pair_idx = tid; pair_idx < pair_count; pair_idx += blockDim.x) {
    const int i0 = pair_idx;
    const int i1 = pair_idx + pair_count;
    if (i1 >= head_dim) {
      continue;
    }
    const float inv_freq = expf(
      -logf(rope_theta) * (static_cast<float>(2 * pair_idx) / static_cast<float>(applied_rope_dim)));
    const float angle = static_cast<float>(position) * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const std::size_t k0_idx = k_cache_base + static_cast<std::size_t>(i0);
    const std::size_t k1_idx = k_cache_base + static_cast<std::size_t>(i1);
    const float x0 = k_cache[k0_idx];
    const float x1 = k_cache[k1_idx];
    k_cache[k0_idx] = x0 * c - x1 * s;
    k_cache[k1_idx] = x1 * c + x0 * s;
  }
}

__global__ void prepare_full_attention_qkv_prefill_chunk_kernel(
  const float * q_gate_packed,
  const float * k_raw,
  const float * v_raw,
  const float * q_norm_weight,
  const float * k_norm_weight,
  float * out_q,
  float * out_gate,
  float * k_cache,
  float * v_cache,
  const int token_count,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int applied_rope_dim,
  const int position_base,
  const float rope_theta,
  const float eps,
  const unsigned long long k_cache_offset_base,
  const unsigned long long v_cache_offset_base) {
  __shared__ float reduction[256];
  __shared__ float inv_norm;

  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int tid = threadIdx.x;
  if (head >= n_heads || token >= token_count) {
    return;
  }

  const int pair_count = applied_rope_dim / 2;
  const std::size_t q_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  const std::size_t q_packed_count = q_count * 2;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv_heads) * static_cast<std::size_t>(head_dim);

  const std::size_t q_token_base = static_cast<std::size_t>(token) * q_count;
  const std::size_t packed_token_base = static_cast<std::size_t>(token) * q_packed_count;
  const std::size_t kv_token_base = static_cast<std::size_t>(token) * kv_count;

  const std::size_t q_head_base = q_token_base + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  const std::size_t packed_head_base = packed_token_base + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim) * 2;

  float q_sq_sum = 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const float qv = q_gate_packed[packed_head_base + static_cast<std::size_t>(d)];
    q_sq_sum += qv * qv;
  }
  reduction[tid] = q_sq_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    inv_norm = rsqrtf((reduction[0] / static_cast<float>(head_dim)) + eps);
  }
  __syncthreads();

  for (int d = tid; d < head_dim; d += blockDim.x) {
    const std::size_t out_idx = q_head_base + static_cast<std::size_t>(d);
    const float qv = q_gate_packed[packed_head_base + static_cast<std::size_t>(d)];
    const float gv = q_gate_packed[packed_head_base + static_cast<std::size_t>(head_dim) + static_cast<std::size_t>(d)];
    out_q[out_idx] = qv * inv_norm * (1.0f + q_norm_weight[d]);
    out_gate[out_idx] = gv;
  }
  __syncthreads();

  const int position = position_base + token;
  for (int pair_idx = tid; pair_idx < pair_count; pair_idx += blockDim.x) {
    const int i0 = pair_idx;
    const int i1 = pair_idx + pair_count;
    if (i1 >= head_dim) {
      continue;
    }
    const float inv_freq = expf(
      -logf(rope_theta) * (static_cast<float>(2 * pair_idx) / static_cast<float>(applied_rope_dim)));
    const float angle = static_cast<float>(position) * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const std::size_t q0_idx = q_head_base + static_cast<std::size_t>(i0);
    const std::size_t q1_idx = q_head_base + static_cast<std::size_t>(i1);
    const float x0 = out_q[q0_idx];
    const float x1 = out_q[q1_idx];
    out_q[q0_idx] = x0 * c - x1 * s;
    out_q[q1_idx] = x1 * c + x0 * s;
  }
  __syncthreads();

  if (head >= n_kv_heads) {
    return;
  }

  const std::size_t kv_head_base = kv_token_base + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);

  float k_sq_sum = 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    const float kv = k_raw[kv_head_base + static_cast<std::size_t>(d)];
    k_sq_sum += kv * kv;
  }
  reduction[tid] = k_sq_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    inv_norm = rsqrtf((reduction[0] / static_cast<float>(head_dim)) + eps);
  }
  __syncthreads();

  const std::size_t cache_token_offset = static_cast<std::size_t>(token) * kv_count;
  const std::size_t k_cache_base = static_cast<std::size_t>(k_cache_offset_base) + cache_token_offset +
                                   static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  const std::size_t v_cache_base = static_cast<std::size_t>(v_cache_offset_base) + cache_token_offset +
                                   static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);

  for (int d = tid; d < head_dim; d += blockDim.x) {
    const std::size_t src_idx = kv_head_base + static_cast<std::size_t>(d);
    k_cache[k_cache_base + static_cast<std::size_t>(d)] = k_raw[src_idx] * inv_norm * (1.0f + k_norm_weight[d]);
    v_cache[v_cache_base + static_cast<std::size_t>(d)] = v_raw[src_idx];
  }
  __syncthreads();

  for (int pair_idx = tid; pair_idx < pair_count; pair_idx += blockDim.x) {
    const int i0 = pair_idx;
    const int i1 = pair_idx + pair_count;
    if (i1 >= head_dim) {
      continue;
    }
    const float inv_freq = expf(
      -logf(rope_theta) * (static_cast<float>(2 * pair_idx) / static_cast<float>(applied_rope_dim)));
    const float angle = static_cast<float>(position) * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const std::size_t k0_idx = k_cache_base + static_cast<std::size_t>(i0);
    const std::size_t k1_idx = k_cache_base + static_cast<std::size_t>(i1);
    const float x0 = k_cache[k0_idx];
    const float x1 = k_cache[k1_idx];
    k_cache[k0_idx] = x0 * c - x1 * s;
    k_cache[k1_idx] = x1 * c + x0 * s;
  }
}

__device__ __forceinline__ float adjust_sampling_logit(
  const float raw,
  const float seen_mask,
  const float repetition_penalty,
  const float temperature) {
  float value = raw;
  if (repetition_penalty > 1.0f && seen_mask > 0.0f) {
    if (value > 0.0f) {
      value /= repetition_penalty;
    } else {
      value *= repetition_penalty;
    }
  }
  if (temperature > 0.0f) {
    value /= temperature;
  }
  return value;
}

__global__ void argmax_blocks_kernel(
  const float * logits,
  const float * seen_token_mask,
  const int vocab_size,
  const float repetition_penalty,
  float * block_values,
  int * block_indices) {
  __shared__ float sh_values[kSamplingArgmaxBlockSize];
  __shared__ int sh_indices[kSamplingArgmaxBlockSize];

  const int tid = threadIdx.x;
  const int block_start = blockIdx.x * kSamplingArgmaxChunkSize;
  const int block_end = min(vocab_size, block_start + kSamplingArgmaxChunkSize);

  float best_value = -1.0e30f;
  int best_index = -1;
  for (int idx = block_start + tid; idx < block_end; idx += blockDim.x) {
    const float value = adjust_sampling_logit(logits[idx], seen_token_mask[idx], repetition_penalty, 0.0f);
    if (value > best_value || (value == best_value && (best_index < 0 || idx < best_index))) {
      best_value = value;
      best_index = idx;
    }
  }

  sh_values[tid] = best_value;
  sh_indices[tid] = best_index;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_value = sh_values[tid + stride];
      const int other_index = sh_indices[tid + stride];
      if (other_value > sh_values[tid] ||
          (other_value == sh_values[tid] && (sh_indices[tid] < 0 || (other_index >= 0 && other_index < sh_indices[tid])))) {
        sh_values[tid] = other_value;
        sh_indices[tid] = other_index;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_values[blockIdx.x] = sh_values[0];
    block_indices[blockIdx.x] = sh_indices[0];
  }
}

__global__ void finalize_argmax_token_kernel(
  const float * block_values,
  const int * block_indices,
  const int block_count,
  float * seen_token_mask,
  const int vocab_size,
  float * out_token) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  float best_value = -1.0e30f;
  int best_index = -1;
  for (int i = 0; i < block_count; ++i) {
    const float value = block_values[i];
    const int idx = block_indices[i];
    if (idx < 0) {
      continue;
    }
    if (value > best_value || (value == best_value && (best_index < 0 || idx < best_index))) {
      best_value = value;
      best_index = idx;
    }
  }

  if (best_index < 0 || best_index >= vocab_size) {
    out_token[0] = -1.0f;
    return;
  }
  seen_token_mask[best_index] = 1.0f;
  out_token[0] = static_cast<float>(best_index);
}

__global__ void topk_blocks_kernel(
  const float * logits,
  const float * seen_token_mask,
  const int vocab_size,
  const float temperature,
  const float repetition_penalty,
  const int top_k,
  float * block_top_values,
  int * block_top_indices) {
  if (threadIdx.x != 0) {
    return;
  }

  const int block_start = blockIdx.x * kSamplingArgmaxChunkSize;
  const int block_end = min(vocab_size, block_start + kSamplingArgmaxChunkSize);
  const int base = blockIdx.x * kGpuSamplingMaxTopK;

  float local_values[kGpuSamplingMaxTopK];
  int local_indices[kGpuSamplingMaxTopK];
  for (int i = 0; i < top_k; ++i) {
    local_values[i] = -1.0e30f;
    local_indices[i] = -1;
  }

  for (int idx = block_start; idx < block_end; ++idx) {
    const float value =
      adjust_sampling_logit(logits[idx], seen_token_mask[idx], repetition_penalty, temperature);
    int min_slot = 0;
    float min_value = local_values[0];
    for (int i = 1; i < top_k; ++i) {
      if (local_values[i] < min_value) {
        min_value = local_values[i];
        min_slot = i;
      }
    }
    if (value > min_value) {
      local_values[min_slot] = value;
      local_indices[min_slot] = idx;
    }
  }

  for (int i = 0; i + 1 < top_k; ++i) {
    int best = i;
    for (int j = i + 1; j < top_k; ++j) {
      if (local_values[j] > local_values[best]) {
        best = j;
      }
    }
    if (best != i) {
      const float tmp_v = local_values[i];
      local_values[i] = local_values[best];
      local_values[best] = tmp_v;
      const int tmp_i = local_indices[i];
      local_indices[i] = local_indices[best];
      local_indices[best] = tmp_i;
    }
  }

  for (int i = 0; i < top_k; ++i) {
    block_top_values[base + i] = local_values[i];
    block_top_indices[base + i] = local_indices[i];
  }
}

__global__ void sample_token_from_block_topk_kernel(
  const float * block_top_values,
  const int * block_top_indices,
  const int block_count,
  const int vocab_size,
  const float top_p,
  const int top_k,
  const float random_u01,
  float * seen_token_mask,
  float * out_token) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  float global_values[kGpuSamplingMaxTopK];
  int global_indices[kGpuSamplingMaxTopK];
  for (int i = 0; i < top_k; ++i) {
    global_values[i] = -1.0e30f;
    global_indices[i] = -1;
  }

  const int candidate_count = block_count * top_k;
  for (int c = 0; c < candidate_count; ++c) {
    const int idx = block_top_indices[c];
    if (idx < 0 || idx >= vocab_size) {
      continue;
    }
    const float value = block_top_values[c];
    int min_slot = 0;
    float min_value = global_values[0];
    for (int i = 1; i < top_k; ++i) {
      if (global_values[i] < min_value) {
        min_value = global_values[i];
        min_slot = i;
      }
    }
    if (value > min_value) {
      global_values[min_slot] = value;
      global_indices[min_slot] = idx;
    }
  }

  for (int i = 0; i + 1 < top_k; ++i) {
    int best = i;
    for (int j = i + 1; j < top_k; ++j) {
      if (global_values[j] > global_values[best]) {
        best = j;
      }
    }
    if (best != i) {
      const float tmp_v = global_values[i];
      global_values[i] = global_values[best];
      global_values[best] = tmp_v;
      const int tmp_i = global_indices[i];
      global_indices[i] = global_indices[best];
      global_indices[best] = tmp_i;
    }
  }

  int valid_count = 0;
  float max_logit = -1.0e30f;
  for (int i = 0; i < top_k; ++i) {
    if (global_indices[i] < 0) {
      continue;
    }
    ++valid_count;
    if (global_values[i] > max_logit) {
      max_logit = global_values[i];
    }
  }
  if (valid_count <= 0) {
    out_token[0] = -1.0f;
    return;
  }

  float probs[kGpuSamplingMaxTopK];
  float prob_sum = 0.0f;
  for (int i = 0; i < valid_count; ++i) {
    probs[i] = expf(global_values[i] - max_logit);
    prob_sum += probs[i];
  }
  if (prob_sum <= 0.0f) {
    const int fallback = global_indices[0];
    if (fallback < 0 || fallback >= vocab_size) {
      out_token[0] = -1.0f;
      return;
    }
    seen_token_mask[fallback] = 1.0f;
    out_token[0] = static_cast<float>(fallback);
    return;
  }
  for (int i = 0; i < valid_count; ++i) {
    probs[i] /= prob_sum;
  }

  int keep = valid_count;
  if (top_p < 1.0f) {
    float cumulative = 0.0f;
    keep = 0;
    for (int i = 0; i < valid_count; ++i) {
      cumulative += probs[i];
      ++keep;
      if (cumulative >= top_p && keep >= 1) {
        break;
      }
    }
    if (keep <= 0) {
      keep = 1;
    }
  }

  float keep_sum = 0.0f;
  for (int i = 0; i < keep; ++i) {
    keep_sum += probs[i];
  }
  if (keep_sum <= 0.0f) {
    const int fallback = global_indices[0];
    if (fallback < 0 || fallback >= vocab_size) {
      out_token[0] = -1.0f;
      return;
    }
    seen_token_mask[fallback] = 1.0f;
    out_token[0] = static_cast<float>(fallback);
    return;
  }

  float clamped_u = random_u01;
  if (clamped_u < 0.0f) {
    clamped_u = 0.0f;
  }
  if (clamped_u >= 1.0f) {
    clamped_u = 0.99999994f;
  }
  const float threshold = clamped_u * keep_sum;
  float cumulative = 0.0f;
  int sampled = global_indices[keep - 1];
  for (int i = 0; i < keep; ++i) {
    cumulative += probs[i];
    if (threshold <= cumulative) {
      sampled = global_indices[i];
      break;
    }
  }

  if (sampled < 0 || sampled >= vocab_size) {
    out_token[0] = -1.0f;
    return;
  }
  seen_token_mask[sampled] = 1.0f;
  out_token[0] = static_cast<float>(sampled);
}

__global__ void full_attention_decode_gqa_kernel_shared(
  const float * q,
  const float * gate,
  const float * k_cache,
  const float * v_cache,
  float * out,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int seq_len) {
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  if (head >= n_heads || seq_len <= 0) {
    return;
  }

  const int n_rep = n_heads / n_kv_heads;
  const int kvh = head / n_rep;
  const int kv_stride = n_kv_heads * head_dim;

  const float * q_ptr = q + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  const float * gate_ptr = gate + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  float * out_ptr = out + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);

  __shared__ float reduction[256];
  __shared__ float sh_alpha;
  __shared__ float sh_beta;
  __shared__ float sh_denom;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const float q_value = (tid < head_dim) ? q_ptr[tid] : 0.0f;
  float acc = 0.0f;
  float running_max = -1.0e30f;
  float running_denom = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    const float * k_ptr = k_cache +
                          (static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_stride)) +
                          (static_cast<std::size_t>(kvh) * static_cast<std::size_t>(head_dim));
    const float * v_ptr = v_cache +
                          (static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_stride)) +
                          (static_cast<std::size_t>(kvh) * static_cast<std::size_t>(head_dim));

    float partial = (tid < head_dim) ? (q_value * k_ptr[tid]) : 0.0f;
    reduction[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float score = reduction[0] * scale;
      const float next_max = fmaxf(running_max, score);
      const float alpha = (running_max <= -1.0e29f) ? 0.0f : expf(running_max - next_max);
      const float beta = expf(score - next_max);
      running_denom = running_denom * alpha + beta;
      running_max = next_max;
      sh_alpha = alpha;
      sh_beta = beta;
      sh_denom = running_denom;
    }
    __syncthreads();
    if (tid < head_dim) {
      acc = acc * sh_alpha + sh_beta * v_ptr[tid];
    }
    __syncthreads();
  }

  if (tid < head_dim) {
    const float denom = (sh_denom > 0.0f) ? sh_denom : 1.0f;
    const float g = gate_ptr[tid];
    const float sig = 1.0f / (1.0f + expf(-g));
    out_ptr[tid] = (acc / denom) * sig;
  }
}

__global__ void full_attention_prefill_gqa_chunk_kernel_shared(
  const float * q,
  const float * gate,
  const float * k_cache,
  const float * v_cache,
  float * out,
  const int token_count,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int position_base) {
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int tid = threadIdx.x;
  if (head >= n_heads || token >= token_count) {
    return;
  }

  const int n_rep = n_heads / n_kv_heads;
  const int kvh = head / n_rep;
  const int kv_stride = n_kv_heads * head_dim;
  const std::size_t q_stride = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);

  const std::size_t token_base = static_cast<std::size_t>(token) * q_stride;
  const float * q_ptr = q + token_base + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  const float * gate_ptr = gate + token_base + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  float * out_ptr = out + token_base + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);

  __shared__ float reduction[256];
  __shared__ float sh_alpha;
  __shared__ float sh_beta;
  __shared__ float sh_denom;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const float q_value = (tid < head_dim) ? q_ptr[tid] : 0.0f;
  float acc = 0.0f;
  float running_max = -1.0e30f;
  float running_denom = 0.0f;

  const int seq_len = position_base + token + 1;
  for (int t = 0; t < seq_len; ++t) {
    const float * k_ptr = k_cache +
                          (static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_stride)) +
                          (static_cast<std::size_t>(kvh) * static_cast<std::size_t>(head_dim));
    const float * v_ptr = v_cache +
                          (static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_stride)) +
                          (static_cast<std::size_t>(kvh) * static_cast<std::size_t>(head_dim));

    float partial = (tid < head_dim) ? (q_value * k_ptr[tid]) : 0.0f;
    reduction[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float score = reduction[0] * scale;
      const float next_max = fmaxf(running_max, score);
      const float alpha = (running_max <= -1.0e29f) ? 0.0f : expf(running_max - next_max);
      const float beta = expf(score - next_max);
      running_denom = running_denom * alpha + beta;
      running_max = next_max;
      sh_alpha = alpha;
      sh_beta = beta;
      sh_denom = running_denom;
    }
    __syncthreads();

    if (tid < head_dim) {
      acc = acc * sh_alpha + sh_beta * v_ptr[tid];
    }
    __syncthreads();
  }

  if (tid < head_dim) {
    const float denom = (sh_denom > 0.0f) ? sh_denom : 1.0f;
    const float g = gate_ptr[tid];
    const float sig = 1.0f / (1.0f + expf(-g));
    out_ptr[tid] = (acc / denom) * sig;
  }
}

__device__ float sigmoidf_device(const float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ float softplusf_device(const float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return expf(x);
  }
  return log1pf(expf(x));
}

__global__ void linear_conv_update_kernel(
  const float * mixed_qkv,
  const float * conv_weights,
  float * conv_state,
  float * conv_out,
  const int channels,
  const int kernel_size) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }

  const int history = kernel_size - 1;
  const float * w_ptr = conv_weights + static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel_size);
  float sum = 0.0f;
  for (int k = 0; k < history; ++k) {
    sum += conv_state[static_cast<std::size_t>(k) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)] * w_ptr[k];
  }
  sum += mixed_qkv[c] * w_ptr[history];
  conv_out[c] = sum * sigmoidf_device(sum);

  for (int r = 0; r + 1 < history; ++r) {
    conv_state[static_cast<std::size_t>(r) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)] =
      conv_state[static_cast<std::size_t>(r + 1) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)];
  }
  if (history > 0) {
    conv_state[static_cast<std::size_t>(history - 1) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)] =
      mixed_qkv[c];
  }
}

__global__ void linear_recurrent_decode_kernel(
  const float * conv_out,
  const float * z,
  const float * b,
  const float * a,
  const float * norm,
  const float * dt_bias,
  const float * ssm_a,
  float * recurrent_state,
  float * out_gated_norm,
  const int linear_num_k_heads,
  const int linear_num_v_heads,
  const int head_k_dim,
  const int head_v_dim,
  const float rms_eps) {
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  if (h >= linear_num_v_heads) {
    return;
  }

  const int q_dim = linear_num_k_heads * head_k_dim;
  const int v_dim = linear_num_v_heads * head_v_dim;
  const int q_base = h * head_k_dim;
  const int v_base = h * head_v_dim;
  if (q_base + head_k_dim > q_dim || v_base + head_v_dim > v_dim) {
    return;
  }

  float * s = recurrent_state +
              static_cast<std::size_t>(h) * static_cast<std::size_t>(head_v_dim) * static_cast<std::size_t>(head_v_dim);
  const float * q_ptr = conv_out + q_base;
  const float * k_ptr = conv_out + q_dim + q_base;
  const float * v_ptr = conv_out + 2 * q_dim + v_base;
  const float * z_ptr = z + v_base;
  float * out_ptr = out_gated_norm + v_base;

  __shared__ float sh_inv_q;
  __shared__ float sh_inv_k;
  __shared__ float sh_alpha;
  __shared__ float sh_beta;
  __shared__ float sh_inv_norm;
  extern __shared__ float shared[];
  float * sh_sk = shared;
  float * sh_core = shared + head_v_dim;

  if (tid == 0) {
    float sq_q = 0.0f;
    float sq_k = 0.0f;
    for (int d = 0; d < head_k_dim; ++d) {
      sq_q += q_ptr[d] * q_ptr[d];
      sq_k += k_ptr[d] * k_ptr[d];
    }
    sh_inv_q = rsqrtf(sq_q + 1.0e-6f) * rsqrtf(static_cast<float>(head_k_dim));
    sh_inv_k = rsqrtf(sq_k + 1.0e-6f);
    sh_beta = sigmoidf_device(b[h]);
    const float pre_gate = softplusf_device(a[h] + dt_bias[h]);
    sh_alpha = expf(pre_gate * ssm_a[h]);
  }
  __syncthreads();

  const int state_count = head_v_dim * head_v_dim;
  for (int idx = tid; idx < state_count; idx += blockDim.x) {
    s[idx] *= sh_alpha;
  }
  __syncthreads();

  for (int i = tid; i < head_v_dim; i += blockDim.x) {
    float sk = 0.0f;
    const float * row = s + static_cast<std::size_t>(i) * static_cast<std::size_t>(head_v_dim);
    for (int j = 0; j < head_v_dim; ++j) {
      sk += row[j] * (k_ptr[j] * sh_inv_k);
    }
    sh_sk[i] = sk;
  }
  __syncthreads();

  for (int i = tid; i < head_v_dim; i += blockDim.x) {
    const float delta = (v_ptr[i] - sh_sk[i]) * sh_beta;
    float * row = s + static_cast<std::size_t>(i) * static_cast<std::size_t>(head_v_dim);
    for (int j = 0; j < head_v_dim; ++j) {
      row[j] += delta * (k_ptr[j] * sh_inv_k);
    }
  }
  __syncthreads();

  for (int i = tid; i < head_v_dim; i += blockDim.x) {
    float core = 0.0f;
    const float * row = s + static_cast<std::size_t>(i) * static_cast<std::size_t>(head_v_dim);
    for (int j = 0; j < head_v_dim; ++j) {
      core += row[j] * (q_ptr[j] * sh_inv_q);
    }
    sh_core[i] = core;
  }
  __syncthreads();

  if (tid == 0) {
    float sq_sum = 0.0f;
    for (int i = 0; i < head_v_dim; ++i) {
      sq_sum += sh_core[i] * sh_core[i];
    }
    sh_inv_norm = rsqrtf(sq_sum / static_cast<float>(head_v_dim) + rms_eps);
  }
  __syncthreads();

  for (int i = tid; i < head_v_dim; i += blockDim.x) {
    const float zv = z_ptr[i];
    out_ptr[i] = sh_core[i] * sh_inv_norm * norm[i] * (zv * sigmoidf_device(zv));
  }
}

__global__ void linear_conv_prefill_chunk_kernel(
  const float * mixed_qkv,
  const float * conv_weights,
  float * conv_state,
  float * conv_out,
  const int token_count,
  const int channels,
  const int kernel_size) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }

  constexpr int kMaxSupportedKernel = 16;
  if (kernel_size > kMaxSupportedKernel) {
    return;
  }

  const int history = kernel_size - 1;
  const float * w_ptr = conv_weights + static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel_size);
  float hist[kMaxSupportedKernel];
  for (int k = 0; k < history; ++k) {
    hist[k] = conv_state[static_cast<std::size_t>(k) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)];
  }

  for (int t = 0; t < token_count; ++t) {
    const float cur = mixed_qkv[static_cast<std::size_t>(t) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)];
    float sum = cur * w_ptr[history];
    for (int k = 0; k < history; ++k) {
      sum += hist[k] * w_ptr[k];
    }
    conv_out[static_cast<std::size_t>(t) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)] =
      sum * sigmoidf_device(sum);
    for (int k = 0; k + 1 < history; ++k) {
      hist[k] = hist[k + 1];
    }
    if (history > 0) {
      hist[history - 1] = cur;
    }
  }

  for (int k = 0; k < history; ++k) {
    conv_state[static_cast<std::size_t>(k) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)] = hist[k];
  }
}

__global__ void linear_recurrent_prefill_chunk_kernel(
  const float * conv_out,
  const float * z,
  const float * b,
  const float * a,
  const float * norm,
  const float * dt_bias,
  const float * ssm_a,
  float * recurrent_state,
  float * out_gated_norm,
  const int token_count,
  const int linear_num_k_heads,
  const int linear_num_v_heads,
  const int head_k_dim,
  const int head_v_dim,
  const float rms_eps) {
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  if (h >= linear_num_v_heads) {
    return;
  }

  const int q_dim = linear_num_k_heads * head_k_dim;
  const int v_dim = linear_num_v_heads * head_v_dim;
  const int channels = q_dim * 2 + v_dim;
  const int q_base = h * head_k_dim;
  const int v_base = h * head_v_dim;
  if (q_base + head_k_dim > q_dim || v_base + head_v_dim > v_dim) {
    return;
  }

  float * s = recurrent_state +
              static_cast<std::size_t>(h) * static_cast<std::size_t>(head_v_dim) * static_cast<std::size_t>(head_v_dim);

  __shared__ float sh_inv_q;
  __shared__ float sh_inv_k;
  __shared__ float sh_alpha;
  __shared__ float sh_beta;
  __shared__ float sh_inv_norm;
  extern __shared__ float shared[];
  float * sh_sk = shared;
  float * sh_core = shared + head_v_dim;

  const int state_count = head_v_dim * head_v_dim;
  for (int t = 0; t < token_count; ++t) {
    const float * conv_t = conv_out + static_cast<std::size_t>(t) * static_cast<std::size_t>(channels);
    const float * q_ptr = conv_t + q_base;
    const float * k_ptr = conv_t + q_dim + q_base;
    const float * v_ptr = conv_t + 2 * q_dim + v_base;
    const float * z_ptr = z + static_cast<std::size_t>(t) * static_cast<std::size_t>(v_dim) + v_base;
    float * out_ptr = out_gated_norm + static_cast<std::size_t>(t) * static_cast<std::size_t>(v_dim) + v_base;

    if (tid == 0) {
      float sq_q = 0.0f;
      float sq_k = 0.0f;
      for (int d = 0; d < head_k_dim; ++d) {
        sq_q += q_ptr[d] * q_ptr[d];
        sq_k += k_ptr[d] * k_ptr[d];
      }
      sh_inv_q = rsqrtf(sq_q + 1.0e-6f) * rsqrtf(static_cast<float>(head_k_dim));
      sh_inv_k = rsqrtf(sq_k + 1.0e-6f);
      sh_beta = sigmoidf_device(b[static_cast<std::size_t>(t) * static_cast<std::size_t>(linear_num_v_heads) + static_cast<std::size_t>(h)]);
      const float pre_gate = softplusf_device(
        a[static_cast<std::size_t>(t) * static_cast<std::size_t>(linear_num_v_heads) + static_cast<std::size_t>(h)] + dt_bias[h]);
      sh_alpha = expf(pre_gate * ssm_a[h]);
    }
    __syncthreads();

    for (int idx = tid; idx < state_count; idx += blockDim.x) {
      s[idx] *= sh_alpha;
    }
    __syncthreads();

    for (int i = tid; i < head_v_dim; i += blockDim.x) {
      float sk = 0.0f;
      const float * row = s + static_cast<std::size_t>(i) * static_cast<std::size_t>(head_v_dim);
      for (int j = 0; j < head_v_dim; ++j) {
        sk += row[j] * (k_ptr[j] * sh_inv_k);
      }
      sh_sk[i] = sk;
    }
    __syncthreads();

    for (int i = tid; i < head_v_dim; i += blockDim.x) {
      const float delta = (v_ptr[i] - sh_sk[i]) * sh_beta;
      float * row = s + static_cast<std::size_t>(i) * static_cast<std::size_t>(head_v_dim);
      for (int j = 0; j < head_v_dim; ++j) {
        row[j] += delta * (k_ptr[j] * sh_inv_k);
      }
    }
    __syncthreads();

    for (int i = tid; i < head_v_dim; i += blockDim.x) {
      float core = 0.0f;
      const float * row = s + static_cast<std::size_t>(i) * static_cast<std::size_t>(head_v_dim);
      for (int j = 0; j < head_v_dim; ++j) {
        core += row[j] * (q_ptr[j] * sh_inv_q);
      }
      sh_core[i] = core;
    }
    __syncthreads();

    if (tid == 0) {
      float sq_sum = 0.0f;
      for (int i = 0; i < head_v_dim; ++i) {
        sq_sum += sh_core[i] * sh_core[i];
      }
      sh_inv_norm = rsqrtf(sq_sum / static_cast<float>(head_v_dim) + rms_eps);
    }
    __syncthreads();

    for (int i = tid; i < head_v_dim; i += blockDim.x) {
      const float zv = z_ptr[i];
      out_ptr[i] = sh_core[i] * sh_inv_norm * norm[i] * (zv * sigmoidf_device(zv));
    }
    __syncthreads();
  }
}

} // namespace

bool begin_inference_session(
  const std::size_t max_input_count,
  const std::size_t max_output_count,
  std::string & error_message) {
  if (max_input_count == 0 || max_output_count == 0) {
    error_message = "Inference session requires non-zero max input/output counts.";
    return false;
  }
  return ensure_session_workspace(max_input_count, max_output_count, error_message);
}

void end_inference_session() {
  if (!g_session.active) {
    return;
  }
  release_session_storage();
  g_session.active = false;
}

bool upload_matrix_f32(
  const std::vector<float> & host_data,
  const int rows,
  const int cols,
  CudaDeviceMatrixF32 & out_matrix,
  std::string & error_message) {
  if (rows <= 0 || cols <= 0) {
    error_message = "Invalid matrix dimensions for CUDA upload.";
    return false;
  }

  const std::size_t expected_count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (host_data.size() != expected_count) {
    error_message = "Host matrix size does not match rows * cols.";
    return false;
  }

  void * device_ptr = nullptr;
  if (!check_cuda(cudaMalloc(&device_ptr, expected_count * sizeof(float)), "cudaMalloc(matrix)", error_message)) {
    return false;
  }
  if (!tracked_memcpy(
        device_ptr,
        host_data.data(),
        expected_count * sizeof(float),
        cudaMemcpyHostToDevice,
        "cudaMemcpy(matrix)",
        error_message)) {
    cudaFree(device_ptr);
    return false;
  }

  out_matrix.data = device_ptr;
  out_matrix.data_bf16 = nullptr;
  out_matrix.rows = rows;
  out_matrix.cols = cols;
  return true;
}

bool upload_matrix_bf16_shadow_from_f32(
  const std::vector<float> & host_data,
  const int rows,
  const int cols,
  CudaDeviceMatrixF32 & matrix,
  std::string & error_message) {
  if (rows <= 0 || cols <= 0) {
    error_message = "Invalid matrix dimensions for CUDA BF16 shadow upload.";
    return false;
  }
  const std::size_t expected_count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (host_data.size() != expected_count) {
    error_message = "Host matrix size does not match rows * cols for BF16 upload.";
    return false;
  }
  if (matrix.rows != rows || matrix.cols != cols) {
    error_message = "BF16 shadow upload dimensions do not match matrix metadata.";
    return false;
  }

  std::vector<std::uint16_t> bf16_values(expected_count);
  for (std::size_t i = 0; i < expected_count; ++i) {
    bf16_values[i] = float_to_bf16_bits(host_data[i]);
  }

  void * bf16_ptr = nullptr;
  if (!check_cuda(cudaMalloc(&bf16_ptr, expected_count * sizeof(std::uint16_t)), "cudaMalloc(matrix_bf16)", error_message)) {
    return false;
  }
  if (!tracked_memcpy(
        bf16_ptr,
        bf16_values.data(),
        expected_count * sizeof(std::uint16_t),
        cudaMemcpyHostToDevice,
        "cudaMemcpy(matrix_bf16)",
        error_message)) {
    cudaFree(bf16_ptr);
    return false;
  }

  if (matrix.data_bf16 != nullptr) {
    cudaFree(matrix.data_bf16);
  }
  matrix.data_bf16 = bf16_ptr;
  return true;
}

void free_matrix_f32(CudaDeviceMatrixF32 & matrix) {
  if (matrix.data != nullptr) {
    cudaFree(matrix.data);
    matrix.data = nullptr;
  }
  if (matrix.data_bf16 != nullptr) {
    cudaFree(matrix.data_bf16);
    matrix.data_bf16 = nullptr;
  }
  matrix.rows = 0;
  matrix.cols = 0;
}

bool run_matvec_f32(
  const CudaDeviceMatrixF32 & matrix,
  const std::vector<float> & input,
  std::vector<float> & output,
  std::string & error_message) {
  if (matrix.data == nullptr || matrix.rows <= 0 || matrix.cols <= 0) {
    error_message = "CUDA matrix is not initialized.";
    return false;
  }
  if (input.size() != static_cast<std::size_t>(matrix.cols)) {
    error_message = "Input size does not match matrix columns.";
    return false;
  }

  if (!ensure_session_workspace(input.size(), static_cast<std::size_t>(matrix.rows), error_message)) {
    return false;
  }
  cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA inference session stream is not initialized.";
    return false;
  }

  if (!tracked_memcpy_async(
        g_session.workspace_input,
        input.data(),
        input.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream,
        "cudaMemcpyAsync(input)",
        error_message)) {
    return false;
  }

  const int block_size = 128;
  const int grid_size = (matrix.rows + block_size - 1) / block_size;
  f32_matvec_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(matrix.data),
    g_session.workspace_input,
    g_session.workspace_output,
    matrix.rows,
    matrix.cols);
  if (!check_cuda(cudaGetLastError(), "f32_matvec_kernel", error_message)) {
    return false;
  }

  output.resize(static_cast<std::size_t>(matrix.rows));
  if (!tracked_memcpy_async(
        output.data(),
        g_session.workspace_output,
        output.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream,
        "cudaMemcpyAsync(output)",
        error_message)) {
    return false;
  }
  if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(matvec)", error_message)) {
    return false;
  }

  return true;
}

bool run_matvec_f32_device(
  const CudaDeviceMatrixF32 & matrix,
  const CudaDeviceBufferF32 & input,
  CudaDeviceBufferF32 & output,
  std::string & error_message) {
  if (matrix.data == nullptr || matrix.rows <= 0 || matrix.cols <= 0) {
    error_message = "CUDA matrix is not initialized.";
    return false;
  }
  if (input.data == nullptr || input.count < static_cast<std::size_t>(matrix.cols)) {
    error_message = "CUDA device input buffer is invalid for matvec.";
    return false;
  }
  if (output.data == nullptr || output.count < static_cast<std::size_t>(matrix.rows)) {
    error_message = "CUDA device output buffer is invalid for matvec.";
    return false;
  }

  if (g_session.prefer_bf16_matvec && matrix.data_bf16 != nullptr) {
    std::string bf16_error;
    if (run_matvec_bf16_device_cublaslt(matrix, input, output, bf16_error)) {
      return true;
    }
  }

  std::string cublas_error;
  if (run_matvec_f32_device_cublaslt(matrix, input, output, cublas_error)) {
    return true;
  }

  const int block_size = 128;
  const int grid_size = (matrix.rows + block_size - 1) / block_size;
  const cudaStream_t stream = active_stream();
  f32_matvec_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(matrix.data),
    static_cast<const float *>(input.data),
    static_cast<float *>(output.data),
    matrix.rows,
    matrix.cols);
  if (check_cuda(cudaGetLastError(), "f32_matvec_kernel(device)", error_message)) {
    return true;
  }
  if (!cublas_error.empty()) {
    error_message += " (fallback reason: " + cublas_error + ")";
  }
  return false;
}

void set_prefer_bf16_matvec(const bool enabled) {
  g_session.prefer_bf16_matvec = enabled;
}

bool gather_matrix_row_f32(
  const CudaDeviceMatrixF32 & matrix,
  const int row_index,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (matrix.data == nullptr || matrix.rows <= 0 || matrix.cols <= 0) {
    error_message = "CUDA matrix is not initialized.";
    return false;
  }
  if (row_index < 0 || row_index >= matrix.rows) {
    error_message = "CUDA matrix row index is out of range.";
    return false;
  }
  if (out.data == nullptr || out.count < static_cast<std::size_t>(matrix.cols)) {
    error_message = "CUDA output buffer is invalid for gather_matrix_row_f32.";
    return false;
  }

  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA inference session stream is not initialized.";
    return false;
  }

  const int block_size = 256;
  const int grid_size = (matrix.cols + block_size - 1) / block_size;
  if (g_session.prefer_bf16_matvec && matrix.data_bf16 != nullptr) {
    gather_matrix_row_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(matrix.data_bf16),
      matrix.cols,
      row_index,
      static_cast<float *>(out.data));
    return check_cuda(cudaGetLastError(), "gather_matrix_row_bf16_kernel", error_message);
  }

  gather_matrix_row_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(matrix.data),
    matrix.cols,
    row_index,
    static_cast<float *>(out.data));
  return check_cuda(cudaGetLastError(), "gather_matrix_row_kernel", error_message);
}

bool gather_matrix_row_f32_from_token_f32(
  const CudaDeviceMatrixF32 & matrix,
  const CudaDeviceBufferF32 & token_id,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (matrix.data == nullptr || matrix.rows <= 0 || matrix.cols <= 0) {
    error_message = "CUDA matrix is not initialized.";
    return false;
  }
  if (token_id.data == nullptr || token_id.count < 1) {
    error_message = "CUDA token buffer is invalid for gather_matrix_row_f32_from_token_f32.";
    return false;
  }
  if (out.data == nullptr || out.count < static_cast<std::size_t>(matrix.cols)) {
    error_message = "CUDA output buffer is invalid for gather_matrix_row_f32_from_token_f32.";
    return false;
  }

  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA inference session stream is not initialized.";
    return false;
  }

  const int block_size = 256;
  const int grid_size = (matrix.cols + block_size - 1) / block_size;
  if (g_session.prefer_bf16_matvec && matrix.data_bf16 != nullptr) {
    gather_matrix_row_from_token_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(matrix.data_bf16),
      matrix.rows,
      matrix.cols,
      static_cast<const float *>(token_id.data),
      static_cast<float *>(out.data));
    return check_cuda(cudaGetLastError(), "gather_matrix_row_from_token_bf16_kernel", error_message);
  }

  gather_matrix_row_from_token_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(matrix.data),
    matrix.rows,
    matrix.cols,
    static_cast<const float *>(token_id.data),
    static_cast<float *>(out.data));
  return check_cuda(cudaGetLastError(), "gather_matrix_row_from_token_kernel", error_message);
}

bool allocate_buffer_f32(
  const std::size_t count,
  CudaDeviceBufferF32 & out_buffer,
  std::string & error_message) {
  if (count == 0) {
    out_buffer.data = nullptr;
    out_buffer.count = 0;
    return true;
  }

  void * device_ptr = nullptr;
  if (!check_cuda(cudaMalloc(&device_ptr, count * sizeof(float)), "cudaMalloc(buffer)", error_message)) {
    return false;
  }
  if (!check_cuda(cudaMemset(device_ptr, 0, count * sizeof(float)), "cudaMemset(buffer)", error_message)) {
    cudaFree(device_ptr);
    return false;
  }

  out_buffer.data = device_ptr;
  out_buffer.count = count;
  return true;
}

void free_buffer_f32(CudaDeviceBufferF32 & buffer) {
  if (buffer.data != nullptr) {
    cudaFree(buffer.data);
    buffer.data = nullptr;
  }
  buffer.count = 0;
}

bool upload_to_buffer_f32(
  const float * host_data,
  const std::size_t count,
  const CudaDeviceBufferF32 & buffer,
  const std::size_t buffer_offset,
  std::string & error_message) {
  if (count == 0) {
    return true;
  }
  if (buffer.data == nullptr) {
    error_message = "CUDA buffer is not initialized.";
    return false;
  }
  if (host_data == nullptr) {
    error_message = "Host data pointer is null.";
    return false;
  }
  if (buffer_offset > buffer.count || count > (buffer.count - buffer_offset)) {
    error_message = "CUDA buffer upload range is out of bounds.";
    return false;
  }

  float * dst = static_cast<float *>(buffer.data) + buffer_offset;
  const cudaStream_t stream = active_stream();
  if (stream != nullptr) {
    return tracked_memcpy_async(
      dst,
      host_data,
      count * sizeof(float),
      cudaMemcpyHostToDevice,
      stream,
      "cudaMemcpyAsync(buffer_upload)",
      error_message);
  }
  return tracked_memcpy(dst, host_data, count * sizeof(float), cudaMemcpyHostToDevice, "cudaMemcpy(buffer_upload)", error_message);
}

bool download_from_buffer_f32(
  const CudaDeviceBufferF32 & buffer,
  const std::size_t count,
  const std::size_t buffer_offset,
  std::vector<float> & out_data,
  std::string & error_message) {
  if (count == 0) {
    out_data.clear();
    return true;
  }
  if (buffer.data == nullptr) {
    error_message = "CUDA buffer is not initialized.";
    return false;
  }
  if (buffer_offset > buffer.count || count > (buffer.count - buffer_offset)) {
    error_message = "CUDA buffer download range is out of bounds.";
    return false;
  }

  out_data.resize(count);
  const float * src = static_cast<const float *>(buffer.data) + buffer_offset;
  const cudaStream_t stream = active_stream();
  if (stream != nullptr) {
    if (!tracked_memcpy_async(
          out_data.data(),
          src,
          count * sizeof(float),
          cudaMemcpyDeviceToHost,
          stream,
          "cudaMemcpyAsync(buffer_download)",
          error_message)) {
      return false;
    }
    return check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(buffer_download)", error_message);
  }
  return tracked_memcpy(
    out_data.data(),
    src,
    count * sizeof(float),
    cudaMemcpyDeviceToHost,
    "cudaMemcpy(buffer_download)",
    error_message);
}

bool run_silu_mul_f32(
  const CudaDeviceBufferF32 & a,
  const CudaDeviceBufferF32 & b,
  const std::size_t count,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (count == 0) {
    return true;
  }
  if (a.data == nullptr || b.data == nullptr || out.data == nullptr) {
    error_message = "CUDA silu_mul requires initialized buffers.";
    return false;
  }
  if (a.count < count || b.count < count || out.count < count) {
    error_message = "CUDA silu_mul buffer range is out of bounds.";
    return false;
  }

  const int block_size = 256;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size));
  const cudaStream_t stream = active_stream();
  silu_mul_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(a.data),
    static_cast<const float *>(b.data),
    static_cast<float *>(out.data),
    count);
  return check_cuda(cudaGetLastError(), "silu_mul_kernel", error_message);
}

bool run_add_f32(
  const CudaDeviceBufferF32 & a,
  const CudaDeviceBufferF32 & b,
  const std::size_t count,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (count == 0) {
    return true;
  }
  if (a.data == nullptr || b.data == nullptr || out.data == nullptr) {
    error_message = "CUDA add requires initialized buffers.";
    return false;
  }
  if (a.count < count || b.count < count || out.count < count) {
    error_message = "CUDA add buffer range is out of bounds.";
    return false;
  }
  const int block_size = 256;
  const int grid_size = static_cast<int>((count + static_cast<std::size_t>(block_size) - 1) / static_cast<std::size_t>(block_size));
  const cudaStream_t stream = active_stream();
  add_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const float *>(a.data),
    static_cast<const float *>(b.data),
    static_cast<float *>(out.data),
    count);
  return check_cuda(cudaGetLastError(), "add_kernel", error_message);
}

bool run_rms_norm_f32(
  const CudaDeviceBufferF32 & input,
  const CudaDeviceBufferF32 & weight,
  const std::size_t count,
  const float eps,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (count == 0) {
    return true;
  }
  if (input.data == nullptr || weight.data == nullptr || out.data == nullptr) {
    error_message = "CUDA rms_norm requires initialized buffers.";
    return false;
  }
  if (input.count < count || weight.count < count || out.count < count) {
    error_message = "CUDA rms_norm buffer range is out of bounds.";
    return false;
  }
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    error_message = "CUDA rms_norm count exceeds kernel limits.";
    return false;
  }

  const cudaStream_t stream = active_stream();
  rms_norm_kernel<<<1, 256, 0, stream>>>(
    static_cast<const float *>(input.data),
    static_cast<const float *>(weight.data),
    static_cast<float *>(out.data),
    static_cast<int>(count),
    eps);
  return check_cuda(cudaGetLastError(), "rms_norm_kernel", error_message);
}

bool run_split_q_gate_f32(
  const CudaDeviceBufferF32 & q_gate_packed,
  const int n_heads,
  const int head_dim,
  CudaDeviceBufferF32 & out_q,
  CudaDeviceBufferF32 & out_gate,
  std::string & error_message) {
  if (n_heads <= 0 || head_dim <= 0) {
    error_message = "Invalid dimensions for split_q_gate.";
    return false;
  }
  const std::size_t packed_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim) * 2;
  const std::size_t q_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  if (q_gate_packed.data == nullptr || out_q.data == nullptr || out_gate.data == nullptr) {
    error_message = "CUDA split_q_gate requires initialized buffers.";
    return false;
  }
  if (q_gate_packed.count < packed_count || out_q.count < q_count || out_gate.count < q_count) {
    error_message = "CUDA split_q_gate buffer range is out of bounds.";
    return false;
  }

  int block_size = 1;
  while (block_size < head_dim && block_size < 256) {
    block_size <<= 1;
  }
  if (block_size < 32) {
    block_size = 32;
  }
  if (block_size > 256) {
    block_size = 256;
  }
  const cudaStream_t stream = active_stream();
  split_q_gate_kernel<<<n_heads, block_size, 0, stream>>>(
    static_cast<const float *>(q_gate_packed.data),
    static_cast<float *>(out_q.data),
    static_cast<float *>(out_gate.data),
    n_heads,
    head_dim);
  return check_cuda(cudaGetLastError(), "split_q_gate_kernel", error_message);
}

bool run_rms_norm_per_head_f32(
  const CudaDeviceBufferF32 & input,
  const CudaDeviceBufferF32 & weight,
  const int n_heads,
  const int head_dim,
  const float eps,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (n_heads <= 0 || head_dim <= 0) {
    error_message = "Invalid dimensions for rms_norm_per_head.";
    return false;
  }
  const std::size_t count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  if (input.data == nullptr || weight.data == nullptr || out.data == nullptr) {
    error_message = "CUDA rms_norm_per_head requires initialized buffers.";
    return false;
  }
  if (input.count < count || weight.count < static_cast<std::size_t>(head_dim) || out.count < count) {
    error_message = "CUDA rms_norm_per_head buffer range is out of bounds.";
    return false;
  }

  int block_size = 1;
  while (block_size < head_dim && block_size < 256) {
    block_size <<= 1;
  }
  if (block_size < 32) {
    block_size = 32;
  }
  if (block_size > 256) {
    block_size = 256;
  }
  const cudaStream_t stream = active_stream();
  rms_norm_per_head_kernel<<<n_heads, block_size, 0, stream>>>(
    static_cast<const float *>(input.data),
    static_cast<const float *>(weight.data),
    static_cast<float *>(out.data),
    n_heads,
    head_dim,
    eps);
  return check_cuda(cudaGetLastError(), "rms_norm_per_head_kernel", error_message);
}

bool run_apply_rope_inplace_f32(
  const CudaDeviceBufferF32 & values,
  const int n_heads,
  const int head_dim,
  const int rope_dim,
  const int position,
  const float rope_theta,
  std::string & error_message) {
  if (n_heads <= 0 || head_dim <= 0 || rope_dim <= 0) {
    error_message = "Invalid dimensions for apply_rope.";
    return false;
  }
  const int applied_rope_dim = std::min(rope_dim, head_dim);
  const std::size_t count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  if (values.data == nullptr || values.count < count) {
    error_message = "CUDA apply_rope buffer range is out of bounds.";
    return false;
  }
  const int pair_count = applied_rope_dim / 2;
  if (pair_count <= 0) {
    return true;
  }
  const int block_size = 128;
  const int grid_y = (pair_count + block_size - 1) / block_size;
  const dim3 grid(n_heads, grid_y, 1);
  const cudaStream_t stream = active_stream();
  rope_inplace_kernel<<<grid, block_size, 0, stream>>>(
    static_cast<float *>(values.data),
    n_heads,
    head_dim,
    applied_rope_dim,
    position,
    rope_theta);
  return check_cuda(cudaGetLastError(), "rope_inplace_kernel", error_message);
}

bool run_prepare_full_attention_qkv_f32(
  const CudaDeviceBufferF32 & q_gate_packed,
  const CudaDeviceBufferF32 & k_raw,
  const CudaDeviceBufferF32 & v_raw,
  const CudaDeviceBufferF32 & q_norm_weight,
  const CudaDeviceBufferF32 & k_norm_weight,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int rope_dim,
  const int position,
  const float rope_theta,
  const float eps,
  CudaDeviceBufferF32 & out_q,
  CudaDeviceBufferF32 & out_gate,
  CudaDeviceBufferF32 & k_cache,
  const std::size_t k_cache_offset,
  CudaDeviceBufferF32 & v_cache,
  const std::size_t v_cache_offset,
  std::string & error_message) {
  if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
    error_message = "Invalid dimensions for full-attention QKV prepare.";
    return false;
  }
  if (n_heads % n_kv_heads != 0) {
    error_message = "n_heads must be divisible by n_kv_heads for GQA.";
    return false;
  }
  const int applied_rope_dim = std::min(rope_dim, head_dim);
  const std::size_t q_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  const std::size_t q_packed_count = q_count * 2;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv_heads) * static_cast<std::size_t>(head_dim);
  if (q_gate_packed.data == nullptr || k_raw.data == nullptr || v_raw.data == nullptr || q_norm_weight.data == nullptr ||
      k_norm_weight.data == nullptr || out_q.data == nullptr || out_gate.data == nullptr || k_cache.data == nullptr ||
      v_cache.data == nullptr) {
    error_message = "CUDA full-attention QKV prepare requires initialized buffers.";
    return false;
  }
  if (q_gate_packed.count < q_packed_count || k_raw.count < kv_count || v_raw.count < kv_count || out_q.count < q_count ||
      out_gate.count < q_count || q_norm_weight.count < static_cast<std::size_t>(head_dim) ||
      k_norm_weight.count < static_cast<std::size_t>(head_dim)) {
    error_message = "CUDA full-attention QKV prepare buffer range is out of bounds.";
    return false;
  }
  if (k_cache_offset > k_cache.count || kv_count > (k_cache.count - k_cache_offset) ||
      v_cache_offset > v_cache.count || kv_count > (v_cache.count - v_cache_offset)) {
    error_message = "CUDA full-attention QKV prepare cache range is out of bounds.";
    return false;
  }

  constexpr int block_size = 256;
  const cudaStream_t stream = active_stream();
  prepare_full_attention_qkv_kernel<<<n_heads, block_size, 0, stream>>>(
    static_cast<const float *>(q_gate_packed.data),
    static_cast<const float *>(k_raw.data),
    static_cast<const float *>(v_raw.data),
    static_cast<const float *>(q_norm_weight.data),
    static_cast<const float *>(k_norm_weight.data),
    static_cast<float *>(out_q.data),
    static_cast<float *>(out_gate.data),
    static_cast<float *>(k_cache.data),
    static_cast<float *>(v_cache.data),
    n_heads,
    n_kv_heads,
    head_dim,
    applied_rope_dim,
    position,
    rope_theta,
    eps,
    static_cast<unsigned long long>(k_cache_offset),
    static_cast<unsigned long long>(v_cache_offset));
  return check_cuda(cudaGetLastError(), "prepare_full_attention_qkv_kernel", error_message);
}

bool run_prepare_full_attention_qkv_prefill_chunk_f32(
  const CudaDeviceBufferF32 & q_gate_packed,
  const CudaDeviceBufferF32 & k_raw,
  const CudaDeviceBufferF32 & v_raw,
  const CudaDeviceBufferF32 & q_norm_weight,
  const CudaDeviceBufferF32 & k_norm_weight,
  const int token_count,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int rope_dim,
  const int position_base,
  const float rope_theta,
  const float eps,
  CudaDeviceBufferF32 & out_q,
  CudaDeviceBufferF32 & out_gate,
  CudaDeviceBufferF32 & k_cache,
  const std::size_t k_cache_offset_base,
  CudaDeviceBufferF32 & v_cache,
  const std::size_t v_cache_offset_base,
  std::string & error_message) {
  if (token_count <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
    error_message = "Invalid dimensions for full-attention prefill QKV prepare.";
    return false;
  }
  if (n_heads % n_kv_heads != 0) {
    error_message = "n_heads must be divisible by n_kv_heads for GQA.";
    return false;
  }

  const int applied_rope_dim = std::min(rope_dim, head_dim);
  const std::size_t q_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  const std::size_t q_packed_count = q_count * 2;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv_heads) * static_cast<std::size_t>(head_dim);
  const std::size_t q_total = static_cast<std::size_t>(token_count) * q_count;
  const std::size_t q_packed_total = static_cast<std::size_t>(token_count) * q_packed_count;
  const std::size_t kv_total = static_cast<std::size_t>(token_count) * kv_count;

  if (q_gate_packed.data == nullptr || k_raw.data == nullptr || v_raw.data == nullptr || q_norm_weight.data == nullptr ||
      k_norm_weight.data == nullptr || out_q.data == nullptr || out_gate.data == nullptr || k_cache.data == nullptr ||
      v_cache.data == nullptr) {
    error_message = "CUDA full-attention prefill QKV prepare requires initialized buffers.";
    return false;
  }
  if (q_gate_packed.count < q_packed_total || k_raw.count < kv_total || v_raw.count < kv_total || out_q.count < q_total ||
      out_gate.count < q_total || q_norm_weight.count < static_cast<std::size_t>(head_dim) ||
      k_norm_weight.count < static_cast<std::size_t>(head_dim)) {
    error_message = "CUDA full-attention prefill QKV prepare buffer range is out of bounds.";
    return false;
  }
  if (k_cache_offset_base > k_cache.count || kv_total > (k_cache.count - k_cache_offset_base) ||
      v_cache_offset_base > v_cache.count || kv_total > (v_cache.count - v_cache_offset_base)) {
    error_message = "CUDA full-attention prefill QKV prepare cache range is out of bounds.";
    return false;
  }

  constexpr int block_size = 256;
  const cudaStream_t stream = active_stream();
  const dim3 grid(static_cast<unsigned int>(n_heads), static_cast<unsigned int>(token_count), 1);
  prepare_full_attention_qkv_prefill_chunk_kernel<<<grid, block_size, 0, stream>>>(
    static_cast<const float *>(q_gate_packed.data),
    static_cast<const float *>(k_raw.data),
    static_cast<const float *>(v_raw.data),
    static_cast<const float *>(q_norm_weight.data),
    static_cast<const float *>(k_norm_weight.data),
    static_cast<float *>(out_q.data),
    static_cast<float *>(out_gate.data),
    static_cast<float *>(k_cache.data),
    static_cast<float *>(v_cache.data),
    token_count,
    n_heads,
    n_kv_heads,
    head_dim,
    applied_rope_dim,
    position_base,
    rope_theta,
    eps,
    static_cast<unsigned long long>(k_cache_offset_base),
    static_cast<unsigned long long>(v_cache_offset_base));
  return check_cuda(cudaGetLastError(), "prepare_full_attention_qkv_prefill_chunk_kernel", error_message);
}

bool copy_buffer_f32(
  const CudaDeviceBufferF32 & src,
  const std::size_t count,
  const std::size_t src_offset,
  const CudaDeviceBufferF32 & dst,
  const std::size_t dst_offset,
  std::string & error_message) {
  if (count == 0) {
    return true;
  }
  if (src.data == nullptr || dst.data == nullptr) {
    error_message = "CUDA copy_buffer requires initialized buffers.";
    return false;
  }
  if (src_offset > src.count || count > (src.count - src_offset) || dst_offset > dst.count || count > (dst.count - dst_offset)) {
    error_message = "CUDA copy_buffer range is out of bounds.";
    return false;
  }
  const float * src_ptr = static_cast<const float *>(src.data) + src_offset;
  float * dst_ptr = static_cast<float *>(dst.data) + dst_offset;
  const cudaStream_t stream = active_stream();
  if (stream != nullptr) {
    return tracked_memcpy_async(
      dst_ptr,
      src_ptr,
      count * sizeof(float),
      cudaMemcpyDeviceToDevice,
      stream,
      "cudaMemcpyAsync(buffer_copy_d2d)",
      error_message);
  }
  return tracked_memcpy(
    dst_ptr,
    src_ptr,
    count * sizeof(float),
    cudaMemcpyDeviceToDevice,
    "cudaMemcpy(buffer_copy_d2d)",
    error_message);
}

bool run_full_attention_decode_gqa(
  const CudaDeviceBufferF32 & q,
  const CudaDeviceBufferF32 & gate,
  const CudaDeviceBufferF32 & k_cache,
  const CudaDeviceBufferF32 & v_cache,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int seq_len,
  CudaDeviceBufferF32 & out,
  CudaDeviceBufferF32 & scratch_scores,
  std::string & error_message) {
  if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || seq_len <= 0) {
    error_message = "Invalid dimensions for full attention decode.";
    return false;
  }
  if ((n_heads % n_kv_heads) != 0) {
    error_message = "Invalid GQA ratio for full attention decode.";
    return false;
  }

  const std::size_t q_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv_heads) * static_cast<std::size_t>(head_dim);
  const std::size_t kv_count = static_cast<std::size_t>(seq_len) * kv_stride;
  if (q.data == nullptr || gate.data == nullptr || k_cache.data == nullptr || v_cache.data == nullptr ||
      out.data == nullptr) {
    error_message = "One or more CUDA buffers are not initialized for full attention decode.";
    return false;
  }
  if (q.count < q_count || gate.count < q_count || out.count < q_count || k_cache.count < kv_count ||
      v_cache.count < kv_count) {
    error_message = "CUDA buffer range is out of bounds for full attention decode.";
    return false;
  }
  (void)scratch_scores;

  int block_size = 1;
  while (block_size < head_dim && block_size < 256) {
    block_size <<= 1;
  }
  if (block_size < 32) {
    block_size = 32;
  }
  if (block_size > 256) {
    block_size = 256;
  }

  const cudaStream_t stream = active_stream();
  full_attention_decode_gqa_kernel_shared<<<n_heads, block_size, 0, stream>>>(
    static_cast<const float *>(q.data),
    static_cast<const float *>(gate.data),
    static_cast<const float *>(k_cache.data),
    static_cast<const float *>(v_cache.data),
    static_cast<float *>(out.data),
    n_heads,
    n_kv_heads,
    head_dim,
    seq_len);
  return check_cuda(cudaGetLastError(), "full_attention_decode_gqa_kernel_shared", error_message);
}

bool run_full_attention_prefill_gqa_chunk(
  const CudaDeviceBufferF32 & q,
  const CudaDeviceBufferF32 & gate,
  const CudaDeviceBufferF32 & k_cache,
  const CudaDeviceBufferF32 & v_cache,
  const int token_count,
  const int n_heads,
  const int n_kv_heads,
  const int head_dim,
  const int position_base,
  CudaDeviceBufferF32 & out,
  std::string & error_message) {
  if (token_count <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || position_base < 0) {
    error_message = "Invalid dimensions for full attention prefill chunk.";
    return false;
  }
  if ((n_heads % n_kv_heads) != 0) {
    error_message = "Invalid GQA ratio for full attention prefill chunk.";
    return false;
  }

  const std::size_t q_count = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(head_dim);
  const std::size_t q_total = static_cast<std::size_t>(token_count) * q_count;
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv_heads) * static_cast<std::size_t>(head_dim);
  const std::size_t kv_required =
    static_cast<std::size_t>(position_base + token_count) * static_cast<std::size_t>(kv_stride);
  if (q.data == nullptr || gate.data == nullptr || k_cache.data == nullptr || v_cache.data == nullptr || out.data == nullptr) {
    error_message = "One or more CUDA buffers are not initialized for full attention prefill chunk.";
    return false;
  }
  if (q.count < q_total || gate.count < q_total || out.count < q_total || k_cache.count < kv_required || v_cache.count < kv_required) {
    error_message = "CUDA buffer range is out of bounds for full attention prefill chunk.";
    return false;
  }

  int block_size = 1;
  while (block_size < head_dim && block_size < 256) {
    block_size <<= 1;
  }
  if (block_size < 32) {
    block_size = 32;
  }
  if (block_size > 256) {
    block_size = 256;
  }

  const cudaStream_t stream = active_stream();
  const dim3 grid(static_cast<unsigned int>(n_heads), static_cast<unsigned int>(token_count), 1);
  full_attention_prefill_gqa_chunk_kernel_shared<<<grid, block_size, 0, stream>>>(
    static_cast<const float *>(q.data),
    static_cast<const float *>(gate.data),
    static_cast<const float *>(k_cache.data),
    static_cast<const float *>(v_cache.data),
    static_cast<float *>(out.data),
    token_count,
    n_heads,
    n_kv_heads,
    head_dim,
    position_base);
  return check_cuda(cudaGetLastError(), "full_attention_prefill_gqa_chunk_kernel_shared", error_message);
}

bool run_linear_attention_decode(
  const CudaDeviceBufferF32 & mixed_qkv,
  const CudaDeviceBufferF32 & z,
  const CudaDeviceBufferF32 & b,
  const CudaDeviceBufferF32 & a,
  const CudaDeviceMatrixF32 & conv1d,
  const CudaDeviceBufferF32 & norm,
  const CudaDeviceBufferF32 & dt_bias,
  const CudaDeviceBufferF32 & ssm_a,
  const int linear_kernel,
  const int linear_num_k_heads,
  const int linear_num_v_heads,
  const int linear_head_k_dim,
  const int linear_head_v_dim,
  const float rms_eps,
  CudaDeviceBufferF32 & conv_state,
  CudaDeviceBufferF32 & recurrent_state,
  CudaDeviceBufferF32 & scratch_conv_out,
  CudaDeviceBufferF32 & out_gated_norm,
  std::string & error_message) {
  if (linear_kernel <= 1 || linear_num_k_heads <= 0 || linear_num_v_heads <= 0 || linear_head_k_dim <= 0 ||
      linear_head_v_dim <= 0) {
    error_message = "Invalid dimensions for linear attention decode.";
    return false;
  }
  if (linear_head_k_dim != linear_head_v_dim) {
    error_message = "Linear attention decode expects head_k_dim == head_v_dim.";
    return false;
  }

  const int linear_q_dim = linear_num_k_heads * linear_head_k_dim;
  const int linear_v_dim = linear_num_v_heads * linear_head_v_dim;
  const int linear_conv_channels = linear_q_dim * 2 + linear_v_dim;
  const int conv_hist = linear_kernel - 1;

  if (mixed_qkv.data == nullptr || z.data == nullptr || b.data == nullptr || a.data == nullptr ||
      conv1d.data == nullptr || norm.data == nullptr || dt_bias.data == nullptr || ssm_a.data == nullptr ||
      conv_state.data == nullptr || recurrent_state.data == nullptr || scratch_conv_out.data == nullptr ||
      out_gated_norm.data == nullptr) {
    error_message = "One or more CUDA buffers are not initialized for linear attention decode.";
    return false;
  }
  if (mixed_qkv.count < static_cast<std::size_t>(linear_conv_channels) || z.count < static_cast<std::size_t>(linear_v_dim) ||
      b.count < static_cast<std::size_t>(linear_num_v_heads) || a.count < static_cast<std::size_t>(linear_num_v_heads) ||
      norm.count < static_cast<std::size_t>(linear_head_v_dim) || dt_bias.count < static_cast<std::size_t>(linear_num_v_heads) ||
      ssm_a.count < static_cast<std::size_t>(linear_num_v_heads) ||
      conv_state.count < static_cast<std::size_t>(conv_hist * linear_conv_channels) ||
      recurrent_state.count <
        static_cast<std::size_t>(linear_num_v_heads) * static_cast<std::size_t>(linear_head_v_dim) *
          static_cast<std::size_t>(linear_head_v_dim) ||
      scratch_conv_out.count < static_cast<std::size_t>(linear_conv_channels) ||
      out_gated_norm.count < static_cast<std::size_t>(linear_v_dim)) {
    error_message = "CUDA buffer range is out of bounds for linear attention decode.";
    return false;
  }
  if (conv1d.rows != linear_conv_channels || conv1d.cols != linear_kernel) {
    error_message = "conv1d device matrix dimensions do not match linear attention dimensions.";
    return false;
  }

  const cudaStream_t stream = active_stream();
  const int conv_block = 256;
  const int conv_grid = (linear_conv_channels + conv_block - 1) / conv_block;
  linear_conv_update_kernel<<<conv_grid, conv_block, 0, stream>>>(
    static_cast<const float *>(mixed_qkv.data),
    static_cast<const float *>(conv1d.data),
    static_cast<float *>(conv_state.data),
    static_cast<float *>(scratch_conv_out.data),
    linear_conv_channels,
    linear_kernel);
  if (!check_cuda(cudaGetLastError(), "linear_conv_update_kernel", error_message)) {
    return false;
  }

  int block_size = 1;
  while (block_size < linear_head_v_dim && block_size < 256) {
    block_size <<= 1;
  }
  if (block_size < 32) {
    block_size = 32;
  }
  if (block_size > 256) {
    block_size = 256;
  }
  const std::size_t shared_bytes = static_cast<std::size_t>(2 * linear_head_v_dim) * sizeof(float);
  linear_recurrent_decode_kernel<<<linear_num_v_heads, block_size, shared_bytes, stream>>>(
    static_cast<const float *>(scratch_conv_out.data),
    static_cast<const float *>(z.data),
    static_cast<const float *>(b.data),
    static_cast<const float *>(a.data),
    static_cast<const float *>(norm.data),
    static_cast<const float *>(dt_bias.data),
    static_cast<const float *>(ssm_a.data),
    static_cast<float *>(recurrent_state.data),
    static_cast<float *>(out_gated_norm.data),
    linear_num_k_heads,
    linear_num_v_heads,
    linear_head_k_dim,
    linear_head_v_dim,
    rms_eps);
  return check_cuda(cudaGetLastError(), "linear_recurrent_decode_kernel", error_message);
}

bool run_linear_attention_prefill_chunk(
  const CudaDeviceBufferF32 & mixed_qkv,
  const CudaDeviceBufferF32 & z,
  const CudaDeviceBufferF32 & b,
  const CudaDeviceBufferF32 & a,
  const CudaDeviceMatrixF32 & conv1d,
  const CudaDeviceBufferF32 & norm,
  const CudaDeviceBufferF32 & dt_bias,
  const CudaDeviceBufferF32 & ssm_a,
  const int token_count,
  const int linear_kernel,
  const int linear_num_k_heads,
  const int linear_num_v_heads,
  const int linear_head_k_dim,
  const int linear_head_v_dim,
  const float rms_eps,
  CudaDeviceBufferF32 & conv_state,
  CudaDeviceBufferF32 & recurrent_state,
  CudaDeviceBufferF32 & scratch_conv_out,
  CudaDeviceBufferF32 & out_gated_norm,
  std::string & error_message) {
  if (token_count <= 0 || linear_kernel <= 1 || linear_num_k_heads <= 0 || linear_num_v_heads <= 0 ||
      linear_head_k_dim <= 0 || linear_head_v_dim <= 0) {
    error_message = "Invalid dimensions for linear attention prefill chunk.";
    return false;
  }
  if (linear_head_k_dim != linear_head_v_dim) {
    error_message = "Linear attention prefill chunk expects head_k_dim == head_v_dim.";
    return false;
  }

  const int linear_q_dim = linear_num_k_heads * linear_head_k_dim;
  const int linear_v_dim = linear_num_v_heads * linear_head_v_dim;
  const int linear_conv_channels = linear_q_dim * 2 + linear_v_dim;
  const int conv_hist = linear_kernel - 1;
  constexpr int kMaxSupportedKernel = 16;
  if (linear_kernel > kMaxSupportedKernel) {
    error_message = "Linear attention prefill chunk currently supports kernel_size <= 16.";
    return false;
  }

  const std::size_t mixed_total = static_cast<std::size_t>(token_count) * static_cast<std::size_t>(linear_conv_channels);
  const std::size_t z_total = static_cast<std::size_t>(token_count) * static_cast<std::size_t>(linear_v_dim);
  const std::size_t head_total = static_cast<std::size_t>(token_count) * static_cast<std::size_t>(linear_num_v_heads);
  if (mixed_qkv.data == nullptr || z.data == nullptr || b.data == nullptr || a.data == nullptr || conv1d.data == nullptr ||
      norm.data == nullptr || dt_bias.data == nullptr || ssm_a.data == nullptr || conv_state.data == nullptr ||
      recurrent_state.data == nullptr || scratch_conv_out.data == nullptr || out_gated_norm.data == nullptr) {
    error_message = "One or more CUDA buffers are not initialized for linear attention prefill chunk.";
    return false;
  }
  if (mixed_qkv.count < mixed_total || z.count < z_total || b.count < head_total || a.count < head_total ||
      norm.count < static_cast<std::size_t>(linear_head_v_dim) || dt_bias.count < static_cast<std::size_t>(linear_num_v_heads) ||
      ssm_a.count < static_cast<std::size_t>(linear_num_v_heads) ||
      conv_state.count < static_cast<std::size_t>(conv_hist * linear_conv_channels) ||
      recurrent_state.count <
        static_cast<std::size_t>(linear_num_v_heads) * static_cast<std::size_t>(linear_head_v_dim) *
          static_cast<std::size_t>(linear_head_v_dim) ||
      scratch_conv_out.count < mixed_total || out_gated_norm.count < z_total) {
    error_message = "CUDA buffer range is out of bounds for linear attention prefill chunk.";
    return false;
  }
  if (conv1d.rows != linear_conv_channels || conv1d.cols != linear_kernel) {
    error_message = "conv1d device matrix dimensions do not match linear attention prefill chunk dimensions.";
    return false;
  }

  const cudaStream_t stream = active_stream();
  const int conv_block = 256;
  const int conv_grid = (linear_conv_channels + conv_block - 1) / conv_block;
  linear_conv_prefill_chunk_kernel<<<conv_grid, conv_block, 0, stream>>>(
    static_cast<const float *>(mixed_qkv.data),
    static_cast<const float *>(conv1d.data),
    static_cast<float *>(conv_state.data),
    static_cast<float *>(scratch_conv_out.data),
    token_count,
    linear_conv_channels,
    linear_kernel);
  if (!check_cuda(cudaGetLastError(), "linear_conv_prefill_chunk_kernel", error_message)) {
    return false;
  }

  int block_size = 1;
  while (block_size < linear_head_v_dim && block_size < 256) {
    block_size <<= 1;
  }
  if (block_size < 32) {
    block_size = 32;
  }
  if (block_size > 256) {
    block_size = 256;
  }
  const std::size_t shared_bytes = static_cast<std::size_t>(2 * linear_head_v_dim) * sizeof(float);
  linear_recurrent_prefill_chunk_kernel<<<linear_num_v_heads, block_size, shared_bytes, stream>>>(
    static_cast<const float *>(scratch_conv_out.data),
    static_cast<const float *>(z.data),
    static_cast<const float *>(b.data),
    static_cast<const float *>(a.data),
    static_cast<const float *>(norm.data),
    static_cast<const float *>(dt_bias.data),
    static_cast<const float *>(ssm_a.data),
    static_cast<float *>(recurrent_state.data),
    static_cast<float *>(out_gated_norm.data),
    token_count,
    linear_num_k_heads,
    linear_num_v_heads,
    linear_head_k_dim,
    linear_head_v_dim,
    rms_eps);
  return check_cuda(cudaGetLastError(), "linear_recurrent_prefill_chunk_kernel", error_message);
}

bool sample_token_from_logits_f32_device_to_buffer(
  const CudaDeviceBufferF32 & logits,
  const CudaDeviceBufferF32 & seen_token_mask,
  const int vocab_size,
  const float temperature,
  const float top_p,
  const int top_k,
  const float repetition_penalty,
  const float random_u01,
  const CudaDeviceBufferF32 & out_token,
  std::string & error_message) {
  if (logits.data == nullptr || seen_token_mask.data == nullptr) {
    error_message = "CUDA sampling requires initialized device buffers.";
    return false;
  }
  if (vocab_size <= 0 || logits.count < static_cast<std::size_t>(vocab_size) ||
      seen_token_mask.count < static_cast<std::size_t>(vocab_size)) {
    error_message = "CUDA sampling buffer size is out of bounds.";
    return false;
  }
  if (top_p <= 0.0f || top_p > 1.0f) {
    error_message = "CUDA sampling requires top_p in (0, 1].";
    return false;
  }
  if (repetition_penalty <= 0.0f) {
    error_message = "CUDA sampling requires repetition_penalty > 0.";
    return false;
  }
  if (temperature < 0.0f) {
    error_message = "CUDA sampling requires temperature >= 0.";
    return false;
  }
  if (temperature > 0.0f) {
    if (top_k <= 0 || top_k > kGpuSamplingMaxTopK) {
      error_message = "CUDA sampling requires top_k in [1, 64] when temperature > 0.";
      return false;
    }
  }
  if (out_token.data == nullptr || out_token.count < 1) {
    error_message = "CUDA sampling output token buffer is not initialized.";
    return false;
  }
  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA sampling stream is not initialized.";
    return false;
  }

  if (!ensure_sampling_workspace(vocab_size, error_message)) {
    return false;
  }

  const int block_count = std::max(1, (vocab_size + kSamplingArgmaxChunkSize - 1) / kSamplingArgmaxChunkSize);

  if (temperature <= 0.0f) {
    argmax_blocks_kernel<<<block_count, kSamplingArgmaxBlockSize, 0, stream>>>(
      static_cast<const float *>(logits.data),
      static_cast<const float *>(seen_token_mask.data),
      vocab_size,
      repetition_penalty,
      g_session.sampling_block_values,
      g_session.sampling_block_indices);
    if (!check_cuda(cudaGetLastError(), "argmax_blocks_kernel", error_message)) {
      return false;
    }
    finalize_argmax_token_kernel<<<1, 1, 0, stream>>>(
      g_session.sampling_block_values,
      g_session.sampling_block_indices,
      block_count,
      static_cast<float *>(seen_token_mask.data),
      vocab_size,
      static_cast<float *>(out_token.data));
    if (!check_cuda(cudaGetLastError(), "finalize_argmax_token_kernel", error_message)) {
      return false;
    }
  } else {
    topk_blocks_kernel<<<block_count, 1, 0, stream>>>(
      static_cast<const float *>(logits.data),
      static_cast<const float *>(seen_token_mask.data),
      vocab_size,
      temperature,
      repetition_penalty,
      top_k,
      g_session.sampling_block_topk_values,
      g_session.sampling_block_topk_indices);
    if (!check_cuda(cudaGetLastError(), "topk_blocks_kernel", error_message)) {
      return false;
    }
    sample_token_from_block_topk_kernel<<<1, 1, 0, stream>>>(
      g_session.sampling_block_topk_values,
      g_session.sampling_block_topk_indices,
      block_count,
      vocab_size,
      top_p,
      top_k,
      random_u01,
      static_cast<float *>(seen_token_mask.data),
      static_cast<float *>(out_token.data));
    if (!check_cuda(cudaGetLastError(), "sample_token_from_block_topk_kernel", error_message)) {
      return false;
    }
  }

  return true;
}

bool sample_token_from_logits_f32_device(
  const CudaDeviceBufferF32 & logits,
  const CudaDeviceBufferF32 & seen_token_mask,
  const int vocab_size,
  const float temperature,
  const float top_p,
  const int top_k,
  const float repetition_penalty,
  const float random_u01,
  const CudaDeviceBufferF32 & topk_values_scratch,
  const CudaDeviceBufferF32 & topk_indices_scratch,
  int & out_token,
  std::string & error_message) {
  (void)topk_values_scratch;
  (void)topk_indices_scratch;

  if (g_session.workspace_output == nullptr || g_session.workspace_output_count < 1) {
    error_message = "CUDA sampling output workspace is not initialized.";
    return false;
  }
  CudaDeviceBufferF32 sampled_token_device;
  sampled_token_device.data = g_session.workspace_output;
  sampled_token_device.count = 1;

  if (!sample_token_from_logits_f32_device_to_buffer(
        logits,
        seen_token_mask,
        vocab_size,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        random_u01,
        sampled_token_device,
        error_message)) {
    return false;
  }

  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA sampling stream is not initialized.";
    return false;
  }

  float token_value = -1.0f;
  if (!tracked_memcpy_async(
        &token_value,
        sampled_token_device.data,
        sizeof(float),
        cudaMemcpyDeviceToHost,
        stream,
        "cudaMemcpyAsync(sampled_token)",
        error_message)) {
    return false;
  }
  if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(sampled_token)", error_message)) {
    return false;
  }

  out_token = static_cast<int>(token_value);
  if (out_token < 0 || out_token >= vocab_size) {
    error_message = "CUDA sampling failed to produce a valid token id.";
    return false;
  }
  return true;
}

bool begin_stream_capture(std::string & error_message) {
  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA stream capture requires an active inference session stream.";
    return false;
  }
  return check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture", error_message);
}

bool end_stream_capture(
  CudaCapturedGraph & out_graph,
  std::string & error_message) {
  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA stream capture requires an active inference session stream.";
    return false;
  }

  cudaGraph_t captured_graph = nullptr;
  if (!check_cuda(cudaStreamEndCapture(stream, &captured_graph), "cudaStreamEndCapture", error_message)) {
    return false;
  }
  if (captured_graph == nullptr) {
    error_message = "CUDA stream capture returned a null graph.";
    return false;
  }

  cudaGraphExec_t graph_exec = nullptr;
  if (!check_cuda(
        cudaGraphInstantiate(&graph_exec, captured_graph, nullptr, nullptr, 0),
        "cudaGraphInstantiate",
        error_message)) {
    cudaGraphDestroy(captured_graph);
    return false;
  }

  free_captured_graph(out_graph);
  out_graph.graph = captured_graph;
  out_graph.exec = graph_exec;
  out_graph.ready = true;
  return true;
}

bool launch_captured_graph(
  const CudaCapturedGraph & graph,
  std::string & error_message) {
  if (!graph.ready || graph.exec == nullptr) {
    error_message = "CUDA captured graph is not initialized.";
    return false;
  }

  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA stream capture requires an active inference session stream.";
    return false;
  }
  return check_cuda(
    cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph.exec), stream),
    "cudaGraphLaunch",
    error_message);
}

bool synchronize_stream(std::string & error_message) {
  const cudaStream_t stream = active_stream();
  if (stream == nullptr) {
    error_message = "CUDA stream is not initialized.";
    return false;
  }
  return check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", error_message);
}

void free_captured_graph(CudaCapturedGraph & graph) {
  if (graph.exec != nullptr) {
    cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph.exec));
    graph.exec = nullptr;
  }
  if (graph.graph != nullptr) {
    cudaGraphDestroy(static_cast<cudaGraph_t>(graph.graph));
    graph.graph = nullptr;
  }
  graph.ready = false;
}

void reset_transfer_stats() {
  g_transfer_stats = CudaTransferStats{};
}

void get_transfer_stats(CudaTransferStats & out_stats) {
  out_stats = g_transfer_stats;
}

} // namespace qwen35x::cuda

#endif
