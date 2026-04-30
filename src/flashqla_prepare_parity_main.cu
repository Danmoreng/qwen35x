#include "variant.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace wmma = nvcuda::wmma;

namespace {

constexpr int kChunk = 64;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kTiles = kChunk / kWmmaM;
constexpr int kLowerTiles = kTiles * (kTiles + 1) / 2;

struct Options {
  float tolerance = 0.0f;
};

struct CaseResult {
  int rows = 0;
  float max_a = 0.0f;
  float max_p = 0.0f;
  float max_g = 0.0f;
  float max_beta = 0.0f;
  std::size_t max_a_index = 0;
  std::size_t max_p_index = 0;
  std::size_t max_g_index = 0;
  std::size_t max_beta_index = 0;
};

template <bool SkipUpperTiles>
__global__ void prepare_kernel(
  const float *qkv_f32,
  const float *beta_buf,
  const float *alpha_buf,
  __nv_bfloat16 *a_out,
  __nv_bfloat16 *p_out,
  float *g_out,
  float *beta_out,
  int rows) {
  const int h = blockIdx.x;
  const int group = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / 32;

  extern __shared__ float smem[];
  float *kk_shared = smem;
  float *qk_shared = kk_shared + kChunk * kChunk;
  float *a_shared = qk_shared + kChunk * kChunk;
  float *g_shared = a_shared + kChunk * kChunk;
  float *beta_shared = g_shared + kChunk;
  __nv_bfloat16 *q_bf16 = reinterpret_cast<__nv_bfloat16 *>(beta_shared + kChunk);
  __nv_bfloat16 *k_bf16 = q_bf16 + kChunk * DN_KEY;

  const std::size_t ws_id = static_cast<std::size_t>(h) * DN_VAL_GROUPS + group;
  __nv_bfloat16 *a_head = a_out + ws_id * kChunk * kChunk;
  __nv_bfloat16 *p_head = p_out + ws_id * kChunk * kChunk;
  float *g_head = g_out + ws_id * kChunk;
  float *beta_head = beta_out + ws_id * kChunk;
  const int gate_head = h * DN_VAL_GROUPS + group;

  for (int idx = tid; idx < kChunk * DN_KEY; idx += blockDim.x) {
    const int t = idx / DN_KEY;
    const int k = idx - t * DN_KEY;
    float qv = 0.0f;
    float kv = 0.0f;
    if (t < rows) {
      const float *base = qkv_f32 + t * DN_CONV_CH;
      qv = base[h * DN_KEY + k];
      kv = base[DN_QK_SIZE + h * DN_KEY + k];
    }
    q_bf16[idx] = __float2bfloat16(qv);
    k_bf16[idx] = __float2bfloat16(kv);
  }

  if (tid == 0) {
    float g = 0.0f;
    for (int t = 0; t < kChunk; ++t) {
      float gt = 0.0f;
      float bt = 0.0f;
      if (t < rows) {
        const int gate_off = t * DN_GATE + gate_head;
        g += logf(fmaxf(alpha_buf[gate_off], 1.0e-20f));
        gt = g;
        bt = beta_buf[gate_off];
      }
      g_shared[t] = gt;
      beta_shared[t] = bt;
      g_head[t] = gt;
      beta_head[t] = bt;
    }
  }
  __syncthreads();

  const int tile_count = SkipUpperTiles ? kLowerTiles : kTiles * kTiles;
  for (int tile = warp; tile < tile_count; tile += blockDim.x / 32) {
    int tile_m = tile / kTiles;
    int tile_n = tile - tile_m * kTiles;
    if constexpr (SkipUpperTiles) {
      int remaining = tile;
      tile_m = 0;
      while (remaining > tile_m) {
        remaining -= tile_m + 1;
        ++tile_m;
      }
      tile_n = remaining;
    }

    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> k_a_frag;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::col_major> k_b_frag;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> kk_acc;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> qk_acc;
    wmma::fill_fragment(kk_acc, 0.0f);
    wmma::fill_fragment(qk_acc, 0.0f);

    for (int k0 = 0; k0 < DN_KEY; k0 += kWmmaK) {
      const __nv_bfloat16 *q_tile = q_bf16 + (tile_m * kWmmaM) * DN_KEY + k0;
      const __nv_bfloat16 *k_a_tile = k_bf16 + (tile_m * kWmmaM) * DN_KEY + k0;
      const __nv_bfloat16 *k_b_tile = k_bf16 + (tile_n * kWmmaN) * DN_KEY + k0;
      wmma::load_matrix_sync(q_frag, q_tile, DN_KEY);
      wmma::load_matrix_sync(k_a_frag, k_a_tile, DN_KEY);
      wmma::load_matrix_sync(k_b_frag, k_b_tile, DN_KEY);
      wmma::mma_sync(qk_acc, q_frag, k_b_frag, qk_acc);
      wmma::mma_sync(kk_acc, k_a_frag, k_b_frag, kk_acc);
    }

    wmma::store_matrix_sync(kk_shared + (tile_m * kWmmaM) * kChunk + tile_n * kWmmaN, kk_acc, kChunk, wmma::mem_row_major);
    wmma::store_matrix_sync(qk_shared + (tile_m * kWmmaM) * kChunk + tile_n * kWmmaN, qk_acc, kChunk, wmma::mem_row_major);
  }
  __syncthreads();

  for (int idx = tid; idx < kChunk * kChunk; idx += blockDim.x) {
    const int t = idx / kChunk;
    const int j = idx - t * kChunk;
    float lower = 0.0f;
    if (t < rows && j < t) {
      lower = beta_shared[t] * expf(g_shared[t] - g_shared[j]) * kk_shared[t * kChunk + j];
    }
    kk_shared[idx] = lower;
    a_shared[idx] = (j == t) ? 1.0f : 0.0f;
  }
  __syncthreads();

  for (int t = 1; t < rows; ++t) {
    for (int j = tid; j < t; j += blockDim.x) {
      float sum = kk_shared[t * kChunk + j];
      for (int m = j + 1; m < t; ++m) {
        sum += kk_shared[t * kChunk + m] * a_shared[m * kChunk + j];
      }
      a_shared[t * kChunk + j] = -sum;
    }
    __syncthreads();
  }

  for (int idx = tid; idx < kChunk * kChunk; idx += blockDim.x) {
    const int t = idx / kChunk;
    const int j = idx - t * kChunk;
    float p = 0.0f;
    if (t < rows && j <= t) {
      p = expf(g_shared[t] - g_shared[j]) * qk_shared[t * kChunk + j];
    }
    a_head[idx] = __float2bfloat16(a_shared[idx]);
    p_head[idx] = __float2bfloat16(p);
  }
}

float deterministic_value(std::uint32_t i, float scale, float bias) {
  const float a = std::sin(static_cast<float>((i * 17u + 13u) % 257u) * 0.071f);
  const float b = std::cos(static_cast<float>((i * 31u + 7u) % 193u) * 0.053f);
  return bias + scale * (0.75f * a + 0.25f * b);
}

bool parse_args(int argc, char **argv, Options &options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tolerance" && i + 1 < argc) {
      options.tolerance = std::strtof(argv[++i], nullptr);
    } else if (arg == "--help") {
      std::cout << "usage: qwen35x_flashqla_prepare_parity [--tolerance <float>]\n";
      return false;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      std::exit(2);
    }
  }
  return true;
}

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::cerr << what << " failed: " << cudaGetErrorString(status) << "\n";
    std::exit(1);
  }
}

CaseResult run_case(int rows) {
  const std::size_t qkv_count = static_cast<std::size_t>(kChunk) * DN_CONV_CH;
  const std::size_t gate_count = static_cast<std::size_t>(kChunk) * DN_GATE;
  const std::size_t heads = static_cast<std::size_t>(DN_HEADS) * DN_VAL_GROUPS;
  const std::size_t mat_count = heads * kChunk * kChunk;
  const std::size_t vec_count = heads * kChunk;

  std::vector<float> h_qkv(qkv_count);
  std::vector<float> h_alpha(gate_count);
  std::vector<float> h_beta(gate_count);
  std::vector<__nv_bfloat16> h_a_ref(mat_count);
  std::vector<__nv_bfloat16> h_p_ref(mat_count);
  std::vector<__nv_bfloat16> h_a_candidate(mat_count);
  std::vector<__nv_bfloat16> h_p_candidate(mat_count);
  std::vector<float> h_g_ref(vec_count);
  std::vector<float> h_beta_ref(vec_count);
  std::vector<float> h_g_candidate(vec_count);
  std::vector<float> h_beta_candidate(vec_count);

  for (std::size_t i = 0; i < h_qkv.size(); ++i) {
    h_qkv[i] = deterministic_value(static_cast<std::uint32_t>(i), 0.035f, 0.0f);
  }
  for (std::size_t i = 0; i < h_alpha.size(); ++i) {
    h_alpha[i] = 0.93f + 0.05f * deterministic_value(static_cast<std::uint32_t>(i + 1009), 1.0f, 0.0f);
    h_beta[i] = 0.55f + 0.20f * deterministic_value(static_cast<std::uint32_t>(i + 2003), 1.0f, 0.0f);
  }

  float *d_qkv = nullptr;
  float *d_alpha = nullptr;
  float *d_beta = nullptr;
  __nv_bfloat16 *d_a_ref = nullptr;
  __nv_bfloat16 *d_p_ref = nullptr;
  __nv_bfloat16 *d_a_candidate = nullptr;
  __nv_bfloat16 *d_p_candidate = nullptr;
  float *d_g_ref = nullptr;
  float *d_beta_ref = nullptr;
  float *d_g_candidate = nullptr;
  float *d_beta_candidate = nullptr;

  check_cuda(cudaMalloc(&d_qkv, h_qkv.size() * sizeof(float)), "cudaMalloc qkv");
  check_cuda(cudaMalloc(&d_alpha, h_alpha.size() * sizeof(float)), "cudaMalloc alpha");
  check_cuda(cudaMalloc(&d_beta, h_beta.size() * sizeof(float)), "cudaMalloc beta");
  check_cuda(cudaMalloc(&d_a_ref, h_a_ref.size() * sizeof(__nv_bfloat16)), "cudaMalloc a ref");
  check_cuda(cudaMalloc(&d_p_ref, h_p_ref.size() * sizeof(__nv_bfloat16)), "cudaMalloc p ref");
  check_cuda(cudaMalloc(&d_a_candidate, h_a_candidate.size() * sizeof(__nv_bfloat16)), "cudaMalloc a candidate");
  check_cuda(cudaMalloc(&d_p_candidate, h_p_candidate.size() * sizeof(__nv_bfloat16)), "cudaMalloc p candidate");
  check_cuda(cudaMalloc(&d_g_ref, h_g_ref.size() * sizeof(float)), "cudaMalloc g ref");
  check_cuda(cudaMalloc(&d_beta_ref, h_beta_ref.size() * sizeof(float)), "cudaMalloc beta ref");
  check_cuda(cudaMalloc(&d_g_candidate, h_g_candidate.size() * sizeof(float)), "cudaMalloc g candidate");
  check_cuda(cudaMalloc(&d_beta_candidate, h_beta_candidate.size() * sizeof(float)), "cudaMalloc beta candidate");

  check_cuda(cudaMemcpy(d_qkv, h_qkv.data(), h_qkv.size() * sizeof(float), cudaMemcpyHostToDevice), "copy qkv");
  check_cuda(cudaMemcpy(d_alpha, h_alpha.data(), h_alpha.size() * sizeof(float), cudaMemcpyHostToDevice), "copy alpha");
  check_cuda(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(float), cudaMemcpyHostToDevice), "copy beta");

  const std::size_t shared_bytes =
    static_cast<std::size_t>(3 * kChunk * kChunk + 2 * kChunk) * sizeof(float) +
    static_cast<std::size_t>(2 * kChunk * DN_KEY) * sizeof(__nv_bfloat16);
  const dim3 grid(DN_HEADS, DN_VAL_GROUPS);
  check_cuda(
    cudaFuncSetAttribute(prepare_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)),
    "set prepare reference shared memory");
  check_cuda(
    cudaFuncSetAttribute(prepare_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)),
    "set prepare candidate shared memory");
  prepare_kernel<false><<<grid, 512, shared_bytes>>>(d_qkv, d_beta, d_alpha, d_a_ref, d_p_ref, d_g_ref, d_beta_ref, rows);
  check_cuda(cudaGetLastError(), "launch prepare reference");
  prepare_kernel<true><<<grid, 512, shared_bytes>>>(d_qkv, d_beta, d_alpha, d_a_candidate, d_p_candidate, d_g_candidate, d_beta_candidate, rows);
  check_cuda(cudaGetLastError(), "launch prepare candidate");
  check_cuda(cudaDeviceSynchronize(), "prepare kernels");

  check_cuda(cudaMemcpy(h_a_ref.data(), d_a_ref, h_a_ref.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy a ref");
  check_cuda(cudaMemcpy(h_p_ref.data(), d_p_ref, h_p_ref.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy p ref");
  check_cuda(cudaMemcpy(h_a_candidate.data(), d_a_candidate, h_a_candidate.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy a candidate");
  check_cuda(cudaMemcpy(h_p_candidate.data(), d_p_candidate, h_p_candidate.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy p candidate");
  check_cuda(cudaMemcpy(h_g_ref.data(), d_g_ref, h_g_ref.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy g ref");
  check_cuda(cudaMemcpy(h_beta_ref.data(), d_beta_ref, h_beta_ref.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy beta ref");
  check_cuda(cudaMemcpy(h_g_candidate.data(), d_g_candidate, h_g_candidate.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy g candidate");
  check_cuda(cudaMemcpy(h_beta_candidate.data(), d_beta_candidate, h_beta_candidate.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy beta candidate");

  CaseResult result;
  result.rows = rows;
  for (std::size_t i = 0; i < h_a_ref.size(); ++i) {
    const float diff = std::fabs(__bfloat162float(h_a_ref[i]) - __bfloat162float(h_a_candidate[i]));
    if (diff > result.max_a) {
      result.max_a = diff;
      result.max_a_index = i;
    }
  }
  for (std::size_t i = 0; i < h_p_ref.size(); ++i) {
    const float diff = std::fabs(__bfloat162float(h_p_ref[i]) - __bfloat162float(h_p_candidate[i]));
    if (diff > result.max_p) {
      result.max_p = diff;
      result.max_p_index = i;
    }
  }
  for (std::size_t i = 0; i < h_g_ref.size(); ++i) {
    const float diff = std::fabs(h_g_ref[i] - h_g_candidate[i]);
    if (diff > result.max_g) {
      result.max_g = diff;
      result.max_g_index = i;
    }
  }
  for (std::size_t i = 0; i < h_beta_ref.size(); ++i) {
    const float diff = std::fabs(h_beta_ref[i] - h_beta_candidate[i]);
    if (diff > result.max_beta) {
      result.max_beta = diff;
      result.max_beta_index = i;
    }
  }

  cudaFree(d_beta_candidate);
  cudaFree(d_g_candidate);
  cudaFree(d_beta_ref);
  cudaFree(d_g_ref);
  cudaFree(d_p_candidate);
  cudaFree(d_a_candidate);
  cudaFree(d_p_ref);
  cudaFree(d_a_ref);
  cudaFree(d_beta);
  cudaFree(d_alpha);
  cudaFree(d_qkv);

  return result;
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parse_args(argc, argv, options)) {
    return 0;
  }

  const std::vector<int> rows_cases = {1, 17, 63, 64};
  bool failed = false;

  std::cout << "FlashQLA prepare workspace parity\n";
  std::cout << "heads=" << DN_HEADS << " groups=" << DN_VAL_GROUPS << " key=" << DN_KEY
            << " chunk=" << kChunk << " tolerance=" << options.tolerance << "\n";

  for (const int rows : rows_cases) {
    const CaseResult result = run_case(rows);
    const bool pass =
      result.max_a <= options.tolerance &&
      result.max_p <= options.tolerance &&
      result.max_g <= options.tolerance &&
      result.max_beta <= options.tolerance;
    failed = failed || !pass;
    std::cout << "rows=" << std::setw(2) << rows
              << " max_a=" << std::scientific << std::setprecision(8) << result.max_a
              << " max_a_index=" << std::defaultfloat << result.max_a_index
              << " max_p=" << std::scientific << std::setprecision(8) << result.max_p
              << " max_p_index=" << std::defaultfloat << result.max_p_index
              << " max_g=" << std::scientific << std::setprecision(8) << result.max_g
              << " max_g_index=" << std::defaultfloat << result.max_g_index
              << " max_beta=" << std::scientific << std::setprecision(8) << result.max_beta
              << " max_beta_index=" << std::defaultfloat << result.max_beta_index
              << " parity=" << (pass ? "PASS" : "FAIL") << "\n";
  }

  if (failed) {
    return 1;
  }
  std::cout << "FlashQLA prepare workspace parity complete.\n";
  return 0;
}
