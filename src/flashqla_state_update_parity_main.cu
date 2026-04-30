#include "variant.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace wmma = nvcuda::wmma;

namespace {

constexpr int kChunk = 64;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

struct Options {
  float tolerance = 1.0e-6f;
};

struct CaseResult {
  int rows = 0;
  float max_abs = 0.0f;
  float max_abs_vs_f32 = 0.0f;
  std::size_t max_index = 0;
  std::size_t max_index_vs_f32 = 0;
};

__device__ __forceinline__ float state_update_scalar_value(
  const __nv_bfloat16 *k,
  const __nv_bfloat16 *vnew,
  const float *initial_state,
  const float state_scale,
  const int rows,
  const int h,
  const int c,
  const int d) {
  float acc = state_scale * initial_state[(h * DN_VAL + c) * DN_KEY + d];
#pragma unroll
  for (int t = 0; t < kChunk; ++t) {
    if (t < rows) {
      const float kid = __bfloat162float(k[(h * kChunk + t) * DN_KEY + d]);
      const float vid = __bfloat162float(vnew[(h * kChunk + t) * DN_VAL + c]);
      acc += kid * vid;
    }
  }
  return acc;
}

__global__ void state_update_reference_kernel(
  const __nv_bfloat16 *k,
  const __nv_bfloat16 *vnew,
  const float *initial_state,
  const float *state_scale,
  float *out,
  const int rows) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = DN_HEADS * DN_VAL * DN_KEY;
  if (idx >= total) {
    return;
  }

  const int d = idx % DN_KEY;
  const int c = (idx / DN_KEY) % DN_VAL;
  const int h = idx / (DN_KEY * DN_VAL);
  out[idx] = state_update_scalar_value(k, vnew, initial_state, state_scale[h], rows, h, c, d);
}

__global__ void state_update_f32_reference_kernel(
  const float *k,
  const float *vnew,
  const float *initial_state,
  const float *state_scale,
  float *out,
  const int rows) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = DN_HEADS * DN_VAL * DN_KEY;
  if (idx >= total) {
    return;
  }

  const int d = idx % DN_KEY;
  const int c = (idx / DN_KEY) % DN_VAL;
  const int h = idx / (DN_KEY * DN_VAL);
  float acc = state_scale[h] * initial_state[idx];
  for (int t = 0; t < kChunk; ++t) {
    if (t < rows) {
      acc += k[(h * kChunk + t) * DN_KEY + d] * vnew[(h * kChunk + t) * DN_VAL + c];
    }
  }
  out[idx] = acc;
}

__global__ void state_update_candidate_kernel(
  const __nv_bfloat16 *k,
  const __nv_bfloat16 *vnew,
  const float *initial_state,
  const float *state_scale,
  float *out,
  const int rows) {
  const int h = blockIdx.x;
  const int key_tile = blockIdx.y;
  const int val_tile = blockIdx.z;
  const int lane = threadIdx.x;
  if (lane >= 32) {
    return;
  }

  __shared__ __nv_bfloat16 a_tile[kWmmaM * kWmmaK];
  __shared__ __nv_bfloat16 b_tile[kWmmaK * kWmmaN];
  __shared__ float out_tile[kWmmaM * kWmmaN];

  wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  for (int tk = 0; tk < kChunk; tk += kWmmaK) {
    for (int idx = lane; idx < kWmmaM * kWmmaK; idx += 32) {
      const int local_d = idx / kWmmaK;
      const int local_t = idx - local_d * kWmmaK;
      const int d = key_tile * kWmmaM + local_d;
      const int t = tk + local_t;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (d < DN_KEY && t < rows) {
        value = k[(h * kChunk + t) * DN_KEY + d];
      }
      a_tile[idx] = value;
    }
    for (int idx = lane; idx < kWmmaK * kWmmaN; idx += 32) {
      const int local_t = idx / kWmmaN;
      const int local_c = idx - local_t * kWmmaN;
      const int t = tk + local_t;
      const int c = val_tile * kWmmaN + local_c;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (t < rows && c < DN_VAL) {
        value = vnew[(h * kChunk + t) * DN_VAL + c];
      }
      b_tile[idx] = value;
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, a_tile, kWmmaK);
    wmma::load_matrix_sync(b_frag, b_tile, kWmmaN);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    __syncthreads();
  }

  wmma::store_matrix_sync(out_tile, acc_frag, kWmmaN, wmma::mem_row_major);
  __syncthreads();

  for (int idx = lane; idx < kWmmaM * kWmmaN; idx += 32) {
    const int local_d = idx / kWmmaN;
    const int local_c = idx - local_d * kWmmaN;
    const int d = key_tile * kWmmaM + local_d;
    const int c = val_tile * kWmmaN + local_c;
    if (d < DN_KEY && c < DN_VAL) {
      const int state_idx = (h * DN_VAL + c) * DN_KEY + d;
      out[state_idx] = state_scale[h] * initial_state[state_idx] + out_tile[idx];
    }
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
      std::cout << "usage: qwen35x_flashqla_state_update_parity [--tolerance <float>]\n";
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
  const std::size_t k_count = static_cast<std::size_t>(DN_HEADS) * kChunk * DN_KEY;
  const std::size_t v_count = static_cast<std::size_t>(DN_HEADS) * kChunk * DN_VAL;
  const std::size_t state_count = static_cast<std::size_t>(DN_HEADS) * DN_VAL * DN_KEY;

  std::vector<float> h_k_f32(k_count);
  std::vector<float> h_vnew_f32(v_count);
  std::vector<__nv_bfloat16> h_k(k_count);
  std::vector<__nv_bfloat16> h_vnew(v_count);
  std::vector<float> h_initial(state_count);
  std::vector<float> h_scale(DN_HEADS);
  std::vector<float> h_ref_f32(state_count);
  std::vector<float> h_ref(state_count);
  std::vector<float> h_candidate(state_count);

  for (std::size_t i = 0; i < h_k.size(); ++i) {
    h_k_f32[i] = deterministic_value(static_cast<std::uint32_t>(i), 0.035f, 0.0f);
    h_k[i] = __float2bfloat16(h_k_f32[i]);
  }
  for (std::size_t i = 0; i < h_vnew.size(); ++i) {
    h_vnew_f32[i] = deterministic_value(static_cast<std::uint32_t>(i + 1009), 0.045f, 0.0f);
    h_vnew[i] = __float2bfloat16(h_vnew_f32[i]);
  }
  for (std::size_t i = 0; i < h_initial.size(); ++i) {
    h_initial[i] = deterministic_value(static_cast<std::uint32_t>(i + 2003), 0.020f, 0.0f);
  }
  for (int h = 0; h < DN_HEADS; ++h) {
    h_scale[h] = 0.78f + 0.011f * static_cast<float>(h % 7);
  }

  __nv_bfloat16 *d_k = nullptr;
  __nv_bfloat16 *d_vnew = nullptr;
  float *d_k_f32 = nullptr;
  float *d_vnew_f32 = nullptr;
  float *d_initial = nullptr;
  float *d_scale = nullptr;
  float *d_ref_f32 = nullptr;
  float *d_ref = nullptr;
  float *d_candidate = nullptr;

  check_cuda(cudaMalloc(&d_k_f32, h_k_f32.size() * sizeof(float)), "cudaMalloc k f32");
  check_cuda(cudaMalloc(&d_vnew_f32, h_vnew_f32.size() * sizeof(float)), "cudaMalloc vnew f32");
  check_cuda(cudaMalloc(&d_k, h_k.size() * sizeof(__nv_bfloat16)), "cudaMalloc k");
  check_cuda(cudaMalloc(&d_vnew, h_vnew.size() * sizeof(__nv_bfloat16)), "cudaMalloc vnew");
  check_cuda(cudaMalloc(&d_initial, h_initial.size() * sizeof(float)), "cudaMalloc initial");
  check_cuda(cudaMalloc(&d_scale, h_scale.size() * sizeof(float)), "cudaMalloc scale");
  check_cuda(cudaMalloc(&d_ref_f32, h_ref_f32.size() * sizeof(float)), "cudaMalloc ref f32");
  check_cuda(cudaMalloc(&d_ref, h_ref.size() * sizeof(float)), "cudaMalloc ref");
  check_cuda(cudaMalloc(&d_candidate, h_candidate.size() * sizeof(float)), "cudaMalloc candidate");

  check_cuda(cudaMemcpy(d_k_f32, h_k_f32.data(), h_k_f32.size() * sizeof(float), cudaMemcpyHostToDevice), "copy k f32");
  check_cuda(cudaMemcpy(d_vnew_f32, h_vnew_f32.data(), h_vnew_f32.size() * sizeof(float), cudaMemcpyHostToDevice), "copy vnew f32");
  check_cuda(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy k");
  check_cuda(cudaMemcpy(d_vnew, h_vnew.data(), h_vnew.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy vnew");
  check_cuda(cudaMemcpy(d_initial, h_initial.data(), h_initial.size() * sizeof(float), cudaMemcpyHostToDevice), "copy initial");
  check_cuda(cudaMemcpy(d_scale, h_scale.data(), h_scale.size() * sizeof(float), cudaMemcpyHostToDevice), "copy scale");

  const int threads = 256;
  const int total = static_cast<int>(state_count);
  const int blocks = (total + threads - 1) / threads;
  state_update_f32_reference_kernel<<<blocks, threads>>>(d_k_f32, d_vnew_f32, d_initial, d_scale, d_ref_f32, rows);
  check_cuda(cudaGetLastError(), "launch f32 reference");
  state_update_reference_kernel<<<blocks, threads>>>(d_k, d_vnew, d_initial, d_scale, d_ref, rows);
  check_cuda(cudaGetLastError(), "launch reference");
  state_update_candidate_kernel<<<dim3(DN_HEADS, (DN_KEY + kWmmaM - 1) / kWmmaM, (DN_VAL + kWmmaN - 1) / kWmmaN), 32>>>(
    d_k, d_vnew, d_initial, d_scale, d_candidate, rows);
  check_cuda(cudaGetLastError(), "launch candidate");
  check_cuda(cudaDeviceSynchronize(), "state update kernels");

  check_cuda(cudaMemcpy(h_ref_f32.data(), d_ref_f32, h_ref_f32.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy ref f32");
  check_cuda(cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy ref");
  check_cuda(cudaMemcpy(h_candidate.data(), d_candidate, h_candidate.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy candidate");

  CaseResult result;
  result.rows = rows;
  for (std::size_t i = 0; i < h_ref.size(); ++i) {
    const float diff = std::fabs(h_ref[i] - h_candidate[i]);
    if (diff > result.max_abs) {
      result.max_abs = diff;
      result.max_index = i;
    }
    const float f32_diff = std::fabs(h_ref_f32[i] - h_candidate[i]);
    if (f32_diff > result.max_abs_vs_f32) {
      result.max_abs_vs_f32 = f32_diff;
      result.max_index_vs_f32 = i;
    }
  }

  cudaFree(d_candidate);
  cudaFree(d_ref);
  cudaFree(d_ref_f32);
  cudaFree(d_scale);
  cudaFree(d_initial);
  cudaFree(d_vnew);
  cudaFree(d_k);
  cudaFree(d_vnew_f32);
  cudaFree(d_k_f32);

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

  std::cout << "FlashQLA state-update parity\n";
  std::cout << "heads=" << DN_HEADS << " key=" << DN_KEY << " value=" << DN_VAL
            << " chunk=" << kChunk << " tolerance=" << options.tolerance << "\n";

  for (const int rows : rows_cases) {
    const CaseResult result = run_case(rows);
    const bool pass = result.max_abs <= options.tolerance;
    failed = failed || !pass;
    std::cout << "rows=" << std::setw(2) << rows
              << " max_abs=" << std::scientific << std::setprecision(8) << result.max_abs
              << " max_index=" << std::defaultfloat << result.max_index
              << " max_abs_vs_f32=" << std::scientific << std::setprecision(8) << result.max_abs_vs_f32
              << " max_index_vs_f32=" << std::defaultfloat << result.max_index_vs_f32
              << " parity=" << (pass ? "PASS" : "FAIL") << "\n";
  }

  if (failed) {
    return 1;
  }
  std::cout << "FlashQLA state-update parity complete.\n";
  return 0;
}
