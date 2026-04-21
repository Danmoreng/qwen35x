# Full GPU Inference Checklist (Qwen3.5)

This checklist is the execution plan to move `--infer-gpu` from CUDA-hybrid to a fully GPU-resident decode/prefill path.

Progress snapshot (April 2026):
- Fully device-resident per-layer decode flow is implemented.
- Full-logits per-token D2H copies are removed.
- Current measured transfer footprint is near control-path scale (`~3-4 bytes D2H per forward token`).
- Current open bottlenecks are sampling optimization depth and prefill specialization.

Scope:
- Model family: Qwen3.5 (current profile `qwen3_5_0_8b.profile.json`)
- Runtime mode: `qwen35x --infer-gpu`
- Goal: remove CPU math and avoid per-token host/device transfers in the hot loop

---

## 0. Baseline and Guardrails

- [x] Capture baseline metrics before changes
  - Command: `.\build\qwen35x.exe --infer-gpu --hf-model-dir models\qwen3.5-0.8b --chat-user "Tell me a short joke." --max-new-tokens 128 --max-context 256 --seed 123 --temperature 0`
  - Record: load time, decode time, tokens/s, generated token ids
- [ ] Add a fixed-prompt parity set for regression checks
  - Create `scripts/bench/parity_prompts.txt` with at least 10 prompts
  - Include the known edge case prompt style with trailing space
- [x] Define acceptance thresholds
  - Token parity vs CPU reference when `temperature=0` and same seed
  - No quality regression on chat prompts with default sampling
  - Throughput target: improve decode tokens/s by removing CPU-bound steps

Files:
- `src/main.cpp`
- `src/runtime/reference_inference.cpp`
- `scripts/`

---

## 1. Instrumentation First (Find Remaining CPU Bottlenecks)

- [x] Add fine-grained timing sections in the decode loop
  - Split timings: embedding, attention block(s), MLP, logits, sampling, stop checks
- [x] Count and log host/device transfer bytes per token
  - Track `cudaMemcpy` calls in inference path and aggregate totals
- [x] Add a `--profile-json <path>` output option for machine-readable benchmarking

Files:
- `src/runtime/reference_inference.cpp`
- `src/runtime/cuda_inference.cu`
- `include/qwen35x/runtime/reference_inference.h`
- `src/main.cpp`

Exit criteria:
- [x] We can point to exact top 3 stalls in per-token decode.

---

## 2. Kill Global MatVec Workspace and Blocking Sync

Current issue:
- `src/runtime/cuda_inference.cu` uses global reusable buffers plus `cudaDeviceSynchronize()` in matvec path.

Tasks:
- [x] Replace global workspace (`g_workspace_input/output`) with per-runtime context buffers
- [x] Introduce explicit CUDA stream ownership in inference runtime context
- [x] Remove unconditional `cudaDeviceSynchronize()` from per-op path
- [x] Use async copies/kernels where possible and synchronize only at required boundaries

Files:
- `include/qwen35x/runtime/cuda_inference.h`
- `src/runtime/cuda_inference.cu`
- `src/runtime/reference_inference.cpp`

Exit criteria:
- [x] No global mutable CUDA workspace state.
- [x] No per-matvec full-device sync in hot loop.

---

## 3. Keep State Fully Device-Resident

Current issue:
- Full-attention and linear states are still updated on CPU and synced to device token-by-token.

Tasks:
- [x] Define device-state structs for:
  - Full-attention KV cache
  - Linear conv state
  - Linear recurrent SSM state
- [x] Stop maintaining CPU-authoritative copies for hot loop state
- [x] Replace `sync_full_state_token_to_cuda` and `sync_linear_state_to_cuda` path with direct GPU writes/updates
- [ ] Keep CPU copies only for optional debug dumps

Files:
- `src/runtime/reference_inference.cpp`
- `include/qwen35x/runtime/cuda_inference.h`
- `src/runtime/cuda_inference.cu`

Exit criteria:
- [x] No per-token cache/state upload from host in normal `--infer-gpu` mode.

---

## 4. Implement GPU Full-Attention Decode Kernel (GQA)

Current issue:
- Full-attention math runs in CPU loops in `run_full_attention_step`.

Tasks:
- [x] Add real kernel replacing `qwen35x_full_decode_gqa_stub`
- [ ] Kernel responsibilities:
  - Q/K normalization handling (or pre/post kernels if cleaner)
  - RoPE for query/key at current position
  - Attention score compute against KV cache
  - Softmax and weighted value reduction
  - Head-group mapping for GQA (`n_heads / n_kv_heads`)
- [x] Add launch wrapper API in `cuda_inference` module
- [x] Validate numeric parity on deterministic mode

Files:
- `src/kernels/cuda/qwen35x_decode_stub.cu` (replace with real kernels)
- `src/runtime/cuda_inference.cu`
- `include/qwen35x/runtime/cuda_inference.h`
- `src/runtime/reference_inference.cpp`

Exit criteria:
- [x] `run_full_attention_step` no longer performs CPU attention loops when `use_cuda=true`.

---

## 5. Implement GPU Linear-Attention Decode Kernel Path

Current issue:
- Linear attention path (`conv + recurrent update + gated norm`) is CPU-native.

Tasks:
- [x] Add real kernel replacing `qwen35x_linear_decode_stub`
- [ ] Implement GPU path for:
  - conv window update and 1D causal conv
  - q/k/v split and per-head norms
  - recurrent matrix state update (`S = alpha*S + beta*...`)
  - output projection input construction
  - gated RMS-style normalization and SiLU gate application
- [x] Keep a clean CPU reference path for parity testing only

Files:
- `src/kernels/cuda/qwen35x_decode_stub.cu`
- `src/runtime/cuda_inference.cu`
- `src/runtime/reference_inference.cpp`

Exit criteria:
- [x] `run_linear_attention_step` does not execute CPU math when `use_cuda=true`.

---

## 6. Move Residual/Norm/MLP Glue to GPU

Current issue:
- Even with CUDA matvec, glue math around layers can remain CPU-side and force transfers.

Tasks:
- [x] Add GPU kernels for:
  - residual add
  - RMSNorm variants used in block
  - SiLU and elementwise multiply in MLP gate path
- [x] Keep hidden states on device across entire layer stack
- [x] Avoid device-to-host copies between layer operations

Files:
- `src/runtime/reference_inference.cpp`
- `src/runtime/cuda_inference.cu`
- `include/qwen35x/runtime/cuda_inference.h`

Exit criteria:
- [x] Hidden/residual tensors stay on GPU from token embedding to logits.

---

## 7. GPU Logits + Sampling + Stop Conditions

Current issue:
- Sampling and stop-text checks are CPU-based in the main inference loop.

Tasks:
- [x] Add GPU sampling kernel path:
  - temperature scaling
  - top-k filter
  - top-p cutoff
  - repetition penalty application
- [x] Keep RNG deterministic with explicit seeded generator state per request
- [x] Return only sampled token id to CPU each step
- [x] Keep stop-token checks on CPU (cheap), but avoid full-logit copies
- [ ] Optional phase: add GPU-side stop-token hit flag

Files:
- `src/runtime/reference_inference.cpp`
- `src/runtime/cuda_inference.cu`
- `include/qwen35x/runtime/cuda_inference.h`
- `src/main.cpp`

Exit criteria:
- [x] No full-logits device-to-host copy per token in `--infer-gpu`.

Note:
- Current GPU sampling implementation supports `top_k <= 64` for `temperature > 0`.

---

## 8. Prefill GPU Path (Beyond Single-Token Decode)

Current issue:
- Runtime currently executes token-by-token forward; prefill is not optimized separately.

Tasks:
- [ ] Add dedicated prefill path for prompt tokens with batched/streaming execution
- [ ] Reuse same device-resident caches/state used by decode
- [ ] Benchmark prefill throughput independently from decode

Files:
- `src/runtime/reference_inference.cpp`
- `include/qwen35x/runtime/reference_inference.h`
- `src/runtime/cuda_inference.cu`

Exit criteria:
- [ ] Prefill has its own timing metric and shows clear gain over token-by-token loop.

---

## 9. Runtime Execution Optimizations

Tasks:
- [ ] Introduce persistent per-request device buffers (avoid per-token alloc/free)
- [ ] Use pinned host memory for minimal control-path transfers
- [ ] Capture steady-state decode in CUDA Graph
- [ ] Add stream strategy (main compute stream + optional transfer stream)
- [ ] Add warmup step and stable benchmark mode

Files:
- `src/runtime/cuda_inference.cu`
- `src/runtime/reference_inference.cpp`
- `src/main.cpp`

Exit criteria:
- [ ] Reduced kernel launch overhead and improved decode stability.

---

## 10. Correctness and Regression Harness

Tasks:
- [ ] Add script for CPU vs GPU deterministic parity runs over prompt suite
- [ ] Add script for throughput benchmarking with CSV/JSON output
- [ ] Add CI-friendly smoke mode (short token count, fixed seed)

Files:
- `scripts/` (new scripts)
- `src/main.cpp` (CLI options as needed)

Exit criteria:
- [ ] One-command parity report and one-command perf report.

---

## 11. Stretch Goals After Full GPU Parity

- [ ] BF16-native inference data path (not only benchmark microkernel)
- [ ] Weight layout packing tuned for decode kernels
- [ ] Optional fused kernels (RMSNorm+Linear, attention epilogues)
- [ ] Quantization track as separate mode after BF16 path is stable

---

## Suggested Execution Order

1. Sections 1, 2, 3 (measure and remove architectural blockers)
2. Section 4 (full-attention kernel)
3. Section 5 (linear-attention kernel)
4. Sections 6 and 7 (end-to-end GPU decode loop)
5. Sections 8 and 9 (prefill and runtime optimization)
6. Section 10 (automation and regression safety)

---

## Definition of Done (Full GPU Decode)

- [x] `--infer-gpu` performs full decode math on GPU for both full and linear attention layers.
- [x] Per-token host/device transfer is limited to minimal control data (for example sampled token id).
- [ ] Deterministic CPU/GPU parity passes on fixed prompt suite.
- [x] Throughput improvement is measured and documented with reproducible commands.
