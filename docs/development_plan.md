# qwen35x Development Plan

This plan summarizes the original project direction and turns it into a public, implementation-focused roadmap.

## Status Snapshot (April 2026)

Completed:
- CPU reference inference pipeline for Qwen3.5-0.8B
- CUDA inference path with full-attention decode kernel + linear-attention decode kernel
- Device-resident per-layer decode flow (hidden/residual/norm/attention/MLP on GPU)
- Legacy CUDA runtime GPU logits + GPU sampling path
- Legacy CUDA runtime device-token decode loop path (sampled token consumed on device for next-step embedding gather)
- CUDA Graph replay for steady-state decode segments (MLP + linear-attention blocks)
- Decode profiling (`--profile-json`) with stage timing and transfer breakdown
- BF16 decode matvec path in the legacy CUDA runtime backend
- Optional synchronized CUDA stage timing mode (`--profile-sync`)
- Packed full-attention projection (`q+gate+k+v`) to reduce full-attention decode matvec launches
- Streaming full-attention decode kernel with online softmax/value accumulation
- Luce megakernel decode backend integrated into `--infer-gpu` as the default Qwen3.5-0.8B decode path
- Luce CUDA sources used by the build moved into `src/kernels/cuda/luce_megakernel/` with local correctness fixes and MIT attribution
- Deterministic CPU/GPU parity harness with minimal and extended prompt suites

Current known constraints:
- Default Luce backend currently supports greedy decode only (`temperature <= 0`)
- Legacy runtime GPU sampling path currently requires `top_k <= 64` when `temperature > 0`
- Stop condition checks remain host-side
- Main Luce inference path currently replays prompt tokens through single-token decode for correctness; batched/specialized prefill is still open

Latest local benchmark snapshot (Qwen3.5-0.8B, same machine):
- Historical chat baseline (early April 2026): ~91 tokens/s
- Current integrated Luce sequential chat benchmark (April 23, 2026, `Runs=3`, `WarmupRuns=1`, `MaxNewTokens=128`, `MaxContext=256`):
  - `gpu-bf16` label: avg `289.43 tok/s` (`289.02` min, `290.15` max)
  - `gpu-f32` label: avg `287.88 tok/s` (`287.43` min, `288.31` max)
  - CSV: `benchmarks/qwen35x-inference-seq-luce-current.csv`
- Historical sequential chat benchmark (April 21, 2026, post streaming full-attention kernel):
  - BF16 matvec ON (historical `--infer-gpu` default at the time): `180.76`, `182.03`, `169.35` tokens/s (avg `177.38`)
  - FP32 matvec (`--gpu-f32-matvec`): `116.99`, `117.77`, `117.35` tokens/s (avg `117.37`)
  - Prior main rerun baseline: BF16 avg `166.90` tokens/s

## Vision

Build a standalone, hardware-aware inference engine specialized for the Qwen3.5 architecture family, starting with `Qwen3.5-0.8B` and scaling to larger variants later.

Core strategy:
- Keep the stack small and explicit
- Prioritize architecture-specific fast paths over generic abstractions
- Use reference implementations for correctness, then replace hotspots with specialized kernels

## Target Profile

- Initial model: `Qwen/Qwen3.5-0.8B`
- Initial platform: NVIDIA GPU class used in this project environment
- Initial precision path: dense reference + CUDA GPU decode path
- Future target: larger Qwen3.5 variants with the same engine architecture

## Design Principles

- Use a compiler/runtime split:
- Compiler: parse model metadata, build static execution plan, prepare packed tensors
- Runtime: execute prefill/decode and manage caches, dispatch, and memory
- Keep correctness and speed work separate:
- CPU reference path is the oracle
- GPU path must match token-by-token before heavy optimization
- Optimize measured bottlenecks first:
- Decode path and cache updates before broad refactors
- Build for extensibility:
- The code should be easy to retarget to other model sizes and GPUs

## Architecture Milestones

1. Baseline Bring-up
- Load HF model config and safetensors
- Implement tokenizer encode/decode
- End-to-end CPU reference inference

2. Hybrid Attention Correctness
- Support Qwen3.5 mixed layer schedule (`linear_attention` + `full_attention`)
- Maintain both full-attention KV cache and linear recurrent/conv states
- Validate stable text generation for simple prompts

3. CUDA-Hybrid Acceleration
- Add `--infer-gpu` path
- Offload matrix-heavy operations to CUDA
- Keep attention math in reference form until parity is stable

4. Generation Controls and Usability
- Add chat prompt wrapper mode
- Add default sampling and override flags
- Add deterministic mode with seed control
- Add stop-token and stop-text handling

5. Kernel Specialization Phase
- Replace CUDA stubs with real decode kernels
- Prioritize full-attention GQA decode kernel
- Add linear-attention recurrent update kernels
- Introduce packed weight layout for target GPU class

6. Prefill and Scheduling
- Optimize prefill separately from decode
- Add persistent runtime buffers
- Add profiling-driven dispatch and graph execution improvements

7. Low-Precision Phase
- Add one clear low-precision path first (instead of many formats at once)
- Keep fallback and correctness tests for every precision mode

Milestone progress:
- Milestones 1-4: completed
- Milestone 5: completed for the Qwen3.5-0.8B default decode path (legacy kernels + integrated Luce megakernel backend)
- Milestones 6-7: in progress

## Validation and Benchmarking Plan

- Correctness tests:
- CPU vs GPU token-level parity for fixed prompts
- Deterministic sampling with known seeds
- Benchmark suite:
- Decode tokens/sec
- Prefill throughput
- Load time and memory footprint
- Regression gates:
- Keep simple prompts and expected token outputs as smoke checks

## Current Performance Plan (April 2026 Update)

Primary objective:
- Reach Luce-level decode throughput and llama.cpp-level prefill throughput on the same hardware class.

Technical direction:
- Use Luce-style decode as the baseline runtime architecture.
- Improve prefill inside the same engine, using llama.cpp-inspired methods (large batched GEMM + flash-style attention), without splitting into two incompatible runtime stacks.
- Keep one canonical cache/state layout so prefill can hand off directly to decode without conversion or reorder steps.

Execution phases:
1. Stabilize runtime baseline
- Keep persistent, device-resident decode flow as the default path.
- Preserve deterministic correctness against CPU reference for fixed prompts/seeds.

2. Decode optimization track (Luce target)
- Prioritize persistent/megakernel-style decode execution and reduce per-token launch overhead.
- Tune decode occupancy controls (`decode_blocks`, launch geometry) per GPU profile.
- Keep sampling fully device-resident and remove avoidable host-side sync points.

3. Prefill optimization track (llama.cpp target)
- Replace token-by-token prefill projections with true batched GEMM paths for QKV and MLP projections.
- Add specialized full-attention prefill kernels with flash-style block processing.
- Add chunked linear-attention prefill with sequence-level kernels instead of per-token micro-dispatch.
- Minimize copy traffic and kernel launch count in prefill (eliminate token-wise buffer shuffles).

4. Unified scheduler and fallback policy
- Use one scheduler that chooses optimized paths by workload shape (prompt length, batch shape, model size, GPU profile).
- Keep safe fallback paths for short prompts and unsupported configurations.

Benchmark gates:
- Track prefill and decode separately in sequential harness runs with fixed settings.
- Require no regression on short-prompt decode while improving long-prompt prefill.
- Maintain comparable A/B runs versus old commits, Luce benchmark harness, and llama.cpp benchmark output.

## Scaling Plan for Larger Qwen3.5 Models

Model progression:
- Start from `Qwen3.5-0.8B`, then adapt to `4B`, `9B`, and `27B`.

Adaptation approach:
1. Parameterize kernel/runtime descriptors
- Drive dimensions (hidden size, head counts, KV heads, layer count, schedule) from model metadata.
- Keep model-specific compile-time specializations only where they provide measurable speedups.

2. Retune per model size
- Generate per-model tuning profiles (tile sizes, decode blocks, chunk sizes, graph capture boundaries).
- Store and reuse tuned profiles per GPU class.

3. Enforce layout compatibility
- Keep cache/state ABI stable across model variants to reuse the same runtime orchestration.
- Avoid model-specific conversion steps between prefill and decode.

4. Expand memory strategy for larger models
- For larger variants, plan for stricter VRAM budgeting, cache paging strategy, and potential multi-GPU/tensor-parallel extensions.

Validation policy for each new size:
- CPU/GPU token-level parity checks.
- Long-prompt prefill and decode throughput benchmarks.
- Regression comparison against the previous model tier before promotion.

## Implementation TODO (Megakernel Adaptation)

- [x] Extract a reusable Luce decode backend from the current benchmark harness.
- [x] Define a runtime-facing decode backend API (`init`, `reset`, `decode_step`, `release`) independent of benchmarking code.
- [x] Map current model weights/states into Luce-style packed decode layout at model-load time.
- [x] Integrate the reusable decode backend into the main GPU inference loop (default for `--infer-gpu`; legacy runtime backend remains selectable).
- [x] Keep a one-size-fits-all runtime path (no prompt-size gating in execution logic).
- [x] Move the Luce CUDA sources used by the build into `src/kernels/cuda/luce_megakernel/` with MIT license attribution.
- [x] Apply local Luce correctness fixes needed for CPU parity: DeltaNet decode decay, host-side barrier reset, repetition-penalty-aware greedy argmax.
- [ ] Unify prefill/decode around one canonical cache/state layout without prompt replay or conversion.
- [ ] Replace correctness-first token replay prefill with batched GEMM + flash-style full-attention prefill kernels.
- [ ] Remove token-wise projection/copy overhead in prefill and move to true batched projection execution.
- [ ] Parameterize kernel/runtime descriptors by model metadata to support `Qwen3.5-0.8B`, `4B`, `9B`, and `27B`.
- [ ] Add per-model/per-GPU autotune profiles (decode blocks, tile sizes, chunk sizes, graph boundaries).
- [x] Establish deterministic CPU vs GPU parity harness + fixed prompt suites (`scripts/benchmark-parity.ps1`, minimal + extended prompt sets). Latest baseline (April 23, 2026): minimal `5/5` pass, extended `12/12` pass.
- [x] Run CPU vs GPU token-parity validation for the Luce default integration and source move.
- [ ] Run prompt-length sweep benchmarks (short/medium/long) after each major optimization batch.
- [x] Compare the integrated Luce default decode milestone against prior qwen35x and Luce decode baselines.
