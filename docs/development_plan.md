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
- Qwen35x CUDA backend integrated into `--infer-gpu` as the default Qwen3.5-0.8B decode path
- Qwen35x CUDA kernel sources used by the build moved into `src/kernels/cuda/qwen35x_megakernel/` with local correctness fixes and MIT attribution
- Batched Qwen35x prefill integrated as the default prompt-processing path, including a backend warmup during initialization to remove one-time CUDA/cuBLAS setup from timed inference
- Qwen35x prefill/decode phase profiling, including per-layer full-attention QK/softmax/PV/gate timing
- Long-context full-attention decode split across context blocks
- Single-path tiled full-attention prefill using the existing canonical cache/state layout
- Deterministic CPU/GPU parity harness with minimal and extended prompt suites
- Optional PyTorch/Transformers parity harness for checking the CPU reference against an external implementation

Current known constraints:
- Default Qwen35x CUDA backend currently supports greedy decode only (`temperature <= 0`)
- Legacy runtime GPU sampling path currently requires `top_k <= 64` when `temperature > 0`
- Stop condition checks remain host-side
- Main Qwen35x CUDA inference path defaults to batched prefill for prompt processing.
- Token replay prefill remains selectable with `--qwen35x-prefill-mode replay` as a conservative fallback.
- Qwen35x CUDA backend initialization includes a dummy one-token prefill warmup and state reset; benchmark load time includes this warmup.
- The PyTorch/Transformers comparison environment is optional and kept outside the C++ build in `.venv-hf-parity`; it is a correctness oracle, not a performance benchmark.

Latest local benchmark snapshot (Qwen3.5-0.8B, same machine):
- Long-context actual-prompt benchmark (April 25, 2026, Wikipedia prompt, ~65k prompt tokens, `MaxContext=65536`, `MaxNewTokens=128`):
  - Current integrated Qwen35x CUDA path after profiling, split decode attention, grouped-GQA decode, decode-block clamp, and single-path tiled prefill tuning: prefill `8,310.58 ms` / `7,870.08 tok/s`; decode `636.25 ms` / `201.18 tok/s`, CSV `benchmarks/qwen35x-wiki-ai-64k-gen128-gqa-decode-default-profile.csv`.
  - Full-attention prefill attention subphase split: total attention `4,694.03 ms`; QK `1,693.86 ms`, softmax `1,656.64 ms`, PV `1,322.32 ms`, gate `15.38 ms`.
  - Decode profile for the same run: effective decode blocks `60/60`, decode kernel `516.41 ms`, LM head `115.30 ms`.
  - llama.cpp `llama-completion` without Flash Attention: prefill `12,181.78 ms` / `5,369.00 tok/s`; decode `907.89 ms` / `139.88 tok/s`, CSV `benchmarks/llama-cli/qwen35x-wiki-ai-64k-gen128.csv`.
  - llama.cpp `llama-completion` with Flash Attention: prefill `6,979.29 ms` / `9,371.15 tok/s`; decode `767.73 ms` / `165.42 tok/s`, CSV `benchmarks/llama-cli/qwen35x-wiki-ai-64k-gen128.csv`.
  - Interpretation: Qwen35x long-context prefill is faster than llama.cpp without Flash Attention but remains behind llama.cpp with Flash Attention; decode is now ahead of the saved llama.cpp runs, with LM head the next material decode bottleneck.
- Current short-context gate after the single-path prefill cleanup (April 25, 2026, `Runs=3`, `WarmupRuns=1`, `MaxContext=256`, `MaxNewTokens=128`):
  - `chat_short_joke/gen128` generation avg `274.30 tok/s` (`275.22`, `268.66`, `279.03` samples), CSV `benchmarks/qwen35x-short-gen128-qwen35x-prefill-single-path.csv`.
- Current integrated Qwen35x CUDA benchmark (April 24, 2026, `Runs=3`, `WarmupRuns=1`):
  - `pp256` prefill-only: avg `19,739.13 tok/s` (`20,112.19`, `18,994.76`, `20,110.50` samples), CSV `benchmarks/qwen35x-pp256-prefill-only-current-rerun.csv`
  - `prompt1/gen128` generation: avg `300.46 tok/s` (`317.89`, `312.77`, `270.72` samples), CSV `benchmarks/qwen35x-tg-prompt1-current-rerun.csv`
  - End-to-end `pp256/gen128` with `MaxContext=384`: prefill avg `18,915.00 tok/s`, generation avg `302.38 tok/s`, CSV `benchmarks/qwen35x-full-pp256-gen128-current-ctx384.csv`
  - Saved llama.cpp comparison artifacts from the earlier run: `pp256` prefill `12,597.12 tok/s` without FA and `13,681.26 tok/s` with FA; `gen128` generation `140.15 tok/s` without FA and `142.59 tok/s` with FA.
- Historical chat baseline (early April 2026): ~91 tokens/s
- Historical sequential chat benchmark (April 21, 2026, post streaming full-attention kernel):
  - BF16 matvec ON (historical `--infer-gpu` default at the time): `180.76`, `182.03`, `169.35` tokens/s (avg `177.38`)
  - FP32 matvec (`--gpu-f32-matvec`): `116.99`, `117.77`, `117.35` tokens/s (avg `117.37`)
  - Prior main rerun baseline: BF16 avg `166.90` tokens/s

Latest validation snapshot (April 24, 2026):
- CPU reference vs PyTorch/Transformers external oracle (`scripts/benchmark-transformers-parity.ps1`, minimal prompt suite, `max_new_tokens=4`): pass `5/5` with prompt-token and generated-token parity.
- Default GPU path vs CPU reference (`scripts/benchmark-parity.ps1`, minimal prompt suite, `gpu-f32`, Qwen35x batched prefill, `max_new_tokens=4`): pass `5/5`.
- Extended CPU/GPU parity suite (`gpu-f32`, Qwen35x batched prefill + warmup, `max_new_tokens=4`): pass `12/12`.

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
- CPU reference path is the local oracle
- Periodically validate the CPU reference against an external implementation such as PyTorch/Transformers
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
- Milestone 5: completed for the Qwen3.5-0.8B default decode path (legacy kernels + integrated Qwen35x kernel backend)
- Milestones 6-7: in progress

## Immediate Priority: Long-Context Performance First

This is the next work item and takes precedence over larger-model generalization and the remaining scaling roadmap.

Rationale:
- The 64k actual-prompt benchmark now shows Qwen35x CUDA generation ahead of the saved llama.cpp runs after grouped-GQA decode, while prefill still trails llama.cpp with Flash Attention.
- The remaining bottlenecks are structural rather than specific to the 0.8B model size: the current materialized full-attention prefill path still spends seconds across QK/softmax/PV at 64k, decode now spends a meaningful share in LM head, prefill beta/alpha projections remain inefficient, and the Qwen35x CUDA path still lacks compatible steady-state decode graph reuse.
- Generalizing the current kernels to larger Qwen3.5 variants before fixing these issues would mostly generalize a long-context design that already fails the target performance shape.

Required work before model-size generalization:
1. Add Qwen35x phase profiling for prefill and decode
- Capture per-layer and per-phase timings for projection, DeltaNet recurrence, full-attention QKV/cache work, attention scan, MLP, LM head, sampling, synchronization, and host/device transfers.
- Current status: implemented for Qwen35x prefill/decode and used on both short-context progress gates and 64k actual-prompt runs.

2. Fix long-context full-attention prefill
- Current status: replaced the naive causal full-attention prefill fallback with one tiled cuBLAS-backed path for all prompt lengths, using larger scratch-safe tiles, 512-thread softmax, fast exponentials, and full QK/softmax/PV/gate profiling.
- Next step: replace the materialized score/probability tiled path with one unified fused/flash-style implementation; do not add prompt-length-specific kernel dispatch.
- Preserve the existing canonical KV cache layout so prefill still hands off directly to decode.
- Continue benchmarking against llama.cpp with and without Flash Attention.

3. Fix long-context decode attention
- Current status: split-context full-attention decode plus grouped-GQA KV sharing is implemented. The 64k generation path improved from `103.31 tok/s` to `201.18 tok/s` on the saved Wikipedia prompt, ahead of the saved llama.cpp runs.
- Decode-block override and effective block reporting are wired into `scripts/benchmark-inference-seq.ps1`; unsafe low overrides are clamped to one block per DeltaNet head.
- Next step: optimize LM head and preserve the current decode attention path as the baseline.
- Keep short-context decode throughput from regressing.

4. Replace inefficient long-prefill recurrent projections
- Replace per-token/per-head beta and alpha scalar matvec launches with packed GEMM or another batched tensor-core path.
- Reduce DeltaNet prefill kernel launch count and global-memory traffic.

5. Add steady-state decode graph reuse where it is compatible with the Qwen35x CUDA path
- Capture/replay stable decode work when context shape and selected kernels allow it.
- Keep correctness and stop-condition behavior unchanged.

Exit criteria for resuming larger-model generalization:
- 64k actual-prompt prefill and decode are materially closer to llama.cpp on the same hardware.
- Short-context benchmarks do not regress from the current Qwen35x CUDA advantage.
- CPU/GPU parity still passes the minimal and extended suites.
- The optimized kernels are structured as reusable primitives or per-variant specializations so later generalization does not require redoing the same work.

## Validation and Benchmarking Plan

- Correctness tests:
- CPU vs GPU token-level parity for fixed prompts
- CPU reference vs PyTorch/Transformers token-level parity for fixed prompts
- Deterministic sampling with known seeds
- Benchmark suite:
- Decode tokens/sec
- Prefill throughput
- Load time and memory footprint
- Regression gates:
- Keep simple prompts and expected token outputs as smoke checks

## Current Performance Plan (April 2026 Update)

Primary objective:
- Maintain current-kernel-level short-context decode throughput and current llama.cpp-leading `pp256` prefill throughput on the same hardware class, while closing the newly identified 64k-context prefill and decode gap versus llama.cpp.

Technical direction:
- Use Qwen35x CUDA-style decode as the baseline runtime architecture.
- Improve prefill inside the same engine, using llama.cpp-inspired methods (large batched GEMM + flash-style attention), without splitting into two incompatible runtime stacks.
- Keep one canonical cache/state layout so prefill can hand off directly to decode without conversion or reorder steps.
- Treat the current initialization warmup as a measurement hygiene improvement, not a substitute for reducing actual prefill kernel work.

Execution phases:
1. Stabilize and instrument runtime baseline
- Keep persistent, device-resident decode flow as the default path.
- Preserve deterministic correctness against CPU reference for fixed prompts/seeds.
- Detailed Qwen35x phase profiling is available and should remain enabled for long-context performance work.

2. Long-context decode optimization track
- Split-context/split-K full-attention decode for long contexts has landed.
- Grouped-GQA decode sharing is implemented, so each KV segment feeds all query heads in its GQA group.
- Decode occupancy controls are exposed and profiled; current default uses dynamic max-safe blocks with a minimum safety clamp.
- Optimize LM head, which is now a visible share of long-context decode time.
- Add compatible CUDA graph reuse for steady-state decode.
- Keep short-prompt decode throughput as a non-regression gate.

3. Long-context prefill optimization track
- The current full-attention prefill path is a single tiled cuBLAS-backed implementation for all prompt lengths.
- Replace the materialized score/probability path with one unified fused/flash-style block implementation.
- Replace beta/alpha per-token scalar matvec launches with packed GEMM or a comparable batched path.
- Continue reducing linear-attention recurrence overhead and prefill kernel launch count.
- Minimize copy traffic and avoid any prompt replay or prefill/decode state conversion.

4. Short-context decode optimization track (Qwen35x CUDA target)
- Prioritize persistent/megakernel-style decode execution and reduce per-token launch overhead.
- Keep sampling fully device-resident and remove avoidable host-side sync points.

5. Unified scheduler and fallback policy
- Keep the Qwen35x CUDA default path free of prompt-length-specific kernel dispatch; use specialized variants by model/GPU capability only when they preserve the same execution policy.
- Keep safe fallback paths for short prompts and unsupported configurations.

Benchmark gates:
- Track prefill and decode separately in sequential harness runs with fixed settings.
- Require no regression on short-prompt decode while improving long-prompt prefill.
- Maintain comparable A/B runs versus old commits, Qwen35x CUDA benchmark harness, and llama.cpp benchmark output.

## Scaling Plan for Larger Qwen3.5 Models

Scaling work is intentionally paused behind the long-context performance priority above. Descriptor and larger-model work should resume only after the long-context exit criteria are met.

Model progression:
- Start from `Qwen3.5-0.8B`, then adapt to `4B`, `9B`, and `27B`.

Current scaling constraint:
- The default Qwen35x CUDA path is still specialized for `Qwen3.5-0.8B`.
- Hard-coded model dimensions currently live in `src/runtime/qwen35x_cuda_backend.cpp`, `src/kernels/cuda/qwen35x_megakernel/kernel.cu`, and `src/kernels/cuda/qwen35x_megakernel/prefill.cu`.
- The hard-coded values include layer count, hidden size, intermediate size, vocab size, full-attention head counts, DeltaNet dimensions, layer schedule, and maximum sequence length.
- The legacy runtime path already uses `ModelProfile`/`RuntimeDims` more broadly and should be used as the descriptor model for generalizing the Qwen35x CUDA path.

Adaptation approach:
1. Introduce a first-class Qwen35x CUDA model descriptor
- Derive a `Qwen35ModelDescriptor`-style structure from HF `config.json` / `ModelProfile`.
- Include layers, hidden size, intermediate size, vocab size, full-attention heads/KV heads/head dim, RoPE dim/theta, DeltaNet head/dim settings, conv kernel, and exact layer schedule.
- Pass this descriptor into `Qwen35xCudaBackendConfig` instead of relying on Qwen35x CUDA-local constants.
- Validate safetensor shapes against the descriptor before allocating or uploading device weights.

2. Keep a stable cache/state ABI
- Full-attention cache layout should remain `full_layer_slot x kv_heads x max_seq x head_dim x {K,V}`.
- Linear-attention state layout should remain `linear_layer_slot x linear_v_heads x key_dim x value_dim`, plus linear conv state.
- Prefill and decode must share the same cache/state layout so larger models do not need handoff conversion.

3. Preserve specialized decode variants
- Do not make the megakernel fully runtime-dynamic if that sacrifices the optimization value.
- Compile or generate decode variants keyed by `(model_variant, sm)` where dimensions affect shared memory, register layout, unrolling, or occupancy.
- Start with the current `0.8B/sm120` variant, then add `4B/sm120`, `9B/sm120`, and `27B/sm120` only after descriptor validation and parity gates pass.
- Dispatch through the existing compiler/runtime/kernel-registry direction rather than embedding model selection inside ad-hoc launch code.

4. Refactor scaling-sensitive kernel assumptions
- Replace single-kernel shared-memory assumptions that scale with `max(hidden, intermediate)` before moving to larger hidden/intermediate sizes.
- Tile the LM head over hidden/vocab dimensions instead of assuming the current hidden size is cheap to stage.
- Keep fast DeltaNet recurrence kernels specialized for known supported shapes, but fail clearly for unsupported descriptors.
- Remove static per-process prefill state that assumes only one layer count or one model variant can be loaded.

5. Make prefill descriptor-driven first
- Prefill should lean on cuBLAS/cuBLASLt GEMM with descriptor-driven `M/N/K` dimensions.
- Custom prefill kernels should specialize mostly on head dimension, RoPE layout, and DeltaNet state shape.
- Replace the current materialized full-attention prefill path with a unified fused/flash-style implementation before promoting long-context larger-model support.
- Chunk prefill scratch by prompt length and available VRAM instead of allocating all scratch from `max_context * largest_dim`.

6. Expand memory strategy for larger models
- Track BF16 weight memory, KV cache memory, recurrent state, prefill scratch, and logits/sampling scratch separately.
- Larger variants require stricter VRAM budgeting, quantized weight paths, cache paging, and possibly multi-GPU/tensor-parallel extensions.
- The scheduler should be able to refuse unsupported model/context/device combinations with a precise diagnostic.

7. Retune per model size
- Generate per-model/per-GPU tuning profiles for decode blocks, block size, LM head tiling, prefill chunk size, cuBLASLt algorithms, and graph capture boundaries.
- Store and reuse tuned profiles per `(model_variant, sm)` instead of relying on a single global default.

Validation policy for each new size:
- CPU/GPU token-level parity checks.
- Long-prompt prefill and decode throughput benchmarks.
- Regression comparison against the previous model tier before promotion.

## Implementation TODO (Megakernel Adaptation)

- [x] Extract a reusable Qwen35x CUDA backend from the current benchmark harness.
- [x] Define a runtime-facing decode backend API (`init`, `reset`, `decode_step`, `release`) independent of benchmarking code.
- [x] Map current model weights/states into Qwen35x CUDA-style packed decode layout at model-load time.
- [x] Integrate the reusable decode backend into the main GPU inference loop (default for `--infer-gpu`; legacy runtime backend remains selectable).
- [x] Keep a one-size-fits-all runtime path (no prompt-size gating in execution logic).
- [x] Move the Qwen35x CUDA kernel sources used by the build into `src/kernels/cuda/qwen35x_megakernel/` with MIT license attribution.
- [x] Apply local Qwen35x CUDA correctness fixes needed for CPU parity: DeltaNet decode decay, host-side barrier reset, repetition-penalty-aware greedy argmax.
- [x] Unify the Qwen35x prefill/decode handoff around one canonical cache/state layout without prompt replay or conversion.
- [x] Replace correctness-first token replay prefill with batched GEMM-based Qwen35x prefill as the default path.
  Current status: `--qwen35x-prefill-mode batched` is the default after sampler/cache-stride fixes, batched recurrence work, and CPU/GPU parity validation.
- [x] Remove prompt-replay and token-wise projection/copy overhead in prefill; use batched projection execution with direct state/cache handoff.
- [x] Warm the Qwen35x prefill backend during initialization and reset state before real inference, keeping one-time cuBLAS/kernel setup out of timed prefill.
- [x] Add detailed Qwen35x phase profiling for prefill and decode, including long-context 64k runs.
- [x] Replace naive full-attention prefill with a single tiled full-attention path for all prompt lengths.
- [ ] Replace the materialized tiled full-attention prefill path with a unified fused/flash-style implementation.
- [x] Add split-context/split-K full-attention decode for long-context generation.
- [x] Add grouped-GQA full-attention decode sharing plus decode-block profiling/clamping.
- [ ] Optimize LM head for long-context token generation.
- [ ] Replace prefill beta/alpha scalar matvec launches with packed GEMM or another batched tensor-core path.
- [ ] Add compatible steady-state decode graph reuse for the Qwen35x CUDA path.
- [ ] Run prompt-length sweep benchmarks (short/medium/long/64k) after each major long-context optimization batch.
- [ ] Reduce actual prefill kernel launch count and recurrence overhead beyond warmup effects.
- [ ] Add a Qwen35x CUDA model descriptor derived from `ModelProfile`/HF config and pass it through `Qwen35xCudaBackendConfig`.
- [ ] Replace Qwen35x CUDA-side 0.8B allocation sizes with descriptor-derived sizes while keeping the current 0.8B kernel as the only enabled compiled variant.
- [ ] Add descriptor validation and clear unsupported-variant errors for larger models until their kernel variants exist.
- [ ] Split megakernel compile-time constants into per-variant generated/configured values for `0.8B`, `4B`, `9B`, and `27B`.
- [ ] Refactor shared-memory and LM-head assumptions that scale with hidden/intermediate size before enabling larger variants.
- [ ] Add prefill chunking and descriptor-driven GEMM dimensions for larger prompt/model shapes.
- [ ] Add per-model/per-GPU autotune profiles (decode blocks, block size, LM head tiling, chunk sizes, graph boundaries).
- [x] Establish deterministic CPU vs GPU parity harness + fixed prompt suites (`scripts/benchmark-parity.ps1`, minimal + extended prompt sets). Latest baseline (April 24, 2026): batched Qwen35x prefill passes minimal `5/5` and extended `12/12`.
- [x] Add optional PyTorch/Transformers parity harness (`scripts/benchmark-transformers-parity.ps1`) to validate tokenizer, prompt formatting, and greedy CPU-reference output against an external implementation. Latest baseline (April 24, 2026): minimal `5/5` pass.
- [x] Run CPU vs GPU token-parity validation for the Qwen35x CUDA default integration and source move.
- [x] Compare the integrated Qwen35x CUDA default decode/prefill milestone against prior qwen35x, Qwen35x kernel bench harness, and saved llama.cpp benchmark artifacts.
