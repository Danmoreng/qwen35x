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
- Qwen35x CUDA backend integrated into `--infer-gpu` as the default decode path, with a single main binary that auto-selects the compiled 0.8B or 4B CUDA layout from the loaded model profile
- Qwen3.5-4B CUDA variant added as a compiled specialization alongside the 0.8B variant
- Qwen35x CUDA kernel sources used by the build live directly under `src/kernels/cuda/` with local correctness fixes and MIT attribution
- Batched Qwen35x prefill integrated as the default prompt-processing path, including a backend warmup during initialization to remove one-time CUDA/cuBLAS setup from timed inference
- Qwen35x prefill/decode phase profiling, including per-layer full-attention QK/softmax/PV/gate timing
- Long-context full-attention decode split across context blocks
- Single-path tiled full-attention prefill using the existing canonical cache/state layout
- Chunked MLP/DeltaNet prefill scratch plus variant-aware full-attention query tiling, preserving 0.8B throughput while making 4B 64k runs fit without VRAM spill collapse
- Deterministic CPU/GPU parity harness with minimal and extended prompt suites
- Optional PyTorch/Transformers parity harness for checking the CPU reference against an external implementation
- Actual-prompt benchmark matrix comparing qwen35x against llama.cpp with and without Flash Attention for 0.8B and 4B

Current known constraints:
- Default Qwen35x CUDA backend currently supports greedy decode only (`temperature <= 0`)
- Legacy runtime GPU sampling path currently requires `top_k <= 64` when `temperature > 0`
- Stop condition checks remain host-side
- Main Qwen35x CUDA inference path defaults to batched prefill for prompt processing.
- Token replay prefill remains selectable with `--qwen35x-prefill-mode replay` as a conservative fallback.
- Qwen35x CUDA backend initialization includes a dummy one-token prefill warmup and state reset; benchmark load time includes this warmup.
- Prefill chunk/tile tuning is controlled by `QWEN35X_PREFILL_MLP_CHUNK_TOKENS` and `QWEN35X_PREFILL_ATTENTION_QUERY_TOKENS`; defaults are chosen per selected model variant.
- The PyTorch/Transformers comparison environment is optional and kept outside the C++ build in `.venv-hf-parity`; it is a correctness oracle, not a performance benchmark.

Latest local benchmark snapshot (same machine):
- Actual-prompt model matrix (April 25, 2026, RTX 5080 Laptop GPU, `MaxNewTokens=128`, CSV `benchmarks/model-matrix/qwen35x-vs-llama-matrix-summary.csv`):
  - 0.8B qwen35x generation is ahead of llama.cpp actual-run generation across 256-4096 context: `325.97`, `331.72`, `312.55`, `316.40`, `306.01 tok/s`.
  - 0.8B llama.cpp + Flash Attention generation for the same contexts: `242.01`, `240.64`, `236.67`, `236.13`, `207.91 tok/s`.
  - 4B qwen35x generation is ahead of or near llama.cpp across 256-4096 context: `61.31`, `61.24`, `57.89`, `58.14`, `55.24 tok/s`.
  - 4B llama.cpp + Flash Attention generation for the same contexts: `48.96`, `49.40`, `47.39`, `46.45`, `49.39 tok/s`.
  - qwen35x prefill is ahead of llama.cpp for these prompt lengths in this matrix; the separate 64k run remains the main long-context prefill stress case.
- Long-context actual-prompt benchmark (April 25, 2026, Wikipedia prompt, ~65k prompt tokens, `MaxContext=65536`, `MaxNewTokens=128`):
  - Current integrated Qwen35x CUDA 0.8B path after chunked MLP/DeltaNet scratch and variant-aware attention tiling: prefill `8,030.47 ms` / `8,144.61 tok/s`; decode `645.69 ms` / `198.24 tok/s`, CSV `benchmarks/qwen35x-0p8b-wiki-ai-64k-gen128-chunked-variant-tile.csv`.
  - Current integrated Qwen35x CUDA 4B path with memory-safe attention query tiling: prefill `30,134.70 ms` / `2,170.42 tok/s`; decode `2,650.58 ms` / `48.29 tok/s`, CSV `benchmarks/qwen35x-4b-wiki-ai-64k-gen128-chunked-prefill.csv`.
  - The first MLP-only chunking pass made 4B 64k fit but still showed likely VRAM spill behavior: prefill `108,558 ms` / `602.49 tok/s`. DeltaNet scratch chunking plus bounded attention score scratch improved the same prefill to roughly `30.18 s` / `2,167 tok/s`.
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
- Descriptor/shape-validation follow-up (April 26, 2026):
  - CUDA safetensor shape validation is wired before allocation/upload for 0.8B and 4B, including the packed full-attention `q+gate` projection shape.
  - Minimal CPU/GPU parity after shape validation: pass `5/5`, CSV `benchmarks/qwen35x-parity-shape-validation.csv`.
  - 0.8B synthetic generation gates after shape validation stayed near the accepted decode range: `prompt1/gen128` avg `324.40 tok/s`, `pp512/gen128` avg `322.67 tok/s`, `pp1024/gen128` avg `311.58 tok/s`, `pp4096/gen128` avg `296.01 tok/s`.
  - 4B synthetic generation gates after shape validation stayed near the accepted decode range: `prompt1/gen128` avg `60.67 tok/s`, `pp512/gen128` avg `60.63 tok/s`, `pp1024/gen128` avg `60.65 tok/s`.

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

## Immediate Priority: Model Generalization

This remains the next work item. The current Qwen3.5-0.8B and 4B CUDA performance is accepted as good enough for now. The main binary now includes both enabled CUDA variants and dispatches by model descriptor, so descriptor cleanup and larger-model variant work should resume before further long-context performance tuning.

Rationale:
- The 64k actual-prompt benchmark after the grouped-GQA decode work is stable enough for the current target: prefill `8,218.60 ms` / `7,958.17 tok/s`; decode `632.65 ms` / `202.33 tok/s`, CSV `benchmarks/qwen35x-wiki-ai-64k-gen128-baseline-after-reboot.csv`.
- A cuBLAS-streaming flash-style prefill prototype was tested and rejected because it regressed the same 64k run to roughly `4,328 tok/s` prefill and `134.6 tok/s` generation. Future flash-style work should be a real fused/tensor-core attention implementation rather than a tiled cuBLAS wrapper.
- The remaining performance work is still valuable, but it should no longer block model-size generalization.

Active work for model-size generalization:
1. Add a Qwen35x CUDA model descriptor
- A first-class descriptor is now derived from `ModelProfile` / HF config and can be passed through `Qwen35xCudaBackendConfig`.
- It includes layer count, hidden size, intermediate size, vocab size, attention heads/KV heads/head dim, RoPE dim/theta, DeltaNet dimensions including grouped value dimension, conv settings, and exact layer schedule.
- Safetensor shapes are validated against the descriptor before CUDA allocation/upload, including the CUDA packed full-attention `q+gate` projection shape.

2. Make allocation and validation descriptor-driven
- Runtime cache/state, RoPE table, decode scratch, and prefill scratch allocation in `src/runtime/qwen35x_cuda_backend.cpp` now use descriptor-derived sizes, with a descriptor/compiled-variant consistency check before allocation.
- Tensor shape validation now runs before CUDA upload for embed/norm/lm head and per-layer MLP, full-attention, and DeltaNet tensors.
- Keep the current 0.8B and 4B compiled kernels as the enabled fast variants in the single main binary until additional variants have explicit kernels and parity gates.
- Add precise unsupported-variant errors for model shapes that do not yet have a compiled kernel variant.

3. Split compile-time kernel constants into per-variant configuration
- Start by preserving the existing `0.8B/sm120` kernel behavior exactly.
- Prepare configured/generated values for `9B` and `27B` after descriptor validation is in place.
- Dispatch through the compiler/runtime/kernel-registry direction instead of adding ad-hoc model checks in launch code.

4. Keep cache/state ABI stable
- Full-attention cache layout stays compatible between prefill and decode.
- Linear-attention recurrent state and conv state stay compatible between prefill and decode.
- Larger variants must not require handoff conversion between prefill and decode.

Deferred performance backlog:
- Replace the materialized tiled full-attention prefill path with a real fused/flash-style implementation.
- Continue reducing 4B long-context prefill memory and launch overhead; current chunking avoids VRAM spill but remains much slower than an eventual flash-style implementation.
- Optimize LM head for long-context token generation.
- Replace prefill beta/alpha scalar matvec launches with packed GEMM or another batched tensor-core path.
- Add compatible steady-state decode graph reuse for the Qwen35x CUDA path.
- Reduce prefill kernel launch count and recurrent overhead beyond warmup effects.

Exit criteria for the first generalization phase:
- The current 0.8B and 4B paths remain behaviorally and performance compatible with the accepted baseline.
- Unsupported larger variants fail with clear descriptor/variant diagnostics instead of silent shape misuse.
- Descriptor-derived allocation and shape validation are wired through the Qwen35x CUDA backend.
- CPU/GPU parity still passes the minimal and extended suites for the enabled 0.8B variant.

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
- Preserve the accepted Qwen3.5-0.8B CUDA performance baseline while making the backend descriptor-driven enough to support larger Qwen3.5 variants.

Technical direction:
- Use Qwen35x CUDA-style decode as the baseline runtime architecture.
- Improve prefill inside the same engine, using llama.cpp-inspired methods (large batched GEMM + flash-style attention), without splitting into two incompatible runtime stacks.
- Keep one canonical cache/state layout so prefill can hand off directly to decode without conversion or reorder steps.
- Treat the current initialization warmup as a measurement hygiene improvement, not a substitute for reducing actual prefill kernel work.

Execution phases:
1. Stabilize and instrument runtime baseline
- Keep persistent, device-resident decode flow as the default path.
- Preserve deterministic correctness against CPU reference for fixed prompts/seeds.
- Detailed Qwen35x phase profiling is available and should remain enabled for regression checks.

2. Model descriptor and variant enablement track
- Add a Qwen35x CUDA model descriptor derived from `ModelProfile` / HF config.
- Use descriptor-derived allocation and shape validation in the Qwen35x CUDA backend.
- Keep the existing 0.8B and 4B kernel variants as the enabled compiled variants until larger variants are explicitly added.
- Add clear unsupported-variant errors for shapes without a matching compiled kernel.

3. Long-context decode optimization track
- Split-context/split-K full-attention decode for long contexts has landed.
- Grouped-GQA decode sharing is implemented, so each KV segment feeds all query heads in its GQA group.
- Decode occupancy controls are exposed and profiled; current default uses dynamic max-safe blocks with a minimum safety clamp.
- Deferred: optimize LM head, which is now a visible share of long-context decode time.
- Deferred: add compatible CUDA graph reuse for steady-state decode.
- Keep short-prompt decode throughput as a non-regression gate.

4. Long-context prefill optimization track
- The current full-attention prefill path is a materialized tiled cuBLAS-backed implementation with variant-aware query tile size.
- MLP and DeltaNet prefill scratch are chunked to avoid allocating the largest intermediates at full `max_context` size.
- Deferred: replace the materialized score/probability path with one unified fused/flash-style block implementation.
- Deferred: replace beta/alpha per-token scalar matvec launches with packed GEMM or a comparable batched path.
- Deferred: continue reducing linear-attention recurrence overhead and prefill kernel launch count.
- Minimize copy traffic and avoid any prompt replay or prefill/decode state conversion.

5. Short-context decode optimization track (Qwen35x CUDA target)
- Prioritize persistent/megakernel-style decode execution and reduce per-token launch overhead.
- Keep sampling fully device-resident and remove avoidable host-side sync points.

6. Unified scheduler and fallback policy
- Keep the Qwen35x CUDA default path free of prompt-length-specific kernel dispatch; use specialized variants by model/GPU capability only when they preserve the same execution policy.
- Keep safe fallback paths for short prompts and unsupported configurations.

Benchmark gates:
- Track prefill and decode separately in sequential harness runs with fixed settings.
- Require no regression on short-prompt decode while improving long-prompt prefill.
- Maintain comparable A/B runs versus old commits, Qwen35x CUDA benchmark harness, and llama.cpp benchmark output.

## Scaling Plan for Larger Qwen3.5 Models

Scaling work is now the active priority. Long-context performance remains tracked, but it no longer blocks descriptor and larger-model work.

Model progression:
- Start from `Qwen3.5-0.8B` and `4B`, then adapt to `9B` and `27B`.

Current scaling constraint:
- The default Qwen35x CUDA path is compiled with both `Qwen3.5-0.8B` and `Qwen3.5-4B` specializations in the main binary, then dispatches by validated model descriptor.
- Runtime allocation sizes in `src/runtime/qwen35x_cuda_backend.cpp` are now descriptor-derived; compiled kernel dimensions still live in `src/kernels/cuda/kernel.cu`, `src/kernels/cuda/prefill.cu`, and the per-variant headers.
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
- Preserve the current `0.8B/sm120` and `4B/sm120` variants, then add `9B/sm120` and `27B/sm120` only after descriptor validation and parity gates pass.
- Dispatch through the existing compiler/runtime/kernel-registry direction rather than embedding model selection inside ad-hoc launch code.

4. Refactor scaling-sensitive kernel assumptions
- Replace single-kernel shared-memory assumptions that scale with `max(hidden, intermediate)` before moving to larger hidden/intermediate sizes.
- Tile the LM head over hidden/vocab dimensions instead of assuming the current hidden size is cheap to stage.
- Keep fast DeltaNet recurrence kernels specialized for known supported shapes, but fail clearly for unsupported descriptors.
- Remove static per-process prefill state that assumes only one layer count or one model variant can be loaded.

5. Make prefill descriptor-driven first
- Prefill should lean on cuBLAS/cuBLASLt GEMM with descriptor-driven `M/N/K` dimensions.
- Custom prefill kernels should specialize mostly on head dimension, RoPE layout, and DeltaNet state shape.
- Keep the current materialized tiled full-attention prefill path as the accepted baseline for the first descriptor-driven phase; revisit fused/flash-style attention as deferred performance work.
- Chunk prefill scratch by prompt length and available VRAM instead of allocating all scratch from `max_context * largest_dim`.

6. Expand memory strategy for larger models
- Track BF16 weight memory, KV cache memory, recurrent state, prefill scratch, and logits/sampling scratch separately.
- Larger variants require stricter VRAM budgeting, quantized weight paths, cache paging, and possibly multi-GPU/tensor-parallel extensions.
- The scheduler should be able to refuse unsupported model/context/device combinations with a precise diagnostic.

7. Retune per model size
- Generate per-model/per-GPU tuning profiles for decode blocks, block size, LM head tiling, prefill chunk size, cuBLASLt algorithms, and graph capture boundaries.
- Store and reuse tuned profiles per `(model_variant, sm)` instead of relying on a single global default.

## NVFP4 and Quantized Cache Objective

The BF16 Qwen35x CUDA path remains the stable default. NVFP4 work should be added as duplicate, opt-in kernel paths rather than by adding heavy precision conditionals inside the current BF16 kernels.

Precision rollout:
1. `BF16 weights + BF16 KV cache`
- Current production path.
- Must remain behaviorally stable and performance-compatible while quantized work lands.

2. `NVFP4 weights + BF16 KV cache`
- First quantized target.
- Add separate NVFP4 weight artifacts, packed weight structs, launchers, and decode kernels.
- Reuse the current BF16 KV cache layout so cache correctness and weight quantization are not debugged at the same time.

3. `NVFP4 weights + quantized KV cache`
- Second quantized target after weight-only decode is validated.
- Add a separate quantized cache layout with scale metadata, prefill cache writes, and decode cache reads/dequantization.
- Keep BF16 KV as an available fallback.

Kernel and dispatch policy:
- Dispatch by `(model_variant, weight_precision, cache_precision, sm)` through the variant/registry path.
- Use separate launchers such as BF16 decode and NVFP4 decode instead of mutating the BF16 kernel with many runtime branches.
- Shared helper code can be factored, but BF16 kernels should stay simple and low-risk.

Validation policy:
- BF16 remains token-exact against CPU reference for greedy parity gates.
- NVFP4 should start with token parity where possible, then add logit/hidden error thresholds and quality smoke prompts if token-exact parity is too strict.
- Every quantized promotion needs side-by-side BF16 fallback runs and benchmark CSVs for both prefill and decode.

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
- [x] Move the Qwen35x CUDA kernel sources used by the build into `src/kernels/cuda/` with MIT license attribution.
- [x] Apply local Qwen35x CUDA correctness fixes needed for CPU parity: DeltaNet decode decay, host-side barrier reset, repetition-penalty-aware greedy argmax.
- [x] Unify the Qwen35x prefill/decode handoff around one canonical cache/state layout without prompt replay or conversion.
- [x] Replace correctness-first token replay prefill with batched GEMM-based Qwen35x prefill as the default path.
  Current status: `--qwen35x-prefill-mode batched` is the default after sampler/cache-stride fixes, batched recurrence work, and CPU/GPU parity validation.
- [x] Remove prompt-replay and token-wise projection/copy overhead in prefill; use batched projection execution with direct state/cache handoff.
- [x] Warm the Qwen35x prefill backend during initialization and reset state before real inference, keeping one-time cuBLAS/kernel setup out of timed prefill.
- [x] Add detailed Qwen35x phase profiling for prefill and decode, including long-context 64k runs.
- [x] Replace naive full-attention prefill with a single tiled full-attention path for all prompt lengths.
- [x] Chunk MLP and DeltaNet prefill scratch and add variant-aware full-attention query tiling so 4B 64k can run without VRAM spill collapse.
- [ ] Replace the materialized tiled full-attention prefill path with a unified fused/flash-style implementation.
- [x] Add split-context/split-K full-attention decode for long-context generation.
- [x] Add grouped-GQA full-attention decode sharing plus decode-block profiling/clamping.
- [ ] Optimize LM head for long-context token generation.
- [ ] Replace prefill beta/alpha scalar matvec launches with packed GEMM or another batched tensor-core path.
- [ ] Add compatible steady-state decode graph reuse for the Qwen35x CUDA path.
- [ ] Run prompt-length sweep benchmarks (short/medium/long/64k) after each major long-context optimization batch.
- [ ] Reduce actual prefill kernel launch count and recurrence overhead beyond warmup effects.
- [x] Add a Qwen35x CUDA model descriptor derived from `ModelProfile`/HF config and pass it through `Qwen35xCudaBackendConfig`.
- [x] Add descriptor validation and clear unsupported-variant errors for model shapes without a matching compiled kernel.
- [x] Build the main `qwen35x` executable with both 0.8B and 4B CUDA variants and dispatch to the matching launchers at runtime.
- [x] Replace Qwen35x CUDA-side runtime allocation/reset sizes with descriptor-derived sizes while keeping the current 0.8B and 4B kernels as the enabled compiled variants.
- [x] Validate safetensor tensor shapes against the Qwen35x CUDA descriptor before allocating or uploading device weights.
- [x] Split megakernel compile-time constants into per-variant configured values for `0.8B` and `4B`.
- [ ] Extend per-variant generated/configured values to `9B` and `27B`.
- [ ] Refactor shared-memory and LM-head assumptions that scale with hidden/intermediate size before enabling larger variants.
- [x] Add first-pass prefill chunking for larger prompt/model shapes while preserving the canonical prefill/decode cache layout.
- [ ] Make prefill chunking descriptor-driven and move tuning defaults into per-model/per-GPU profiles.
- [ ] Add per-model/per-GPU autotune profiles (decode blocks, block size, LM head tiling, chunk sizes, graph boundaries).
- [x] Add explicit Qwen35x CUDA precision/cache mode plumbing: `weight_precision={bf16,nvfp4}` and `cache_precision={bf16,quantized}`, defaulting to `bf16/bf16`.
- [x] Add fail-fast diagnostics for unsupported `nvfp4` and quantized-cache modes until kernels and artifacts exist.
- [x] Include weight/cache precision labels in CLI output, profile JSON, benchmark CSVs, and Qwen35x runtime diagnostics.
  Current status: `--qwen35x-weight-precision bf16` and `--qwen35x-cache-precision bf16` are the default supported path. `--qwen35x-weight-precision nvfp4` runs with BF16 KV cache when a ModelOpt NVFP4 checkpoint is loaded. `--qwen35x-cache-precision quantized` still fails during backend initialization until the cache ABI is implemented.
- [x] Use the AxionML/ModelOpt NVFP4 checkpoints as the canonical quantized artifact source instead of local max-abs conversion.
- [x] Document the AxionML safetensors tensor naming, packed payload dtype, FP8 scale tensors, and per-tensor shape rules for 0.8B and 4B.
  Current status: `--validate-nvfp4-model` checks ModelOpt `hf_quant_config.json`, group size 16, null KV-cache quantization, `U8` packed `.weight` tensors shaped `[rows, cols / 2]`, `F8_E4M3` `.weight_scale` tensors shaped `[rows, cols / 16]`, and scalar `F32` `.input_scale` / `.weight_scale_2` tensors. The downloaded AxionML 0.8B checkpoint validates 186 quantized modules.
- [x] Add Qwen35x CUDA loader support for AxionML/ModelOpt NVFP4 safetensors while preserving the current BF16 loader and shape-validation path.
  Current status: `--qwen35x-weight-precision nvfp4` validates ModelOpt `U8` packed weights, `F8_E4M3` weight scales, and scalar `F32` scale metadata. Decode now consumes native packed NVFP4 projection weights through a separate per-layer pointer table while keeping BF16 embedding, norms, LM head, recurrent state, and KV cache.
- [x] Add duplicated NVFP4 weight structs for the decode path with BF16 KV cache.
- [x] Implement a correctness-first native packed NVFP4 decode matvec path.
  Current status: full-attention Q/K/V/O, DeltaNet QKV/Z/beta/alpha/out, and MLP gate/up/down projections can consume packed ModelOpt NVFP4 tensors. The implementation dequantizes E2M1 weights and E4M3 block scales inside warp matvecs; it is native packed inference, but not yet Blackwell tensor-core accelerated.
- [ ] Add a separate NVFP4 decode launcher/dispatch path so BF16 kernels have no runtime precision checks.
- [ ] Replace scalar dequantized NVFP4 matvecs with Blackwell-optimized kernels:
  - [x] Add a cuBLASLt FP4/NVFP4 block-scale GEMM probe for prefill-sized projections using `CUDA_R_4F_E2M1` and `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`.
    Current status: `--probe-nvfp4-cublaslt` confirms CUDA/cuBLASLt can select and execute a Blackwell FP4 block-scale algorithm for an AxionML tensor family.
  - [x] Validate AxionML packed weight/scale layout against cuBLASLt operand layout requirements without repacking at runtime.
    Current status: direct AxionML row-major packed weights and `[out,K/16]` scale tensors are not numerically compatible with the raw cuBLASLt FP4 path. Production FP4 execution needs a backend-specific packed projection layout rather than passing safetensor rows directly to cuBLASLt.
  - [x] Add NVIDIA tiled 1D block-scale layout handling to the diagnostic probe.
    Current status: the probe pads scales to `M % 128 == 0` and `(K/16) % 4 == 0`, then applies the documented FP4 1D block-scale tile order used for Blackwell block-scaled GEMM scale tensors.
  - [x] Establish the cuBLASLt operand orientation for native FP4 projection.
    Current status: `input[1,K] x weight[rows,K]^T -> output[1,rows]` with input scales bound to operand A and weight scales bound to operand B produces numerically plausible output. A full `gate_proj` probe over 3,584 rows reports `max_abs_expected ~= 0.467`, `max_abs_actual ~= 0.465`, and `max_abs_error ~= 0.051`; the remaining gap is consistent with FP4 activation quantization and needs thresholding against model quality tests.
  - [x] Implement a ModelOpt FP4 projection layout object that stores padded packed weights, swizzled FP8 block scales, `alpha=input_scale*weight_scale_2`, and original output size per tensor.
    Current status: the CUDA loader now materializes raw scalar-fallback tensors and a second tensor-core-ready projection layout for every ModelOpt NVFP4 projection. The tensor-core packed-weight copy uses cuBLASLt FP4 nibble order, and the scale copy uses the tiled UE4M3 block-scale layout. Decode still uses the scalar fallback until the FP4 projection backend is wired into the runtime path.
  - [ ] Add a reusable custom Blackwell FP4 projection backend using AxionML/ModelOpt packed weights and backend-specific scale layouts.
    Current status: the cuBLASLt probe remains a diagnostic and layout oracle, but the decode hot path is moving to owned kernels. `--bench-nvfp4-projection` now runs standalone custom CUDA projection variants over raw AxionML packed weights and checks the first rows against a CPU dequantized reference. The default `scale-group` kernel decodes one E4M3 scale per 16-weight group and is faster than the original row-parallel baseline. Sequential 0.8B layer-0 timings are about `0.052 ms` for `mlp.gate_proj` (`3584x1024`), `0.069 ms` for `mlp.down_proj` (`1024x3584`), and `0.085 ms` for `linear_attn.in_proj_qkv` (`6144x1024`), all with max absolute reference error near `1e-9`.
  - [ ] Replace the scale-group custom projection with a Blackwell FP4 tensor-core implementation.
    Current status: the current custom projection kernel is a correctness and benchmarking scaffold, not yet tensor-core FP4 inference. Correction: RTX 50 / SM120 should not use the SM100/103/110 `tcgen05` programming model. PTX 9.2 documents the SM120 native path as `mma.sync.aligned...kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`, and CUDA 13.2 accepts `compute_120a` / `compute_120f` targets. A missing `tcgen05` SM120 opcode is therefore not a reason to wait for a future CUDA release; it means the wrong Blackwell programming model was being checked. The build script now exposes `-CudaArchitectures`; `.\scripts\build.ps1 ... -BuildDir build-sm120a -CudaArchitectures 120a` builds successfully and CMake emits `--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]`. `--nvfp4-projection-kernel blackwell-fp4` now pre-packs the deterministic benchmark input and AxionML/ModelOpt weights into validated SM120 fragments, then runs a complete batch-1 projection with one warp per 8 output rows and a loop over all 64-column K blocks. Loader-side ModelOpt NVFP4 initialization now also materializes SM120 fragment-packed weight and scale buffers in addition to the raw scalar-fallback and cuBLASLt diagnostic layouts. A reusable `run_nvfp4_sm120_projection_device` API now performs GPU activation fragment packing plus native SM120 projection, and the decode dry-run hook can launch it against loader-prepacked weights using persistent SM120 activation scratch. Actual decode output substitution is not wired yet.
    Latest correction: the reusable SM120 projection API now includes the ModelOpt/NVFP4 second-level per-tensor weight scale in its production contract. Activation fragment packing quantizes raw activations, and the MMA projection applies the global `alpha = input_scale * weight_scale_2` factor after accumulation so tiny `input_scale` values do not underflow the activation E4M3 block scales. The `blackwell-fp4` benchmark now validates against the quantized activation/weight reference, reports failure using expected/actual magnitude plus absolute and relative error, and no longer treats a valid negative row-0 output as the SM120a build sentinel. On the AxionML 0.8B `layers.0.mlp.gate_proj` tensor, the stricter SM120a projection check passes with `max_abs_error=0.00017204` and `avg_iteration_ms=0.0379267` for a 3584x1024 batch-1 projection. Decode-callable SM120 gate/up+SiLU, full MLP, and MLP residual/writeback bridge primitives now run under `QWEN35X_DRY_RUN_FP4_DECODE_GATE_UP`, `QWEN35X_DRY_RUN_FP4_DECODE_MLP`, and `QWEN35X_DRY_RUN_FP4_DECODE_MLP_RESIDUAL`; gate/up share one activation pack, the MLP bridge feeds the SiLU product into the SM120 down projection, and the residual bridge writes `bf16 hidden_out = down + residual`. Actual decode output substitution is still pending because the current decode kernel owns the complete layer loop and must be split at the MLP boundary.
    Performance conclusion: host-launched FP4 projection substitution is only a correctness bridge. To beat BF16 decode, the SM120 MMA projection must be integrated into the persistent decode structure or a small number of fused layer-phase kernels. The reusable projection wrapper no longer forces a device-wide synchronize on unprofiled calls, only creates CUDA events when timing is requested, guards event cleanup, and caches the device capability check after the first call. The first production target remains fused MLP gate/up/down because gate/up share one activation pack and dominate decode work.
    Implementation direction:
    - [ ] Keep `scale-group` as the correctness and regression fallback while the tensor-core path is brought up.
    - [x] Build the first executable SM120a synthetic probe around `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`.
    - [x] Replace the pure accumulator probe with a tile-buffer probe that loads packed E2M1 A/B fragment registers and packed UE4M3 scale-vector registers from device memory.
    - [x] Validate a nonzero all-ones `m16n8k64` FP4 matrix product through the standalone SM120a tile-buffer probe.
    - [x] Derive and validate the exact non-uniform A/B fragment mapping for `m16n8k64` so the standalone tile can reproduce arbitrary tile products.
    - [x] Pack one real AxionML/ModelOpt activation and weight tile into the validated SM120 fragment layout and compare against a CPU reference.
    - [x] Expand the standalone tile path across output rows and K blocks to produce a complete batch-1 projection output.
    - [x] Move SM120 fragment packing for ModelOpt weights into the CUDA loader so decode can reuse prepacked tensor-core-ready layouts.
    - [x] Move FP4 activation quantization and fragment packing onto GPU for decode inputs.
    - [x] Add a decode-callable SM120 projection launcher that consumes loader-prepacked weights and persistent activation scratch.
    - [x] Add `alpha = input_scale * weight_scale_2` support to the SM120 projection API and apply it after MMA accumulation.
    - [x] Replace the current loose benchmark check with strict FP4-reference validation: absolute error, relative error, expected/actual magnitude checks, and near-zero-output guards.
    - [x] Remove production-path overhead from the SM120 projection wrapper: no forced synchronize, no per-call event allocation unless profiling, no per-call device-property query, and guarded event cleanup.
    - [x] Add a decode-callable SM120 gate/up+SiLU launcher and dry-run hook so the MLP replacement can be tested as a pair before decode substitution.
    - [x] Share one SM120 activation pack across gate/up projections in the gate/up bridge.
    - [x] Add an SM120 down projection bridge and full MLP dry-run hook that validates gate/up/SiLU/down plumbing together.
    - [x] Add SM120 down projection residual/writeback fusion at the bridge-primitive level.
    - [ ] Split the decode path around MLP gate/up/down so the SM120 projection launcher can replace scalar NVFP4 matvecs instead of running as a dry-run beside them.
    - [x] Fuse MLP gate/up SM120 projections with one activation pack and a GPU `SiLU(gate) * up` stage at the bridge-primitive level.
    - [ ] Add SM120 MLP substitution inside the decode layer execution path, reusing the bridge output contract without corrupting the cooperative-kernel layer loop.
    - [ ] Extend the same projection backend to attention and DeltaNet projections only after MLP parity and throughput improve over BF16.
    - [ ] Run inference benchmarks with `scripts/benchmark-inference-seq.ps1 -Runs 3 -WarmupRuns 1 -MaxNewTokens 128 -MaxContext 256`; promotion requires NVFP4 decode to exceed the BF16 baseline, currently in the `300+ tok/s` range on the 0.8B smoke path.
    - [ ] Validate the microkernel on synthetic tiles first, then on AxionML projection tiles against the existing CPU and `scale-group` references.
    - [ ] Match the documented SM120 constraints: TN operand orientation, block-scaled E2M1 weights/activations, UE4M3 scale vectors, fixed 1x1x1 cluster shape, and supported tile shapes.
    - [ ] Use cuBLASLt/CUTLASS behavior as a correctness and layout oracle, but keep the decode hot path in owned kernels.
    - [ ] Integrate the proven standalone projection into MLP gate/up/down first, then extend to attention/DeltaNet projections, then run parity and decode-throughput benchmarks.
  - [ ] Replace MLP gate/up scalar NVFP4 matvecs with custom FP4 projections plus a GPU SiLU multiply.
    Current status: a runtime-facing gate/up primitive using cuBLASLt exists as a diagnostic, but generic host-launched FP4 GEMM is not the preferred decode path. Integration should wait until the standalone custom projection primitive beats the scalar fallback and is competitive with BF16 at batch-1 decode shapes.
  - [x] Add a GPU activation quantization probe stage that emits packed E2M1 activations plus tiled UE4M3 per-16 scales without host round-trips.
    Current status: `--probe-nvfp4-cublaslt` now quantizes the input vector on GPU, self-checks the packed activation and tiled scale buffers against the host reference, then feeds those GPU-produced buffers into cuBLASLt.
  - [ ] Move the GPU activation quantization stage from the probe into a reusable decode/prefill projection backend.
  - [x] Add a custom decode projection benchmark for batch-1/token shapes because cuBLASLt GEMM is inefficient at current shapes.
  - [ ] Keep scalar NVFP4 matvecs as the correctness fallback until tensor-core kernels pass error-threshold and quality smoke tests.
- [ ] Expand NVFP4 native coverage to embedding/LM-head or document why those remain BF16.
- [ ] Validate `NVFP4 weights + BF16 KV cache` with parity/error-threshold tests and short/medium/long decode benchmarks.
  Current smoke status: `Hello`, 8-token greedy decode on the AxionML 0.8B checkpoint completes with native packed NVFP4 projections and coherent output. Throughput is currently about `95 tok/s` for this tiny smoke, versus about `330 tok/s` for the BF16 path on the same prompt; this confirms the scalar fallback is for correctness bring-up, not the final Blackwell performance path.
- [ ] Add NVFP4 prefill support after decode-only weight quantization is validated, preferably through cuBLASLt FP4/NVFP4 GEMM where available or a measured custom fallback.
- [ ] Define the quantized KV cache ABI: storage type, per-head/per-block scale metadata, cache strides, and descriptor-derived allocation sizes.
- [ ] Add quantized full-attention cache writes in prefill/decode and matching decode-side dequantized reads.
- [ ] Validate `NVFP4 weights + quantized KV cache` as a separate opt-in mode against BF16 KV fallback before promotion.
- [ ] Add memory-accounting and benchmark reporting for BF16 weights, NVFP4 weights/scales, BF16 KV, quantized KV/scales, recurrent state, and prefill scratch.
- [x] Establish deterministic CPU vs GPU parity harness + fixed prompt suites (`scripts/benchmark-parity.ps1`, minimal + extended prompt sets). Latest baseline (April 24, 2026): batched Qwen35x prefill passes minimal `5/5` and extended `12/12`.
- [x] Add optional PyTorch/Transformers parity harness (`scripts/benchmark-transformers-parity.ps1`) to validate tokenizer, prompt formatting, and greedy CPU-reference output against an external implementation. Latest baseline (April 24, 2026): minimal `5/5` pass.
- [x] Run CPU vs GPU token-parity validation for the Qwen35x CUDA default integration and source move.
- [x] Compare the integrated Qwen35x CUDA default decode/prefill milestone against prior qwen35x, Qwen35x kernel bench harness, and saved llama.cpp benchmark artifacts.
- [x] Add the actual-prompt qwen35x vs llama.cpp matrix benchmark script for 0.8B and 4B (`scripts/benchmark-qwen35x-vs-llama-matrix.ps1`).
