# CPU Q8 implementation roadmap

Status date: 2026-08-31  
Target model: Qwen3.5 0.8B GGUF Q8_0  
Working branch: `codex/cpu-q8-avx2`  
Initial optimized baseline: `bac3e6d`  
External-review documentation baseline: `b81e2e8`

## Purpose

This document is the durable implementation and measurement plan for the
Qwen3.5 0.8B CPU inference path. It translates the external source review and
local benchmark findings into independently testable changes.

The immediate target is the Intel Core i7-8750H laptop:

- 6 physical cores / 12 logical threads;
- AVX2, FMA, and F16C;
- no AVX-512 or VNNI;
- dual-channel DDR4-2666 and limited sustained laptop power/thermal headroom.

The later target is a newer x86-64 machine with some combination of 256-bit
AVX-VNNI, AVX-512, AVX-512 VNNI, and possibly AMX. Those backends must be added
without removing the portable scalar and AVX2 paths.

The end state should normally be one universal binary containing separately
compiled ISA implementations selected safely at runtime. Optional CPU-specific
release binaries may be added later for PGO/LTO, but they are not required for
correctness or normal distribution.

## Performance objective

The primary product workload is local speech-transcript cleanup:

1. restore a cached invariant system-prompt state when available;
2. prefill the variable raw transcript, commonly up to about 2,048 tokens;
3. generate corrected text with low time-to-first-token and high decode speed;
4. preserve the transcript's meaning while fixing wording, punctuation, and
   formatting.

The engine currently exceeds the contemporaneous six-thread llama.cpp short
benchmark by about 1.4% in pp256 and 17.4% in prompt-1/tg128. At 2,048 input and
128 output tokens, qwen35x is about 4.1% slower in prefill, 17.7% faster in
decode, and roughly 1.2% faster by the sum of separately timed model phases.

The next milestone is therefore not merely a higher synthetic decode number.
It is a reproducible reduction in total latency for transcript-shaped requests,
without losing the scalar fallback or introducing material output degradation.

## Non-negotiable engineering rules

1. Make one performance hypothesis per commit whenever practical.
2. Build and run the full CPU test suite before performance measurement.
3. Add differential scalar-versus-SIMD tests before changing numerically
   sensitive kernels.
4. Use only the repository's sequential benchmark harnesses; write results to
   CSV and never benchmark variants concurrently.
5. Compare against the immediately preceding stable binary under similar
   temperature, frequency, power, and background-load conditions.
6. Retain a change only when repeated measurements show a useful improvement
   or when it has an independently valuable correctness/portability benefit.
7. Report regressions and reverted ideas in this document so they are not
   rediscovered later.
8. Do not keep duplicate permanent model-weight layouts. A new production
   packing must replace and release the old representation.
9. Do not compile the entire application with `-march=native`, AVX2, AVX-512,
   or AMX. ISA-specific code remains isolated in translation units.
10. Runtime feature checks must include OS extended-state support through
    XGETBV where required, not CPUID flags alone.

## Measurement matrix

Every material optimization should run the smallest relevant subset first and
the full matrix before a milestone merge.

| Workload | Primary purpose |
| --- | --- |
| pp64 | Small-prefill dispatch and fixed overhead |
| pp256 | Continuity with the established comparison |
| pp2048 | Realistic transcript prefill and cache behavior |
| prompt-1 / tg128 | Base decode and projection throughput |
| context-256 / tg128 | Medium attention and KV behavior |
| context-2048 / tg128 | Long-context attention and GQA behavior |
| 2048 input / 128 output | Frequent practical cleanup-shaped request |
| 2048 input / 2048 output | Conservative target upper bound |

Default progress runs use six physical-core threads, greedy sampling, one
warmup and at least three measured repetitions. Changes under roughly 2% should
use more repetitions and preferably an alternating A-B-B-A order. Record the
median, spread, temperature, effective clocks when available, and total process
RSS for data-layout changes.

The comparison must explicitly state timer boundaries. qwen35x decode profiles
currently include greedy sampling; `llama-bench` excludes sampling,
tokenization, model loading, and chat-template work.

## Workstream A: laptop AVX2/FMA/F16C

This is the active implementation track. Items are ordered to front-load
low-risk removal of redundant work, then progress toward layout and algorithm
changes.

### A0 — Establish repeatable baselines

Status: pending

- Preserve the current `bac3e6d` implementation binary or rebuild it in a
  separate baseline directory.
- Capture pp64, pp256, pp2048, prompt-1/tg128, and 2048/128 results with six
  threads.
- Capture the same relevant llama.cpp six-thread measurements.
- Record CPU topology, governor/power mode, effective frequency if available,
  compiler version, model hash, and commit hashes.
- Prefer physical-core affinity when a reliable portable mechanism has been
  implemented; until then keep the machine state as consistent as possible.

Acceptance: complete CSV and system metadata sufficient to compare all later
retained commits against the stable baseline.

### A1 — Remove redundant activation-quantization work

Status: pending

Implementation:

- Replace the AVX2 quantizer's software `float_to_half()` call with F16C
  `_cvtss_sh` under the already-required AVX2/FMA/F16C backend.
- Reuse the four loaded YMM source vectors during quantization rather than
  reloading the same 32 FP32 values through pointer-based helpers.
- Add `q8_0_quantize_with_scales`, which stores the FP16 block scale and the
  exact same rounded value as an FP32 sidecar in one pass.
- Use the fused interface in batched prefill and remove the subsequent
  `q8_0_scales_to_f32` pass over quantized activations.
- Investigate token-parallel quantization only after the fused serial path is
  measured; thread dispatch must be amortized by batch size.

Validation:

- preserve bit-identical finite normal quantization results where F16C and the
  portable conversion agree;
- explicitly cover zero, signed zero, subnormal boundaries, maximum finite
  half, overflow, infinity, and NaN behavior;
- compare scalar and AVX2 dequantized results and sidecar scales.

Measurement gate: pp64/pp256/pp2048 and prompt-1/tg128. Keep individual pieces
separate if only one is measurably beneficial.

### A2 — Request-level RoPE cache and AVX2 rotation

Status: pending

Implementation:

- Compute `inv_freq` once when the model/profile is initialized.
- Maintain request-level cosine/sine tables indexed by absolute position.
- Grow tables lazily or allocate through `max_context`; keep allocation outside
  timed token/layer loops.
- Reuse one table across all six full-attention layers.
- Move batched-prefill table construction above the per-layer invocation so a
  prompt chunk is not recomputed for every layer.
- Add an AVX2 rotation kernel over eight pairs with safe scalar tails.

Validation:

- scalar table generation versus the old direct formula over positions
  including 0, 1, 2,047, and maximum context;
- scalar versus AVX2 rotation at odd head counts and tail dimensions;
- deterministic generation smoke tests.

Measurement gate: prompt-1/tg128, pp64, pp256, and 2048/128. Profile RoPE time
separately if existing stage timers are insufficient.

### A3 — Persistent request workspace and span-based projections

Status: pending

Implementation:

- Introduce a 64-byte-aligned `CpuForwardWorkspace` owned per inference request,
  not by shared immutable model weights.
- Size persistent buffers once from model dimensions and maximum context.
- Reuse hidden, norm, projection, intermediate, attention, score, quantized,
  and scale buffers across tokens and layers.
- Replace copied Q/K/V and packed projection slices with non-owning spans where
  lifetimes allow.
- Perform residual updates in place where aliasing is demonstrably safe.
- Move current `CpuQ8Runtime::quantized_input` and `quantized_batch` scratch into
  request-owned state to permit future concurrent requests.

Validation:

- ASan/UBSan run after the lifetime/aliasing conversion;
- repeated requests through one loaded model;
- two concurrent request workspaces, even if the current executor still
  serializes jobs;
- output comparison with the pre-workspace implementation.

Measurement gate: allocation counts, peak RSS, prompt-1/tg128, pp64, pp256,
pp2048, and 2048/128. Land incrementally so a large workspace refactor is not a
single unreviewable commit.

### A4 — Ring-buffer causal convolution with kernel-major AVX2

Status: pending

Implementation:

- Replace history shifting and the per-token `conv_window` with a three-slot
  ring state and a current index for the known kernel size four.
- Repack convolution weights once from `[channel][kernel]` to
  `[kernel][channel]`, replacing the original runtime representation when safe.
- Fully unroll the four taps.
- Process eight channels per AVX2 iteration with scalar tails.
- Fuse the immediately following SiLU when this does not force an extra store
  or alter tested numerical behavior.
- For prefill, process token then contiguous channel vectors while maintaining
  correct recurrent ordering for every channel partition.

Validation:

- new scalar reference versus ring scalar versus AVX2 for one token and batches;
- zero state, random state, multiple consecutive calls, non-multiple-of-eight
  channel tails, and exact state after every step;
- full generation smoke and regression tests.

Measurement gate: prompt-1/tg128, pp64, pp256, pp2048, and convolution stage
time. The previously interrupted untested prototype must not simply be restored;
the state layout and correctness tests come first.

### A5 — Per-kernel threading policy and laptop affinity

Status: pending

Implementation:

- Extend executor jobs with an optional maximum worker count or cost estimate.
- Keep large Q8 projections, LM head, and batched GEMM on six physical cores.
- Run small norm/residual jobs inline.
- Sweep one to six workers for eight-head full attention and DeltaNet work,
  separately for short and long contexts.
- Reduce or adapt the current long spin phase before blocking.
- Add optional Linux and Windows physical-core affinity without making affinity
  mandatory or silently pinning unrelated application threads.
- Consider `atomic::wait/notify` only after simpler policy/spin changes are
  measured.

Validation: executor stress tests, nested/busy behavior, shutdown during idle
and active work, affinity-disabled behavior, and multiple topology shapes.

Measurement gate: all short workloads plus sustained repeated 2048/128 runs;
track clocks/temperature because reduced spinning may improve performance by
preserving turbo rather than by reducing a directly timed stage.

### A6 — Greedy LM-head matvec plus argmax fusion

Status: pending

Implementation:

- Add a greedy-only Q8 matvec variant returning one deterministic local maximum
  per worker rather than materializing all vocabulary logits.
- Apply repetition penalty during each worker's reduction.
- Define and test tie-breaking explicitly, preferring the lower token ID when
  adjusted values are equal if that matches current behavior.
- Reduce worker maxima deterministically.
- Retain the normal logits path for temperature sampling, top-k/top-p, debug
  output, and APIs that explicitly request logits.
- Stop allocating the embedding/LM-head FP32 scale sidecar if no retained batch
  kernel consumes it; expected resident-memory saving is about 30 MiB.

Validation: fused versus materialized greedy selection across randomized
weights/inputs, repeated tokens, positive/negative logits, ties, and penalty 1;
multi-token deterministic generation comparison.

Measurement gate: prompt-1/tg128, context-256/tg128, context-2048/tg128, LM-head
stage time, total decode, and peak RSS.

### A7 — Full-attention low-risk fixes and test expansion

Status: pending

Implementation:

- Add a direct scalar-versus-AVX2 attention differential test before changing
  the kernel.
- In sigmoid, compute one reciprocal and multiply the negative branch by it
  instead of issuing a second vector division.
- Normalize each softmax probability row once in place and make weighted sum
  consume probabilities, avoiding repeated normalization for each 64-column
  output tile.

Test context lengths: 1, 7, 8, 9, 63, 64, 65, 255, 256, 257, and 2,048, with
causal masking and numerical edge cases.

Measurement gate: pp64/pp256/pp2048 and all three decode context depths. Small
changes require higher repetition counts because attention is only part of the
overall runtime.

### A8 — DeltaNet algebra and value-row tiling

Status: pending

Implementation:

- Algebraically compute old-state `dot(k)` and `dot(q)` in one read pass.
- Compute `k dot q` once per head.
- Use a second pass for the single final state write.
- Process two or four value rows together so Q and K remain resident and are
  reused.
- Keep the old implementation available during differential validation.

Validation: long sequence state/output differential tests, adversarial alpha
and beta values, chunk boundary tests, scalar-versus-AVX2 comparisons, and a
deterministic transcript-quality suite. The altered FP32 reduction order makes
quality validation mandatory even when the algebra is equivalent.

Measurement gate: prompt-1/tg128, pp256, pp2048, 2048/128, and DeltaNet stage
time. FP16 recurrent state is explicitly deferred as a separate quality-risk
experiment.

### A9 — Grouped-query attention and online softmax

Status: pending

Implementation order:

1. group four query heads by each shared K/V head during decode;
2. load each K vector once per context position for four Q dots;
3. reuse each V vector across four probability streams;
4. after the grouped kernel is correct, evaluate blockwise numerically stable
   online softmax to eliminate full score rows;
5. adapt prefill only after decode results justify the complexity.

Validation: differential tests at the A7 context lengths, extreme scores,
causal boundaries, and grouped/un-grouped parity.

Measurement gate: context-256/tg128, context-2048/tg128, pp2048, memory usage,
and attention stage time. Expect little benefit at context one.

### A10 — Producer-to-Q8 fusion

Status: pending

Candidates:

- RMSNorm to Q8;
- SiLU-multiply to Q8;
- Delta output gate to Q8;
- final RMSNorm to Q8 for the LM head.

Each producer should compute its 32-value Q8 block maximum and immediately emit
the rounded FP16 scale, FP32 sidecar only when required, and quantized bytes,
without storing and rereading a full FP32 output vector. Start with final RMS to
Q8 because its single consumer is the expensive LM head.

Validation: fused versus unfused Q8 blocks, logits, selected tokens, and longer
quality runs. Measurement must include the removed FP32 traffic as well as any
extra recomputation needed for the local maximum pass.

## Workstream B: system-prompt state cache

Status: pending

This is likely the largest product-level latency gain for repeated transcript
cleanup requests and can proceed independently of newer ISA work.

Snapshot after the invariant prefix:

- DeltaNet recurrent states for every linear-attention layer;
- causal-convolution histories and ring indices;
- full-attention K/V cache content;
- absolute token position and any cache lengths;
- tokenizer/chat-template identity needed to reproduce the exact token prefix.

Cache key and format must include:

- model-content hash and architecture/profile;
- exact system-prompt token IDs;
- weight, cache, and activation precision choices;
- engine state-format ABI/version;
- maximum-context/cache layout fields;
- endianness and any ISA-independent serialization constraints.

First implement an in-memory clone/restore API. Add disk serialization only
after state ownership and invalidation are proven. Measure restore/copy time,
memory cost, saved prefill tokens, time to first generated token, and total
cleanup latency.

## Workstream C: future modern x86 backends

These items must be implemented and benchmarked on suitable hardware. They are
documented now but are not to be compiled into the AVX2 path accidentally.

### C1 — Per-operation kernel dispatch table

Status: future-hardware pending

Replace the increasingly coarse global backend choice with a resolved immutable
kernel set, for example quantize, matvec, matmul, RMS, SiLU, convolution,
DeltaNet, attention, and LM-head-argmax function pointers. Resolve once during
model/request setup rather than repeatedly inside small helpers.

Feature detection should distinguish at least:

- scalar x86-64;
- optional SSSE3/SSE4.1;
- AVX2 + FMA + F16C;
- AVX-VNNI (256-bit);
- AVX-512F/BW/VL as actually required by each FP32/int8 kernel;
- AVX-512 VNNI;
- AMX INT8 plus operating-system tile-state permission.

### C2 — 256-bit AVX-VNNI Q8 backend

Status: future-hardware pending

This is the first modern Q8 dot-product backend. A Q8 block's 32 bytes fit one
YMM register exactly. Initially preserve the established sign/absolute-value
mapping and replace the AVX2 multiply-add sequence with the appropriate VNNI
dot-product instruction. Benchmark decode and prefill independently.

### C3 — AVX-512 FP32 kernels and 256-bit EVEX/VNNI tiling

Status: future-hardware pending

Use AVX-512 where it naturally benefits normalization, residuals, SiLU, RoPE,
convolution, attention, and DeltaNet. For Q8, first evaluate 256-bit EVEX/VNNI
using the larger register file and larger output tiles before assuming ZMM is
better. Avoid unnecessary frequency reduction from wide instructions.

### C4 — Replacing Q8 weight layout

Status: future-hardware pending

Prototype an aligned separated layout:

```text
quants: [row tile][block][row in tile][32 int8]
scales: [row tile][block][row in tile] binary16
```

Use eight-row tiles for AVX2/256-bit VNNI and evaluate 16-row tiles for
AVX-512. The packed representation must replace the canonical in-memory blocks
after load; measure load time and RSS and release the source blocks. Do not
repeat the already-rejected duplicate-layout experiment.

### C5 — Native 512-bit Q8 and AMX prefill experiments

Status: future-hardware pending

Only attempt native ZMM Q8 after the scale/layout problem is addressed. Two
Q8_0 blocks in a ZMM have independent scales and are not contiguous in the
canonical 34-byte layout. AMX INT8 is a later batched-prefill experiment with
explicit tile packing and amortized setup; it is not the first decode target.

### C6 — Older x86 acceleration

Status: future-hardware pending

If broad pre-AVX2 performance is a product requirement, add an SSSE3/SSE4.1 Q8
dot backend using `pabsb`, `psignb`, `pmaddubsw`, and `pmaddwd`. Keep the fully
scalar path as the final correctness fallback.

## Known rejected approaches

Do not repeat these unchanged:

- precomputed FP32 decode scales that add a second memory stream;
- duplicate permanent contiguous or packed weight representations;
- block-major activations with a larger working set;
- blanket 2x4 or 1x8 prefill tiles without a new packing hypothesis;
- CPU prefill chunk size 256 on this host;
- eight inference threads on the six-core laptop;
- generic helper dispatch for tiny decode residual loops;
- `-mtune=haswell` on the i7-8750H;
- the interrupted convolution prototype that lacked build/test/benchmark proof.

## Progress log

Update this table in the same commit that lands or rejects each experiment.

| ID | Status | Commit | Evidence / decision |
| --- | --- | --- | --- |
| A0 | Pending | — | Stable reference figures exist; expanded controlled baseline still required |
| A1 | Pending | — | External review identified F16C call/reloads and a second scale pass |
| A2 | Pending | — | Decode and each full-attention prefill layer currently recompute trigonometry |
| A3 | Pending | — | Local vectors and model-global quantization scratch remain |
| A4 | Pending | — | Current decode creates/copies a convolution window; clean redesign required |
| A5 | Pending | — | Global `min_parallel_rows=1` and long spin phase remain |
| A6 | Pending | — | Greedy path avoids a copy but still materializes and scans all logits |
| A7 | Pending | — | Attention test and low-risk normalization/division fixes remain |
| A8 | Pending | — | Algebra/row tiling unimplemented |
| A9 | Pending | — | GQA heads still largely independent |
| A10 | Pending | — | FP32 producer buffers still precede Q8 quantization |
| B | Pending | — | Prefix state cache not implemented |
| C1-C6 | Future hardware | — | Requires suitable ISA hosts for validation |

## Completion definition

The laptop phase is complete when retained A-series changes are tested,
benchmarked against the stable baseline and current llama.cpp, documented here,
and validated on at least one realistic transcript workload. Any deferred item
must state whether it is blocked by hardware, low expected return, complexity,
or measured regression.

The modern-CPU phase is complete when the same source builds one safe universal
binary, dispatches each kernel only when both CPU and OS support it, passes the
portable/scalar test suite, and has reproducible measurements for AVX2,
AVX-VNNI, AVX-512, AVX-512 VNNI, and any retained AMX path on appropriate hosts.
