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

Status: completed; retained as a neutral cleanup with no claimed end-to-end
throughput gain

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

Status: completed; retained as a neutral architectural cleanup

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

Status: completed; retained with a clear prefill gain and a smaller decode gain

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

Status: partially completed; unused embedding/LM-head scale sidecar removed,
greedy matvec/argmax fusion still pending

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

Status: completed; retained with a small measured gain and substantially better
test coverage

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

## Workstream D: lower-bit weights and transcript-quality evaluation

This workstream is active on the AVX2 laptop. It was added after the second
external review and takes priority over speculative sub-four-bit kernels. The
central hypothesis is that decode is sufficiently weight-bandwidth-bound for a
good W4A8 implementation to provide a larger gain than further tuning of the
existing Q8 matvec.

Quantization algorithm and runtime storage format are separate decisions. RTN,
an importance matrix, GPTQ/AWQ-style optimization, or AutoRound may choose
better codes while the runtime still consumes a simple Q4_0-like layout.

### D0 — llama.cpp format and quality preselection

Status: initial laptop screen completed; production quality expansion pending

Benchmark these Qwen3.5 0.8B GGUF variants on the same i7-8750H and current
llama.cpp revision:

| Format | Role in the comparison |
| --- | --- |
| Q8_0 | Current quality and performance reference |
| Q6_K | High-quality lower-bandwidth reference |
| Q5_K_M | Conservative mixed five-bit candidate |
| Q4_K_M | Common quality-oriented four-bit candidate |
| Q4_0 | Simplest prospective custom AVX2 kernel format |
| IQ4_NL | Same nominal block size with nonlinear codebook |
| IQ4_XS | Smaller four-bit reference with more scale decoding |

Prefer files quantized directly from BF16/FP16/FP32 and generated with a
documented importance matrix where available. Do not requantize the existing
Q8_0 file to Q4. Record repository, revision/file name, exact byte size, SHA-256,
GGUF metadata, and quantization provenance.

Performance matrix:

| Workload | Required measurement |
| --- | --- |
| pp256 | Prompt processing and comparison continuity |
| pp2048 | Realistic long transcript prefill |
| prompt-1 / tg128 | Base bandwidth-bound decode |
| context-256 / tg128 | Medium-context attention contribution |
| context-2048 / tg128 | Long-context GQA/KV contribution |
| realistic cleanup | End-to-end prefill, decode, and wall clock |

Use six physical-core threads, CPU-only execution, identical llama.cpp build,
batch/ubatch/cache settings, one warmup and at least three measured runs. Run
formats sequentially, rotate their order for small differences, and record
median/spread as well as averages. Model loading and file size must be reported
separately from timed inference.

### D0 quality mini-suite

Run greedy decoding with the exact same system instruction, chat template,
maximum output length, and deterministic settings for all formats. Start with
the following eight rewrite inputs; preserve the raw input and generated output
for review rather than recording only a subjective score.

1. **German fillers and punctuation**

   ```text
   hallo äh ich wollte nur sagen das treffen ist morgen um zehn uhr und bitte
   bring noch die unterlagen mit
   ```

   Required facts: meeting is tomorrow at 10:00; bring the documents.

2. **German self-correction**

   ```text
   wir liefern am donnerstag also nein entschuldigung am freitag den
   vierzehnten und zwar zwölf kartons an frau schneider
   ```

   Required facts: Friday the 14th, 12 boxes, recipient Frau Schneider; the
   superseded Thursday date must not remain as the final commitment.

3. **German names, identifier, and amount**

   ```text
   notier bitte für herrn özdemir projekt alpha sieben ticket a c vier neun
   zwei budget sind dreizehntausendfünfhundert euro
   ```

   Required facts: Herr Özdemir, Projekt Alpha 7, ticket AC-492, EUR 13,500.

4. **German medical-style dictation without invention**

   ```text
   patient berichtet seit drei tagen über kopfschmerzen kein fieber keine
   übelkeit termin zur kontrolle am zweiten september um acht uhr dreißig
   ```

   Required facts: three days, headache, no fever, no nausea, September 2 at
   08:30. The model must only clean the wording and must not add diagnoses or
   treatment advice.

5. **English fillers and action items**

   ```text
   okay so um send the revised contract to Maya Chen before five p m and copy
   Daniel on the email but do not attach the old pricing sheet
   ```

   Required facts: Maya Chen, before 5 p.m., copy Daniel, exclude old pricing.

6. **English numbers and correction**

   ```text
   the invoice total is four thousand six hundred and eighteen dollars no wait
   four thousand six hundred and eighty dollars and payment is due june twenty
   first
   ```

   Required facts: corrected total USD 4,680 and due date June 21; USD 4,618
   must not be presented as final.

7. **Mixed-language technical transcript**

   ```text
   bitte update den inference server auf version zwei punkt vier punkt eins
   aber lass den cuda driver unverändert und starte danach nur den qwen worker
   neu
   ```

   Required facts: inference server 2.4.1, do not change CUDA driver, restart
   only the Qwen worker.

8. **Longer disfluent paragraph with negation**

   ```text
   also für das protokoll wir haben heute entschieden dass der test nicht am
   montag startet sondern am mittwoch zuerst nur mit fünf nutzern und falls
   fehler auftreten wird nicht automatisch ausgerollt sondern wir prüfen das
   am donnerstag nochmal gemeinsam
   ```

   Required facts: starts Wednesday, not Monday; five users; no automatic
   rollout on errors; joint review Thursday.

For every output score or flag:

- preservation of names, identifiers, numbers, dates, times, negations, and
  corrected statements;
- omissions and unsupported additions;
- punctuation, casing, readability, and removal of disfluencies;
- repetition, truncation, malformed text, and instruction leakage;
- output token count and wall-clock latency;
- exact match of critical-fact assertions;
- optional teacher comparison: token-level KLD/top-k agreement against BF16 or
  at least Q8_0 when a compatible logits harness exists.

The first pass may use a small manual rubric: `critical facts` (pass/fail),
`unsupported additions` (count), and `rewrite quality` (1–5). Keep all raw
outputs so later human review can override automated judgments. A format is not
approved merely because its prose sounds fluent.

D0 decision gate:

- eliminate any format with repeated critical-fact corruption or instability;
- identify the fastest acceptable Q4 and a conservative Q5/Q6 fallback;
- compare prefill and decode separately because Q4 unpacking can improve decode
  while reducing prompt-processing speed;
- select no custom runtime format until these llama.cpp measurements exist.

Laptop result (2026-08-31): the three-round, alternating-order llama.cpp screen
selected Q4_0 as the first native format. Against Q8_0 it measured +45.1%
decode at context 1, +44.6% at context 256, +38.2% at context 2,048, and +23.7%
pp2048. IQ4_NL was effectively tied in throughput but showed a less desirable
long-rewrite failure. The eight-case suite found baseline model failures even in
Q8_0, so production quality approval remains open. A separate three-repetition
2,048-input/2,048-output run measured 68.477 seconds for Q4_0 versus 91.425
seconds for Q8_0: 25.1% lower combined model-compute latency. See
`docs/llama-cpp-quantization-screen.md` for hashes, raw-artifact locations,
methodology, full results, and limitations.

### D1 — Native Q4_0 × Q8 backend

Status: active; laptop throughput target achieved, expanded validation pending

The first implementation slice now includes GGUF Q4_0 parsing, a scalar
reference, isolated AVX2/FMA/F16C dispatch, canonical block kernels, and a
size-neutral eight-row packed representation. Decode, the tied LM head,
embedding row gather, and prefill all consume the packed representation after
load, and the canonical Q4 blocks plus FP32 weight-scale sidecar are released.
The retained prefill kernel computes eight tokens by eight output rows; direct
F32-to-interleaved-Q8 preparation avoids the intermediate 34-byte activation
layout for complete four-token groups. Executor partitions use eight-row tiles.

Current laptop result (six threads, three measured runs after one warmup):

| Workload | qwen35x native Q4_0 | llama.cpp Q4_0 | Difference |
| --- | ---: | ---: | ---: |
| pp256 | 281.52 tok/s | 227.90 tok/s | +23.5% |
| pp2048 | 237.58 tok/s | 202.53 tok/s | +17.3% |
| prompt-1 / tg128 | 60.57 tok/s | 45.50 tok/s | +33.1% |

The original canonical native-Q4 implementation measured 171.82 tok/s at
pp256 and 55.30 tok/s at prompt-1/tg128. A row-major 1x8 prefill tile reached
180.15 tok/s, packed 4-token x 8-row reached 204.38 tok/s, direct packed-Q8
preparation reached 205.40 tok/s, and the retained 8-token x 8-row kernel
reached 235.19 tok/s before canonical-weight removal. Packed-only decode then
improved generation from 56.10 to 59.01 tok/s without a pp256 regression. A
single F16C conversion of all eight packed weight scales, followed by one lane
permutation, produced the current 249.94 pp256, 214.36 pp2048, and 60.78 decode
results.

The next retained A/B replaced signed nibble expansion with unsigned Q4 values
and an exact `-8 * activation_sum` correction. Prepared Q8 blocks now carry one
int16 sum per token, produced during quantization. It improved pp256 from
249.94 to 268.86 tok/s (+7.6%) and pp2048 from 214.36 to 227.71 tok/s (+6.2%),
while prompt-1/tg128 remained effectively neutral at 60.68 versus 60.78 tok/s.
Prepared Q8 then changed activation-scale storage from FP16 to the exact
FP16-rounded value expanded once to FP32 during quantization. This removes
repeated F16C conversions in every output tile and improves pp256 from 268.86
to 281.52 tok/s (+4.7%) and pp2048 from 227.71 to 237.58 tok/s (+4.3%);
decode remains effectively neutral at 60.57 tok/s.

The Q4-specific external-review plan is incorporated as this ordered checklist:

1. **Q4 semantics and correctness — substantially complete.** Preserve signed
   FP16 scales, low/high nibble ordering, exact per-block int32 dots, and scalar
   versus AVX2 differential coverage. Expand the matrix to all requested row,
   block, and batch tails plus explicit zero-scale and extreme-Q8 known answers.
2. **One permanent packed weight layout — complete for the model path.** Eight
   scales and 128 quant bytes occupy exactly 144 bytes, equal to eight canonical
   blocks. Concatenated projections are joined before final packing. Canonical
   model weights are released.
3. **Prepared Q8 activations — scale/sum slice complete.** Four-token groups are
   quantized directly into interleaved scratch and exact int16 activation sums
   plus FP16-rounded FP32 scales are emitted in the same pass. A future A/B should test a
   fully separated 32-byte-aligned quant/scales/sums SoA only when the unsigned
   zero-point kernel is ready; do not add another unconditional repacking copy.
4. **Decode alternatives — unsigned kernel retained.** The canonical eight-lane
   kernel established 55.30 tok/s. The packed output-vector kernel reached
   60.78 tok/s, and unsigned Q4 plus activation-sum correction is neutral at
   60.68 tok/s while materially improving prefill. This is the AVX-VNNI bridge.
5. **Prefill tile search — active.** Retain 8-token x 8-row on this laptop.
   Record 2x3 canonical, 1x8 canonical, 4x8 packed, direct-4x8, and 8x8 results.
   Test 2x4/4x4 or 16x8 only when evidence justifies another experiment.
6. **Tile executor and tied embedding/LM head — complete.** Worker ranges use
   eight-row tiles, with no silent Q8 or FP32 fallback.
7. **Assembly cleanup — active.** GCC emits no helper calls in the retained dot
   loop but spills YMM temporaries for 8x8. The spills are provisionally accepted
   because the end-to-end gain is large. Inspect Clang when available.
8. **Validation gate — active.** ASan/UBSan and a forced-scalar full-model smoke
   pass. Multi-position full-model comparisons, the D0 rewrite suite,
   long-context repeats, and packing-time plus peak/permanent-RSS measurements
   remain. The 4/5/6/8-thread pp256 sweep selected six threads at 197.94,
   218.53, 234.73, and 188.30 tok/s.
9. **Future ISA reuse — planned.** Reuse the format/dispatch for AVX-VNNI,
   AVX-512 VNNI, and possibly AMX. IQ4_NL, Q4_K, Hadamard Q4, KV/state
   quantization, and vocabulary approximation remain outside D1.

Initial merge targets on this laptop:

- at least 20–25% faster decode than the retained Q8 path;
- no more than a small, explicitly measured prefill regression;
- scalar/AVX2 differential tests at all block/tile tails;
- no material failure in the D0 transcript suite;
- total resident model memory must fall rather than contain Q8 and Q4 copies.

### D2 — Native IQ4_NL × Q8 backend

Status: pending D1

Reuse executor and row tiling from D1. Expand nibbles to indices, use a
lane-safe duplicated 16-entry table with `VPSHUFB`, and feed the resulting
signed bytes to the established Q8 dot sequence. Compare cycles/weight,
prefill, decode, RSS, and D0 quality directly against Q4_0.

### D3 — Mixed tensor precision

Status: pending D1/D2

Represent weight format per tensor and dispatch by format, ISA, and
decode/prefill mode. First candidate:

- large MLP and DeltaNet matrices: Q4_0 or IQ4_NL;
- full-attention projections: IQ4_NL or Q5;
- LM head: Q5_0 initially;
- small norms, convolution weights, and parameter vectors: FP16/FP32/current
  representation as appropriate.

Measure tensor ablations and prioritize names, numbers, negation, and output
projection sensitivity. `Q4_K_M` is a mixed model policy, not one homogeneous
block type, and should not be conflated with a simple Q4 runtime kernel.

### D4 — Calibration and offline rounding

Status: pending a stable runtime format

Generate candidates directly from BF16/FP16 using transcript-specific
calibration data:

- naive RTN Q4_0;
- llama.cpp importance-matrix Q4;
- AutoRound Q4_0 and Q4_K_M;
- selected AWQ/GPTQ-like rounding exported into the same runtime layout;
- mixed-precision output/embedding alternatives.

Calibration data must contain German and English raw transcripts, fillers,
self-corrections, names, identifiers, dates, times, numbers, short/long inputs,
and the real system prompt/chat template. Runtime-kernel performance and offline
quantizer quality must remain independently measurable.

### D5 — Custom Hadamard Q4 format

Status: research target after D1–D4

This is the preferred custom-format research direction before dense EXL3:

```text
Q4_H128 = randomized-sign Hadamard-128 regularization + calibrated simple Q4
```

Offline transform weight input blocks by the orthogonal signed Hadamard matrix;
at runtime transform each activation block once, quantize it to Q8, and reuse it
across all projections sharing that input. Proposed metadata includes transform
ID/version, block size, sign seed, quantization method, scale granularity, and
format ABI. Benchmark the transform overhead separately and ensure transformed
activations are reused across packed QKV, gate/up, and DeltaNet projections.

The goal is to capture much of QTIP/EXL3's outlier-regularization benefit while
retaining an extremely simple hardware-native Q4 dot kernel.

### D6 — Quantized KV cache and DeltaNet state

Status: deferred quality-risk experiments

Evaluate Q8 KV cache only at long contexts after weight-only Q4. Evaluate FP16
DeltaNet state with FP32 compute only after long deterministic recurrence and
transcript tests. Neither is expected to help context-one decode as much as
lower-bit weights, and recurrent error accumulation makes state quantization
more sensitive.

### D7 — Dense EXL3/QTIP/Trellis research backend

Status: future research, primarily modern hardware

Treat existing EXL3 CPU code as a reference rather than a drop-in dense engine;
it targets MoE expert offload. A future dense prototype requires CPU-native
packing, Hadamard-128 activation transforms, scalar and AVX2 references,
AVX-512 VNNI/VBMI implementations, and separate dense GEMV and prefill kernels.
Compare two-, three-, and four-bit quality only after D0–D5 establish a strong
four-bit baseline. On the i7-8750H, Q4_0/IQ4_NL should be preferred over Trellis
at four bits because the latter adds decode/transform complexity without VNNI,
AVX-512, or VBMI.

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
| A1 | Completed, neutral | `511d9a5` | F16C conversion and loaded-YMM reuse verified in disassembly; fused scale sidecar removes the second pass. All tests pass. Ordered/reverse A/B pairs were dominated by laptop drift: combined pp256 was 198.59 versus 199.63 tok/s, decode 36.61 versus 36.74 tok/s, and pp2048 168.22 versus 169.56 tok/s (A1 versus baseline). No speedup is claimed. |
| A2 | Completed, neutral | `961478d` | Request tables are sized to the required context, shared by all full-attention layers, and consumed by tested AVX2 rotation. pp256 was 198.32 versus 198.12 tok/s. Ordered/reverse decode pairs averaged 36.88 versus 36.90 tok/s (A2 versus immediate baseline), so no laptop throughput gain is claimed. The cache removes redundant transcendental work and prepares prefix-state caching. |
| A3 | Pending | — | Local vectors and model-global quantization scratch remain |
| A4 | Completed, retained | `4c74dd8` | Three-slot ring state removes decode window copies; kernel-major weights, four-tap AVX2, and fused SiLU are covered by scalar/SIMD state and output tests. Ordered/reverse pp256 pairs averaged 210.48 versus 198.33 tok/s (+6.1%); decode averaged 37.14 versus 36.72 tok/s (+1.2%). pp2048 was 175.48 versus 165.58 tok/s (+6.0%) even though the optimized run was second. A 2,048-token integration run crossed 32 prefill chunk boundaries successfully. |
| A5 | Pending | — | Global `min_parallel_rows=1` and long spin phase remain |
| A6 | Partial | `a0524f2` | The embedding/LM-head tensor no longer allocates an unused FP32 scale sidecar, saving 7,946,240 floats (30.31 MiB) for this model. Greedy matvec/argmax fusion remains pending. |
| A7 | Completed, retained | `d49aec4` | Added scalar/AVX2 differential coverage for contexts 1, 7, 8, 9, 63, 64, 65, 255, 256, 257, and 2,048. Softmax probabilities are normalized once and sigmoid uses one vector division. pp256 was 211.31 versus 209.37 tok/s (+0.9%). Ordered/reverse decode pairs averaged 37.44 versus 37.17 tok/s (+0.7%). |
| A8 | Pending | — | Algebra/row tiling unimplemented |
| A9 | Pending | — | GQA heads still largely independent |
| A10 | Pending | — | FP32 producer buffers still precede Q8 quantization |
| B | Pending | — | Prefix state cache not implemented |
| C1-C6 | Future hardware | — | Requires suitable ISA hosts for validation |
| D0 | Initial screen completed | `4ca364c` | Seven formats, three alternating-order performance rounds, 56 deterministic rewrite outputs, and a three-run 2k/2k Q8-versus-Q4_0 comparison select Q4_0 for the first native backend; production quality expansion remains open |
| D1 | Active; laptop throughput target achieved | pending | Native Q4_0 scalar/AVX2, packed-only model weights, direct prepared activations with FP32 scales and sums, unsigned Q4 correction, tile scheduling, embedding/LM-head coverage, and 8x8 prefill are implemented. Against llama.cpp Q4_0, pp256 is +23.5%, pp2048 is +17.3%, and decode is +33.1%. Expanded correctness, quality, memory, and long-context gates remain. |
| D2-D4 | Pending D1 validation | — | IQ4_NL, mixed precision, and calibrated offline rounding follow the validated Q4_0 baseline |
| D5 | Research target | — | Custom Hadamard-regularized `Q4_H128` is the preferred custom-format direction after simple Q4 baselines |
| D6-D7 | Deferred/future hardware | — | State quantization is quality-sensitive; dense sub-four-bit research benefits materially from newer SIMD ISAs |

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
