# Native CPU Q4_0 implementation report

Status date: 2026-08-31  
Host: Intel Core i7-8750H, 6 cores / 12 threads, AVX2/FMA/F16C  
Branch: `codex/cpu-q8-avx2`

## Outcome

The native Q4_0-weight/Q8-activation path is operational for the complete
Qwen3.5 0.8B CPU model. On the six-core laptop it now exceeds the saved
llama.cpp Q4_0 comparison in short prefill and decode, and is approximately
26% faster at 2,048-token prefill.

| Workload | Native qwen35x | llama.cpp Q4_0 | Native difference |
| --- | ---: | ---: | ---: |
| pp256 | 319.01 tok/s mean | 227.90 tok/s | +40.0% |
| pp2048 | 264.82 tok/s mean | 202.53 tok/s median | +30.8% |
| prompt-1 / tg128 | 62.75 tok/s mean | 45.50 tok/s | +37.9% |

The qwen35x decode timer includes greedy sampling, while `llama-bench` excludes
sampling. Timer boundaries therefore favor llama.cpp slightly in that row.

## Implemented path

- GGUF Q4_0 tensor validation and reading with the canonical 18-byte block.
- Portable scalar dequantization, Q4_0 x Q8_0 dot, matvec, and batched matmul.
- AVX2/FMA/F16C in an isolated translation unit with safe runtime dispatch.
- Eight-output-row decode and a packed output-vector decode kernel.
- A size-neutral 144-byte `Q4_0BlockX8`: eight FP16 scales plus 128 interleaved
  quant bytes, with nibble sign bits flipped once at load time.
- Direct four-token F32-to-packed-Q8 quantization for prefill.
- Token-major prepared-Q8 quant bytes, replacing the former eight-byte
  interleave and allowing two contiguous 16-byte activation loads.
- Exact int16 activation sums emitted during packed-Q8 quantization.
- FP16-rounded activation scales expanded to FP32 once during quantization.
- A prepared single-vector decode path that computes Q8 bytes, scale, and sum
  once per projection rather than once per eight output rows.
- Greedy-only tied LM-head reduction to one deterministic maximum per executor
  partition, including repetition penalty and lower-token tie-breaking.
- Four-value-row AVX2 tiling for the batched DeltaNet scan; Q/K vectors are
  reused and the intermediate decayed-state store/reload is eliminated while
  preserving the original output reduction order.
- Decode-only two-query-head GQA tiling. Each pair shares one KV head, so K/V
  cache vectors are loaded once for two score and value streams while four
  executor tasks remain available on the six-core host.
- Unsigned Q4 AVX2 dot products with exact `-8 * activation_sum` correction.
- Retained eight-token x eight-output-row AVX2 prefill kernel.
- Eight-row-tile executor scheduling, including tail-token packed matvec.
- Packed embedding row gather and packed tied LM-head matvec.
- Release of canonical Q4 blocks and FP32 weight-scale sidecars after final
  projection concatenation and packing; no permanent duplicate Q4 model copy.

## Controlled performance history

All engine measurements used `scripts/benchmark-inference-seq.sh`, one warmup,
three measured runs, six threads, and the same pure Q4_0 GGUF.

| Variant | pp256 | prompt-1 / tg128 | Decision |
| --- | ---: | ---: | --- |
| Canonical Q4, initial prefill tile | 171.82 | 55.30 | Correct baseline |
| Canonical row-major 1x8 | 180.15 | not repeated | Better but insufficient |
| Packed 4-token x 8-row | 204.38 | not repeated | Retained foundation |
| Direct packed-Q8 4x8 | 205.40 | not repeated | Retained architecture; small isolated gain |
| Direct packed-Q8 8x8 | 235.19 | 56.10 | Retained prefill winner |
| Packed-only weights and decode | 234.73 | 59.01 | Retained |
| Vector F16C load for eight weight scales | 249.94 | 60.78 | Retained foundation |
| Unsigned Q4 plus prepared activation sums | 268.86 | 60.68 | Retained foundation |
| Prepared FP32 activation scales | 281.52 | 60.57 | Retained foundation |
| Token-major Q8 plus prepared decode | 306.26 | 61.41 | Retained foundation |
| Fused greedy LM-head/argmax | unchanged | 62.75 | Current implementation |
| Batched-only DeltaNet 4-row tile | 319.01 | neutral | Current implementation |
| Two-head GQA decode tile | unchanged | 62.81 | Retained; larger at long context |

The current step also measured 255.91 tok/s at pp2048 versus 237.58 before it
(+7.7%). Its pp256 gain is +8.8%. The five-run decode confirmation is +1.4%
over the immediate 60.57 tok/s baseline. A 16-token x 8-row prefill tile was
rejected before this change: 282.72 pp256 and 239.31 pp2048 were only +0.4%
and +0.7%, respectively, within normal host variance.

Fused greedy selection raises the five-run decode mean from 61.41 to 62.75
tok/s (+2.2%). Eight earlier fusion runs measured 62.10--62.20 tok/s before
scratch was reduced from one result per vocabulary tile to one per executor
partition. A 16-token end-to-end comparison against a separately built
`12f9ecf` binary produced identical token IDs with repetition penalty enabled.
The probabilistic temperature path remains on the materialized-logit path and
also passes a full-model smoke test.

The safe batched DeltaNet tile improves pp256 from 306.26 to 319.01 tok/s
(+4.2%) and pp2048 from 255.91 to 264.82 tok/s (+3.5%). Ordered/reverse decode
A/B pairs were inconsistent: the reverse pair measured 62.52 versus 62.50
tok/s (new versus baseline), so no decode gain or regression is claimed. The
single-token kernel remains unchanged. An algebraic output shortcut using
`old_state dot q + delta * (k dot q)` reached 320.65 pp256 and 265.33 pp2048,
but changed long generation and materially worsened one identifier/budget
rewrite; it was rejected. The retained version matches the old 128-token
sequence and all three previously divergent rewrite outputs exactly.

The decode-only GQA pair kernel is neutral at context one (62.81 versus 62.75
tok/s), improves context-256/tg128 from 61.66 to 62.03 tok/s (+0.6%), and
improves context-2048/tg128 from 52.20 to 54.98 tok/s (+5.3%). A second
post-baseline context-2048 run measured 55.12 tok/s, confirming the gain. The
new differential test covers contexts 1 through 2,048 with both odd and native
head dimensions, F16 caches, and scalar/AVX2 dispatch. A 128-token end-to-end
generation remains token-identical to the pre-A9 binary.

Two follow-up A9 variants were rejected. Grouping all four query heads per KV
head reduced the executor to two tasks and measured 61.62 tok/s at short
context and 53.47 tok/s at context 2,048, both below the retained pair kernel.
Fusing probability normalization into the first V tile initially produced a
noisy 55.81 tok/s result, but the controlled baseline/new pair measured 54.96
versus 54.10 tok/s. The original normalization pass is retained.

A10 producer-to-Q8 fusion was tested at three scopes without a retained gain.
Final RMSNorm-to-Q8 measured 62.68 versus 62.65 tok/s over five runs. Direct
RMSNorm-to-Q8 before all combined layer projections measured 62.94 versus
62.96 tok/s. Direct SiLU(gate)-times-up-to-Q8 before every MLP down projection
measured 62.98 versus 62.96 tok/s. All produced the same 128-token sequence,
but packed-Q4 weight traffic dominates enough that removing these FP32 stores
does not change end-to-end decode on this host. All prototypes were reverted.

The pre-vector-scale pp256 thread sweep measured 197.94, 218.53, 234.73, and
188.30 tok/s at 4, 5, 6, and 8 threads. Six physical-core threads remain the
default for this host.

## Correctness and assembly observations

The test suite covers canonical and packed layouts, nibble ordering, negative
Q4 scales, direct packed-Q8 equality, scalar/AVX2 dots, matvec, matmul, output
strides, embedding-row gather, GGUF parsing, and the existing executor and
DeltaNet tests. The complete five-test CPU suite passes in both the Release and
ASan/UBSan builds. Decode-GQA additionally compares the paired path against the
established row kernel at eleven context boundaries, head dimensions 17 and
256, and FP16 cache storage. A forced-scalar full-model smoke run also completes
using the same packed model representation.

GCC keeps the hot AVX2 arithmetic free of function calls, but the 8x8 kernel
does spill YMM values because AVX2 exposes only sixteen vector registers. The
measured weight reuse more than compensates on this host. A future split or
16-token variant must be accepted only by controlled end-to-end A/B results,
not by spill count alone.

## Remaining D1 gates

- Add the full edge-shape matrix, explicit zero-scale/extreme-value known
  answers, and multi-position scalar-versus-AVX2 full-model comparisons.
- Re-run the deterministic transcript-rewrite quality suite on the exact pure
  Q4_0 file used here.
- Measure long-context decode, sustained 2,048-input/2,048-output wall time,
  packing time, permanent RSS, and peak load RSS.
- Inspect Clang assembly and try a lower-spill split kernel only if it improves
  pp256 and pp2048 without reducing decode.

The detailed future work order is maintained in
`docs/cpu-q8-implementation-roadmap.md`, section D1.
