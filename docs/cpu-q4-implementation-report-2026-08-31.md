# Native CPU Q4_0 implementation report

Status date: 2026-08-31  
Host: Intel Core i7-8750H, 6 cores / 12 threads, AVX2/FMA/F16C  
Branch: `codex/cpu-q8-avx2`

## Outcome

The native Q4_0-weight/Q8-activation path is operational for the complete
Qwen3.5 0.8B CPU model. On the six-core laptop it now exceeds the saved
llama.cpp Q4_0 comparison in short prefill and decode, and is approximately
tied at 2,048-token prefill.

| Workload | Native qwen35x | llama.cpp Q4_0 | Native difference |
| --- | ---: | ---: | ---: |
| pp256 | 268.86 tok/s mean | 227.90 tok/s | +18.0% |
| pp2048 | 227.71 tok/s mean | 202.53 tok/s median | +12.4% |
| prompt-1 / tg128 | 60.68 tok/s mean | 45.50 tok/s | +33.4% |

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
- Exact int16 activation sums emitted during packed-Q8 quantization.
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
| Unsigned Q4 plus prepared activation sums | 268.86 | 60.68 | Current implementation |

The pre-vector-scale pp256 thread sweep measured 197.94, 218.53, 234.73, and
188.30 tok/s at 4, 5, 6, and 8 threads. Six physical-core threads remain the
default for this host.

## Correctness and assembly observations

The test suite covers canonical and packed layouts, nibble ordering, negative
Q4 scales, direct packed-Q8 equality, scalar/AVX2 dots, matvec, matmul, output
strides, embedding-row gather, GGUF parsing, and the existing executor and
DeltaNet tests. The complete five-test CPU suite passes in both the Release and
ASan/UBSan builds. A forced-scalar full-model smoke run also completes using the
same packed model representation.

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
- A/B signed nibble expansion against unsigned nibbles plus activation sums;
  the latter is the direct bridge to AVX-VNNI `VPDPBUSD`.
- Inspect Clang assembly and try a lower-spill split kernel only if it improves
  pp256 and pp2048 without reducing decode.

The detailed future work order is maintained in
`docs/cpu-q8-implementation-roadmap.md`, section D1.
