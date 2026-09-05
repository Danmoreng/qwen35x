# CPU review implementation, 2026-09-05

Origin AVX2 changes `33b1b6a` were pulled before any experiment. The new X8 AVX2
kernel from that commit is preserved. Review claims are checked against the
code; the supplied document's prototype ZIP was not supplied, so DOT4 was
implemented independently from its stated layout.

## Completed correctness and dispatch work

- `bf0ec32`: completion uses atomic wait/notify, including zero-spin and
  oversubscribed generation/empty-partition tests. The evaluator rejects
  NaN/Inf raw logits with file/position/index and rejects empty dumps and
  non-finite KL. Three Python CLI regression tests pass (including six
  non-finite input cases); seven CTest suites pass.
- `b443bd1`: independent usable CPU capabilities and compiler checks; synthetic
  VEX/EVEX combinations; safe Q8 AVX2 fallback; actual EVEX Q4 decode/argmax
  and safe EVEX prefill tails. MSVC AVX512 dispatch checks the full extension
  set enabled by /arch:AVX512. A strict CLI ISA request is available via
  `--cpu-isa-strict`, also exposed as `-CpuIsaStrict` by the benchmark script.
  JSON and console profiles identify selected per-operation kernels.
  Actual EVEX `vpdpbusd` (62 prefix) was checked in the object disassembly.

## DOT4 packing

Encoding IDs 5 (plain Q4) and 6 (H128 Q4) store eight FP16 scales and 128 bytes
per eight-row/32-column block. For each row r and eight-value group t:
`qs[32*t+4*r+j] = u[r][8*t+j] | (u[r][8*t+4+j] << 4)`.
Old IDs 1-4 remain supported and are never reinterpreted. The existing
`--layout cpu-packed` default is unchanged; use `--layout cpu-dot4` explicitly.
All ISAs can read both formats. No dequantized duplicate or runtime repacking
is introduced. Embedding, decode, fused argmax, prefill and tails support DOT4.

The AVX2 DOT4 kernel accumulates pair-products in int16, with an explicit bound
of `16*15*128=30720` per lane, then performs one widening pair sum. VEX/EVEX
use two independent integer accumulator chains. FP32 cross-block FMA order
is preserved. EVEX prefill uses 16 tokens; AVX2/VEX use four.

New local artifact: `models/qwen3.5-0.8b/model-q4-h128-cpu-dot4.q35h` (not committed).
Generated directly from the original BF16 model. Independent NumPy verification
against the canonical checkpoint checks every FP16 scale, Q4 nibble and F32
byte, including merged projections: 320 canonical tensors to 230 packed.
Eight CTest suites pass. DOT4 tests cover exact packing roundtrip, embedding,
decode, prefill including 16+4-token tails, argmax/penalty/offset, negative and
zero scales, Q8 -128/+127, six K widths, four row counts and all available ISAs.
Full-model logits for a 65-token prompt plus three forced tokens are byte-exact
between X8 and DOT4, separately for AVX2 and AVX512-VNNI.

## Sequential measurements

AMD Ryzen 9 9955HX3D, Windows/MSVC Release, 12 threads, FP16 KV. Process affinity
0x0000ffff for both sides, inherited by child processes. This is experimental
control only, not an engine default or a claim about other CPU families.
All runs use `scripts/benchmark-inference-seq.ps1`, three runs after one warmup,
128 new tokens, max context 256. Per-workload A/B/B/A gives six samples per
variant. No simultaneous compilation/conversion/benchmarks. Throughput uses
normal greedy generation, not the forced-logit mode used for correctness.
The CSV retains every measured sample, including unfavorable ones.

DOT4 versus the post-dispatch X8 baseline, medians:

| ISA | Workload | X8 tok/s | DOT4 tok/s | Change |
|---|---|---:|---:|---:|
| avx2 | decode128 | 125.12 | 125.21 | +0.08% |
| avx2 | prefill256 | 763.08 | 808.10 | +5.90% |
| avx512-vnni | decode128 | 126.12 | 126.36 | +0.19% |
| avx512-vnni | prefill256 | 957.96 | 1069.38 | +11.63% |

The tiny decode differences are within run variation; no decode gain is claimed
for this layout alone. The AVX2 numbers are AVX2 dispatch on this Ryzen, not a
new measurement on the user's older Intel laptop.

## Further measured experiments

- DOT4 ZMM/16-row decode: 120.46 to 116.60 tok/s (-3.20%); reverted.
  Source patch remains locally at `benchmarks/review-zmm16-experiment.patch`.
- Four instead of two EVEX integer chains: decode 122.53 to 122.35 tok/s;
  prefill 1058.74 to 1085.86 tok/s. No decode gain and the small prefill
  difference was not established by a repeat; retained two chains. Patch:
  `benchmarks/review-chains4-experiment.patch`.
- Per-worker mailboxes and participants=min(pool size, work items): decode
  122.03 to 123.68 tok/s (+1.35%), prefill 1097.38 to 1117.92 (+1.87%).
  This eliminates empty participants in four-item GQA jobs and also removes
  the shared job mutex. Inactive workers do not read or acknowledge reused
  job payload. Release/acquire completion protects the next mailbox write;
  mailbox generation and completion both use atomic wait/notify.
  Zero-spin, 64-thread oversubscription, repeated generations, empty jobs,
  busy/reentrant rejection, all eight CTest suites and persistent-model
  prefix/cache-precision tests pass. Full model logits are byte-identical.
  Pool size and spin defaults remain configurable and unchanged; no universal
  physical-core/topology optimum is inferred from this machine.

- Persistent per-session prefill workspaces, with separate buffers for forward,
  linear and full attention: 1110.68 to 1696.05 prefill tok/s (+52.70%).
  Linear attention normalizes Q/K directly in the convolution output and
  writes alpha/beta directly into the head-major batch layout. Full attention
  writes gates directly into the batch output; final normalization consumes
  the last hidden row without copying. Arithmetic order is unchanged.
  All eight CTest suites, byte-exact full-model logits and persistent-session
  prefix/FP16/FP32 switch tests pass. Retained capacities trade resident scratch
  memory for removal of per-layer allocations and zero-initializations.

- FP16-only KV storage: remove the unused FP32 mirror when FP16 is selected.
  At context 256 the six full-attention layers use 3 MiB rather than 9 MiB
  for K/V arrays. FP32 mode remains unchanged. Prefix snapshots already store
  only the selected representation and invalidate on precision/backend changes;
  no FP32 reconstruction from rounded FP16 is attempted. Scalar attention now
  also accepts FP16-only pointers. Kernel and session-switch tests pass, and
  full model logits are byte-identical. Decode 122.19 to 122.11 tok/s and
  prefill 1668.12 to 1693.31 tok/s: no substantial speedup claimed at this context,
  retained for deterministic memory savings and removal of duplicate stores.

- Register-held EVEX-256 DeltaNet-128: 122.21 to 122.38 decode tok/s
  (+0.14%, noise). Full logits remain byte-exact; not retained without gain.
  The local patch and kernel are in benchmarks/review-delta-experiment.patch
  and benchmarks/review-delta-net-evex.cpp.
- Prefill chunks: 64 versus 32 gives 1625.96 versus 1613.47 tok/s;
  64 versus 128 gives 1651.01 versus 1670.70 tok/s. Retained 64; no robust
  improvement established by these small differences.
- Private reusable attention score row per partition: 1704.88 to 1708.59
  prefill tok/s (no speed claim). With 12 threads, the 64-token/eight-head
  chunk needs 12 context rows instead of 512. At context 32768 that is
  1.5 MiB rather than 64 MiB (calculated, not a 32768-context timing).
  Public kernel API retains the original score layout by default. The new
  mode explicitly requires private scratch; each runtime partition owns one
  row. Softmax and summation order are unchanged. Guarded scratch tests
  cover nonzero row starts and contexts up to 2048 on all available ISAs.
  Full logits are byte-exact for 65- and 257-token prompts, and prefix/cache
  precision tests pass.

H256 is deferred at the user's explicit request.

## Final local tuning (no further optimization trials after this point)

- Zero spin: decode 122.21 to 109.71 tok/s; prefill 1688.01 to 1516.51.
  Shortened spin (65536): decode 124.98 to 124.02; prefill 1684.80 to 1707.52.
  Retained the existing default; no consistent throughput gain.
- Fixed K=1024/2048/3584 decode/argmax variants: an initial EVEX result of
  121.64 to 122.99 did not repeat (114.26 to 113.02). AVX2 initially improved
  111.86 to 114.53, but confirmation was 120.06 to 120.37 (+0.26%, noise).
  Reverted all fixed-K variants; local source patch is
  benchmarks/review-fixedk-experiment.patch.
- Thread count with the same 0xffff affinity: 12 versus 8 gives decode
  118.77 versus 121.34 and prefill 1699.23 versus 2145.76. A separate 12 versus
  16 comparison gives decode 116.71 versus 115.05 and prefill 1668.29 versus
  1951.82. These are settings for this placement on this machine, not universal
  defaults. Pool configuration remains unchanged.
- Some early thread trials aborted because the CSV was open in an editor.
  Labels threads8-* and threadscan8-* are incomplete and are excluded from
  comparisons. The complete threadfinal* series is used above. The runner now
  retries only Windows sharing/lock violations after inference; later series
  write separate files. Raw samples are retained, including incomplete series.

The user stopped additional optimization trials and requested a final comparison
against updated llama.cpp. H256 remains explicitly deferred. A new H128
quantization recipe needs a separate held-out quality evaluation; the repository's
single arithmetic regression case is insufficient for acceptance. LUT/T-MAC and
SSE/other-topology backends remain separate, unimplemented research proposals.
No claim that every possible CPU optimization has been exhausted is made.
