# Qwen3.5 0.8B Q8 CPU optimization report

Date: 2026-08-31  
Branch: `codex/cpu-q8-avx2`  
Stable implementation baseline: `bac3e6d` (`Vectorize DeltaNet L2 normalization`)

## Executive summary

This work added a native, direct-GGUF Q8_0 CPU inference path for Qwen3.5 0.8B
and optimized both prompt processing (prefill) and autoregressive generation
(decode). The implementation remains portable: it has scalar kernels for
generic x86-64 CPUs and runtime dispatches to AVX2/FMA/F16C kernels when the
host supports them. It does not require AVX-512.

On the test host, an Intel Core i7-8750H with six physical cores, the latest
six-thread build reaches:

| Workload | qwen35x | current llama.cpp | qwen35x difference |
| --- | ---: | ---: | ---: |
| 256-token prefill | 198.90 tok/s | 196.08 tok/s | +1.4% |
| 128-token decode after a one-token prompt | 37.05 tok/s | 31.56 tok/s | +17.4% |

For a more realistic transcript-cleanup run with 2,048 input tokens and 128
generated tokens, qwen35x measured 11.582 s of prefill and 3.774 s of decode,
or 15.355 s for those two model-compute phases. The comparable llama.cpp phase
sum was 15.541 s. qwen35x is therefore about 1.2% faster overall in this
particular test, with faster decode compensating for a 4.1% slower long-prompt
prefill. Timer boundaries are not perfectly identical; details are below.

The current result is a stable and tested checkpoint. An experimental causal
convolution vectorization was deliberately excluded because it had not yet
been built, validated, or benchmarked.

## Target use case

The intended application is local cleanup of speech-to-text transcripts:

1. a short, reusable system instruction describes the cleanup task;
2. an upstream speech model supplies a raw transcript;
3. Qwen3.5 0.8B rewrites it into clean text locally;
4. typical upper-bound planning assumes roughly 2,000 input and 2,000 output
   tokens, though many cleanup responses should be substantially shorter.

Both throughput dimensions matter. Prefill determines how quickly the system
prompt and raw transcript are consumed; decode determines how quickly the
cleaned text is emitted. A future persistent system-prompt state cache could
remove repeated computation for the invariant prompt prefix.

## Test platform and software

- CPU: Intel Core i7-8750H, 6 physical cores / 12 logical threads
- Supported SIMD on this host: AVX2, FMA and F16C; no AVX-512
- Primary engine thread count: 6
- Model: Qwen3.5 0.8B, GGUF Q8_0
- llama.cpp comparison revision: `9723942a`
- qwen35x branch: `codex/cpu-q8-avx2`
- compiler tuning for the AVX2 translation units: `-mtune=skylake`
- portable ISA contract: scalar fallback or AVX2/FMA/F16C selected at runtime

The model file, llama.cpp build tree, generated profiles, CSV files, and local
build directories are intentionally not part of the source-review archive.

## Earlier GTX 1070 compatibility work

Before the CPU path, commit `3b06090` made the existing CUDA engine build and
run on Pascal-class GPUs such as the GTX 1070 while preserving newer paths for
modern GPUs:

- BF16 WMMA compilation is enabled only for Ampere-or-newer targets;
- Pascal uses compatible scalar/storage paths and non-WMMA fallbacks;
- PTX load and memory-fence instructions that require newer architectures have
  guarded Pascal alternatives;
- CUDA 12.8-only NVFP4/cuBLASLt code is compile-time guarded and reports a clear
  unsupported diagnostic on older toolkits;
- newer GPUs still select their available BF16 WMMA, NVFP4, and architecture-
  specific implementations.

This compatibility layer is independent of the CPU Q8 work described below.

## CPU architecture implemented

### Model loading and dispatch

- The reference runtime can load the Qwen3.5 0.8B Q8_0 GGUF directly.
- Projection tensors retain packed Q8_0 weights rather than expanding the
  complete model to FP32.
- Runtime feature detection chooses scalar or AVX2/FMA/F16C backends.
- Unsupported older CPUs keep a correctness-oriented scalar implementation.
- A persistent CPU executor avoids repeatedly constructing worker threads.

### Batched prefill

The original conservative path replayed the prompt one token at a time. The
new layerwise batch path performs prompt work in larger matrices and retains
`--qwen35x-prefill-mode replay` as a fallback. The optimized path contains:

- packed Q8_0 QKV and MLP gate/up projections;
- activation-scale precomputation;
- an AVX2 2x3 batch matrix microtile selected after benchmark comparisons;
- channel-parallel depthwise causal convolution;
- head-parallel DeltaNet recurrence;
- token/head-parallel causal grouped-query attention;
- AVX2 QK and PV operations plus vectorized softmax and gating;
- an FP16 K/V read cache for AVX2 prompt attention while retaining canonical
  FP32 state needed by decode;
- shared per-chunk RoPE lookup tables;
- vectorized SiLU, RMS normalization, residual add, and DeltaNet L2/output
  normalization stages.

### Decode

The optimized single-token path contains:

- AVX2 Q8_0 dot products with four independent accumulators;
- an eight-output-row microkernel that reuses each quantized activation block;
- reuse of activation absolute values with sign transferred to weight vectors;
- optimized full-attention decode shared with the batched attention kernels;
- parallel attention heads and FP16 K/V reads;
- greedy sampling without copying the complete approximately 1 MB logits
  buffer, while applying repetition penalty during the argmax scan;
- vectorized attention SiLU, RMS, and DeltaNet normalization operations.

## Optimization history

| Commit | Change | Main observed result |
| --- | --- | --- |
| `98c4795` | Add optimized Q8 CPU inference path | Direct GGUF Q8 foundation and runtime ISA dispatch |
| `44f66d9` | Optimize batched Q8 CPU prefill | True layerwise batch path; pp256 172.84 tok/s and pp2048 150.03 tok/s in the earlier comparison |
| `0098426` | Optimize CPU full-attention decode | FP16 KV update/read and parallel head attention; decode 30.85 to 32.43 tok/s |
| `0f08352` | Unroll AVX2 Q8 decode dot products | Four independent accumulators; 32.61 tok/s |
| `c743b04` | Avoid greedy sampling logit copies | Sampling stage about 52 ms to 42 ms per 128 tokens |
| `8750a91` | Reuse Q8 activations across AVX2 matvec rows | Four-row reuse kernel; 33.40 tok/s |
| `4936935` | Expand AVX2 Q8 matvec activation reuse | Eight-row reuse kernel; 34.35 tok/s |
| `8838628` | Reuse AVX2 activation signs across Q8 rows | Avoid repeated activation sign work; 34.54 tok/s |
| `52dca9e` | Tune AVX2 CPU kernels for Skylake scheduling | Small host-specific improvement without raising the AVX2 ISA requirement |
| `99cb336` | Vectorize CPU SiLU activations | Cephes-derived AVX2 exponential; pp256 187.06 and decode 35.13 tok/s |
| `b487e24` | Vectorize CPU attention SiLU stages | Convolution and DeltaNet output gates; pp256 190.98 and decode 37.18 tok/s |
| `aaae229` | Vectorize CPU RMS normalization | Global and per-head RMS kernels; pp256 194.72 tok/s |
| `2a67f20` | Vectorize batched CPU residual adds | Kept only where dispatch overhead is amortized; pp256 195.21 tok/s |
| `4646926` | Vectorize DeltaNet output normalization | One combined RMS and one wide SiLU call; pp256 195.90 tok/s |
| `bac3e6d` | Vectorize DeltaNet L2 normalization | Combined L2 and Q scaling; pp256 198.90 and decode 37.05 tok/s |

Intermediate results are useful for direction, not as statistically exact
attribution: laptop clock, temperature, and background activity varied during
the session. The final comparison uses contemporaneous repeated runs.

## Benchmark methodology

All progress measurements ran sequentially and wrote machine-readable results;
ad-hoc concurrent benchmark invocations were avoided. qwen35x uses
`scripts/benchmark-inference-seq.sh`, and llama.cpp uses
`scripts/benchmark-llama-cpu-seq.sh` or the transcript-oriented
`scripts/benchmark-llama-cleanup-seq.sh`.

The standard short qwen35x measurement uses three measured runs after one
warmup, greedy decoding, 128 new tokens, a maximum context of 256, Q8_0 weights,
AVX2 auto-dispatch, batched prefill, and six CPU threads. The llama.cpp short
comparison uses its official `llama-bench` executable with CPU-only execution,
the same Q8_0 GGUF, 256 prompt tokens, 128 generation evaluations, and the
specified thread count.

Important timer distinction:

- `llama-bench` measures model execution and excludes loading, tokenization,
  chat templates, and sampling;
- qwen35x profile decode time includes its greedy sampling work;
- llama.cpp's generation count represents timed one-token evaluations, while a
  production decoder can sample its first output from final prefill logits.

The comparison is consequently close and useful, but not cycle-identical.

## Current benchmark results

### Short six-thread comparison

Final qwen35x results are averages of three measured runs after one warmup.

| Engine | Threads | Prompt work | Prefill | Decode work | Decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen35x | 6 | 256 tokens | 198.90 tok/s | prompt 1 / generate 128 | 37.05 tok/s |
| llama.cpp `9723942a` | 6 | pp256 | 196.08 tok/s | tg128 | 31.56 tok/s |
| llama.cpp `9723942a` | 4 | pp256 | 164.19 tok/s | tg128 | 30.99 tok/s |

Moving llama.cpp from four to six threads improved prompt processing by 19.4%
but decode by only 1.9%. This is expected for memory-bandwidth-heavy batch-one
matrix-vector work. Six threads were also the best observed qwen35x setting on
this six-core host; earlier engine decode measurements were 32.76 tok/s at four
threads, 34.54 tok/s at six threads, and 33.67 tok/s at eight threads. Those
three engine figures predate the final activation vectorization and should be
used only as a thread-scaling indication.

### Realistic 2,048-input / 128-output cleanup profile

| Engine | Prefill | Prefill time | Decode | Decode time | Phase sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen35x, 6 threads | 176.83 tok/s | 11.582 s | 33.92 tok/s | 3.774 s | 15.355 s |
| llama.cpp, 6 threads | 184.47 tok/s | 11.102 s | 28.83 tok/s | 4.439 s | 15.541 s |

The qwen35x model load took about 0.738 s in this run. llama.cpp's combined
phase measured 17.064 s and 127.52 aggregate tok/s, but the separately measured
prefill/decode phase sum is the more transparent comparison above. At this
context qwen35x prefill is 4.1% slower, decode is 17.7% faster, and the phase
sum is about 1.2% faster.

A projected 2,048-input / 2,048-output run at the observed qwen35x rates would
take approximately 11.6 s of prefill plus 60.4 s of decode, or roughly 72 s,
before application overhead. This is only a linear projection; an actual long
generation benchmark is still required because decode cost grows with context,
especially in full-attention layers. Cleanup outputs substantially shorter than
the input will have much lower latency.

### Earlier prefill milestone

Before the later activation work, the first optimized batch implementation
measured 172.84 tok/s for 256 tokens and 150.03 tok/s for 2,048 tokens, compared
with saved llama.cpp measurements of 163.36 and 138.34 tok/s respectively. The
2,048-token qwen35x result was roughly 13.65 s, down from approximately 54 s in
the original one-token replay path. These older results were taken under a
different machine state and should not be mixed with the final head-to-head
numbers.

## Validation and quality checks

The stable checkpoint passed the four CTest targets covering:

- persistent CPU executor behavior;
- GGUF reading;
- scalar and AVX2 Q8/activation operations;
- DeltaNet recurrence.

Unit coverage compares scalar and AVX2 implementations for SiLU, RMS
normalization, vector addition, and L2 normalization. The AVX2 exponential is
a Cephes-derived approximation, so small floating-point and token-selection
differences from the scalar path are expected and tested within tolerances.

A real German transcript-cleanup smoke prompt remained coherent after the SIMD
changes:

> Korrigiere dieses Rohtranskript knapp: hallo äh ich wollte nur sagen das
> treffen ist morgen um zehn uhr.

The 32-token smoke limit ended while the model was introducing the corrected
version, but inference, state updates, and text generation remained valid. A
larger deterministic quality/equivalence corpus is still needed before treating
the approximate activation path as production-qualified.

## Experiments rejected or reverted

These attempts were not retained because they regressed the comparable
benchmark or were not validated:

- FP32 scale arrays precomputed for decode matvec: 30.58 tok/s, likely extra
  memory traffic;
- head-parallel one-token DeltaNet: 32.13 tok/s, dispatch/synchronization cost
  exceeded useful parallelism;
- direct packed MLP gate/up path intended to avoid a copy: formal 34.29 versus
  34.54 tok/s;
- `-mtune=haswell`: prefill fell to 169.97 tok/s; `skylake` scheduling won on
  this host;
- block-major packed activations and duplicate contiguous quantized weights:
  both increased movement/working-set cost;
- alternate batch tiles 2x4 and 1x8: slower than the retained 2x3 tile;
- prefill chunk size 256: slower; the tested CPU path retained 64;
- eight engine threads: slower than six on this six-core CPU;
- dispatching every tiny decode residual add through the generic AVX2 helper:
  overhead outweighed SIMD savings, so decode kept its inline loop;
- a kernel-major AVX2 causal convolution prototype: removed from the stable
  tree because it was interrupted before build, tests, and benchmarks.

Linux `perf` hardware-counter profiling was unavailable because the host has
`perf_event_paranoid=4`. Optimization decisions therefore used stage timers,
controlled A/B builds, and repeated wall-clock throughput measurements.

## Main source entry points

| Area | Files |
| --- | --- |
| Q8_0 scalar/dispatch and AVX2 kernels | `src/cpu/q8_0.cpp`, `src/cpu/q8_0_avx2.cpp`, `src/cpu/q8_0_internal.h` |
| SIMD activation and normalization | `src/cpu/activation.cpp`, `src/cpu/activation_avx2.cpp`, `include/qwen35x/cpu/activation.h` |
| DeltaNet | `src/cpu/delta_net.cpp`, `src/cpu/delta_net_avx2.cpp` |
| Full attention | `src/cpu/full_attention.cpp`, `src/cpu/full_attention_avx2.cpp` |
| Persistent threading | `src/cpu/executor.cpp`, `include/qwen35x/cpu/executor.h` |
| CPU batch prefill orchestration | `src/runtime/reference_inference_internal_cpu_prefill.inl` |
| Single-token layers/decode | `src/runtime/reference_inference_internal_layers.inl` |
| Forward loop and sampling | `src/runtime/reference_inference_internal_forward.inl` |
| Model weights and workspaces | `src/runtime/reference_inference_internal_weights_workspace.inl` |
| Tests | `tests/q8_0_test.cpp`, `tests/delta_net_test.cpp`, `tests/cpu_executor_test.cpp`, `tests/gguf_reader_test.cpp` |
| Sequential benchmarks | `scripts/benchmark-inference-seq.sh`, `scripts/benchmark-llama-cpu-seq.sh`, `scripts/benchmark-llama-cleanup-seq.sh` |

## Highest-value next investigations

1. **Depthwise causal convolution layout and SIMD.** Repack its tiny kernels in
   a channel-contiguous/kernel-major form, vectorize eight channels per step,
   and benchmark both prefill and decode. Keep packing cost out of inference.
2. **Batch Q8 GEMM packing/cache blocking.** The current 2x3 microtile won among
   tested local variants, but a deliberately packed activation panel and
   cache-aware multi-level kernel may improve the long-prompt gap.
3. **Grouped-query decode reuse.** Four query heads share each K/V head. A
   grouped attention job may reuse K/V cache reads and reduce executor overhead.
4. **LM-head decode cost.** Profiling placed roughly 1.06-1.12 s of a 128-token
   run (about 28%) in the vocabulary projection. Explore better row blocking,
   NUMA/cache behavior, and an exact fused argmax. Vocabulary shortlisting is
   only acceptable if a quality evaluation shows no material loss.
5. **Persistent scratch buffers.** Audit inference-loop allocations and reuse
   workspace storage across tokens and requests.
6. **Additional runtime backends.** Add AVX-512 and VNNI kernels behind feature
   dispatch without changing the scalar/AVX2 compatibility baseline.
7. **System-prompt cache.** Serialize or clone the recurrent DeltaNet state and
   full-attention KV cache immediately after the invariant system prefix. Cache
   keys must include model hash, prompt token IDs, precision, and engine ABI.
8. **Full 2,048-to-2,048 measurement.** Run repeated end-to-end cleanup-style
   benchmarks and report time-to-first-token, prefill, decode, sampling, and
   total wall clock separately.
9. **Thread policy.** Test pinning to physical cores and dynamic thread counts:
   prefill benefits strongly from six cores, while decode saturates earlier.
10. **Transcript quality suite.** Build German and English cleanup cases with
    deterministic expected properties, punctuation/casing metrics, factual
    preservation checks, and scalar-versus-SIMD comparisons.

## Review questions for a follow-up code analysis

An external reviewer should particularly examine:

- correctness at tensor tails, unusual dimensions, and scalar fallback paths;
- data layout and cache reuse in `q8_0_avx2.cpp`;
- whether executor task granularity can be reduced without oversubscription;
- opportunities to fuse normalization, activation, quantization, and projection;
- KV/state ownership and what is required for safe prefix-state caching;
- numerical error from the approximate exponential over realistic activations;
- redundant conversions between FP32 activations, FP16 cache data, and Q8
  projection inputs;
- build portability across GCC, Clang, MSVC, older AVX2 CPUs, and future
  AVX-512/VNNI hosts;
- benchmark timer equivalence with llama.cpp and any remaining methodological
  bias.

## Reproduction notes

Representative qwen35x short benchmark:

```bash
scripts/benchmark-inference-seq.sh \
  --executable build-cpu-q8/qwen35x \
  --mode cpu-q8 \
  --cpu-gguf models/gguf/Qwen3.5-0.8B-Q8_0.gguf \
  --cpu-threads 6 \
  --cpu-isa auto \
  --prefill-mode batched \
  --prompt-repeat-token 198 \
  --prompt-token-count 256 \
  --max-new-tokens 0 \
  --max-context 256 \
  --prefill-only \
  --warmup-runs 1 \
  --runs 3 \
  --csv-out benchmarks/qwen-cpu/review-pp256-t6.csv
```

Representative llama.cpp thread comparison:

```bash
scripts/benchmark-llama-cpu-seq.sh \
  --threads 4,6 \
  --prompt-tokens 256 \
  --gen-tokens 128 \
  --repetitions 3 \
  --model models/gguf/Qwen3.5-0.8B-Q8_0.gguf \
  --csv-out benchmarks/llama-cpu/review-t4-t6.csv
```

The benchmark outputs and model are excluded from the review ZIP. The scripts,
source, tests, CMake configuration, and this report are included.
