# CPU Multi-Request Implementation Progress

Base: 3cc0e07. The staged design is in
[cpu-multi-request-plan-2026-09-06.md](cpu-multi-request-plan-2026-09-06.md).
Measurements and implementation in this series target the Ryzen only.

## Step 1a: Explicit execution context

Removed mutable runtime pointers from TensorData and execution ownership from
ModelWeights. CPU projections, attention and forward helpers now receive an
explicit CpuExecutionContext pointer. The synchronous request/session owns the
executor and scratch, separately from its weights. Existing CUDA entry points
continue to build with the propagated context argument.

Validation: all nine CPU CTest suites, the full-model prefix/cache/precision
regression, and byte-exact baseline logits for P65/P257 plus three forced tokens
pass. The normal CUDA-enabled Release executable also builds.

Single-request ABBA, three runs plus one warmup per leg, eight threads and
0xffff affinity, same DOT4 model and FP16 KV, official sequential CSV runner:

| Workload | Baseline token/s | Explicit context token/s |
|---|---:|---:|
| Decode P128, 128 outputs | 121.30 | 125.64 |
| Prefill P512 | 2207.50 | 2205.46 |

There is considerable run variation, including baseline decode samples from
111.22 to 126.19 token/s and one candidate prefill sample at 1287.83 token/s.
No outliers were removed. These runs do not establish a speedup; the refactor
is retained as an independently validated structural prerequisite.

Raw records: benchmarks/cpu-experiments-2026-09-06/multi-context.
The frozen pre-change executable is benchmarks/multi-request-2026-09-06/baseline.exe.

## Step 1b: Persistent incremental requests

Added immutable `CpuModel` sharing and an explicit single-owner `CpuEngine` API
with submit, bounded advance, result, cancellation and release. Each request
owns hybrid state, position, a pending token, RNG, penalties and stopping state.
Resident and context limits apply before allocation; IDs are never reused.
The synchronous API remains compatible and uses the same forward helpers.

The optional `qwen35x_cpu_engine_model_test` compares complete raw logits and
outputs against the synchronous API, interleaving prompts of 65 and 257 tokens
with both forced tokens and seeded sampling. Results are byte-identical. It
also checks cancellation, admission limits, stopping, IDs and model sharing
between engines. The existing full-model cache/precision regression passes.
These are structural changes; no additional speedup is claimed.

## Next

Steps 2-7 remain pending. Step 2 adds actual shared decode projections and
measurement of aggregate throughput over a common wall-clock interval.
