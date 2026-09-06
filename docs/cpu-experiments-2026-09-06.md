# CPU experiments on Ryzen and Intel, September 6, 2026

## 1. Decode diagnostics

Added optional `--profile-cpu-decode` (runner: `-ProfileCpuDecode`). Records
cover decode forwards only, excluding prefill and its first predicted token.
Q4 activation preparation, projection/argmax wall time, DeltaNet and full
attention are separate. Executor dispatch, caller partition and post-caller
wait are elapsed intervals, not worker CPU accounting. Wait is already included
in operation wall time and must not be added again. Enabled diagnostics perturb
execution and are not the performance acceptance measurements.

Ryzen 9 9955HX3D, eight threads, affinity 0xffff, FP16 KV, DOT4 checkpoint,
128 greedy output tokens (127 timed forwards), one diagnostic measured run:

| Prompt | Decode ms | Q4 projections ms | LM-head ms | Preparation ms | DeltaNet ms | Full attention ms | Included wait ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1016.45 | 610.19 | 317.25 | 10.97 | 40.93 | 7.27 | 41.55 |
| 512 | 1057.94 | 626.55 | 324.99 | 11.10 | 42.70 | 22.46 | 27.05 |

Normal execution with profiling disabled: ABBA, three measured runs plus one
warmup per leg (six samples per version), official sequential CSV runner.
Frozen 88b6982 binary versus instrumented build: decode P128 125.36 versus
125.80 token/s; prefill P512 2239.82 versus 2236.67 token/s. No speedup is
claimed for instrumentation. All nine CTest suites pass. Additional model
regression switches diagnostics on/off across prefix replays and requires
byte-identical full-vocabulary logits; it passes. Executor tests exercise
profiled empty, serial and parallel jobs.

Local raw records: `benchmarks/cpu-experiments-2026-09-06/{diagnostic,profile-overhead}`.
The starting binary SHA-256 is
`e6db3b22ad9436350989f7c0b1e5eb7779eb3bbc9bce33e1b765c572bbfec08e`.

## Remaining experiments

Workload-aware participant counts, four integer chains in prefill, and an
AVX2 LUT prototype on the Intel i7-8750H are being evaluated separately.
H256 remains deferred. No outcome is claimed for unfinished experiments.
