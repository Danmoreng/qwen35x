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

## 2. Workload-aware participant counts: rejected

Decode-only trial caps Q4 matvec participants at floor(rows*columns/min_work),
clamped to at least one and at most the configured pool. The LM-head is unchanged.
DeltaNet has a separate participant cap. Each comparison uses ABBA, six measured
samples per cell. Outputs for 65/257 prompt tokens plus three forced tokens are
byte-identical for the combined 524288/four-thread candidate.

| Candidate | Prompt | Baseline token/s | Candidate token/s |
|---|---:|---:|---:|
| min work 524288, DeltaNet 4 | 128 | 124.71 | 125.68 |
| min work 524288, DeltaNet 4 | 512 | 122.91 | 123.20 |
| DeltaNet 4 alone | 128 | 125.68 | 125.60 |
| min work 524288 alone | 128 | 129.13 | 129.50 |
| min work 1048576, DeltaNet 2 | 128 | 129.40 | 129.46 |

Small initial gains are not supported by the isolated/repeat comparisons.
All participant changes and experimental environment switches were reverted.
This does not exhaust topology-aware scheduling or every possible thread count;
it resolves these concrete, profile-guided candidates on this placement.

## 3. Four integer chains in EVEX prefill: rejected

Repeated the older unconfirmed prefill indication, now restricted to X4
activation blocks: decode and argmax keep two chains. Existing integer bounds
and FP32 FMA order are preserved. 288 DOT4 test combinations pass, and full
model logits for 65/257 prompt tokens plus three forced tokens are byte-identical.

| Prefill tokens | Two chains token/s | Four chains token/s | Change |
|---|---:|---:|---:|
| 512 | 2199.97 | 2206.01 | +0.27% |
| 4096 | 1916.48 | 1894.24 | -1.16% |

ABBA, six samples per cell; no outliers removed. One baseline P512 measurement
was 1285.54 token/s, retained in the CSV. The larger-context result and near-equal
short-context medians do not support adoption. Reverted the candidate.

The opt-in diagnostic commit af53d38 was pulled and built on the Intel i7-8750H:
all nine tests and the model cache/precision/diagnostic regression pass there.
Experimental patches remain under `benchmarks/cpu-experiments-2026-09-06`.

## 4. Intel AVX2 LUT experiment

Pending. H256 remains deferred.
