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

Implemented and tested a bounded **exact int16 LUT prototype for the LM-head**,
not a complete T-MAC backend. It uses four bit-planes of the unchanged Q4
codes, 16 subset sums for each four Q8 activations, byte-shuffle lookups and
exact int32 accumulation. FP16 scales and the FP32 FMA order are unchanged.
The packed payload remains 144 bytes per eight rows / 32 columns.

For this prototype the LM-head's bit-plane matrix is prepared once before timed
prefill/decode; original DOT4 embedding/prefill storage is also retained (about
136.41 MiB additional resident weights). Activation tables are built once per
LM-head call and shared by its workers. Conversion/loading and SSH transfer
are excluded from inference timing. A production adoption would require its
own checkpoint encoding rather than the prototype's extra runtime packing.

Machine: Intel i7-8750H, GCC 15.2, Linux, CPU Release. Sequential measurements
use the **same PowerShell benchmark-inference-seq.ps1 runner on Windows**, with
an executable adapter invoking SSH and copying back the engine's profile JSON.
The runner's reported times are the remote engine's internal elapsed intervals.
Six-thread tests bind to CPUs 0-5 (one logical CPU per physical core); twelve
threads bind to CPUs 0-11. Both variants use the same mask within a comparison.
AVX2 is requested strictly, FP16 KV, P128, 128 greedy output tokens, context256.
ABBA with three measured runs plus one warmup per leg; six samples per cell.

| Threads | Existing DOT4 token/s | LUT prototype token/s | Change |
|---|---:|---:|---:|
| 6 | 65.83 | 53.19 | -19.21% |
| 12 | 64.38 | 51.43 | -20.11% |

**Rejected and reverted on both machines.** The extra lookup work and table
traffic are plausible costs, but these timings alone do not prove which one
causes the regression. No claim is made that every LUT design or a fully tuned
T-MAC backend is slower. Six threads are faster than twelve in this measured
configuration; this is not a universal thread-count default.

Validation on Intel: all 72 LUT shape/pattern cases inside the DOT4 test pass
against the existing AVX2 kernel bit-for-bit (extreme -128/127 activations,
zero/negative scales, random weights; argmax also checked). The surrounding
DOT4 suite reports 144 backend cases. Full-vocabulary model logits for P65/P257
plus three forced outputs are byte-identical. The persistent-session test with
FP16/FP32 switching, prefix replay and diagnostic toggling passes. All 24
recorded benchmark runs produce identical 128-token greedy continuations.

Source for reproduction is retained as separate experimental patches:

- [Thread participants](experiments/cpu-workload-participants-2026-09-06.patch)
- [Four-chain prefill](experiments/cpu-chains4-prefill-2026-09-06.patch)
- [Intel LUT](experiments/cpu-avx2-lut-2026-09-06.patch)

Apply each independently to production source at 7c75993 (or af53d38) in a
separate checkout. The LUT patch is an x64/AVX2 research build, not a portable
production backend; enable with QWEN35X_EXPERIMENT_LUT=1. Build through
scripts/build.sh --ninja --no-cuda --all on Linux. Normal inference source
contains none of these experimental switches after reversion.

Baseline Intel executable SHA-256:
`732ea20dc8f5191d629ac500f3639e764207d68b86871e3147008d108d658e55`.
LUT executable SHA-256:
`25c5a234f6c4756081a84985d79eb2bc01ccd516ae2a4baadf71a88d0730abdd`.
Both use the checkpoint SHA-256
`e73de30bf646dee502dd5e519939221f7d2b60068ccbbe16dc1701597919c42f`.

## Outcome

The diagnostic facility is retained; none of the three measured optimization
candidates demonstrated a reliable improvement and none changes production
inference. This completes these four experiments, not every possible CPU
optimization. H256 remains deferred.

Final validation after reverting the candidates: normal CUDA-enabled Windows
Release build and Linux CPU Release build both succeed; all nine CTest suites
and the full-model cache/precision/diagnostic regression pass on both hosts.
The checked-in [Ryzen samples](cpu-experiments-2026-09-06-samples.csv),
[Intel samples](cpu-experiments-2026-09-06-intel-samples.csv) and
[diagnostic aggregates](cpu-experiments-2026-09-06-diagnostics.json) retain the
results. In the aggregate JSON, full-attention columns=0 groups its varying
context lengths; it does not indicate a zero-width executed operation.
