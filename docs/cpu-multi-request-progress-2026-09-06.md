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

## Step 2a: Static decode with shared projections

`advance_batch` jointly projects independent rows through all layers, including
the tied LM-head. It produces full logits, applies request-local sampling, and
retains the existing fast path for B=1. Stateful attention and recurrence are
initially dispatched per request. Counters expose the shared projection path.

The model test checks byte-exact raw logits and seeded outputs for B=2/3/4/5/17,
including different positions (3/65/128/257 prompt tokens), row permutations,
completion/removal and duplicate-ID rejection. All nine CTest suites pass and
the normal CUDA-enabled executable builds. On P512, every output token in all
three B4/B8 batched runs also matches the corresponding serial workload.

### Static throughput baseline

Ryzen 9 9955HX3D, eight threads, 0xffff affinity, FP16 KV, same DOT4 artifact.
Medians of three measured runs after one warmup, official sequential runner.
Decode counts B*(128-1) forwards after all first tokens have been produced,
including sampling and the caller loop. No shared-prefix reuse yet.

| Prompt per request | B | Aggregate token/s | Tick p50 (ms) | Peak process RSS (MiB) |
|---:|---:|---:|---:|---:|
| 512 | 1 | 125.16 | 7.95 | 469 |
| 512 | 2 | 163.21 | 12.23 | 501 |
| 512 | 4 | 318.92 | 12.47 | 561 |
| 512 | 8 | 345.61 | 23.06 | 681 |
| 512 | 16 | 503.23 | 31.78 | 920 |
| 512 | 32 | 517.01 | 61.57 | 1400 |
| 4096 | 1 | 99.01 | 10.06 | 512 |
| 4096 | 4 | 181.53 | 22.01 | 733 |
| 4096 | 8 | 202.43 | 39.50 | 1024 |
| 4096 | 16 | 238.84 | 67.16 | 1607 |
| 4096 | 32 | 236.94 | 133.66 | 2773 |

Serial execution of the same P512 B4/B8 workloads gives 119.35/119.10 token/s.
Shared projections therefore improve aggregate throughput by 2.67x/2.90x in
those directly matched cases. Larger batches trade latency for throughput;
these results do not establish a universal optimum. The P4096 B2 point and
extensions above B32 are deferred until thread/layout experiments.

[All 39 samples](cpu-multi-request-static-2026-09-06-samples.csv) and
[binary hashes, settings and topology](cpu-multi-request-static-2026-09-06-metadata.json).
Metadata records the parent revision plus the exact measured executable hash;
the static implementation was uncommitted when those binaries were measured.
See [API and reproduction instructions](cpu-engine-api.md).

Single-request ABBA confirmation against the frozen pre-refactor executable
(six samples each): decode P128 is 126.03 versus 125.33 token/s (-0.56%);
prefill P512 is 2250.95 versus 2217.13 token/s (-1.50%). The candidate includes
one 1277.55 token/s prefill outlier, retained in the data, and its fastest sample
is 2265.47 versus 2256.59 for baseline. These noisy runs do not show a material
single-request regression. Raw records:
`benchmarks/cpu-experiments-2026-09-06/multi-static-single-confirm`.

## Step 2b: Shared greedy vocabulary reduction

For all-greedy batches, prepare each activation once, scan shared vocabulary
tiles, and reduce private repetition-penalized maxima with lowest-token tie
breaking. Each worker uses a small local logits tile instead of writing and
rereading full vocabulary rows. General sampling, forced tokens and callbacks
retain full logits. The batch greedy path is enabled by default and can be
disabled explicitly for comparison.

P512, eight threads, 0xffff, sequential A/B/B/A with three measured runs and
one warmup per leg (six samples per path):

| B | Full logits token/s | Greedy tiles token/s | Gain | Tick p50 before / after (ms) |
|---:|---:|---:|---:|---:|
| 4 | 318.51 | 338.39 | +6.2% | 12.54 / 11.80 |
| 16 | 501.77 | 566.48 | +12.9% | 31.75 / 28.24 |

Every 128-token request output matches the full-logits reference in every
measured run. The full-model test also compares 32 greedy outputs with a 1.2
repetition penalty at B=3/4/5/16/17, reverses row order between ticks and checks
that the shared greedy LM-head ran on all 31 decode ticks. All cases pass;
the existing complete-logit/isolation tests continue to pass.

[Samples](cpu-multi-request-greedy-2026-09-06-samples.csv) ·
[Binary hashes and settings](cpu-multi-request-greedy-2026-09-06-metadata.json).

## Next

Confirm thread/topology effects on the updated state-dispatch path and extend
selected batch sizes. Steps 3-7 remain pending.

## Step 2c: Joint dispatch of private state updates

When B >= executor thread count, process independent attention/DeltaNet rows in
one shared dispatch per layer. Each row has private scratch. Worker callbacks
run the stateful kernel directly and never recursively dispatch the executor.
Smaller batches and B=1 retain their earlier paths. A worker failure poisons the
whole partially updated batch instead of retrying state mutations.

P512, eight threads, 0xffff, greedy vocabulary tiles enabled on both paths,
sequential A/B/B/A (six samples per path):

| B | Per-request dispatch token/s | Joint dispatch token/s | Gain | Tick p50 before / after (ms) |
|---:|---:|---:|---:|---:|
| 8 | 424.34 | 482.51 | +13.7% | 18.84 / 16.57 |
| 16 | 538.87 | 647.21 | +20.1% | 29.43 / 24.74 |

All measured output arrays match. Full-vocabulary model comparisons exercise
the joint path with two and eight worker threads, varying prompt positions,
reordered rows and completion tails; logits remain byte-identical. The shared
state-dispatch counter is exactly 24*127 = 3048 for each measured batch.

[Samples](cpu-multi-request-state-2026-09-06-samples.csv) ·
[Binary hashes and settings](cpu-multi-request-state-2026-09-06-metadata.json).
