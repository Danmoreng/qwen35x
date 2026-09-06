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

## Step 3 checkpoint: Experimental copy-based hybrid prefixes

Implemented explicit immutable prefix registration, bounded payload storage,
namespace/model/configuration checks, private hybrid-state restore, and complete
hits using cached normalized hidden vectors with request-local sampling.
Prefix restore resumes at the remaining part of the original prefill chunk,
avoiding a shift of all subsequent chunk boundaries.

The full-model test checks prefix lengths 65/128/257 with suffix lengths
0/3/129 under forced tokens and seeded sampling: all 18 full-logit comparisons
are byte-identical. Deduplication, namespace/token rejection, distinct loaded
model rejection and pinned-budget eviction tests pass. The existing engine
model tests passed, and all nine CTest tests passed at this checkpoint.

However, the longer benchmark exposes an unresolved numerical difference:
with a 511-token prefix and one private suffix token, 5 of 16 greedy output
sequences differ from cold prefill, starting as early as output index 1.
Every warm measured run reproduces those differences. This is not yet evidence
of a quality loss, but it prevents claiming general equivalence. Investigate
prefix construction and split prefill arithmetic, add a representative
regression fixture and evaluate logit/state error before declaring step 3 done.
The earlier short-token tests are insufficient. Prefix use remains explicit
and experimental; existing requests do not automatically enable it.

P512, B16, eight threads, 0xffff, 128 outputs, sequential cold/warm/warm/cold,
three measured runs after one warmup per leg (six samples per variant):

| Prefix case | Cold initialization ms | Warm initialization ms | Separate prefix build ms | Output equality |
|---|---:|---:|---:|---|
| 511 tokens + private final token | 4675.79 | 209.00 | 403.96 | 5/16 sequences differ |
| Complete 512-token hit | 3761.18 | 108.94 | 301.84 | All identical |

Initialization includes allocation, restore and first-token selection for all
16 requests. Prefix build is excluded from warm time, so the complete-hit
34.5x initialization ratio is not an end-to-end or cold-start speedup. The
complete snapshot contains 26,499,072 payload bytes; total restore-copy time is
16.12 ms. Warm complete hits compute zero new prefill tokens. Decode throughput
is 671.06 cold versus 668.06 warm token/s: no demonstrated decode gain.
Partial-hit timing is diagnostic only until numerical validation is resolved.

## Thread and topology confirmation

Using the committed joint-state/greedy path, P512/B16/128 outputs, same frozen
binary, sequential A/B/B/A with six measured samples per configuration:

| Configuration | Decode token/s | Range |
|---|---:|---:|
| 8 threads, 96 MiB L3 domain, mask 0xffff | 674.96 | 673.11-678.78 |
| 16 threads, one per physical core, mask 0x55555555 | 698.05 | 678.73-709.77 |

All generated output arrays match. The 3.4% median gain applies to this batch
and topology; it is not a universal thread recommendation. Thread count and
placement both change. Larger batches, 24/32 threads, and the selective
B10/20/50/100 extension remain to be measured.

[Samples](cpu-multi-request-prefix-threads-2026-09-06-samples.csv) and
[binary hashes/settings](cpu-multi-request-prefix-threads-2026-09-06-metadata.json)
retain all measured runs. A first-difference index of 128 means no mismatch
within the 128 generated tokens. Thread runs predate the new prefix counters,
so blank prefix fields mean unavailable, not zero measured work.

## Remaining work, in order

1. Finish step 3 numerical validation for realistic partial prefixes and
   suffixes, including long contexts and batched branches. Do not generalize
   the passing complete-hit result to arbitrary prefixes.
2. Complete selected step 2 scale/thread measurements, including larger batches
   and SMT candidates; report latency and memory alongside throughput.
3. Step 4: bounded admission/queues, continuous batching with fair prefill/decode
   ticks, cancellation, output backpressure and single-flight prefix builds.
   Current submit allocates private state immediately and callers manage batches.
4. Step 5: physically shared immutable KV pages, private suffixes and tail-page
   copy-on-write, keeping DeltaNet/convolution state private.
5. Step 6: grouped attention over common prefix panels with stable partial
   softmax merging; demonstrate gains over the same shared-page baseline.
6. Step 7: mixed prefill/decode projection experiment; adopt only measured gains.

Portable topology detection, optional saved workload calibration and a manual
thread override are an additional design direction discussed with the user,
not an implemented autotuner or a prerequisite for finishing prefix validation.
The original implementation plan remains the roadmap; steps 1 and 2 core paths
are implemented, step 3 is experimental, and steps 4-7 are not implemented.

## Step 3 follow-up: Short-tail attention dispatch fixed

The partial-hit discrepancy was caused by the automatic attention selector:
fewer than four remaining prefill tokens switched to row attention, although
the same positions in cold prefill used tiled attention. Removing that batch-size
gate keeps tiled arithmetic beyond position 128, including one-token suffixes.
This changes short prefill tails, not the separate single-token decode kernel.

The realistic token fixture now checks prefixes 127/128/129/255/511/512/1023/4095
with suffixes 1/3/129 and eight forced full-vocabulary outputs. All 24 cases have
zero logit error and byte-identical vectors. The original 18 prefix cases,
identity/budget checks, independent engine model test, and all nine CTest tests
also pass.

Repeated P512/B16/128-output cold/warm/warm/cold comparison (six measured samples
per path, eight threads, mask 0xffff): all request output arrays now match.
Median cold initialization is 4244.75 ms versus 210.59 ms warm, with 255.70 ms
separately measured prefix construction. Cold/warm decode is 668.13/672.27
token/s; this is a prefill reuse improvement, not a demonstrated decode gain.
The earlier mismatch records above are retained as historical evidence, not an
outstanding failure of this revision. Step 3's observed numerical blocker is
resolved; broader serving and shared-page work remains pending.

[Samples](cpu-prefix-tail-fix-2026-09-06-samples.csv) and
[settings/hashes](cpu-prefix-tail-fix-2026-09-06-metadata.json).

## Step 4a: Bounded continuous scheduler

Serving mode now queues request metadata before allocating private model state,
admits at most one request per tick and alternates bounded prefill chunks with
decode batches. Independent queue, resident and decode limits, round-robin
progress, cancellation, terminal-state retirement and per-request output
backpressure are implemented. Stop-prefix withholding prevents streaming tokens
that a later stop match must remove. Direct execution mode remains available.

The full-model scheduler test passes with delayed arrivals, prompt lengths
65/129/257/1024, exact full logits against direct execution, a slow consumer,
bounded queues, stop withholding, cancellation before/after admission and idle
backpressure. The existing engine model test also passes.

A finite burst of C16 requests, A16 residents, P512, 128 outputs, registered
511-token prefix, eight threads/mask 0xffff compares decode batch limit 1 with
16. Sequential A/B/B/A, six measured samples per path; all outputs match:

| Decode batch limit | Delivered token/s | TTFT p50 ms | ITL p50 ms | ITL p95 ms |
|---:|---:|---:|---:|---:|
| 1 | 119.20 | 202.83 | 133.40 | 134.58 |
| 16 | 593.20 | 236.44 | 23.98 | 28.86 |

This 4.98x delivered-throughput ratio includes queueing, admission, restore,
prefill/head work, decode, sampling and immediate output draining. Immutable
model loading and the separately reported prefix build are outside the interval.
This is a burst experiment, not constant concurrency or decode-only timing.
Both paths use the same scheduler and cache mode; only the decode batch limit
changes. Peak RSS medians are 958.30/963.84 MB. Wider arrival traces remain to
be evaluated. Automatic single-flight builds and shared KV pages are separate
subsequent changes.

[Samples](cpu-serving-2026-09-06-samples.csv) and
[settings/hashes](cpu-serving-2026-09-06-metadata.json).

## Selected batch/thread scale extension

P512, 128 outputs, three measured runs after one warmup per configuration,
sequentially. These are screening medians, not ABBA confirmation at every B.

| B | 8 threads, 96 MiB L3 token/s | 16 physical threads token/s | 16-thread tick p50 ms | 16-thread peak RSS GB |
|---:|---:|---:|---:|---:|
| 10 | not measured | 440.62 | 22.70 | 0.758 |
| 20 | not measured | 707.42 | 28.25 | 1.056 |
| 32 | 708.36 | 746.53 | 42.68 | 1.410 |
| 50 | 699.86 | 757.49 | 65.26 | 1.942 |
| 100 | 696.37 | 800.86 | 124.67 | 3.419 |

At B32, 24/32 threads over all logical CPUs give 713.57/714.37 token/s,
below 16 physical threads. At B100, 16 physical threads improve screening
throughput by 15.0% over eight threads, but increase aggregate throughput only
5.7% over B50 while nearly doubling tick latency. The B10 result also cautions
against using 16 threads universally: joint state dispatch requires B >= threads.
All 30 measured request arrays match their corresponding rows in the B100
reference, including changes in batch size and thread placement.

The original selected B10/20/50/100 extension is now measured at P512. Wider
contexts, arrival traces and platform calibration remain workload-specific
follow-ups, not evidence that 100 is the optimal serving batch size.

[Samples](cpu-batch-scale-2026-09-06-samples.csv) and
[settings/hashes](cpu-batch-scale-2026-09-06-metadata.json).
