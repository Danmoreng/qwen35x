# CPU Multi-Request Implementation Order on Ryzen

Planning date: September 6, 2026, repository revision `3cc0e07`. This plan is
based on the supplied `Qwen35x_Multi_Request_Codex_Plan_DE.md`, checked against
the current code. It describes proposed implementation work, not an implemented
feature or new performance results.

The goal is higher aggregate throughput for independent requests with reusable
system-prompt prefixes. Keep H128/Q4-G32-DOT4 weights, FP16 KV, and FP32 DeltaNet
state. Initially, implement and measure on the Ryzen machine only.

## Current Starting Point

- `CpuModelSessionState` owns weights and serializes complete inference calls.
  `ModelWeights::cpu_q8_runtime` and `TensorData::q8_0_runtime` still couple
  weights to mutable workspaces, the executor, and diagnostic outputs.
- `ModelState` contains full-attention KV, convolution, and DeltaNet state.
  Position, sampling, and generation control also live within the complete API
  call.
- `run_forward_cpu_q8_batch` processes consecutive tokens of **one** sequence.
  The LM-head uses only its last hidden row. This is not a multi-request API.
- The prefix snapshot already copies the complete hybrid state. The current
  contract requires at least one uncached prompt token; the snapshot does not
  contain the final normalized hidden vector.
- Tiled attention and decode diagnostics are available. However, the public
  tiled interface still consumes contiguous KV for one sequence and returns a
  finalized output. Segment access and exported partial softmax states are
  not yet supported.

## 1. Ownership and Incremental Single-Sequence Execution

Separate immutable weights from an explicit execution context. One engine owner
manages the shared executor and scratch space. Pass this context explicitly to
projections; do not switch runtime pointers in weights between requests.

Persistent request state includes KV, DeltaNet, the convolution ring, absolute
position, prompt cursor, committed tokens, a pending output token, and private
RNG, penalty, and stopping state. Preserve the existing synchronous call as a
compatible wrapper. Initially, use private contiguous KV buffers with an
explicit limit on resident requests.

Acceptance: pausing and resuming matches the existing single call; A and B can
advance alternately without affecting each other. No single-request correctness
or measured performance regression. Do not assume this structural change alone
improves throughput.

## 2. Static Multi-Request Decode, Including the LM-Head

Initially, initialize prompts individually using the existing fast prefill.
Then jointly project one independent input row per active request, layer by
layer, on each tick. Attention, RoPE, convolution, and DeltaNet use each request's
own state and position. Padding must never modify state.

Track output rows and their request mapping explicitly. Batch the LM-head from
the start, initially producing full logits as a verifiable reference. Then test
request-local greedy argmax over shared vocabulary tiles as a separate
optimization; retain full logits for general sampling. A fast single-request
path may remain available for B=1. Matching the first generated tokens alone
does not establish quality preservation.

Initial measurements: B=1/2/4/8/16/32, P512/P4096, and 128 output tokens. Also
check isolation and tails at B=3/5/17. Then selectively extend the sweep to 100,
including 10/20/50 and additional boundaries around multiples of 4 and 16 from
the Pro plan. Expand context and output lengths after this initial validation.

Acceptance: compare A alone with A in a batch, reorder the batch, advance or
remove other sequences independently, and test different positions and
convolution-ring boundaries. Require bitwise equality when arithmetic order is
unchanged; otherwise evaluate logit/state errors and model quality separately.
Instrumentation must demonstrate that batching actually uses shared projections.

This is the first decisive performance milestone: more tokens actually produced
per shared decode wall-clock interval, with tick duration and throughput per
request explicitly reported.

## 3. Registered Hybrid Prefixes with Copy-Based Restore

Move this stage earlier than in the Pro plan so that a shared system prompt can
provide a separately measurable benefit sooner. Start with explicitly registered
immutable prefixes and a simple bounded store, without a radix tree.

An entry contains exact token IDs, a common committed position, all full- and
linear-attention states, and the final normalized hidden vector. The latter
supports a complete prompt hit without processing the last token again. Logits
and sampling remain request-local; do not reuse a cached sample.

Bind identity to the loaded immutable model and a compatible execution
configuration, including RoPE, precision, and state ABI. Do not derive model
identity solely from path, size, and mtime; perform necessary content
identification once at load time, not in the decode hot path. Include a trust
namespace in the store contract. Budgets must also count DeltaNet snapshots.

Acceptance: cold and warm restores with different suffixes, complete hits, and
invalid identities. A later DeltaNet state must not be truncated to an earlier
token position. Initially, this stage saves prefill computation while still
creating private KV copies.

## 4. Continuous Batching and Single-Flight Prefix Builds

Extend the validated runner with `submit`, `cancel`, and bounded `step` ticks.
Limit queue size, resident sequences, decode batch size, and prefill work
separately. Queued requests do not yet receive full model state. Initially,
alternate decode ticks with bounded prefill ticks, guaranteeing progress for
prefill. Do not wait for a full batch under light load.

This stage includes request admission, completion, cancellation, slot
generations, overload handling, and bounded output buffers. Compute workers see
stable state throughout each tick; slow output consumers must not block the
entire executor.

Integrate single-flight now: build identical prefixes arriving concurrently
once, then let waiting requests continue independently. Leader cancellation and
errors must resolve or reschedule followers.

Acceptance: varied arrival times, short and long prompts, early completion,
cancellation, full buffers, and fairness. Measure C=1/5/10/20/50/100 open requests
with separate limits on resident sequences A and decode batch size B.
Distinguish constant concurrency from prescribed arrival rates and bursts.
HTTP is not a prerequisite.

## 5. Physically Shared KV Prefixes

First extend the attention interface to consume contiguous KV segments and
partial softmax accumulators, and validate against contiguous KV. Then share
references to immutable prefix pages and write only to private suffixes.
Partially filled tail pages require an explicit copy-on-write contract. Do not
copy the entire prefix on each decode tick or perform per-scalar hash lookups.

Copy DeltaNet and convolution state into private storage once at the branch
point. Budget KV page size and the spacing of complete hybrid snapshots
separately. Active handles pin data; eviction must not recycle it.

Acceptance: branch isolation, tail copy-on-write, page boundaries, and eviction
with active requests. Compare against copy-based restore using identical
requests, reporting physical memory, restore time, and inference latency
separately. Memory savings alone do not demonstrate faster decode.

## 6. Shared Prefix Attention

Compute queries from multiple requests against the same prefix panels. Process
private suffixes separately; stably merge partial maxima, denominators, and
unnormalized outputs. Apply the output gate exactly once.

Acceptance: empty segments, different suffix lengths, causality, and numerical
model checks. Measure gains against the same shared-KV baseline without this
query grouping. The benefit on this CPU remains unproven.

## 7. Mixed Prefill/Decode Projections

As a separate final experiment, jointly project decode rows and segmented
prefills. Compare against the interleaving scheduler from step 4 using the same
prefix mode and workload traces. Preserve attention and recurrence boundaries
and fairness. Adopt only if benefits are demonstrated.

## Common Measurement and Adoption Requirements

- Continue running all performance measurements through
  `scripts/benchmark-inference-seq.ps1`; extend it for the new in-process
  benchmark modes. Sequential execution and CSV output remain mandatory.
  Do not run benchmarks alongside builds or tests.
- Initially retain the same eight threads and 0xffff affinity as before.
  Explore thread counts and topology only after batching works correctly.
- Static decode timing starts after prompt initialization. Serving time to
  first token (TTFT) starts at arrival and includes queueing, restore, and
  prefill.
- Report aggregate delivered tokens per shared wall-clock interval, actual
  decode forwards, newly computed prefill tokens, and cache-hit tokens
  separately. Do not add individual tokens/s values or count cached tokens as
  newly computed prefill. Also report inter-token latency (ITL) p50/p95/p99,
  queue time, and peak RSS.
- Start with a small informative sweep, then expand promising candidates.
  Do not run the full Cartesian product of all parameters upfront. Report
  latency and aggregate throughput together; do not prescribe a universal
  optimal batch size yet.
- Run single-request regression and isolation tests after every stage, with
  traceable commit and measurement boundaries. Preserve existing fast paths
  as references.

The first implementation block covers steps 1 and 2 with separate intermediate
results. System-prompt restore and serving build on that foundation.
