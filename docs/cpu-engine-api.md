# Incremental CPU Engine

`qwen35x/runtime/cpu_engine.h` exposes immutable `CpuModel` weights and a
single-owner `CpuEngine`. It currently loads the CPU H128/Q4 artifact. Independent
engines can share a loaded model; each engine owns one executor and its scratch.
Methods and logits callbacks execute synchronously. Callbacks must not reenter
their engine, and their context must outlive the request.

```cpp
auto model = CpuModel::load(profile, load_options, error);
auto engine = CpuEngine::create(model,
    {.threads = 8, .max_resident_requests = 32, .max_context = 8192}, error);
CpuRequestSpec request;
request.prompt_tokens = prompt_token_ids;
request.max_new_tokens = 128;
request.sampling.temperature = 0;
auto id = engine->submit(std::move(request), error);
CpuRequestResult result;
do {
  if (!engine->advance(id, error)) { /* handle failure */ break; }
  engine->result(id, result);
} while (result.status == CpuRequestStatus::prefill ||
         result.status == CpuRequestStatus::decode);
engine->release(id);
```

Check load/create/submit results before use. `submit` returns zero on rejection.
Request IDs are scoped to their engine and never reused. Cancelled, failed and
completed requests retain their result until `release`; the initial admission
limit counts all stored requests. Context capacity includes the prompt plus
`max_new_tokens - 1` committed decode inputs. The final sampled output need not
be committed to KV/DeltaNet unless generation continues.

To batch independent requests, initialize each prompt with `advance` until its
status is `decode`. Pass distinct active IDs to `advance_batch`. It jointly
projects one row per request through every layer and the LM-head. Stateful
attention and recurrence remain private. Remove completed/cancelled IDs before
the next batch. An empty batch is a no-op; B=1 retains the single-request path.
Invalid batch IDs are rejected before state changes. A failed forward marks
affected requests failed because partial state updates cannot be retried.

When B is at least the executor's thread count, private attention/DeltaNet rows
also share one dispatch per layer. Each worker owns row scratch and executes
its state kernel directly, without nested executor calls. Smaller batches keep
per-request state dispatch. Set `parallel_state_batches = false` (or benchmark
`-SerialStateBatches`) to compare the earlier scheduling path.

The legacy synchronous inference API remains available. This initial API does
not yet provide a background scheduler, HTTP service or shared KV pages.
An experimental registered prefix store is described below. Further stages follow the
[implementation plan](cpu-multi-request-plan-2026-09-06.md).

## Reproducible static decode measurement

All-greedy batches now reduce shared vocabulary tiles directly, avoiding full
logit storage. Penalties and tie breaking remain request-local. General
sampling, forced-token tests and logits callbacks use the full-logits path.
Set `CpuEngineConfig::batch_greedy = false` or pass `-FullLogits` to the batch
benchmark wrapper to retain that reference explicitly. B=1 keeps its existing
fused greedy path regardless of this batch-only option.

Build `qwen35x_cpu_engine_bench` with `scripts/build.ps1 -BuildAll`. The benchmark
uses explicit token IDs, initializes prompts individually and starts the common
decode timer after every request has produced its first token. It counts actual
decode forwards, not all output tokens including those produced during prefill.
Sampling and the caller loop are inside the decode interval. Each request has
the same prefix with a deterministic distinct final prompt token; there is no
prefix reuse in this baseline.

```powershell
.\scripts\benchmark-cpu-batch.ps1 -BatchSizes 1,4,8 -PromptLengths 512 `
  -OutputDir benchmarks/my-static-sweep
.\scripts\benchmark-cpu-batch.ps1 -BatchSizes 4,8 -PromptLengths 512 `
  -SerialReference -OutputDir benchmarks/my-serial-reference
```

The wrapper delegates every run to `benchmark-inference-seq.ps1`, sequentially,
with CSV output. Defaults are eight threads, 0xffff affinity, three runs after
one warmup and 128 outputs. Do not overlap measurement with builds or tests.
Use a fresh output directory. The generic runner accepts `-CpuBatch` and
`-CpuBatchSerial` only with the dedicated executable; regular `qwen35x.exe`
does not implement these flags.

Profiles include actual forwards, aggregate throughput, throughput divided by
batch size, tick p50/p95/p99, peak process RSS, all output token IDs and counts
of shared projections/LM-head calls. Tick times describe this fixed static
workload; they are not arrival-to-first-token or continuous-serving latency.
Reported prefill time includes request allocation and first-token selection.
No stopping tokens are configured, so every request delivers the fixed count.

The optional `qwen35x_cpu_engine_model_test <HF directory> <Q4 artifact>` checks
full-model correctness using local model files. It is separate from CTest so
ordinary source builds do not require downloading a checkpoint.

## Experimental registered prefixes

Call `register_prefix(token_ids, trust_namespace, error)` to build a synchronous,
immutable hybrid snapshot. Set `CpuRequestSpec::prefix` to its handle, set the
same nonempty `trust_namespace`, and supply the complete prompt including those
exact prefix tokens. Restore copies KV, DeltaNet and convolution state into
private request storage. Complete hits use the cached final normalized hidden
vector, but compute the LM-head and sample independently for each request.

Compatibility requires the same loaded `CpuModel` object and prefill chunk
configuration. Handles retain that object, preventing address reuse. Reloading
even the same artifact creates a different identity and is rejected; this is an
in-process cache, with no persistent snapshot format or filename-based identity.
The default 256 MiB budget counts snapshot payload including DeltaNet, convolution,
KV, token IDs and final hidden data, not allocator overhead or request copies.
Externally held handles pin entries; only unpinned entries can be evicted.

The initial partial-hit discrepancy is fixed: short suffixes now retain tiled
attention beyond position 128, matching their cold-prefill arithmetic. All
outputs match in the repeated P512/B16 benchmark, and realistic full-logit tests
cover prefix boundaries through 4095 tokens. Prefix use remains explicit. See the
[progress report](cpu-multi-request-progress-2026-09-06.md) for measured scope.

`scripts/benchmark-cpu-prefix.ps1` compares cold/warm/warm/cold runs through the
sequential benchmark runner. `-FullHits` uses identical complete prompts;
otherwise each request has a distinct final token after the shared prefix.
Prefix construction is timed separately from warm request initialization.
Cached tokens are reported separately and never counted as newly computed
prefill throughput. The optional `qwen35x_cpu_prefix_model_test` takes the same
two model paths as the engine model test.

## Bounded serving mode

Set a nonzero max_queued_requests to enable the single-owner scheduler.
submit queues metadata without allocating KV or recurrent state; step admits
at most one request and performs at most one prefill chunk or decode batch.
max_resident_requests, max_decode_batch_size and max_queued_requests are
independent bounds. Prefill and decode alternate when both are runnable;
each class rotates request IDs so long prompts or small decode limits cannot
starve later active requests. No tick waits for a full batch.

Drive queued mode through step(progressed, error), not advance/advance_batch.
A successful idle tick returns progressed=false, for example when every resident
request is blocked on output. Read confirmed tokens with read_output and release
terminal result records when no longer needed. Completion and cancellation free
private model state; terminal results still count toward the bounded record
limit (resident limit plus queue limit) until release.

max_buffered_tokens bounds undrained generated tokens, including withheld partial
stop sequences. A full buffer pauses only that request. Stop sequences longer
than the buffer are rejected to prevent a holdback deadlock. read_output appends
only confirmed tokens; result.output_tokens is a diagnostic history and may
temporarily contain an incomplete stop prefix. Output history itself is bounded
by max_new_tokens/context. This API remains synchronous and has no HTTP server
or background owner thread.

The optional cpu_scheduler_model_test compares complete logits with direct
execution under delayed arrivals and mixed prompt lengths, exercises slow
consumers, stop-sequence withholding, queue limits, cancellation and slot reuse.

For serving measurements, pass -Serving to benchmark-cpu-batch.ps1 together
with -Residents, -DecodeBatchLimit and optionally -ArrivalGapMs. BatchSizes
then specifies total requests in a finite arrival trace, not constant
concurrency. Zero gap is a burst; nonzero gap prescribes arrival times. TTFT
includes lateness before the owner can submit, queueing and prompt work. ITL
uses delivered-token timestamps. Registered-prefix construction remains outside
the measured serving interval and is reported separately. The CSV field
tokens_per_second denotes aggregate delivered serving throughput when
cpu_serving=true; it must not be compared directly with decode-only throughput.
Separate prefill/decode wall-time fields are blank for this interleaved mode.

## Single-flight prefix construction

In serving mode, set request.register_prefix_tokens and a nonempty
trust_namespace to register a leading token sequence automatically. Do not also
supply an explicit prefix handle. The earliest admitted matching request builds
the prefix in bounded prefill ticks; followers remain queued without model
state. Once its complete hybrid snapshot is published, followers restore it
independently. Counts distinguish newly computed and restored tokens.

Cancellation or failure of an unpublished leader allows another queued request
to become the builder. If a snapshot cannot fit the cache budget or its
allocation fails, current followers fall back to independent cold execution.
No waiting request remains attached to a dead builder. The cache key uses exact
leading token IDs, namespace and the engine's immutable model/configuration.
This is explicit prefix-length registration, not automatic radix-tree matching.

Stats expose successful prefix_builds and prefix_build_fallbacks. In the serving
benchmark, -SingleFlightPrefixTokens enables this path; construction is inside
the common serving interval, unlike -RegisteredPrefixTokens, which prepares a
snapshot before the interval. Neither mode is enabled by default.
