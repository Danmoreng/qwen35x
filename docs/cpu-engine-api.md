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

The legacy synchronous inference API remains available. This initial API does
not yet provide a background scheduler, HTTP service, shared prefix store or
shared KV pages. Those are subsequent stages in the
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
