# CPU Server Release Scope

The user requested a bounded finish to the performance experiment series on
September 6. The target is a usable Qwen3.5-0.8B inference server with measured
single- and multi-request behavior and documented x86-64 compatibility.

## Freeze the validated engine foundation

Keep H128/Q4-DOT4 weights, FP16 KV, private FP32 recurrent state, incremental
requests, continuous batching and hybrid prefix reuse. Finish the ongoing
shared-KV-page comparison and record its numerical, memory and timing results.
Keep explicit reference paths and configuration switches. Defer grouped shared
prefix attention and mixed prefill/decode projections (original steps 6 and 7)
until after the release baseline; their performance benefit is not established.

## Connect a usable server

Add an HTTP entry point over the single-owner scheduler, with model loading
once at startup, native tokenization/chat formatting, streamed output, request
cancellation on disconnect and bounded submission/output queues. Expose model
readiness and configuration. Test concurrent requests and slow clients through
the actual interface, not just the in-process benchmark. The current CpuEngine
API is a serving foundation, not yet an HTTP inference server.

## Establish a matched llama.cpp comparison

Use current, recorded engine revisions and Q4 artifacts derived from the same
BF16 source. The two Q4 formats differ; performance comparisons do not establish
a quality ranking. Preserve fixed token inputs and comparable generation rules,
KV precision, CPU allocation and output counts. Compare both single requests
and selected concurrent loads, with cold/unrelated prompts and shared prefixes
reported separately. Include delivered throughput, TTFT, ITL and peak memory.
Use the sequential benchmark runner and retain raw samples and model hashes.

The current 17-22% historical single-request decode advantage cannot be applied
to the new multi-request results. Report the measured ratio at each workload,
including any cases where llama.cpp performs better.

## Validate the supported CPU/platform matrix

Verify the current integrated path on this Ryzen and the available Intel
i7-8750H laptop, including Linux and Windows builds as available. Exercise
scalar/AVX2 and newer ISA paths; forcing AVX2 on a newer CPU is useful but does
not replace testing on actual older hardware. Confirm portable compiler
baselines and CPU/OS feature gating, and document minimum supported x86-64
requirements instead of claiming all x86 CPUs without evidence.

Retain explicit thread overrides and measured guidance. Machine-specific
affinity masks belong only to benchmark scripts. Portable topology-aware
defaults are useful release work; optional saved calibration can follow later.

## Completion criteria

- A documented startup command serves real concurrent streaming requests.
- Isolation, stopping, cancellation and bounded resource behavior pass tests.
- The README contains a reproducible matched single/multi-request comparison.
- Supported operating systems/CPU levels are stated with actual validation.
- Completed changes and results are committed; deferred experiments are listed.
