# qwen35x Development Plan

This plan summarizes the original project direction and turns it into a public, implementation-focused roadmap.

## Status Snapshot (April 2026)

Completed:
- CPU reference inference pipeline for Qwen3.5-0.8B
- CUDA inference path with full-attention decode kernel + linear-attention decode kernel
- Device-resident per-layer decode flow (hidden/residual/norm/attention/MLP on GPU)
- GPU logits + GPU sampling path
- Device-token GPU decode loop path (sampled token consumed on device for next-step embedding gather)
- CUDA Graph replay for steady-state MLP decode work
- Decode profiling (`--profile-json`) with stage timing and transfer breakdown

Current known constraints:
- GPU sampling path currently requires `top_k <= 64` when `temperature > 0`
- Stop condition checks remain host-side when stop tokens/sequences are configured
- Prefill path is still token-by-token and not yet batched/specialized

Latest local benchmark snapshot (Qwen3.5-0.8B, same machine):
- Deterministic prompt path: ~127 tokens/s
- Chat prompt path: ~91 tokens/s

## Vision

Build a standalone, hardware-aware inference engine specialized for the Qwen3.5 architecture family, starting with `Qwen3.5-0.8B` and scaling to larger variants later.

Core strategy:
- Keep the stack small and explicit
- Prioritize architecture-specific fast paths over generic abstractions
- Use reference implementations for correctness, then replace hotspots with specialized kernels

## Target Profile

- Initial model: `Qwen/Qwen3.5-0.8B`
- Initial platform: NVIDIA GPU class used in this project environment
- Initial precision path: dense reference + CUDA GPU decode path
- Future target: larger Qwen3.5 variants with the same engine architecture

## Design Principles

- Use a compiler/runtime split:
- Compiler: parse model metadata, build static execution plan, prepare packed tensors
- Runtime: execute prefill/decode and manage caches, dispatch, and memory
- Keep correctness and speed work separate:
- CPU reference path is the oracle
- GPU path must match token-by-token before heavy optimization
- Optimize measured bottlenecks first:
- Decode path and cache updates before broad refactors
- Build for extensibility:
- The code should be easy to retarget to other model sizes and GPUs

## Architecture Milestones

1. Baseline Bring-up
- Load HF model config and safetensors
- Implement tokenizer encode/decode
- End-to-end CPU reference inference

2. Hybrid Attention Correctness
- Support Qwen3.5 mixed layer schedule (`linear_attention` + `full_attention`)
- Maintain both full-attention KV cache and linear recurrent/conv states
- Validate stable text generation for simple prompts

3. CUDA-Hybrid Acceleration
- Add `--infer-gpu` path
- Offload matrix-heavy operations to CUDA
- Keep attention math in reference form until parity is stable

4. Generation Controls and Usability
- Add chat prompt wrapper mode
- Add default sampling and override flags
- Add deterministic mode with seed control
- Add stop-token and stop-text handling

5. Kernel Specialization Phase
- Replace CUDA stubs with real decode kernels
- Prioritize full-attention GQA decode kernel
- Add linear-attention recurrent update kernels
- Introduce packed weight layout for target GPU class

6. Prefill and Scheduling
- Optimize prefill separately from decode
- Add persistent runtime buffers
- Add profiling-driven dispatch and graph execution improvements

7. Low-Precision Phase
- Add one clear low-precision path first (instead of many formats at once)
- Keep fallback and correctness tests for every precision mode

Milestone progress:
- Milestones 1-4: completed
- Milestone 5: in progress (decode kernels implemented, packed projections added, CUDA Graph MLP replay added)
- Milestones 6-7: in progress

## Validation and Benchmarking Plan

- Correctness tests:
- CPU vs GPU token-level parity for fixed prompts
- Deterministic sampling with known seeds
- Benchmark suite:
- Decode tokens/sec
- Prefill throughput
- Load time and memory footprint
- Regression gates:
- Keep simple prompts and expected token outputs as smoke checks

## Practical Next Steps

1. Extend CUDA Graph usage beyond MLP replay to larger steady-state decode segments
2. Optimize GPU sampling further (lift current `top_k <= 64` limit while preserving throughput)
3. Add dedicated prefill implementation (batched/streaming prefill separate from decode)
4. Add parity and performance harness scripts for repeatable CPU/GPU checks
5. Add decode-critical weight/layout packing and kernel-tuned memory layouts
