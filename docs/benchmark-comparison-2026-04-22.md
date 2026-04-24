# Benchmark Comparison

This document summarizes local benchmark results for Qwen3.5-0.8B on a single Windows CUDA machine. It started as the April 22 comparison and now includes the April 24 follow-up after batched Luce prefill became the default qwen35x prompt-processing path.

## Environment

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU (SM 12.0, 60 SMs)
- OS: Windows
- Model: `Qwen/Qwen3.5-0.8B` (BF16)

## Implementations compared

- `qwen35x` custom CUDA-hybrid path
- `qwen35x` default integrated Luce backend
- Luce megakernel benchmark harness (`qwen35x_lucebench`)
- `llama.cpp` BF16 CUDA
- `llama.cpp` BF16 CUDA with FlashAttention

## 2026-04-24 Follow-up: current default Luce path

Current `qwen35x` uses batched Luce prefill by default and warms the prefill backend during initialization. Timed prefill and decode measurements therefore exclude one-time CUDA/cuBLAS setup; `load_time` includes the warmup.

Source `qwen35x` CSVs:

- `benchmarks/qwen35x-pp256-prefill-only-current-rerun.csv`
- `benchmarks/qwen35x-tg-prompt1-current-rerun.csv`
- `benchmarks/qwen35x-full-pp256-gen128-current-ctx384.csv`

Saved `llama.cpp` artifacts reused from the earlier local comparison:

- `benchmarks/llama-bench/qwen3.5-0.8b-bf16-pp256-current.json`
- `benchmarks/llama-bench/qwen3.5-0.8b-bf16-pp256-fa-current.json`
- `benchmarks/llama-bench/qwen3.5-0.8b-bf16-tg-current.json`
- `benchmarks/llama-bench/qwen3.5-0.8b-bf16-tg-fa-current.json`

| Metric | qwen35x | llama.cpp | llama.cpp + FA | qwen35x vs llama.cpp | qwen35x vs llama.cpp + FA |
|---|---:|---:|---:|---:|---:|
| `pp256` prefill tok/s | `19,739.13` | `12,597.12` | `13,681.26` | `1.57x` | `1.44x` |
| `prompt1/gen128` generation tok/s | `300.46` | `140.15` | `142.59` | `2.14x` | `2.11x` |

End-to-end `qwen35x` `pp256/gen128` with `MaxContext=384` measured `18,915.00 tok/s` prefill and `302.38 tok/s` generation. The same `pp256/gen128` shape cannot use `MaxContext=256` in `qwen35x` because the prompt length plus requested new tokens exceeds the context guard, so the direct comparison uses split prefill-only and generation-only runs plus the `MaxContext=384` end-to-end sanity run.

Current takeaways:

1. The current default `qwen35x` Luce path is ahead of the saved `llama.cpp` BF16 CUDA artifacts for both `pp256` prefill and `prompt1/gen128` generation on this machine.
2. The backend warmup improves measurement hygiene; remaining prefill work should reduce actual kernel work, launch count, and recurrence overhead.
3. The April 22 results below are historical and describe the pre-batched-prefill custom path.

## Workload settings

### Decode-oriented comparison

- `max_new_tokens=128`
- context budget around `256`
- deterministic sampling (`temperature=0`)

### Prefill-oriented comparison

- prompt length `256` tokens

Notes:

- Historical April 22 `qwen35x` prefill was measured from the previous single-token forward loop, before the default batched Luce prefill path.
- `llama.cpp` prefill numbers come from `llama-bench` `n_prompt=256, n_gen=0`.

## 2026-04-22 Historical Results

Source CSV:

- `benchmarks/benchmark-compare-luce-our-llama.csv`

| Implementation | Decode tok/s | Prefill tok/s | Notes |
|---|---:|---:|---|
| Luce megakernel (CUDA) | 267.45 | 8358.51 | decode prompt=1/gen=128; prefill prompt=256 |
| qwen35x custom gpu-bf16 | 199.17 | 184.50 | decode prompt=1/gen=128; prefill prompt=256/gen=1 |
| llama.cpp bf16 cuda | 199.74 | 13761.77 | `flash_attn=false`, prompt=256, gen=128 |
| llama.cpp bf16 cuda + FA | 222.62 | 14372.79 | `flash_attn=true`, prompt=256, gen=128 |

### FlashAttention effect in llama.cpp

- Decode: `199.74 -> 222.62 tok/s` (`+11.5%`)
- Prefill: `13761.77 -> 14372.79 tok/s` (`+4.4%`)

## Luce kernel tuning summary

Source CSV:

- `benchmarks/luce-tuning-summary.csv`

2D tuning (`BLOCK_SIZE x decode_blocks`) selected:

- `BLOCK_SIZE=256`
- `decode_blocks=52`

Best means by block size:

| BLOCK_SIZE | Best decode_blocks | Mean decode tok/s |
|---:|---:|---:|
| 256 | 52 | 272.06 |
| 384 | 24 | 240.98 |
| 512 | 48 | 236.11 |

Additional revalidation CSVs:

- `benchmarks/luce-revalidation-bs256-db52.csv`
- `benchmarks/luce-revalidation-bs512-db32.csv`

Head-to-head revalidation showed `256/52` outperforming `512/32` by about `11.75%` decode throughput.

## Key takeaways

1. On this machine, Luce decode path is fastest in this comparison set.
2. `llama.cpp` gains substantially from FlashAttention for decode.
3. Historical `qwen35x` decode was near `llama.cpp` non-FA decode, but historical prefill was much lower before the true batched Luce prefill path.
4. Luce tuning is hardware-sensitive; occupancy-safe defaults are not necessarily throughput-optimal.
