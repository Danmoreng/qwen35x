# Benchmark Comparison (2026-04-22)

This document summarizes local benchmark results for Qwen3.5-0.8B on a single Windows CUDA machine.

## Environment

- GPU: NVIDIA GeForce RTX 5080 Laptop GPU (SM 12.0, 60 SMs)
- OS: Windows
- Model: `Qwen/Qwen3.5-0.8B` (BF16)

## Implementations compared

- `qwen35x` custom CUDA-hybrid path
- Luce megakernel benchmark harness (`qwen35x_lucebench`)
- `llama.cpp` BF16 CUDA
- `llama.cpp` BF16 CUDA with FlashAttention

## Workload settings

### Decode-oriented comparison

- `max_new_tokens=128`
- context budget around `256`
- deterministic sampling (`temperature=0`)

### Prefill-oriented comparison

- prompt length `256` tokens

Notes:

- `qwen35x` prefill is currently measured from the existing single-token forward loop (not yet a dedicated batched prefill kernel path).
- `llama.cpp` prefill numbers come from `llama-bench` `n_prompt=256, n_gen=0`.

## Results

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
3. `qwen35x` decode is near `llama.cpp` non-FA decode, but prefill is currently much lower due to lack of a true batched prefill kernel path.
4. Luce tuning is hardware-sensitive; occupancy-safe defaults are not necessarily throughput-optimal.
