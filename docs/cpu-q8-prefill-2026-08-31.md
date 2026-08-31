# CPU Q8_0 prefill status (2026-08-31)

The Qwen3.5 0.8B direct-GGUF CPU path now has a true layerwise batched prefill
implementation. It keeps token replay as `--qwen35x-prefill-mode replay`, while
`batched` uses runtime-dispatched portable scalar or AVX2/FMA/F16C kernels.

Implemented prefill optimizations:

- packed Q8_0 QKV and gate/up projections with batched matrix multiplication;
- precomputed activation and weight scales plus a spill-free AVX2 2x3 tile;
- head-major, head-parallel DeltaNet recurrence and channel-parallel causal convolution;
- token/head-parallel causal GQA with AVX2 QK/PV and vectorized softmax/gating;
- FP16 K/V read cache for AVX2 prefill while retaining the canonical F32 cache for decode;
- per-chunk RoPE tables shared by all Q and K heads.

Local test machine: Intel Core i7-8750H (6 cores / 12 threads, AVX2, no AVX-512),
Qwen3.5-0.8B-Q8_0.gguf, six engine threads. The sequential benchmark harness used
three measured runs after one warmup.

| Workload | qwen35x CPU Q8_0 | llama.cpp Q8_0 | Difference |
| --- | ---: | ---: | ---: |
| 256-token prefill | 172.84 tok/s | 163.36 tok/s | +5.8% |
| 2,048-token prefill | 150.03 tok/s | 138.34 tok/s | +8.5% |

The 2,048-token engine result corresponds to about 13.65 seconds of prefill,
down from roughly 54 seconds in the original token-replay path. Decode remains
a separate path and is not included in these prefill-only figures.
