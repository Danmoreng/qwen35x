# FlashQLA Current State - 2026-04-29

This note documents the current FlashQLA-style DeltaNet prefill work after adding the experimental BF16 tensor-core path for the local RTX 5080 Laptop GPU target.

## Reference

- Reference repository: `third_party/reference/FlashQLA`
- Upstream source: `git@github.com:QwenLM/FlashQLA.git`
- Integration mode: reference-only submodule; no reference code is linked into the build.

## Implemented paths

The FlashQLA experiments are all opt-in. The default prefill path is unchanged unless an environment flag is set.

| Flag | Path | Status |
|---|---|---|
| `QWEN35X_ENABLE_FLASHQLA_GDR_PREFILL=1` | scalar correctness scaffold | Useful for comparison only. |
| `QWEN35X_ENABLE_FLASHQLA_GDR_TILED_PREFILL=1` | first tiled recurrence prototype | Faster than scalar, still scalar-heavy. |
| `QWEN35X_ENABLE_FLASHQLA_GDR_CUDA_PREFILL=1` | direct reference-form CUDA implementation | Closest to the FlashQLA math, but very slow. |
| `QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL=1` | BF16 WMMA tensor-core tiled implementation | Current experimental fast path. |

The current tensor-core path is implemented in `src/kernels/cuda/prefill_flashqla.cu` as `pf_deltanet_recurrence_flashqla64_tc_tiled`.

Current properties:

- Uses BF16 WMMA tensor-core operations for the 64 by 64 chunk matrices `K * K^T` and `Q * K^T`.
- Does not use NVFP4 or the SM120-only MXF4/NVF4 assembly path.
- Builds for the local RTX 5080 Laptop GPU with `-CudaArchitectures 120a`.
- Processes value columns in `COLS=32` groups per block.
- Is still opt-in because parity is not clean yet.

## Local build command

Verified build command for the current machine:

```powershell
.\scripts\build.ps1 -UseNinja -EnableCuda -Configuration Release -Target qwen35x -CudaArchitectures 120a -BuildDir build-flashqla-gdr-verify
```

## Local benchmark snapshot

Benchmarks were run with the repository benchmark scripts, not ad-hoc commands. Comparable settings were `-Runs 3 -WarmupRuns 1 -MaxNewTokens 128 -MaxContext 256`.

| Implementation | Source CSV | Avg prefill | Avg prefill tok/s | DeltaNet total | DeltaNet recurrence | Minimal parity |
|---|---|---:|---:|---:|---:|---:|
| Current baseline SM120 prefill | `benchmarks/qwen35x-sm120-pp256-baseline-prefill-20260429.csv` | `25.425 ms` | `10133.62` | `12.059 ms` | `3.760 ms` | previously clean |
| FlashQLA CUDA reference-form path | `benchmarks/qwen35x-sm120-pp256-flashqla-cuda-ref-prefill-20260429.csv` | `2431.465 ms` | not competitive | not listed | `2411.326 ms` | `5/5` |
| Earlier FlashQLA tiled prototype | `benchmarks/qwen35x-sm120-pp256-flashqla-tiled-prefill-20260429.csv` | `206.873 ms` | not competitive | not listed | `183.015 ms` | experimental |
| FlashQLA BF16 WMMA TC, `COLS=16` | `benchmarks/qwen35x-sm120-pp256-flashqla-tc-cols16-prefill-20260429.csv` | `107.613 ms` | `2379.62` | `92.682 ms` | `83.785 ms` | `4/5` |
| FlashQLA BF16 WMMA TC, `COLS=32` | `benchmarks/qwen35x-sm120-pp256-flashqla-tc-cols32-prefill-20260429.csv` | `96.814 ms` | `2644.89` | `83.200 ms` | `74.162 ms` | `3/5` |

The `COLS=32` tensor-core path is much faster than the scalar reference-form implementation and the first tiled prototype, but it is still slower than the current baseline prefill path and has parity failures.

## What is still missing versus full FlashQLA

The current BF16 WMMA path only accelerates the chunk attention-like matrices. It does not yet implement the full producer/consumer kernel structure that gives FlashQLA most of its speed.

Main missing pieces:

- The triangular solve / delta computation is still scalar per value column.
- Output accumulation is still scalar over the chunk.
- State update is still scalar over key lanes and chunk rows.
- `K * K^T` and `Q * K^T` are recomputed for each value-column tile instead of being amortized more aggressively across the value dimension.
- The implementation does not yet use a large value tile comparable to the reference kernel's high-throughput layout.
- Shared-memory layout and scheduling are still simple and not tuned around sustained tensor-core occupancy.
- Parity needs to be fixed before the path can be considered for default use.

## Next work

1. Fix parity first, likely by comparing `COLS=16` and `COLS=32` against the clean CUDA reference-form path at intermediate tensors.
2. Move the lower-triangular delta solve toward a chunk-level implementation that is computed once and reused across value columns where possible.
3. Tensorize the value-side work: state/output updates should use BF16 MMA over larger value tiles instead of scalar per-column loops.
4. Rework shared-memory layout to keep chunk matrices and value tiles resident without recomputing them per value tile.
5. Rebenchmark on the RTX 5080 Laptop GPU with the standard sequential benchmark script after each substantial change.

