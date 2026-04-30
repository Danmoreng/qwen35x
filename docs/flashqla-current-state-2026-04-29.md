# FlashQLA Current State - 2026-04-29

This note documents the current FlashQLA-style DeltaNet prefill work after adding the experimental BF16 tensor-core path for the local RTX 5080 Laptop GPU target.

Update 2026-04-30: the TC path now uses a split workspace structure. Chunk-level matrices are prepared once per DeltaNet head/value group/chunk and reused by value-column consumer blocks. This removes the earlier 4x recomputation of `KK`, `QK`, `A`, and `P` for the 0.8B layout.

Update 2026-04-30: the TC path now exposes separate profile timings for the FlashQLA prepare and consume stages. The combined `recurrence_ms` remains the sum of those stages for compatibility with existing reports.

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

The current tensor-core path is implemented in `src/kernels/cuda/prefill_flashqla.cu` as a two-stage path:

- `pf_deltanet_flashqla64_tc_prepare`: computes per-chunk `A`, `P`, `g`, and `beta` into an arena-backed workspace.
- `pf_deltanet_recurrence_flashqla64_tc_tiled`: consumes those matrices for `A @ W`, `P @ Vnew`, output, and state update over value-column tiles.

The launch layer now has explicit wrappers for both stages:

- `launch_pf_deltanet_flashqla64_tc_prepare`
- `launch_pf_deltanet_recurrence_flashqla64_tc_consume`
- `launch_pf_deltanet_recurrence_flashqla64_tc_tiled`, retained as a compatibility wrapper that calls prepare then consume.

Current properties:

- Uses BF16 WMMA tensor-core operations for the 64 by 64 chunk matrices `K * K^T` and `Q * K^T`.
- Does not use NVFP4 or the SM120-only MXF4/NVF4 assembly path.
- Builds for the local RTX 5080 Laptop GPU with `-CudaArchitectures 120a`.
- Processes value columns in `COLS=32` groups per block.
- Reuses chunk matrices across value tiles through `pf_flashqla_workspace`.
- Is still opt-in because it is not yet full FlashQLA-reference structure or faster than the old custom recurrence.

## Local build command

Verified build command for the current machine:

```powershell
.\scripts\build.ps1 -UseNinja -EnableCuda -Configuration Release -Target qwen35x -CudaArchitectures 120a -BuildDir build-flashqla-gdr-verify
```

## Iteration harness

The one-command FlashQLA iteration harness is:

```powershell
.\scripts\test-flashqla-iteration.ps1
```

Default behavior:

1. Builds `qwen35x` with CUDA/Ninja Release.
2. Sets `QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL=1`.
3. Runs minimal parity with `scripts/bench/parity_prompts_minimal.txt`.
4. Runs the 64k Wikipedia benchmark with profiling enabled.
5. Writes timestamped CSVs under `benchmarks/`.

Optional switches are available for local loops:

- `-SkipBuild`
- `-SkipParity`
- `-SkipBenchmark`

## Local benchmark snapshot

Benchmarks were run with the repository benchmark scripts, not ad-hoc commands. Comparable settings were `-Runs 3 -WarmupRuns 1 -MaxNewTokens 128 -MaxContext 256`.

| Implementation | Source CSV | Avg prefill | Avg prefill tok/s | DeltaNet total | DeltaNet recurrence | Minimal parity |
|---|---|---:|---:|---:|---:|---:|
| Current baseline SM120 prefill | `benchmarks/qwen35x-sm120-pp256-baseline-prefill-20260429.csv` | `25.425 ms` | `10133.62` | `12.059 ms` | `3.760 ms` | previously clean |
| FlashQLA CUDA reference-form path | `benchmarks/qwen35x-sm120-pp256-flashqla-cuda-ref-prefill-20260429.csv` | `2431.465 ms` | not competitive | not listed | `2411.326 ms` | `5/5` |
| Earlier FlashQLA tiled prototype | `benchmarks/qwen35x-sm120-pp256-flashqla-tiled-prefill-20260429.csv` | `206.873 ms` | not competitive | not listed | `183.015 ms` | experimental |
| FlashQLA BF16 WMMA TC, `COLS=16` | `benchmarks/qwen35x-sm120-pp256-flashqla-tc-cols16-prefill-20260429.csv` | `107.613 ms` | `2379.62` | `92.682 ms` | `83.785 ms` | `4/5` |
| FlashQLA BF16 WMMA TC, `COLS=32` | `benchmarks/qwen35x-sm120-pp256-flashqla-tc-cols32-prefill-20260429.csv` | `96.814 ms` | `2644.89` | `83.200 ms` | `74.162 ms` | `3/5` |

The earlier `COLS=32` tensor-core path was much faster than the scalar reference-form implementation and the first tiled prototype, but it was still slower than the current baseline prefill path and had parity failures.

After the split-workspace change, build and parity were verified with:

```powershell
.\scripts\build.ps1 -UseNinja -EnableCuda -Configuration Release -Target qwen35x
$env:QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL='1'
.\scripts\benchmark-parity.ps1 -Executable build\qwen35x.exe -PromptsFile scripts\bench\parity_prompts_minimal.txt -CsvOut benchmarks\qwen35x-flashqla-split-workspace-minimal-parity.csv -RunLabel flashqla-split-workspace-minimal -MaxNewTokens 4 -MaxContext 256 -GpuMode gpu-f32 -Qwen35xPrefillMode batched
.\scripts\benchmark-parity.ps1 -Executable build\qwen35x.exe -PromptsFile scripts\bench\parity_prompts.txt -CsvOut benchmarks\qwen35x-flashqla-split-workspace-extended-parity.csv -RunLabel flashqla-split-workspace-extended -MaxNewTokens 4 -MaxContext 256 -GpuMode gpu-f32 -Qwen35xPrefillMode batched
```

Parity results:

- Minimal parity: `5/5`
- Extended parity: `12/12`

64k Wikipedia benchmark settings:

```powershell
$env:QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL='1'
.\scripts\benchmark-inference-seq.ps1 -Executable build\qwen35x.exe -CsvOut benchmarks\qwen35x-flashqla-split-workspace-wiki-ai-64k-gen128.csv -RunLabel flashqla-split-workspace-wiki-ai-64k-gen128 -Modes gpu-f32 -PromptMode prompt-file -PromptFile benchmarks\inputs\wiki_artificial_intelligence_64k_prompt.txt -PromptName wiki_ai_64k_gen128 -Runs 3 -WarmupRuns 1 -MaxNewTokens 128 -MaxContext 65536 -Qwen35xPrefillMode batched
```

| Implementation | Source CSV | Avg prefill | Avg prefill tok/s | Avg decode | Avg total tok/s |
|---|---|---:|---:|---:|---:|
| Previous FlashQLA TC path | `benchmarks/qwen35x-current-flashqla-wiki-ai-64k-gen128.csv` | `27731.82 ms` | `2358.73` | `735.52 ms` | `174.03` |
| Split-workspace FlashQLA TC path | `benchmarks/qwen35x-flashqla-split-workspace-wiki-ai-64k-gen128.csv` | `13073.39 ms` | `5004.03` | `645.76 ms` | `198.40` |
| Split workspace + row-parallel KKT solve | `benchmarks/qwen35x-flashqla-parallel-kkt-wiki-ai-64k-gen128.csv` | `10910.53 ms` | `5994.76` | `650.20 ms` | `196.87` |
| Old custom recurrence path | `benchmarks/qwen35x-current-old-custom-wiki-ai-64k-gen128.csv` | `9512.08 ms` | `6885.11` | `707.23 ms` | `180.99` |
| Profile-split FlashQLA TC path | `benchmarks/qwen35x-flashqla-iteration-wiki-ai-64k-gen128-20260430-145601.csv` | `10875.03 ms` | `6014.43` | `656.53 ms` | `194.97` |

The split-workspace change recovered the largest obvious duplicate-work penalty: about 2.1x faster prefill than the previous FlashQLA TC path on 64k. The row-parallel KKT solve then reduced prefill from `13073.39 ms` to `10910.53 ms`. It is still slower than the old custom recurrence by about 15 percent on prefill latency.

The profile-split run keeps the same algorithmic structure as the row-parallel KKT path, but records the two FlashQLA sub-stages separately:

| Metric | Avg over 3 measured 64k runs |
|---|---:|
| DeltaNet recurrence total | `4313.68 ms` |
| FlashQLA prepare | `360.63 ms` |
| FlashQLA consume | `3953.05 ms` |

The consumer stage is therefore the next high-value optimization target.

## What is still missing versus full FlashQLA

The current BF16 WMMA path only accelerates the chunk attention-like matrices. It does not yet implement the full producer/consumer kernel structure that gives FlashQLA most of its speed.

Main missing pieces:

- The triangular KKT solve for `A` is row-parallel, but still row-synchronous and not equivalent to the reference's optimized solve schedule.
- The path uses a two-kernel global-workspace split instead of the reference's tighter producer/consumer shared-memory schedule.
- Output accumulation is tensorized through `P @ Vnew`, but final output/state handling still uses scalar warp reductions.
- State update is still scalar over key lanes and chunk rows.
- Scalar `expf`/`logf` work remains in prepare and consumer loops.
- Shared-memory layout and scheduling are still simple and not tuned around sustained tensor-core occupancy.
- Parity against this repository's CPU reference is clean for the current test set, but direct parity against the upstream Qwen FlashQLA reference is still not implemented.

## Next work

1. Improve the KKT solve schedule further; the current row-synchronous version is correct and faster, but still conservative.
2. Reduce consumer-side scalar work without regressing prefill. A tested `exp(g)` workspace variant preserved parity but was slower on 64k, so it was not kept.
3. Move more value-side work into tensor-core-friendly tiles, especially state/output update surfaces.
4. Rework shared-memory layout and scheduling toward a producer/consumer structure that avoids global workspace traffic where practical on `sm_120a`.
5. Add direct comparison tests against the upstream Qwen FlashQLA reference outputs.
6. Rebenchmark on the RTX 5080 Laptop GPU with `scripts/test-flashqla-iteration.ps1` after each substantial change.
