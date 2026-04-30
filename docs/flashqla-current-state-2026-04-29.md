# FlashQLA Current State - 2026-04-29

This note documents the current FlashQLA-style DeltaNet prefill work after adding the experimental BF16 tensor-core path for the local RTX 5080 Laptop GPU target.

Update 2026-04-30: the TC path now uses a split workspace structure. Chunk-level matrices are prepared once per DeltaNet head/value group/chunk and reused by value-column consumer blocks. This removes the earlier 4x recomputation of `KK`, `QK`, `A`, and `P` for the 0.8B layout.

Update 2026-04-30: the TC path now exposes separate profile timings for the FlashQLA prepare and consume stages. The combined `recurrence_ms` remains the sum of those stages for compatibility with existing reports.

Update 2026-04-30: the consumer now reuses the chunk scalar scratch for `exp(g[t])` after beta is dead, and skips per-element bounds work on full 64-token, in-bounds value tiles. A larger shared `Q/K` staging attempt was tested and reverted because minimal parity failed `0/5`.

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

For tighter correctness loops during consumer refactors, use the first-token prefill parity harness:

```powershell
.\scripts\test-flashqla-prefill-parity.ps1
```

This script builds `qwen35x`, sets `QWEN35X_ENABLE_FLASHQLA_GDR_TC_PREFILL=1`, and runs the minimal CPU/GPU parity prompts with `-MaxNewTokens 1`. It catches prefill/logit/state regressions at the first generated token before running the full 64k performance harness.

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
| Consumer scratch/fast-path cleanup | `benchmarks/qwen35x-flashqla-iteration-wiki-ai-64k-gen128-20260430-152345.csv` | `10675.06 ms` | `6126.97` | `654.36 ms` | `195.62` |
| Rejected 32-lane K-slice staging | `benchmarks/qwen35x-flashqla-iteration-wiki-ai-64k-gen128-20260430-160746.csv` | `10685.69 ms` | `6120.82` | not recorded here | not recorded here |

The split-workspace change recovered the largest obvious duplicate-work penalty: about 2.1x faster prefill than the previous FlashQLA TC path on 64k. The row-parallel KKT solve then reduced prefill from `13073.39 ms` to `10910.53 ms`. It is still slower than the old custom recurrence by about 15 percent on prefill latency.

The profile-split run keeps the same algorithmic structure as the row-parallel KKT path, but records the two FlashQLA sub-stages separately:

| Metric | Avg over 3 measured 64k runs |
|---|---:|
| DeltaNet recurrence total | `4313.68 ms` |
| FlashQLA prepare | `360.63 ms` |
| FlashQLA consume | `3953.05 ms` |

The consumer stage is therefore the next high-value optimization target.

After the consumer scratch/fast-path cleanup, the 64k profile split was:

| Metric | Avg over 3 measured 64k runs |
|---|---:|
| DeltaNet recurrence total | `4012.27 ms` |
| FlashQLA prepare | `358.31 ms` |
| FlashQLA consume | `3653.96 ms` |

The first attempt to stage `Q` and `K` directly in the consumer used extra dynamic shared memory after the `w_bf16` tile. It built, but failed minimal parity for all 5 prompts, so it was reverted. The next producer/consumer refactor should be developed against a narrower prefill parity harness before rerunning full 64k performance.

A smaller RTX 5080-oriented staging attempt copied one 32-lane `K` slice at a time for the final state update while reusing the existing dynamic shared-memory allocation. It preserved minimal parity (`5/5`), but regressed the 64k profile split to `4049.93 ms` recurrence total, `353.65 ms` prepare, and `3696.28 ms` consume. The added staging loops and barriers cost more than the reduced global `K` reloads on this kernel shape, so the code change was reverted.

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

The next session should focus on structural changes, not more small shared-memory caching inside the current monolithic consumer. The rejected K-slice staging result suggests that this kernel is limited by synchronization/register/shared-memory scheduling pressure more than by a single obvious global `K` load.

Recommended order:

1. Add a narrow state-update parity harness. This is now available as `scripts/test-flashqla-state-update-parity.ps1`, which builds `qwen35x_flashqla_state_update_parity` and checks synthetic `state += K^T @ Vnew` cases for chunk row counts `1`, `17`, `63`, and `64`.
2. Tensorize the state update in the isolated harness first. The BF16 WMMA candidate now matches a BF16 scalar reference with max error around `1.5e-8`; compared with the current FP32 scalar production update, the synthetic per-chunk drift is around `5.0e-5`.
3. Do not directly drop the BF16 WMMA state update into the current monolithic consumer. A direct gated production attempt passed first-token prefill parity, but failed 4-token minimal generation parity because the recurrent state drift affected decode. The production kernel change was reverted.
4. If shared memory blocks a parity-preserving tensorized state update, split the consumer by responsibility: one kernel for tensor-core `P @ Vnew` output and one kernel for final state update. This may add global traffic, but it can reduce barrier pressure and make each kernel easier to schedule on `sm_120a`.
5. Retest tile shapes only after the state-update split/tensorization exists. Worth trying first: `CHUNK=32, COLS=32` and `CHUNK=64, COLS=16`, using the iteration harness for 64k comparisons.
6. Revisit scalar `expf`/`logf` reduction only after the consumer structure changes. The earlier precomputed `exp(g)` workspace variant preserved parity but was slower in the current structure.
7. Add direct comparison tests against upstream Qwen FlashQLA outputs once the local algorithmic surfaces match: KKT solve, output accumulation, state update, and gating/decay handling.
8. Rebenchmark on the RTX 5080 Laptop GPU with `scripts/test-flashqla-iteration.ps1` after each substantial change. Keep using `scripts/benchmark-inference-seq.ps1` through the harness for comparable performance tracking.
