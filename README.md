# qwen35x

`qwen35x` is a vibe-coded inference engine project focused on **Qwen3.5**.

The goal is not a generic multi-model runtime. The goal is a small, hardware-aware engine that can be aggressively optimized for a specific Qwen3.5 architecture and target GPU class.

## What This Repo Is

- A C++/CUDA codebase for Qwen3.5-focused inference experiments
- A hybrid reference runtime for correctness and architecture bring-up
- A playground for moving from reference CPU logic to specialized GPU kernels

## Current Capabilities

- Load Qwen3.5 profile/config from Hugging Face model folders
- Native tokenizer (`vocab.json` + `merges.txt` + added tokens)
- Reference generation via `--infer-reference` (CPU) and `--infer-gpu` (CUDA)
- `--infer-gpu` defaults to the in-tree Qwen35x CUDA backend and auto-selects the compiled 0.8B or 4B CUDA layout from the loaded model profile
- Legacy CUDA runtime decode backend remains available with `--gpu-decode-backend default`
- Batched Qwen35x prefill is the default prompt-processing path and warms the prefill backend during initialization
- Device-resident decode path for per-layer hidden/residual/norm/attention/MLP math in `--infer-gpu`
- GPU logits + GPU sampling path in the legacy runtime decode backend
- Device-token decode loop path in the legacy runtime decode backend (sampled token stays on GPU for next-step embedding gather)
- CUDA Graph replay for steady-state decode segments (MLP + linear-attention blocks)
- BF16 decode matvec path for the legacy CUDA runtime decode backend
- Packed full-attention projection path (`q+gate+k+v`) to reduce decode matvec launches in full-attention blocks
- Streaming full-attention decode kernel with online softmax accumulation (single pass over sequence)
- Qwen-style default sampling (`temperature=0.7`, `top_p=0.8`, `top_k=20`, `repeat_penalty=1.05`)
- Deterministic mode with `--seed` and `--temperature 0`
- Stop controls: `--stop-token`, `--stop-text`, `--stop-on-im-end`
- BF16 tensor benchmark path (`--bench-bf16`)
- Profiling output via `--profile-json` (stage timings + H2D/D2H transfer stats)
- Optional synchronized CUDA stage timing via `--profile-sync`
- Optional Qwen35x phase profiling via `--qwen35x-profile`, including full-attention prefill QK/softmax/PV/gate timing
- Optional PyTorch/Transformers parity oracle for checking the local CPU reference against an external implementation

Current GPU sampling constraint:
- The default Qwen35x CUDA backend currently supports greedy decode only (`--temperature 0`) and applies repetition penalty during argmax.
- For `temperature > 0`, use `--gpu-decode-backend default`; that GPU sampling path currently supports `top_k` in `[1, 64]`.

Current Qwen35x prefill behavior:
- `--qwen35x-prefill-mode batched` is the default path.
- `--qwen35x-prefill-mode replay` remains available as a conservative fallback.
- The default backend performs a one-token prefill warmup during initialization, then resets recurrent/cache state before real inference. This keeps one-time cuBLAS/kernel setup outside timed prefill and decode paths.
- Prefill scratch is chunked for MLP and DeltaNet work so 4B long-context runs do not allocate all large intermediates at `max_context` size.
- `QWEN35X_PREFILL_MLP_CHUNK_TOKENS` overrides the MLP/DeltaNet chunk size. The default is `4096`.
- `QWEN35X_PREFILL_ATTENTION_QUERY_TOKENS` overrides the materialized full-attention query tile. Defaults are variant-aware: large tile for 0.8B throughput, 64-token tile for 4B VRAM safety.

Current Qwen35x precision behavior:
- `--qwen35x-weight-precision bf16` and `--qwen35x-cache-precision bf16` are the stable default path.
- `--qwen35x-weight-precision nvfp4` validates AxionML/ModelOpt NVFP4 weights and runs decode with native packed NVFP4 projection weights plus BF16 KV cache. The current decode path is a correctness-first scalar dequantized warp matvec; standalone custom projection kernels are being brought up before replacing decode hot-path projections.
- `--qwen35x-cache-precision quantized` is reserved for the upcoming quantized-cache path and currently fails fast with a clear unsupported-mode diagnostic.
- AxionML/ModelOpt NVFP4 checkpoint layout can be checked before CUDA loading with:
  `.\build\qwen35x.exe --validate-nvfp4-model --hf-model-dir models/qwen3.5-0.8b-nvfp4-axionml`
- The first NVFP4 CUDA primitive can be checked against a CPU dequantized reference with:
  `.\build\qwen35x.exe --check-nvfp4-tensor --hf-model-dir models/qwen3.5-0.8b-nvfp4-axionml --nvfp4-tensor model.language_model.layers.0.mlp.gate_proj`
- The custom batch-1 NVFP4 projection benchmark can be run with:
  `.\build\qwen35x.exe --bench-nvfp4-projection --hf-model-dir models/qwen3.5-0.8b-nvfp4-axionml --nvfp4-tensor model.language_model.layers.0.mlp.gate_proj`
  This benchmark currently defaults to a scale-group CUDA reduction over AxionML packed weights, with `--nvfp4-projection-kernel row|warp|scale-group` available for comparison. It validates against a CPU dequantized reference and is the staging point for the Blackwell FP4 tensor-core projection kernel.
- `--nvfp4-projection-kernel blackwell-fp4` is reserved for the native SM120 tensor-core path. For RTX 50 / SM120 the public native FP4 route is `mma.sync.aligned...kind::mxf4nvf4.block_scale`, not the SM100/103/110 `tcgen05` path. Do not treat a missing `tcgen05` SM120 opcode as a CUDA/hardware blocker. Build native SM120 experiments with `.\scripts\build.ps1 ... -CudaArchitectures 120a` or `120f`; plain `sm_120` is not sufficient for the architecture-specific MMA opcode. The path currently pre-packs AxionML/ModelOpt weights into validated SM120 fragments, packs activations on GPU, then runs a full batch-1 projection through the native MMA instruction. Model loading also materializes SM120 fragment-packed weight buffers for decode integration; decode launch wiring remains next.
- Blackwell/cuBLASLt FP4 availability and layout assumptions can be probed with:
  `.\build\qwen35x.exe --probe-nvfp4-cublaslt --hf-model-dir models/qwen3.5-0.8b-nvfp4-axionml --nvfp4-tensor model.language_model.layers.0.mlp.gate_proj`
  This probe uses the NVIDIA-documented padded/tiled 1D block-scale layout for FP4 scale tensors and the native projection orientation `input[1,K] x weight[rows,K]^T`. It remains diagnostic while the production path moves toward a reusable Blackwell FP4 projection backend.

Current decode control behavior:
- The default Qwen35x CUDA path returns the selected token id each step and performs stop checks on the host.
- `--gpu-decode-blocks <n>` can override decode grid size for tuning; the Qwen35x CUDA kernel clamps unsafe low values to one block per DeltaNet head and reports requested/effective block counts in profile JSON.
- The legacy runtime backend buffers generated token ids on device when no stop tokens/sequences are configured.
- If stop tokens/sequences are configured, decoding uses per-token host-visible checks to preserve immediate stop behavior.

## Quick Start (Windows / PowerShell)

1. Download model

```powershell
.\scripts\download-hf-model.ps1 -InstallDeps
```

2. Build

```powershell
.\scripts\build.ps1 -UseNinja -EnableCuda -Configuration Release
```

The main `qwen35x.exe` includes the supported 0.8B and 4B CUDA variants and selects the correct one automatically from `--hf-model-dir`. The build script still accepts `-CudaVariant` for older benchmark workflows, but it no longer changes the main inference binary.

3. Run chat inference (CPU reference)

```powershell
.\build\qwen35x.exe --infer-reference --hf-model-dir models/qwen3.5-0.8b --chat-user "Tell me a short joke." --max-new-tokens 64 --max-context 256 --stop-on-im-end
```

4. Run chat inference (CUDA / Qwen35x CUDA default)

```powershell
.\build\qwen35x.exe --infer-gpu --hf-model-dir models/qwen3.5-0.8b --chat-user "Tell me a short joke." --max-new-tokens 64 --max-context 256 --temperature 0 --stop-on-im-end
```

The same binary can run the 4B model:

```powershell
.\build\qwen35x.exe --infer-gpu --hf-model-dir models/qwen3.5-4b --chat-user "Tell me a short joke." --max-new-tokens 64 --max-context 256 --temperature 0 --stop-on-im-end
```

5. Run with profiling JSON

```powershell
.\build\qwen35x.exe --infer-gpu --hf-model-dir models/qwen3.5-0.8b --chat-user "Tell me a short joke." --max-new-tokens 64 --max-context 256 --temperature 0 --profile-json build/last_profile.json
```

## Performance Snapshot

Latest local actual-prompt matrix benchmark (April 25, 2026, RTX 5080 Laptop GPU, `MaxNewTokens=128`). Table cells are `prefill tok/s | generation tok/s`.

| Model | Ctx | qwen35x | llama.cpp + FA | llama.cpp |
|---|---:|---:|---:|---:|
| 0.8B | 256 | `17,544.39` \| `325.97` | `7,758.50` \| `244.68` | `3,429.27` \| `252.01` |
| 0.8B | 512 | `21,726.28` \| `331.72` | `16,240.52` \| `247.80` | `6,868.95` \| `225.39` |
| 0.8B | 1024 | `22,783.52` \| `312.55` | `14,456.74` \| `243.20` | `9,205.30` \| `247.52` |
| 0.8B | 2048 | `20,562.67` \| `316.40` | `14,933.44` \| `234.52` | `11,450.36` \| `242.46` |
| 0.8B | 4096 | `18,446.63` \| `306.01` | `15,315.75` \| `217.94` | `12,615.59` \| `222.91` |
| 4B | 256 | `4,660.76` \| `61.31` | `1,942.74` \| `52.03` | `1,520.10` \| `52.25` |
| 4B | 512 | `4,915.72` \| `61.24` | `2,729.85` \| `51.39` | `2,277.65` \| `51.43` |
| 4B | 1024 | `4,844.64` \| `57.89` | `3,385.12` \| `50.28` | `2,976.50` \| `50.77` |
| 4B | 2048 | `4,716.04` \| `58.14` | `3,594.46` \| `48.32` | `3,262.50` \| `47.97` |
| 4B | 4096 | `4,010.30` \| `55.24` | `3,311.87` \| `51.32` | `3,061.02` \| `50.03` |

Source: `benchmarks/model-matrix/qwen35x-vs-llama-matrix-summary.csv`. qwen35x uses the sequential inference harness with real prompt files; llama.cpp uses warmed `llama-completion` actual prompt/eval timings. These actual-run prompt-eval timings are not directly equivalent to synthetic `llama-bench pp*` prefill-only timings, especially at small context sizes. Reproduce with:

```powershell
.\scripts\benchmark-qwen35x-vs-llama-matrix.ps1
```

Latest local long-context Qwen35x CUDA benchmark snapshot (April 25, 2026, RTX 5080 Laptop GPU):

| Workload | qwen35x | llama.cpp | llama.cpp + FA |
|---|---:|---:|---:|
| 0.8B 64k Wikipedia prompt prefill | `8,144.61 tok/s` | `5,369.00 tok/s` | `9,371.15 tok/s` |
| 0.8B 64k Wikipedia prompt generation | `198.24 tok/s` | `139.88 tok/s` | `165.42 tok/s` |
| 4B 64k Wikipedia prompt prefill | `2,170.42 tok/s` | `1,795.70 tok/s` | `2,894.05 tok/s` |
| 4B 64k Wikipedia prompt generation | `48.29 tok/s` | `38.23 tok/s` | `43.87 tok/s` |

The current Qwen35x long-context prefill path still uses materialized tiled full attention, but its scratch allocation is now chunked for MLP/DeltaNet and uses variant-aware attention query tiling. This keeps the 0.8B 64k prefill path near the previous throughput while allowing the 4B 64k Wikipedia benchmark to complete without the prior VRAM spill collapse. The 0.8B path remains faster than llama.cpp without Flash Attention on 64k prefill, but still trails llama.cpp with Flash Attention.

Full-attention prefill subphase split for the current 64k run:

| Subphase | Time |
|---|---:|
| QK | `1,693.86 ms` |
| Softmax | `1,656.64 ms` |
| PV | `1,322.32 ms` |
| Gate | `15.38 ms` |

Latest local short-context integrated Qwen35x CUDA benchmark snapshot (April 24-25, 2026, Qwen3.5-0.8B, RTX 5080 Laptop GPU):

| Workload | qwen35x avg | Samples | CSV |
|---|---:|---|---|
| `pp256` prefill-only | `19,739.13 tok/s` | `20,112.19`, `18,994.76`, `20,110.50` | `benchmarks/qwen35x-pp256-prefill-only-current-rerun.csv` |
| `prompt1/gen128` generation | `300.46 tok/s` | `317.89`, `312.77`, `270.72` | `benchmarks/qwen35x-tg-prompt1-current-rerun.csv` |
| `pp256/gen128`, `MaxContext=384`, prefill | `18,915.00 tok/s` | end-to-end run | `benchmarks/qwen35x-full-pp256-gen128-current-ctx384.csv` |
| `pp256/gen128`, `MaxContext=384`, generation | `302.38 tok/s` | end-to-end run | `benchmarks/qwen35x-full-pp256-gen128-current-ctx384.csv` |
| `chat_short_joke/gen128`, single-path prefill cleanup | `274.30 tok/s` | `275.22`, `268.66`, `279.03` | `benchmarks/qwen35x-short-gen128-qwen35x-prefill-single-path.csv` |

Comparison against saved llama.cpp BF16 CUDA artifacts from the earlier local run:

| Metric | qwen35x | llama.cpp | llama.cpp + FA | qwen35x vs llama.cpp | qwen35x vs llama.cpp + FA |
|---|---:|---:|---:|---:|---:|
| `pp256` prefill tok/s | `19,739.13` | `12,597.12` | `13,681.26` | `1.57x` | `1.44x` |
| `prompt1/gen128` generation tok/s | `300.46` | `140.15` | `142.59` | `2.14x` | `2.11x` |

Commands:

```powershell
$tokens = (1..256 | ForEach-Object { "198" }) -join ","
.\scripts\benchmark-inference-seq.ps1 -Modes gpu-f32 -PromptMode prompt-tokens -PromptName pp_256_prefill_only_current_rerun -PromptTokensCsv $tokens -Runs 3 -WarmupRuns 1 -MaxNewTokens 0 -MaxContext 256 -PrefillOnly -CsvOut benchmarks\qwen35x-pp256-prefill-only-current-rerun.csv -RunLabel qwen35x-pp256-prefill-only-current-rerun
.\scripts\benchmark-inference-seq.ps1 -Modes gpu-f32 -PromptMode prompt-tokens -PromptName tg_prompt1_current_rerun -PromptTokensCsv "198" -Runs 3 -WarmupRuns 1 -MaxNewTokens 128 -MaxContext 256 -CsvOut benchmarks\qwen35x-tg-prompt1-current-rerun.csv -RunLabel qwen35x-tg-prompt1-current-rerun
```

With the default Qwen35x CUDA backend, the `gpu-bf16`/`gpu-f32` labels are legacy benchmark modes; both exercise the same Qwen35x CUDA decode path.

Benchmark comparison history is documented here:

- [docs/benchmark-comparison-2026-04-22.md](docs/benchmark-comparison-2026-04-22.md)

It includes:

- qwen35x vs the historical kernel harness vs llama.cpp (with/without FlashAttention)
- decode + prefill numbers
- Qwen35x kernel block-size/decode-block tuning results
- CSV source file references for reproducibility

Quick headline numbers from the latest comparison update:

- Current qwen35x Qwen35x CUDA default: `300.46 tok/s` generation and `19,739.13 tok/s` `pp256` prefill
- Current qwen35x 64k Wikipedia prompt: `7,870.08 tok/s` prefill and `201.18 tok/s` generation
- llama.cpp BF16 + FA saved run: `142.59 tok/s` generation and `13,681.26 tok/s` `pp256` prefill
- llama.cpp BF16 + FA 64k Wikipedia prompt: `9,371.15 tok/s` prefill and `165.42 tok/s` generation
- Historical standalone Qwen35x kernel bench harness decode: `267.45 tok/s`

Parity status (April 25, 2026):
- CPU reference vs PyTorch/Transformers external oracle (`max_new_tokens=4`): `5/5` minimal prompts pass for both prompt token IDs and generated token IDs.
- Default GPU path vs CPU reference (`--infer-gpu`, Qwen35x batched prefill, `gpu-f32`, `max_new_tokens=4`): `5/5` minimal prompts pass and `12/12` extended prompts pass.
- The CPU reference remains the primary local oracle for GPU work. `scripts/benchmark-transformers-parity.ps1` cross-checks that oracle against PyTorch/Transformers.

Historical pre-default-kernel local sequential benchmark command (kept for reference):

Command:

```powershell
.\scripts\benchmark-inference-seq.ps1 -Executable build/qwen35x.exe -HFModelDir models/qwen3.5-0.8b -PromptMode chat-user -PromptText "Tell me a short joke." -Runs 3 -WarmupRuns 1 -MaxNewTokens 128 -MaxContext 256 -Temperature 0 -TopP 0.8 -TopK 20 -RepeatPenalty 1.05 -Seed 123 -CsvOut benchmarks/qwen35x-inference-seq-post-streaming-attn.csv
```

Results:
- BF16 matvec ON (historical `--infer-gpu` default at the time): `180.76`, `182.03`, `169.35` tokens/s (avg `177.38`)
- FP32 matvec (`--gpu-f32-matvec`): `116.99`, `117.77`, `117.35` tokens/s (avg `117.37`)
- Recent optimization delta:
  - Prior main rerun checkpoint: BF16 avg `166.90` tokens/s
  - Post streaming full-attention kernel: BF16 avg `177.38` tokens/s (peak `182.03`)

## Useful Commands

- Deterministic run:

```powershell
.\build\qwen35x.exe --infer-gpu --hf-model-dir models/qwen3.5-0.8b --chat-user "Summarize this project in one sentence." --max-new-tokens 64 --max-context 256 --temperature 0 --seed 1234
```

- Force the legacy runtime backend with FP32 CUDA matvec:

```powershell
.\build\qwen35x.exe --infer-gpu --gpu-decode-backend default --gpu-f32-matvec --hf-model-dir models/qwen3.5-0.8b --prompt-text "Once upon a time" --max-new-tokens 32 --temperature 0 --seed 123
```

- Enable synchronized CUDA stage timing:

```powershell
.\build\qwen35x.exe --infer-gpu --profile-sync --hf-model-dir models/qwen3.5-0.8b --prompt-text "Once upon a time" --max-new-tokens 32 --temperature 0 --seed 123
```

- BF16 benchmark:

```powershell
.\build\qwen35x.exe --bench-bf16 --hf-model-dir models/qwen3.5-0.8b
```

- Deterministic CPU vs GPU parity smoke (short prompts, minimal generated tokens):

```powershell
.\scripts\benchmark-parity.ps1 -CsvOut benchmarks\qwen35x-parity.csv
```

- Deterministic CPU vs GPU parity (extended prompt suite):

```powershell
.\scripts\benchmark-parity.ps1 -PromptsFile scripts\bench\parity_prompts.txt -CsvOut benchmarks\qwen35x-parity-extended.csv
```

- Install optional PyTorch/Transformers parity dependencies:

```powershell
.\scripts\setup-transformers-parity.ps1
```

The setup script creates `.venv-hf-parity` by default and installs Transformers from source/main, because the current local Qwen3.5 config uses a dev architecture (`qwen3_5`) that may not be available in the latest PyPI release.
To use an existing Python environment instead, pass `-PythonExe python` to the benchmark wrapper.
This path is for correctness only; the local validation used the torch fallback implementation, not the optional optimized linear-attention dependencies.

- Compare the CPU reference against Hugging Face Transformers:

```powershell
.\scripts\benchmark-transformers-parity.ps1 -PromptsFile scripts\bench\parity_prompts_minimal.txt -CsvOut benchmarks\qwen35x-transformers-parity.csv
```

- Sequential inference benchmark to CSV (minimal schema):

```powershell
.\scripts\benchmark-inference-seq.ps1 -RunLabel "baseline" -Runs 3 -WarmupRuns 1 -CsvOut benchmarks\qwen35x-inference-seq.csv
```

## Project Layout

- `src/`, `include/`: engine implementation
- `src/kernels/cuda/`: in-tree MIT-licensed Qwen35x CUDA kernel sources used by the build
- `configs/`: model profile(s)
- `scripts/`: build/download/benchmark utilities
- `scripts/hf/`: optional PyTorch/Transformers comparison tooling
- `docs/architecture.md`: architecture notes
- `docs/development_plan.md`: public development plan
- `third_party/reference/*`: pinned read-only reference submodules

## Roadmap

See [docs/development_plan.md](docs/development_plan.md) for the staged plan from reference parity to specialized Qwen3.5 GPU kernels.

## References

See [docs/references.md](docs/references.md) for submodule provenance and reference workflow.

## License

- Project code: MIT ([LICENSE](LICENSE))
- The in-tree Qwen35x CUDA kernels are adapted from Lucebox MIT sources; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Third-party attribution: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
