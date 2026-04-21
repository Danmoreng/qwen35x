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
- Device-resident decode path for per-layer hidden/residual/norm/attention/MLP math in `--infer-gpu`
- GPU logits + GPU sampling path
- Device-token decode loop path for `--infer-gpu` (sampled token stays on GPU for next-step embedding gather)
- CUDA Graph replay for steady-state decode segments (MLP + linear-attention blocks)
- BF16 decode matvec path for CUDA inference (`--infer-gpu` defaults to BF16 matvec, override with `--gpu-f32-matvec`)
- Packed full-attention projection path (`q+gate+k+v`) to reduce decode matvec launches in full-attention blocks
- Qwen-style default sampling (`temperature=0.7`, `top_p=0.8`, `top_k=20`, `repeat_penalty=1.05`)
- Deterministic mode with `--seed` and `--temperature 0`
- Stop controls: `--stop-token`, `--stop-text`, `--stop-on-im-end`
- BF16 tensor benchmark path (`--bench-bf16`)
- Profiling output via `--profile-json` (stage timings + H2D/D2H transfer stats)
- Optional synchronized CUDA stage timing via `--profile-sync`

Current GPU sampling constraint:
- For `temperature > 0`, GPU sampling currently supports `top_k` in `[1, 64]`.

Current decode control behavior:
- If no stop tokens/sequences are configured, generated token ids are buffered on device and copied back in bulk after generation.
- If stop tokens/sequences are configured, decoding uses per-token host-visible sampling to preserve immediate stop behavior.

## Quick Start (Windows / PowerShell)

1. Download model

```powershell
.\scripts\download-hf-model.ps1 -InstallDeps
```

2. Build

```powershell
.\scripts\build.ps1 -UseNinja -EnableCuda -Configuration Release
```

3. Run chat inference (CPU reference)

```powershell
.\build\qwen35x.exe --infer-reference --hf-model-dir models/qwen3.5-0.8b --chat-user "Tell me a short joke." --max-new-tokens 64 --max-context 256 --stop-on-im-end
```

4. Run chat inference (CUDA-hybrid)

```powershell
.\build\qwen35x.exe --infer-gpu --hf-model-dir models/qwen3.5-0.8b --chat-user "Tell me a short joke." --max-new-tokens 64 --max-context 256 --stop-on-im-end
```

5. Run with profiling JSON

```powershell
.\build\qwen35x.exe --infer-gpu --hf-model-dir models/qwen3.5-0.8b --chat-user "Tell me a short joke." --max-new-tokens 64 --max-context 256 --profile-json build/last_profile.json
```

## Performance Snapshot

Latest local sequential benchmark (April 21, 2026, same machine):

Command:

```powershell
.\build\qwen35x.exe --infer-gpu --hf-model-dir models\qwen3.5-0.8b --chat-user "Tell me a short joke." --max-new-tokens 128 --max-context 256 --seed 123 --temperature 0
```

Results:
- BF16 matvec ON (`--infer-gpu` default): `183.34`, `180.06`, `183.52` tokens/s
- FP32 matvec (`--gpu-f32-matvec`): typically `~115-119` tokens/s on stable runs
- Recent optimization delta:
  - Pre packed full-attention projection checkpoint: BF16 avg `178.91` tokens/s
  - Post packed full-attention projection checkpoint: BF16 avg `182.31` tokens/s

## Useful Commands

- Deterministic run:

```powershell
.\build\qwen35x.exe --infer-gpu --hf-model-dir models/qwen3.5-0.8b --chat-user "Summarize this project in one sentence." --max-new-tokens 64 --max-context 256 --temperature 0 --seed 1234
```

- Force FP32 CUDA matvec (disable BF16 decode matvec):

```powershell
.\build\qwen35x.exe --infer-gpu --gpu-f32-matvec --hf-model-dir models/qwen3.5-0.8b --prompt-text "Once upon a time" --max-new-tokens 32 --temperature 0 --seed 123
```

- Enable synchronized CUDA stage timing:

```powershell
.\build\qwen35x.exe --infer-gpu --profile-sync --hf-model-dir models/qwen3.5-0.8b --prompt-text "Once upon a time" --max-new-tokens 32 --temperature 0 --seed 123
```

- BF16 benchmark:

```powershell
.\build\qwen35x.exe --bench-bf16 --hf-model-dir models/qwen3.5-0.8b
```

- Sequential inference benchmark to CSV (minimal schema):

```powershell
.\scripts\benchmark-inference-seq.ps1 -RunLabel "baseline" -Runs 3 -WarmupRuns 1 -CsvOut benchmarks\qwen35x-inference-seq.csv
```

## Project Layout

- `src/`, `include/`: engine implementation
- `configs/`: model profile(s)
- `scripts/`: build/download/benchmark utilities
- `docs/architecture.md`: architecture notes
- `docs/development_plan.md`: public development plan
- `third_party/reference/*`: pinned read-only reference submodules

## Roadmap

See [docs/development_plan.md](docs/development_plan.md) for the staged plan from reference parity to specialized Qwen3.5 GPU kernels.

## References

See [docs/references.md](docs/references.md) for submodule provenance and reference workflow.

## License

- Project code: MIT ([LICENSE](LICENSE))
- Third-party attribution: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
