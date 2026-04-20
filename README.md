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
- Reference generation via `--infer-reference` (CPU) and `--infer-gpu` (CUDA-hybrid)
- Qwen-style default sampling (`temperature=0.7`, `top_p=0.8`, `top_k=20`, `repeat_penalty=1.05`)
- Deterministic mode with `--seed` and `--temperature 0`
- Stop controls: `--stop-token`, `--stop-text`, `--stop-on-im-end`
- BF16 tensor benchmark path (`--bench-bf16`)

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

## Useful Commands

- Deterministic run:

```powershell
.\build\qwen35x.exe --infer-gpu --hf-model-dir models/qwen3.5-0.8b --chat-user "Summarize this project in one sentence." --max-new-tokens 64 --max-context 256 --temperature 0 --seed 1234
```

- BF16 benchmark:

```powershell
.\build\qwen35x.exe --bench-bf16 --hf-model-dir models/qwen3.5-0.8b
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
