# Reference Code

For publishing and attribution summary, see `THIRD_PARTY_NOTICES.md` in the repo root.

Note: this repository does not vendor third-party reference source files in `docs/reference`; references are maintained via submodules and notes.

## llama.cpp

- Path: `third_party/reference/llama.cpp`
- Source: `https://github.com/ggml-org/llama.cpp`
- Integration mode: reference-only (read-only design/code reference, not linked into the main build)

### Why submodule

- Keeps upstream history and exact provenance.
- Pins an exact commit for reproducible analysis.
- Makes updates explicit (`git submodule update --remote`).

### Common commands

```powershell
git submodule status
git submodule update --init --recursive
git submodule update --remote third_party/reference/llama.cpp
```

### Current pin

- `98dc1418ea0491d62948f712ed534ece3b773564`

## tinygrad

- Path: `third_party/reference/tinygrad`
- Source: `https://github.com/tinygrad/tinygrad`
- Integration mode: reference-only (read-only compiler/lowering/runtime design reference, not linked into the main build)

### Common commands

```powershell
git submodule update --remote third_party/reference/tinygrad
```

### Current pin

- `8eeb77a905aff0108e365dfdc41d36c75b5790a2`

## PyTorch / Transformers

- Path: optional local virtual environment `.venv-hf-parity`
- Requirements: `scripts/hf/requirements-transformers-parity.txt`
- Runner: `scripts/hf/transformers_inference.py`
- Wrapper: `scripts/benchmark-transformers-parity.ps1`
- Integration mode: optional correctness oracle only; not linked into the C++ build and not required for normal inference
- Transformers source/main is the default install target because the local Qwen3.5 config uses the dev `qwen3_5` architecture.
- Latest local validation (April 24, 2026): `torch=2.11.0+cpu`, `transformers=5.7.0.dev0`, CPU-only torch fallback path, minimal parity `5/5`.

### Common commands

```powershell
.\scripts\setup-transformers-parity.ps1
.\scripts\benchmark-transformers-parity.ps1 -PromptsFile scripts\bench\parity_prompts_minimal.txt -CsvOut benchmarks\qwen35x-transformers-parity.csv
```
