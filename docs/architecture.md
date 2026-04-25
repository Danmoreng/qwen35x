# Architecture Notes

The scaffold follows the public project roadmap in `docs/development_plan.md`.

## Current Runtime Status

- CPU reference path remains the local correctness oracle for GPU work.
- Optional PyTorch/Transformers tooling can cross-check the CPU reference against an external implementation.
- `--infer-gpu` defaults to the in-tree Qwen35x CUDA backend for Qwen3.5-0.8B.
- The legacy CUDA runtime decode backend remains selectable with `--gpu-decode-backend default`.
- CUDA decode paths use device-resident layer math across hidden/residual/norm/attention/MLP work.
- Device-token GPU decode loop is implemented for the no-stop-controls path.
- CUDA Graph replay is implemented for steady-state MLP decode work.
- Per-token D2H transfer is reduced to minimal control-path data.
- Profiling is available via `--profile-json` for stage timing and transfer accounting.

## Design split

- Compiler layer
  - Input: model profile (later HF config + safetensors metadata)
  - Output: static packed tensor plan and op schedule
  - Responsibility: model-specific fingerprint checks and pack-layout decisions

- Runtime layer
  - Input: target GPU info and compiled plan
  - Output: dispatch table for prefill/decode operators
  - Responsibility: cache initialization, kernel variant selection, scheduling hooks

- Kernel layer
  - Input: key `(op, mode, dtype, layout, sm)`
  - Output: callable kernel symbol
  - Responsibility: architecture-specific CUDA fast paths and reference fallbacks
  - In-tree Qwen35x CUDA kernel sources live under `src/kernels/cuda/qwen35x_megakernel/`

## Initial scope

- Family: Qwen3.5
- Starter profile: 0.8B
- Target GPU class: Blackwell (SM120)
- Precision phase 1: BF16/FP16 dense path
- Precision phase 2: low precision path (FP8/FP4 or custom packed W4A16-style)

## Milestone order

1. Parse real HF `config.json` and `model.safetensors.index.json`.
2. Build offline packer output (`.q35xpack`) from safetensors shards.
3. Replace correctness-first prompt replay with dedicated batched/specialized prefill.
4. Expand CUDA Graph or persistent-kernel coverage where it still benefits the legacy runtime path.
5. Add quantized path and architecture-specific autotuning.
