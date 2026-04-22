# Vendored Luce Megakernel Source

This directory contains a vendored copy of:

- `kernel.cu` from `Luce-Org/lucebox-hub` (`megakernel/kernel.cu`)

The file is used by the local benchmark target `qwen35x_lucebench`.

## Provenance

- Upstream repository: https://github.com/Luce-Org/lucebox-hub
- Upstream path: `megakernel/kernel.cu`
- License: MIT
- Local license copy: `LICENSE.Lucebox`

## Local modifications

The vendored file includes local changes for:

- occupancy-safe decode launch sizing on non-3090 GPUs
- decode block override/query hooks for throughput tuning

These changes are benchmark-focused and intended to make the kernel runnable/tunable on this machine class.
