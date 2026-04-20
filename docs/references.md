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

- `cf8b0dbda9ac0eac30ee33f87bc6702ead1c4664`

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
