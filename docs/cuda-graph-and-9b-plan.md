# CUDA Graph Optimization and Qwen3.5-9B Bring-Up Plan

Status date: 2026-07-20

Working branch: `codex/4b-cuda-graph`

Initial implementation commit: `5393af3` (`Add 4B CUDA Graph decode mode`)

This document is the handoff plan for continuing development on Linux. It
records the measured 4B baseline, the next CUDA Graph experiments, and the
staged path to a practical Qwen3.5-9B runtime on a 16 GB Blackwell GPU.

## Current State and Measured Baseline

The initial graph implementation is opt-in through:

```text
--qwen35x-decode-execution graph
```

The megakernel remains the default. Graph mode currently supports the 4B BF16
variant only. It captures 32 layer kernels followed by final normalization and
the LM head. Token id and position are updated through an 8-byte device control
buffer, allowing one graph executable to be reused for all decode positions.

Validation completed on an NVIDIA GeForce RTX 5080 Laptop GPU (SM120, 16,303
MiB VRAM):

- Release CUDA/Ninja build passed with the 0.8B and 4B specializations.
- An 8-token deterministic smoke test was token-exact between the megakernel
  and graph paths.
- Both paths emitted: `2, 220, 17, 15, 16, 24, 95859, 96328`.
- The captured graph contained 99 nodes and used 60 occupancy-safe blocks.
- The 0.8B megakernel regression smoke passed at 329.28 tok/s.
- Selecting graph mode for 0.8B failed fast with the intended 4B-only error.

The comparable sequential benchmark used one warmup, three measured runs, 128
generated tokens, context 256, deterministic greedy sampling, and the same
14-token chat prompt for both paths:

| Decode execution | Runs (tok/s) | Mean (tok/s) | Mean (ms/token) |
|---|---|---:|---:|
| Megakernel | 51.697, 52.577, 51.587 | 51.954 | 19.248 |
| CUDA Graph | 51.591, 50.890, 50.907 | 51.129 | 19.558 |

The first graph implementation is therefore 1.59% slower than the megakernel.
All benchmark runs produced the same 128-token sequence. This is the baseline
to beat; graph mode should remain experimental until it is measurably faster.

## Linux Handoff

The model directories and benchmark CSVs are Git-ignored. They must be restored
on the Linux system and are not contained in the branch.

### Checkout and dependencies

```bash
git clone git@github.com:Danmoreng/qwen35x.git
cd qwen35x
git switch codex/4b-cuda-graph

python3 -m pip install --upgrade huggingface_hub
```

Install a CUDA 13.x toolkit compatible with the installed NVIDIA driver, CMake,
GNU Make, and a C++20 host compiler. Confirm the GPU before building:

```bash
nvidia-smi
nvcc --version
cmake --version
make --version
```

### Restore the current models

Use a single download worker if the machine is being used interactively:

```bash
hf download Qwen/Qwen3.5-0.8B \
  --local-dir models/qwen3.5-0.8b \
  --max-workers 1

hf download Qwen/Qwen3.5-4B \
  --local-dir models/qwen3.5-4b \
  --max-workers 1
```

The Windows copies occupied approximately 1.65 GiB and 8.70 GiB respectively.

### Build and smoke test

```bash
./scripts/build.sh --config Release --target qwen35x --arch native

./build/qwen35x --infer-gpu \
  --hf-model-dir models/qwen3.5-4b \
  --prompt-tokens 198 \
  --max-new-tokens 8 \
  --max-context 256 \
  --temperature 0 \
  --top-p 0.8 \
  --top-k 20 \
  --repeat-penalty 1.05 \
  --seed 123 \
  --qwen35x-decode-execution megakernel

./build/qwen35x --infer-gpu \
  --hf-model-dir models/qwen3.5-4b \
  --prompt-tokens 198 \
  --max-new-tokens 8 \
  --max-context 256 \
  --temperature 0 \
  --top-p 0.8 \
  --top-k 20 \
  --repeat-penalty 1.05 \
  --seed 123 \
  --qwen35x-decode-execution graph
```

Before performance work, confirm that both commands produce the same eight
tokens recorded above. Use an architecture-specific `--arch 120a` build when
enabling the native SM120 NVFP4 MMA experiments.

### Reproduce the 4B benchmark

Use the native Bash sequential harness on Linux; it preserves the CSV schema
and execution policy without requiring PowerShell.

```bash
./scripts/benchmark-inference-seq.sh \
  --executable build/qwen35x \
  --hf-model-dir models/qwen3.5-4b \
  --mode gpu-f32 \
  --runs 3 \
  --warmup-runs 1 \
  --max-new-tokens 128 \
  --max-context 256 \
  --qwen35x-decode-execution megakernel \
  --run-label 4b-megakernel \
  --csv-out benchmarks/qwen35x-4b-decode-execution.csv

./scripts/benchmark-inference-seq.sh \
  --executable build/qwen35x \
  --hf-model-dir models/qwen3.5-4b \
  --mode gpu-f32 \
  --runs 3 \
  --warmup-runs 1 \
  --max-new-tokens 128 \
  --max-context 256 \
  --qwen35x-decode-execution graph \
  --run-label 4b-graph \
  --csv-out benchmarks/qwen35x-4b-decode-execution.csv
```

## 4B CUDA Graph Performance Plan

### G0. Add the missing control experiment — completed

Add an uncaptured `multi_kernel` decode execution mode that launches the exact
same layer sequence and uses the same device buffers as graph mode.

Compare three paths:

```text
megakernel -> uncaptured multi-kernel -> captured CUDA Graph
```

This separates the launch-overhead savings from the cost of decomposing the
megakernel. Keep the megakernel, uncaptured, and graph implementations on the
same mathematical path so token comparisons remain meaningful.

Implemented as `--qwen35x-decode-execution multi-kernel`. It uses the same
per-layer reset kernel, layer kernels, final norm, LM head, device buffers, and
decode grid as graph mode, but launches them directly on the stream.

First Linux measurement (RTX 5080 Laptop GPU, `Runs=3`, `WarmupRuns=1`,
`MaxNewTokens=128`, `MaxContext=256`, prompt token `198`):

| Decode execution | Runs (tok/s) | Mean (tok/s) | Delta vs megakernel |
|---|---|---:|---:|
| Megakernel | 52.205, 52.064, 51.869 | 52.046 | — |
| Uncaptured multi-kernel | 51.660, 51.571, 51.528 | 51.586 | -0.88% |
| CUDA Graph | 51.396, 50.844, 51.020 | 51.086 | -1.84% |

All three paths were token-exact for deterministic 8-token and 128-token
runs. The captured path is currently 0.97% slower than the identical
uncaptured multi-kernel path, so capture alone does not yet recover the cost
of decomposition.

Deliverables:

- `Qwen35xDecodeExecution::multi_kernel` and CLI/JSON/CSV labels.
- Exact-token parity for all three paths.
- Overall decode timing without per-token profiling synchronization.
- An Nsight Systems trace showing CPU launch gaps and graph replay duration.

### G1. Collapse barrier reset nodes — completed

The current graph contains two `cudaMemsetAsync` nodes before every layer and
one before the LM head. These reset nodes account for 65 of the 99 graph nodes.

Implement per-layer barrier state arrays and capture one small reset kernel at
the start of each graph replay. The reset kernel should initialize all layer
counters, layer generations, and the LM-head counter in one node. Each layer
kernel then receives its own counter/generation slot.

Expected graph shape:

```text
1 reset-all + 32 layers + 1 final norm + 1 LM head = 35 nodes
```

The graph now has exactly 35 nodes on the Linux 4B run (60 safe decode
blocks). `launch_decode_graph_reset` initializes the 32 per-layer counter and
generation slots plus the LM-head counter in one captured reset kernel.

For a persistent generation scheme, seed every block from a stable generation
value only after the previous group has completed and left its counter at zero.
Do not allow blocks to sample a generation concurrently with a reset or first
grid barrier; that can race and deadlock the cooperative synchronization.

Validation gates:

- Exact token parity for 8-token and 128-token deterministic runs.
- Counter/generation stress test across at least 16K decode replays.
- No deadlock at both automatic and manually reduced decode-block counts.
- Measure the node reduction and its isolated throughput impact before further
  kernel changes.

### G2. Remove per-thread layer schedule work

`decode_graph_layer_kernel` currently scans all preceding layers on every
thread to derive DeltaNet and full-attention slots, then branches on layer type.
All of this is known when the graph is captured.

Replace it with separate DeltaNet and full-attention graph launchers that
receive precomputed values or direct pointers for:

- Layer weights.
- DeltaNet recurrent-state slot.
- Full-attention K/V cache slot.
- Layer-specific cache/state strides.
- Layer index only where it is still required by packed-weight tables.

Prefer direct layer/cache pointers in captured kernel arguments. Confirm with
Nsight Compute that instruction count falls and occupancy does not regress.

### G3. Sweep captured layer-group size

One kernel per layer is only one point in the design space. Add templated
grouped-layer launchers and benchmark graph group sizes:

```text
1, 2, 4, 8, and 32 layers per kernel
```

The 32-layer case is the existing megakernel limit. Specialize group starts and
lengths at compile time where possible so the sweep does not add runtime layer
schedule loops.

Record for each group size:

- Graph node count.
- Effective decode blocks and occupancy.
- Kernel duration by DeltaNet/full-attention group.
- End-to-end tokens per second at contexts 256, 2K, and 8K.
- Exact output-token parity.

Promote the smallest/flexiblest grouping that is statistically faster than the
megakernel rather than assuming maximum decomposition is desirable.

### Linux experiment log — 2026-07-20

The Linux benchmark harness is now the native Bash script
`scripts/benchmark-inference-seq.sh`; it preserves the sequential JSON/CSV
workflow and does not require PowerShell or Ninja. All results below use the
same deterministic prompt-token benchmark: one warmup, three measured runs,
128 generated tokens, and context 256 on the RTX 5080 Laptop GPU.

Completed graph-layout experiments:

- Direct DeltaNet/full-attention launchers removed the per-layer schedule scan,
  but did not beat the grouped path.
- The natural `3 x DeltaNet + 1 x full-attention` group reduced the captured
  graph from 35 nodes to 11. Fusing final RMSNorm reduced it to 10, and
  persistent barrier generations reduced that to 9. All three execution modes
  remained exact for deterministic 8-token and 128-token runs.
- A requested 16-block grouped grid exposed that DeltaNet needs its 32 gate
  heads covered; graph and multi-kernel initialization now clamp to that safe
  minimum instead of issuing an invalid launch.
- The current graph layout captures two nodes: a monolithic device-controlled
  decode kernel followed by the LM head. It initializes the LM reduction inside
  the decode kernel and carries cooperative barrier generations between graph
  replays. The captured graph was verified to contain two nodes and to remain
  token-exact for 128 tokens.

The two-node graph is close to the direct megakernel, but has not yet produced
a stable win. At 48 decode blocks, one ordered pair measured Graph at 51.503
tok/s versus megakernel at 51.247 tok/s (+0.50%), while reversing the order
measured Graph at 51.086 tok/s versus 51.277 tok/s (-0.37%). A further
BF16-only layer-helper template specialization regressed to about 50.23 tok/s
and was reverted. The remaining variation is too large to promote graph mode;
the next experiment should use kernel-level profiling and a material change to
the BF16 matvec/MLP path rather than further launch-count reductions.

Follow-up experiments added device-resident decode control: the LM head writes
the next token and position into the graph-stable control buffer, and the
monolithic graph decode node marks the current token for repetition penalty.
After the first decode step, Graph therefore avoids both per-token H2D uploads.
The default graph grid is now 48 blocks (unless explicitly overridden), while
the direct megakernel retains its own safe-grid default. The fixed 4B layer
schedule also avoids the graph kernel's layer-type lookup, reducing its
register count from 187 to 186.

The default-path follow-up pairs measured Graph at 51.655 versus megakernel at
51.270 tok/s (+0.75%), then Graph at 51.314 versus megakernel at 51.265
tok/s (+0.10%) with the order reversed. The aggregate +0.42% lead is promising
but remains below the 2–3% promotion target. A Graph-only LM-head dot-product
variant with four independent partial sums remains token-exact for 128
deterministic tokens. Its final ordered pair was 51.471 tok/s for Graph versus
51.190 tok/s for megakernel (+0.55%); this is an encouraging but unreplicated
result and does not change the promotion decision. An eight-way variant and a
256-block LM-head grid were slower. Final retained-configuration parity passed
for 128 deterministic Graph versus megakernel tokens; the same earlier test
also passed for multi-kernel mode. Further work should target a Blackwell
tensor-core or cuBLASLt projection path for the MLP/LM head.

### G4. Split only operations that gain better hardware utilization

After the synchronization and grouping sweeps, split expensive operations only
when an independent launch enables better geometry or Blackwell Tensor Cores:

- NVFP4 MLP gate/up/down projections.
- Full-attention projections.
- DeltaNet projections and recurrence.
- LM head projection/reduction.

Keep normalization, activation, gating, and residual additions fused around
those operations when they do not benefit from separate scheduling.

For BF16 batch-one decode, compare custom GEMV kernels against cuBLAS/cuBLASLt
instead of assuming a GEMM library call will win at `M=1`. For NVFP4, prioritize
the native SM120 MMA route because graph replay alone cannot compensate for
scalar dequantization.

### G5. Optimize the LM head independently

The vocabulary remains 248,320 entries, making the LM head a substantial fixed
cost for every model size. Add dedicated experiments for:

- Hidden-dimension and vocabulary tiling.
- Persistent or two-stage reduction without a host-visible synchronization.
- NVFP4/FP8 LM-head storage where the checkpoint and quality gates permit it.
- Reusing graph-stable scratch and resetting its synchronization state in the
  graph-wide reset kernel.

Keep LM-head changes independently selectable until they pass token/logit
parity gates.

### G6. Promotion criteria

Graph mode remains opt-in until all of the following hold:

- BF16 greedy output is token-exact with the megakernel.
- Mean decode throughput is at least 2-3% above the megakernel in comparable
  three-run sequential measurements.
- The improvement holds at contexts 256, 2K, and 8K.
- No 0.8B regression is introduced.
- No graph recapture or per-token graph-node parameter patching is required.
- Initialization/capture errors fail clearly and leave CUDA resources clean.

## Qwen3.5-9B Bring-Up Plan

The official configuration is available at
<https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json>.

The 9B text model keeps the same broad hybrid schedule as 4B:

- 32 transformer layers.
- Eight repetitions of three DeltaNet layers followed by one full-attention
  layer.
- 16 attention query heads, 4 KV heads, and head dimension 256.
- 16 DeltaNet Q/K heads and 32 value heads with head dimension 128.
- Vocabulary size 248,320.

The principal dimension changes are:

| Dimension | 4B | 9B |
|---|---:|---:|
| Hidden size | 2,560 | 4,096 |
| Intermediate size | 9,216 | 12,288 |
| Layers | 32 | 32 |

The official configuration also advertises a 262,144-token native context.
Initial bring-up remains text-only, single-token autoregressive decode. Vision
weights and the trained MTP layer are explicitly deferred.

### N0. Add descriptor and memory diagnostics first

Before allocating weights, calculate and report:

- Selected text-weight bytes by precision.
- BF16 embedding, norm, and LM-head bytes.
- DeltaNet recurrent-state bytes.
- KV-cache bytes for the requested maximum context.
- Prefill and decode workspace bytes.
- CUDA Graph and tuning scratch.
- Available and reserved device memory from `cudaMemGetInfo`.

Reject unsupported configurations with required-versus-available numbers. The
current 16 GB RTX 5080 Laptop cannot hold the full BF16 repository plus runtime
state, so a partial allocation failure is not an acceptable diagnostic.

### N1. Add the compiled 9B specialization

Create `src/kernels/cuda/variant_9b.cuh` and add it to the multi-variant build.
Extend:

- CMake symbol renaming and CUDA object targets.
- `VariantDescriptor` and runtime dispatch.
- Build-script variant validation.
- Descriptor/shape-validation tests.
- Kernel and profile layer-count assumptions.

Initial constants:

```text
NUM_LAYERS=32
HIDDEN_SIZE=4096
INTERMEDIATE_SIZE=12288
VOCAB_SIZE=248320
FA_Q_HEADS=16
FA_KV_HEADS=4
FA_HEAD_DIM=256
FA_ROTARY_DIM=64
DN_QK_HEADS=16
DN_VALUE_HEADS=32
DN_KEY_DIM=128
DN_VALUE_HEAD_DIM=128
DN_CONV_KERNEL=4
```

Compile both 256- and 512-thread candidates. `MAX_ACT_DIM=12288` implies a
48 KiB float shared activation array before other compiler resource usage, so
block size and occupancy must be measured rather than copied from 4B.

### N2. Establish a BF16 correctness reference

The BF16 variant is still required as the numerical reference even though it
will not fit on the current 16 GB GPU. Validate it on a GPU with at least 24 GB
or through a deliberately implemented offload/reference path.

Gates:

- Exact safetensor shape validation.
- Successful packed-layer table construction.
- One-layer projection and hidden-state comparisons.
- Full single-token decode comparison.
- Deterministic short-generation token parity.
- Prefill/decode cache handoff parity.

Do not optimize or graph-capture 9B before the BF16 specialization is known to
be mathematically correct.

### N3. Make NVFP4 the practical 16 GB path

Use the existing ModelOpt loader with the canonical artifact:

<https://huggingface.co/AxionML/Qwen3.5-9B-NVFP4>

Download it only when the Linux machine has enough disk space:

```bash
hf download AxionML/Qwen3.5-9B-NVFP4 \
  --local-dir models/qwen3.5-9b-nvfp4-axionml \
  --max-workers 1
```

The repository download is approximately 9.4 GB because it also contains
unquantized components such as vision weights. The runtime should load only the
text tensors it needs. The model card describes NVFP4 linear operators with a
BF16 KV cache, matching the repository's current precision rollout.

Required work:

- Extend ModelOpt packed-weight validation to the 4096/12288 projections.
- Reuse the existing U8 E2M1 payload and FP8 E4M3 scale ABI.
- Keep BF16 embedding, norms, recurrent state, and KV cache initially.
- Measure actual peak allocation before promising a supported context length.
- Replace correctness-first scalar dequantized matvecs with native SM120 MMA.
- Keep a clear BF16 fallback diagnostic for larger-memory GPUs.

### N4. Validate NVFP4 numerically and behaviorally

Quantization may prevent exact token parity on every prompt. Use layered gates:

1. Per-projection max/mean error and cosine similarity against BF16.
2. Per-layer hidden-state cosine and norm-ratio checks.
3. Logit/top-candidate comparisons.
4. Deterministic generation stability and valid token ranges.
5. A fixed quality-smoke prompt set for reasoning, code, and multilingual text.
6. Repeated 128-token and long-context runs without NaN, invalid state, or
   synchronization failure.

Record the BF16 reference outputs in small tracked fixtures when licensing and
size permit, so Linux and future systems can rerun correctness gates without
depending on Windows artifacts.

### N5. Tune 9B megakernel before graph mode

Establish a stable NVFP4 megakernel baseline on the Linux GPU and tune:

- Decode block count.
- CUDA block size.
- Native FP4 projection tiles.
- LM-head launch geometry.
- Prefill MLP chunk size.
- Full-attention query tile.
- Maximum context allowed by the memory estimator.

Store results by `(model_variant, precision, SM)` instead of changing global
defaults shared by 0.8B and 4B.

### N6. Port only the winning graph design

Do not copy the initial 99-node 4B graph to 9B. After the 4B grouping and
synchronization experiments identify a winner, implement that graph layout for
the 9B NVFP4 megakernel/projection path.

9B graph promotion gates:

- Meets the NVFP4 numerical and quality thresholds.
- Beats the tuned 9B megakernel by at least 2-3%.
- Fits within the memory estimator's safety margin.
- Requires no per-token recapture.
- Preserves stable execution at short and long contexts.

## Recommended Execution Order

1. Reproduce the 4B Linux parity and throughput baseline.
2. Add uncaptured `multi_kernel` mode.
3. Replace 65 reset nodes with one graph-wide reset kernel.
4. Specialize DeltaNet/full-attention layer arguments and remove schedule scans.
5. Sweep grouped-layer graph granularity.
6. Promote graph mode only if it beats the megakernel.
7. Add 9B memory diagnostics and the compiled BF16 specialization.
8. Validate 9B BF16 on suitable hardware.
9. Bring up AxionML 9B NVFP4 on the 16 GB Blackwell GPU.
10. Tune the 9B megakernel, then port the winning graph layout.

This order keeps correctness references intact, measures each graph change in
isolation, and prevents the larger-model effort from inheriting a graph design
that has not yet demonstrated a performance advantage.
