# Q4_H128 Design and Evaluation Plan

## Scope

This workstream develops an engine-native four-bit weight format for
Qwen3.5-0.8B. Compatibility with GGUF or llama.cpp is not a requirement. The
source of truth for every conversion is the local BF16 safetensors checkpoint;
an existing Q4 or Q8 artifact must never be requantized into Q4_H128.

The primary format is `Q4_H128`. An H64 variant is useful only as a controlled
ablation if H128 transform overhead or padding proves material. The initial
implementation targets portable scalar x86-64 and AVX2/FMA/F16C. AVX-VNNI and
AVX-512 backends remain independent follow-up work on capable machines.

## Mathematical contract

For every eligible weight row, partition the input dimension into 128-element
blocks. For block `b`, define a deterministic diagonal sign matrix `D_b` and
the normalized Walsh-Hadamard matrix `H_128`. The orthogonal transform is

```text
R_b = H_128 D_b / sqrt(128)
```

and inference preserves the original projection algebra as

```text
y = W x = (W R^T) (R x).
```

The converter transforms `W R^T` once and quantizes the transformed rows to
signed four-bit values. The runtime transforms each activation block once,
quantizes it to Q8, and reuses it for every projection that consumes the same
activation. The same transform definition, sign generator, normalization,
rounding rule, scale grouping, and integer packing are part of the format ABI.

The first correctness implementation uses one symmetric FP16 scale per 32
weights and the existing signed Q4 nibble convention. Scale groups of 64 and
128 are later quality/performance ablations. This separates Hadamard quality
effects from aggressive scale-metadata changes.

## Artifact ABI

The custom artifact is a single little-endian file with:

- an eight-byte magic value, format version, endian marker, and header size;
- the model dimensions and tensor count needed to reject incompatible models;
- a tensor directory containing UTF-8 name, rank, dimensions, encoding,
  transform kind, transform block size, scale group, sign-seed identifier,
  payload offset, payload size, and payload checksum;
- aligned tensor payloads so SIMD readers do not need unaligned metadata
  recovery;
- explicit model fingerprint and converter ABI metadata.

Projection matrices whose input dimensions can use complete 128-element blocks
are encoded as Q4_H128. One-dimensional tensors, norms, convolution parameters,
and other small tensors stay F32 initially. Token embeddings and the tied LM
head stay untransformed in the first artifact so vocabulary semantics and
weight tying remain simple; they may still use the established Q4_0 encoding.
The loader must reject unknown versions, overflowed dimensions or offsets,
overlapping payloads, invalid alignment, and truncated files before allocating
large buffers.

## Conversion stages

1. Enumerate the exact Qwen3.5-0.8B tensors required by the runtime and read
   them from BF16 safetensors as F32.
2. Verify tensor names and shapes against `ModelProfile` before conversion.
3. Apply deterministic randomized-sign H128 transforms to eligible projection
   input blocks.
4. Quantize directly from transformed F32 values, write the engine-native
   artifact, and emit a manifest with per-tensor quantization error.
5. Reload the completed artifact through the production loader and verify it;
   do not benchmark converter-owned buffers.

Conversion must be deterministic: the same BF16 input and converter options
must produce byte-identical payloads and checksums.

## Quality evaluation

### Tier 1: transform and packing correctness

- Verify `R^T R = I` numerically for deterministic random and adversarial
  blocks.
- Compare `W x` with `(W R^T)(R x)` before quantization.
- Round-trip every nibble value, scale, odd row boundary, and aligned payload.
- Compare scalar and AVX2 kernels bit-exactly where their integer accumulation
  order is the same, otherwise within a documented floating-point tolerance.

### Tier 2: projection error

For representative attention, DeltaNet, MLP, embedding, and LM-head tensors,
report normalized MSE, relative L2 error, maximum absolute error, and cosine
similarity against BF16/F32 projections. Include synthetic outlier-heavy inputs
and real activations captured from calibration transcripts.

### Tier 3: teacher-forced model divergence

Use `--forced-output-text` or `--forced-output-tokens` together with
`--logits-out` to run the same target sequence through BF16, Q8_0, Q4_0, and
Q4_H128. Compare full-vocabulary dumps with
`scripts/compare-logit-dumps.py`. For each output position it computes stable
full-softmax `D_KL(P_BF16 || P_candidate)`, target-token NLL, top-k agreement,
centered-logit RMSE, and centered cosine similarity.

Teacher forcing is mandatory for position-wise KL. Comparing independently
generated sequences is invalid after the first token divergence because later
logits are conditioned on different histories. Logits are captured before
sampling penalties and forced-token selection.

The calibration split contains the real cleanup system prompt plus German and
English transcripts with fillers, corrections, negation, names, identifiers,
numbers, dates, and punctuation. A disjoint held-out split is the acceptance
set. Calibration examples may guide rounding or mixed precision but must not be
reported as held-out quality.

Acceptance is relative rather than based on an arbitrary universal KL number:

- Q8_0 establishes that the evaluator and BF16 teacher are aligned.
- Existing Q4_0 is the minimum four-bit quality baseline.
- Q4_H128 must improve held-out mean and tail KL/NLL over Q4_0 without repeated
  critical-fact regressions in generated transcript-cleanup outputs.
- Any speed claim must use the same prompts, forced targets, thread count,
  context, and model residency state. Full-logit dumping is a quality mode and
  is never used for throughput claims.

The initial two-position evaluator smoke test produced exact zero KL for a Q4_0
self-comparison, mean KL 0.006824 and 100% top-1 agreement for BF16 versus Q8_0,
and mean KL 1.96188 for BF16 versus Q4_0. This tiny synthetic sample validates
the measurement path only; it is not a model-quality conclusion.

## Performance evaluation

Measure H128 activation transform time, activation Q8 packing time, projection
time, and end-to-end inference separately. Activation transforms must be cached
within a layer and reused across packed QKV, gate/up, and related projections.
Persistent model sessions and prefix caches remain enabled for the intended
HTTP-worker use case.

Repository performance runs use `scripts/benchmark-inference-seq.sh` with
sequential execution and CSV output. The standard progress settings are three
measured runs, one warmup, 128 generated tokens, and context 256. Report prefill
and decode independently against the same-engine Q4_0 backend and the pinned
llama.cpp Q4_0 baseline. Longer 2,000-input/2,000-output transcript simulations
are final wall-clock checks, not inner-loop tuning runs.

## Delivery sequence

1. Land teacher-forced logit dumping and the streaming comparison tool.
2. Land the format definitions, safe reader/writer, and deterministic H128
   scalar primitives with unit tests.
3. Land the BF16-to-Q4_H128 converter and produce a verified local artifact.
4. Land the scalar runtime path and establish complete correctness/quality
   baselines.
5. Land the AVX2 transform and Q4_H128 x Q8 kernels only when each change is
   benchmark-positive and passes the same quality gates.
6. Run held-out transcript quality and comparable prefill/decode benchmarks;
   record rejected experiments as well as accepted ones.

## Initial laptop artifact

The first deterministic conversion completed on the i7-8750H with:

```bash
build-cpu-q8/qwen35x_q4_h128_convert \
  --hf-model-dir models/qwen3.5-0.8b \
  --output models/qwen3.5-0.8b/model-q4-h128.q35h
```

It writes 320 text-model tensors, excludes unused vision and MTP tensors, and
produces a 424,976,960-byte engine artifact. Conversion plus a full payload
checksum pass took 13.12 seconds and peaked at 1,494,688 KiB RSS. Two independent
conversions were byte-identical with SHA-256
`ac3018478e4b0398152870257b23e071464ee503750f39e02708521079abfef9`.
The artifact is intentionally ignored by Git; only its converter, ABI, tests,
and reproducibility record belong in the repository.

The first correctness-first runtime integration reuses the existing packed
Q4×Q8 projection kernels and performs H128 activation transforms in the portable
scalar implementation. On the six-core i7-8750H, three runs after one warmup
gave 252.01 tokens/s for prefill-256 and 59.29 tokens/s for 128-token decode.
The same build and run order gave 319.92 and 62.82 tokens/s respectively for
the existing Q4_0 artifact. Thus the initial transform costs 21.2% of prefill
throughput and 5.6% of decode throughput; this is a correctness baseline, not a
performance acceptance. An AVX2 transform and more aggressive transform reuse
are the next optimization targets.

The first isolated AVX2 H128 implementation preserves bit-exact scalar output
and raises Q4_H128 to 282.33 prefill tokens/s and 59.95 decode tokens/s. This is
+12.0% prefill and +1.1% decode over the scalar-transform baseline, while still
trailing Q4_0 by 11.8% and 4.6% respectively. Transform reuse/fusion remains
necessary before claiming throughput parity.

Vectorizing the remaining eight-lane butterflies and computing each 64-bit
sign word only once per H128 block raises the result again to 304.99 prefill
tokens/s and 61.99 decode tokens/s. Relative to the original scalar transform,
that is +21.0% prefill and +4.6% decode. The remaining gap to the contemporaneous
Q4_0 measurements is 4.7% prefill and 1.3% decode.

Distributing independent prompt-token transforms over the existing persistent
CPU executor raises prefill again to 317.58 tokens/s. That is within 0.7% of the
319.92-token/s Q4_0 run while preserving the exact transform and logits. Decode
is unaffected because its batch contains one activation row.

On the deliberately small two-position evaluator smoke sample, BF16 versus
Q4_H128 measured mean KL 1.96881, compared with 1.96188 for the prior Q4_0
sample. This is statistically insufficient to accept or reject the format and
does not replace the held-out transcript suite.

The exact 23-token arithmetic regression is stored in
`scripts/data/q4-h128-quality-cases.json`:

```text
Was ist 2 + 2? Antworte nur mit der Zahl.
```

With the same chat tokens, 12 threads, greedy sampling, and repetition penalty
1.05, BF16, Q4_0, and Q4_H128 generated the same four-token empty thinking
wrapper. At the fifth step Q4_0 selected token 17 (`2`), while BF16 and Q4_H128
selected token 19 (`4`). Because token 17 already occurs in the prompt, its
positive raw logit is divided by 1.05 before greedy selection. The resulting
decision margins were:

| Weights | post-penalty `logit(4) - logit(2)` | next token |
| --- | ---: | ---: |
| BF16 | +4.313247 | 19 (`4`) |
| Q4_0 | -1.038412 | 17 (`2`) |
| Q4_H128 | +3.360266 | 19 (`4`) |

Across all five teacher-aligned positions, raw full-vocabulary comparison
against BF16 measured mean/max KL 0.416220/2.009178 and 80% top-1 agreement for
Q4_0. Q4_H128 measured mean/max KL 0.024985/0.101301 and 100% top-1 agreement.
This case therefore demonstrates a material H128 quality improvement, although
the broader held-out transcript suite remains the production acceptance gate.
