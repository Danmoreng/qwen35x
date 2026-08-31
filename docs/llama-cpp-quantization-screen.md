# llama.cpp lower-bit quantization screen

- Date: 2026-08-31
- Host: Intel Core i7-8750H, six physical cores / twelve logical CPUs, AVX2
- llama.cpp: `9723942adc518b43c4b95dc4dce6906903eb5e09` (`b10711`)
- Model source: `bartowski/Qwen_Qwen3.5-0.8B-GGUF` at
  `f36b1ea49a332ede8fe5f389bbf5b3575ef71f48`

## Outcome

`Q4_0` is the first format to implement in the native engine. On this laptop it
is both substantially faster than Q8 decode and faster than Q8 prompt
processing. `IQ4_NL` has indistinguishable throughput, but its more complex
codebook kernel is not justified before a simple Q4_0 baseline, and it showed a
less desirable long-rewrite output in the small quality suite.

This is a format preselection result, not a production-quality sign-off. The
0.8B model made material identifier/number errors even in Q8_0, so Q8 output is
not a ground-truth oracle.

## Reproduction

All performance tests were CPU-only, used six threads, F16 K/V cache,
batch/ubatch 2048/512, one llama-bench warmup per phase, and three measured
rounds. Models ran sequentially. Round two reversed format order; round three
returned to forward order. Run:

```bash
scripts/benchmark-llama-quant-sweep.sh \
  --output-dir benchmarks/llama-cpu/quant-sweep-f36b1ea
```

The harness writes model byte sizes and SHA-256 hashes, exact per-model
commands/system metadata, all 189 phase samples, and aggregate CSV output below
the ignored `benchmarks/` directory.

The deterministic rewrite suite used the same system prompt, embedded chat
template, greedy decoding, reasoning disabled, 160 maximum output tokens, and
eight identical German/English inputs for every format:

```bash
scripts/evaluate-llama-quant-rewrites.py \
  --output-dir benchmarks/llama-quality/quant-rewrites-f36b1ea
```

It preserves stdout, stderr, conversation text, an automated critical-fact
screen, and a Markdown report containing all 56 raw outputs. Automated regex
flags are triage aids only; human inspection remains authoritative.

## Model artifacts

| Format | Bytes | SHA-256 |
| --- | ---: | --- |
| Q8_0 | 835,325,024 | `7182e2362766bb9569209bbc24cf1a4cdfbb8ab161babdb2080c84fa62c08c2f` |
| Q6_K | 691,461,216 | `976220309a81b4eb26462657a77570bc6e7d936e8425161c54d6d85488567f95` |
| Q5_K_M | 646,093,920 | `7f5b4185c9a3fd54daae8925927377c80e16e3afa094ea9444b3c0a9077136ab` |
| Q4_K_M | 579,615,840 | `fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526` |
| Q4_0 | 537,517,152 | `407c9dec5a813ebe4ed381e26b9b04f62b9041ea96f1a054ff9aa67c2fe46b36` |
| IQ4_NL | 537,640,032 | `5b83cb0d0ccc8bebe91799426dbb857df3074224072d81ccfbda1db014cd0ef7` |
| IQ4_XS | 523,058,272 | `0208d8c9491848f83fdd9b125fdcdb7495f4656314f323e5c78da8ae6c5d9426` |

No test model was requantized from Q8.

## Performance

The table reports the median of three alternating-order rounds. `pp2048` is
prompt processing. Decode columns are 128 timed one-token evaluations after
the stated context depth.

| Format | pp2048 tok/s | tg128 @ ctx 1 | tg128 @ ctx 256 | tg128 @ ctx 2048 | Estimated 2000 in + 2000 out |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 163.74 | 31.36 | 30.80 | 27.31 | 85.44 s |
| Q6_K | 142.31 | 36.56 | 35.91 | 31.10 | 78.37 s |
| Q5_K_M | 122.95 | 37.93 | 37.33 | 32.13 | 78.52 s |
| Q4_K_M | 179.59 | 41.93 | 41.10 | 35.42 | 67.60 s |
| **Q4_0** | **202.53** | **45.50** | **44.55** | **37.73** | **62.88 s** |
| IQ4_NL | 205.33 | 45.09 | 44.62 | 37.68 | 62.82 s |
| IQ4_XS | 114.46 | 44.20 | 43.23 | 36.02 | 73.01 s |

The last column is a comparison estimate:

```text
2000 / pp2048_rate + 2000 / decode_at_context_2048_rate
```

It is not an actual end-to-end generation measurement and excludes model load,
tokenization, sampling, and chat-template time. It is optimistic because the
decode context grows from roughly 2,000 to 4,000 tokens. A separate long
end-to-end run is required for absolute latency.

### Measured 2,048 input / 2,048 output model compute

A subsequent three-repetition long run compared the selected format with Q8_0
under the same six-thread llama-bench settings:

| Format | pp2048 | Decode after 2048 context | Combined 2048 + 2048 | Combined throughput |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 11.286 s | 78.889 s | 91.425 s | 44.80 tok/s |
| **Q4_0** | **10.522 s** | **58.081 s** | **68.477 s** | **59.82 tok/s** |

Q4_0 therefore removes 22.948 seconds, or 25.1%, from the long combined
model-compute latency and raises combined throughput by 33.5%. Its fixed-depth
long decode is 26.4% lower latency / 35.8% higher throughput. Model load,
tokenization, sampling, and chat-template time remain outside llama-bench's
timer, so application wall clock will be slightly higher.

The dedicated sustained run measured only a 7.3% Q4_0 pp2048 advantage
(194.72 versus 181.47 tok/s), smaller than the alternating format sweep's 23.7%
median advantage. Q4_0 won in both measurements, but prompt-processing gains
should be reported with this run-order/thermal sensitivity rather than as one
universal percentage.

Relative to Q8_0, Q4_0 measured:

- +45.1% decode at context 1;
- +44.6% decode at context 256;
- +38.2% decode at context 2048;
- +23.7% prompt processing at 2,048 tokens;
- 35.7% lower GGUF file size.

Q6_K and Q5_K_M improve decode but regress pp2048 by 13.1% and 24.9%,
respectively. IQ4_XS improves decode but regresses pp2048 by 30.1%. Those
trade-offs are poor for transcript cleanup, where long prefill is a first-class
part of latency.

## Quality observations

The automated screen returned four strict passes out of eight for every
format. That equality is misleading: strict regexes intentionally flag retained
self-correction fragments, while some semantic corruptions require human
inspection.

Common baseline failures:

- Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_NL, and IQ4_XS changed “Friday the
  fourteenth” to “Friday the fourth”. Q4_0 preserved “fourteenth” but produced
  awkward surrounding prose.
- Every format failed to reliably reconstruct both `AC-492` and EUR 13,500 from
  the intentionally difficult spoken-character/number case. Q8_0 produced
  `A: C: V: N: 2` and EUR 3,900, demonstrating that this is not solely a
  low-bit regression.
- Every format retained the obsolete USD 4,618 clause before the corrected USD
  4,680 value instead of removing the false start. The final value remained
  unambiguous, but cleanup quality was incomplete.

Format-specific observations:

- Q4_0 preserved all required facts in the date-correction case and did not
  show a repeated critical-fact regression relative to Q8_0 in this small run.
- IQ4_NL changed the English addressee construction (“Maya Chen, send ... to
  you”) and converted the long German rewrite into an invented protocol schema
  that hit the output limit.
- IQ4_XS used the same unwanted protocol schema on the long case and was much
  slower for prefill, so it has no advantage here despite being the smallest
  file.
- Q4_K_M did not demonstrate a compensating quality advantage over Q4_0 in
  these eight cases.

Before choosing a production quantizer, expand the dataset, add multiple real
transcripts, manually score every output, and compare importance-matrix and
calibrated candidates. The present evidence is sufficient to start the native
Q4_0 kernel, not to declare all Q4_0 models production-safe.

## Decision

1. Implement native Q4_0 x Q8 first, with separate decode and prefill kernels.
2. Retain Q8 as a diagnostic reference, not as an assumed quality oracle.
3. Revisit IQ4_NL only after Q4_0 is correct and benchmarked in this engine.
4. Expand quality calibration before mixed-precision and Hadamard-Q4 decisions.
5. Keep `Q4_H128` as the preferred custom-format research direction after the
   simple Q4 baseline.
