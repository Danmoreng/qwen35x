# Qwen3.5-0.8B deterministic text benchmark (1024 input / 512 output)

This benchmark compares single-stream, fixed-length BF16 text inference across:

- `qwen35x`
- Moondream Photon / Kestrel
- vLLM

It is intentionally different from the image-heavy ChartQA serving benchmark published for Photon 2.0. The goal here is narrower: measure long text prefill followed by sustained text decode without image encoding, batching, early stopping, or prefix-cache reuse.

## Pinned workload

The workload is defined by `configs/qwen3_5_0_8b_text_1024x512.json`.

- Model: `Qwen/Qwen3.5-0.8B`
- Hugging Face revision: `2fc06364715b967f1860aea9cf38778875588b17`
- Precision: BF16
- Prompt: exactly 1024 token IDs, including the Qwen chat framing
- Output: exactly 512 generated tokens
- Maximum context: 2048
- Temperature: 0
- Top-p: 0.8 (irrelevant under greedy decoding, but identical across engines)
- Repetition penalty: 1.0
- Concurrency: 1
- Prefix cache: disabled
- Warmups: 2
- Measured runs: 5, executed sequentially

The prompt text, exact prompt token IDs, tokenizer checksum, and weight checksum are committed. The result combiner rejects a run if any engine did not process exactly 1024 input and 512 output tokens.

## Fixed-length decode and Photon

`qwen35x` naturally runs to `max_new_tokens` when no stop token is supplied. vLLM uses its public `ignore_eos=True` option.

Photon's public chat API does not expose an equivalent fixed-length option. For this raw-decode benchmark, `scripts/benchmark-photon-seq.py --ignore-eos` replaces only the Python prompt template's stop-ID lists after engine initialization. It does not alter Photon kernels, scheduling, model weights, sampling, or timing. This small benchmark adapter is included in the repository for inspection. Without it, Photon can stop early and the engines perform different amounts of decode work.

## Setup

Download the complete model snapshot at the pinned revision:

```bash
python3 -m venv .venv-download
.venv-download/bin/pip install huggingface-hub
.venv-download/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
    local_dir="models/qwen3.5-0.8b",
)
PY
```

Build `qwen35x` on Linux:

```bash
./scripts/build.sh --clean --ninja --config Release --target qwen35x
```

Create isolated Photon and vLLM environments:

```bash
python3 -m venv .venv-photon
.venv-photon/bin/pip install -r scripts/bench/requirements-photon-text-benchmark.txt

python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -r scripts/bench/requirements-vllm-text-benchmark.txt
```

The published run used Python 3.14 and CUDA 13 wheels.

## Run all engines

```bash
./scripts/benchmark-text-1024x512-seq.sh
```

Alternative environment paths can be supplied explicitly:

```bash
PHOTON_PYTHON=/path/to/photon/bin/python \
VLLM_PYTHON=/path/to/vllm/bin/python \
QWEN_EXECUTABLE=/path/to/qwen35x \
RESULT_DIR=/path/to/results \
./scripts/benchmark-text-1024x512-seq.sh
```

The runner records system information, raw per-engine CSV files, a combined CSV, and a JSON summary. It executes engines sequentially so they do not contend for the GPU.

## Timing definitions

Each engine's own request metrics provide prefill and decode durations. The normalized summary reports:

- `prefill tokens/s = 1024 / prefill time`
- `steady-state decode tokens/s = 511 / decode time`
- `model time = prefill time + decode time`

The first generated token belongs to prefill, hence 511 steady-state decode steps for 512 output tokens. A second `reported_decode_tokens_per_second` column retains the common engine convention of dividing all 512 output tokens by decode time.

These are model execution metrics, not HTTP serving throughput. Startup is reported separately and should not be mixed into warm request latency.

## RTX 5080 Laptop result

Hardware and software details are in `docs/qwen3_5_0_8b_text_1024x512_rtx5080_laptop_system.txt`. Raw normalized rows are in `docs/qwen3_5_0_8b_text_1024x512_rtx5080_laptop.csv`.

| Engine | Prefill tokens/s | Steady decode tokens/s | Model time |
|---|---:|---:|---:|
| qwen35x | 25,680.84 | **404.58** | **1,302.93 ms** |
| Photon 2.0.1 | 32,716.55 | 318.42 | 1,636.11 ms |
| vLLM 0.26.0 | **40,140.11** | 349.90 | 1,485.94 ms |

On this machine and workload:

- `qwen35x` decode was 27.06% faster than Photon.
- `qwen35x` decode was 15.63% faster than vLLM.
- `qwen35x` model time was 20.36% lower than Photon.
- `qwen35x` model time was 12.32% lower than vLLM.
- vLLM had the fastest long-prompt prefill; `qwen35x` had the fastest sustained decode.

## Scope and limitations

This result does not disprove Moondream's published H100/ChartQA measurements. It demonstrates that those image-serving request-throughput results do not automatically generalize to sustained text decode on a laptop SM120 GPU.

For credible comparisons, publish the exact GPU, power state, package versions, prompt/output lengths, cache behavior, concurrency, raw rows, and timing definition. More machines and more measured runs are preferable. The recorded run did not lock GPU clocks or power, so small differences should not be overinterpreted; the observed decode gaps were substantially larger than the run-to-run variation.
