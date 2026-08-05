#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

QWEN_EXECUTABLE="${QWEN_EXECUTABLE:-build/qwen35x}"
PHOTON_PYTHON="${PHOTON_PYTHON:-.venv-photon/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-.venv-vllm/bin/python}"
RESULT_DIR="${RESULT_DIR:-benchmarks/text-1024x512}"
CONFIG="configs/qwen3_5_0_8b_text_1024x512.json"
PROMPT="configs/qwen3_5_0_8b_text_1024x512_prompt.txt"
TOKENS="configs/qwen3_5_0_8b_text_1024x512_tokens.csv"
MODEL_DIR="models/qwen3.5-0.8b"

for path in "$QWEN_EXECUTABLE" "$PHOTON_PYTHON" "$VLLM_PYTHON" "$CONFIG" "$PROMPT" "$TOKENS"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$RESULT_DIR"
{
  date --iso-8601=seconds
  uname -a
  git rev-parse HEAD
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
  "$PHOTON_PYTHON" -c 'import moondream, torch; print("Photon:", moondream.__version__, "Torch:", torch.__version__)'
  "$VLLM_PYTHON" -c 'import vllm, torch; print("vLLM:", vllm.__version__, "Torch:", torch.__version__)'
  sha256sum "$PROMPT" "$TOKENS" "$MODEL_DIR/tokenizer.json" "$MODEL_DIR/model.safetensors-00001-of-00001.safetensors"
} > "$RESULT_DIR/system.txt"

printf '\n=== qwen35x ===\n'
./scripts/benchmark-inference-seq.sh \
  --executable "$QWEN_EXECUTABLE" \
  --hf-model-dir "$MODEL_DIR" \
  --mode gpu-bf16 \
  --runs 5 \
  --warmup-runs 2 \
  --max-new-tokens 512 \
  --max-context 2048 \
  --repeat-penalty 1.0 \
  --prompt-tokens-file "$TOKENS"

printf '\n=== Photon ===\n'
"$PHOTON_PYTHON" scripts/benchmark-photon-seq.py \
  --model Qwen/Qwen3.5-0.8B \
  --model-path "$MODEL_DIR" \
  --prompt-file "$PROMPT" \
  --expected-input-tokens 1024 \
  --ignore-eos \
  --require-full-output \
  --runs 5 \
  --warmup-runs 2 \
  --max-new-tokens 512 \
  --max-batch-size 1 \
  --kv-cache-pages 2048 \
  --csv-out "$RESULT_DIR/photon.csv"

printf '\n=== vLLM ===\n'
"$VLLM_PYTHON" scripts/benchmark-vllm-seq.py \
  --model-path "$MODEL_DIR" \
  --prompt-tokens-file "$TOKENS" \
  --runs 5 \
  --warmup-runs 2 \
  --max-new-tokens 512 \
  --max-context 2048 \
  --gpu-memory-utilization 0.8 \
  --csv-out "$RESULT_DIR/vllm.csv"

python3 scripts/summarize-text-benchmark.py \
  --config "$CONFIG" \
  --qwen-profile-dir build/bench-profiles \
  --photon-csv "$RESULT_DIR/photon.csv" \
  --vllm-csv "$RESULT_DIR/vllm.csv" \
  --csv-out "$RESULT_DIR/combined.csv" \
  --summary-out "$RESULT_DIR/summary.json"
