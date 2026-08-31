#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_BENCH="third_party/reference/llama.cpp/build-cpu-native/bin/llama-bench"
MODEL="models/gguf/Qwen3.5-0.8B-Q8_0.gguf"
CSV_OUT="benchmarks/llama-cpu/qwen3.5-0.8b-q8_0-thread-sweep.csv"
SYSTEM_OUT="benchmarks/llama-cpu/qwen3.5-0.8b-q8_0-thread-sweep-system.txt"
THREADS="1,2,4,6,8,12"
PROMPT_TOKENS=256
GEN_TOKENS=128
REPETITIONS=3
DELAY_SECONDS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llama-bench)
      LLAMA_BENCH="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --csv-out)
      CSV_OUT="$2"
      shift 2
      ;;
    --system-out)
      SYSTEM_OUT="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --prompt-tokens)
      PROMPT_TOKENS="$2"
      shift 2
      ;;
    --gen-tokens)
      GEN_TOKENS="$2"
      shift 2
      ;;
    --repetitions)
      REPETITIONS="$2"
      shift 2
      ;;
    --delay-seconds)
      DELAY_SECONDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

resolve_repo_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$path"
  fi
}

LLAMA_BENCH="$(resolve_repo_path "$LLAMA_BENCH")"
MODEL="$(resolve_repo_path "$MODEL")"
CSV_OUT="$(resolve_repo_path "$CSV_OUT")"
SYSTEM_OUT="$(resolve_repo_path "$SYSTEM_OUT")"

for required in "$LLAMA_BENCH" "$MODEL"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

if (( PROMPT_TOKENS < 1 || GEN_TOKENS < 1 || REPETITIONS < 1 || DELAY_SECONDS < 0 )); then
  echo "Token counts and repetitions must be positive; delay must be non-negative." >&2
  exit 2
fi

mkdir -p "$(dirname "$CSV_OUT")" "$(dirname "$SYSTEM_OUT")"

LLAMA_CPP_DIR="$(git -C "$(dirname "$LLAMA_BENCH")" rev-parse --show-toplevel)"
{
  date --iso-8601=seconds
  uname -a
  if command -v lscpu >/dev/null 2>&1; then
    lscpu
  elif command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -Command \
      "Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed"
  fi
  printf '\nllama.cpp commit: '
  git -C "$LLAMA_CPP_DIR" rev-parse HEAD
  printf 'llama.cpp description: '
  git -C "$LLAMA_CPP_DIR" describe --always --dirty
  printf 'model sha256: '
  sha256sum "$MODEL"
  printf '\ncommand:\n'
  printf '%q ' "$LLAMA_BENCH" \
    -m "$MODEL" \
    -p "$PROMPT_TOKENS" \
    -n "$GEN_TOKENS" \
    -r "$REPETITIONS" \
    -t "$THREADS" \
    --delay "$DELAY_SECONDS" \
    -ngl 0 \
    -dev none \
    -o csv
  printf '\n'
} > "$SYSTEM_OUT"

tmp_csv="${CSV_OUT}.tmp"
trap 'rm -f "$tmp_csv"' EXIT

echo "Starting sequential llama.cpp CPU benchmark"
echo "  model: $MODEL"
echo "  threads: $THREADS"
echo "  prompt/gen: $PROMPT_TOKENS/$GEN_TOKENS"
echo "  repetitions: $REPETITIONS"

"$LLAMA_BENCH" \
  -m "$MODEL" \
  -p "$PROMPT_TOKENS" \
  -n "$GEN_TOKENS" \
  -r "$REPETITIONS" \
  -t "$THREADS" \
  --delay "$DELAY_SECONDS" \
  -ngl 0 \
  -dev none \
  --progress \
  -o csv > "$tmp_csv"

mv "$tmp_csv" "$CSV_OUT"
trap - EXIT

echo "CSV written to: $CSV_OUT"
echo "System metadata written to: $SYSTEM_OUT"
