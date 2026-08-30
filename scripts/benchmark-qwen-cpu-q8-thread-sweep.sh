#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTABLE="build-cpu-q8/qwen35x"
HF_MODEL_DIR="models/qwen3.5-0.8b"
CPU_GGUF="models/gguf/Qwen3.5-0.8B-Q8_0.gguf"
THREADS="1,2,4,6,8,12"
RUNS=3
WARMUP_RUNS=1
MAX_NEW_TOKENS=128
MAX_CONTEXT=256
CSV_OUT="benchmarks/qwen-cpu/qwen3.5-0.8b-q8_0-thread-sweep.csv"
SYSTEM_OUT="benchmarks/qwen-cpu/qwen3.5-0.8b-q8_0-thread-sweep-system.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --executable) EXECUTABLE="$2"; shift 2 ;;
    --hf-model-dir) HF_MODEL_DIR="$2"; shift 2 ;;
    --cpu-gguf) CPU_GGUF="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --warmup-runs) WARMUP_RUNS="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --max-context) MAX_CONTEXT="$2"; shift 2 ;;
    --csv-out) CSV_OUT="$2"; shift 2 ;;
    --system-out) SYSTEM_OUT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

resolve_repo_path() {
  if [[ "$1" = /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$REPO_ROOT" "$1"; fi
}

EXECUTABLE="$(resolve_repo_path "$EXECUTABLE")"
HF_MODEL_DIR="$(resolve_repo_path "$HF_MODEL_DIR")"
CPU_GGUF="$(resolve_repo_path "$CPU_GGUF")"
CSV_OUT="$(resolve_repo_path "$CSV_OUT")"
SYSTEM_OUT="$(resolve_repo_path "$SYSTEM_OUT")"

if [[ ! "$THREADS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  echo "Invalid --threads list: $THREADS" >&2
  exit 2
fi
for required in "$EXECUTABLE" "$CPU_GGUF" "$HF_MODEL_DIR/config.json"; do
  if [[ ! -e "$required" ]]; then echo "Missing required path: $required" >&2; exit 1; fi
done

mkdir -p "$(dirname "$CSV_OUT")" "$(dirname "$SYSTEM_OUT")"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwen35x-cpu-q8-sweep.XXXXXX")"
cleanup() { rm -r -- "$TEMP_DIR"; }
trap cleanup EXIT

{
  date --iso-8601=seconds
  uname -a
  lscpu
  printf '\nqwen35x commit: '
  git -C "$REPO_ROOT" rev-parse HEAD
  printf 'qwen35x description: '
  git -C "$REPO_ROOT" describe --always --dirty
  printf 'model sha256: '
  sha256sum "$CPU_GGUF"
  printf 'threads: %s\nruns: %s\nwarmup runs: %s\nmax new tokens: %s\nmax context: %s\n' \
    "$THREADS" "$RUNS" "$WARMUP_RUNS" "$MAX_NEW_TOKENS" "$MAX_CONTEXT"
} > "$SYSTEM_OUT"

IFS=',' read -r -a thread_list <<< "$THREADS"
part_files=()
for thread_count in "${thread_list[@]}"; do
  part_csv="$TEMP_DIR/threads-${thread_count}.csv"
  part_files+=("$part_csv")
  echo "Running qwen35x Q8_0 with $thread_count CPU thread(s)"
  "$REPO_ROOT/scripts/benchmark-inference-seq.sh" \
    --executable "$EXECUTABLE" \
    --hf-model-dir "$HF_MODEL_DIR" \
    --mode cpu-q8 \
    --cpu-gguf "$CPU_GGUF" \
    --cpu-threads "$thread_count" \
    --cpu-isa auto \
    --runs "$RUNS" \
    --warmup-runs "$WARMUP_RUNS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-context "$MAX_CONTEXT" \
    --repeat-penalty 1 \
    --prompt-tokens 198 \
    --run-label "q8_0-t${thread_count}" \
    --csv-out "$part_csv"
done

python3 - "$CSV_OUT" "${part_files[@]}" <<'PY'
import csv
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
rows = []
fieldnames = None
for input_name in sys.argv[2:]:
    with pathlib.Path(input_name).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        rows.extend(reader)
if not fieldnames or not rows:
    raise RuntimeError("thread sweep produced no benchmark rows")
with output.open("w", newline="", encoding="utf-8") as target:
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\nThread sweep averages:")
for threads in sorted({int(row["cpu_threads"]) for row in rows}):
    selected = [row for row in rows if int(row["cpu_threads"]) == threads]
    prefill = sum(float(row["prefill_tokens_per_second"]) for row in selected) / len(selected)
    decode = sum(float(row["tokens_per_second"]) for row in selected) / len(selected)
    print(f"  {threads:2d} threads: prefill={prefill:8.2f} tok/s decode={decode:8.2f} tok/s")
PY

echo "CSV written to: $CSV_OUT"
echo "System metadata written to: $SYSTEM_OUT"
