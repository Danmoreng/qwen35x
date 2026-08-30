#!/bin/bash
set -e

# Simplified benchmark script for Linux
# usage: ./scripts/benchmark-inference-seq.sh --mode gpu-f32 --max-new-tokens 128 ...

EXECUTABLE="${EXECUTABLE:-build/qwen35x}"
HF_MODEL_DIR="${HF_MODEL_DIR:-models/qwen3.5-0.8b-nvfp4}"
RUNS=3
WARMUP_RUNS=1
MAX_NEW_TOKENS=128
MAX_CONTEXT=256
PROMPT_TOKENS="198" # Default to single token prompt
REPEAT_PENALTY=1.05
PREFILL_ONLY=false
MODE="gpu-f32"
CPU_GGUF="models/gguf/Qwen3.5-0.8B-Q8_0.gguf"
CPU_THREADS=0
CPU_ISA="auto"
CSV_OUT="benchmarks/qwen35x-inference-seq.csv"
RUN_LABEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --executable)
      EXECUTABLE="$2"
      shift 2
      ;;
    --hf-model-dir)
      HF_MODEL_DIR="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --warmup-runs)
      WARMUP_RUNS="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --cpu-gguf)
      CPU_GGUF="$2"
      shift 2
      ;;
    --cpu-threads)
      CPU_THREADS="$2"
      shift 2
      ;;
    --cpu-isa)
      CPU_ISA="$2"
      shift 2
      ;;
    --csv-out)
      CSV_OUT="$2"
      shift 2
      ;;
    --run-label)
      RUN_LABEL="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --max-context)
      MAX_CONTEXT="$2"
      shift 2
      ;;
    --repeat-penalty)
      REPEAT_PENALTY="$2"
      shift 2
      ;;
    --prefill-only)
      PREFILL_ONLY=true
      shift
      ;;
    --prompt-tokens)
      PROMPT_TOKENS="$2"
      shift 2
      ;;
    --prompt-tokens-file)
      PROMPT_TOKENS="$(tr -d '[:space:]' < "$2")"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOLVED_EXE="$EXECUTABLE"
[[ "$RESOLVED_EXE" = /* ]] || RESOLVED_EXE="$REPO_ROOT/$RESOLVED_EXE"
RESOLVED_MODEL_DIR="$HF_MODEL_DIR"
[[ "$RESOLVED_MODEL_DIR" = /* ]] || RESOLVED_MODEL_DIR="$REPO_ROOT/$RESOLVED_MODEL_DIR"
RESOLVED_CPU_GGUF="$CPU_GGUF"
[[ "$RESOLVED_CPU_GGUF" = /* ]] || RESOLVED_CPU_GGUF="$REPO_ROOT/$RESOLVED_CPU_GGUF"
RESOLVED_CSV_OUT="$CSV_OUT"
[[ "$RESOLVED_CSV_OUT" = /* ]] || RESOLVED_CSV_OUT="$REPO_ROOT/$RESOLVED_CSV_OUT"
BUILD_DIR="$REPO_ROOT/build"
BENCH_DIR="$BUILD_DIR/bench-profiles"
mkdir -p "$BENCH_DIR"
mkdir -p "$(dirname "$RESOLVED_CSV_OUT")"

echo "Starting benchmark: mode=$MODE, runs=$RUNS, warmup=$WARMUP_RUNS, max_new_tokens=$MAX_NEW_TOKENS, max_context=$MAX_CONTEXT"

run_once() {
  local run_type=$1
  local index=$2
  local profile_json="$BENCH_DIR/${run_type}_${index}.json"
  
  local args=(
    "--hf-model-dir" "$RESOLVED_MODEL_DIR"
    "--max-new-tokens" "$MAX_NEW_TOKENS"
    "--max-context" "$MAX_CONTEXT"
    "--temperature" "0"
    "--top-p" "0.8"
    "--repeat-penalty" "$REPEAT_PENALTY"
    "--seed" "123"
    "--profile-json" "$profile_json"
    "--prompt-tokens" "$PROMPT_TOKENS"
  )
  
  if [ "$MODE" == "gpu-f32" ]; then
    args+=("--infer-gpu" "--gpu-f32-matvec")
  elif [ "$MODE" == "gpu-bf16" ]; then
    args+=("--infer-gpu" "--gpu-bf16")
  elif [ "$MODE" == "nvfp4" ]; then
    args+=("--infer-gpu" "--qwen35x-weight-precision" "nvfp4")
  elif [ "$MODE" == "cpu-q8" ]; then
    args+=(
      "--infer-reference"
      "--cpu-gguf" "$RESOLVED_CPU_GGUF"
      "--cpu-threads" "$CPU_THREADS"
      "--cpu-isa" "$CPU_ISA"
    )
  else
    args+=("--infer-reference")
  fi
  
  if [ "$PREFILL_ONLY" == true ]; then
    args+=("--prefill-only")
  fi

  "$RESOLVED_EXE" "${args[@]}" > /dev/null
  echo "$profile_json"
}

# Warmup
for ((i=1; i<=WARMUP_RUNS; i++)); do
  echo "Warmup run $i..."
  run_once "warmup" "$i" > /dev/null
done

# Benchmark runs
PROFILES=()
for ((i=1; i<=RUNS; i++)); do
  echo "Benchmark run $i..."
  p=$(run_once "run" "$i")
  PROFILES+=("$p")
done

# Python summary
PY_PREFILL_ONLY="False"
if [ "$PREFILL_ONLY" == true ]; then
    PY_PREFILL_ONLY="True"
fi

# Use Python to summarize
python3 - <<EOF
import json
import sys
import os
import csv

profiles = [ "$(echo ${PROFILES[@]} | sed 's/ /", "/g')" ]
if not profiles:
    sys.exit(0)

prefill_tps_list = []
decode_tps_list = []
rows = []

for run_index, p_path in enumerate(profiles, start=1):
    if not p_path: continue
    if not os.path.exists(p_path): continue
    with open(p_path, 'r') as f:
        data = json.load(f)
        prefill_tps_list.append(data.get('prefill_tokens_per_second', 0))
        decode_tps_list.append(data.get('tokens_per_second', 0))
        rows.append({
            'run_label': '$RUN_LABEL',
            'run_index': run_index,
            'mode': '$MODE',
            'cpu_threads': $CPU_THREADS,
            'cpu_isa': '$CPU_ISA',
            'prompt_tokens': data.get('prompt_tokens', 0),
            'generated_tokens': data.get('generated_tokens', 0),
            'load_time_ms': data.get('load_time_ms', 0),
            'prefill_time_ms': data.get('prefill_time_ms', 0),
            'prefill_tokens_per_second': data.get('prefill_tokens_per_second', 0),
            'decode_time_ms': data.get('decode_time_ms', 0),
            'tokens_per_second': data.get('tokens_per_second', 0),
            'max_new_tokens': $MAX_NEW_TOKENS,
            'max_context': $MAX_CONTEXT,
        })

if rows:
    with open('$RESOLVED_CSV_OUT', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def avg(l):
    return sum(l) / len(l) if l else 0

print("\nBenchmark Results Summary:")
print(f"  Prefill tokens/s: {avg(prefill_tps_list):.2f} (avg of {len(prefill_tps_list)} runs)")
if not $PY_PREFILL_ONLY:
    print(f"  Generation tokens/s: {avg(decode_tps_list):.2f} (avg of {len(decode_tps_list)} runs)")
print("  CSV: $RESOLVED_CSV_OUT")
EOF
