#!/bin/bash
set -e

# Simplified benchmark script for Linux
# usage: ./scripts/benchmark-inference-seq.sh --mode gpu-f32 --max-new-tokens 128 ...

EXECUTABLE="build/qwen35x"
HF_MODEL_DIR="models/qwen3.5-0.8b-nvfp4"
RUNS=3
WARMUP_RUNS=1
MAX_NEW_TOKENS=128
MAX_CONTEXT=256
PROMPT_TOKENS="198" # Default to single token prompt
PREFILL_ONLY=false
MODE="gpu-f32"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
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
    --prefill-only)
      PREFILL_ONLY=true
      shift
      ;;
    --prompt-tokens)
      PROMPT_TOKENS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOLVED_EXE="$REPO_ROOT/$EXECUTABLE"
RESOLVED_MODEL_DIR="$REPO_ROOT/$HF_MODEL_DIR"
BUILD_DIR="$REPO_ROOT/build"
BENCH_DIR="$BUILD_DIR/bench-profiles"
mkdir -p "$BENCH_DIR"

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

profiles = [ "$(echo ${PROFILES[@]} | sed 's/ /", "/g')" ]
if not profiles:
    sys.exit(0)

prefill_tps_list = []
decode_tps_list = []

for p_path in profiles:
    if not p_path: continue
    if not os.path.exists(p_path): continue
    with open(p_path, 'r') as f:
        data = json.load(f)
        prefill_tps_list.append(data.get('prefill_tokens_per_second', 0))
        decode_tps_list.append(data.get('tokens_per_second', 0))

def avg(l):
    return sum(l) / len(l) if l else 0

print("\nBenchmark Results Summary:")
print(f"  Prefill tokens/s: {avg(prefill_tps_list):.2f} (avg of {len(prefill_tps_list)} runs)")
if not $PY_PREFILL_ONLY:
    print(f"  Generation tokens/s: {avg(decode_tps_list):.2f} (avg of {len(decode_tps_list)} runs)")
EOF
