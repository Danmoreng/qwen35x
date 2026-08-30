#!/usr/bin/env bash
set -euo pipefail

# Sequential llama.cpp CPU benchmark for transcript-cleanup workloads.
#
# Each profile produces three measurements with official llama-bench semantics:
#   prefill:              -p INPUT
#   decode_after_input:   -d INPUT -n OUTPUT
#   combined:             -pg INPUT,OUTPUT
#
# llama-bench measures model execution only. Model loading, tokenization,
# chat-template processing, and sampling are intentionally outside the timer.
# OUTPUT is llama-bench's number of timed one-token generation evaluations. A
# production decoder can sample its first output from the final prefill logits,
# so an exact N-token response commonly needs N-1 subsequent decode evaluations.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_BENCH="third_party/reference/llama.cpp/build-cpu-native/bin/llama-bench"
MODEL="models/gguf/Qwen3.5-0.8B-Q8_0.gguf"
OUTPUT_PREFIX="benchmarks/llama-cpu/qwen3.5-0.8b-q8_0-cleanup"
PROFILES="256:256,512:512,1024:1024,2048:2048"
THREADS=4
REPETITIONS=3
BATCH_SIZE=2048
UBATCH_SIZE=512
DELAY_SECONDS=1
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: scripts/benchmark-llama-cleanup-seq.sh [options]

Options:
  --llama-bench PATH       llama-bench binary
  --model PATH             Q8 GGUF model
  --output-prefix PATH     Prefix for -summary.csv, -samples.csv, and -system.txt
  --profiles LIST          Comma-separated INPUT:OUTPUT pairs
                           (default: 256:256,512:512,1024:1024,2048:2048)
  --threads N              CPU inference threads (default: 4 on this host)
  --repetitions N          Timed repetitions per phase (default: 3)
  --batch-size N           Logical prompt batch size (default: 2048)
  --ubatch-size N          Physical prompt micro-batch size (default: 512)
  --delay-seconds N        Pause between sequential phases (default: 1)
  --dry-run                Print the exact phase commands without running them
  -h, --help               Show this help

Examples:
  # Fast validation before the full suite
  scripts/benchmark-llama-cleanup-seq.sh --profiles 256:256

  # Conservative 2k input / 2k output target workload
  scripts/benchmark-llama-cleanup-seq.sh --profiles 2048:2048 --threads 4
EOF
}

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
    --output-prefix)
      OUTPUT_PREFIX="$2"
      shift 2
      ;;
    --profiles)
      PROFILES="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --repetitions)
      REPETITIONS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --ubatch-size)
      UBATCH_SIZE="$2"
      shift 2
      ;;
    --delay-seconds)
      DELAY_SECONDS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
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

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_non_negative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

LLAMA_BENCH="$(resolve_repo_path "$LLAMA_BENCH")"
MODEL="$(resolve_repo_path "$MODEL")"
OUTPUT_PREFIX="$(resolve_repo_path "$OUTPUT_PREFIX")"

if [[ ! "$PROFILES" =~ ^[1-9][0-9]*:[1-9][0-9]*(,[1-9][0-9]*:[1-9][0-9]*)*$ ]]; then
  echo "Invalid --profiles value: $PROFILES (expected INPUT:OUTPUT[,INPUT:OUTPUT...])" >&2
  exit 2
fi

for value_name in THREADS REPETITIONS BATCH_SIZE UBATCH_SIZE; do
  if ! is_positive_integer "${!value_name}"; then
    echo "$value_name must be a positive integer, got: ${!value_name}" >&2
    exit 2
  fi
done

if ! is_non_negative_integer "$DELAY_SECONDS"; then
  echo "DELAY_SECONDS must be a non-negative integer, got: $DELAY_SECONDS" >&2
  exit 2
fi

for required in "$LLAMA_BENCH" "$MODEL"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

COMMON_ARGS=(
  -m "$MODEL"
  -r "$REPETITIONS"
  -t "$THREADS"
  -b "$BATCH_SIZE"
  -ub "$UBATCH_SIZE"
  -ngl 0
  -dev none
  -ctk f16
  -ctv f16
  -fa auto
  -lm mmap
  --progress
  -o jsonl
)

print_command() {
  printf '%q ' "$LLAMA_BENCH" "${COMMON_ARGS[@]}" "$@"
  printf '\n'
}

if [[ "$DRY_RUN" == true ]]; then
  echo "Sequential cleanup benchmark plan"
  IFS=',' read -r -a profile_list <<< "$PROFILES"
  for profile in "${profile_list[@]}"; do
    input_tokens="${profile%%:*}"
    output_tokens="${profile##*:}"
    echo "profile=${input_tokens}in/${output_tokens}out phase=prefill"
    print_command -p "$input_tokens" -n 0
    echo "profile=${input_tokens}in/${output_tokens}out phase=decode_after_input"
    print_command -p 0 -n "$output_tokens" -d "$input_tokens"
    echo "profile=${input_tokens}in/${output_tokens}out phase=combined"
    print_command -p 0 -n 0 -pg "${input_tokens},${output_tokens}"
  done
  exit 0
fi

SUMMARY_CSV="${OUTPUT_PREFIX}-summary.csv"
SAMPLES_CSV="${OUTPUT_PREFIX}-samples.csv"
SYSTEM_OUT="${OUTPUT_PREFIX}-system.txt"
mkdir -p "$(dirname "$OUTPUT_PREFIX")"

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwen35x-llama-cleanup.XXXXXX")"
cleanup() {
  if [[ -n "${TEMP_DIR:-}" && -d "$TEMP_DIR" ]]; then
    rm -r -- "$TEMP_DIR"
  fi
}
trap cleanup EXIT

MANIFEST="$TEMP_DIR/manifest.tsv"
: > "$MANIFEST"

LLAMA_CPP_DIR="$(git -C "$(dirname "$LLAMA_BENCH")" rev-parse --show-toplevel 2>/dev/null || true)"
{
  printf 'timestamp: '
  date --iso-8601=seconds
  uname -a
  lscpu
  printf '\nllama.cpp directory: %s\n' "${LLAMA_CPP_DIR:-unknown}"
  if [[ -n "$LLAMA_CPP_DIR" ]]; then
    printf 'llama.cpp commit: '
    git -C "$LLAMA_CPP_DIR" rev-parse HEAD
    printf 'llama.cpp description: '
    git -C "$LLAMA_CPP_DIR" describe --always --dirty
  fi
  printf 'model sha256: '
  sha256sum "$MODEL"
  printf 'profiles: %s\n' "$PROFILES"
  printf 'threads: %s\n' "$THREADS"
  printf 'repetitions: %s\n' "$REPETITIONS"
  printf 'batch size: %s\n' "$BATCH_SIZE"
  printf 'ubatch size: %s\n' "$UBATCH_SIZE"
  printf 'timer boundary: llama-bench model execution only; excludes model loading, tokenization, sampling, and chat templates\n'
  printf 'output accounting: OUTPUT is n_gen (timed one-token evaluations); production generation may need OUTPUT-1 evaluations after prefill\n'
  printf '\nphase commands:\n'
  IFS=',' read -r -a metadata_profiles <<< "$PROFILES"
  for profile in "${metadata_profiles[@]}"; do
    input_tokens="${profile%%:*}"
    output_tokens="${profile##*:}"
    print_command -p "$input_tokens" -n 0
    print_command -p 0 -n "$output_tokens" -d "$input_tokens"
    print_command -p 0 -n 0 -pg "${input_tokens},${output_tokens}"
  done
} > "$SYSTEM_OUT"

run_phase() {
  local scenario="$1"
  local phase="$2"
  local input_tokens="$3"
  local output_tokens="$4"
  shift 4

  local result_jsonl="$TEMP_DIR/${scenario}-${phase}.jsonl"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$scenario" "$phase" "$input_tokens" "$output_tokens" "$result_jsonl" >> "$MANIFEST"

  echo "Running scenario=$scenario phase=$phase (${input_tokens} input, ${output_tokens} output)"
  "$LLAMA_BENCH" "${COMMON_ARGS[@]}" "$@" > "$result_jsonl"

  if [[ ! -s "$result_jsonl" ]]; then
    echo "llama-bench produced no result for scenario=$scenario phase=$phase" >&2
    exit 1
  fi
}

IFS=',' read -r -a profile_list <<< "$PROFILES"
for profile in "${profile_list[@]}"; do
  input_tokens="${profile%%:*}"
  output_tokens="${profile##*:}"
  scenario="in${input_tokens}_out${output_tokens}"

  run_phase "$scenario" prefill "$input_tokens" "$output_tokens" \
    -p "$input_tokens" -n 0
  if (( DELAY_SECONDS > 0 )); then sleep "$DELAY_SECONDS"; fi

  run_phase "$scenario" decode_after_input "$input_tokens" "$output_tokens" \
    -p 0 -n "$output_tokens" -d "$input_tokens"
  if (( DELAY_SECONDS > 0 )); then sleep "$DELAY_SECONDS"; fi

  run_phase "$scenario" combined "$input_tokens" "$output_tokens" \
    -p 0 -n 0 -pg "${input_tokens},${output_tokens}"
  if (( DELAY_SECONDS > 0 )); then sleep "$DELAY_SECONDS"; fi
done

python3 - "$MANIFEST" "$SUMMARY_CSV" "$SAMPLES_CSV" <<'PY'
import csv
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
samples_path = pathlib.Path(sys.argv[3])

summary_fields = [
    "scenario",
    "phase",
    "input_tokens",
    "output_tokens",
    "context_depth",
    "measured_prompt_tokens",
    "measured_generation_tokens",
    "rate_basis_tokens",
    "threads",
    "batch_size",
    "ubatch_size",
    "repetitions",
    "avg_ns",
    "stddev_ns",
    "avg_seconds",
    "stddev_seconds",
    "avg_tokens_per_second",
    "stddev_tokens_per_second",
    "build_commit",
    "model_filename",
]

sample_fields = [
    "scenario",
    "phase",
    "repetition",
    "input_tokens",
    "output_tokens",
    "context_depth",
    "measured_prompt_tokens",
    "measured_generation_tokens",
    "rate_basis_tokens",
    "threads",
    "elapsed_ns",
    "elapsed_seconds",
    "tokens_per_second",
]

summary_rows = []
sample_rows = []

with manifest_path.open(encoding="utf-8") as manifest:
    for line_number, line in enumerate(manifest, start=1):
        scenario, phase, input_text, output_text, jsonl_text = line.rstrip("\n").split("\t")
        input_tokens = int(input_text)
        output_tokens = int(output_text)
        records = []
        with pathlib.Path(jsonl_text).open(encoding="utf-8") as result_file:
            for result_line in result_file:
                if result_line.strip():
                    records.append(json.loads(result_line))

        if len(records) != 1:
            raise RuntimeError(
                f"Expected exactly one llama-bench row for manifest line {line_number}, got {len(records)}"
            )

        record = records[0]
        measured_prompt = int(record["n_prompt"])
        measured_generation = int(record["n_gen"])
        context_depth = int(record["n_depth"])
        rate_basis_tokens = measured_prompt + measured_generation
        samples_ns = [int(value) for value in record["samples_ns"]]
        samples_ts = [float(value) for value in record["samples_ts"]]

        if len(samples_ns) != len(samples_ts):
            raise RuntimeError(f"Mismatched sample arrays for {scenario}/{phase}")

        summary_rows.append(
            {
                "scenario": scenario,
                "phase": phase,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "context_depth": context_depth,
                "measured_prompt_tokens": measured_prompt,
                "measured_generation_tokens": measured_generation,
                "rate_basis_tokens": rate_basis_tokens,
                "threads": record["n_threads"],
                "batch_size": record["n_batch"],
                "ubatch_size": record["n_ubatch"],
                "repetitions": len(samples_ns),
                "avg_ns": record["avg_ns"],
                "stddev_ns": record["stddev_ns"],
                "avg_seconds": float(record["avg_ns"]) / 1e9,
                "stddev_seconds": float(record["stddev_ns"]) / 1e9,
                "avg_tokens_per_second": record["avg_ts"],
                "stddev_tokens_per_second": record["stddev_ts"],
                "build_commit": record["build_commit"],
                "model_filename": record["model_filename"],
            }
        )

        for repetition, (elapsed_ns, tokens_per_second) in enumerate(
            zip(samples_ns, samples_ts), start=1
        ):
            sample_rows.append(
                {
                    "scenario": scenario,
                    "phase": phase,
                    "repetition": repetition,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "context_depth": context_depth,
                    "measured_prompt_tokens": measured_prompt,
                    "measured_generation_tokens": measured_generation,
                    "rate_basis_tokens": rate_basis_tokens,
                    "threads": record["n_threads"],
                    "elapsed_ns": elapsed_ns,
                    "elapsed_seconds": elapsed_ns / 1e9,
                    "tokens_per_second": tokens_per_second,
                }
            )

with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
    writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

with samples_path.open("w", newline="", encoding="utf-8") as samples_file:
    writer = csv.DictWriter(samples_file, fieldnames=sample_fields)
    writer.writeheader()
    writer.writerows(sample_rows)

print("\nCleanup benchmark summary (model compute only):")
print(f"{'scenario':<20} {'phase':<20} {'seconds':>10} {'tokens/s':>12}")
for row in summary_rows:
    print(
        f"{row['scenario']:<20} {row['phase']:<20} "
        f"{row['avg_seconds']:>10.3f} {float(row['avg_tokens_per_second']):>12.2f}"
    )
PY

echo "Summary CSV written to: $SUMMARY_CSV"
echo "Per-repetition CSV written to: $SAMPLES_CSV"
echo "System metadata written to: $SYSTEM_OUT"
