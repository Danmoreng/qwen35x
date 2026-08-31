#!/usr/bin/env bash
set -euo pipefail

# Compare GGUF weight formats without running models concurrently. Each round
# reverses the model order to reduce first/last thermal bias. The existing
# cleanup harness remains the single source of llama-bench invocation details.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="models/gguf/bartowski-f36b1ea"
OUTPUT_DIR="benchmarks/llama-cpu/quant-sweep"
QUANTS="Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_0,IQ4_NL,IQ4_XS"
PROFILES="1:128,256:128,2048:128"
THREADS=6
ROUNDS=3
BATCH_SIZE=2048
UBATCH_SIZE=512
DELAY_SECONDS=1

usage() {
  cat <<'EOF'
Usage: scripts/benchmark-llama-quant-sweep.sh [options]

Options:
  --model-dir PATH        Directory containing Bartowski GGUF files
  --output-dir PATH       Benchmark artifact directory
  --quants LIST           Comma-separated quant names
  --profiles LIST         Comma-separated INPUT:OUTPUT pairs
  --threads N             CPU threads (default: 6)
  --rounds N              Alternating-order rounds (default: 3)
  --batch-size N          Logical prompt batch size (default: 2048)
  --ubatch-size N         Physical prompt batch size (default: 512)
  --delay-seconds N       Pause between phases (default: 1)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --quants) QUANTS="$2"; shift 2 ;;
    --profiles) PROFILES="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --ubatch-size) UBATCH_SIZE="$2"; shift 2 ;;
    --delay-seconds) DELAY_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

resolve_repo_path() {
  if [[ "$1" = /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$REPO_ROOT" "$1"; fi
}

MODEL_DIR="$(resolve_repo_path "$MODEL_DIR")"
OUTPUT_DIR="$(resolve_repo_path "$OUTPUT_DIR")"
HARNESS="$REPO_ROOT/scripts/benchmark-llama-cleanup-seq.sh"

for value_name in THREADS ROUNDS BATCH_SIZE UBATCH_SIZE; do
  if [[ ! "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer, got: ${!value_name}" >&2
    exit 2
  fi
done
if [[ ! "$DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "DELAY_SECONDS must be a non-negative integer, got: $DELAY_SECONDS" >&2
  exit 2
fi

IFS=',' read -r -a quant_list <<< "$QUANTS"
if (( ${#quant_list[@]} == 0 )); then
  echo "At least one quant is required" >&2
  exit 2
fi
for quant in "${quant_list[@]}"; do
  model="$MODEL_DIR/Qwen_Qwen3.5-0.8B-${quant}.gguf"
  if [[ ! -f "$model" ]]; then
    echo "Missing model: $model" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/model-manifest.txt"
{
  printf 'timestamp: '; date --iso-8601=seconds
  printf 'model source: bartowski/Qwen_Qwen3.5-0.8B-GGUF\n'
  printf 'model revision: f36b1ea49a332ede8fe5f389bbf5b3575ef71f48\n'
  printf 'quants: %s\nprofiles: %s\nthreads: %s\nrounds: %s\n' \
    "$QUANTS" "$PROFILES" "$THREADS" "$ROUNDS"
  for quant in "${quant_list[@]}"; do
    model="$MODEL_DIR/Qwen_Qwen3.5-0.8B-${quant}.gguf"
    stat --printf='%n\t%s bytes\n' "$model"
    sha256sum "$model"
  done
} > "$MANIFEST"

for (( round=1; round<=ROUNDS; ++round )); do
  order=("${quant_list[@]}")
  if (( round % 2 == 0 )); then
    reversed=()
    for (( index=${#order[@]}-1; index>=0; --index )); do reversed+=("${order[index]}"); done
    order=("${reversed[@]}")
  fi

  for quant in "${order[@]}"; do
    model="$MODEL_DIR/Qwen_Qwen3.5-0.8B-${quant}.gguf"
    prefix="$OUTPUT_DIR/round-${round}/${quant}"
    echo "Quant sweep: round=$round/$ROUNDS quant=$quant"
    "$HARNESS" \
      --model "$model" \
      --output-prefix "$prefix" \
      --profiles "$PROFILES" \
      --threads "$THREADS" \
      --repetitions 1 \
      --batch-size "$BATCH_SIZE" \
      --ubatch-size "$UBATCH_SIZE" \
      --delay-seconds "$DELAY_SECONDS"
  done
done

python3 - "$OUTPUT_DIR" "$ROUNDS" "${quant_list[@]}" <<'PY'
import csv
import pathlib
import statistics
import sys

output_dir = pathlib.Path(sys.argv[1])
rounds = int(sys.argv[2])
quants = sys.argv[3:]
samples = []

for round_number in range(1, rounds + 1):
    for quant in quants:
        path = output_dir / f"round-{round_number}" / f"{quant}-samples.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["quant"] = quant
                row["round"] = round_number
                samples.append(row)

sample_fields = ["quant", "round"] + [key for key in samples[0] if key not in {"quant", "round"}]
with (output_dir / "all-samples.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=sample_fields)
    writer.writeheader()
    writer.writerows(samples)

groups = {}
for row in samples:
    key = (row["quant"], row["scenario"], row["phase"])
    groups.setdefault(key, []).append(float(row["tokens_per_second"]))

summary_rows = []
for (quant, scenario, phase), values in groups.items():
    summary_rows.append({
        "quant": quant,
        "scenario": scenario,
        "phase": phase,
        "samples": len(values),
        "mean_tokens_per_second": statistics.fmean(values),
        "median_tokens_per_second": statistics.median(values),
        "min_tokens_per_second": min(values),
        "max_tokens_per_second": max(values),
        "stddev_tokens_per_second": statistics.stdev(values) if len(values) > 1 else 0.0,
    })

fields = list(summary_rows[0])
with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary_rows)

print("\nQuant sweep medians (tokens/s):")
for row in summary_rows:
    print(f"{row['quant']:<8} {row['scenario']:<18} {row['phase']:<20} "
          f"{row['median_tokens_per_second']:>10.2f}")
PY

echo "Combined samples written to: $OUTPUT_DIR/all-samples.csv"
echo "Aggregate summary written to: $OUTPUT_DIR/summary.csv"
