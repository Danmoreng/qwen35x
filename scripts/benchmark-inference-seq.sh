#!/usr/bin/env bash
set -euo pipefail

# Sequential inference benchmark for Linux. Each warmup and measured run starts
# a fresh process; measured runs append one row each to the requested CSV.

usage() {
    cat <<'EOF'
Usage: scripts/benchmark-inference-seq.sh [options]

Options:
  --executable PATH                 Executable relative to the repository (default: build/qwen35x)
  --hf-model-dir PATH               Hugging Face model directory (default: models/qwen3.5-0.8b)
  --csv-out PATH                    CSV output path (default: benchmarks/qwen35x-inference-seq.csv)
  --run-label TEXT                  Label recorded in the CSV
  --mode MODE                       gpu-f32, gpu-bf16, nvfp4, or cpu-reference (default: gpu-f32)
  --runs N                          Number of measured runs (default: 3)
  --warmup-runs N                   Number of warmup runs (default: 1)
  --max-new-tokens N                Number of generated tokens (default: 128)
  --max-context N                   Maximum context length (default: 256)
  --prompt-tokens CSV               Token IDs used as the prompt (default: 198)
  --prompt-text TEXT                Plain-text prompt
  --prompt-file PATH                Prompt file relative to the repository
  --chat-user TEXT                  Chat prompt
  --gpu-decode-blocks N             Override decode grid size
  --qwen35x-decode-execution MODE   megakernel, multi-kernel, or graph (default: megakernel)
  --temperature N                   Sampling temperature (default: 0)
  --top-p N                         Top-p value (default: 0.8)
  --top-k N                         Top-k value (default: 20)
  --repeat-penalty N                Repeat penalty (default: 1.05)
  --seed N                          Random seed (default: 123)
  --prefill-only                    Do not generate decode tokens
  --profile-sync                    Request synchronized CUDA stage timings
  --qwen35x-profile                 Keep detailed Qwen35x profile data
  --keep-profiles                   Keep per-run profile JSON files
  --profile-dir PATH                Profile directory (default: build/bench-profiles)
  -h, --help                        Show this help
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

executable="build/qwen35x"
hf_model_dir="models/qwen3.5-0.8b"
csv_out="benchmarks/qwen35x-inference-seq.csv"
run_label=""
mode="gpu-f32"
runs=3
warmup_runs=1
max_new_tokens=128
max_context=256
prompt_kind="prompt-tokens"
prompt_value="198"
gpu_decode_blocks=0
decode_execution="megakernel"
temperature=0
top_p=0.8
top_k=20
repeat_penalty=1.05
seed=123
prefill_only=false
profile_sync=false
qwen35x_profile=false
keep_profiles=false
profile_dir="build/bench-profiles"

repo_path() {
    if [[ "$1" = /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s/%s\n' "$repo_root" "$1"
    fi
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --executable) require_value "$@"; executable="$2"; shift 2 ;;
        --hf-model-dir) require_value "$@"; hf_model_dir="$2"; shift 2 ;;
        --csv-out) require_value "$@"; csv_out="$2"; shift 2 ;;
        --run-label) require_value "$@"; run_label="$2"; shift 2 ;;
        --mode) require_value "$@"; mode="$2"; shift 2 ;;
        --runs) require_value "$@"; runs="$2"; shift 2 ;;
        --warmup-runs) require_value "$@"; warmup_runs="$2"; shift 2 ;;
        --max-new-tokens) require_value "$@"; max_new_tokens="$2"; shift 2 ;;
        --max-context) require_value "$@"; max_context="$2"; shift 2 ;;
        --prompt-tokens) require_value "$@"; prompt_kind="prompt-tokens"; prompt_value="$2"; shift 2 ;;
        --prompt-text) require_value "$@"; prompt_kind="prompt-text"; prompt_value="$2"; shift 2 ;;
        --prompt-file) require_value "$@"; prompt_kind="prompt-file"; prompt_value="$2"; shift 2 ;;
        --chat-user) require_value "$@"; prompt_kind="chat-user"; prompt_value="$2"; shift 2 ;;
        --gpu-decode-blocks) require_value "$@"; gpu_decode_blocks="$2"; shift 2 ;;
        --qwen35x-decode-execution) require_value "$@"; decode_execution="$2"; shift 2 ;;
        --temperature) require_value "$@"; temperature="$2"; shift 2 ;;
        --top-p) require_value "$@"; top_p="$2"; shift 2 ;;
        --top-k) require_value "$@"; top_k="$2"; shift 2 ;;
        --repeat-penalty) require_value "$@"; repeat_penalty="$2"; shift 2 ;;
        --seed) require_value "$@"; seed="$2"; shift 2 ;;
        --prefill-only) prefill_only=true; shift ;;
        --profile-sync) profile_sync=true; shift ;;
        --qwen35x-profile) qwen35x_profile=true; shift ;;
        --keep-profiles) keep_profiles=true; shift ;;
        --profile-dir) require_value "$@"; profile_dir="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$mode" in gpu-f32|gpu-bf16|nvfp4|cpu-reference) ;; *) echo "Unsupported mode: $mode" >&2; exit 2 ;; esac
case "$decode_execution" in megakernel|multi-kernel|graph) ;; *) echo "Unsupported decode execution: $decode_execution" >&2; exit 2 ;; esac
[[ "$runs" =~ ^[1-9][0-9]*$ ]] || { echo "--runs must be >= 1" >&2; exit 2; }
[[ "$warmup_runs" =~ ^[0-9]+$ ]] || { echo "--warmup-runs must be >= 0" >&2; exit 2; }
[[ "$gpu_decode_blocks" =~ ^[0-9]+$ ]] || { echo "--gpu-decode-blocks must be >= 0" >&2; exit 2; }

resolved_executable="$(repo_path "$executable")"
resolved_model_dir="$(repo_path "$hf_model_dir")"
resolved_csv_out="$(repo_path "$csv_out")"
resolved_profile_dir="$(repo_path "$profile_dir")"
if [[ "$prompt_kind" == "prompt-file" ]]; then
    prompt_value="$(repo_path "$prompt_value")"
fi

[[ -x "$resolved_executable" ]] || { echo "Executable not found or not executable: $resolved_executable" >&2; exit 1; }
[[ -d "$resolved_model_dir" ]] || { echo "Model directory not found: $resolved_model_dir" >&2; exit 1; }
[[ "$prompt_kind" != "prompt-file" || -f "$prompt_value" ]] || { echo "Prompt file not found: $prompt_value" >&2; exit 1; }
mkdir -p "$(dirname "$resolved_csv_out")" "$resolved_profile_dir"

record_csv_row() {
    local profile_path="$1"
    local run_index="$2"
    local profile_path_for_csv=""
    if [[ "$keep_profiles" == true || "$qwen35x_profile" == true ]]; then
        profile_path_for_csv="$profile_path"
    fi
    python3 - "$profile_path" "$resolved_csv_out" "$run_label" "$mode" "$decode_execution" "$gpu_decode_blocks" "$run_index" "$prompt_kind" "$profile_path_for_csv" <<'PY'
import csv
import json
import os
import sys
from datetime import datetime, timezone

profile_path, csv_path, run_label, mode, requested_execution, requested_blocks, run_index, prompt_kind, profile_path_for_csv = sys.argv[1:]
with open(profile_path, encoding="utf-8") as profile_file:
    profile = json.load(profile_file)

qwen_profile = profile.get("qwen35x_profile") or profile.get("luce_profile") or {}
decode_profile = qwen_profile.get("decode") or {}
row = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "run_label": run_label,
    "mode": mode,
    "qwen35x_decode_execution": profile.get("qwen35x_decode_execution", requested_execution),
    "qwen35x_decode_blocks_requested": requested_blocks,
    "prefill_only": profile.get("prefill_only", False),
    "run_index": run_index,
    "prompt_kind": prompt_kind,
    "prompt_tokens": profile.get("prompt_tokens", ""),
    "generated_tokens": profile.get("generated_tokens", ""),
    "load_time_ms": profile.get("load_time_ms", ""),
    "prefill_time_ms": profile.get("prefill_time_ms", ""),
    "prefill_tokens_per_second": profile.get("prefill_tokens_per_second", ""),
    "decode_time_ms": profile.get("decode_time_ms", ""),
    "tokens_per_second": profile.get("tokens_per_second", ""),
    "profile_json": profile_path_for_csv,
    "qwen35x_decode_graph_node_count": decode_profile.get("graph_node_count", ""),
    "qwen35x_decode_blocks": decode_profile.get("decode_blocks", ""),
    "qwen35x_decode_max_safe_blocks": decode_profile.get("max_safe_decode_blocks", ""),
    "qwen35x_decode_graph_launch_ms": decode_profile.get("graph_launch_ms", ""),
    "qwen35x_decode_kernel_ms": decode_profile.get("decode_kernel_ms", ""),
}
new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=row.keys())
    if new_file:
        writer.writeheader()
    writer.writerow(row)
print(row["tokens_per_second"])
PY
}

run_once() {
    local run_type="$1"
    local run_index="$2"
    local profile_path="$resolved_profile_dir/${run_type}_${mode}_${run_index}_$(date +%s%N).json"
    local args=(
        --hf-model-dir "$resolved_model_dir"
        --max-new-tokens "$max_new_tokens"
        --max-context "$max_context"
        --temperature "$temperature"
        --top-p "$top_p"
        --top-k "$top_k"
        --repeat-penalty "$repeat_penalty"
        --seed "$seed"
        --profile-json "$profile_path"
        "--$prompt_kind" "$prompt_value"
    )

    case "$mode" in
        gpu-f32) args+=(--infer-gpu --gpu-f32-matvec) ;;
        gpu-bf16) args+=(--infer-gpu --gpu-bf16) ;;
        nvfp4) args+=(--infer-gpu --qwen35x-weight-precision nvfp4) ;;
        cpu-reference) args+=(--infer-reference) ;;
    esac
    if [[ "$mode" != "cpu-reference" && "$decode_execution" != "megakernel" ]]; then args+=(--qwen35x-decode-execution "$decode_execution"); fi
    if (( gpu_decode_blocks > 0 )); then args+=(--gpu-decode-blocks "$gpu_decode_blocks"); fi
    [[ "$prefill_only" == true ]] && args+=(--prefill-only)
    [[ "$profile_sync" == true ]] && args+=(--profile-sync)
    [[ "$qwen35x_profile" == true ]] && args+=(--qwen35x-profile)

    "$resolved_executable" "${args[@]}" >/dev/null
    [[ -f "$profile_path" ]] || { echo "Missing profile JSON output: $profile_path" >&2; exit 1; }
    printf '%s\n' "$profile_path"
}

echo "Sequential benchmark start: mode=$mode, decode_execution=$decode_execution, warmup=$warmup_runs, runs=$runs"
echo "CSV output: $resolved_csv_out"
for ((run_index = 1; run_index <= warmup_runs; run_index++)); do
    warmup_profile="$(run_once warmup "$run_index")"
    echo "Warmup completed: $run_index/$warmup_runs"
    [[ "$keep_profiles" == true ]] || rm -f "$warmup_profile"
done

for ((run_index = 1; run_index <= runs; run_index++)); do
    profile_path="$(run_once run "$run_index")"
    tokens_per_second="$(record_csv_row "$profile_path" "$run_index")"
    echo "Recorded: mode=$mode run=$run_index/$runs tps=$tokens_per_second"
    if [[ "$keep_profiles" != true && "$qwen35x_profile" != true ]]; then rm -f "$profile_path"; fi
done

echo "Benchmark complete. CSV written to: $resolved_csv_out"
