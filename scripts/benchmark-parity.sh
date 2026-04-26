#!/bin/bash
set -e

# Default values
EXECUTABLE="build/qwen35x"
HF_MODEL_DIR="models/qwen3.5-0.8b"
PROMPTS_FILE="scripts/bench/parity_prompts_minimal.txt"
CSV_OUT="benchmarks/qwen35x-parity.csv"
RUN_LABEL="parity"
MAX_NEW_TOKENS=4
MAX_CONTEXT=256
TEMPERATURE=0.0
SEED=123
GPU_MODE="gpu-f32"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --exe)
      EXECUTABLE="$2"
      shift 2
      ;;
    --model)
      HF_MODEL_DIR="$2"
      shift 2
      ;;
    --prompts)
      PROMPTS_FILE="$2"
      shift 2
      ;;
    --csv)
      CSV_OUT="$2"
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
RESOLVED_PROMPTS="$PROMPTS_FILE"
[[ "$RESOLVED_PROMPTS" = /* ]] || RESOLVED_PROMPTS="$REPO_ROOT/$RESOLVED_PROMPTS"
RESOLVED_CSV_OUT="$CSV_OUT"
[[ "$RESOLVED_CSV_OUT" = /* ]] || RESOLVED_CSV_OUT="$REPO_ROOT/$RESOLVED_CSV_OUT"

PROFILE_TMP_DIR="$REPO_ROOT/build/parity-profiles"
mkdir -p "$PROFILE_TMP_DIR"
mkdir -p "$(dirname "$RESOLVED_CSV_OUT")"

echo "Parity run start: prompts=$RESOLVED_PROMPTS"

cat <<EOF > "$PROFILE_TMP_DIR/parity_helper.py"
import json
import subprocess
import os
import sys
import csv
from datetime import datetime

def run_inference(exe, mode, model_dir, prompt_mode, prompt_text, max_new_tokens, max_context, temperature, seed, profile_path):
    args = [exe]
    if mode == "cpu":
        args.append("--infer-reference")
    else:
        args.extend(["--infer-gpu", "--gpu-f32-matvec"])
    
    args.extend([
        "--hf-model-dir", model_dir,
        "--max-new-tokens", str(max_new_tokens),
        "--max-context", str(max_context),
        "--temperature", str(temperature),
        "--seed", str(seed),
        "--profile-json", profile_path
    ])
    
    if prompt_mode == "chat-user":
        args.extend(["--chat-user", prompt_text])
    else:
        args.extend(["--prompt-text", prompt_text])
        
    print(f"Running {mode} inference for: {prompt_text[:30]}...")
    subprocess.run(args, check=True, capture_output=True)
    
    with open(profile_path, 'r') as f:
        return json.load(f)

def main():
    exe = sys.argv[1]
    model_dir = sys.argv[2]
    prompts_file = sys.argv[3]
    csv_out = sys.argv[4]
    profile_tmp_dir = sys.argv[5]
    
    with open(prompts_file, 'r') as f:
        prompts = [json.loads(line) for line in f if line.strip() and not line.startswith('#')]
        
    results = []
    for i, p in enumerate(prompts):
        name = p['name']
        mode = p['mode']
        text = p['text']
        
        cpu_profile_path = os.path.join(profile_tmp_dir, f"cpu_{i}.json")
        gpu_profile_path = os.path.join(profile_tmp_dir, f"gpu_{i}.json")
        
        cpu_profile = run_inference(exe, "cpu", model_dir, mode, text, $MAX_NEW_TOKENS, $MAX_CONTEXT, $TEMPERATURE, $SEED, cpu_profile_path)
        gpu_profile = run_inference(exe, "gpu", model_dir, mode, text, $MAX_NEW_TOKENS, $MAX_CONTEXT, $TEMPERATURE, $SEED, gpu_profile_path)
        
        cpu_tokens = cpu_profile.get('output_token_ids', [])
        gpu_tokens = gpu_profile.get('output_token_ids', [])
        
        match = cpu_tokens == gpu_tokens
        print(f"  Prompt '{name}': {'PASS' if match else 'FAIL'}")
        
        results.append({
            'timestamp_utc': datetime.utcnow().isoformat(),
            'prompt_name': name,
            'token_parity_pass': str(match).lower(),
            'cpu_tokens': str(cpu_tokens),
            'gpu_tokens': str(gpu_tokens)
        })
        
    with open(csv_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()
EOF

python3 "$PROFILE_TMP_DIR/parity_helper.py" "$RESOLVED_EXE" "$RESOLVED_MODEL_DIR" "$RESOLVED_PROMPTS" "$RESOLVED_CSV_OUT" "$PROFILE_TMP_DIR"

echo "Parity run complete. CSV written to $RESOLVED_CSV_OUT"
