#!/usr/bin/env python3
"""Validate and combine qwen35x, Photon, and vLLM text benchmark results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(rows: list[dict[str, object]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qwen-profile-dir", type=Path, required=True)
    parser.add_argument("--photon-csv", type=Path, required=True)
    parser.add_argument("--vllm-csv", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))

    def resolve(path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else repo_root / value

    prompt_file = resolve(config["prompt_file"])
    tokens_file = resolve(config["prompt_tokens_file"])
    model_dir = resolve(config["model_dir"])
    if sha256(prompt_file) != config["prompt_sha256"]:
        raise RuntimeError(f"prompt checksum mismatch: {prompt_file}")
    if sha256(tokens_file) != config["prompt_tokens_sha256"]:
        raise RuntimeError(f"prompt-token checksum mismatch: {tokens_file}")
    tokenizer_file = model_dir / "tokenizer.json"
    weights_file = model_dir / "model.safetensors-00001-of-00001.safetensors"
    if sha256(tokenizer_file) != config["tokenizer_sha256"]:
        raise RuntimeError(f"tokenizer checksum mismatch: {tokenizer_file}")
    if sha256(weights_file) != config["weights_sha256"]:
        raise RuntimeError(f"weight checksum mismatch: {weights_file}")
    prompt_ids = [int(value) for value in tokens_file.read_text().split(",") if value.strip()]
    if len(prompt_ids) != config["prompt_tokens"]:
        raise RuntimeError(f"expected {config['prompt_tokens']} prompt tokens, got {len(prompt_ids)}")

    qwen_profile_dir = resolve(args.qwen_profile_dir)
    photon_csv = resolve(args.photon_csv)
    vllm_csv = resolve(args.vllm_csv)
    rows: list[dict[str, object]] = []
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()

    for run_index in range(1, int(config["measured_runs"]) + 1):
        profile_path = qwen_profile_dir / f"run_{run_index}.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "engine": "qwen35x",
                "version": commit,
                "run_index": run_index,
                "input_tokens": profile["prompt_tokens"],
                "output_tokens": profile["generated_tokens"],
                "prefill_time_ms": profile["prefill_time_ms"],
                "decode_time_ms": profile["decode_time_ms"],
                "startup_or_load_time_ms": profile["load_time_ms"],
            }
        )

    with photon_csv.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            rows.append(
                {
                    "engine": "photon",
                    "version": (
                        f"moondream {source['moondream_version']} / kestrel {source['kestrel_version']} "
                        f"/ kernels {source['kestrel_kernels_version']}"
                    ),
                    "run_index": int(source["run_index"]),
                    "input_tokens": int(source["input_tokens"]),
                    "output_tokens": int(source["output_tokens"]),
                    "prefill_time_ms": float(source["prefill_time_ms"]),
                    "decode_time_ms": float(source["decode_time_ms"]),
                    "startup_or_load_time_ms": float(source["init_time_ms"]),
                }
            )

    with vllm_csv.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            rows.append(
                {
                    "engine": "vllm",
                    "version": source["vllm_version"],
                    "run_index": int(source["run_index"]),
                    "input_tokens": int(source["input_tokens"]),
                    "output_tokens": int(source["output_tokens"]),
                    "prefill_time_ms": float(source["prefill_time_ms"]),
                    "decode_time_ms": float(source["decode_time_ms"]),
                    "startup_or_load_time_ms": float(source["init_time_ms"]),
                }
            )

    expected_runs = int(config["measured_runs"])
    for engine in ("qwen35x", "photon", "vllm"):
        engine_rows = [row for row in rows if row["engine"] == engine]
        if len(engine_rows) != expected_runs:
            raise RuntimeError(f"expected {expected_runs} {engine} rows, got {len(engine_rows)}")
        for row in engine_rows:
            if row["input_tokens"] != config["prompt_tokens"]:
                raise RuntimeError(f"{engine} input-token mismatch: {row['input_tokens']}")
            if row["output_tokens"] != config["max_new_tokens"]:
                raise RuntimeError(f"{engine} output-token mismatch: {row['output_tokens']}")

    for row in rows:
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        prefill_ms = float(row["prefill_time_ms"])
        decode_ms = float(row["decode_time_ms"])
        row["prefill_tokens_per_second"] = input_tokens * 1000.0 / prefill_ms
        row["reported_decode_tokens_per_second"] = output_tokens * 1000.0 / decode_ms
        row["steady_state_decode_tokens"] = max(output_tokens - 1, 0)
        row["steady_state_decode_tokens_per_second"] = max(output_tokens - 1, 0) * 1000.0 / decode_ms
        row["model_time_ms"] = prefill_ms + decode_ms

    csv_out = resolve(args.csv_out)
    summary_out = resolve(args.summary_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "config": config,
        "results": {},
    }
    for engine in ("qwen35x", "photon", "vllm"):
        engine_rows = [row for row in rows if row["engine"] == engine]
        summary["results"][engine] = {
            "prefill_tokens_per_second_mean": mean(engine_rows, "prefill_tokens_per_second"),
            "steady_state_decode_tokens_per_second_mean": mean(
                engine_rows, "steady_state_decode_tokens_per_second"
            ),
            "reported_decode_tokens_per_second_mean": mean(
                engine_rows, "reported_decode_tokens_per_second"
            ),
            "model_time_ms_mean": mean(engine_rows, "model_time_ms"),
            "startup_or_load_time_ms_mean": mean(engine_rows, "startup_or_load_time_ms"),
        }
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Combined CSV: {csv_out}")
    print(f"Summary JSON: {summary_out}")
    for engine, result in summary["results"].items():
        print(
            f"{engine}: prefill={result['prefill_tokens_per_second_mean']:.2f} tok/s "
            f"decode={result['steady_state_decode_tokens_per_second_mean']:.2f} tok/s "
            f"model_time={result['model_time_ms_mean']:.2f} ms"
        )


if __name__ == "__main__":
    main()
