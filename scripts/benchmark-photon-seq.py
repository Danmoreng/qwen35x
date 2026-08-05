#!/usr/bin/env python3
"""Run sequential Photon benchmarks and write per-run metrics to CSV."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import moondream as md
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-path", type=Path, default=Path("models/qwen3.5-0.8b"))
    parser.add_argument("--csv-out", type=Path, default=Path("benchmarks/photon-inference-seq.csv"))
    parser.add_argument("--prompt", default="Count from 1 to 1000, separated by commas.")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--expected-input-tokens", type=int)
    parser.add_argument("--require-full-output", action="store_true")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Benchmark fixed-length raw decode by disabling Photon skill stop IDs.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--kv-cache-pages", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.warmup_runs < 0:
        raise ValueError("runs must be >= 1 and warmup-runs must be >= 0")

    repo_root = Path(__file__).resolve().parent.parent
    model_path = args.model_path if args.model_path.is_absolute() else repo_root / args.model_path
    csv_out = args.csv_out if args.csv_out.is_absolute() else repo_root / args.csv_out
    prompt_text = args.prompt
    if args.prompt_file is not None:
        prompt_file = args.prompt_file if args.prompt_file.is_absolute() else repo_root / args.prompt_file
        prompt_text = prompt_file.read_text(encoding="utf-8")
    settings = {"temperature": 0.0, "top_p": 0.8, "max_tokens": args.max_new_tokens}
    messages = [{"role": "user", "content": prompt_text}]

    print(f"Loading Photon model={args.model} path={model_path}", flush=True)
    init_started = time.perf_counter()
    model = md.photon(
        args.model,
        model_path=str(model_path),
        max_batch_size=args.max_batch_size,
        kv_cache_pages=args.kv_cache_pages,
        enable_prefix_cache=False,
        enable_cuda_graphs=True,
    )
    init_time_ms = (time.perf_counter() - init_started) * 1000.0
    print(f"Photon initialized in {init_time_ms:.2f} ms", flush=True)

    if args.ignore_eos:
        async def disable_stop_ids() -> None:
            runtime = model._engine._runtimes[model.model_id]
            original = runtime.prompt_template

            class FixedLengthPromptTemplate:
                bos_id = original.bos_id
                eos_id = -1
                answer_id = original.answer_id
                thinking_id = original.thinking_id

                def query(self):
                    template = original.query()
                    return replace(template, stop_token_ids=[]) if template is not None else None

                def chat(self):
                    template = original.chat()
                    return replace(template, turn_end_ids=[]) if template is not None else None

            runtime.prompt_template = FixedLengthPromptTemplate()

        model._run(disable_stop_ids())
        print("Photon stop IDs disabled for fixed-length raw decode", flush=True)

    def run_once():
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = model._run(  # Metrics are not exposed by the public convenience wrapper.
            model._model.chat(
                messages=messages,
                stream=False,
                settings=settings,
                reasoning=False,
            )
        )
        torch.cuda.synchronize()
        wall_time_ms = (time.perf_counter() - started) * 1000.0
        return result, wall_time_ms

    rows: list[dict[str, object]] = []
    try:
        for index in range(1, args.warmup_runs + 1):
            result, wall_time_ms = run_once()
            print(
                f"Warmup {index}/{args.warmup_runs}: "
                f"tokens={result.metrics.output_tokens} wall_ms={wall_time_ms:.3f}",
                flush=True,
            )

        for index in range(1, args.runs + 1):
            result, wall_time_ms = run_once()
            metrics = result.metrics
            if args.expected_input_tokens is not None and metrics.input_tokens != args.expected_input_tokens:
                raise RuntimeError(
                    f"expected {args.expected_input_tokens} input tokens, got {metrics.input_tokens}"
                )
            if args.require_full_output and metrics.output_tokens != args.max_new_tokens:
                raise RuntimeError(
                    f"expected {args.max_new_tokens} output tokens, got {metrics.output_tokens}"
                )
            decode_tps = (
                metrics.output_tokens * 1000.0 / metrics.decode_time_ms
                if metrics.decode_time_ms > 0
                else 0.0
            )
            prefill_tps = (
                metrics.input_tokens * 1000.0 / metrics.prefill_time_ms
                if metrics.prefill_time_ms > 0
                else 0.0
            )
            rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "engine": "photon",
                    "moondream_version": version("moondream"),
                    "kestrel_version": version("kestrel"),
                    "kestrel_kernels_version": version("kestrel-kernels"),
                    "torch_version": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0),
                    "model": args.model,
                    "model_path": str(model_path),
                    "run_index": index,
                    "warmup_runs": args.warmup_runs,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": 0.0,
                    "top_p": 0.8,
                    "input_tokens": metrics.input_tokens,
                    "output_tokens": metrics.output_tokens,
                    "cached_tokens": metrics.cached_tokens,
                    "finish_reason": result.finish_reason,
                    "init_time_ms": init_time_ms,
                    "prefill_time_ms": metrics.prefill_time_ms,
                    "prefill_tokens_per_second": prefill_tps,
                    "decode_time_ms": metrics.decode_time_ms,
                    "tokens_per_second": decode_tps,
                    "ttft_ms": metrics.ttft_ms,
                    "wall_time_ms": wall_time_ms,
                }
            )
            print(
                f"Run {index}/{args.runs}: tokens={metrics.output_tokens} "
                f"prefill_tps={prefill_tps:.2f} decode_tps={decode_tps:.2f} "
                f"wall_ms={wall_time_ms:.3f}",
                flush=True,
            )
    finally:
        model.close()

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    decode_values = [float(row["tokens_per_second"]) for row in rows]
    prefill_values = [float(row["prefill_tokens_per_second"]) for row in rows]
    wall_values = [float(row["wall_time_ms"]) for row in rows]
    print("\nPhoton benchmark summary:")
    print(f"  Prefill tokens/s: {statistics.mean(prefill_values):.2f}")
    print(f"  Generation tokens/s: {statistics.mean(decode_values):.2f}")
    print(f"  Request wall time ms: {statistics.mean(wall_values):.3f}")
    print(f"  CSV: {csv_out}")


if __name__ == "__main__":
    main()
