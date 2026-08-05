#!/usr/bin/env python3
"""Run sequential vLLM offline benchmarks and write per-run metrics to CSV."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


DEFAULT_PROMPT_TOKENS = (
    "248045,846,198,2427,494,220,16,310,220,16,15,15,15,11,18101,539,"
    "73982,13,248046,198,248045,74455,198,248068,271,248069,271"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/qwen3.5-0.8b"))
    parser.add_argument("--csv-out", type=Path, default=Path("benchmarks/vllm-inference-seq.csv"))
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--prompt-tokens-file", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.warmup_runs < 0:
        raise ValueError("runs must be >= 1 and warmup-runs must be >= 0")

    repo_root = Path(__file__).resolve().parent.parent
    model_path = args.model_path if args.model_path.is_absolute() else repo_root / args.model_path
    csv_out = args.csv_out if args.csv_out.is_absolute() else repo_root / args.csv_out
    prompt_tokens_csv = args.prompt_tokens
    if args.prompt_tokens_file is not None:
        prompt_tokens_file = (
            args.prompt_tokens_file
            if args.prompt_tokens_file.is_absolute()
            else repo_root / args.prompt_tokens_file
        )
        prompt_tokens_csv = prompt_tokens_file.read_text(encoding="utf-8")
    prompt_token_ids = [int(token) for token in prompt_tokens_csv.split(",") if token.strip()]
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
    sampling = SamplingParams(
        temperature=0.0,
        top_p=0.8,
        top_k=0,
        repetition_penalty=1.0,
        seed=123,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
        detokenize=False,
    )

    print(f"Loading vLLM model path={model_path}", flush=True)
    init_started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        runner="generate",
        dtype="bfloat16",
        max_model_len=args.max_context,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        skip_tokenizer_init=True,
        disable_log_stats=False,
        seed=123,
    )
    init_time_ms = (time.perf_counter() - init_started) * 1000.0
    print(f"vLLM initialized in {init_time_ms:.2f} ms", flush=True)

    def run_once():
        started = time.perf_counter()
        request = llm.generate(prompt, sampling, use_tqdm=False)[0]
        wall_time_ms = (time.perf_counter() - started) * 1000.0
        output = request.outputs[0]
        metrics = request.metrics
        if metrics is None:
            raise RuntimeError("vLLM request metrics are unavailable")
        return request, output, metrics, wall_time_ms

    rows: list[dict[str, object]] = []
    for index in range(1, args.warmup_runs + 1):
        _request, output, _metrics, wall_time_ms = run_once()
        print(
            f"Warmup {index}/{args.warmup_runs}: "
            f"tokens={len(output.token_ids)} wall_ms={wall_time_ms:.3f}",
            flush=True,
        )

    for index in range(1, args.runs + 1):
        request, output, metrics, wall_time_ms = run_once()
        output_tokens = len(output.token_ids)
        if output_tokens != args.max_new_tokens:
            raise RuntimeError(f"expected {args.max_new_tokens} output tokens, got {output_tokens}")
        prefill_time_ms = max(metrics.first_token_ts - metrics.scheduled_ts, 0.0) * 1000.0
        decode_time_ms = max(metrics.last_token_ts - metrics.first_token_ts, 0.0) * 1000.0
        queued_time_ms = max(metrics.scheduled_ts - metrics.queued_ts, 0.0) * 1000.0
        prefill_tps = (
            len(prompt_token_ids) * 1000.0 / prefill_time_ms if prefill_time_ms > 0 else 0.0
        )
        # Match qwen35x/Photon's reported convention: all output tokens divided
        # by decode time, even though vLLM generates the first token in prefill.
        decode_tps = output_tokens * 1000.0 / decode_time_ms if decode_time_ms > 0 else 0.0
        standard_decode_tps = (
            (output_tokens - 1) * 1000.0 / decode_time_ms
            if decode_time_ms > 0 and output_tokens > 1
            else 0.0
        )
        rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "engine": "vllm",
                "vllm_version": version("vllm"),
                "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "model_path": str(model_path),
                "run_index": index,
                "warmup_runs": args.warmup_runs,
                "max_new_tokens": args.max_new_tokens,
                "max_context": args.max_context,
                "temperature": 0.0,
                "top_p": 0.8,
                "repeat_penalty": 1.0,
                "input_tokens": len(prompt_token_ids),
                "output_tokens": output_tokens,
                "cached_tokens": request.num_cached_tokens or 0,
                "finish_reason": output.finish_reason,
                "init_time_ms": init_time_ms,
                "queued_time_ms": queued_time_ms,
                "prefill_time_ms": prefill_time_ms,
                "prefill_tokens_per_second": prefill_tps,
                "decode_time_ms": decode_time_ms,
                "tokens_per_second": decode_tps,
                "standard_decode_tokens_per_second": standard_decode_tps,
                "ttft_ms": metrics.first_token_latency * 1000.0,
                "request_compute_time_ms": prefill_time_ms + decode_time_ms,
                "wall_time_ms": wall_time_ms,
            }
        )
        print(
            f"Run {index}/{args.runs}: tokens={output_tokens} "
            f"prefill_tps={prefill_tps:.2f} decode_tps={decode_tps:.2f} "
            f"wall_ms={wall_time_ms:.3f}",
            flush=True,
        )

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    decode_values = [float(row["tokens_per_second"]) for row in rows]
    prefill_values = [float(row["prefill_tokens_per_second"]) for row in rows]
    compute_values = [float(row["request_compute_time_ms"]) for row in rows]
    wall_values = [float(row["wall_time_ms"]) for row in rows]
    print("\nvLLM benchmark summary:")
    print(f"  Prefill tokens/s: {statistics.mean(prefill_values):.2f}")
    print(f"  Generation tokens/s: {statistics.mean(decode_values):.2f}")
    print(f"  Request compute time ms: {statistics.mean(compute_values):.3f}")
    print(f"  Request wall time ms: {statistics.mean(wall_values):.3f}")
    print(f"  CSV: {csv_out}")


if __name__ == "__main__":
    main()
