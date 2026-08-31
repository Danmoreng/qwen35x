#!/usr/bin/env python3
"""Run deterministic transcript rewrites sequentially across GGUF formats."""

import argparse
import csv
import json
import pathlib
import re
import subprocess
import time


DEFAULT_QUANTS = "Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_0,IQ4_NL,IQ4_XS"


def parse_args():
    repo = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", type=pathlib.Path,
                        default=repo / "third_party/reference/llama.cpp/build-cpu-native/bin/llama-cli")
    parser.add_argument("--model-dir", type=pathlib.Path,
                        default=repo / "models/gguf/bartowski-f36b1ea")
    parser.add_argument("--cases", type=pathlib.Path,
                        default=repo / "scripts/data/transcript-cleanup-rewrite-cases.json")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=repo / "benchmarks/llama-quality/quant-rewrites")
    parser.add_argument("--quants", default=DEFAULT_QUANTS)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--context", type=int, default=1024)
    return parser.parse_args()


def extract_assistant(path):
    text = path.read_text(encoding="utf-8")
    marker = "Assistant:\n"
    return text.split(marker, 1)[1].strip() if marker in text else text.strip()


def main():
    args = parse_args()
    if args.threads < 1 or args.max_tokens < 1 or args.context < 1:
        raise SystemExit("threads, max-tokens, and context must be positive")
    for path in (args.llama_cli, args.cases):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")

    suite = json.loads(args.cases.read_text(encoding="utf-8"))
    quants = [item.strip() for item in args.quants.split(",") if item.strip()]
    models = {
        quant: args.model_dir / f"Qwen_Qwen3.5-0.8B-{quant}.gguf"
        for quant in quants
    }
    for model in models.values():
        if not model.is_file():
            raise SystemExit(f"Missing model: {model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for quant, model in models.items():
        for case in suite["cases"]:
            case_dir = args.output_dir / quant / case["id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = case_dir / "conversation.txt"
            stdout_path = case_dir / "stdout.txt"
            stderr_path = case_dir / "stderr.txt"
            command = [
                str(args.llama_cli), "-m", str(model),
                "-sys", suite["system_prompt"], "-p", case["input"],
                "-n", str(args.max_tokens), "-c", str(args.context),
                "-t", str(args.threads), "-tb", str(args.threads),
                "-ngl", "0", "-dev", "none", "-ctk", "f16", "-ctv", "f16",
                "--temp", "0", "--seed", "123", "--reasoning", "off",
                "--single-turn", "--no-display-prompt", "--simple-io",
                "--no-warmup", "--perf", "--output", str(transcript_path),
            ]
            print(f"Quality evaluation: quant={quant} case={case['id']}", flush=True)
            started = time.monotonic()
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            wall_seconds = time.monotonic() - started
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(
                    f"llama-cli failed for {quant}/{case['id']} with exit code "
                    f"{completed.returncode}; see {stderr_path}"
                )

            output = extract_assistant(transcript_path)
            required_results = [bool(re.search(pattern, output)) for pattern in case["required_regex"]]
            forbidden_results = [bool(re.search(pattern, output)) for pattern in case["forbidden_regex"]]
            timing_text = completed.stdout + "\n" + completed.stderr
            prompt_rate = re.search(r"Prompt:\s*([0-9.]+)\s*t/s", timing_text)
            generation_rate = re.search(r"Generation:\s*([0-9.]+)\s*t/s", timing_text)
            rows.append({
                "quant": quant,
                "case": case["id"],
                "critical_facts_pass": all(required_results) and not any(forbidden_results),
                "required_passed": sum(required_results),
                "required_total": len(required_results),
                "forbidden_hits": sum(forbidden_results),
                "rewrite_quality_1_to_5": "",
                "unsupported_additions": "",
                "wall_seconds": f"{wall_seconds:.6f}",
                "prompt_tokens_per_second": prompt_rate.group(1) if prompt_rate else "",
                "generation_tokens_per_second": generation_rate.group(1) if generation_rate else "",
                "output": output,
                "artifact": str(transcript_path),
            })

    fields = list(rows[0])
    csv_path = args.output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report_path = args.output_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("# llama.cpp quantized transcript rewrite evaluation\n\n")
        report.write(f"System prompt: `{suite['system_prompt']}`\n\n")
        for row in rows:
            report.write(f"## {row['quant']} — {row['case']}\n\n")
            report.write(
                f"Automated critical facts: **{row['critical_facts_pass']}** "
                f"({row['required_passed']}/{row['required_total']} required, "
                f"{row['forbidden_hits']} forbidden hits)  \n"
            )
            report.write(
                f"Wall: {row['wall_seconds']} s; prompt: {row['prompt_tokens_per_second']} t/s; "
                f"generation: {row['generation_tokens_per_second']} t/s\n\n"
            )
            report.write("```text\n" + row["output"] + "\n```\n\n")

    print(f"CSV written to: {csv_path}")
    print(f"Review report written to: {report_path}")


if __name__ == "__main__":
    main()
