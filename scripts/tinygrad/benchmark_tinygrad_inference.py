#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tinygrad-root",
        default=str(repo_root / "third_party" / "reference" / "tinygrad"),
        help="Path to tinygrad checkout.",
    )
    parser.add_argument(
        "--model",
        default=str(repo_root / "models" / "gguf" / "qwen3.5-0.8b-bf16.gguf"),
        help="Path to local GGUF model file.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("chat-user", "prompt-text"),
        default="chat-user",
    )
    parser.add_argument("--prompt-text", default="Tell me a short joke.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="")
    parser.add_argument("--stop-on-eos", action="store_true")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.device:
        os.environ["DEV"] = args.device

    tinygrad_root = Path(args.tinygrad_root).resolve()
    if not tinygrad_root.exists():
        raise FileNotFoundError(f"tinygrad root not found: {tinygrad_root}")
    sys.path.insert(0, str(tinygrad_root))

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {model_path}")

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    from tinygrad import Tensor
    from tinygrad.llm.cli import SimpleTokenizer
    from tinygrad.llm.model import Transformer

    Tensor.manual_seed(args.seed)

    load_start = time.perf_counter()
    gguf = Tensor(model_path)
    model, kv = Transformer.from_gguf(gguf, max_context=args.max_context)
    tokenizer = SimpleTokenizer.from_gguf_kv(kv)
    load_time_ms = (time.perf_counter() - load_start) * 1000.0

    if args.prompt_mode == "chat-user":
        prompt_tokens = (
            tokenizer.prefix()
            + tokenizer.role("user")
            + tokenizer.encode(args.prompt_text)
            + tokenizer.end_turn()
            + tokenizer.role("assistant")
        )
    else:
        prompt_tokens = tokenizer.prefix() + tokenizer.encode(args.prompt_text)

    # Keep at least one token slot for generation.
    if len(prompt_tokens) >= args.max_context:
        prompt_tokens = prompt_tokens[-(args.max_context - 1) :]

    decode_start = time.perf_counter()
    generated_tokens: list[int] = []
    token_stream = model.generate(prompt_tokens.copy(), temperature=args.temperature)
    for _ in range(args.max_new_tokens):
        next_id = int(next(token_stream))
        if args.stop_on_eos and tokenizer.is_end(next_id):
            break
        generated_tokens.append(next_id)
    decode_time_ms = (time.perf_counter() - decode_start) * 1000.0

    tokens_per_second = 0.0
    if decode_time_ms > 0.0:
        tokens_per_second = len(generated_tokens) / (decode_time_ms / 1000.0)

    payload = {
        "backend": os.environ.get("DEV", "DEFAULT"),
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": len(generated_tokens),
        "load_time_ms": load_time_ms,
        "decode_time_ms": decode_time_ms,
        "tokens_per_second": tokens_per_second,
        "sampling": {
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "output_token_ids": generated_tokens,
    }

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"tinygrad run complete backend={payload['backend']} "
        f"prompt_tokens={payload['prompt_tokens']} generated_tokens={payload['generated_tokens']} "
        f"tps={payload['tokens_per_second']:.6f}"
    )
    print(f"profile_json={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
