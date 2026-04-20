#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot into a local directory.")
    parser.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dest", required=True, help="Destination directory")
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, falls back to HF_TOKEN/HUGGINGFACE_HUB_TOKEN env vars.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full repo snapshot (default is inference-focused files only).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Install with: python -m pip install huggingface_hub", file=sys.stderr)
        return 2

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    allow_patterns = None
    if not args.full:
        allow_patterns = [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "*.model",
            "model.safetensors",
            "model.safetensors.index.json",
            "model.safetensors-*.safetensors",
            "model-*.safetensors",
        ]

    path = snapshot_download(
        repo_id=args.repo,
        repo_type="model",
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        token=token,
    )

    print(f"download complete: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
