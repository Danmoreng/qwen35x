#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run deterministic HF Transformers inference for qwen35x parity checks.")
    parser.add_argument("--model-dir", default=str(repo_root / "models" / "qwen3.5-0.8b"))
    parser.add_argument("--prompt-mode", choices=("chat-user", "prompt-text", "prompt-tokens"), default="chat-user")
    parser.add_argument("--prompt-text", default="Tell me a short joke.")
    parser.add_argument("--prompt-tokens", default="")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repeat-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--model-auto-class", choices=("auto", "causal-lm", "image-text-to-text"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--top-logits", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def parse_token_csv(csv: str) -> list[int]:
    tokens: list[int] = []
    for part in csv.split(","):
        stripped = part.strip()
        if stripped:
            tokens.append(int(stripped))
    if not tokens:
        raise ValueError("prompt token list is empty")
    return tokens


def build_prompt_text(mode: str, text: str) -> str:
    if mode == "chat-user":
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    return text


def torch_dtype_from_name(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def apply_repetition_penalty(logits: Any, seen_token_ids: set[int], repeat_penalty: float, torch: Any) -> Any:
    if repeat_penalty <= 1.0 or not seen_token_ids:
        return logits
    ids = torch.tensor(sorted(seen_token_ids), device=logits.device, dtype=torch.long)
    selected = logits.index_select(0, ids)
    adjusted = torch.where(selected > 0.0, selected / repeat_penalty, selected * repeat_penalty)
    out = logits.clone()
    out.index_copy_(0, ids, adjusted)
    return out


def top_logits_payload(logits: Any, k: int, torch: Any) -> list[dict[str, float | int]]:
    if k <= 0:
        return []
    count = min(k, int(logits.numel()))
    values, indices = torch.topk(logits.detach().float().cpu(), count)
    return [
        {"token_id": int(indices[i].item()), "logit": float(values[i].item())}
        for i in range(count)
    ]


def model_auto_class_candidates(transformers: Any, name: str) -> list[tuple[str, Any]]:
    if name == "causal-lm":
        return [("causal-lm", transformers.AutoModelForCausalLM)]
    if name == "image-text-to-text":
        cls = getattr(transformers, "AutoModelForImageTextToText", None)
        return [("image-text-to-text", cls)] if cls is not None else []

    candidates: list[tuple[str, Any]] = [("causal-lm", transformers.AutoModelForCausalLM)]
    image_text_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if image_text_cls is not None:
        candidates.append(("image-text-to-text", image_text_cls))
    return candidates


def load_model_from_candidates(
    transformers: Any,
    model_dir: Path,
    args: argparse.Namespace,
    torch_dtype: Any,
    local_files_only: bool,
) -> tuple[str, Any]:
    errors: list[str] = []
    candidates = model_auto_class_candidates(transformers, args.model_auto_class)
    if not candidates:
        raise RuntimeError(f"no available model auto class candidates for {args.model_auto_class}")

    for class_name, auto_class in candidates:
        try:
            model = auto_class.from_pretrained(
                str(model_dir),
                torch_dtype=torch_dtype,
                trust_remote_code=args.trust_remote_code,
                local_files_only=local_files_only,
            )
            return class_name, model
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")

    raise RuntimeError("\n".join(errors))


def main() -> int:
    args = parse_args()
    if args.temperature != 0.0:
        print("Transformers parity runner currently supports only greedy temperature=0.", file=sys.stderr)
        return 2
    if args.max_new_tokens <= 0:
        print("max-new-tokens must be > 0.", file=sys.stderr)
        return 2
    if args.repeat_penalty < 1.0:
        print("repeat-penalty must be >= 1.0.", file=sys.stderr)
        return 2

    try:
        import torch
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(
            "Missing PyTorch/Transformers dependency. Run: "
            "pwsh -File scripts/setup-transformers-parity.ps1",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 3

    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"model directory not found: {model_dir}", file=sys.stderr)
        return 4

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    local_files_only = not args.allow_download
    load_start = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=args.trust_remote_code,
            local_files_only=local_files_only,
        )
        loaded_auto_class, model = load_model_from_candidates(
            transformers,
            model_dir,
            args,
            torch_dtype_from_name(torch, args.dtype),
            local_files_only,
        )
    except Exception as exc:
        print("Failed to load model with HF Transformers.", file=sys.stderr)
        print("This Qwen3.5 snapshot may require a source/dev Transformers build.", file=sys.stderr)
        print("Install/update the optional parity environment with: pwsh -File scripts/setup-transformers-parity.ps1", file=sys.stderr)
        print("For model repos with custom modeling code, rerun with --trust-remote-code and a full local snapshot.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 5

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA device requested, but torch.cuda.is_available() is false.", file=sys.stderr)
        return 6

    model.to(device)
    model.eval()
    load_time_ms = (time.perf_counter() - load_start) * 1000.0

    if args.prompt_mode == "prompt-tokens":
        prompt_token_ids = parse_token_csv(args.prompt_tokens)
        prompt_text = ""
    else:
        prompt_text = build_prompt_text(args.prompt_mode, args.prompt_text)
        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    if not prompt_token_ids:
        print("prompt produced zero tokens.", file=sys.stderr)
        return 7
    if len(prompt_token_ids) > args.max_context:
        print(
            f"prompt token count {len(prompt_token_ids)} exceeds max_context {args.max_context}.",
            file=sys.stderr,
        )
        return 7

    seen_token_ids = set(int(token_id) for token_id in prompt_token_ids)
    generated_tokens: list[int] = []
    top_logits_by_step: list[dict[str, Any]] = []

    def run_full_context_forward() -> Any:
        ids = prompt_token_ids + generated_tokens
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        return model(input_ids=input_ids, use_cache=False)

    with torch.inference_mode():
        prefill_start = time.perf_counter()
        if args.no_cache:
            outputs = run_full_context_forward()
            past_key_values = None
            attention_mask = None
        else:
            input_ids = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = getattr(outputs, "past_key_values", None)
        prefill_time_ms = (time.perf_counter() - prefill_start) * 1000.0

        decode_start = time.perf_counter()
        for step in range(args.max_new_tokens):
            logits = outputs.logits[0, -1, :].float()
            sampled_logits = apply_repetition_penalty(logits, seen_token_ids, args.repeat_penalty, torch)
            next_token = int(torch.argmax(sampled_logits).item())
            generated_tokens.append(next_token)
            seen_token_ids.add(next_token)

            if args.top_logits > 0:
                top_logits_by_step.append(
                    {
                        "step": step,
                        "selected_token_id": next_token,
                        "top_logits": top_logits_payload(sampled_logits, args.top_logits, torch),
                    }
                )

            if step + 1 >= args.max_new_tokens:
                break

            if args.no_cache:
                outputs = run_full_context_forward()
            else:
                next_input = torch.tensor([[next_token]], device=device, dtype=torch.long)
                if attention_mask is not None:
                    next_mask = torch.ones((1, 1), device=device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, next_mask], dim=1)
                outputs = model(
                    input_ids=next_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = getattr(outputs, "past_key_values", None)
        decode_time_ms = (time.perf_counter() - decode_start) * 1000.0

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    tokens_per_second = 0.0
    if decode_time_ms > 0.0:
        tokens_per_second = len(generated_tokens) / (decode_time_ms / 1000.0)

    payload: dict[str, Any] = {
        "backend": "hf-transformers",
        "device": device,
        "dtype": args.dtype,
        "model_auto_class": loaded_auto_class,
        "model_dir": str(model_dir),
        "prompt_mode": args.prompt_mode,
        "prompt_text": prompt_text,
        "prompt_tokens": len(prompt_token_ids),
        "prompt_token_ids": prompt_token_ids,
        "generated_tokens": len(generated_tokens),
        "output_token_ids": generated_tokens,
        "generated_text": generated_text,
        "load_time_ms": load_time_ms,
        "prefill_time_ms": prefill_time_ms,
        "decode_time_ms": decode_time_ms,
        "tokens_per_second": tokens_per_second,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repeat_penalty": args.repeat_penalty,
            "seed": args.seed,
        },
    }
    if top_logits_by_step:
        payload["top_logits_by_step"] = top_logits_by_step

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"hf-transformers run complete device={device} prompt_tokens={len(prompt_token_ids)} "
        f"generated_tokens={len(generated_tokens)} tps={tokens_per_second:.6f}"
    )
    print(f"profile_json={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
