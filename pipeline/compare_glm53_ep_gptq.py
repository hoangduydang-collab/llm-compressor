"""Compare representative EP checkpoints with the offloaded DDP4 control."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np


DEFAULT_PROMPT = (
    "Explain why deterministic collective ordering matters when expert owners "
    "publish GPTQ parameters across distributed ranks."
)


def compare_logit_arrays(
    reference: np.ndarray,
    candidates: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> dict:
    errors = []
    details = {}
    reference = np.asarray(reference)
    if not np.isfinite(reference).all():
        errors.append("DDP4 reference logits contain non-finite values")
    for label, raw in candidates.items():
        value = np.asarray(raw)
        finite = bool(np.isfinite(value).all())
        same_shape = value.shape == reference.shape
        max_abs = (
            float(np.max(np.abs(value - reference)))
            if finite and same_shape and value.size
            else None
        )
        close = (
            finite
            and same_shape
            and bool(np.allclose(value, reference, rtol=rtol, atol=atol))
        )
        details[label] = {
            "shape": list(value.shape),
            "finite": finite,
            "max_abs_error": max_abs,
            "close": close,
        }
        if not finite:
            errors.append(f"{label} logits contain non-finite values")
        elif not same_shape:
            errors.append(
                f"{label} logits shape differs: {value.shape} != {reference.shape}"
            )
        elif not close:
            errors.append(
                f"{label} logits differ from DDP4 at rtol={rtol} atol={atol}"
            )
    return {
        "ok": not errors,
        "rtol": rtol,
        "atol": atol,
        "errors": errors,
        "reference_shape": list(reference.shape),
        "candidates": details,
    }


def _last_token_logits(checkpoint: Path, tokenizer_path: Path, prompt: str) -> np.ndarray:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        add_special_tokens=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    inputs = {name: value.to("cuda:0") for name, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :].float().cpu().numpy()
    del model, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return logits


def _candidate(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--candidate expects LABEL=CHECKPOINT")
    label, path = raw.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--candidate expects LABEL=CHECKPOINT")
    return label, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=_candidate, action="append", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    reference = _last_token_logits(args.reference, args.tokenizer, args.prompt)
    candidates = {
        label: _last_token_logits(path, args.tokenizer, args.prompt)
        for label, path in args.candidate
    }
    report = compare_logit_arrays(
        reference, candidates, rtol=args.rtol, atol=args.atol
    )
    report.update(
        {
            "reference": str(args.reference),
            "candidate_paths": {
                label: str(path) for label, path in args.candidate
            },
            "prompt": args.prompt,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
