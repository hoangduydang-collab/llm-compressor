"""Load a checkpoint (or slice) in SGLang and record a deterministic probe.

Two jobs in one script, deliberately:

  1. Answer "does the engine load this artifact at all", which for a converted
     checkpoint is the first thing worth knowing and needs no reference.
  2. Emit a comparable JSON artifact -- generated token ids plus per-position
     logprobs -- so two runs can be diffed numerically.

BOTH ARMS MUST RUN THIS SAME FILE. The comparison it feeds is only meaningful if
the sampling parameters, prompt text, tokenizer handling and logprob extraction
are bit-identical between arms; two scripts that "do the same thing" are how a
difference in the harness gets attributed to the model. The only thing that
differs between arms is ``--model`` and ``--quantization``.

WHY A FILE AND NOT A HEREDOC. SGLang spawns its scheduler with multiprocessing
"spawn", which re-imports ``__main__``. Fed on stdin, ``__main__.__file__`` is
``<stdin>`` and every child dies with FileNotFoundError before a single weight
is read -- observed 2026-08-30. Hence a real module with a ``__main__`` guard.

WHAT A COMPARISON OF TWO PROBES CAN AND CANNOT SHOW. Run against slices of the
same layers, one BF16 and one quantized, the diff measures end-to-end
quantization error through the real engine and needs no reference
implementation of the model's forward pass. But a truncated model's activations
never passed through the missing layers, so the absolute values are meaningless
and only the BF16-vs-quantized COMPARISON carries information. Perplexity and
text quality require the full model.

Usage:
    python -m pipeline.sglang_load_probe --model <dir> --out probe.json \\
        [--quantization w4afp8] [--max-new-tokens 8] [--top-logprobs 5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Fixed, deterministic, and deliberately dull. Content does not matter -- the
# probe compares two engines on identical input -- but it must be IDENTICAL
# across arms and across time, so it is hard-coded rather than parameterised.
PROMPTS = (
    "The capital of France is",
    "def add(a, b):",
    "1, 2, 3, 4,",
)


def run(
    model: Path,
    out: Path,
    quantization: str | None = None,
    max_new_tokens: int = 8,
    top_logprobs: int = 5,
    tp_size: int = 1,
    mem_fraction: float = 0.6,
) -> int:
    import sglang as sgl

    record: dict = {
        "model": str(model),
        "quantization": quantization,
        "sglang_version": getattr(sgl, "__version__", "unknown"),
        "prompts": list(PROMPTS),
        "max_new_tokens": max_new_tokens,
        "top_logprobs": top_logprobs,
        "tp_size": tp_size,
        "status": "starting",
    }

    def save() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    save()
    engine = None
    try:
        kwargs: dict = {
            "model_path": str(model),
            "tp_size": tp_size,
            "mem_fraction_static": mem_fraction,
            "trust_remote_code": True,
            # Off because graph capture is a separate failure surface and this
            # probe is about weight loading and numerics, not throughput.
            "disable_cuda_graph": True,
            "log_level": "info",
        }
        if quantization:
            kwargs["quantization"] = quantization
        engine = sgl.Engine(**kwargs)
        record["status"] = "engine_up"
        save()
        print("ENGINE UP", flush=True)

        outputs = engine.generate(
            list(PROMPTS),
            {
                # Greedy: any sampling randomness would make two arms differ for
                # reasons that have nothing to do with the weights.
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
            },
            return_logprob=True,
            top_logprobs_num=top_logprobs,
        )
        record["outputs"] = _normalize(outputs)
        record["status"] = "ok"
        save()
        print(f"GENERATE OK: {len(record['outputs'])} result(s)", flush=True)
        for item in record["outputs"]:
            print(f"  text={item.get('text')!r} "
                  f"tokens={item.get('output_ids')}", flush=True)
        return 0
    except Exception:
        record["status"] = "error"
        record["traceback"] = traceback.format_exc()
        save()
        traceback.print_exc()
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:  # noqa: BLE001 - shutdown noise must not mask the result
                pass


def _normalize(outputs) -> list[dict]:
    """Reduce SGLang's output to the JSON-safe fields a comparison needs.

    Kept deliberately narrow: text, output ids and the per-position top
    logprobs. Storing the whole meta_info would bake in engine-version details
    that differ between runs for reasons unrelated to the weights, and a
    comparison over noisy fields is a comparison nobody trusts.
    """
    normalized = []
    for item in outputs if isinstance(outputs, list) else [outputs]:
        meta = item.get("meta_info", {}) if isinstance(item, dict) else {}
        entry = {
            "text": item.get("text") if isinstance(item, dict) else None,
            "output_ids": meta.get("output_token_ids")
            or (item.get("output_ids") if isinstance(item, dict) else None),
            "output_top_logprobs": _clean_logprobs(
                meta.get("output_top_logprobs")
            ),
            "input_top_logprobs": _clean_logprobs(
                meta.get("input_top_logprobs")
            ),
        }
        normalized.append(entry)
    return normalized


def _clean_logprobs(raw):
    """SGLang returns [[(logprob, token_id, token_text), ...], ...]."""
    if not raw:
        return None
    cleaned = []
    for position in raw:
        if position is None:
            cleaned.append(None)
            continue
        cleaned.append([
            [float(entry[0]), int(entry[1])]
            for entry in position
            if entry is not None and entry[0] is not None and entry[1] is not None
        ])
    return cleaned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--quantization", default=None,
                        help="engine quantization name, e.g. w4afp8; omit for "
                             "an unquantized checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction", type=float, default=0.6)
    args = parser.parse_args(argv)

    code = run(
        args.model, args.out,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
        top_logprobs=args.top_logprobs,
        tp_size=args.tp_size,
        mem_fraction=args.mem_fraction,
    )
    print(f"PROBE RESULT: {'PASS' if code == 0 else 'FAIL'} -> {args.out}",
          flush=True)
    return code


if __name__ == "__main__":
    # Guard matters: without it SGLang's spawned scheduler re-executes this
    # module's top level and recursively starts engines.
    sys.exit(main())
