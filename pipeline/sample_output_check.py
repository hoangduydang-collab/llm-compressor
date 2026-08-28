"""Generate a few tokens from a saved checkpoint and fail on degenerate output.

This is the M3-style sanity check: load the quantized checkpoint, generate, and
look at the text. It exists as a standalone script because the in-run version
(`pipeline/quantize.py::_sample_generation`) is skipped exactly when we need it --
it runs only when `save_checkpoint and not dist_ctx.enabled`, and every GLM-5.2 run
is distributed, so `sample_generation` is off in those configs (also because
offloaded generation on a 743B model gated the save for many minutes during the M3
incident of 2026-07-07).

WHY A CHECK AND NOT JUST A PRINT. The failure this catches is not subtle
degradation, it is collapse: the M3 full-calibration AWQ run produced
`"arringarringarring..."` (BUGS_AND_FIXES.md, "full-calib AWQ garbage output"),
caused by the shared expert being dropped in every MoE layer. That is trivially
detectable, so the script decides rather than leaving it to whoever reads the log:
degenerate repetition and empty output are hard failures.

WHAT IT DOES NOT TELL YOU. Coherent text here means the checkpoint loads and the
forward path is not broken. It says nothing about benchmark quality -- a checkpoint
can produce fluent text and still have lost accuracy. Quality needs the eval
harness, not this.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

DEFAULT_PROMPTS = [
    "Explain in two sentences why the sky appears blue.",
    "Write a Python function that returns the nth Fibonacci number.",
    "What is the capital of France, and what river runs through it?",
]


def degeneracy_report(text: str) -> dict:
    """Cheap detectors for collapsed output.

    The primary metric is ``distinct_4gram_ratio`` -- distinct character 4-grams
    over total. It was chosen by measurement, after a first attempt using
    "share of the most common 4-gram" FAILED on the actual M3 failure string:
    ``"arringarring..."`` has no spaces (so word-level checks never trigger) and
    cycles through only six 4-grams, giving a top share of just 1/6 = 0.17 --
    under any threshold that ordinary prose survives.

    Measured on real samples:

        healthy prose        0.914          repeated word       0.111
        terse valid answer   1.000          repeated sentence   0.051
        code                 0.750          "arring" collapse   0.017
                                            single character    0.005

    Healthy text clusters at >= 0.75 and every degenerate case sits <= 0.111, so
    0.30 separates them with a wide margin in both directions.
    """
    stripped = text.strip()
    words = stripped.split()
    grams = [stripped[i : i + 4] for i in range(max(0, len(stripped) - 3))]
    counts = collections.Counter(grams)
    return {
        "chars": len(stripped),
        "words": len(words),
        "unique_word_ratio": (len(set(words)) / len(words)) if words else 0.0,
        "distinct_4gram_ratio": (len(counts) / len(grams)) if grams else 1.0,
        "top_4gram_share": (
            counts.most_common(1)[0][1] / len(grams) if grams else 0.0
        ),
    }


def judge(report: dict, min_chars: int = 20) -> list[str]:
    """Return a list of failure reasons; empty means the output looks sane."""
    problems = []
    if report["chars"] < min_chars:
        problems.append(f"output too short ({report['chars']} chars < {min_chars})")
    # Only applied to text long enough for the ratio to mean anything; a 20-char
    # correct answer legitimately has a ratio near 1.0 and must not be judged on
    # repetition at all.
    if report["chars"] >= 40 and report["distinct_4gram_ratio"] < 0.30:
        problems.append(
            f"degenerate repetition (distinct-4gram ratio "
            f"{report['distinct_4gram_ratio']:.3f} < 0.30): this is the "
            "'arringarring' collapse signature"
        )
    if report["words"] >= 8 and report["unique_word_ratio"] < 0.15:
        problems.append(
            f"word-level loop (unique-word ratio "
            f"{report['unique_word_ratio']:.2f} < 0.15)"
        )
    if report["top_4gram_share"] > 0.35:
        problems.append(
            f"a single 4-gram is {report['top_4gram_share']:.0%} of the output"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=48,
                        help="kept small: offloaded generation on a 743B model "
                             "streams experts off disk for every token")
    parser.add_argument("--offload-folder", default=None,
                        help="required for a model too large for VRAM+RAM")
    parser.add_argument("--prompt", action="append", default=None,
                        help="repeatable; defaults to three short factual prompts")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args(argv)

    prompts = args.prompt or DEFAULT_PROMPTS

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"==> loading {args.ckpt}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.ckpt), trust_remote_code=args.trust_remote_code
    )
    load_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    if args.offload_folder:
        load_kwargs["offload_folder"] = args.offload_folder
    model = AutoModelForCausalLM.from_pretrained(str(args.ckpt), **load_kwargs)
    model.eval()
    print("==> loaded; generating", flush=True)

    failures = 0
    for index, prompt in enumerate(prompts, start=1):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy: a collapse must not be blamed on sampling
            )
        full = tokenizer.decode(out[0], skip_special_tokens=True)
        completion = full[len(prompt):] if full.startswith(prompt) else full
        report = degeneracy_report(completion)
        problems = judge(report)

        print("\n" + "-" * 70)
        print(f"PROMPT {index}: {prompt}")
        print(f"OUTPUT : {completion!r}")
        print(f"STATS  : {report}")
        if problems:
            failures += 1
            for problem in problems:
                print(f"[FAIL] {problem}")
        else:
            print("[ok]   output looks non-degenerate")

    print("\n" + "=" * 70)
    if failures:
        print(f"RESULT: FAILED on {failures}/{len(prompts)} prompt(s)")
        return 1
    print(f"RESULT: PASSED -- {len(prompts)} prompt(s) produced non-degenerate text")
    print("NOTE: this proves the forward path works, NOT that quality is preserved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
