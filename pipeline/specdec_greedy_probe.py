#!/usr/bin/env python3
"""Greedy-equivalence probe for the EAGLE3 spec-dec A/B (M3_SPECDEC_EAGLE3_PLAN.md).

Speculative decoding with rejection sampling is distribution-preserving, so at
temperature 0 a spec-dec arm should emit the same tokens as the no-spec control.
Small late divergence is expected (different batch shapes change floating-point
reduction order); early divergence is not, and would mean the multi-query verify
path is wrong -- the real risk on M3, whose sparse-attention indexer and cudagraph
capture have only ever run one query position per sequence.

Writes one JSON per arm; ``--compare`` diffs an arm against the control offline.
Stdlib only (runs from any venv on the node).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# Fixed, deterministic prompt set: short prompts (cheap), mixed domains so a
# single lucky match cannot carry the gate.
PROMPTS = [
    "List the first 12 prime numbers, separated by commas.",
    "Explain in three sentences why matrix multiplication is not commutative.",
    "Write a Python function that reverses a linked list in place.",
    "What is 17 * 243? Show the intermediate steps.",
    "Name the seven continents in order of land area, largest first.",
    "Summarize the difference between TCP and UDP in two sentences.",
    "Translate 'the weather is cold today' into French and Spanish.",
    "Given the sequence 2, 6, 12, 20, 30, what is the next term and why?",
]


def one(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        # Match how we benchmark M3: thinking on.
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    msg = payload["choices"][0]["message"]
    return {
        "prompt": prompt,
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "finish_reason": payload["choices"][0].get("finish_reason"),
        "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
    }


def run(args: argparse.Namespace) -> int:
    out = {"arm": args.arm, "base_url": args.base_url, "results": []}
    for prompt in PROMPTS:
        try:
            out["results"].append(one(args.base_url, args.model, prompt,
                                      args.max_tokens, args.timeout))
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            out["results"].append({"prompt": prompt, "error": repr(exc)})
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    errors = sum(1 for r in out["results"] if "error" in r)
    print(json.dumps({"arm": args.arm, "probed": len(PROMPTS), "errors": errors}))
    # A probe that cannot reach the endpoint is a harness failure, not a verdict.
    return 1 if errors == len(PROMPTS) else 0


def compare(args: argparse.Namespace) -> int:
    ref = json.load(open(args.compare))["results"]
    arm = json.load(open(args.against))["results"]
    rows, matched = [], 0
    for i, (a, b) in enumerate(zip(ref, arm)):
        # Reasoning text is the first thing streamed, so compare it when present.
        ta = (a.get("reasoning") or "") + (a.get("content") or "")
        tb = (b.get("reasoning") or "") + (b.get("content") or "")
        # Character prefix as a tokenizer-free proxy for "first 32 tokens".
        pref = 0
        for ca, cb in zip(ta, tb):
            if ca != cb:
                break
            pref += 1
        ok = pref >= args.min_prefix_chars
        matched += ok
        rows.append({"i": i, "prefix_chars": pref, "identical": ta == tb, "pass": ok})
    verdict = {
        "reference": args.compare,
        "arm": args.against,
        "min_prefix_chars": args.min_prefix_chars,
        "passed": matched,
        "of": len(rows),
        "gate": "PASS" if matched >= args.min_pass else "SUSPECT",
        "rows": rows,
    }
    print(json.dumps(verdict, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="MiniMaxAI/MiniMax-M3")
    p.add_argument("--arm", default="unknown")
    p.add_argument("--out", default="greedy-probe.json")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--compare", help="reference (control) probe JSON; enables compare mode")
    p.add_argument("--against", help="arm probe JSON to compare against the reference")
    # ~32 tokens of English is ~120 characters; deliberately conservative.
    p.add_argument("--min-prefix-chars", type=int, default=120)
    p.add_argument("--min-pass", type=int, default=6)
    args = p.parse_args()
    if args.compare:
        if not args.against:
            p.error("--compare needs --against")
        return compare(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
