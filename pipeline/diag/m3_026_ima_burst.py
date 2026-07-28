"""Minimal faithful reproducer for the vLLM 0.26.0 MiniMax-M3 conc-10 IMA.

The crash signature we are chasing (see docs/m3-dspark-blockers-026.md):

    total_num_scheduled_tokens=97, num_scheduled_tokens={req_a: 1, req_b: 96}
    req_b: prompt_token_ids_len=8288, num_computed_tokens=8192

i.e. a MIXED batch -- one 1-token decode alongside a 96-token chunked-prefill
tail. The 8192-token prefix hit is not luck: the benchmark arm runs
``--warmup-request-count 10`` over the same prompts it then measures, so the
measured pass re-sends prompts whose blocks are already in the prefix cache and
only a sub-block remainder needs prefilling. That remainder lands in the same
batch as another request's decode.

This script reproduces exactly that with no aiperf dependency:

  phase "warm"  -- fire the first N prompts concurrently, drain them (this
                   populates the prefix cache, block-aligned)
  phase "burst" -- fire the SAME N prompts concurrently again; the sub-block
                   remainders now co-schedule with in-flight decodes

Exit status is the evidence, and it is fail-closed:
  0  both phases completed every request AND /health still answers
  1  a request failed, or the engine stopped answering /health (the IMA)
  2  could not even reach the server

Deliberately does NOT trust HTTP 200 alone: it checks /health after each phase,
because a dead engine still lets the API server return errors with a rc that
looks fine (cf. the aiperf exit-code trap).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def load_prompts(path: str, n: int) -> list[str]:
    out: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line)["text"])
            if len(out) == n:
                break
    if len(out) < n:
        raise SystemExit(f"FATAL: only {len(out)} prompts in {path}, need {n}")
    return out


def health(base: str, timeout: float = 10.0) -> bool:
    try:
        return requests.get(f"{base}/health", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def one(base: str, model: str, prompt: str, max_tokens: int, idx: int) -> dict:
    t0 = time.time()
    try:
        r = requests.post(
            f"{base}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "stream": False,
            },
            timeout=600,
        )
    except requests.RequestException as exc:
        return {"i": idx, "ok": False, "why": f"{type(exc).__name__}: {exc}"}
    dt = time.time() - t0
    if r.status_code != 200:
        return {
            "i": idx,
            "ok": False,
            "why": f"HTTP {r.status_code}: {r.text[:200]}",
            "s": round(dt, 1),
        }
    body = r.json()
    usage = body.get("usage") or {}
    return {
        "i": idx,
        "ok": True,
        "s": round(dt, 1),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
    }


def phase(name: str, base: str, model: str, prompts: list[str], max_tokens: int) -> bool:
    print(f"\n--- phase {name}: {len(prompts)} concurrent requests ---", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        results = list(
            pool.map(
                lambda a: one(base, model, a[1], max_tokens, a[0]),
                enumerate(prompts),
            )
        )
    wall = time.time() - t0
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    for r in sorted(results, key=lambda r: r["i"]):
        if r["ok"]:
            print(
                f"  req{r['i']:02d} OK   {r['s']:>6}s  prompt={r['prompt_tokens']} "
                f"cached={r['cached_tokens']} out={r['completion_tokens']}",
                flush=True,
            )
        else:
            print(f"  req{r['i']:02d} FAIL {r['why']}", flush=True)
    alive = health(base)
    print(
        f"  phase {name}: {len(ok)}/{len(prompts)} completed in {wall:.1f}s, "
        f"engine_alive={alive}",
        flush=True,
    )
    return not bad and alive


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. http://localhost:8120")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="speedbench 8k-low.jsonl")
    ap.add_argument("-n", type=int, default=10, help="concurrency == request count")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    base = args.base.rstrip("/")
    if not health(base):
        print(f"FATAL: {base}/health does not answer before we start")
        return 2

    prompts = load_prompts(args.prompts, args.n)
    print(f"loaded {len(prompts)} prompts, max_tokens={args.max_tokens}", flush=True)

    warm_ok = phase("warm", base, args.model, prompts, args.max_tokens)
    if not warm_ok:
        print("\nRESULT: FAILED ALREADY IN WARM PHASE (no prefix-cache reuse yet)")
        return 1

    burst_ok = phase("burst", base, args.model, prompts, args.max_tokens)
    if not burst_ok:
        print("\nRESULT: FAILED IN BURST PHASE -- reproduced (mixed prefill+decode)")
        return 1

    print("\nRESULT: CLEAN -- both phases completed, engine alive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
