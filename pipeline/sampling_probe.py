#!/usr/bin/env python3
"""Sampling-sensitivity probe for the AWQ GPQA non-termination phenomenon.

Question: the tok64k run found AWQ arms non-terminate on GPQA far more than
BF16/GPTQ (in-house AWQ 38.9%, cyankiwi 55.6% budget-exhausted at 64k, vs
~12% for the healthy arms). Those runs used GREEDY decoding (temp=0, the
suite's §7 convention). Reasoning models are known to loop under greedy
(DeepSeek-R1 model card; MiniMax recommends temp 1.0 / top_p 0.95). This
probe re-runs ONLY the exhausted GPQA docs (plus a small terminated control)
against a live endpoint under BOTH regimes at a FIXED budget, so greedy vs
sampling is apples-to-apples:

  * greedy  : temperature 0.0, top_p 1.0            (1 generation / doc)
  * sampled : temperature 1.0, top_p 0.95           (N generations / doc)

Per generation we record finish_reason, completion_tokens, the reasoning +
visible text, a repetition signature (zlib compression ratio of the reasoning
tail — degenerate loops compress to ~0.004), and a best-effort GPQA answer
letter for approximate correctness. The headline metric is the per-arm
non-termination RATE (finish_reason == 'length'); if sampling collapses AWQ's
rate toward the healthy arms, the tok64k regression was largely a greedy x
quant interaction rather than pure recipe damage.

Doc source: the tok64k samples_*.jsonl for one arm. A doc is "exhausted" if
its stored visible response is empty (the same definition used in the tok64k
per-item analysis). Prompts are reconstructed verbatim from the stored
`arguments`, so the probe hits the identical questions.

Env / CLI (all overridable):
  SAMPLES_GLOB  glob for the arm's tok64k samples files (latest wins)
  BASE_URL      e.g. http://localhost:8004
  SERVED_NAME   model id to send (MiniMaxAI/MiniMax-M3)
  OUT           output jsonl
  N_SAMPLES     sampled generations per doc            (default 5)
  MAX_TOKENS    generation budget per generation       (default 32768)
  N_CONTROL     terminated docs to include as control  (default 25)
  N_EXHAUSTED   cap exhausted docs (0 = all; evenly    (default 0)
                spaced by doc_id when the cap bites)
  CONCURRENCY   in-flight requests                      (default 24)
  REQUEST_TIMEOUT_S                                     (default 3600)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

_TS_RE = re.compile(r"samples_.+_(\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")
_ANS_RE = re.compile(r"answer\s*(?:is|:)?\s*\(?\s*([ABCD])\s*\)?", re.IGNORECASE)
_PAREN_RE = re.compile(r"\(\s*([ABCD])\s*\)")


def _latest(glob_pat: str) -> str:
    best, best_ts = None, ""
    for p in glob.glob(glob_pat, recursive=True):
        m = _TS_RE.search(os.path.basename(p))
        ts = m.group(1) if m else ""
        if best is None or ts > best_ts:
            best, best_ts = p, ts
    if best is None:
        raise SystemExit(f"no samples files match {glob_pat}")
    return best


def _prompt_text(row: dict) -> str:
    """Reconstruct the user prompt from lm-eval stored arguments."""
    arg0 = row["arguments"]["gen_args_0"]["arg_0"]
    raw = arg0[0] if isinstance(arg0, list) else arg0
    try:
        parts = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw if isinstance(raw, str) else str(raw)
    if isinstance(parts, list):
        return "".join(p.get("content", "") if isinstance(p, dict) else str(p)
                        for p in parts)
    return str(parts)


def _target_letter(row: dict):
    t = row.get("target") or ""
    m = _PAREN_RE.search(t) or re.search(r"([ABCD])", t)
    return m.group(1).upper() if m else None


def _resp_empty(row: dict) -> bool:
    resp = row.get("resps") or [[""]]
    text = resp[0][0] if resp and resp[0] else ""
    return text.strip() == ""


def _extract_answer(content: str):
    if not content:
        return None
    ms = list(_ANS_RE.finditer(content))
    if ms:
        return ms[-1].group(1).upper()
    ms = list(_PAREN_RE.finditer(content))
    return ms[-1].group(1).upper() if ms else None


def _rep_ratio(text: str):
    tail = (text or "")[-8000:]
    if len(tail) < 200:
        return None
    b = tail.encode("utf-8", "ignore")
    return round(len(zlib.compress(b, 6)) / max(1, len(b)), 4)


def _load_docs(samples_path: str):
    docs = {}
    with open(samples_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            did = row["doc_id"]
            docs[did] = {
                "doc_id": did,
                "prompt": _prompt_text(row),
                "target": _target_letter(row),
                "exhausted": _resp_empty(row),
            }
    return docs


def _generate(base_url, served_name, prompt, temperature, top_p, max_tokens, timeout):
    body = {
        "model": served_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    r = requests.post(base_url.rstrip("/") + "/v1/chat/completions",
                      json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    msg = ch.get("message", {}) or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    usage = j.get("usage", {}) or {}
    return {
        "finish_reason": ch.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "reasoning_rep_ratio": _rep_ratio(reasoning),
        "answer": _extract_answer(content),
    }


def main() -> int:
    samples_glob = os.environ["SAMPLES_GLOB"]
    base_url = os.environ["BASE_URL"]
    served = os.environ.get("SERVED_NAME", "MiniMaxAI/MiniMax-M3")
    out = os.environ["OUT"]
    n_samples = int(os.environ.get("N_SAMPLES", "5"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "32768"))
    n_control = int(os.environ.get("N_CONTROL", "25"))
    n_exhausted = int(os.environ.get("N_EXHAUSTED", "0"))
    conc = int(os.environ.get("CONCURRENCY", "24"))
    timeout = float(os.environ.get("REQUEST_TIMEOUT_S", "3600"))

    docs = _load_docs(_latest(samples_glob))
    exhausted = sorted(d for d, v in docs.items() if v["exhausted"])
    terminated = sorted(d for d, v in docs.items() if not v["exhausted"])
    # cap exhausted docs (evenly spaced by doc_id for representativeness)
    if n_exhausted > 0 and len(exhausted) > n_exhausted:
        step = max(1, len(exhausted) // n_exhausted)
        exhausted = exhausted[::step][:n_exhausted]
    # evenly spaced control from the terminated docs
    if terminated and n_control > 0:
        step = max(1, len(terminated) // n_control)
        control = terminated[::step][:n_control]
    else:
        control = []
    selected = [(d, "exhausted") for d in exhausted] + [(d, "control") for d in control]

    print(f"[probe] docs total={len(docs)} exhausted={len(exhausted)} "
          f"control={len(control)} base_url={base_url}", flush=True)

    # build the work list: (doc_id, cohort, regime, sample_idx, params)
    jobs = []
    for did, cohort in selected:
        jobs.append((did, cohort, "greedy", 0, 0.0, 1.0))
        for i in range(n_samples):
            jobs.append((did, cohort, "sampled", i, 1.0, 0.95))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = 0
    fails = 0
    with open(out, "w", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=conc) as ex:
        fut = {}
        for (did, cohort, regime, idx, temp, top_p) in jobs:
            f = ex.submit(_generate, base_url, served, docs[did]["prompt"],
                          temp, top_p, max_tokens, timeout)
            fut[f] = (did, cohort, regime, idx)
        for f in as_completed(fut):
            did, cohort, regime, idx = fut[f]
            rec = {"doc_id": did, "cohort": cohort, "regime": regime,
                   "sample_idx": idx, "target": docs[did]["target"]}
            try:
                g = f.result()
                rec.update(g)
                rec["correct"] = (g["answer"] is not None
                                  and g["answer"] == docs[did]["target"])
            except Exception as e:  # fail-open per generation
                rec["error"] = repr(e)[:300]
                fails += 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 25 == 0:
                print(f"[probe] {done}/{len(jobs)} (fails={fails})", flush=True)
    print(f"[probe] DONE {done}/{len(jobs)} fails={fails} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
