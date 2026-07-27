#!/usr/bin/env python3
"""Aggregate the EAGLE3 spec-dec A/B window into one table (M3_SPECDEC_EAGLE3_PLAN.md).

Reads, per arm: the AA sweep summary, the acceptance lines vLLM's
SpecDecodingLogging wrote to serve.log, and the greedy-equivalence probe. Emits
markdown + JSON. Speedups are always stated against the k0 control measured in
the SAME window on the same checkpoint and kernel.

    python pipeline/specdec_aggregate.py --root /mnt/.../m3-specdec-eagle3/<TS>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

BENCH = "/mnt/nfs/hoangduy/projects/benchmarks"
ARMS = ["k0-control", "k1", "k3", "k5"]


def aa_cells(root: str, arm: str, run_ts: str) -> dict:
    """(input, conc) -> cell dict, from the arm's AA sweep summary."""
    path = os.path.join(
        BENCH, "results", f"minimax-m3-specdec-{arm}",
        "self-hosted", "perf", "aa-sweep", run_ts, "aa_sweep_summary.json")
    if not os.path.exists(path):
        # Fall back to whatever the arm recorded, in case the timestamp drifted.
        recorded = os.path.join(root, f"arm-{arm}", "aa-results.path")
        if os.path.exists(recorded):
            cand = os.path.join(open(recorded).read().strip(), "aa_sweep_summary.json")
            if os.path.exists(cand):
                path = cand
    if not os.path.exists(path):
        return {}
    doc = json.load(open(path))
    out = {}
    for cell in doc.get("cells", []):
        out[(cell.get("input"), cell.get("concurrency"))] = cell
    return out


ACC_LEN = re.compile(r"Mean acceptance length:\s*([0-9.]+)")
ACC_RATE = re.compile(r"Avg Draft acceptance rate:\s*([0-9.]+)%")
PER_POS = re.compile(r"Per-position acceptance rate:\s*([0-9.,\s]+)")
DRAFTED = re.compile(r"Drafted:\s*(\d+)\s*tokens")
ACCEPTED = re.compile(r"Accepted:\s*(\d+)\s*tokens")


def acceptance(root: str, arm: str) -> dict:
    """Last acceptance report in the arm's log, plus totals over all reports."""
    path = os.path.join(root, f"arm-{arm}", "spec-metrics.log")
    if not os.path.exists(path):
        return {}
    lines = [ln for ln in open(path, errors="replace") if "acceptance" in ln.lower()]
    if not lines:
        return {}
    last = lines[-1]
    def one(rx, text, cast=float):
        m = rx.search(text)
        return cast(m.group(1)) if m else None
    per_pos = None
    m = PER_POS.search(last)
    if m:
        per_pos = [float(x) for x in m.group(1).replace(" ", "").strip(",").split(",") if x]
    return {
        "reports": len(lines),
        "mean_acceptance_length": one(ACC_LEN, last),
        "avg_draft_acceptance_pct": one(ACC_RATE, last),
        "per_position_acceptance": per_pos,
        "drafted_tokens_last": one(DRAFTED, last, int),
        "accepted_tokens_last": one(ACCEPTED, last, int),
    }


def greedy(root: str, arm: str) -> dict:
    path = os.path.join(root, f"arm-{arm}", "greedy-probe.json")
    if not os.path.exists(path):
        return {}
    doc = json.load(open(path))
    res = doc.get("results", [])
    return {
        "probed": len(res),
        "errors": sum(1 for r in res if "error" in r),
        "texts": [(r.get("reasoning") or "") + (r.get("content") or "") for r in res],
    }


def prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run-ts", default=None, help="defaults to the root's basename")
    ap.add_argument("--min-prefix-chars", type=int, default=120)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    run_ts = args.run_ts or os.path.basename(args.root.rstrip("/"))

    data = {arm: {"aa": aa_cells(args.root, arm, run_ts),
                  "spec": acceptance(args.root, arm),
                  "greedy": greedy(args.root, arm)} for arm in ARMS}
    present = [a for a in ARMS if data[a]["aa"]]
    ctrl = data["k0-control"]["aa"]

    lines = [f"# EAGLE3 spec-dec A/B — window {run_ts}", ""]
    if not ctrl:
        lines.append("**No control cells found — speedups cannot be stated.**")
    lines += ["## Output speed (tok/s, AA p50) — x = vs k0 control", "",
              "| cell | " + " | ".join(present) + " |",
              "|---" * (len(present) + 1) + "|"]

    cells = sorted({c for a in present for c in data[a]["aa"]},
                   key=lambda c: (str(c[0]), c[1] or 0))
    for cell in cells:
        row = [f"{cell[0]} x conc {cell[1]}"]
        base = (ctrl.get(cell) or {}).get("output_speed_tps", {}).get("p50")
        for arm in present:
            c = data[arm]["aa"].get(cell) or {}
            v = (c.get("output_speed_tps") or {}).get("p50")
            if v is None:
                row.append("n/a")
            elif base and arm != "k0-control":
                row.append(f"{v:.1f} ({v / base:.2f}x)")
            else:
                row.append(f"{v:.1f}")
        lines.append("| " + " | ".join(row) + " |")

    for label, key, fmt in [
        ("TTFT p50 (ms)", "ttft_ms", "{:.0f}"),
        ("Aggregate output tok/s", "aggregate_output_tps", "{:.1f}"),
        ("Natural OSL (avg tokens)", "natural_osl_tokens", "{:.0f}"),
    ]:
        lines += ["", f"## {label}", "",
                  "| cell | " + " | ".join(present) + " |",
                  "|---" * (len(present) + 1) + "|"]
        for cell in cells:
            row = [f"{cell[0]} x conc {cell[1]}"]
            for arm in present:
                c = data[arm]["aa"].get(cell) or {}
                v = c.get(key)
                if isinstance(v, dict):
                    v = v.get("p50", v.get("avg"))
                row.append(fmt.format(v) if isinstance(v, (int, float)) else "n/a")
            lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Acceptance (from serve.log SpecDecodingLogging)", "",
              "| arm | reports | mean accepted length | avg draft acceptance | per-position |",
              "|---|---|---|---|---|"]
    for arm in ARMS:
        s = data[arm]["spec"]
        if not s:
            lines.append(f"| {arm} | — | — | — | — |")
            continue
        pp = ", ".join(f"{p:.2f}" for p in (s.get("per_position_acceptance") or [])) or "—"
        mal = s.get("mean_acceptance_length")
        rate = s.get("avg_draft_acceptance_pct")
        lines.append(f"| {arm} | {s.get('reports')} | "
                     f"{mal if mal is not None else '—'} | "
                     f"{str(rate) + '%' if rate is not None else '—'} | {pp} |")

    lines += ["", "## Greedy equivalence vs control (temp 0)", "",
              "| arm | prompts | errors | >= "
              f"{args.min_prefix_chars} identical chars | identical outputs |",
              "|---|---|---|---|---|"]
    ref = data["k0-control"]["greedy"].get("texts") or []
    for arm in ARMS:
        g = data[arm]["greedy"]
        if not g:
            lines.append(f"| {arm} | — | — | — | — |")
            continue
        texts = g.get("texts") or []
        if arm == "k0-control" or not ref:
            lines.append(f"| {arm} | {g['probed']} | {g['errors']} | (reference) | (reference) |")
            continue
        pref = [prefix_len(a, b) for a, b in zip(ref, texts)]
        ok = sum(1 for p in pref if p >= args.min_prefix_chars)
        same = sum(1 for a, b in zip(ref, texts) if a == b)
        lines.append(f"| {arm} | {g['probed']} | {g['errors']} | "
                     f"{ok}/{len(pref)} | {same}/{len(pref)} |")

    md = "\n".join(lines)
    print(md)
    if args.out_json:
        serializable = {a: {"aa": {f"{k[0]}|{k[1]}": v for k, v in d["aa"].items()},
                            "spec": d["spec"],
                            "greedy": {kk: vv for kk, vv in d["greedy"].items()
                                       if kk != "texts"}}
                        for a, d in data.items()}
        with open(args.out_json, "w") as fh:
            json.dump(serializable, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
