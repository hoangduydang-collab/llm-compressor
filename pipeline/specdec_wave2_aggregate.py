#!/usr/bin/env python3
"""Aggregate EAGLE3 spec-dec wave 2 (M3_SPECDEC_EAGLE3_PLAN.md).

Three phases, each control-vs-k3, joined per cell with the acceptance measured for
THAT cell from /metrics counter deltas:

    mean accepted length = 1 + d(num_accepted_tokens) / d(num_drafts)
    acceptance rate      =     d(num_accepted_tokens) / d(num_draft_tokens)
    per-position rate[i] =     d(per_pos{position=i})  / d(num_drafts)

Speed numbers come from the analyzer's points.csv (interactivity_tps = per-request
output speed, output_tps = server total), so they are the same quantities the
two-axis report uses.

    python pipeline/specdec_wave2_aggregate.py --root /mnt/.../<TS>-wave2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

BENCH = "/mnt/nfs/hoangduy/projects/benchmarks"
PHASES = ["natural", "load", "lowconc"]
ARMS = {"k0": "control", "k3": "eagle3 k=3"}
COUNTER = re.compile(r"^(vllm:spec_decode_[a-z_]*?)(?:_total)?\{([^}]*)\}\s+([0-9.eE+-]+)")


def read_counters(path: str) -> dict:
    """name -> float, plus per_pos -> {position: float}. Ignores _created stamps."""
    out: dict = {"per_pos": {}}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        if line.startswith("#") or "spec_decode" not in line:
            continue
        m = COUNTER.match(line.strip())
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2), float(m.group(3))
        if name.endswith("_created"):
            continue
        if "per_pos" in name:
            pos = re.search(r'position="(\d+)"', labels)
            if pos:
                out["per_pos"][int(pos.group(1))] = out["per_pos"].get(int(pos.group(1)), 0.0) + value
        else:
            out[name] = out.get(name, 0.0) + value
    return out


def acceptance(root: str, arm: str, cell: str) -> dict:
    """Acceptance for one cell, from the pre/post snapshots taken around it."""
    base = os.path.join(root, f"arm-{arm}", "metrics")
    pre = read_counters(os.path.join(base, f"{cell}-pre.txt"))
    post = read_counters(os.path.join(base, f"{cell}-post.txt"))
    if not post or "vllm:spec_decode_num_drafts" not in post:
        return {}
    def d(key):
        return post.get(key, 0.0) - pre.get(key, 0.0)
    drafts = d("vllm:spec_decode_num_drafts")
    draft_tokens = d("vllm:spec_decode_num_draft_tokens")
    accepted = d("vllm:spec_decode_num_accepted_tokens")
    if drafts <= 0:
        return {"drafts": 0}
    per_pos = {}
    for pos, val in sorted(post.get("per_pos", {}).items()):
        per_pos[pos] = (val - pre.get("per_pos", {}).get(pos, 0.0)) / drafts
    return {
        "drafts": int(drafts),
        "mean_accepted_length": 1 + accepted / drafts,
        "acceptance_rate": accepted / draft_tokens if draft_tokens else None,
        "per_position": [round(per_pos[p], 3) for p in sorted(per_pos)],
    }


def points(path: str) -> dict:
    """concurrency -> row, from an analyzer points.csv."""
    if not os.path.exists(path):
        return {}
    rows = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                rows[int(float(row["concurrency"]))] = row
            except (KeyError, ValueError):
                continue
    return rows


def natural_points(root: str, arm: str, tag: str) -> dict:
    return points(os.path.join(root, f"arm-{arm}", "natural", tag, "points.csv"))


def reasoning_points(phase: str, k: str, run_ts: str) -> dict:
    pat = os.path.join(BENCH, "results", f"minimax-m3-inhouse-specdec-w2-{phase}-{k}",
                       "vllm", "perf", "reasoning", run_ts, "points.csv")
    hits = glob.glob(pat)
    return points(hits[0]) if hits else {}


def fmt(row: dict, key: str, digits: int = 1) -> str:
    try:
        return f"{float(row[key]):.{digits}f}"
    except (KeyError, ValueError, TypeError):
        return "n/a"


def table(title: str, concs: list, c0: dict, c3: dict, acc: dict, note: str = "") -> list:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines += ["| conc | control speed | k3 speed | × | control total | k3 total | × | k3 accepted len | k3 per-position |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in concs:
        r0, r3 = c0.get(c, {}), c3.get(c, {})
        s0, s3 = r0.get("interactivity_tps"), r3.get("interactivity_tps")
        t0, t3 = r0.get("output_tps"), r3.get("output_tps")
        ratio = f"{float(s3) / float(s0):.2f}x" if s0 and s3 else "n/a"
        tratio = f"{float(t3) / float(t0):.2f}x" if t0 and t3 else "n/a"
        a = acc.get(c, {})
        mal = f"{a['mean_accepted_length']:.2f}" if a.get("mean_accepted_length") else "—"
        pp = ", ".join(str(x) for x in a.get("per_position", [])) or "—"
        lines.append(f"| {c} | {fmt(r0, 'interactivity_tps')} | {fmt(r3, 'interactivity_tps')} | "
                     f"{ratio} | {fmt(r0, 'output_tps')} | {fmt(r3, 'output_tps')} | {tratio} | "
                     f"{mal} | {pp} |")
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run-ts", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    root = args.root.rstrip("/")
    # Window dir is <RUN_TS>-wave2; the suite's RUN_TS is the bare timestamp.
    run_ts = args.run_ts or os.path.basename(root).replace("-wave2", "")

    out = [f"# EAGLE3 spec-dec wave 2 — window {os.path.basename(root)}", "",
           "Speed = per-request output speed (tok/s); total = server output tok/s.",
           "× is k=3 over the in-window control. Acceptance is per cell, from",
           "/metrics counter deltas around that cell.", ""]
    blob = {}

    out.append("## Phase A — ShareGPT natural prompts, natural output")
    out.append("")
    for tag, temp in [("t06", "0.6 (production sampling)"), ("t0", "0 (upper bound)")]:
        c0 = natural_points(root, "natural-k0", tag)
        c3 = natural_points(root, "natural-k3", tag)
        acc = {c: acceptance(root, "natural-k3", f"natural-{tag}-c{c}") for c in (1, 10)}
        out += table(f"temp {temp}", [1, 10], c0, c3, acc)
        blob[f"natural_{tag}"] = {"control": c0, "k3": c3, "acceptance": acc}

    out.append("## Phase B — load sweep (synthetic 1k in / 8k pinned out, temp 0.6)")
    out.append("")
    c0 = reasoning_points("load", "k0", run_ts)
    c3 = reasoning_points("load", "k3", run_ts)
    acc = {c: acceptance(root, "load-k3", f"reasoning-c{c}") for c in (16, 32, 64)}
    out += table("conc 16 / 32 / 64", [16, 32, 64], c0, c3, acc,
                 "A total-throughput ratio below 1.00x is the point where spec-dec "
                 "starts costing capacity.")
    blob["load"] = {"control": c0, "k3": c3, "acceptance": acc}

    out.append("## Phase C — like-for-like vs the two-axis tables (same shape, conc 1/4)")
    out.append("")
    c0 = reasoning_points("lowconc", "k0", run_ts)
    c3 = reasoning_points("lowconc", "k3", run_ts)
    acc = {c: acceptance(root, "lowconc-k3", f"reasoning-c{c}") for c in (1, 4)}
    out += table("conc 1 / 4", [1, 4], c0, c3, acc)
    blob["lowconc"] = {"control": c0, "k3": c3, "acceptance": acc}

    md = "\n".join(out)
    print(md)
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(blob, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
