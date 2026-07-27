#!/usr/bin/env python3
"""Aggregate the phase G quantized-drafter A/B window into a comparison table.

Each arm holds two configs on the same node -- `int4/` and `fp/` -- with identical
cells. The primary metric is the ITL ratio between them at equal accepted length:
if acceptance is unchanged, the number of decode steps for a given output is
unchanged, so ITL is proportional to step cost and the ratio is the drafting-cost
delta directly. No k=0 control is needed for that comparison and none is run in
this window; phase D's k=0 numbers are quoted only as context.

The bf16 half re-runs 8k-low conc 1 at the end as cell `8k-low-repeat`. That
same-serve repeat is the noise floor: an int4-vs-fp delta smaller than the drift
between two identical runs is not a result.

Usage:
    python pipeline/specdec_int4drafter_aggregate.py --root <window> [--out-json f]
"""

from __future__ import annotations

import argparse
import json
import os
import re

CELLS = (("8k-low", 1), ("8k-high", 1), ("8k-low", 10), ("8k-high", 10))
# Phase D, same target checkpoint / TP8 / serve config, k=0 control (ms).
PHASED_K0_ITL = {("8k-low", 1): 7.313, ("8k-high", 1): 7.315,
                 ("8k-low", 10): 13.751, ("8k-high", 10): 12.830}


def _metric(path: str, name: str, pos: int | None = None) -> float | None:
    if not os.path.exists(path):
        return None
    pat = re.compile(rf"^vllm:{name}(?:\{{(.*?)\}})?\s+([0-9.eE+]+)$", re.M)
    tot, hit = 0.0, False
    with open(path) as fh:
        body = fh.read()
    for lab, val in pat.findall(body):
        if pos is not None and f'position="{pos}"' not in lab:
            continue
        tot += float(val)
        hit = True
    return tot if hit else None


def accepted(cdir: str, cell: str, conc: int, k: int):
    pre = f"{cdir}/metrics/sb-{cell}-c{conc}-pre.txt"
    post = f"{cdir}/metrics/sb-{cell}-c{conc}-post.txt"
    d0 = _metric(pre, "spec_decode_num_drafts_total")
    a0 = _metric(pre, "spec_decode_num_accepted_tokens_total")
    d1 = _metric(post, "spec_decode_num_drafts_total")
    a1 = _metric(post, "spec_decode_num_accepted_tokens_total")
    if None in (d0, a0, d1, a1) or d1 == d0:
        return None, []
    per = []
    for i in range(k):
        p0 = _metric(pre, "spec_decode_num_accepted_tokens_per_pos_total", i)
        p1 = _metric(post, "spec_decode_num_accepted_tokens_per_pos_total", i)
        if p0 is None or p1 is None:
            break
        per.append((p1 - p0) / (d1 - d0))
    return 1 + (a1 - a0) / (d1 - d0), per


def perf(cdir: str, cell: str, conc: int):
    f = f"{cdir}/speedbench/{cell}/conc_{conc}/profile_export_aiperf.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    g = lambda key: (d.get(key) or {}).get("avg")
    return dict(itl=g("inter_token_latency"), ttft=g("time_to_first_token"),
                osl=g("output_sequence_length"),
                tps_user=g("output_token_throughput_per_user"),
                tps_server=g("output_token_throughput"))


def pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    out: dict = {"root": args.root, "arms": {}}
    print(f"window {args.root}\n")
    hdr = (f"{'k':>2s} {'cell':9s} {'c':>3s} | {'acc4':>6s} {'acc16':>6s} {'dAcc%':>7s} "
           f"| {'itl4':>7s} {'itl16':>7s} {'dITL%':>7s} | {'step4':>7s} {'step16':>7s} {'dStep%':>7s}")
    print(hdr)
    print("-" * len(hdr))

    for k in (3, 4, 5):
        arm = f"{args.root}/arm-int4-k{k}"
        if not os.path.isdir(arm):
            continue
        rows = {}
        for cell, conc in CELLS:
            r4, r16 = {}, {}
            for tag, cdir in (("int4", f"{arm}/int4"), ("fp", f"{arm}/fp")):
                p = perf(cdir, cell, conc)
                acc, per = accepted(cdir, cell, conc, k)
                (r4 if tag == "int4" else r16).update(
                    dict(itl=(p or {}).get("itl"), ttft=(p or {}).get("ttft"),
                         osl=(p or {}).get("osl"), tps_user=(p or {}).get("tps_user"),
                         acc=acc, per=per))
            if not (r4.get("itl") and r16.get("itl") and r4.get("acc") and r16.get("acc")):
                print(f"{k:>2d} {cell:9s} {conc:>3d} |  (incomplete: int4="
                      f"{bool(r4.get('itl'))} fp={bool(r16.get('itl'))})")
                continue
            s4 = r4["itl"] * r4["acc"]
            s16 = r16["itl"] * r16["acc"]
            rows[f"{cell}|{conc}"] = dict(
                int4=r4, fp=r16, step_int4=s4, step_fp=s16,
                d_acc_pct=pct(r4["acc"], r16["acc"]),
                d_itl_pct=pct(r4["itl"], r16["itl"]),
                d_step_pct=pct(s4, s16),
                k0_itl=PHASED_K0_ITL.get((cell, conc)))
            print(f"{k:>2d} {cell:9s} {conc:>3d} | {r4['acc']:6.3f} {r16['acc']:6.3f} "
                  f"{pct(r4['acc'], r16['acc']):+7.2f} | {r4['itl']:7.3f} {r16['itl']:7.3f} "
                  f"{pct(r4['itl'], r16['itl']):+7.2f} | {s4:7.3f} {s16:7.3f} "
                  f"{pct(s4, s16):+7.2f}")

        # Noise floor: identical cell, same serve, ~90 min apart.
        base = perf(f"{arm}/fp", "8k-low", 1)
        rep = perf(f"{arm}/fp", "8k-low-repeat", 1)
        drift = None
        if base and rep and base["itl"] and rep["itl"]:
            drift = pct(rep["itl"], base["itl"])
            print(f"{k:>2d} {'drift(fp)':9s} {1:>3d} | same cell, same serve, later: "
                  f"ITL {base['itl']:.3f} -> {rep['itl']:.3f} ({drift:+.2f}%)")
        out["arms"][f"k{k}"] = {"cells": rows, "fp_drift_pct": drift,
                                "loaded_gib": {
                                    t: (open(f"{arm}/{t}/model-loading-gib.txt").read().strip()
                                        if os.path.exists(f"{arm}/{t}/model-loading-gib.txt") else None)
                                    for t in ("int4", "fp")}}
        print()

    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=1)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
