#!/usr/bin/env python3
"""Aggregate the unified one-node spec-dec window, with replicate statistics.

Runs safely against a PARTIAL window, so it doubles as the checkpoint tool while the
allocation is still live: every serve that has finished its cells is included, and
anything absent is reported as missing rather than silently dropped.

WHAT MAKES THIS DIFFERENT FROM THE PER-PHASE AGGREGATORS
--------------------------------------------------------
Phases D-I.2 each had one measurement per configuration, so their aggregators printed
raw deltas and left significance to prose. This window replicates the tight
comparisons on purpose, so the arithmetic belongs here:

  * configurations are keyed by what the EVIDENCE says (each serve's cell-config.txt:
    k, backend, drafter, kernel), not by parsing the label -- a mislabelled dir cannot
    silently join the wrong group;
  * replicates are pooled into mean / sd / se = sd/sqrt(n);
  * a difference between two configs carries se_diff = sqrt(seA^2 + seB^2) and is
    reported in se units, so "significant" is computed rather than asserted;
  * where n=1 there is no measured sd, so the ASSUMED historical floor is used
    (conc-1 1.02%, conc-10 0.16%, from the four cross-window replicates of phases
    H/I/I.2) and the row is flagged `assumed-sd`. Never present those as measured.

PRIMARY METRIC is per-user output tok/s (aiperf `output_token_throughput_per_user`,
the mean of per-request 1/ITL). Step cost = ITL * accepted length is the
acceptance-adjusted view and is the right lens for axes 2 and 3, which change only how
the drafter's arithmetic is done and must not be credited with acceptance scatter.

Usage:
    python pipeline/specdec_unified_aggregate.py --root <window> [--out-json f]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics as st

CONCS = (1, 10)
CELLS = ("8k-low", "8k-high")
# Assumed sd (%) when a config has a single replicate. Measured across four
# fresh-engine cross-window replicates on gpu-h113 in phases H/I/I.2.
ASSUMED_SD_PCT = {1: 1.02, 10: 0.16}


# ---------------------------------------------------------------- evidence readers
def _prom(path: str, name: str) -> float | None:
    if not os.path.exists(path):
        return None
    pat = re.compile(rf"^vllm:{name}(?:\{{(.*?)\}})?\s+([0-9.eE+]+)$", re.M)
    with open(path) as fh:
        vals = [float(v) for _, v in pat.findall(fh.read())]
    return sum(vals) if vals else None


def accepted(sdir: str, cell: str, conc: int) -> float | None:
    pre = f"{sdir}/metrics/sb-{cell}-c{conc}-pre.txt"
    post = f"{sdir}/metrics/sb-{cell}-c{conc}-post.txt"
    d0 = _prom(pre, "spec_decode_num_drafts_total")
    a0 = _prom(pre, "spec_decode_num_accepted_tokens_total")
    d1 = _prom(post, "spec_decode_num_drafts_total")
    a1 = _prom(post, "spec_decode_num_accepted_tokens_total")
    if None in (d0, a0, d1, a1) or d1 == d0:
        return None
    return 1 + (a1 - a0) / (d1 - d0)


def perf(sdir: str, cell: str, conc: int) -> dict | None:
    p = f"{sdir}/speedbench/{cell}/conc_{conc}/profile_export_aiperf.json"
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        d = json.load(fh)

    def g(k):
        v = d.get(k)
        return v.get("avg") if isinstance(v, dict) else v

    out = {
        "itl": g("inter_token_latency"),
        "per_user": g("output_token_throughput_per_user"),
        "server": g("output_token_throughput"),
        "ttft": g("time_to_first_token"),
        "osl": g("output_sequence_length"),
    }
    return out if out["per_user"] else None


def read_config(sdir: str) -> dict | None:
    """Config comes from the serve's own recorded evidence, never from its label."""
    p = f"{sdir}/cell-config.txt"
    if not os.path.exists(p):
        return None
    cfg = {}
    with open(p) as fh:
        for line in fh:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                cfg[k] = v
    if not {"k", "backend", "drafter", "kernel"} <= set(cfg):
        return None
    return cfg


def key_of(cfg: dict, cell: str) -> tuple:
    return (int(cfg["k"]), cfg["backend"], cfg["drafter"], cfg["kernel"], cell)


def key_str(key: tuple) -> str:
    k, backend, drafter, kernel, cell = key
    bits = [f"k={k}", backend]
    if k > 0:
        bits += [drafter, kernel]
    return f"{cell:8s} {' '.join(bits)}"


# ---------------------------------------------------------------- statistics
class Group:
    """Replicates of one configuration in one cell at one concurrency."""

    def __init__(self):
        self.rows: list[dict] = []
        self.labels: list[str] = []

    def add(self, label: str, row: dict):
        self.rows.append(row)
        self.labels.append(label)

    @property
    def n(self):
        return len(self.rows)

    def vals(self, metric):
        return [r[metric] for r in self.rows if r.get(metric) is not None]

    def mean(self, metric):
        v = self.vals(metric)
        return st.mean(v) if v else None

    def sd_pct(self, metric, conc):
        """Sample sd as a percentage of the mean; None when n<2 (caller substitutes)."""
        v = self.vals(metric)
        if len(v) < 2:
            return None
        m = st.mean(v)
        return st.stdev(v) / m * 100 if m else None

    def se_pct(self, metric, conc) -> tuple[float, bool]:
        """(se as % of mean, assumed_flag)."""
        sd = self.sd_pct(metric, conc)
        if sd is None:
            return ASSUMED_SD_PCT[conc] / math.sqrt(max(self.n, 1)), True
        return sd / math.sqrt(self.n), False


def compare(a: Group, b: Group, metric: str, conc: int) -> dict | None:
    """b relative to a, in percent, with se of the difference and its size in se."""
    ma, mb = a.mean(metric), b.mean(metric)
    if not ma or not mb:
        return None
    sea, fa = a.se_pct(metric, conc)
    seb, fb = b.se_pct(metric, conc)
    se = math.hypot(sea, seb)
    delta = (mb / ma - 1) * 100
    return {
        "delta_pct": delta,
        "se_pct": se,
        "n_se": abs(delta) / se if se else None,
        "assumed_sd": fa or fb,
        "n_a": a.n,
        "n_b": b.n,
        "mean_a": ma,
        "mean_b": mb,
    }


def verdict(c: dict | None) -> str:
    if not c or c["n_se"] is None:
        return "     -"
    tag = "sig" if c["n_se"] >= 2.0 else "ns "
    star = "~" if c["assumed_sd"] else " "
    return f"{tag}{star}{c['n_se']:4.1f}se"


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arm", default="unified")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    arm = f"{args.root}/arm-{args.arm}"
    groups: dict[tuple, dict[int, Group]] = {}
    served: list[str] = []
    missing: list[str] = []

    expected = []
    slist = f"{args.root}/serve-list.txt"
    if os.path.exists(slist):
        with open(slist) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    expected.append(line.split(";")[0])

    if expected:
        labels = expected
    elif os.path.isdir(arm):
        labels = sorted(d for d in os.listdir(arm) if os.path.isdir(f"{arm}/{d}"))
    else:
        labels = []

    for label in labels:
        sdir = f"{arm}/{label}"
        cfg = read_config(sdir)
        if cfg is None:
            missing.append(label)
            continue
        got_any = False
        for cell in CELLS:
            for conc in CONCS:
                pf = perf(sdir, cell, conc)
                if pf is None:
                    continue
                pf["accepted"] = accepted(sdir, cell, conc)
                if pf["accepted"] and pf["itl"]:
                    pf["step"] = pf["itl"] * pf["accepted"]
                groups.setdefault(key_of(cfg, cell), {}).setdefault(conc, Group()).add(label, pf)
                got_any = True
        (served if got_any else missing).append(label)

    print(f"window : {args.root}")
    print(f"arm    : {args.arm}")
    print(f"serves : {len(served)} with data, {len(missing)} missing/incomplete")
    if missing:
        print(f"         missing: {','.join(missing)}")
    print("         (a partial window is expected while the allocation is live)\n")

    def grp(k, backend, drafter, kernel, cell, conc) -> Group | None:
        return groups.get((k, backend, drafter, kernel, cell), {}).get(conc)

    hdr = (f"{'configuration':44s} {'c':>3} {'n':>2} | {'per-user':>9} {'sd%':>6} | "
           f"{'ITL ms':>7} | {'accepted':>8} | {'step ms':>8}")

    # ---------------- all measured configurations ----------------
    print("=" * 100)
    print("ALL CONFIGURATIONS (replicate means; sd% is measured only where n>=2)")
    print("=" * 100)
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(groups, key=lambda x: (x[4], x[0], x[1], x[2], x[3])):
        for conc in CONCS:
            g = groups[key].get(conc)
            if not g:
                continue
            sd = g.sd_pct("per_user", conc)
            sds = f"{sd:5.2f}%" if sd is not None else "    - "
            acc = g.mean("accepted")
            stp = g.mean("step")
            accs = f"{acc:8.3f}" if acc else f"{'-':>8}"
            stps = f"{stp:8.3f}" if stp else f"{'-':>8}"
            print(f"{key_str(key):44s} {conc:>3} {g.n:>2} | {g.mean('per_user'):9.1f} {sds:>6} | "
                  f"{g.mean('itl'):7.3f} | {accs} | {stps}")

    out = {"root": args.root, "served": served, "missing": missing, "axes": {}}

    def show(title, rows):
        """rows: list of (label, base Group, cand Group, conc)"""
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        h = (f"{'comparison':46s} {'c':>3} | {'base':>8} {'cand':>8} {'delta':>8} "
             f"{'se':>6} {'verdict':>11}")
        print(h)
        print("-" * len(h))
        res = []
        for lab, a, b, conc, metric in rows:
            if not a or not b:
                print(f"{lab:46s} {conc:>3} | {'(missing)':>8}")
                continue
            c = compare(a, b, metric, conc)
            if not c:
                print(f"{lab:46s} {conc:>3} | {'(no data)':>8}")
                continue
            print(f"{lab:46s} {conc:>3} | {c['mean_a']:8.2f} {c['mean_b']:8.2f} "
                  f"{c['delta_pct']:+7.2f}% {c['se_pct']:5.2f}% {verdict(c):>11}")
            res.append({"comparison": lab, "conc": conc, "metric": metric, **c})
        return res

    # ---------------- axis 0: barebone kernel ----------------
    rows = []
    for cell in CELLS:
        for conc in CONCS:
            rows.append((f"{cell} k=0: CUTLASS -> Humming (per-user)",
                         grp(0, "cutlass", "none", "default", cell, conc),
                         grp(0, "humming", "none", "default", cell, conc),
                         conc, "per_user"))
    out["axes"]["0-barebone-kernel"] = show(
        "AXIS 0 -- barebone kernel: W4AFP8 on CUTLASS vs Humming, no spec-dec\n"
        "This is the rung the published 95->137 figure covers; measured here on\n"
        "SPEED-Bench so it chains with the spec-dec legs on one workload.", rows)

    # ---------------- axis 1: draft depth ----------------
    rows = []
    for cell, ks in (("8k-low", (5, 6, 7)), ("8k-high", (1, 2, 3))):
        base0 = {c: grp(0, "humming", "none", "default", cell, c) for c in CONCS}
        for k in ks:
            for conc in CONCS:
                rows.append((f"{cell} k=0 -> k={k} (per-user speedup)",
                             base0[conc], grp(k, "humming", "int4", "default", cell, conc),
                             conc, "per_user"))
    out["axes"]["1-draft-depth-vs-control"] = show(
        "AXIS 1 -- draft depth vs the in-window k=0 Humming control", rows)

    rows = []
    for cell, pairs in (("8k-low", ((5, 6), (6, 7))), ("8k-high", ((1, 2), (2, 3)))):
        for a_k, b_k in pairs:
            for conc in CONCS:
                rows.append((f"{cell} k={a_k} -> k={b_k} (per-user)",
                             grp(a_k, "humming", "int4", "default", cell, conc),
                             grp(b_k, "humming", "int4", "default", cell, conc),
                             conc, "per_user"))
    out["axes"]["1-draft-depth-adjacent"] = show(
        "AXIS 1 -- adjacent k steps: this is where the optimum is decided,\n"
        "and where the replicates were spent (effects here are 1-3%).", rows)

    # ---------------- axis 2: drafter kernel ----------------
    rows = []
    for cell, k in (("8k-low", 5), ("8k-high", 2)):
        for kern in ("hum-lmhead", "hum-all", "machete-all"):
            for conc in CONCS:
                for metric in ("per_user", "step"):
                    rows.append((f"{cell} k={k} default -> {kern} ({metric})",
                                 grp(k, "humming", "int4", "default", cell, conc),
                                 grp(k, "humming", "int4", kern, cell, conc),
                                 conc, metric))
    out["axes"]["2-drafter-kernel"] = show(
        "AXIS 2 -- drafter W4A16 kernel assignment (step cost is the honest lens:\n"
        "the drafter reads identical bytes, so acceptance must not move)", rows)

    # ---------------- axis 3: drafter precision ----------------
    rows = []
    for cell, k in (("8k-low", 5), ("8k-high", 2)):
        for conc in CONCS:
            for metric in ("per_user", "step", "accepted"):
                rows.append((f"{cell} k={k} bf16 -> INT4 drafter ({metric})",
                             grp(k, "humming", "bf16", "default", cell, conc),
                             grp(k, "humming", "int4", "default", cell, conc),
                             conc, metric))
    out["axes"]["3-drafter-precision"] = show(
        "AXIS 3 -- drafter precision (accepted length is the quality control:\n"
        "it must NOT move; step cost is the speed claim)", rows)

    print("\nverdict column: sig/ns at 2 se; '~' means the sd was ASSUMED from the")
    print("historical floor because that configuration has a single replicate.")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True, default=str)
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
