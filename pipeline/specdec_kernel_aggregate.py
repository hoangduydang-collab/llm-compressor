#!/usr/bin/env python3
"""Aggregate the phase I drafter-kernel window into a comparison table.

Cells (all k=5, 8k-low, conc 1 and 10, one node, serial) differ ONLY in which
W4A16 kernel serves the drafter's 9 linears:

    A-baseline      Machete x8 + Marlin (lm_head)      <- what phases D-H measured
    B-hum-lmhead    Machete x8 + Humming (lm_head)
    C-hum-all       Humming x9
    D-machete-all   Machete x9 (draft vocab padded 200064 -> 200704)
    A-repeat        A re-served at window end           <- drift floor

PRIMARY METRIC is per-user output tok/s, the AA convention: aiperf's
`output_token_throughput_per_user`, which is the mean of per-request 1/ITL. Note
that is NOT 1/mean(ITL), so it need not agree in sign with the ITL column; where
they conflict the cell is inside the noise band and both are printed.

Acceptance MUST NOT move across these cells -- the drafter weights and the prompts
are byte-identical, so only the arithmetic kernel differs. Any acceptance shift is
either noise or a numerics bug (cell D changes the lm_head's padded shard), so it is
reported as a control rather than folded into a speedup.

Step cost = ITL * accepted_length. Since acceptance is fixed by construction here,
step cost and ITL should tell the same story; a divergence between them is a signal
that acceptance drifted and the comparison is contaminated.

Usage:
    python pipeline/specdec_kernel_aggregate.py --root <window> [--out-json f]
"""

from __future__ import annotations

import argparse
import json
import os
import re

CELLS = ("A-baseline", "B-hum-lmhead", "C-hum-all", "D-machete-all", "A-repeat")
CONCS = (1, 10)
# Cross-window reference: phase H measured k=5 / 8k-low on this same node (gpu-h113),
# so A-baseline should reproduce it inside the 1-2% node-variance band. A large gap
# means something other than the kernel changed between windows.
PHASEH_K5 = {1: {"itl": 3.014, "per_user": 334.1}, 10: {"itl": 6.707, "per_user": 153.5}}


def _metric(path: str, name: str) -> float | None:
    if not os.path.exists(path):
        return None
    pat = re.compile(rf"^vllm:{name}(?:\{{(.*?)\}})?\s+([0-9.eE+]+)$", re.M)
    with open(path) as fh:
        body = fh.read()
    vals = [float(v) for _, v in pat.findall(body)]
    return sum(vals) if vals else None


def accepted(cdir: str, cell: str, conc: int) -> float | None:
    pre = f"{cdir}/metrics/sb-{cell}-c{conc}-pre.txt"
    post = f"{cdir}/metrics/sb-{cell}-c{conc}-post.txt"
    d0 = _metric(pre, "spec_decode_num_drafts_total")
    a0 = _metric(pre, "spec_decode_num_accepted_tokens_total")
    d1 = _metric(post, "spec_decode_num_drafts_total")
    a1 = _metric(post, "spec_decode_num_accepted_tokens_total")
    if None in (d0, a0, d1, a1) or d1 == d0:
        return None
    return 1 + (a1 - a0) / (d1 - d0)


def perf(cdir: str, cell: str, conc: int) -> dict | None:
    p = f"{cdir}/speedbench/{cell}/conc_{conc}/profile_export_aiperf.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))

    def g(k):
        v = d.get(k)
        return v.get("avg") if isinstance(v, dict) else v

    return {
        "itl": g("inter_token_latency"),
        "per_user": g("output_token_throughput_per_user"),
        "server": g("output_token_throughput"),
        "ttft": g("time_to_first_token"),
        "osl": g("output_sequence_length"),
    }


def read1(path: str) -> str:
    try:
        return open(path).read().strip()
    except OSError:
        return "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arm", default="kernel")
    ap.add_argument("--cell", default="8k-low")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    arm = f"{args.root}/arm-{args.arm}"
    rows: dict[tuple[str, int], dict] = {}
    meta: dict[str, dict] = {}

    for label in CELLS:
        cdir = f"{arm}/{label}"
        # A-repeat writes into "<cell>-repeat" so it cannot collide with A-baseline.
        cell = f"{args.cell}-repeat" if label == "A-repeat" else args.cell
        meta[label] = {
            "kernels": read1(f"{cdir}/wna16-kernels.txt"),
            "marlin_pad_warning": read1(f"{cdir}/marlin-pad-warning.txt"),
            "loaded_gib": read1(f"{cdir}/model-loading-gib.txt"),
        }
        for conc in CONCS:
            pf = perf(cdir, cell, conc)
            if pf is None:
                continue
            pf["accepted"] = accepted(cdir, cell, conc)
            if pf["accepted"] and pf["itl"]:
                pf["step"] = pf["itl"] * pf["accepted"]
            rows[(label, conc)] = pf

    print(f"window: {args.root}")
    print(f"arm={args.arm} cell={args.cell} (k=5, INT4 drafter, single node)\n")

    print("KERNEL ASSIGNMENT PER CELL (gated at serve time, fail-closed)")
    print(f"{'cell':16s} {'WNA16 kernels':22s} {'marlin pad warn':16s} {'loaded GiB':10s}")
    for label in CELLS:
        m = meta[label]
        if m["kernels"] == "-":
            continue
        print(f"{label:16s} {m['kernels']:22s} {m['marlin_pad_warning']:16s} {m['loaded_gib']:10s}")

    base = {c: rows.get(("A-baseline", c)) for c in CONCS}

    print("\nPER-USER OUTPUT tok/s (primary) and supporting metrics")
    hdr = (f"{'cell':16s} {'c':>3} | {'per-user':>9} {'vs A':>7} | {'ITL ms':>7} {'vs A':>7} | "
           f"{'accepted':>8} {'vs A':>7} | {'step ms':>8} {'vs A':>7} | {'server':>8}")
    print(hdr)
    print("-" * len(hdr))
    for label in CELLS:
        for conc in CONCS:
            r = rows.get((label, conc))
            if r is None:
                continue
            b = base[conc]

            def d(key):
                if not b or b.get(key) in (None, 0) or r.get(key) is None:
                    return "     - "
                return f"{(r[key] / b[key] - 1) * 100:+6.2f}%"

            acc = f"{r['accepted']:8.3f}" if r.get("accepted") else "       -"
            step = f"{r['step']:8.3f}" if r.get("step") else "       -"
            print(f"{label:16s} {conc:>3} | {r['per_user']:9.1f} {d('per_user')} | "
                  f"{r['itl']:7.3f} {d('itl')} | {acc} {d('accepted')} | "
                  f"{step} {d('step')} | {r['server']:8.1f}")

    # Drift floor: A-repeat vs A-baseline, same config on a fresh engine at window end.
    print("\nDRIFT CONTROL (A-repeat vs A-baseline: identical config, fresh engine, window end)")
    for conc in CONCS:
        a, z = base[conc], rows.get(("A-repeat", conc))
        if not a or not z:
            continue
        print(f"  conc {conc:>2}: per-user {a['per_user']:7.1f} -> {z['per_user']:7.1f} "
              f"({(z['per_user'] / a['per_user'] - 1) * 100:+.2f}%)   "
              f"ITL {a['itl']:.3f} -> {z['itl']:.3f} ms")
    print("  Any kernel effect smaller than this gap is not a result.")

    print("\nCROSS-WINDOW CHECK (phase H measured k=5 / 8k-low on this same node)")
    for conc in CONCS:
        a = base[conc]
        ref = PHASEH_K5.get(conc)
        if not a or not ref:
            continue
        print(f"  conc {conc:>2}: A-baseline {a['per_user']:7.1f} vs phase H {ref['per_user']:7.1f} "
              f"tok/s/user ({(a['per_user'] / ref['per_user'] - 1) * 100:+.2f}%)   "
              f"ITL {a['itl']:.3f} vs {ref['itl']:.3f} ms")

    print("\nACCEPTANCE CONTROL -- must be flat: same drafter weights, same prompt bytes.")
    print("A shift here is noise or a numerics bug (cell D pads the lm_head shard),")
    print("not a kernel speedup; it would contaminate the ITL comparison.")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(
                {"root": args.root, "arm": args.arm, "cell": args.cell,
                 "meta": meta,
                 "rows": {f"{k[0]}|c{k[1]}": v for k, v in rows.items()}},
                fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
