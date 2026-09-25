#!/usr/bin/env python3
"""Is the shared expert related to the routed experts?  One MoE layer.

Neurons inside an expert can be reordered freely, so a cosine between flattened
matrices is meaningless.  Three permutation-invariant tests instead, where a
"neuron" is (gate row | up row | down column), a vector in R^(3*hidden):

  A. neuron matching  each routed neuron -> max |cos| against every shared
                      neuron (is any routed neuron a copy of a shared one?)
  B. subspace overlap fraction of a routed expert's energy inside the top-k
                      hidden-space principal directions of the shared expert
  C. shared as a base after the best neuron permutation (Hungarian on |cos|)
                      and a signed per-neuron scale, the routed energy left
                      over -- i.e. how well "routed = shared + delta" works

Each test also runs routed-vs-routed (another routed expert in place of the
shared one) and on iid-Gaussian experts of the same shape, so "correlated"
means "more than two routed experts are with each other / than noise".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

from pipeline.basis_codec_layer_analysis import PROJS, load_layer, log

SUBSPACE_KS = (64, 256, 1024, 2048)


def load_shared(ckpt: str, layer: int) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    wmap = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    pat = re.compile(rf"(?:^|\.)layers\.{layer}\.mlp\.shared_experts\.(gate|up|down)_proj\.weight$")
    found = {m[1]: (k, f) for k, f in wmap.items() if (m := pat.search(k))}
    if set(found) != set(PROJS):
        sys.exit(f"FAIL: layer {layer} shared expert tensors found: {sorted(found)}")
    out = {}
    for p, (k, f) in found.items():
        with safe_open(os.path.join(ckpt, f), framework="pt", device="cpu") as h:
            out[p] = h.get_tensor(k)
    return out


def neurons(g, u, d) -> torch.Tensor:
    """[I, H] gate, [I, H] up, [H, I] down -> [I, 3H] fp32 neuron vectors."""
    return torch.cat([g.float(), u.float(), d.float().T], 1)


def unit(x):
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-30)


def max_abs_cos(A, B):
    return (unit(A) @ unit(B).T).abs().amax(1)


def hidden_rows(W: torch.Tensor, p: str) -> torch.Tensor:
    """Hidden-space vectors of a projection: gate/up rows, down columns."""
    return (W.T if p == "down" else W).float()


def top_dirs(X: torch.Tensor) -> torch.Tensor:
    """Principal hidden-space directions of X [n, H], descending, as columns."""
    lam, U = torch.linalg.eigh((X.T @ X).double())
    return U.flip(1).float()


def energy_in(X, U, k):
    return ((X @ U[:, :k]).pow(2).sum() / X.pow(2).sum()).item()


def delta_residual(N_e: torch.Tensor, N_b: torch.Tensor) -> float:
    """Energy of N_e left after matching each neuron to one neuron of the base N_b
    (Hungarian on |cos|) and fitting a signed per-neuron scale."""
    from scipy.optimize import linear_sum_assignment

    C = (unit(N_e) @ unit(N_b).T).abs().cpu().numpy()
    ri, ci = linear_sum_assignment(-C)
    ri, ci = torch.as_tensor(ri, device=N_e.device), torch.as_tensor(ci, device=N_e.device)
    a, b = N_e[ri], N_b[ci]
    alpha = (a * b).sum(1) / b.pow(2).sum(1).clamp(min=1e-30)
    left = (a - alpha[:, None] * b).pow(2).sum()
    unmatched = N_e.pow(2).sum() - a.pow(2).sum()     # if I_routed > I_shared
    return ((left + unmatched) / N_e.pow(2).sum()).item()


def q(x: torch.Tensor) -> dict:
    x = x.float().cpu()
    return {"p50": x.median().item(), "p99": torch.quantile(x, 0.99).item(),
            "max": x.max().item(), "frac_gt_0.5": (x > 0.5).float().mean().item()}


def analyze(R: dict, S: dict, dev, sample: list[int], match: list[int]) -> dict:
    E = R["gate"].shape[0]
    Ns = neurons(S["gate"].to(dev), S["up"].to(dev), S["down"].to(dev))
    Ne = lambda e: neurons(R["gate"][e].to(dev), R["up"][e].to(dev), R["down"][e].to(dev))
    partner = lambda e: (e + 1) % E
    out = {"energy_ratio_shared_over_mean_routed": {
        p: (S[p].float().pow(2).sum() / R[p].float().pow(2).sum((1, 2)).mean()).item() for p in PROJS}}

    # A. neuron matching, all experts
    a_sh, a_rr = [], []
    for e in range(E):
        n = Ne(e)
        a_sh.append(max_abs_cos(n, Ns))
        a_rr.append(max_abs_cos(n, Ne(partner(e))))
    out["A_neuron_max_abs_cos"] = {"vs_shared": q(torch.cat(a_sh)), "vs_other_routed": q(torch.cat(a_rr))}
    log("  A:", json.dumps(out["A_neuron_max_abs_cos"]))

    # B. hidden-space subspace overlap
    B = {}
    for p in PROJS:
        Us = top_dirs(hidden_rows(S[p].to(dev), p))
        H = Us.shape[0]
        ks = [k for k in SUBSPACE_KS if k <= H]
        sh = {k: 0.0 for k in ks}
        rr = {k: 0.0 for k in ks}
        for e in sample:
            X = hidden_rows(R[p][e].to(dev), p)
            Up = top_dirs(hidden_rows(R[p][partner(e)].to(dev), p))
            for k in ks:
                sh[k] += energy_in(X, Us, k) / len(sample)
                rr[k] += energy_in(X, Up, k) / len(sample)
        own = {k: energy_in(hidden_rows(S[p].to(dev), p), Us, k) for k in ks}
        B[p] = {f"k{k}": {"routed_in_shared": sh[k], "routed_in_other_routed": rr[k],
                          "isotropic": k / H, "shared_in_own": own[k]} for k in ks}
    out["B_subspace_energy"] = B
    log("  B:", json.dumps({p: {k: round(v["routed_in_shared"], 4) for k, v in B[p].items()} for p in B}))

    # C. routed = permuted, scaled shared + delta
    try:
        c_sh = [delta_residual(Ne(e), Ns) for e in match]
        c_rr = [delta_residual(Ne(e), Ne(partner(e))) for e in match]
        out["C_delta_residual_energy"] = {"vs_shared": q(torch.tensor(c_sh)), "vs_other_routed": q(torch.tensor(c_rr))}
        log("  C:", json.dumps(out["C_delta_residual_energy"]))
    except ImportError as ex:
        out["C_delta_residual_energy"] = {"skipped": str(ex)}
    return out


def gaussian_like(R, S, experts, gen):
    Rg = {p: torch.randn((experts, *R[p].shape[1:]), generator=gen).to(torch.bfloat16) for p in PROJS}
    Sg = {p: torch.randn(S[p].shape, generator=gen).to(torch.bfloat16) for p in PROJS}
    return Rg, Sg


def fmt(x):
    return f"{x:.4f}"


def summarize(res: dict, ctrl: dict) -> str:
    L = ["# Shared vs routed experts", ""]
    for name, r in (("real", res), ("iid-Gaussian control", ctrl)):
        L += [f"## {name}", "", "Energy ratio shared / mean routed: "
              + ", ".join(f"{p} {v:.3f}" for p, v in r["energy_ratio_shared_over_mean_routed"].items()), "",
              "| test | vs shared | vs other routed |", "|---|---|---|"]
        a = r["A_neuron_max_abs_cos"]
        for s in ("p50", "p99", "max", "frac_gt_0.5"):
            L.append(f"| A neuron max abs cos, {s} | {fmt(a['vs_shared'][s])} | {fmt(a['vs_other_routed'][s])} |")
        c = r["C_delta_residual_energy"]
        if "skipped" not in c:
            for s in ("p50", "max"):
                L.append(f"| C delta residual energy, {s} | {fmt(c['vs_shared'][s])} | {fmt(c['vs_other_routed'][s])} |")
        L += ["", "| proj | k | routed in shared top-k | routed in other-routed top-k | isotropic | shared in own top-k |",
              "|---|---|---|---|---|---|"]
        for p, byk in r["B_subspace_energy"].items():
            for k, v in byk.items():
                L.append(f"| {p} | {k[1:]} | {fmt(v['routed_in_shared'])} | {fmt(v['routed_in_other_routed'])} | "
                         f"{fmt(v['isotropic'])} | {fmt(v['shared_in_own'])} |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=None, help="default: first MoE layer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subspace-experts", type=int, default=16)
    ap.add_argument("--match-experts", type=int, default=32)
    ap.add_argument("--control-experts", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    R, meta = load_layer(args.ckpt, args.layer, None, args.threads)
    S = load_shared(args.ckpt, meta["layer"])
    meta["shared_shapes"] = {p: list(S[p].shape) for p in PROJS}
    log("shared expert shapes", meta["shared_shapes"])
    E = R["gate"].shape[0]

    def spread(n, total):
        return list(range(0, total, max(1, total // n)))[:n]

    res = analyze(R, S, dev, spread(args.subspace_experts, E), spread(args.match_experts, E))
    log("== iid-Gaussian control")
    Rg, Sg = gaussian_like(R, S, args.control_experts, gen)
    ctrl = analyze(Rg, Sg, dev, spread(args.subspace_experts, args.control_experts),
                   spread(args.match_experts, args.control_experts))
    blob = {"meta": meta, "results": res, "gaussian_control": ctrl, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "shared_expert_results.json"), "w") as f:
        json.dump(blob, f, indent=1)
    summary = summarize(res, ctrl)
    with open(os.path.join(args.out, "shared_expert_summary.md"), "w") as f:
        f.write(summary)
    print("\n" + summary, flush=True)
    log("done ->", args.out)


if __name__ == "__main__":
    main()
