"""What the KV residual looks like after subtracting codebook anchors (CPU, offline).

Reads dump/NNN.pt written by kv_anchor_deep --dump-layers (held-out latents, per-key
attention, codebooks) and reports, per layer:

* per-token residual size ||x - P|| / ||x||, and its spread across tokens
  (||r_j|| / median ||r||) -- a ch group shares one scale across 128 tokens;
* the crest factor max|v| / rms(v) of each ch group (128 tokens, one channel) and each
  tok group (one token, 128 channels), raw latent vs residual -- the peak sets the scale;
* the share of tokens that pick the zero centroid, and the attention they receive;
* residual size of the 1% most-attended tokens vs the rest.

    python -m pipeline.kv_residual_stats <out_dir>/dump [--cb cb4096]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pipeline import kv_anchor_probe as kp

G = 128
QS = (0.1, 0.5, 0.9, 0.99)


def quantiles(v: torch.Tensor) -> dict:
    v = v.flatten().float()
    if v.numel() > 1 << 24:                       # torch.quantile input limit
        v = v[torch.randperm(v.numel())[: 1 << 24]]
    return {f"p{int(q * 100)}": round(torch.quantile(v, q).item(), 4) for q in QS}


def crest(V: torch.Tensor, grouping: str) -> torch.Tensor:
    """max|v| / rms(v) per group. V [T, d]; ch groups run over tokens, tok groups over channels."""
    T, d = V.shape
    g = V.T.reshape(d, T // G, G) if grouping == "ch" else V.reshape(T, d // G, G)
    return g.abs().amax(-1) / g.pow(2).mean(-1).sqrt().clamp(min=1e-12)


def layer_stats(dump: dict, cb: str) -> dict:
    C = dump[cb].float()
    Cz = torch.cat([C, torch.zeros_like(C[:1])])
    rows = {k: [] for k in ("rel", "spread", "rel_top", "rel_rest")}
    cr = {k: [] for k in ("raw_ch", "res_ch", "raw_tok", "res_tok")}
    zero_frac, zero_mass = [], []
    for X, km in zip(dump["X"], dump["key_mass"]):
        X = X.float()
        idx = kp.nearest(X, Cz)
        R = X - Cz[idx]
        rn, xn = R.norm(dim=1), X.norm(dim=1).clamp(min=1e-12)
        rel = rn / xn
        rows["rel"].append(rel)
        rows["spread"].append(rn / rn.median())
        z = idx == C.shape[0]
        zero_frac.append(z.float().mean().item())
        if km is not None:
            zero_mass.append(km[z].sum().item())
            top = torch.zeros_like(z)
            top[km.float().topk(max(1, X.shape[0] // 100)).indices] = True
            rows["rel_top"].append(rel[top])
            rows["rel_rest"].append(rel[~top])
        for g in ("ch", "tok"):
            cr[f"raw_{g}"].append(crest(X, g))
            cr[f"res_{g}"].append(crest(R, g))
    mean = lambda v: sum(v) / len(v) if v else None
    return {"residual_over_token": quantiles(torch.cat(rows["rel"])),
            "residual_over_median_residual": quantiles(torch.cat(rows["spread"])),
            "residual_over_token_top1pct_attended": quantiles(torch.cat(rows["rel_top"])) if rows["rel_top"] else None,
            "residual_over_token_rest": quantiles(torch.cat(rows["rel_rest"])) if rows["rel_rest"] else None,
            "crest_factor": {k: quantiles(torch.cat([c.flatten() for c in v])) for k, v in cr.items()},
            "zero_anchor_fraction": mean(zero_frac), "attn_mass_on_zero_anchor_tokens": mean(zero_mass)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--cb", default="cb4096")
    args = ap.parse_args(argv)
    out = {}
    for f in sorted(Path(args.dump_dir).glob("*.pt")):
        out[int(f.stem)] = layer_stats(torch.load(f), args.cb)
        print(f"layer {int(f.stem)}: {json.dumps(out[int(f.stem)])}", flush=True)
    (Path(args.dump_dir) / f"residual_stats_{args.cb}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
