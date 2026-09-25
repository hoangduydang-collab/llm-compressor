#!/usr/bin/env python3
"""Basis-codec feasibility on the routed experts of one MoE layer (GLM-5.3 default).

Question: can a per-layer basis plus quantized coefficients -- no residual --
reach INT4-g128 weight error at ~3 bits/weight?  Four measurements, one read of
the layer's BF16 weights:

  1. target   RTN INT{2,3,4}, groups of 128 along the input axis, symmetric
              [-2^(b-1), 2^(b-1)-1], FP16 scale, 41-point MSE clip search.
  2. dense    pooled d-block covariance spectrum (transform-coding bound
              0.5*log2(AM/GM)) and a measured PCA codec at a fixed rate
              (greedy integer bit allocation per direction), with an
              identity-basis control that isolates the PCA gain.
  3. sparse   overcomplete dictionary (MOD updates + batched OMP), each block
              = s atoms x quantized coefficients, configs sized to 3.0 bpw.
  4. experts  256x256 Gram spectrum of the flattened experts ("eigen-experts").

Every arm also runs on an iid-Gaussian control of the same shape: iid Gaussian
weights cannot beat INT4 at 3 bpw with ANY coder (Shannon: 2^-6 = 1.56% vs
~1.2% for INT4), so a real-weight result near its control means no exploitable
structure.  Weight-space relative MSE only; activation-weighted error is out of
scope.  Bits/weight always include scale and index overhead.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch

PROJS = ("gate", "up", "down")
CLIP = torch.linspace(0.2, 1.0, 41).tolist()
SCALE_GROUP = 128
# (d, K, s, coef_bits): s * (log2 K + coef_bits) / d == 3.0 bits/weight
SPARSE_CONFIGS = ((8, 256, 2, 4), (16, 256, 4, 4), (32, 4096, 6, 4))
DENSE_DIMS = (8, 16, 32, 64, 128)
DENSE_RATES = (3.0, 4.0)
EIG_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- quantizer
def quant_groups(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Dequantized x [G, S]: one FP16 scale per row-group, MSE clip search."""
    if bits <= 0:
        return torch.zeros_like(x)
    if bits == 1:  # sign quantizer; MSE-optimal magnitude is mean |x|
        s = x.abs().mean(1).half().float()
        return torch.sign(x) * s[:, None]
    qmax, qmin = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    s0 = torch.maximum(x.amax(1).clamp(min=0) / qmax, (-x.amin(1)).clamp(min=0) / -qmin)
    best = torch.full((x.shape[0],), float("inf"), device=x.device)
    out = torch.zeros_like(x)
    for m in CLIP:
        s = (s0 * m).half().float()
        safe = torch.where(s > 0, s, torch.ones_like(s))
        xq = torch.clamp(torch.round(x / safe[:, None]), qmin, qmax) * s[:, None]
        err = (xq - x).pow(2).sum(1)
        better = err < best
        best = torch.where(better, err, best)
        out[better] = xq[better]
    return out


def quant_dirs(y: torch.Tensor, bits: int, group: int = SCALE_GROUP) -> torch.Tensor:
    """Quantize each row of y [dirs, L] in its own consecutive groups."""
    L = y.shape[1]
    pad = (-L) % group
    g = torch.nn.functional.pad(y, (0, pad)).reshape(-1, group)
    return quant_groups(g, bits).reshape(y.shape[0], -1)[:, :L]


def rtn_err(w: torch.Tensor, bits: int) -> tuple[float, float]:
    """(squared error, energy) of RTN g128 on a [m, n] matrix."""
    wq = quant_groups(w.reshape(-1, SCALE_GROUP), bits).reshape(w.shape)
    return (wq - w).pow(2).sum().item(), w.pow(2).sum().item()


# ---------------------------------------------------------------- loading
def load_layer(ckpt: str, layer: int | None, max_experts: int | None, threads: int):
    from safetensors import safe_open

    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    if "quantization_config" in cfg:
        sys.exit("FAIL: source config has quantization_config; need an unquantized checkpoint")
    first_moe = int(cfg.get("first_k_dense_replace", 0))
    layer = first_moe if layer is None else layer
    if layer < first_moe:
        sys.exit(f"FAIL: layer {layer} is dense (first_k_dense_replace={first_moe})")
    n_exp = int(cfg["n_routed_experts"])
    E = min(n_exp, max_experts) if max_experts else n_exp

    wmap = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    pat = re.compile(rf"(?:^|\.)layers\.{layer}\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
    found = {}
    for k, f in wmap.items():
        m = pat.search(k)
        if m and int(m[1]) < E:
            found[(int(m[1]), m[2])] = (k, f)
    missing = [(e, p) for e in range(E) for p in PROJS if (e, p) not in found]
    if missing:
        sample = [k for k in wmap if f"layers.{layer}." in k][:20]
        sys.exit(f"FAIL: {len(missing)} expert tensors missing, e.g. {missing[:4]}; "
                 f"layer keys look like: {sample}")

    by_file = defaultdict(list)
    for (e, p), (k, f) in found.items():
        by_file[f].append((e, p, k))

    def read(item):
        f, entries = item
        with safe_open(os.path.join(ckpt, f), framework="pt", device="cpu") as h:
            return [(e, p, h.get_tensor(k)) for e, p, k in entries]

    W: dict[str, torch.Tensor] = {}
    t0 = time.time()
    nbytes = 0
    with ThreadPoolExecutor(threads) as ex:
        for chunk in ex.map(read, sorted(by_file.items())):
            for e, p, t in chunk:
                if t.dtype != torch.bfloat16:
                    sys.exit(f"FAIL: expert {e} {p} dtype {t.dtype}, expected bfloat16")
                if p not in W:
                    W[p] = torch.empty((E, *t.shape), dtype=t.dtype)
                W[p][e] = t
                nbytes += t.numel() * t.element_size()
    dt = time.time() - t0
    log(f"loaded layer {layer}: {E} experts, {nbytes/1e9:.2f} GB in {dt:.0f}s "
        f"({nbytes/1e6/max(dt,1e-9):.0f} MB/s) from {len(by_file)} shards")
    meta = {"layer": layer, "experts": E, "n_routed_experts": n_exp,
            "first_k_dense_replace": first_moe, "shards": sorted(by_file),
            "shapes": {p: list(W[p].shape[1:]) for p in PROJS},
            "load_seconds": round(dt, 1), "load_bytes": nbytes,
            "model_type": cfg.get("model_type"), "architectures": cfg.get("architectures")}
    return W, meta


# ---------------------------------------------------------------- 1. target
def part_target(W: torch.Tensor, dev) -> dict:
    err = {b: 0.0 for b in (2, 3, 4)}
    energy = 0.0
    m4 = 0.0
    n = 0
    for e in range(W.shape[0]):
        w = W[e].to(dev, torch.float32)
        for b in err:
            se, en = rtn_err(w, b)
            err[b] += se
        energy += w.pow(2).sum().item()
        z = w / w.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-30)
        m4 += z.pow(4).sum().item()
        n += z.numel()
    return {f"int{b}_g128": {"rel_mse": err[b] / energy, "bpw": b + 16 / SCALE_GROUP}
            for b in err} | {"row_normalized_excess_kurtosis": m4 / n - 3.0}


# ---------------------------------------------------------------- 2. dense
def allocate_bits(lam: torch.Tensor, total: int, cap: int = 8) -> list[int]:
    """Greedy reverse water-filling with integer bits (distortion ~ lam*4^-b)."""
    b = [0] * len(lam)
    lam = lam.tolist()
    for _ in range(total):
        k = max((i for i in range(len(b)) if b[i] < cap), key=lambda i: lam[i] * 4.0 ** -b[i])
        b[k] += 1
    return b


def part_dense(W: torch.Tensor, dev) -> dict:
    E, m, n = W.shape
    out = {}
    for d in DENSE_DIMS:
        if n % d:
            continue
        C = torch.zeros(d, d, dtype=torch.float64, device=dev)
        N = 0
        for e in range(E):
            B = W[e].to(dev, torch.float64).reshape(-1, d)
            C += B.T @ B
            N += B.shape[0]
        C /= N
        lam, U = torch.linalg.eigh(C)
        lam, U = lam.flip(0).clamp(min=1e-300), U.flip(1)
        am, gm = lam.mean().item(), torch.exp(torch.log(lam).mean()).item()
        rec = {"bound_bits_saved_amgm": 0.5 * math.log2(am / gm),
               "top_quarter_energy": (lam[: d // 4].sum() / lam.sum()).item(),
               "eigs_head": [round(x, 6) for x in (lam[:8] / lam.mean()).tolist()]}
        for rate in DENSE_RATES:
            for arm, basis, var in (("pca", U.float(), lam), ("ident", None, torch.diagonal(C))):
                bits = allocate_bits(var, round(rate * d))
                groups = defaultdict(list)  # same bit count -> one batched quantize
                for k, bk in enumerate(bits):
                    groups[bk].append(k)
                se = en = 0.0
                for e in range(E):
                    Y = W[e].to(dev, torch.float32).reshape(-1, d)
                    if basis is not None:
                        Y = Y @ basis
                    en += Y.pow(2).sum().item()
                    for bk, ks in groups.items():
                        y = Y[:, ks].T.contiguous()   # [dirs, m*n/d], row-major coefficient order
                        se += (quant_dirs(y, bk) - y).pow(2).sum().item()
                active = sum(1 for x in bits if x > 0)
                rec[f"{arm}@{rate}"] = {
                    "rel_mse": se / en, "bits": bits,
                    "bpw": rate + (active / d) * 16 / SCALE_GROUP
                           + (d * d * 16 / (E * m * n) if basis is not None else 0.0)}
        out[f"d{d}"] = rec
        log(f"  dense d={d}: amgm_bits={rec['bound_bits_saved_amgm']:.3f} "
            f"pca@3={rec['pca@3.0']['rel_mse']:.5f} ident@3={rec['ident@3.0']['rel_mse']:.5f}")
    return out


# ---------------------------------------------------------------- 3. sparse
def omp(X: torch.Tensor, D: torch.Tensor, s: int, chunk: int = 32768):
    """Batched orthogonal matching pursuit. X [N,d], D [K,d] unit rows."""
    N = X.shape[0]
    idx_all = torch.empty(N, s, dtype=torch.long, device=X.device)
    coef_all = torch.empty(N, s, device=X.device)
    eye = torch.eye(s, device=X.device)
    for a in range(0, N, chunk):
        x = X[a:a + chunk]
        r = x
        idx = torch.empty(x.shape[0], 0, dtype=torch.long, device=X.device)
        for t in range(s):
            corr = (r @ D.T).abs()
            if t:
                corr.scatter_(1, idx, -1.0)
            idx = torch.cat([idx, corr.argmax(1, keepdim=True)], 1)
            A = D[idx]                                            # [n, t+1, d]
            G = A @ A.transpose(1, 2) + 1e-6 * eye[: t + 1, : t + 1]
            c = torch.linalg.solve(G, A @ x.unsqueeze(2))         # [n, t+1, 1]
            r = x - (c.transpose(1, 2) @ A).squeeze(1)
        idx_all[a:a + chunk] = idx
        coef_all[a:a + chunk] = c.squeeze(2)
    return idx_all, coef_all


def reconstruct(D, idx, coef):
    return (coef.unsqueeze(1) @ D[idx]).squeeze(1)


def fit_dictionary(X: torch.Tensor, K: int, s: int, iters: int, gen) -> torch.Tensor:
    """MOD dictionary learning with dead-atom replacement."""
    N, d = X.shape
    keep = X.norm(dim=1) > 0
    Xn = X[keep]
    if Xn.shape[0] < K:
        raise ValueError(f"only {Xn.shape[0]} nonzero fit blocks for a {K}-atom dictionary")
    D = Xn[torch.randperm(Xn.shape[0], generator=gen, device="cpu")[:K].to(X.device)]
    D = D / D.norm(dim=1, keepdim=True)
    for it in range(iters):
        idx, coef = omp(X, D, s)
        R = X - reconstruct(D, idx, coef)
        err = (R.pow(2).sum() / X.pow(2).sum()).item()
        CtC = torch.zeros(K * K, dtype=torch.float64, device=X.device)
        CtC.index_add_(0, (idx.unsqueeze(2) * K + idx.unsqueeze(1)).reshape(-1),
                       (coef.unsqueeze(2) * coef.unsqueeze(1)).reshape(-1).double())
        CtC = CtC.view(K, K)
        CtX = torch.zeros(K, d, dtype=torch.float64, device=X.device)
        CtX.index_add_(0, idx.reshape(-1), (coef.unsqueeze(2) * X.unsqueeze(1)).reshape(-1, d).double())
        ridge = 1e-6 * CtC.diagonal().mean().clamp(min=1e-30)
        Dn = torch.linalg.solve(CtC + ridge * torch.eye(K, dtype=torch.float64, device=X.device), CtX).float()
        norms = Dn.norm(dim=1)
        used = torch.bincount(idx.reshape(-1), minlength=K) > 0
        dead = (~used) | (norms < 1e-12)
        if dead.any():  # replace with the worst-fit blocks
            worst = R.pow(2).sum(1).topk(int(dead.sum())).indices
            Dn[dead] = X[worst]
            norms = Dn.norm(dim=1)
        D = Dn / norms.clamp(min=1e-30).unsqueeze(1)
        if it == 0 or it == iters - 1 or (it + 1) % 5 == 0:
            log(f"    MOD it {it+1}/{iters}: fit rel_err={err:.5f} dead={int(dead.sum())}")
    return D


def part_sparse(W: torch.Tensor, dev, row_stride: int, n_fit: int, iters: int, gen) -> dict:
    E, m, n = W.shape
    eval_rows = torch.arange(0, m, row_stride)
    fit_rows = torch.arange(row_stride // 2, m, row_stride)
    out = {}
    for d, K, s, cb in SPARSE_CONFIGS:
        if n % d:
            continue
        log(f"  sparse d={d} K={K} s={s} coef_bits={cb}")
        Xfit = torch.cat([W[e][fit_rows].to(dev, torch.float32).reshape(-1, d) for e in range(E)])
        if Xfit.shape[0] > n_fit:
            Xfit = Xfit[torch.randperm(Xfit.shape[0], generator=gen)[:n_fit].to(dev)]
        D = fit_dictionary(Xfit, K, s, iters, gen)
        del Xfit
        w = torch.cat([W[e][eval_rows].to(dev, torch.float32) for e in range(E)])  # [E*r, n]
        X = w.reshape(-1, d)
        idx, coef = omp(X, D, s)
        se_f = (X - reconstruct(D, idx, coef)).pow(2).sum().item()
        # coefficient scales: groups of 128 consecutive coefficients within each weight row
        cq = quant_dirs(coef.reshape(w.shape[0], -1), cb).reshape(coef.shape)
        se_q = (X - reconstruct(D, idx, cq)).pow(2).sum().item()
        en = X.pow(2).sum().item()
        ref = {b: rtn_err(w, b)[0] for b in (3, 4)}
        del w, X, idx, coef, cq
        bpw = s * (math.log2(K) + cb) / d + (s / d) * 16 / SCALE_GROUP + K * d * 16 / (E * m * n)
        out[f"d{d}_K{K}_s{s}_c{cb}"] = {
            "rel_mse": se_q / en, "rel_mse_float_coefs": se_f / en, "bpw": bpw,
            "same_rows_int3_g128": ref[3] / en, "same_rows_int4_g128": ref[4] / en}
        log(f"    -> rel_mse={se_q/en:.5f} (float coefs {se_f/en:.5f}) bpw={bpw:.3f} "
            f"| same rows INT4={ref[4]/en:.5f} INT3={ref[3]/en:.5f}")
    return out


# ---------------------------------------------------------------- 4. experts
def part_experts(W: torch.Tensor, dev, chunk: int = 1 << 20) -> dict:
    E = W.shape[0]
    F = W.reshape(E, -1)
    G = torch.zeros(E, E, dtype=torch.float64, device=dev)
    for a in range(0, F.shape[1], chunk):
        c = F[:, a:a + chunk].to(dev, torch.float64)
        G += c @ c.T
    H = torch.eye(E, dtype=torch.float64, device=dev) - 1.0 / E
    rec = {}
    for name, M in (("raw", G), ("centered", H @ G @ H)):
        lam = torch.linalg.eigvalsh(M).flip(0).clamp(min=0)
        tot = lam.sum()
        rec[name] = {f"K{k}": (1 - lam[:k].sum() / tot).item() for k in EIG_KS if k <= E}
    rec["mean_expert_energy_fraction"] = (G.sum() / E / G.trace()).item()
    return rec


# ---------------------------------------------------------------- driver
def analyze(W: dict, dev, args, gen) -> dict:
    res = {}
    for p in PROJS:
        log(f"== {p}_proj {tuple(W[p].shape)}")
        Wp = W[p].to(dev)  # bf16, resident once; parts upcast per expert
        r = {"target": part_target(Wp, dev)}
        log(f"  target: " + " ".join(f"{k}={v['rel_mse']:.5f}" for k, v in r["target"].items()
                                     if isinstance(v, dict)))
        if not args.skip_dense:
            r["dense"] = part_dense(Wp, dev)
        if not args.skip_sparse:
            r["sparse"] = part_sparse(Wp, dev, args.eval_row_stride, args.n_fit, args.mod_iters, gen)
        if not args.skip_experts and Wp.shape[0] > 1:
            r["experts"] = part_experts(Wp, dev)
            log("  experts raw 1-eta: " + " ".join(f"{k}={v:.4f}" for k, v in r["experts"]["raw"].items()))
        res[p] = r
        del Wp
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return res


def gaussian_like(W: dict, experts: int, gen) -> dict:
    return {p: torch.randn((experts, *W[p].shape[1:]), generator=gen).to(torch.bfloat16) for p in PROJS}


def fmt(x):
    return f"{100*x:.3f}%"


def summarize(res: dict, ctrl: dict | None) -> str:
    L = ["# Basis-codec feasibility summary", "",
         "Relative weight MSE (lower is better). Target: beat INT4-g128 at <=3.2 bpw.", ""]
    for p, r in res.items():
        t = r["target"]
        L += [f"## {p}_proj", "",
              f"INT4-g128 {fmt(t['int4_g128']['rel_mse'])} @4.125 | INT3-g128 "
              f"{fmt(t['int3_g128']['rel_mse'])} @3.125 | INT2-g128 {fmt(t['int2_g128']['rel_mse'])} "
              f"@2.125 | row-normalized excess kurtosis {t['row_normalized_excess_kurtosis']:.2f}", ""]
        if "dense" in r:
            L += ["| d | AM/GM bound (bits saved) | top-quarter energy | PCA@3 | ident@3 | PCA@3 bpw | PCA@4 | ident@4 |",
                  "|---|---|---|---|---|---|---|---|"]
            for d, v in r["dense"].items():
                L.append(f"| {d[1:]} | {v['bound_bits_saved_amgm']:.3f} | {v['top_quarter_energy']:.3f} | "
                         f"{fmt(v['pca@3.0']['rel_mse'])} | {fmt(v['ident@3.0']['rel_mse'])} | "
                         f"{v['pca@3.0']['bpw']:.3f} | {fmt(v['pca@4.0']['rel_mse'])} | {fmt(v['ident@4.0']['rel_mse'])} |")
            L.append("")
        if "sparse" in r:
            L += ["| config | rel MSE | float-coef MSE | bpw | same-rows INT4 | same-rows INT3 | Gaussian ctrl |",
                  "|---|---|---|---|---|---|---|"]
            for k, v in r["sparse"].items():
                c = ctrl[p]["sparse"].get(k, {}).get("rel_mse") if ctrl and "sparse" in ctrl[p] else None
                L.append(f"| {k} | {fmt(v['rel_mse'])} | {fmt(v['rel_mse_float_coefs'])} | {v['bpw']:.3f} | "
                         f"{fmt(v['same_rows_int4_g128'])} | {fmt(v['same_rows_int3_g128'])} | "
                         f"{fmt(c) if c is not None else '-'} |")
            L.append("")
        if "experts" in r:
            ex = r["experts"]
            L += ["Eigen-experts, residual energy fraction 1-eta_K (no-residual error floor):", "",
                  "| K | " + " | ".join(k[1:] for k in ex["raw"]) + " |",
                  "|---|" + "---|" * len(ex["raw"]),
                  "| raw | " + " | ".join(f"{v:.4f}" for v in ex["raw"].values()) + " |",
                  "| centered | " + " | ".join(f"{v:.4f}" for v in ex["centered"].values()) + " |",
                  "", f"Mean-expert energy fraction: {ex['mean_expert_energy_fraction']:.4f}", ""]
    if ctrl:
        L += ["## iid-Gaussian control (same shapes)", ""]
        for p, r in ctrl.items():
            t = r["target"]
            line = f"- {p}: INT4 {fmt(t['int4_g128']['rel_mse'])}, INT3 {fmt(t['int3_g128']['rel_mse'])}"
            if "dense" in r:
                v = r["dense"].get("d16")
                if v:
                    line += f", dense d16 AM/GM bound {v['bound_bits_saved_amgm']:.3f} bits, PCA@3 {fmt(v['pca@3.0']['rel_mse'])}"
            L.append(line)
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="unquantized HF checkpoint dir")
    ap.add_argument("--layer", type=int, default=None, help="default: first MoE layer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-experts", type=int, default=None)
    ap.add_argument("--control-experts", type=int, default=16, help="0 disables the Gaussian control")
    ap.add_argument("--eval-row-stride", type=int, default=64)
    ap.add_argument("--n-fit", type=int, default=262144)
    ap.add_argument("--mod-iters", type=int, default=15)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-dense", action="store_true")
    ap.add_argument("--skip-sparse", action="store_true")
    ap.add_argument("--skip-experts", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    env = {"torch": torch.__version__, "device": str(dev),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "argv": sys.argv, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    log("env", env)

    W, meta = load_layer(args.ckpt, args.layer, args.max_experts, args.threads)
    res = analyze(W, dev, args, gen)
    ctrl = None
    if args.control_experts:
        log(f"== iid-Gaussian control, {args.control_experts} experts")
        ctrl = analyze(gaussian_like(W, args.control_experts, gen), dev, args, gen)

    blob = {"meta": meta, "env": env, "results": res, "gaussian_control": ctrl,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(blob, f, indent=1)
    summary = summarize(res, ctrl)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(summary)
    print("\n" + summary, flush=True)
    log("done ->", args.out)


if __name__ == "__main__":
    main()
