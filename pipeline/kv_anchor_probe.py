#!/usr/bin/env python3
"""KV-cache anchor predictability on GLM-5.3's MLA latent, early layers.

MLA caches one latent per token per layer: the kv_a_layernorm output
(kv_lora_rank values; the separate RoPE key part is left alone here).  The
question: does "one anchor per chunk of N tokens + low-bit residual" beat
quantizing each token directly at the same bits/value?

  K_t ~= P_t + Q(K_t - P_t),   P_t = alpha_t * a_c   (anchor a_c per chunk)

Measured per layer on real AA-LCR text:
  1. lag correlation of the mean-centred latent (how fast similarity decays)
  2. predictability: residual energy of each predictor vs the trivial
     global-mean control, plus a token-shuffled control
  3. quantized error at matched bits/value -- on the latent AND on the layer's
     real attention output (self_attn re-run with the latent swapped in)
  4. DSA clustering (layer 0, long context): how many distinct chunks the
     indexer's top-k touches, vs random selection

Predictors: direct (none), gmean (one per-channel mean), first (chunk's first
token), mean (chunk mean), rank1 (chunk's top singular direction + per-token
fp16 alpha), cbM (nearest of M k-means centroids fit on held-out windows).
Anchors are stored fp8 with one fp16 scale; residuals are computed against
the STORED anchor (closed loop).  bits/value counts residual bits, residual
scales, anchor, alpha and codebook index.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pipeline.basis_codec_layer_analysis import log, quant_groups

LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CHUNKS = (4, 8, 16, 32, 64, 128)
ATTN_CHUNKS = (16, 64)
BITS = (2, 3, 4)
GROUPINGS = ("tok", "ch")          # residual scale groups: 128 channels of a token / 128 tokens of a channel
CB_SIZES = (256, 4096)
G = 128
FP8_MAX = 448.0


# ---------------------------------------------------------------- codecs
def fp8_rows(x: torch.Tensor) -> torch.Tensor:
    """fp8-e4m3 with one fp16 scale per row (last dim)."""
    s = (x.abs().amax(-1, keepdim=True) / FP8_MAX).clamp(min=1e-12).half().float()
    # clamp: fp32/fp16 rounding can push the row max past 448, which e4m3fn does not saturate
    return (x / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s


def fp8_unscaled(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()


def quant_residual(R: torch.Tensor, grouping: str, bits: int) -> torch.Tensor:
    T, d = R.shape
    if grouping == "tok":
        return quant_groups(R.reshape(-1, G), bits).reshape(T, d)
    pad = (-T) % G
    Rt = F.pad(R.T, (0, pad))
    return quant_groups(Rt.reshape(-1, G), bits).reshape(d, -1)[:, :T].T.contiguous()


def nearest(X: torch.Tensor, C: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    cn = C.pow(2).sum(1)
    return torch.cat([(2 * X[a:a + chunk] @ C.T - cn).argmax(1) for a in range(0, X.shape[0], chunk)])


def kmeans(X: torch.Tensor, M: int, iters: int, gen: torch.Generator) -> torch.Tensor:
    if X.shape[0] < M:
        raise ValueError(f"{X.shape[0]} training tokens for a {M}-centroid codebook")
    # k-means++ seeding: each new centroid drawn with probability ~ squared distance
    first = int(torch.randint(0, X.shape[0], (1,), generator=gen))
    C = torch.empty(M, X.shape[1], device=X.device, dtype=X.dtype)
    C[0] = X[first]
    d2 = (X - C[0]).pow(2).sum(1)
    for j in range(1, M):
        p = (d2 / d2.sum().clamp(min=1e-30)).cpu()
        C[j] = X[int(torch.multinomial(p, 1, generator=gen))]
        d2 = torch.minimum(d2, (X - C[j]).pow(2).sum(1))
    for _ in range(iters):
        idx = nearest(X, C)
        sums = torch.zeros_like(C).index_add_(0, idx, X)
        cnt = torch.bincount(idx, minlength=M)
        empty = cnt == 0
        C = torch.where(empty[:, None], C, sums / cnt.clamp(min=1)[:, None])
        if empty.any():
            C[empty] = X[torch.randperm(X.shape[0], generator=gen)[: int(empty.sum())].to(X.device)]
    return C.half().float()


def predict(X: torch.Tensor, scheme: str, N: int | None, ctx: dict) -> tuple[torch.Tensor, float]:
    """(prediction P [T, d], anchor bits/value)."""
    T, d = X.shape
    if scheme == "direct":
        return torch.zeros_like(X), 0.0
    if scheme == "gmean":
        return ctx["mu"].expand(T, d), 0.0
    if scheme.startswith("cb"):
        C = ctx[scheme]
        return C[nearest(X, C)], math.log2(C.shape[0]) / d
    if T % N:
        raise ValueError(f"window {T} not divisible by chunk {N}")
    Xc = X.view(T // N, N, d)
    cost = (8 * d + 16) / (N * d)
    if scheme == "first":
        P = fp8_rows(Xc[:, 0])[:, None].expand_as(Xc)
    elif scheme == "mean":
        P = fp8_rows(Xc.mean(1))[:, None].expand_as(Xc)
    elif scheme == "rank1":
        a = fp8_rows(torch.linalg.svd(Xc, full_matrices=False).Vh[:, 0])        # [nc, d], stored
        alpha = (Xc @ a.unsqueeze(-1)).squeeze(-1).half().float()             # fp16 per token
        P = alpha.unsqueeze(-1) * a[:, None]
        cost += 16 / d
    else:
        raise ValueError(scheme)
    return P.reshape(T, d), cost


def all_arms(chunks=CHUNKS, cbs=CB_SIZES) -> list[tuple]:
    arms = [("fp8_unscaled", None, None, None), ("fp8_tok", None, None, None)]
    for s in ("direct", "gmean", *(f"cb{m}" for m in cbs)):
        arms += [(s, None, g, b) for g in GROUPINGS for b in BITS]
    for s in ("first", "mean", "rank1"):
        arms += [(s, n, g, b) for n in chunks for g in GROUPINGS for b in BITS]
    return arms


def attn_arms(arms: list[tuple], attn_chunks=ATTN_CHUNKS, cb_attn: str = f"cb{CB_SIZES[-1]}") -> list[tuple]:
    keep = {"fp8_unscaled", "fp8_tok", "direct", "gmean", cb_attn}
    return [a for a in arms if a[0] in keep or (a[1] in attn_chunks)]


def apply_arm(X: torch.Tensor, arm: tuple, ctx: dict) -> tuple[torch.Tensor, float]:
    s, N, g, b = arm
    if s == "fp8_unscaled":
        return fp8_unscaled(X), 8.0
    if s == "fp8_tok":
        return fp8_rows(X), 8 + 16 / X.shape[1]
    P, cost = predict(X, s, N, ctx)
    return P + quant_residual(X - P, g, b), b + 16 / G + cost


def arm_name(arm):
    s, N, g, b = arm
    return s if g is None else (f"{s}-N{N}-{g}-int{b}" if N else f"{s}-{g}-int{b}")


# ---------------------------------------------------------------- model hooks
class Tap:
    """Hooks on each layer's kv_a_layernorm (the cached latent) and self_attn."""

    def __init__(self, layers):
        self.latent, self.kwargs, self.out, self.topk = {}, {}, {}, {}
        self.replace: dict[int, torch.Tensor] = {}
        self.record = False
        self.handles = []
        for i, layer in enumerate(layers):
            att = layer.self_attn
            self.handles.append(att.kv_a_layernorm.register_forward_hook(self._norm_hook(i)))
            self.handles.append(att.register_forward_pre_hook(self._pre_hook(i), with_kwargs=True))
            self.handles.append(att.register_forward_hook(self._out_hook(i)))

    def _norm_hook(self, i):
        def hook(mod, inp, out):
            if self.record:
                self.latent[i] = out.detach()
            if i in self.replace:
                return self.replace[i].to(out.dtype)
        return hook

    def _pre_hook(self, i):
        def hook(mod, args, kwargs):
            if self.record:
                self.kwargs[i] = (args, dict(kwargs))
        return hook

    def _out_hook(self, i):
        def hook(mod, inp, out):
            if self.record:
                self.out[i] = out[0].detach()
                self.topk[i] = out[2].detach() if len(out) > 2 and out[2] is not None else None
        return hook

    def rerun(self, layers, i, latent: torch.Tensor | None) -> torch.Tensor:
        args, kwargs = self.kwargs[i]
        if latent is not None:
            self.replace[i] = latent
        try:
            return layers[i].self_attn(*args, **kwargs)[0]
        finally:
            self.replace.pop(i, None)

    def close(self):
        for h in self.handles:
            h.remove()


def base_model(model):
    return model.model if hasattr(model, "model") else model


@torch.no_grad()
def forward_window(model, tap: Tap, ids: torch.Tensor):
    tap.latent.clear(), tap.kwargs.clear(), tap.out.clear(), tap.topk.clear()
    tap.record = True
    try:
        base_model(model)(input_ids=ids[None], use_cache=False)
    finally:
        tap.record = False


# ---------------------------------------------------------------- DSA clustering (layer 0)
@torch.no_grad()
def layer0_topk_chunked(model, ids: torch.Tensor, q_chunk: int = 256) -> torch.Tensor:
    """Layer-0 DSA indexer top-k for a long sequence, computed in query chunks
    (the module materializes [S, H, S]).  Same math as GlmMoeDsaIndexer.forward."""
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as mm

    bm = base_model(model)
    layer = bm.layers[0]
    att, idx = layer.self_attn, layer.self_attn.indexer
    h = layer.input_layernorm(bm.embed_tokens(ids[None]))
    pos = torch.arange(ids.shape[0], device=ids.device)[None]
    cos, sin = bm.rotary_emb(h, position_ids=pos)
    q_resid = att.q_a_layernorm(att.q_a_proj(h))
    B, S, _ = h.shape
    q = idx.wq_b(q_resid).view(B, S, idx.n_heads, idx.head_dim)
    q_rot, q_pass = torch.split(q, [idx.qk_rope_head_dim, idx.head_dim - idx.qk_rope_head_dim], dim=-1)
    k = idx.k_norm(idx.wk(h)).unsqueeze(2)
    k_rot, k_pass = torch.split(k, [idx.qk_rope_head_dim, idx.head_dim - idx.qk_rope_head_dim], dim=-1)
    q_rot, k_rot = mm.apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_rot, q_pass], -1)
    k = torch.cat([k_rot, k_pass], -1).squeeze(2)                       # [B, S, D]
    w = idx.weights_proj(h.to(idx.weights_proj.weight.dtype)).float() * (idx.n_heads ** -0.5)
    topk = min(idx.index_topk, S)
    out = torch.empty(S, topk, dtype=torch.long, device=ids.device)
    kT = k.transpose(-1, -2).float().unsqueeze(1)                          # [B, 1, D, S]
    for a in range(0, S, q_chunk):
        sc = F.relu(torch.matmul(q[:, a:a + q_chunk].float(), kT) * idx.softmax_scale)   # [B, c, H, S]
        sc = torch.matmul(w[:, a:a + q_chunk].unsqueeze(-2), sc).squeeze(-2)          # [B, c, S]
        causal = torch.arange(S, device=ids.device)[None, :] > torch.arange(a, a + sc.shape[1], device=ids.device)[:, None]
        out[a:a + sc.shape[1]] = sc[0].masked_fill(causal, float("-inf")).topk(topk, dim=-1).indices
    return out


def topk_overlap(a: torch.Tensor, b: torch.Tensor, start: int) -> float:
    """Mean set overlap of two top-k index tensors [S, k] over query rows >= start."""
    fr = []
    for s in range(start, a.shape[0]):
        x, y = set(a[s].tolist()), set(b[s].tolist())
        fr.append(len(x & y) / max(1, len(x)))
    return sum(fr) / max(1, len(fr))


def chunk_clustering(topk: torch.Tensor, chunks=(16, 64), stride: int = 256) -> dict:
    """Distinct chunks touched by each query's selection vs uniform random selection.
    Only rows whose past exceeds k tokens (a real selection) are counted."""
    S, k = topk.shape
    out = {}
    for N in chunks:
        reads, ratio = [], []
        for s in range(k, S, stride):
            sel = topk[s]
            sel = sel[sel <= s]
            D = torch.unique(sel // N).numel()
            nc = s // N + 1
            p_empty = (1 - sel.numel() / (s + 1)) ** N
            reads.append(D / sel.numel())
            ratio.append(D / (nc * (1 - p_empty)))
        if reads:
            out[f"N{N}"] = {"chunks_per_selected_token": sum(reads) / len(reads),
                            "vs_random_selection": sum(ratio) / len(ratio), "queries": len(reads)}
    return out


# ---------------------------------------------------------------- probe
def lag_stats(X: torch.Tensor, mu: torch.Tensor, acc: dict):
    Xc = X - mu
    for L in LAGS:
        if L >= X.shape[0]:
            continue
        a, b = Xc[:-L], Xc[L:]
        s = acc.setdefault(L, [0.0, 0.0, 0.0])
        s[0] += (a * b).sum().item()
        s[1] += a.pow(2).sum().item()
        s[2] += b.pow(2).sum().item()


@torch.no_grad()
def probe(model, train_windows, eval_windows, *, chunks=CHUNKS, attn_chunks=ATTN_CHUNKS, cbs=CB_SIZES,
          kmeans_iters=10, seed=0, long_ids=None) -> dict:
    gen = torch.Generator().manual_seed(seed)
    layers = base_model(model).layers
    tap = Tap(layers)
    L = len(layers)
    try:
        # pass 1: fit per-channel mean and codebooks on held-out windows
        train = {i: [] for i in range(L)}
        for w in train_windows:
            forward_window(model, tap, w)
            for i in range(L):
                train[i].append(tap.latent[i][0].float())
        ctx = {}
        for i in range(L):
            Xtr = torch.cat(train[i])
            ctx[i] = {"mu": Xtr.mean(0, keepdim=True).half().float()}
            for m in cbs:
                ctx[i][f"cb{m}"] = kmeans(Xtr, m, kmeans_iters, gen)
        del train
        log(f"fitted means + codebooks {cbs} on {len(train_windows)} windows")

        arms = all_arms(chunks, cbs)
        aarms = set(attn_arms(arms, attn_chunks, f"cb{cbs[-1]}"))
        preds = [("gmean", None), *((f"cb{m}", None) for m in cbs),
                 *((s, n) for s in ("first", "mean", "rank1") for n in chunks)]
        acc = {i: {"lat": {}, "attn": {}, "bits": {}, "pred": {}, "pred_shuf": {}, "lag": {},
                   "energy": 0.0, "energy_centered": 0.0, "noop_attn_rel": 0.0} for i in range(L)}

        # pass 2: evaluate
        for wi, w in enumerate(eval_windows):
            forward_window(model, tap, w)
            for i in range(L):
                a = acc[i]
                X = tap.latent[i][0].float()
                mu = ctx[i]["mu"]
                a["energy"] += X.pow(2).sum().item()
                a["energy_centered"] += (X - mu).pow(2).sum().item()
                lag_stats(X, mu, a["lag"])
                Xs = X[torch.randperm(X.shape[0], generator=gen).to(X.device)]
                for s, n in preds:
                    key = f"{s}-N{n}" if n else s
                    a["pred"][key] = a["pred"].get(key, 0.0) + (X - predict(X, s, n, ctx[i])[0]).pow(2).sum().item()
                    if n:
                        a["pred_shuf"][key] = a["pred_shuf"].get(key, 0.0) + (Xs - predict(Xs, s, n, ctx[i])[0]).pow(2).sum().item()
                o0 = tap.out[i].float()
                on = o0.pow(2).sum().item()
                a["noop_attn_rel"] = max(a["noop_attn_rel"],
                                         (tap.rerun(layers, i, tap.latent[i]).float() - o0).pow(2).sum().item() / on)
                for arm in arms:
                    Xh, bits = apply_arm(X, arm, ctx[i])
                    nm = arm_name(arm)
                    a["bits"][nm] = bits
                    a["lat"][nm] = a["lat"].get(nm, 0.0) + (Xh - X).pow(2).sum().item()
                    if arm in aarms:
                        o = tap.rerun(layers, i, Xh[None]).float()
                        a["attn"][nm] = a["attn"].get(nm, [0.0, 0.0])
                        a["attn"][nm][0] += (o - o0).pow(2).sum().item()
                        a["attn"][nm][1] += on
            log(f"eval window {wi + 1}/{len(eval_windows)} done")

        res = {"layers": {}}
        for i in range(L):
            a = acc[i]
            E, Ec = a["energy"], a["energy_centered"]
            res["layers"][i] = {
                "centered_energy_fraction": Ec / E,
                "noop_attn_rel": a["noop_attn_rel"],
                "lag_corr": {str(k): v[0] / math.sqrt(v[1] * v[2]) for k, v in a["lag"].items()},
                "pred_resid_vs_gmean": {k: v / Ec for k, v in a["pred"].items()},
                "pred_resid_vs_gmean_shuffled": {k: v / Ec for k, v in a["pred_shuf"].items()},
                "arms": {nm: {"bits": a["bits"][nm], "latent_rel_mse": a["lat"][nm] / E,
                              "attn_rel_mse": (a["attn"][nm][0] / a["attn"][nm][1]) if nm in a["attn"] else None}
                         for nm in a["lat"]},
            }
        if long_ids is not None:
            cfg = model.config
            if cfg.indexer_types[0] != "full":
                res["dsa_layer0"] = {"skipped": f"indexer_types[0]={cfg.indexer_types[0]}"}
            else:
                # validate the chunked indexer against the module on the last eval window
                forward_window(model, tap, eval_windows[-1])
                tap_topk = tap.topk[0][0].long()
                mine = layer0_topk_chunked(model, eval_windows[-1])
                k = tap_topk.shape[1]
                overlap = topk_overlap(mine, tap_topk, start=k)
                if overlap < 0.99:
                    raise RuntimeError(f"chunked layer-0 indexer disagrees with the module: overlap {overlap:.4f}")
                res["dsa_layer0"] = {"validation_overlap": overlap, "index_topk": k, "docs": []}
                for ids in long_ids:
                    t = layer0_topk_chunked(model, ids)
                    res["dsa_layer0"]["docs"].append({"tokens": ids.shape[0], **chunk_clustering(t, attn_chunks)})
                    log(f"layer-0 DSA clustering on {ids.shape[0]} tokens done")
        return res
    finally:
        tap.close()


# ---------------------------------------------------------------- data / model
def aa_lcr_sets(root: Path) -> dict[str, str]:
    sets = {}
    base = root / "extracted_text" / "lcr"
    for d in sorted(p for p in base.glob("*/*") if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.is_file())
        if files:
            sets[f"{d.parent.name}/{d.name}"] = "\n\n".join(f.read_text(encoding="utf-8") for f in files)
    if len(sets) < 30:
        sys.exit(f"FAIL: found {len(sets)} AA-LCR document sets under {base}, expected 30")
    return sets


def make_windows(tok, sets: dict[str, str], W: int, offset: int, dev):
    train, evals, names = [], [], []
    for j, (name, text) in enumerate(sorted(sets.items())):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < W:
            continue
        o = min(offset, len(ids) - W)
        (train if j % 2 == 0 else evals).append(torch.tensor(ids[o:o + W], device=dev))
        names.append((name, "train" if j % 2 == 0 else "eval", len(ids)))
    return train, evals, names


def tiny_config(base_cfg=None):
    """A small GlmMoeDsa config with the real per-layer types, for the preflight."""
    from transformers import GlmMoeDsaConfig

    d = (base_cfg.to_dict() if base_cfg is not None else GlmMoeDsaConfig().to_dict())
    n = 4
    d.update(hidden_size=256, intermediate_size=512, moe_intermediate_size=128, n_routed_experts=4,
             num_experts_per_tok=2, n_group=1, topk_group=1, n_shared_experts=1, num_attention_heads=4,
             num_key_value_heads=4, q_lora_rank=128, kv_lora_rank=128, qk_rope_head_dim=16,
             qk_nope_head_dim=32, v_head_dim=32, index_n_heads=2, index_head_dim=32, index_topk=32,
             num_hidden_layers=n, vocab_size=1000, num_nextn_predict_layers=0, first_k_dense_replace=3,
             pad_token_id=0, bos_token_id=1, eos_token_id=2)   # real ids exceed the tiny vocab
    for key in ("mlp_layer_types", "indexer_types", "layer_types"):
        if isinstance(d.get(key), list):
            d[key] = d[key][:n]
    if isinstance(d.get("mlp_layer_types"), list):
        d["mlp_layer_types"] = ["dense"] * 3 + ["sparse"]
    for key in ("qk_head_dim", "head_dim", "_name_or_path", "transformers_version", "architectures",
                "quantization_config", "auto_map"):
        d.pop(key, None)
    return GlmMoeDsaConfig(**d)


def load_model(path, dev):
    """CPU load then move: device_map needs `accelerate`, which the serving image lacks."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()


def run_preflight(base_cfg, dev) -> dict:
    import tempfile

    from transformers import GlmMoeDsaForCausalLM

    torch.manual_seed(0)
    cfg = tiny_config(base_cfg)
    with tempfile.TemporaryDirectory() as tmp:   # exercise the same save/load path as the real model
        GlmMoeDsaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(tmp)
        model = load_model(tmp, dev)
    W = 256
    g = torch.Generator().manual_seed(0)
    mk = lambda: torch.randint(0, cfg.vocab_size, (W,), generator=g).to(dev)
    res = probe(model, [mk() for _ in range(3)], [mk() for _ in range(2)], chunks=(8, 16), attn_chunks=(16,),
                cbs=(16,), kmeans_iters=2, long_ids=[torch.randint(0, cfg.vocab_size, (512,), generator=g).to(dev)])
    for i, r in res["layers"].items():
        if r["noop_attn_rel"] > 1e-6:
            raise RuntimeError(f"preflight: re-running layer {i} attention with its own latent is not exact")
        for nm, v in r["arms"].items():
            vals = [v["latent_rel_mse"]] + ([v["attn_rel_mse"]] if v["attn_rel_mse"] is not None else [])
            if not all(math.isfinite(x) for x in vals):
                raise RuntimeError(f"preflight: non-finite error for {nm} at layer {i}")
    return res


def fmt(x):
    return "-" if x is None else f"{100 * x:.3f}%"


def summarize(res: dict, meta: dict) -> str:
    L = ["# KV anchor probe — GLM-5.3 MLA latent", "",
         f"Layers {list(res['layers'])}, eval windows {meta['eval_windows']} x {meta['window']} tokens, "
         f"kv_lora_rank {meta['kv_lora_rank']}.", ""]
    for i, r in res["layers"].items():
        L += [f"## layer {i}", "",
              f"centered energy fraction {r['centered_energy_fraction']:.4f} (1 - share of the global per-channel mean); "
              f"no-op attention re-run error {r['noop_attn_rel']:.2e}", "",
              "Lag correlation of the centred latent: "
              + ", ".join(f"{k}:{v:.3f}" for k, v in r["lag_corr"].items()), "",
              "| predictor | residual / gmean residual | bits saved vs gmean | token-shuffled |", "|---|---|---|---|"]
        for k, v in r["pred_resid_vs_gmean"].items():
            sh = r["pred_resid_vs_gmean_shuffled"].get(k)
            L.append(f"| {k} | {v:.4f} | {0.5 * math.log2(1 / max(v, 1e-12)):.3f} | {'-' if sh is None else f'{sh:.4f}'} |")
        L += ["", "| arm | bits/value | latent rel MSE | attention rel MSE |", "|---|---|---|---|"]
        for nm, v in sorted(r["arms"].items(), key=lambda kv: (kv[1]["attn_rel_mse"] is None, kv[1]["bits"], kv[0])):
            if v["attn_rel_mse"] is not None:
                L.append(f"| {nm} | {v['bits']:.3f} | {fmt(v['latent_rel_mse'])} | {fmt(v['attn_rel_mse'])} |")
        L.append("")
    if "dsa_layer0" in res:
        L += ["## DSA clustering, layer 0", "", "```", json.dumps(res["dsa_layer0"], indent=1), "```", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, help="BF16 HF snapshot")
    ap.add_argument("--subset-dir", required=True, help="where to build the depth-truncated copy (local disk)")
    ap.add_argument("--layers", type=int, default=4, help="keep decoder layers [0, LAYERS)")
    ap.add_argument("--aa-lcr-root", required=True, help="AA-LCR work dir holding extracted_text/lcr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--offset", type=int, default=1024, help="skip each document's first tokens")
    ap.add_argument("--long-context", type=int, default=32768)
    ap.add_argument("--long-docs", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    from transformers import AutoConfig, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    import transformers
    meta = {"transformers": transformers.__version__, "torch": torch.__version__, "window": args.window,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    log("env", meta)

    base_cfg = AutoConfig.from_pretrained(args.snapshot)
    for k in ("kv_lora_rank", "qk_rope_head_dim", "num_attention_heads", "index_topk", "index_n_heads",
              "index_head_dim", "first_k_dense_replace"):
        meta[k] = getattr(base_cfg, k, None)
    meta["indexer_types_head"] = list(getattr(base_cfg, "indexer_types", [])[: args.layers])
    log("config", {k: meta[k] for k in meta if k not in ("started_utc",)})
    if base_cfg.kv_lora_rank % G:
        sys.exit(f"FAIL: kv_lora_rank {base_cfg.kv_lora_rank} not a multiple of {G}")

    log("preflight on a tiny random model with the real layer types")
    run_preflight(base_cfg, dev)
    log("preflight PASS")

    # data first: fail in seconds, not after the subset build
    tok = AutoTokenizer.from_pretrained(args.snapshot)
    sets = aa_lcr_sets(Path(args.aa_lcr_root))
    train, evals, names = make_windows(tok, sets, args.window, args.offset, dev)
    meta.update(train_windows=len(train), eval_windows=len(evals), windows=names)
    log(f"{len(train)} train / {len(evals)} eval windows of {args.window} tokens")
    long_ids = []
    for name, text in sorted(sets.items())[1::2]:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= args.long_context:
            long_ids.append(torch.tensor(ids[: args.long_context], device=dev))
        if len(long_ids) == args.long_docs:
            break
    log(f"{len(long_ids)} long-context docs of {args.long_context} tokens")

    sub = Path(args.subset_dir)
    if not (sub / "config.json").exists():
        from pipeline.build_subset_checkpoint import build
        t0 = time.time()
        build(Path(args.snapshot), sub, args.layers)
        log(f"subset built in {time.time() - t0:.0f}s")
    model = load_model(sub, dev)
    log("model loaded", type(model).__name__, f"{len(base_model(model).layers)} layers")

    res = probe(model, train, evals, long_ids=long_ids or None)
    blob = {"meta": meta, "results": res, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(args.out, "kv_anchor_results.json"), "w") as f:
        json.dump(blob, f, indent=1)
    summary = summarize(res, meta)
    with open(os.path.join(args.out, "kv_anchor_summary.md"), "w") as f:
        f.write(summary)
    print("\n" + summary, flush=True)
    log("done ->", args.out)


if __name__ == "__main__":
    main()
