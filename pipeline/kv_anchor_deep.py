#!/usr/bin/env python3
"""KV anchor probe at every depth of GLM-5.3, streaming one decoder layer at a time.

A deep layer's MLA latent depends on every layer below it, and the model does
not fit on a node in BF16. So the model is streamed: all probe windows are
pushed through layer 0 with their hidden states kept on the GPU, then layer 1
is swapped in, and so on. Each layer is written as a one-layer checkpoint and
loaded with transformers' own loader (which handles the MoE weight format);
the next layer's shards are prefetched from cephfs to local disk meanwhile.

--weights w4afp8 streams our SERVED SGLang w4afp8 checkpoint, dequantizing it
exactly (int4 experts: nibble unpack x per-128 scale; block-FP8 layers:
128x128 block dequant), with optional FP8 activation emulation (--act-fp8:
dynamic per-token-group FP8 on block-FP8 linear inputs, static scale-1.0 FP8
cast on expert inputs as the serve does; the expert down_proj input is not
emulated). --weights bf16 streams the BF16 source.

Codecs at each layer, scored on real attention-output error (kv_anchor_probe):
  cb4096 / cb256 + INT2/3/4 residual (ours)
  lexico45  Lexico-style sparse code: 4096-atom per-layer dictionary, s=45 FP8
            coefficients + 16-bit indices, no residual (our re-implementation:
            MOD/OMP dictionary, not their Adam training; no FP16 recent-token
            buffer for any arm)
  qvg64     Quant VideoGen-style per-window k-means (64 BF16 centroids) + INT2
Held-out AA-LCR windows and windows of GLM-5.3's own generated text are scored
separately; everything is fit on AA-LCR training windows only.

Gates: a tiny model must stream to the same latents as its full forward; each
layer must load with no missing/unexpected weights; re-running attention with
the layer's own latent must be exact; (w4afp8) dequantized weights must sit at
quantization-level distance from the BF16 source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from pipeline import basis_codec_layer_analysis as bc
from pipeline import kv_anchor_probe as kp

log = bc.log
BITS = (2, 3, 4)
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_ALLOWED_MISSING = re.compile(r"(embed_tokens|lm_head|model\.norm)\.")


# ---------------------------------------------------------------- weights
def copy_parallel(src: Path, dst: Path, threads: int = 16, chunk: int = 64 << 20) -> int:
    """src -> dst with parallel range reads (cephfs is ~30 MB/s per stream)."""
    size = src.stat().st_size
    tmp = dst.with_name(dst.name + ".part")
    with open(tmp, "wb") as f:
        f.truncate(size)

    def work(off):
        with open(src, "rb", buffering=0) as s, open(tmp, "r+b", buffering=0) as d:
            s.seek(off)
            d.seek(off)
            d.write(s.read(min(chunk, size - off)))

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(work, range(0, size, chunk)))
    os.replace(tmp, dst)
    return size


def dequantize_sglang_w4afp8(raw: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], set[str]]:
    """SGLang w4afp8 tensors -> BF16. Returns (tensors, stems of block-FP8 linears)."""
    from pipeline.sglang_w4afp8_kernels import dequantize_block_fp8, unpack_nibbles_int8

    out, fp8 = {}, set()
    for k, t in raw.items():
        if k.endswith((".weight_scale_inv", ".input_scale")):
            continue
        s = raw.get(k[: -len("weight")] + "weight_scale_inv") if k.endswith(".weight") else None
        if s is None:
            out[k] = t
        elif t.dtype == torch.int8:                       # int4 expert, group along input
            q = unpack_nibbles_int8(t).float()
            out[k] = (q * s.float().repeat_interleave(q.shape[1] // s.shape[1], dim=1)).to(torch.bfloat16)
        elif t.dtype == torch.float8_e4m3fn:              # 128x128 block FP8
            out[k] = dequantize_block_fp8(t, s.float()).to(torch.bfloat16)
            fp8.add(k[: -len(".weight")])
        else:
            raise ValueError(f"{k}: scale present but dtype {t.dtype} is neither int8 nor fp8")
    return out, fp8


class LayerSource:
    """Stages one decoder layer's shards on local disk and writes it as a
    one-layer checkpoint loadable by AutoModelForCausalLM."""

    def __init__(self, snapshot: str, stage: str, threads: int = 16, dequant: bool = False):
        self.snapshot, self.stage = Path(snapshot), Path(stage)
        self.stage.mkdir(parents=True, exist_ok=True)
        self.threads, self.dequant = threads, dequant
        self.cfg = json.loads((self.snapshot / "config.json").read_text())
        idx = self.snapshot / "model.safetensors.index.json"
        if idx.exists():
            self.wmap = json.loads(idx.read_text())["weight_map"]
        else:   # single-file checkpoint (tiny preflight)
            from safetensors import safe_open
            with safe_open(str(self.snapshot / "model.safetensors"), framework="pt") as h:
                self.wmap = {k: "model.safetensors" for k in h.keys()}
        self.by_layer: dict[int, list[str]] = {}
        self.other: list[str] = []
        for k in self.wmap:
            m = _LAYER.search(k)
            (self.by_layer.setdefault(int(m[1]), []) if m else self.other).append(k)
        self.n_layers = int(self.cfg["num_hidden_layers"])
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}
        self.bytes, self.seconds = 0, 0.0

    def shards(self, L: int) -> set[str]:
        return {self.wmap[k] for k in self.by_layer[L]}

    def fetch(self, name: str) -> Path:
        dst = self.stage / name
        with self._lock:
            ev = self._inflight.get(name)
            mine = ev is None and not dst.exists()
            if mine:
                ev = self._inflight[name] = threading.Event()
        if mine:
            t0 = time.time()
            try:
                n = copy_parallel(self.snapshot / name, dst, self.threads)
                with self._lock:
                    self.bytes += n
                    self.seconds += time.time() - t0
            finally:                       # never leave a waiter hanging on a failed copy
                with self._lock:
                    del self._inflight[name]
                ev.set()
        elif ev is not None:
            ev.wait()
            if not dst.exists():           # the prefetching thread failed: copy in the foreground
                return self.fetch(name)
        return dst

    def prefetch(self, L: int):
        def work():
            try:
                for s in sorted(self.shards(L)):
                    self.fetch(s)
            except Exception as ex:        # the foreground fetch retries and raises properly
                log(f"prefetch of layer {L} failed: {ex!r}")
        if L < self.n_layers:
            threading.Thread(target=work, daemon=True).start()

    def release(self, keep_from: int):
        """Delete staged shards that no layer >= keep_from needs (non-layer shards are kept)."""
        needed = {self.wmap[k] for k in self.other}
        for L in range(keep_from, self.n_layers):
            needed |= self.shards(L)
        for p in self.stage.glob("*.safetensors"):
            if p.name not in needed:
                p.unlink(missing_ok=True)

    def tensor(self, key: str) -> torch.Tensor:
        from safetensors import safe_open
        with safe_open(str(self.fetch(self.wmap[key])), framework="pt") as h:
            return h.get_tensor(key)

    def layer_config(self, L: int) -> dict:
        d = dict(self.cfg)
        n0, mtp = self.n_layers, int(self.cfg.get("num_nextn_predict_layers") or 0)
        for k, v in self.cfg.items():
            if isinstance(v, list) and len(v) in {n0, n0 + mtp}:
                d[k] = [v[L]]
        d.update(num_hidden_layers=1, num_nextn_predict_layers=0, vocab_size=8, tie_word_embeddings=False,
                 pad_token_id=0, bos_token_id=1, eos_token_id=2)
        if "mlp_layer_types" not in self.cfg:
            d["first_k_dense_replace"] = 1 if L < int(self.cfg.get("first_k_dense_replace", 0)) else 0
        d.pop("quantization_config", None)
        return d

    def raw_layer(self, L: int) -> dict[str, torch.Tensor]:
        from safetensors import safe_open
        out = {}
        for name in sorted(self.shards(L)):
            with safe_open(str(self.fetch(name)), framework="pt") as h:
                for k in self.by_layer[L]:
                    if self.wmap[k] == name:
                        out[k] = h.get_tensor(k)
        return out

    def build(self, L: int) -> tuple[Path, set[str]]:
        from safetensors.torch import save_file
        raw = self.raw_layer(L)
        tensors, fp8 = dequantize_sglang_w4afp8(raw) if self.dequant else (raw, set())
        rename = lambda k: k.replace(f"layers.{L}.", "layers.0.", 1)
        out = self.stage / f"layer{L:03d}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        save_file({rename(k): v.contiguous() for k, v in tensors.items()}, str(out / "model.safetensors"),
                  metadata={"format": "pt"})
        (out / "config.json").write_text(json.dumps(self.layer_config(L)))
        return out, {rename(s) for s in fp8}


def fp8_token_group(x: torch.Tensor, g: int = 128) -> torch.Tensor:
    """Dynamic per-token-group FP8 (e4m3), the activation scheme of block-FP8 linears."""
    shp = x.shape
    xg = x.float().reshape(*shp[:-1], shp[-1] // g, g)
    s = (xg.abs().amax(-1, keepdim=True) / kp.FP8_MAX).clamp(min=1e-12)
    return ((xg / s).clamp(-kp.FP8_MAX, kp.FP8_MAX).to(torch.float8_e4m3fn).float() * s).reshape(shp).to(x.dtype)


def add_act_fp8_hooks(model, fp8_stems: set[str]) -> int:
    n = 0
    for name, mod in model.named_modules():
        if name in fp8_stems and isinstance(mod, torch.nn.Linear) and mod.in_features % 128 == 0:
            mod.register_forward_pre_hook(lambda m, a: (fp8_token_group(a[0]), *a[1:]))
            n += 1
        elif type(mod).__name__.endswith("NaiveMoe"):      # static scale 1.0 FP8 on expert inputs
            mod.register_forward_pre_hook(lambda m, a: (kp.fp8_unscaled(a[0].float()).to(a[0].dtype), *a[1:]))
            n += 1
    return n


def load_layer(src: LayerSource, L: int, dev, act_fp8: bool = False):
    from transformers import AutoModelForCausalLM

    path, fp8 = src.build(L)
    model, info = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager",
                                                       output_loading_info=True)
    shutil.rmtree(path, ignore_errors=True)
    missing = [k for k in info.get("missing_keys", []) if not _ALLOWED_MISSING.search(k)]
    bad = missing + list(info.get("unexpected_keys", [])) + [str(x) for x in info.get("mismatched_keys", [])]
    if bad:
        raise RuntimeError(f"layer {L}: weights did not load cleanly, e.g. {bad[:5]}")
    model = model.to(dev).eval()
    hooks = add_act_fp8_hooks(model, fp8) if act_fp8 else 0
    bm = kp.base_model(model)
    return bm.layers[0], bm.rotary_emb, hooks


@torch.no_grad()
def run_layer(layer, rotary, h: torch.Tensor, prev_topk):
    pos = torch.arange(h.shape[1], device=h.device)[None]
    out, topk = layer(h, attention_mask=None, position_ids=pos, position_embeddings=rotary(h, position_ids=pos),
                      prev_topk_indices=prev_topk, use_cache=False)
    return out, topk


# ---------------------------------------------------------------- baselines
class _tf32:
    """TF32 only for OMP atom selection / dictionary fitting; errors are scored in FP32."""
    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.prev


def lexico_encode(X: torch.Tensor, D: torch.Tensor, s: int) -> tuple[torch.Tensor, float]:
    with _tf32():
        idx, coef = bc.omp(X, D, s, chunk=8192)
    return bc.reconstruct(D, idx, kp.fp8_unscaled(coef)), (24 * s + 16) / X.shape[1]


def qvg_encode(X: torch.Tensor, gen, C: int, bits: int = 2) -> tuple[torch.Tensor, float]:
    T, d = X.shape
    cent = kp.kmeans(X, C, 5, gen).to(torch.bfloat16).float()
    P = cent[kp.nearest(X, cent)]
    Rq = bc.quant_groups((X - P).reshape(-1, 64), bits).reshape(T, d)
    return P + Rq, bits + 16 / 64 + C * 16 / T + math.log2(C) / d


# ---------------------------------------------------------------- data
def assistant_text(convo) -> str:
    msg = convo[1]
    reasoning, content = str(msg.get("reasoning_content") or ""), str(msg.get("content") or "")
    if reasoning and content:
        if content in reasoning:
            return reasoning
        if reasoning in content:
            return content
        return reasoning + "\n" + content
    return reasoning or content


def gen_windows(db: str, tok, W: int, n: int, offset: int, dev) -> list[torch.Tensor]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    texts = []
    for (v,) in con.execute("select value from Cache"):
        d = json.loads(v)
        if isinstance(d, dict) and "convo" in d and len(d["convo"]) > 1:
            texts.append(assistant_text(d["convo"]))
    con.close()
    out = []
    for t in sorted(texts, key=lambda s: hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) >= offset + W:
            out.append(torch.tensor(ids[offset:offset + W], device=dev))
        if len(out) == n:
            return out
    sys.exit(f"FAIL: only {len(out)} generated completions of >= {offset + W} tokens in {db}")


# ---------------------------------------------------------------- per-layer evaluation
class Acc:
    def __init__(self):
        self.lat, self.attn, self.bits, self.pred, self.lag = {}, {}, {}, {}, {}
        self.E = self.Ec = 0.0

    def add(self, nm, bits, Xh, X, o, o0):
        self.bits[nm] = bits
        self.lat[nm] = self.lat.get(nm, 0.0) + (Xh - X).pow(2).sum().item()
        a = self.attn.setdefault(nm, [0.0, 0.0])
        a[0] += (o - o0).pow(2).sum().item()
        a[1] += o0.pow(2).sum().item()

    def result(self) -> dict:
        return {"centered_energy_fraction": self.Ec / self.E,
                "lag_corr": {str(k): v[0] / math.sqrt(v[1] * v[2]) for k, v in self.lag.items()},
                "pred_resid_vs_gmean": {k: v / self.Ec for k, v in self.pred.items()},
                "arms": {nm: {"bits": self.bits[nm], "latent_rel_mse": self.lat[nm] / self.E,
                              "attn_rel_mse": self.attn[nm][0] / self.attn[nm][1]} for nm in self.lat}}


def deep_arms(cbs) -> list[tuple]:
    small, big = f"cb{cbs[0]}", f"cb{cbs[1]}"
    arms = [("fp8_tok", None, None, None), ("fp8_unscaled", None, None, None)]
    arms += [("direct", None, "ch", b) for b in BITS] + [("gmean", None, "ch", 2)]
    arms += [(big, None, g, b) for g in ("ch", "tok") for b in BITS] + [(small, None, "ch", 2)]
    return arms


@torch.no_grad()
def eval_window(layer, tap, X, o0, ctx, acc: Acc, gen, cfg: dict):
    mu = ctx["mu"]
    acc.E += X.pow(2).sum().item()
    acc.Ec += (X - mu).pow(2).sum().item()
    kp.lag_stats(X, mu, acc.lag)
    for m in cfg["cbs"]:
        acc.pred[f"cb{m}"] = acc.pred.get(f"cb{m}", 0.0) + (X - kp.predict(X, f"cb{m}", None, ctx)[0]).pow(2).sum().item()
    rerun = lambda Xh: tap.rerun([layer], 0, Xh[None]).float()
    for arm in deep_arms(cfg["cbs"]):
        Xh, bits = kp.apply_arm(X, arm, ctx)
        acc.add(kp.arm_name(arm), bits, Xh, X, rerun(Xh), o0)
    Xh, bits = qvg_encode(X, gen, cfg["qvg_c"])
    acc.add(f"qvg{cfg['qvg_c']}-int2", bits, Xh, X, rerun(Xh), o0)
    if "lexico" in ctx:
        Xh, bits = lexico_encode(X, ctx["lexico"], cfg["lexico_s"])
        acc.add(f"lexico{cfg['lexico_s']}", bits, Xh, X, rerun(Xh), o0)


@torch.no_grad()
def stream(src: LayerSource, windows: dict, dev, out_dir: Path, *, layers: int | None = None, act_fp8=False,
           cbs=(256, 4096), kmeans_iters=10, lexico_every=8, lexico_atoms=4096, lexico_s=45,
           lexico_fit_tokens=8192, lexico_iters=5, qvg_c=64, state_every=8, seed=0) -> dict:
    """windows: {"train": [...], "eval": [...], "gen": [...]} equal-length token-id tensors."""
    gen = torch.Generator().manual_seed(seed)
    cfg = {"cbs": cbs, "qvg_c": qvg_c, "lexico_s": lexico_s}
    names = [(s, i) for s in ("train", "eval", "gen") for i in range(len(windows[s]))]
    (out_dir / "layers").mkdir(parents=True, exist_ok=True)
    state_p = out_dir / "stream_state.pt"
    if state_p.exists():
        st = torch.load(state_p, map_location=dev)
        H, topk, start = st["H"], st["topk"], st["next_layer"]
        log(f"resumed at layer {start}")
    else:
        emb = src.tensor(next(k for k in src.other if k.endswith("embed_tokens.weight"))).to(dev)
        H = torch.stack([F.embedding(windows[s][i], emb) for s, i in names])
        del emb
        topk, start = [None] * len(names), 0
    last = src.n_layers if layers is None else min(layers, src.n_layers)
    src.prefetch(start)
    results = {}
    for L in range(start, last):
        t0 = time.time()
        src.prefetch(L + 1)
        layer, rotary, hooks = load_layer(src, L, dev, act_fp8)
        t_load = time.time() - t0
        tap = kp.Tap([layer])
        try:
            def step(j):
                tap.record = True
                try:
                    out, tk = run_layer(layer, rotary, H[j:j + 1], topk[j])
                finally:
                    tap.record = False
                H[j], topk[j] = out[0], tk
                return tap.latent[0][0].float(), tap.out[0].float()

            # pass A: training windows -> global mean, codebooks, dictionary
            train = [step(j)[0] for j, (s, _) in enumerate(names) if s == "train"]
            Xtr = torch.cat(train)
            del train
            ctx = {"mu": Xtr.mean(0, keepdim=True).half().float()}
            for m in cbs:
                ctx[f"cb{m}"] = kp.kmeans(Xtr, m, kmeans_iters, gen)
            if lexico_every and (L % lexico_every == 0 or L == src.n_layers - 1):
                sub = Xtr[torch.randperm(Xtr.shape[0], generator=gen)[:lexico_fit_tokens].to(dev)]
                with _tf32():
                    ctx["lexico"] = bc.fit_dictionary(sub, lexico_atoms, lexico_s, lexico_iters, gen)
            del Xtr
            # pass B: held-out AA-LCR and generated windows
            accs, noop = {"eval": Acc(), "gen": Acc()}, 0.0
            for j, (s, _) in enumerate(names):
                if s == "train":
                    continue
                X, o0 = step(j)
                if noop == 0.0:
                    noop = (tap.rerun([layer], 0, X[None].to(torch.bfloat16)).float() - o0).pow(2).sum().item() / o0.pow(2).sum().item()
                    if noop > 1e-6:
                        raise RuntimeError(f"layer {L}: re-running attention with its own latent is not exact ({noop:.2e})")
                eval_window(layer, tap, X, o0, ctx, accs[s], gen, cfg)
        finally:
            tap.close()
        res = {"layer": L, "act_fp8_hooks": hooks, "lexico": "lexico" in ctx,
               "seconds": {"load": round(t_load, 1), "total": round(time.time() - t0, 1)},
               "staged_MBps": round(src.bytes / 1e6 / max(src.seconds, 1e-9), 1),
               **{s: a.result() for s, a in accs.items() if a.E}}
        (out_dir / "layers" / f"{L:03d}.json").write_text(json.dumps(res, indent=1))
        results[L] = res
        e = res.get("eval", {}).get("arms", {})
        log(f"layer {L}: {res['seconds']}  cb{cbs[-1]}-ch-int2 attn={e.get(f'cb{cbs[-1]}-ch-int2', {}).get('attn_rel_mse', float('nan')):.2e}"
            f" direct-ch-int4 attn={e.get('direct-ch-int4', {}).get('attn_rel_mse', float('nan')):.2e}  staged {res['staged_MBps']} MB/s")
        del layer, rotary, ctx
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        src.release(L + 1)
        if state_every and (L + 1) % state_every == 0 and L + 1 < last:
            torch.save({"H": H, "topk": topk, "next_layer": L + 1}, state_p)
    return results


# ---------------------------------------------------------------- gates
@torch.no_grad()
def preflight_stream(base_cfg, dev, tmp: Path) -> float:
    """A tiny model with the real layer types must stream to the latents of its full forward."""
    from transformers import GlmMoeDsaForCausalLM

    torch.manual_seed(0)
    full_dir = tmp / "tiny_full"
    GlmMoeDsaForCausalLM(kp.tiny_config(base_cfg)).to(torch.bfloat16).save_pretrained(full_dir)
    model = kp.load_model(full_dir, dev)
    ids = torch.randint(0, model.config.vocab_size, (256,), generator=torch.Generator().manual_seed(1)).to(dev)
    tap = kp.Tap(kp.base_model(model).layers)
    kp.forward_window(model, tap, ids)
    ref = {i: t.float() for i, t in tap.latent.items()}
    tap.close()
    src = LayerSource(str(full_dir), str(tmp / "tiny_stage"))
    h = F.embedding(ids, src.tensor(next(k for k in src.other if k.endswith("embed_tokens.weight"))).to(dev))[None]
    topk, worst = None, 0.0
    for L in range(src.n_layers):
        layer, rotary, _ = load_layer(src, L, dev)
        t = kp.Tap([layer])
        t.record = True
        h, topk = run_layer(layer, rotary, h, topk)
        t.close()
        worst = max(worst, ((t.latent[0].float() - ref[L]).pow(2).sum() / ref[L].pow(2).sum()).item())
    if worst > 1e-6:
        raise RuntimeError(f"preflight: streamed latents differ from the full forward (rel {worst:.2e})")
    return worst


def dequant_check(qsrc: LayerSource, bf16_snapshot: str, L: int, n_experts: int = 4) -> dict:
    """Dequantized weights of layer L vs the BF16 source: must be quantization-level close."""
    from safetensors import safe_open

    deq, fp8 = dequantize_sglang_w4afp8(qsrc.raw_layer(L))
    wmap = json.loads((Path(bf16_snapshot) / "model.safetensors.index.json").read_text())["weight_map"]
    rows = {"int4_expert": [], "fp8_block": []}
    picks = [k for k in deq if ".mlp.experts." in k and k.endswith(".weight")][: 3 * n_experts]
    picks += [s + ".weight" for s in sorted(fp8)][:6]
    for k in picks:
        with safe_open(str(Path(bf16_snapshot) / wmap[k]), framework="pt") as h:
            ref = h.get_tensor(k).float()
        rel = ((deq[k].float() - ref).pow(2).sum() / ref.pow(2).sum()).item()
        rows["int4_expert" if ".mlp.experts." in k else "fp8_block"].append((k, rel))
    worst = max(r for v in rows.values() for _, r in v)
    if worst > 0.2:
        raise RuntimeError(f"dequantized weights are far from BF16 (worst rel MSE {worst:.3f}): {rows}")
    return rows


# ---------------------------------------------------------------- report
KEY_ARMS = ("direct-ch-int4", "cb4096-ch-int2", "cb4096-ch-int3", "cb4096-ch-int4", "qvg64-int2", "lexico45", "fp8_tok")


def summarize(results: dict, meta: dict) -> str:
    L = ["# KV anchor probe, all layers — GLM-5.3 " + meta["weights"], "",
         f"{meta['eval_windows']} AA-LCR eval + {meta['gen_windows']} generated windows x {meta['window']} tokens; "
         f"codebooks fit on {meta['train_windows']} AA-LCR windows. Attention-output relative MSE.", ""]
    for split in ("eval", "gen"):
        bits = next((r[split]["arms"] for r in results.values() if split in r), {})
        L += [f"## {split}", "", "| layer | cb4096 resid | " + " | ".join(
            f"{a} ({bits[a]['bits']:.2f}b)" if a in bits else a for a in KEY_ARMS) + " |",
              "|---|---|" + "---|" * len(KEY_ARMS)]
        for l, r in sorted(results.items()):
            if split not in r:
                continue
            arms = r[split]["arms"]
            L.append(f"| {l} | {r[split]['pred_resid_vs_gmean'].get('cb4096', float('nan')):.3f} | " + " | ".join(
                f"{100 * arms[a]['attn_rel_mse']:.3f}%" if a in arms else "-" for a in KEY_ARMS) + " |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", choices=("bf16", "w4afp8"), required=True)
    ap.add_argument("--snapshot", required=True, help="BF16 source snapshot (tokenizer, dequant reference)")
    ap.add_argument("--quant-ckpt", help="SGLang w4afp8 checkpoint (--weights w4afp8)")
    ap.add_argument("--stage-dir", required=True, help="node-local scratch")
    ap.add_argument("--aa-lcr-root", required=True)
    ap.add_argument("--gen-db", required=True, help="AA cache.db with GLM-5.3 completions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--offset", type=int, default=1024)
    ap.add_argument("--gen-windows", type=int, default=8)
    ap.add_argument("--layers", type=int, default=None, help="stop after this many layers (smoke runs)")
    ap.add_argument("--act-fp8", action="store_true")
    ap.add_argument("--lexico-every", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import transformers
    from transformers import AutoConfig, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    meta = {"weights": args.weights + ("+act_fp8" if args.act_fp8 else ""), "window": args.window,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "source": args.quant_ckpt if args.weights == "w4afp8" else args.snapshot,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    log("env", meta)

    base_cfg = AutoConfig.from_pretrained(args.snapshot)
    log("gate 1: tiny-model streaming == full forward")
    meta["gate_stream_rel"] = preflight_stream(base_cfg, dev, Path(args.stage_dir) / "preflight")
    shutil.rmtree(Path(args.stage_dir) / "preflight", ignore_errors=True)
    log(f"gate 1 PASS ({meta['gate_stream_rel']:.1e})")

    if args.weights == "w4afp8":
        if not args.quant_ckpt:
            sys.exit("FAIL: --weights w4afp8 needs --quant-ckpt")
        qcfg = json.loads((Path(args.quant_ckpt) / "config.json").read_text()).get("quantization_config", {})
        if qcfg.get("quant_method") != "w4afp8":
            sys.exit(f"FAIL: {args.quant_ckpt} is not an SGLang w4afp8 checkpoint ({qcfg.get('quant_method')})")
        src = LayerSource(args.quant_ckpt, args.stage_dir, args.threads, dequant=True)
        first_moe = int(base_cfg.first_k_dense_replace)
        log(f"gate 2: dequantized layer {first_moe} vs BF16")
        meta["dequant_check"] = dequant_check(src, args.snapshot, first_moe)
        log("gate 2 PASS", json.dumps(meta["dequant_check"]))
    else:
        src = LayerSource(args.snapshot, args.stage_dir, args.threads)

    tok = AutoTokenizer.from_pretrained(args.snapshot)
    train, evals, names = kp.make_windows(tok, kp.aa_lcr_sets(Path(args.aa_lcr_root)), args.window, args.offset, dev)
    gens = gen_windows(args.gen_db, tok, args.window, args.gen_windows, args.offset, dev)
    meta.update(train_windows=len(train), eval_windows=len(evals), gen_windows=len(gens), windows=names)
    log(f"{len(train)} train / {len(evals)} eval AA-LCR + {len(gens)} generated windows")

    stream(src, {"train": train, "eval": evals, "gen": gens}, dev, out, layers=args.layers,
           act_fp8=args.act_fp8, lexico_every=args.lexico_every)
    # every finished layer, including ones from before a resume
    results = {int(p.stem): json.loads(p.read_text()) for p in sorted((out / "layers").glob("*.json"))}
    (out / "kv_anchor_deep_meta.json").write_text(json.dumps(meta, indent=1))
    summary = summarize(results, meta)
    (out / "kv_anchor_deep_summary.md").write_text(summary)
    print("\n" + summary, flush=True)
    log("done ->", out)


if __name__ == "__main__":
    main()
