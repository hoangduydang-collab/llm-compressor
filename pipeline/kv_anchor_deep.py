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
_ST_DTYPE = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F64": torch.float64,
             "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
             "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn}


def read_span(path: Path, lo: int, hi: int, threads: int = 16, chunk: int = 64 << 20) -> bytearray:
    """Bytes [lo, hi) of a file with parallel range reads (cephfs is ~30 MB/s per stream)."""
    buf = bytearray(hi - lo)
    view = memoryview(buf)

    def work(off):
        n = min(chunk, hi - off)
        with open(path, "rb", buffering=0) as f:
            f.seek(off)
            got = 0
            while got < n:
                r = f.readinto(view[off - lo + got: off - lo + n])
                if not r:
                    raise IOError(f"short read at {off + got} in {path}")
                got += r

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(work, range(lo, hi, chunk)))
    return buf


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
    """Reads one decoder layer's tensors by byte range (never whole shards: the
    served checkpoint packs ~10 layers into each 46.6 GiB shard) and writes the
    layer as a one-layer checkpoint loadable by AutoModelForCausalLM."""

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
        self._headers: dict[str, tuple[int, dict]] = {}
        self._pending: dict[int, object] = {}
        self._pool = ThreadPoolExecutor(1)
        self._lock = threading.Lock()
        self.bytes, self.seconds = 0, 0.0

    def _header(self, shard: str) -> tuple[int, dict]:
        with self._lock:
            if shard not in self._headers:
                with open(self.snapshot / shard, "rb") as f:
                    n = int.from_bytes(f.read(8), "little")
                    self._headers[shard] = (8 + n, json.loads(f.read(n)))
            return self._headers[shard]

    def read(self, keys: list[str]) -> dict[str, torch.Tensor]:
        """Tensors by name, reading only the byte span that covers them in each shard."""
        by_shard: dict[str, list[str]] = {}
        for k in keys:
            by_shard.setdefault(self.wmap[k], []).append(k)
        out = {}
        for shard, ks in sorted(by_shard.items()):
            base, hdr = self._header(shard)
            lo = min(hdr[k]["data_offsets"][0] for k in ks)
            hi = max(hdr[k]["data_offsets"][1] for k in ks)
            t0 = time.time()
            buf = read_span(self.snapshot / shard, base + lo, base + hi, self.threads)
            with self._lock:
                self.bytes += hi - lo
                self.seconds += time.time() - t0
            for k in ks:
                a, b = hdr[k]["data_offsets"]
                dt = _ST_DTYPE[hdr[k]["dtype"]]
                t = torch.frombuffer(buf, dtype=torch.uint8, count=b - a, offset=a - lo) if b > a else torch.empty(0, dtype=torch.uint8)
                out[k] = t.view(dt).reshape(hdr[k]["shape"]).clone()
            del buf
        return out

    def prefetch(self, L: int):
        if L < self.n_layers and L not in self._pending:
            self._pending[L] = self._pool.submit(self.read, self.by_layer[L])

    def tensor(self, key: str) -> torch.Tensor:
        return self.read([key])[key]

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
        fut = self._pending.pop(L, None)
        return fut.result() if fut is not None else self.read(self.by_layer[L])

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


# ---------------------------------------------------------------- SGLang's served KV formats (v0.5.17)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """round to the nearest FP4 E2M1 value (|x| <= 6 assumed; ties go up)."""
    vals = x.new_tensor(E2M1)
    bounds = x.new_tensor([(a + b) / 2 for a, b in zip(E2M1, E2M1[1:])])
    return torch.sign(x) * vals[torch.bucketize(x.abs().clamp(max=6.0), bounds, right=True)]


def fp8_tile128(X: torch.Tensor) -> tuple[torch.Tensor, float]:
    """DSA FP8 K cache (kernels/ops/attention/dsa/quant_k_cache.py): per token, one fp32
    amax/448 scale per 128-value tile, e4m3 values."""
    T, d = X.shape
    t = X.reshape(T, d // 128, 128)
    # the reference computes the scale on the bf16 latent, so it is bf16-rounded
    s = (t.to(torch.bfloat16).abs().amax(-1, keepdim=True) / 448.0).float().clamp(min=1e-30)
    q = (t / s).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * s).reshape(T, d), 8 + 32 / 128


def fp4_mx_block16(X: torch.Tensor) -> tuple[torch.Tensor, float]:
    """--kv-cache-dtype fp4_mx_block16 (kvfp4_tensor.FP4MXBlock16KVQuantizeUtil): E2M1 values,
    one power-of-two scale ceil(log2(amax/6)) per 16 values."""
    T, d = X.shape
    b = X.reshape(T, d // 16, 16)
    e = torch.ceil(torch.log2(torch.clamp(b.abs().amax(-1, keepdim=True) / 6.0, min=1e-10)))
    return (_round_e2m1(b / torch.exp2(e)) * torch.exp2(e)).reshape(T, d), 4 + 8 / 16


def nvfp4(X: torch.Tensor) -> tuple[torch.Tensor, float]:
    """--kv-cache-dtype nvfp4: E2M1 values, e4m3 scale per 16 values, one fp32 global scale
    (here from the window itself -- a served static scale can only be worse)."""
    T, d = X.shape
    b = X.reshape(T, d // 16, 16)
    g = (X.abs().amax() / (448.0 * 6.0)).clamp(min=1e-30)
    sb = ((b.abs().amax(-1, keepdim=True) / 6.0) / g).clamp(max=448).to(torch.float8_e4m3fn).float() * g
    sb = sb.clamp(min=1e-30)
    return (_round_e2m1((b / sb).clamp(-6, 6)) * sb).reshape(T, d), 4 + 8 / 16 + 32 / (T * d)


def hadamard(d: int, device=None) -> torch.Tensor:
    """orthonormal Sylvester Hadamard [d, d]; in absorbed MLA it folds exactly into W_UK / W_UV."""
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    if H.shape[0] != d:
        raise ValueError(f"no Sylvester Hadamard of size {d}")
    return H / math.sqrt(d)


def rotated_arms(X: torch.Tensor, ctx: dict, big: str) -> list[tuple[str, torch.Tensor, float]]:
    """established 4-bit baselines with a Hadamard rotation (QuaRot / SAW-INT4), and our
    anchor combined with NVFP4 for the residual."""
    H = ctx["H"]
    int4 = lambda V: kp.quant_residual(V, "tok", 4)
    b_int4 = 4 + 16 / kp.G
    P, cost = kp.predict(X, big + "z", None, ctx)
    R = X - P
    Xn, b_nv = nvfp4(X @ H)
    return [("H-nvfp4", Xn @ H.T, b_nv),
            ("H-direct-tok-int4", int4(X @ H) @ H.T, b_int4),
            (f"{big}z-nvfp4", P + nvfp4(R)[0], b_nv + cost),
            (f"H-{big}z-nvfp4", P + nvfp4(R @ H)[0] @ H.T, b_nv + cost),
            (f"H-{big}z-tok-int4", P + int4(R @ H) @ H.T, b_int4 + cost)]


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
        self.lat, self.attn, self.bits, self.pred, self.lag, self.tok0 = {}, {}, {}, {}, {}, {}
        self.E = self.Ec = self.E0 = 0.0
        self.sink_mass, self.norm_ratio = [], []
        self.small_frac, self.small_mass, self.zero_pick, self.pre_ratio, self.pre_eps = [], [], [], [], []
        self.tiny_frac = []

    def add(self, nm, bits, Xh, X, o, o0):
        self.bits[nm] = bits
        self.lat[nm] = self.lat.get(nm, 0.0) + (Xh - X).pow(2).sum().item()
        self.tok0[nm] = self.tok0.get(nm, 0.0) + (Xh[0] - X[0]).pow(2).sum().item()
        a = self.attn.setdefault(nm, [0.0, 0.0])
        a[0] += (o - o0).pow(2).sum().item()
        a[1] += o0.pow(2).sum().item()

    def result(self) -> dict:
        mean = lambda v: sum(v) / len(v) if v else None
        return {"centered_energy_fraction": self.Ec / self.E,
                "attn_mass_on_token0": mean(self.sink_mass),
                "token0_norm_over_median": mean(self.norm_ratio),
                # tokens 1.. with latent norm < 0.1 x median: share of tokens, attention mass they receive
                "small_token_fraction": mean(self.small_frac),
                "attn_mass_on_small_tokens": mean(self.small_mass),
                "zero_anchor_fraction": mean(self.zero_pick),
                "tiny_exact_fraction": mean(self.tiny_frac),   # tokens under the training-median threshold (|tiny arms)
                # kv_a_layernorm input RMS of token 0: over the median token's, and over sqrt(eps)
                "token0_prenorm_rms_over_median": mean(self.pre_ratio),
                "token0_prenorm_rms_over_sqrt_eps": mean(self.pre_eps),
                "lag_corr": {str(k): v[0] / math.sqrt(v[1] * v[2]) for k, v in self.lag.items()},
                "pred_resid_vs_gmean": {k: v / self.Ec for k, v in self.pred.items()},
                "arms": {nm: {"bits": self.bits[nm], "latent_rel_mse": self.lat[nm] / self.E,
                              "token0_latent_rel_mse": self.tok0[nm] / self.E0,
                              "attn_rel_mse": self.attn[nm][0] / self.attn[nm][1]} for nm in self.lat}}


def deep_arms(cbs) -> list[tuple]:
    small, big = f"cb{cbs[0]}", f"cb{cbs[1]}"
    arms = [("fp8_tok", None, None, None), ("fp8_unscaled", None, None, None)]
    arms += [("direct", None, g, b) for g in ("ch", "tok") for b in BITS] + [("gmean", None, "ch", 2)]
    arms += [(s, None, g, b) for s in (big, big + "z") for g in ("ch", "tok") for b in BITS] + [(small, None, "ch", 2)]
    # attention-weighted codebooks (w1: mean attention weight, w2: mean squared weight) and dedicated sink centroids
    arms += [(s, None, "tok", b) for s in (big + "w1", big + "w2", big + "zs") for b in BITS]
    arms += [(big + "w2", None, "ch", 2)]
    return arms


def tiny_arms(cbs) -> set[str]:
    """arms also scored with tiny tokens (norm < 0.1 x training median) kept exact"""
    big = f"cb{cbs[1]}"
    return {"direct-tok-int2", "direct-tok-int4", *(f"{big}z-tok-int{b}" for b in BITS)}


def diag_arms(cbs) -> set[str]:
    big = f"cb{cbs[1]}"
    return {"direct-ch-int2", "gmean-ch-int2", f"{big}-ch-int2", f"{big}-tok-int2", f"{big}z-ch-int2", f"{big}z-tok-int2"}


@torch.no_grad()
def eval_window(layer, tap, X, o0, ctx, acc: Acc, gen, cfg: dict):
    """sink_keep: the first k tokens of every window stay exact in every arm
    (the attention-sink exemption practical KV quantizers make). diag: also
    score a few arms with k = 1 and 4 when the run itself keeps none."""
    mu, k = ctx["mu"], cfg.get("sink_keep", 0)
    acc.E += X.pow(2).sum().item()
    acc.Ec += (X - mu).pow(2).sum().item()
    acc.E0 += X[0].pow(2).sum().item()
    n = X.norm(dim=1)
    acc.norm_ratio.append((n[0] / n.median()).item())
    if tap.sink_mass.get(0) is not None:
        acc.sink_mass.append(tap.sink_mass[0])
    small = n[1:] < 0.1 * n.median()
    acc.small_frac.append(small.float().mean().item())
    if tap.key_mass.get(0) is not None:
        acc.small_mass.append(tap.key_mass[0][1:][small].sum().item())
    C = ctx[f"cb{cfg['cbs'][-1]}"]
    acc.zero_pick.append((kp.nearest(X, torch.cat([C, torch.zeros_like(C[:1])])) == C.shape[0]).float().mean().item())
    pre = tap.prenorm[0][0].float().pow(2).mean(-1).sqrt()
    norm = layer.self_attn.kv_a_layernorm
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", None))
    acc.pre_ratio.append((pre[0] / pre.median()).item())
    if eps:
        acc.pre_eps.append(pre[0].item() / math.sqrt(eps))
    kp.lag_stats(X, mu, acc.lag)
    for m in cfg["cbs"]:
        acc.pred[f"cb{m}"] = acc.pred.get(f"cb{m}", 0.0) + (X - kp.predict(X, f"cb{m}", None, ctx)[0]).pow(2).sum().item()
    rerun = lambda Xh: tap.rerun([layer], 0, Xh[None]).float()

    def keep(Xh, n):
        if not n:
            return Xh
        Xh = Xh.clone()
        Xh[:n] = X[:n]
        return Xh

    diag = diag_arms(cfg["cbs"]) if cfg.get("diag") and not k else set()
    tiny = n < ctx["tiny_thr"]
    acc.tiny_frac.append(tiny.float().mean().item())
    tiny_cost = tiny.float().mean().item() * (1 + math.log2(X.shape[0]) / X.shape[1])   # x (16 - b) bits, + position
    coded = [(kp.arm_name(a), *kp.apply_arm(X, a, ctx)) for a in deep_arms(cfg["cbs"])]
    Xq, bq = qvg_encode(X, gen, cfg["qvg_c"])
    coded.append((f"qvg{cfg['qvg_c']}-int2", Xq, bq))
    coded += [(nm, *f(X)) for nm, f in (("sglang_fp8_tile128", fp8_tile128), ("sglang_fp4_mx16", fp4_mx_block16),
                                         ("sglang_nvfp4", nvfp4))]
    coded += rotated_arms(X, ctx, f"cb{cfg['cbs'][-1]}")
    if "lexico" in ctx:
        Xl, bl = lexico_encode(X, ctx["lexico"], cfg["lexico_s"])
        coded.append((f"lexico{cfg['lexico_s']}", Xl, bl))
    for nm, Xh, bits in coded:
        Xh = keep(Xh, k)
        acc.add(nm, bits, Xh, X, rerun(Xh), o0)
        for n in (1, 4) if nm in diag else ():
            Xk = keep(Xh, n)
            acc.add(f"{nm}|keep{n}", bits, Xk, X, rerun(Xk), o0)
        if nm in tiny_arms(cfg["cbs"]):
            Xt = torch.where(tiny[:, None], X, Xh)
            acc.add(f"{nm}|tiny", bits + tiny_cost * (16 - bits), Xt, X, rerun(Xt), o0)


@torch.no_grad()
def stream(src: LayerSource, windows: dict, dev, out_dir: Path, *, layers: int | None = None, act_fp8=False,
           cbs=(256, 4096), kmeans_iters=10, lexico_every=8, lexico_atoms=4096, lexico_s=45,
           lexico_fit_tokens=8192, lexico_iters=5, qvg_c=64, state_every=8, seed=0, sink_keep=0,
           diag=False, dump_layers=()) -> dict:
    """windows: {"train": [...], "eval": [...], "gen": [...]} equal-length token-id tensors.
    dump_layers: save the eval latents, per-key attention, codebook and mean of these layers to dump/NNN.pt."""
    gen = torch.Generator().manual_seed(seed)
    cfg = {"cbs": cbs, "qvg_c": qvg_c, "lexico_s": lexico_s, "sink_keep": sink_keep, "diag": diag}
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
            # exempt sink tokens are never coded, so they are not fit either
            train, w1, w2 = [], [], []
            for j, (s, _) in enumerate(names):
                if s == "train":
                    train.append(step(j)[0][sink_keep:])
                    km, km2 = tap.key_mass.get(0), tap.key_mass2.get(0)
                    w1.append(km[sink_keep:] if km is not None else torch.ones_like(train[-1][:, 0]))
                    w2.append(km2[sink_keep:] if km2 is not None else torch.ones_like(train[-1][:, 0]))
            Xtr, w1, w2 = torch.cat(train), torch.cat(w1), torch.cat(w2)
            del train
            ctx = {"mu": Xtr.mean(0, keepdim=True).half().float(), "H": hadamard(Xtr.shape[1], Xtr.device)}
            for m in cbs:
                ctx[f"cb{m}"] = kp.kmeans(Xtr, m, kmeans_iters, gen)
            # attention-weighted codebooks; tiny-token threshold and dedicated sink centroids
            ctx[f"cb{cbs[-1]}w1"] = kp.kmeans(Xtr, cbs[-1], kmeans_iters, gen, w=w1)
            ctx[f"cb{cbs[-1]}w2"] = kp.kmeans(Xtr, cbs[-1], kmeans_iters, gen, w=w2)
            ntr = Xtr.norm(dim=1)
            ctx["tiny_thr"] = 0.1 * ntr.median()
            Xs = Xtr[ntr < ctx["tiny_thr"]]
            ctx["sink"] = kp.kmeans(Xs, min(4, Xs.shape[0]), 20, gen) if Xs.shape[0] else Xtr[:0]
            del w1, w2, Xs
            if lexico_every and (L % lexico_every == 0 or L == src.n_layers - 1):
                sub = Xtr[torch.randperm(Xtr.shape[0], generator=gen)[:lexico_fit_tokens].to(dev)]
                with _tf32():
                    ctx["lexico"] = bc.fit_dictionary(sub, lexico_atoms, lexico_s, lexico_iters, gen)
            del Xtr
            # pass B: held-out AA-LCR and generated windows
            accs, noop, dump = {"eval": Acc(), "gen": Acc()}, 0.0, {"X": [], "key_mass": []}
            for j, (s, _) in enumerate(names):
                if s == "train":
                    continue
                X, o0 = step(j)
                if L in dump_layers and s == "eval":
                    dump["X"].append(X.half().cpu())
                    km = tap.key_mass.get(0)
                    dump["key_mass"].append(None if km is None else km.cpu())
                if noop == 0.0:
                    noop = (tap.rerun([layer], 0, X[None].to(torch.bfloat16)).float() - o0).pow(2).sum().item() / o0.pow(2).sum().item()
                    if noop > 1e-6:
                        raise RuntimeError(f"layer {L}: re-running attention with its own latent is not exact ({noop:.2e})")
                eval_window(layer, tap, X, o0, ctx, accs[s], gen, cfg)
        finally:
            tap.close()
        if L in dump_layers:
            (out_dir / "dump").mkdir(exist_ok=True)
            torch.save({**dump, "mu": ctx["mu"].cpu(), **{f"cb{m}": ctx[f"cb{m}"].cpu() for m in cbs}},
                       out_dir / "dump" / f"{L:03d}.pt")
        g = layer.self_attn.kv_a_layernorm.weight.float().abs()
        res = {"layer": L, "act_fp8_hooks": hooks, "lexico": "lexico" in ctx,
               "kv_norm_gain_abs": {"min": g.min().item(), "median": g.median().item(), "max": g.max().item()},
               "seconds": {"load": round(t_load, 1), "total": round(time.time() - t0, 1)},
               "staged_MBps": round(src.bytes / 1e6 / max(src.seconds, 1e-9), 1),
               **{s: a.result() for s, a in accs.items() if a.E}}
        (out_dir / "layers" / f"{L:03d}.json").write_text(json.dumps(res, indent=1))
        results[L] = res
        e = res.get("eval", {}).get("arms", {})
        at = lambda nm: e.get(nm, {}).get('attn_rel_mse', float('nan'))
        big = f"cb{cbs[-1]}"
        four = {"fp8_tile": "sglang_fp8_tile128", "fp8_tok": "fp8_tok", "nvfp4": "sglang_nvfp4", "H-nvfp4": "H-nvfp4",
                "tok4": "direct-tok-int4", "z4": f"{big}z-tok-int4", "w2-4": f"{big}w2-tok-int4",
                "z4|tiny": f"{big}z-tok-int4|tiny", "H-z-nvfp4": f"H-{big}z-nvfp4"}
        log(f"layer {L}: {res['seconds']}  attn ~4b+: " + " ".join(f"{k} {at(v):.2e}" for k, v in four.items())
            + f"  staged {res['staged_MBps']} MB/s")
        del layer, rotary, ctx
        if dev.type == "cuda":
            torch.cuda.empty_cache()
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
KEY_ARMS = ("sglang_fp8_tile128", "sglang_nvfp4", "sglang_fp4_mx16", "H-nvfp4", "H-direct-tok-int4",
            "cb4096z-tok-int4", "cb4096w2-tok-int4", "H-cb4096z-nvfp4")


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
    ap.add_argument("--sink-keep", type=int, default=0, help="first k tokens of each window stay exact in every arm")
    ap.add_argument("--diag", action="store_true", help="also score key arms with k=1,4 exact sink tokens")
    ap.add_argument("--dump-layers", default="", help="comma list: save eval latents + codebooks of these layers")
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

    meta.update(sink_keep=args.sink_keep, diag=args.diag)
    stream(src, {"train": train, "eval": evals, "gen": gens}, dev, out, layers=args.layers,
           act_fp8=args.act_fp8, lexico_every=args.lexico_every, sink_keep=args.sink_keep, diag=args.diag,
           dump_layers={int(x) for x in args.dump_layers.split(",") if x})
    # every finished layer, including ones from before a resume
    results = {int(p.stem): json.loads(p.read_text()) for p in sorted((out / "layers").glob("*.json"))}
    (out / "kv_anchor_deep_meta.json").write_text(json.dumps(meta, indent=1))
    summary = summarize(results, meta)
    (out / "kv_anchor_deep_summary.md").write_text(summary)
    print("\n" + summary, flush=True)
    log("done ->", out)


if __name__ == "__main__":
    main()
