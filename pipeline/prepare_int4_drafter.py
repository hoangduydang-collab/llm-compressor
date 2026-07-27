#!/usr/bin/env python3
"""Derive a vLLM-loadable INT4 EAGLE3 drafter from Sebesky/MiniMax-M3-EAGLE3-RTN-INT4.

WHY THIS SCRIPT EXISTS
----------------------
The published checkpoint quantizes `embed_tokens` to INT4 alongside the linears and
`lm_head`. On the vLLM build we serve with, that single choice is both (a) fatal and
(b) pointless:

(a) Fatal. `v1/spec_decode/llm_base_proposer.py::_maybe_share_embeddings` decides
    whether the draft can reuse the target's embedding table by running

        isinstance(self.model.model.embed_tokens.weight, torch.Tensor)

    with no `hasattr` guard on the draft side (contrast `_maybe_share_lm_head`,
    which *does* guard). A compressed-tensors quantized embedding registers
    `weight_packed` / `weight_scale` / `weight_shape` and NO `weight`
    (`compressed_tensors_embedding.py::CompressedTensorsEmbeddingWNA16Int
    .create_weights`), and `process_eagle_weight` sets `has_own_embed_tokens=True`
    for any loaded name containing "embed_tokens" -- including
    `embed_tokens.weight_packed`. So the guarded branch is entered and the
    unguarded attribute access raises AttributeError during drafter load.

(b) Pointless. When the draft's embedding is bit-identical to the target's, vLLM
    *deletes* it and points the draft at the target's table. That is what happens
    with the unquantized Inferact drafter (proven in our own phase D serve log:
    "Detected EAGLE model with embed_tokens identical to the target model"). A
    tensor that gets deleted at load cannot make drafting faster, so quantizing it
    buys zero latency while adding quantization error to every draft input.

This script therefore produces the artifact the experiment actually wants: INT4 on
everything the drafter *computes* with (attention + MLP + fc + lm_head, the 2.03 B
parameters that are read on every draft forward), and the ORIGINAL bf16
`embed_tokens` so the sharing path engages exactly as it does for the fp drafter.

Minimality is enforced, not asserted: every carried tensor is compared byte-for-byte
against the published checkpoint, and the embedding byte-for-byte against Inferact's.
The only edits are (1) swap the embedding, (2) drop the `group_embed` quantization
group so vLLM builds an unquantized embedding for it.

Usage:
    python pipeline/prepare_int4_drafter.py \
        --int4 /mnt/nfs/hoangduy/hf_assets/Sebesky/MiniMax-M3-EAGLE3-RTN-INT4 \
        --fp   /mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3 \
        --out  /mnt/nfs/hoangduy/hf_assets/derived/MiniMax-M3-EAGLE3-INT4-bf16embed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EMBED_PREFIX = "embed_tokens."
EMBED_FP_KEY = "embed_tokens.weight"


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--int4", required=True, type=Path, help="published RTN-INT4 dir")
    ap.add_argument("--fp", required=True, type=Path, help="Inferact bf16 drafter dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing {args.out} (use --force)")
    args.out.mkdir(parents=True, exist_ok=True)

    int4_st = args.int4 / "model.safetensors"
    fp_st = args.fp / "model.safetensors"
    for p in (int4_st, fp_st):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    manifest: dict = {
        "derived_from": {
            "int4": {
                "path": str(args.int4),
                "safetensors_sha256": sha256_file(int4_st),
            },
            "fp": {
                "path": str(args.fp),
                "safetensors_sha256": sha256_file(fp_st),
            },
        },
        "edits": [
            "replaced embed_tokens.{weight_packed,weight_scale,weight_shape} with the "
            "bf16 embed_tokens.weight from the fp drafter",
            "removed config_groups.group_embed so vLLM builds an unquantized embedding",
        ],
    }

    # --- config.json: drop only the embedding quantization group -----------------
    cfg = json.loads((args.int4 / "config.json").read_text())
    qc = cfg.get("quantization_config")
    if not qc:
        raise SystemExit("published checkpoint has no quantization_config")
    groups = qc.get("config_groups") or {}
    if "group_embed" not in groups:
        raise SystemExit(
            "expected config_groups.group_embed in the published checkpoint; "
            f"found {sorted(groups)} -- upstream layout changed, re-read the source"
        )
    embed_group = groups.pop("group_embed")
    manifest["removed_group_embed"] = embed_group
    # The linear + lm_head groups must survive untouched: they are the experiment.
    for required in ("group_0", "group_lmhead"):
        if required not in groups:
            raise SystemExit(f"missing {required} after edit -- refusing to write")
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    # --- weights ----------------------------------------------------------------
    carried: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    with safe_open(int4_st, framework="pt") as f4:
        for k in f4.keys():
            if k.startswith(EMBED_PREFIX):
                dropped.append(k)
                continue
            carried[k] = f4.get_tensor(k)
    if not dropped:
        raise SystemExit("no embed_tokens.* tensors found -- layout changed")

    with safe_open(fp_st, framework="pt") as ffp:
        fp_keys = set(ffp.keys())
        if EMBED_FP_KEY not in fp_keys:
            raise SystemExit(f"fp drafter lacks {EMBED_FP_KEY}")
        embed = ffp.get_tensor(EMBED_FP_KEY)
        embed_sha = sha256_tensor(embed)

    if embed.dtype != torch.bfloat16:
        raise SystemExit(f"fp embedding is {embed.dtype}, expected bfloat16")
    if embed.shape[0] != cfg["vocab_size"] or embed.shape[1] != cfg["hidden_size"]:
        raise SystemExit(f"fp embedding shape {tuple(embed.shape)} does not match config")

    # Every non-embedding tensor must be the published bytes, unchanged. Assert it
    # rather than trust it: this is what makes the artifact a minimal derivation.
    expected = {k: sha256_tensor(v) for k, v in carried.items()}

    out = dict(carried)
    out[EMBED_FP_KEY] = embed
    save_file(out, str(args.out / "model.safetensors"), metadata={"format": "pt"})

    with safe_open(args.out / "model.safetensors", framework="pt") as fo:
        out_keys = set(fo.keys())
        if out_keys != set(out):
            raise SystemExit("written key set does not match intended key set")
        for k, want in expected.items():
            if sha256_tensor(fo.get_tensor(k)) != want:
                raise SystemExit(f"byte drift on carried tensor {k}")
        if sha256_tensor(fo.get_tensor(EMBED_FP_KEY)) != embed_sha:
            raise SystemExit("byte drift on spliced embedding")

    # Single-file layout: regenerate the index so it cannot disagree with reality.
    total = sum(t.numel() * t.element_size() for t in out.values())
    index = {
        "metadata": {"total_size": total},
        "weight_map": {k: "model.safetensors" for k in sorted(out)},
    }
    (args.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    for extra in ("README.md",):
        src = args.int4 / extra
        if src.is_file():
            shutil.copy2(src, args.out / extra)

    manifest["dropped_tensors"] = sorted(dropped)
    manifest["spliced"] = {EMBED_FP_KEY: {"sha256": embed_sha, "shape": list(embed.shape)}}
    manifest["carried_tensor_count"] = len(carried)
    manifest["total_size_bytes"] = total
    manifest["out_safetensors_sha256"] = sha256_file(args.out / "model.safetensors")
    (args.out / "derivation-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"carried {len(carried)} published tensors byte-identical")
    print(f"dropped {dropped}")
    print(f"spliced {EMBED_FP_KEY} {tuple(embed.shape)} bf16 sha={embed_sha[:12]}")
    print(f"wrote {args.out} ({total / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
