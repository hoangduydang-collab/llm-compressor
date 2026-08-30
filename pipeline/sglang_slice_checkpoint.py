"""Build a truncated, loadable view of a large checkpoint for engine testing.

WHY. The decisive question about a converted checkpoint is whether the target
engine's REAL loader accepts it -- right tensor names, shapes, dtypes, and scale
semantics. Answering that by serving the full model needs 8 H100s on one node,
which on this cluster means waiting for the single node that has them free. But
the failure modes we are hunting are all per-layer: a wrong ``w13_weight`` shape,
a missing ``weight_scale_inv``, an ``input_scale`` under a name the loader does
not look for. None of them need 78 layers to appear.

So: write a directory that looks like a complete N-layer model of the same
architecture. ``config.json`` says ``num_hidden_layers = N``, the index carries
only those layers plus the embeddings, final norm and head, and the shards
themselves are SYMLINKS to the originals -- no copying, no extra bytes. The
engine then exercises its entire genuine loading path (config parsing, quant
method selection, create_weights, weight_loader,
process_weights_after_loading) on real converted tensors, on one GPU.

The point of doing it this way rather than driving the engine's quantization
classes directly is that hand-driving them means re-encoding my own beliefs about
their conventions in glue code -- which is precisely the mistake that sent a
394 GB job down a wrong path earlier today. A truncated model has no glue: the
engine reads the files exactly as it would in production.

WHAT IT CANNOT TELL YOU. Truncated weights are not a truncated *model*: layer 3
of a 4-layer model receives activations that never passed through layers 4-77, so
generated text is meaningless and perplexity is not comparable to anything. This
answers "does it load and run a forward pass", not "is it any good". Quality
still needs the full model.

Keep N >= first_k_dense_replace + 1 so at least one MoE layer is included --
otherwise the expert loader, the whole reason for the exercise, is never touched.

Usage:
    python -m pipeline.sglang_slice_checkpoint \
        --ckpt <full converted checkpoint> --out <slice dir> [--layers 4]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# Kept regardless of layer index: without these the model has no input
# embedding, no output head and no final norm, and the engine fails on a missing
# key rather than on anything we are trying to learn.
_ALWAYS_KEEP = (
    "model.embed_tokens.",
    "model.norm.",
    "lm_head.",
)


def layer_of(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def keep(name: str, layers: int) -> bool:
    index = layer_of(name)
    if index is not None:
        return index < layers
    return any(name.startswith(prefix) for prefix in _ALWAYS_KEEP)


def truncate_config(config: dict, layers: int) -> dict:
    """An N-layer config of the same architecture.

    Per-layer LISTS have to be truncated too, not just the layer count. GLM
    carries a 78-entry ``indexer_types``, and the engine indexes it by layer
    number to decide whether to build a DSA indexer -- leaving it at full length
    is harmless, but leaving it SHORTER than the layer count would crash on a
    lookup, so anything list-shaped whose length matches the original depth gets
    cut to N.
    """
    out = dict(config)
    original = int(config.get("num_hidden_layers", 0))
    out["num_hidden_layers"] = layers
    # No MTP in a slice: the draft layer sits at index num_hidden_layers and
    # would now point at a layer the index does not contain.
    if "num_nextn_predict_layers" in out:
        out["num_nextn_predict_layers"] = 0

    for key, value in config.items():
        if isinstance(value, list) and original and len(value) == original:
            out[key] = value[:layers]
    return out


def slice_checkpoint(ckpt: Path, out: Path, layers: int = 4) -> int:
    index_path = ckpt / "model.safetensors.index.json"
    config_path = ckpt / "config.json"
    for path in (index_path, config_path):
        if not path.is_file():
            print(f"error: {path} not found", flush=True)
            return 2

    config = json.loads(config_path.read_text(encoding="utf-8"))
    depth = int(config.get("num_hidden_layers", 0))
    dense = int(config.get("first_k_dense_replace", 0))
    if layers <= dense:
        print(f"error: --layers {layers} is not greater than "
              f"first_k_dense_replace {dense}, so the slice would contain no MoE "
              f"layer and would not exercise the expert loader at all", flush=True)
        return 2
    if layers > depth:
        print(f"error: --layers {layers} exceeds the model's {depth}", flush=True)
        return 2

    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    kept = {k: v for k, v in weight_map.items() if keep(k, layers)}
    if not kept:
        print("error: slice selected no tensors", flush=True)
        return 2

    missing = sorted(
        prefix for prefix in _ALWAYS_KEEP
        if not any(k.startswith(prefix) for k in kept)
    )
    if missing:
        # Loudly, because the engine's error for a missing embedding is obscure
        # and would send the reader hunting the quantization format instead.
        print(f"[slice] WARNING: no tensors matched {missing}; the engine will "
              f"fail on a missing key unrelated to quantization", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    shards = sorted({v for v in kept.values()})
    total = 0
    how = set()
    for shard in shards:
        link = out / shard
        if link.exists() or link.is_symlink():
            link.unlink()
        how.add(_link(ckpt / shard, link))
        total += (ckpt / shard).stat().st_size
    print(f"[slice] shard linking: {'/'.join(sorted(how))}", flush=True)

    # Prove the links are usable before handing the directory to an engine.
    # A dangling symlink is a directory entry that `ls` shows and `is_file()`
    # rejects, so this is cheap -- and it is exactly the failure the HF cache's
    # relative symlinks produced, surfacing at load time as "incomplete
    # download?" rather than as anything about linking. Opening the header too,
    # because a link can resolve and still point at a truncated blob.
    from safetensors import safe_open

    unusable = []
    for shard in shards:
        path = out / shard
        if not path.is_file():
            unusable.append(f"{shard}: not a readable file (dangling link?)")
            continue
        try:
            with safe_open(str(path), framework="pt") as handle:
                next(iter(handle.keys()), None)
        except Exception as err:  # noqa: BLE001
            unusable.append(f"{shard}: {type(err).__name__}: {err}")
    if unusable:
        for line in unusable:
            print(f"[slice] FAIL {line}", flush=True)
        return 1
    _ok = len(shards)
    print(f"[slice] verified {_ok} linked shard(s) are readable", flush=True)

    # total_size must describe the KEPT tensors, not the symlinked files, which
    # still hold the layers we excluded.
    from safetensors import safe_open

    kept_bytes = 0
    for shard in shards:
        with safe_open(str(ckpt / shard), framework="pt") as handle:
            for key in handle.keys():
                if key in kept:
                    slice_ = handle.get_slice(key)
                    numel = 1
                    for dim in slice_.get_shape():
                        numel *= dim
                    kept_bytes += numel * _dtype_bytes(slice_.get_dtype())

    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": kept_bytes}, "weight_map": kept}, indent=2
        ),
        encoding="utf-8",
    )
    (out / "config.json").write_text(
        json.dumps(truncate_config(config, layers), indent=2), encoding="utf-8"
    )
    for extra in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ):
        source = ckpt / extra
        if source.is_file():
            (out / extra).write_bytes(source.read_bytes())

    moe = layers - dense
    print(f"[slice] layers 0..{layers - 1} ({dense} dense + {moe} MoE)", flush=True)
    print(f"[slice] tensors {len(kept)} of {len(weight_map)}", flush=True)
    print(f"[slice] kept bytes {kept_bytes / 1e9:.2f} GB "
          f"({len(shards)} linked shard(s) holding {total / 1e9:.1f} GB, the "
          f"excess being layers this slice excludes)", flush=True)
    print(f"[slice] -> {out}", flush=True)
    return 0


def _link(source: Path, dest: Path) -> str:
    """Point ``dest`` at ``source`` without copying bytes.

    Hard link first: it needs no privilege (Windows refuses ``os.symlink`` to an
    unelevated process, WinError 1314) and is zero-copy on any single
    filesystem, which cephfs is here. Symlink is the fallback for a
    cross-filesystem layout.

    Never falls back to copying. A silent copy would duplicate the shards, and
    on the real checkpoint that is hundreds of gigabytes appearing on a shared
    volume because a link failed -- a failure worth surfacing, not absorbing.
    """
    # RESOLVE FIRST. A HuggingFace cache snapshot is a symlink farm: every
    # snapshots/<rev>/*.safetensors entry is a symlink to ../../blobs/<sha>.
    # os.link on Linux does NOT follow symlinks -- it hardlinks the symlink
    # inode, target string and all -- so linking one into a directory at a
    # different depth reproduces a RELATIVE target that now resolves to
    # nowhere. The result is a slice full of dangling symlinks that pass `ls`
    # and fail at load time as "incomplete download?", which is what happened
    # on 2026-08-30. Resolving turns the source into the blob's real path.
    real = source.resolve(strict=True)
    try:
        os.link(real, dest)
        return "hardlink"
    except OSError as hard_err:
        try:
            os.symlink(real, dest)
            return "symlink"
        except OSError as soft_err:
            raise OSError(
                f"cannot link {dest} -> {real}: hard link failed "
                f"({hard_err}); symlink failed ({soft_err}). Refusing to copy, "
                f"which would duplicate the shards on a shared volume."
            ) from soft_err


def _dtype_bytes(dtype: str) -> int:
    return {
        "F64": 8, "I64": 8, "U64": 8,
        "F32": 4, "I32": 4, "U32": 4,
        "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
        "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
    }.get(dtype, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=4,
                        help="must exceed first_k_dense_replace so at least one "
                             "MoE layer is included")
    args = parser.parse_args(argv)
    return slice_checkpoint(args.ckpt, args.out, args.layers)


if __name__ == "__main__":
    sys.exit(main())
