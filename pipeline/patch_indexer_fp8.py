"""Retrofit the DSA indexer to block FP8 in an already-converted w4afp8 checkpoint.

WHY THIS EXISTS RATHER THAN A RE-CONVERSION. ``to_sglang_w4afp8`` now handles the
indexer (see ``_ENGINE_FP8_SUFFIXES``), but re-running it on GLM-5.3 costs 2h26m
because it redoes all 57,600 int4 expert repacks -- work the existing artifact
already contains, correctly and bit-exactly. Only 42 tensors are wrong, spread
over 19 of 40 shards, so rewriting those 19 and hardlinking the other 21 is
~380 GB of pure I/O and no repacking: measured 2h26m versus roughly 32 min.

WHY THE SHARDS MUST BE REWRITTEN, rather than the new tensors appended. Tempting
and wrong. SGLang's loader (``model_loader/loader.py:510``) GLOBS the directory
for ``*.safetensors`` and then filters "files not found in the index" --
``filter_duplicate_safetensors_files`` builds a set of FILE paths, not tensor
names. ``safetensors_weights_iterator`` then yields EVERY tensor in each
surviving file. A superseded BF16 ``indexer.wk.weight`` left inside an
still-referenced shard would therefore be yielded alongside its e4m3 replacement,
and both handed to the same parameter: dtype error at best, order-dependent
corruption at worst. Dropping it from the index is not enough; the bytes have to
go.

WHAT IS NOT FIXED HERE. The expert ``input_scale`` tensors keep the names the old
converter wrote (``experts.{i}.{gate,up,down}_proj.input_scale`` instead of
``experts.{i}.{w1,w3,w2}.input_scale``). Those 58,368 tensors live in every
expert shard, so renaming them would mean rewriting the whole artifact. It costs
nothing to leave them:

  * the loader skips names it does not recognise -- ``layer.py:285`` does
    ``if ("mlp.experts." in name) and name not in params_dict: continue`` -- so
    they are inert, not fatal;
  * ``w13_input_scale`` / ``w2_input_scale`` are registered as ``torch.ones``, so
    a scale that never loads already holds the value we would have written; and
  * measured 2026-08-30, every ``input_scale`` in BOTH existing w4afp8 releases
    (PhalaCloud GLM-5.2, 35,951 downloads, and GLM-5.3) is exactly 1.0, while the
    official ``zai-org/GLM-5.3`` FP8 release ships none at all and declares
    ``activation_scheme: "dynamic"``. There is no measured value anywhere in the
    ecosystem to prefer over ones.

THE INDEXER SPLIT IS NOT GUESSWORK. ``zai-org/GLM-5.3`` -- the official FP8
release -- quantizes ``indexer.wk`` and ``indexer.wq_b`` and leaves
``indexer.weights_proj`` and ``indexer.k_norm`` in BF16, at block size [128,128].
That matches what SGLang's ``dsa_indexer.py`` requires (wk/wq_b built with
``quant_config``, weights_proj without) and what both PhalaCloud releases ship.

Safe by construction: writes a NEW directory, never mutates ``--src``. Idempotent
-- a checkpoint whose indexer is already e4m3 is reported and skipped.

Usage:
    python -m pipeline.patch_indexer_fp8 --src <converted> --out <new dir> \\
        [--base <bf16 snapshot>] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from pipeline.sglang_w4afp8_kernels import (
    DEFAULT_BLOCK,
    dequantize_block_fp8,
    quantize_block_fp8,
)

# The two modules SGLang builds with a quant_config. Kept as an explicit local
# constant rather than imported, so this tool keeps working on an artifact
# produced by any converter version.
TARGET_SUFFIXES = (
    ".self_attn.indexer.wk.weight",
    ".self_attn.indexer.wq_b.weight",
)

# Block-FP8 of a BF16 weight costs one e4m3 rounding, ~0.0265 relative. 0.05
# leaves room for the tail of the distribution while still catching a transposed
# axis or a reciprocal-vs-multiplier mistake, which land an order of magnitude up.
_MAX_RESID = 0.05

_AUX_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "recipe.json",
)


def _link(source: Path, dest: Path) -> str:
    """Hardlink, else symlink, never copy.

    RESOLVE FIRST: os.link does not follow symlinks, so linking a symlink
    reproduces its target verbatim -- and a relative target resolves to nowhere
    from a directory at a different depth. That produced nine dangling shards
    the first time the slicer did this.
    """
    real = source.resolve(strict=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(real, dest)
        return "hardlink"
    except OSError:
        os.symlink(real, dest)
        return "symlink"


def find_targets(src: Path, weight_map: dict) -> tuple[list[str], list[str]]:
    """(names needing conversion, names already e4m3)."""
    from safetensors import safe_open

    todo: list[str] = []
    done: list[str] = []
    opened: dict[str, object] = {}
    for name in sorted(k for k in weight_map if k.endswith(TARGET_SUFFIXES)):
        shard = weight_map[name]
        if shard not in opened:
            opened[shard] = safe_open(str(src / shard), framework="pt")
        dtype = opened[shard].get_slice(name).get_dtype()
        if dtype == "F8_E4M3":
            done.append(name)
        else:
            todo.append(name)
    return todo, done


def patch(
    src: Path,
    out: Path,
    base: Path | None = None,
    dry_run: bool = False,
) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = src / "model.safetensors.index.json"
    if not index_path.is_file():
        print(f"error: {index_path} not found", flush=True)
        return 2
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict = index["weight_map"]

    todo, done = find_targets(src, weight_map)
    if not todo:
        if done:
            print(f"[patch] nothing to do: all {len(done)} indexer tensor(s) are "
                  f"already e4m3", flush=True)
            return 0
        print("error: no DSA indexer wk/wq_b tensors found at all; wrong "
              "checkpoint, or the naming changed", flush=True)
        return 2
    if done:
        print(f"[patch] WARNING: {len(done)} indexer tensor(s) already e4m3 while "
              f"{len(todo)} are not; converting only the latter", flush=True)

    affected = sorted({weight_map[n] for n in todo})
    every = sorted(set(weight_map.values()))
    clean = [s for s in every if s not in set(affected)]
    aff_bytes = sum((src / s).stat().st_size for s in affected)
    print(f"[patch] {len(todo)} tensor(s) to convert across {len(affected)} of "
          f"{len(every)} shard(s) ({aff_bytes / 1e9:.0f} GB to rewrite); "
          f"{len(clean)} shard(s) will be hardlinked", flush=True)
    if dry_run:
        for name in todo:
            print(f"    {name} -> {weight_map[name]}")
        return 0

    out.mkdir(parents=True, exist_ok=True)

    # Optional provenance check. The converter copied these tensors through from
    # the AWQ checkpoint, which never touched them, so they should equal the BF16
    # source exactly. If they do not, something folded them and quantizing the
    # copy would bake in that fold.
    base_map = None
    if base is not None:
        from pipeline.serve_ignore import weight_map_of

        base_map = weight_map_of(base)

    new_entries: dict[str, str] = {}
    resids: list[tuple[float, str]] = []

    for shard in affected:
        tensors: dict = {}
        with safe_open(str(src / shard), framework="pt") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)

        for name in [n for n in todo if weight_map[n] == shard]:
            weight = tensors[name].float()
            if base_map is not None and name in base_map:
                with safe_open(str(base / base_map[name]), framework="pt") as bh:
                    ref = bh.get_tensor(name).float()
                if not torch.equal(weight, ref):
                    delta = ((weight - ref).norm() / ref.norm()).item()
                    print(f"error: {name} differs from the BF16 source by "
                          f"{delta:.4f}; it is not a clean passthrough, so "
                          f"quantizing it would bake in whatever changed it",
                          flush=True)
                    return 2

            qweight, scale_inv = quantize_block_fp8(weight, DEFAULT_BLOCK)
            back = dequantize_block_fp8(qweight, scale_inv, DEFAULT_BLOCK)
            resids.append((((back - weight).norm() / weight.norm()).item(), name))

            tensors[name] = qweight
            scale_name = name[: -len(".weight")] + ".weight_scale_inv"
            tensors[scale_name] = scale_inv
            new_entries[scale_name] = shard

        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        print(f"[patch] rewrote {shard}: {len(tensors)} tensors", flush=True)
        del tensors

    # ---- residual gate, before anything becomes loadable ---------------------
    from statistics import median

    ordered = sorted(resids)
    print(f"[patch] indexer block-fp8 residual: n={len(ordered)} "
          f"min={ordered[0][0]:.4f} median={median(v for v, _ in ordered):.4f} "
          f"max={ordered[-1][0]:.4f} (bound {_MAX_RESID})", flush=True)
    if ordered[-1][0] > _MAX_RESID:
        print(f"error: residual {ordered[-1][0]:.4f} on {ordered[-1][1]} exceeds "
              f"{_MAX_RESID}; refusing to write an index, so the output stays "
              f"unloadable rather than silently wrong", flush=True)
        return 1

    for shard in clean:
        _link(src / shard, out / shard)
    print(f"[patch] linked {len(clean)} unchanged shard(s)", flush=True)

    # ---- index, with total_size recomputed from the headers ------------------
    merged = dict(weight_map)
    merged.update(new_entries)
    total = 0
    for shard in every:
        with safe_open(str(out / shard), framework="pt") as handle:
            for key in handle.keys():
                if key not in merged:
                    continue
                slice_ = handle.get_slice(key)
                numel = 1
                for dim in slice_.get_shape():
                    numel *= dim
                total += numel * _dtype_bytes(slice_.get_dtype())
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": merged},
                   indent=2),
        encoding="utf-8",
    )

    for name in _AUX_FILES:
        source = src / name
        if source.is_file():
            shutil.copy2(source, out / name)

    # These modules are quantized now, so an ignore entry for them would be a
    # lie. SGLang never reads the field (from_config drops it), but vLLM does --
    # and reads it FIRST, which is the serve_ignore.py incident.
    config_path = out / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quant = config.get("quantization_config")
        if isinstance(quant, dict):
            stems = {n[: -len(".weight")] for n in todo}
            before = list(quant.get("ignored_layers", []) or [])
            after = [e for e in before if e not in stems]
            if len(after) != len(before):
                quant["ignored_layers"] = after
                config_path.write_text(json.dumps(config, indent=2),
                                       encoding="utf-8")
                print(f"[patch] dropped {len(before) - len(after)} now-quantized "
                      f"module(s) from ignored_layers", flush=True)

    print(f"[patch] OK -> {out} ({total / 1e9:.1f} GB indexed, "
          f"{len(merged)} tensors)", flush=True)
    return 0


def _dtype_bytes(dtype: str) -> int:
    return {
        "F64": 8, "I64": 8, "U64": 8,
        "F32": 4, "I32": 4, "U32": 4,
        "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
        "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
    }.get(dtype, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path,
                        help="converted w4afp8 checkpoint with a BF16 indexer")
    parser.add_argument("--out", required=True, type=Path,
                        help="new directory; --src is never modified")
    parser.add_argument("--base", type=Path, default=None,
                        help="BF16 snapshot, to verify the indexer tensors are "
                             "clean passthroughs before quantizing them")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return patch(args.src, args.out, base=args.base, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
