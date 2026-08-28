"""Build a truncated-depth copy of a large checkpoint on fast local storage.

Motivation (measured on the Rancher infermesh-test cluster, 2026-08-28)
----------------------------------------------------------------------
Every calibration experiment on GLM-5.2 pays ~63 min of setup before it measures
anything: transformers' ``Loading weights`` phase reads all 59,585 tensors off
cephfs (and discards ~99% of them -- see BUGS_AND_FIXES.md), then dispatch copies
a few GB into shm. That setup cost is what makes probe runs expensive enough to
hurt when one dies at 2 min or gets confounded by filesystem contention.

The GPU nodes each have ~642 GB free on a node-local NVMe device (``/dev/nvme1n1p2``,
ext4) that measures **1.7 GB/s** read -- flat from 1 to 16 concurrent readers --
against cephfs's 31 MB/s single-stream and 135-260 MB/s across 8 readers. The
whole model (1403 GiB) does not fit. But a depth-truncated copy does: layers 0..3
are **24.2 GiB** and touch only 9 of 282 source shards.

Why truncation is faithful for per-layer calibration timing
-----------------------------------------------------------
The sequential pipeline calibrates one decoder layer at a time, and layer L's
calibration inputs are produced by layers 0..L-1. GLM-5.2 has
``first_k_dense_replace = 3``, so layer 3 is the FIRST MoE layer, and a 0..3
subset reproduces its inputs exactly: same dense prefix, same weights, same
activations. What the subset removes is layers 4..77, which a
``stop_after_last_target`` run never calibrates anyway. So the measured cost of
calibrating layer 3 is unchanged, while setup drops from ~63 min to under a
minute.

This is NOT a general-purpose model surgeon. It produces a checkpoint valid for
depth-local work (calibration timing, layer-level numerics); the truncated model
is not a usable language model and must never be served or evaluated for quality.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

_LAYER = re.compile(r"\.layers\.(\d+)\.")

# Files that describe the weight layout of the SOURCE and must not be copied
# verbatim -- they are rewritten for the subset.
_REWRITTEN = {"model.safetensors.index.json", "config.json"}

# Bound resident memory during the copy. Tensors are accumulated into an output
# shard buffer and flushed when it exceeds this, so peak RSS is roughly this plus
# the largest single tensor rather than the whole subset.
DEFAULT_SHARD_MAX_BYTES = 8 * 1024**3


def layer_of(key: str) -> int | None:
    """Decoder-layer index a weight belongs to, or None for non-layer weights
    (embeddings, final norm, lm_head)."""
    match = _LAYER.search(key)
    return int(match.group(1)) if match else None


def select_keys(weight_map: dict[str, str], num_layers: int) -> list[str]:
    """Keys to carry over: every non-layer weight, plus layers [0, num_layers).

    Non-layer weights are always kept because the model cannot be constructed
    without embed_tokens / norm / lm_head (GLM-5.2 sets
    ``tie_word_embeddings: False``, so lm_head is a real 2.9 GiB tensor, not an
    alias). Layers at or beyond the cut -- including the MTP layer, which sits at
    index ``num_hidden_layers`` -- are dropped.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    kept = []
    for key in weight_map:
        index = layer_of(key)
        if index is None or index < num_layers:
            kept.append(key)
    return sorted(kept)


def plan_shards(
    keys: list[str],
    weight_map: dict[str, str],
    sizes: dict[str, int],
    shard_max_bytes: int = DEFAULT_SHARD_MAX_BYTES,
) -> list[list[str]]:
    """Group kept keys into output shards, walking SOURCE shards in order.

    Grouping by source shard matters: each source file is then opened exactly
    once and read front-to-back, which is what makes the copy sequential rather
    than a scatter of seeks across 282 files.
    """
    by_source: dict[str, list[str]] = {}
    for key in keys:
        by_source.setdefault(weight_map[key], []).append(key)

    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for source in sorted(by_source):
        for key in sorted(by_source[source]):
            size = sizes.get(key, 0)
            if current and current_bytes + size > shard_max_bytes:
                groups.append(current)
                current, current_bytes = [], 0
            current.append(key)
            current_bytes += size
    if current:
        groups.append(current)
    return groups


def patch_config(config: dict, num_layers: int) -> dict:
    """Depth-truncate a config.

    ``num_nextn_predict_layers`` is forced to 0: GLM-5.2 ships 1, and the MTP
    layer's weights live at index ``num_hidden_layers`` (78) which the subset
    does not carry. Leaving it at 1 makes model construction look for a layer
    that is not there.

    PER-LAYER LISTS MUST BE TRUNCATED TOO. GLM-5.2 carries ``mlp_layer_types``,
    a 78-element list, and transformers validates its length against
    ``num_hidden_layers`` (configuration_utils.py::validate_layer_type), so
    setting depth alone raises::

        ValueError: `num_hidden_layers` (4) must be equal to the number of
                    `mlp_layer_types` (78)

    Rather than hardcode that one key -- other architectures use ``layer_types``,
    ``attn_layer_types``, per-layer expert counts and so on -- every list whose
    length matches the ORIGINAL depth (with or without the MTP layers) is
    truncated, and the names are returned in ``_subset_truncated_lists`` so the
    transformation is visible in the artifact rather than implicit.
    """
    patched = dict(config)
    original_depth = config.get("num_hidden_layers")
    mtp = config.get("num_nextn_predict_layers") or 0
    per_layer_lengths = {original_depth, original_depth + mtp} - {None}

    truncated = []
    for key, value in config.items():
        if isinstance(value, list) and len(value) in per_layer_lengths:
            patched[key] = value[:num_layers]
            truncated.append(key)

    patched["num_hidden_layers"] = num_layers
    if patched.get("num_nextn_predict_layers"):
        patched["num_nextn_predict_layers"] = 0
    if truncated:
        patched["_subset_truncated_lists"] = sorted(truncated)
    # Record provenance so a stray subset can never be mistaken for the model.
    patched["_subset_of"] = config.get("_name_or_path", "unknown")
    patched["_subset_num_layers"] = num_layers
    patched["_subset_warning"] = (
        "Depth-truncated copy for calibration timing only. Not a usable language "
        "model; never serve or evaluate for quality."
    )
    return patched


def _read_header(path: Path) -> dict:
    """safetensors header, without mapping any tensor data."""
    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        return json.loads(handle.read(length))


_DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "I8": 1, "U8": 1, "BOOL": 1,
    "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8,
}


def tensor_sizes(snapshot: Path, weight_map: dict[str, str]) -> dict[str, int]:
    """Byte size of every tensor, from shard headers only (no data read)."""
    by_source: dict[str, list[str]] = {}
    for key, source in weight_map.items():
        by_source.setdefault(source, []).append(key)
    sizes: dict[str, int] = {}
    for source, keys in by_source.items():
        header = _read_header(snapshot / source)
        for key in keys:
            meta = header.get(key)
            if meta is None:
                continue
            count = 1
            for dim in meta["shape"]:
                count *= dim
            sizes[key] = count * _DTYPE_BYTES.get(meta["dtype"], 2)
    return sizes


def build(
    snapshot: Path,
    out: Path,
    num_layers: int,
    shard_max_bytes: int = DEFAULT_SHARD_MAX_BYTES,
) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    keys = select_keys(weight_map, num_layers)
    sizes = tensor_sizes(snapshot, weight_map)
    groups = plan_shards(keys, weight_map, sizes, shard_max_bytes)

    out.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(sizes.get(k, 0) for k in keys)
    print(
        f"[subset] {len(keys)} of {len(weight_map)} tensors, "
        f"{total_bytes / 1024**3:.2f} GiB, {len(groups)} output shard(s), "
        f"from {len({weight_map[k] for k in keys})} source shard(s)",
        flush=True,
    )

    new_map: dict[str, str] = {}
    for shard_index, group in enumerate(groups, start=1):
        name = f"model-{shard_index:05d}-of-{len(groups):05d}.safetensors"
        # Read grouped by source file so each is opened once.
        by_source: dict[str, list[str]] = {}
        for key in group:
            by_source.setdefault(weight_map[key], []).append(key)
        tensors = {}
        for source in sorted(by_source):
            with safe_open(str(snapshot / source), framework="pt") as handle:
                for key in sorted(by_source[source]):
                    tensors[key] = handle.get_tensor(key)
        save_file(tensors, str(out / name), metadata={"format": "pt"})
        for key in group:
            new_map[key] = name
        written = sum(sizes.get(k, 0) for k in group)
        print(
            f"[subset]   wrote {name}: {len(group)} tensors, "
            f"{written / 1024**3:.2f} GiB",
            flush=True,
        )
        del tensors

    # Fail closed: a silently incomplete subset would surface much later as a
    # missing-weight error inside model construction.
    missing = set(keys) - set(new_map)
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected tensors were not written, e.g. "
            f"{sorted(missing)[:5]}"
        )
    stray = {k for k in new_map if (i := layer_of(k)) is not None and i >= num_layers}
    if stray:
        raise RuntimeError(f"subset contains out-of-range layers: {sorted(stray)[:5]}")

    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": new_map},
            indent=2,
        ),
        encoding="utf-8",
    )
    config = json.loads((snapshot / "config.json").read_text())
    (out / "config.json").write_text(
        json.dumps(patch_config(config, num_layers), indent=2), encoding="utf-8"
    )

    # Everything else (tokenizer, chat template, remote modelling code) verbatim.
    copied = []
    for path in sorted(snapshot.iterdir()):
        if path.is_dir() or path.name in _REWRITTEN:
            continue
        if path.name.endswith(".safetensors"):
            continue
        target = out / path.name
        # Snapshot entries are symlinks into the HF blob store; copy contents.
        shutil.copyfile(path, target, follow_symlinks=True)
        copied.append(path.name)
    print(f"[subset] copied {len(copied)} auxiliary file(s): {copied}", flush=True)

    return {
        "tensors": len(keys),
        "bytes": total_bytes,
        "shards": len(groups),
        "auxiliary_files": copied,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path,
                        help="source HF snapshot dir (with model.safetensors.index.json)")
    parser.add_argument("--out", required=True, type=Path,
                        help="destination dir, normally on node-local NVMe")
    parser.add_argument("--layers", required=True, type=int,
                        help="keep decoder layers [0, LAYERS). For GLM-5.2, 4 keeps "
                             "the 3 dense layers plus the first MoE layer (24.2 GiB)")
    parser.add_argument("--shard-max-bytes", type=int, default=DEFAULT_SHARD_MAX_BYTES)
    args = parser.parse_args(argv)

    if not (args.snapshot / "model.safetensors.index.json").exists():
        print(f"error: no model.safetensors.index.json under {args.snapshot}")
        return 2
    summary = build(args.snapshot, args.out, args.layers, args.shard_max_bytes)
    print(f"[subset] DONE {summary['bytes'] / 1024**3:.2f} GiB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
