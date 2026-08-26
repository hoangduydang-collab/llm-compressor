#!/usr/bin/env python
"""Graft a model's MTP (multi-token-prediction) head into a quantized checkpoint.

Why this exists
---------------
transformers does not model MTP for ANY architecture. `eh_proj` and `shared_head`
appear in zero files package-wide; `num_nextn_predict_layers` appears only in two
*configuration* modules; and `deepseek_v3/modeling_deepseek_v3.py` — the canonical
MTP model — builds exactly `config.num_hidden_layers` layers. Hugging Face
implements the base LM and leaves MTP/EAGLE draft heads to the serving engines,
so the released checkpoints' MTP tensors are loaded-and-ignored as unexpected
keys.

Consequently any llm-compressor run over GLM-5.2 (or DeepSeek-V3/V4) emits a
checkpoint with layers 0..N-1 and no draft head, which cannot serve a production
configuration that enables speculative decoding — and a quality evaluation scores
that checkpoint perfectly clean, because only decode speed changes. Upgrading
transformers does not fix it. Grafting the head back does.

The head does not have to match the body's precision: EAGLE verifies drafts
against the target model, so draft-head precision costs acceptance *rate*, not
output quality. Copying it at source precision (one layer out of N+1) is a sound
default; the graft is a byte-for-byte copy, so whatever the source holds is what
lands.

What it does
------------
Copies every `model.layers.<L>.*` tensor from `--source` into `--target`, writing
new shards, patching the Safetensors index, and setting
`num_nextn_predict_layers` in the target config. Payloads are copied as raw bytes
with their dtype string preserved, so no dtype is ever reinterpreted — an int4,
FP8 or BF16 head all survive unchanged.

Fail-closed throughout: it refuses to run if the target already has MTP keys, if
the target does not look like it is missing exactly that layer, if a shard it
would write already exists, or if the post-write verification does not reproduce
every source key with matching dtype and shape. `--source` is opened read-only
and never modified.

Usage
-----
    python -m pipeline.graft_mtp_head --target OUR_CKPT --source RELEASE_SNAPSHOT
    python -m pipeline.graft_mtp_head --target ... --source ... --dry-run
    python -m pipeline.graft_mtp_head --target ... --source ... --verify-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from pipeline.reexport_minimax_m3_vllm import (
    _build_tensor_reader,
    _load_index,
    _read_safetensors_header,
)

INDEX_NAME = "model.safetensors.index.json"
DEFAULT_MAX_SHARD_BYTES = 4 * 1024**3  # 4 GiB, comfortably under HF conventions


def mtp_key_pattern(layer: int) -> re.Pattern[str]:
    """Every tensor belonging to decoder layer `layer`, with or without prefix."""
    return re.compile(rf"^(?:model\.)?layers\.{layer}\.")


def layer_indices(keys: object) -> set[int]:
    out: set[int] = set()
    for key in keys:  # type: ignore[union-attr]
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
        if match:
            out.add(int(match.group(1)))
    return out


def plan_graft(
    target_index: dict[str, Any],
    source_index: dict[str, Any],
    layer: int,
) -> list[str]:
    """Return the source keys to copy, or raise with a precise reason."""
    pattern = mtp_key_pattern(layer)
    src_map = source_index["weight_map"]
    tgt_map = target_index["weight_map"]

    keys = sorted(k for k in src_map if pattern.match(k))
    if not keys:
        raise ValueError(
            f"--source has no tensors for layer {layer}; nothing to graft. "
            f"Source layers present: {sorted(layer_indices(src_map))[-3:]}"
        )

    already = sorted(k for k in tgt_map if pattern.match(k))
    if already:
        raise ValueError(
            f"--target already has {len(already)} layer-{layer} tensors "
            f"(e.g. {already[0]}); refusing to graft twice"
        )

    tgt_layers = layer_indices(tgt_map)
    if not tgt_layers:
        raise ValueError("--target index has no layers.N tensors; is it a checkpoint?")
    if max(tgt_layers) != layer - 1:
        raise ValueError(
            f"--target's highest layer is {max(tgt_layers)}, expected {layer - 1}. "
            f"Grafting layer {layer} onto it would leave a gap or an overlap"
        )
    return keys


def _write_shard(
    path: Path,
    entries: list[tuple[str, bytes, str, list[int]]],
) -> int:
    """Write one Safetensors shard from raw payloads. Returns payload bytes."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing shard: {path}")
    header: dict[str, Any] = {}
    cursor = 0
    for name, raw, dtype, shape in entries:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(raw)],
        }
        cursor += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Safetensors requires the header be 8-byte aligned.
    blob += b" " * (-len(blob) % 8)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    with tmp.open("wb") as fh:
        fh.write(len(blob).to_bytes(8, "little"))
        fh.write(blob)
        for _, raw, _, _ in entries:
            fh.write(raw)
    tmp.rename(path)
    return cursor


def verify_graft(target: Path, source: Path, layer: int) -> dict[str, Any]:
    """Re-read both checkpoints and confirm the graft is complete and faithful."""
    pattern = mtp_key_pattern(layer)
    tgt_index = _load_index(target / INDEX_NAME)
    src_index = _load_index(source / INDEX_NAME)
    src_keys = {k for k in src_index["weight_map"] if pattern.match(k)}
    tgt_keys = {k for k in tgt_index["weight_map"] if pattern.match(k)}

    missing = src_keys - tgt_keys
    if missing:
        raise ValueError(
            f"verification failed: {len(missing)} MTP keys absent from target "
            f"(e.g. {sorted(missing)[0]})"
        )
    extra = tgt_keys - src_keys
    if extra:
        raise ValueError(
            f"verification failed: target has {len(extra)} layer-{layer} keys the "
            f"source does not (e.g. {sorted(extra)[0]})"
        )

    # dtype/shape must match, and every shard must actually resolve.
    src_meta: dict[str, dict[str, Any]] = {}
    for shard in sorted({src_index["weight_map"][k] for k in src_keys}):
        header, _ = _read_safetensors_header(source / shard)
        src_meta.update({k: v for k, v in header.items() if k in src_keys})
    tgt_meta: dict[str, dict[str, Any]] = {}
    for shard in sorted({tgt_index["weight_map"][k] for k in tgt_keys}):
        path = target / shard
        if not path.exists():
            raise ValueError(f"verification failed: index names a missing {shard}")
        header, _ = _read_safetensors_header(path)
        tgt_meta.update({k: v for k, v in header.items() if k in tgt_keys})

    for key in sorted(src_keys):
        s, t = src_meta[key], tgt_meta[key]
        if s["dtype"] != t["dtype"] or list(s["shape"]) != list(t["shape"]):
            raise ValueError(
                f"verification failed: {key} is {t['dtype']}{t['shape']} in target "
                f"but {s['dtype']}{s['shape']} in source"
            )

    cfg_path = target / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    return {
        "layer": layer,
        "tensors": len(src_keys),
        "dtypes": sorted({m["dtype"] for m in tgt_meta.values()}),
        "num_nextn_predict_layers": cfg.get("num_nextn_predict_layers"),
        "verified": True,
    }


def graft(
    target: Path,
    source: Path,
    layer: int,
    *,
    dry_run: bool = False,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> dict[str, Any]:
    tgt_index_path = target / INDEX_NAME
    tgt_index = _load_index(tgt_index_path)
    src_index = _load_index(source / INDEX_NAME)

    keys = plan_graft(tgt_index, src_index, layer)
    read = _build_tensor_reader(source, src_index["weight_map"])

    # Size the plan first so --dry-run reports real numbers.
    src_meta: dict[str, dict[str, Any]] = {}
    for shard in sorted({src_index["weight_map"][k] for k in keys}):
        header, _ = _read_safetensors_header(source / shard)
        src_meta.update({k: v for k, v in header.items() if k in set(keys)})
    total_bytes = sum(
        m["data_offsets"][1] - m["data_offsets"][0] for m in src_meta.values()
    )
    n_shards = max(1, -(-total_bytes // max_shard_bytes))

    summary = {
        "layer": layer,
        "tensors": len(keys),
        "bytes": total_bytes,
        "shards": n_shards,
        "dtypes": sorted({m["dtype"] for m in src_meta.values()}),
        "source_shards_touched": len({src_index["weight_map"][k] for k in keys}),
    }
    if dry_run:
        summary["dry_run"] = True
        return summary

    # Group keys into shards, keeping each under the size cap.
    groups: list[list[str]] = [[]]
    running = 0
    for key in keys:
        size = src_meta[key]["data_offsets"][1] - src_meta[key]["data_offsets"][0]
        if running + size > max_shard_bytes and groups[-1]:
            groups.append([])
            running = 0
        groups[-1].append(key)
        running += size

    written = 0
    weight_map = dict(tgt_index["weight_map"])
    for i, group in enumerate(groups, start=1):
        name = f"model-mtp-{i:05d}-of-{len(groups):05d}.safetensors"
        entries = []
        for key in group:
            raw, dtype, shape = read(key)
            entries.append((key, raw, dtype, shape))
        written += _write_shard(target / name, entries)
        for key in group:
            weight_map[key] = name

    # Patch the index: new keys, and total_size if the source tracked one.
    tgt_index["weight_map"] = weight_map
    meta = tgt_index.setdefault("metadata", {})
    if isinstance(meta.get("total_size"), int):
        meta["total_size"] += written
    tgt_index_path.write_text(json.dumps(tgt_index, indent=2), encoding="utf-8")

    # Tell the serving engine the head is there.
    cfg_path = target / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        src_cfg_path = source / "config.json"
        n = 1
        if src_cfg_path.exists():
            n = json.loads(src_cfg_path.read_text(encoding="utf-8")).get(
                "num_nextn_predict_layers", 1
            )
        cfg["num_nextn_predict_layers"] = n
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        summary["num_nextn_predict_layers"] = n

    summary["written_bytes"] = written
    summary["shard_names"] = [
        f"model-mtp-{i:05d}-of-{len(groups):05d}.safetensors"
        for i in range(1, len(groups) + 1)
    ]
    # Verify here rather than in the CLI: a programmatic caller must not be able
    # to obtain an unverified graft by skipping a wrapper.
    summary |= verify_graft(target, source, layer)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, type=Path, help="quantized checkpoint")
    ap.add_argument("--source", required=True, type=Path, help="release snapshot dir")
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="MTP layer index (default: source's highest layer)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES)
    args = ap.parse_args()

    layer = args.layer
    if layer is None:
        src_index = _load_index(args.source / INDEX_NAME)
        found = layer_indices(src_index["weight_map"])
        if not found:
            print("--source has no layers.N tensors", file=sys.stderr)
            return 2
        layer = max(found)
        print(f"==> --layer not given; using source's highest layer: {layer}")

    try:
        if args.verify_only:
            result = verify_graft(args.target, args.source, layer)
        else:
            # graft() verifies itself unless --dry-run.
            result = graft(
                args.target,
                args.source,
                layer,
                dry_run=args.dry_run,
                max_shard_bytes=args.max_shard_bytes,
            )
    except (ValueError, FileExistsError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
