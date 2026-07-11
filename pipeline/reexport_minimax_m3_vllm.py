#!/usr/bin/env python
"""Create a portable MiniMax-M3 checkpoint with vLLM routed-expert aliases.

MiniMax-M3 checkpoints exported by recent Transformers versions name routed
expert projections ``gate_proj``, ``down_proj``, and ``up_proj``. The current
vLLM M3 loader expects ``w1``, ``w2``, and ``w3`` respectively. This utility
rewrites Safetensors headers and copies their raw payloads unchanged, avoiding
re-quantization and full-shard tensor materialization.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROUTED_EXPERT_MARKER = ".block_sparse_moe.experts."
_ROUTED_PROJECTION_ALIASES = {
    ".gate_proj.": ".w1.",
    ".down_proj.": ".w2.",
    ".up_proj.": ".w3.",
}
_COPY_BUFFER_BYTES = 64 * 1024 * 1024


def rename_routed_expert_key(name: str) -> str:
    """Convert one routed-expert projection key to the vLLM ``w1/w2/w3`` form."""
    if _ROUTED_EXPERT_MARKER not in name:
        return name
    for source, target in _ROUTED_PROJECTION_ALIASES.items():
        if source in name:
            return name.replace(source, target, 1)
    return name


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path} is not a complete Safetensors file")
        header_length = int.from_bytes(raw_length, "little")
        raw_header = handle.read(header_length)
    if len(raw_header) != header_length:
        raise ValueError(f"{path} has a truncated Safetensors header")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError(f"{path} has a non-object Safetensors header")
    return header, header_length


def rewrite_safetensors_shard(source: Path, output: Path) -> int:
    """Rewrite a shard header and copy its tensor payload byte-for-byte.

    Returns the number of renamed routed-expert tensor keys. ``output`` must
    not already exist so a partial or accidental re-export cannot overwrite a
    checkpoint.
    """
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing shard: {output}")
    header, header_length = _read_safetensors_header(source)
    rewritten: dict[str, Any] = {}
    renamed = 0
    for name, metadata in header.items():
        target_name = (
            name if name == "__metadata__" else rename_routed_expert_key(name)
        )
        if target_name in rewritten:
            raise ValueError(
                f"Key collision while rewriting {source.name}: {name} -> {target_name}"
            )
        rewritten[target_name] = metadata
        renamed += target_name != name
    encoded_header = json.dumps(
        rewritten, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")

    with source.open("rb") as src, output.open("xb") as dst:
        src.seek(8 + header_length)
        dst.write(len(encoded_header).to_bytes(8, "little"))
        dst.write(encoded_header)
        shutil.copyfileobj(src, dst, length=_COPY_BUFFER_BYTES)
    return renamed


def _load_index(path: Path) -> dict[str, Any]:
    index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"{path} has no Safetensors weight_map")
    return index


def verify_reexport(source: Path, output: Path) -> dict[str, int]:
    """Statically verify index keys, shard headers, and raw payload sizes."""
    source_index = _load_index(source / "model.safetensors.index.json")
    output_index = _load_index(output / "model.safetensors.index.json")
    expected_map = {
        rename_routed_expert_key(name): shard
        for name, shard in source_index["weight_map"].items()
    }
    if expected_map != output_index["weight_map"]:
        raise ValueError("Output Safetensors index does not match renamed source keys")

    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for name, shard in expected_map.items():
        expected_by_shard[shard].add(name)
    for shard, expected_keys in expected_by_shard.items():
        source_header, source_header_length = _read_safetensors_header(
            source / shard
        )
        output_header, output_header_length = _read_safetensors_header(
            output / shard
        )
        source_keys = set(source_header) - {"__metadata__"}
        output_keys = set(output_header) - {"__metadata__"}
        if output_keys != expected_keys:
            raise ValueError(f"{shard} output header keys do not match its index")
        if len(source_keys) != len(output_keys):
            raise ValueError(f"{shard} changed its tensor count")
        source_payload = (source / shard).stat().st_size - 8 - source_header_length
        output_payload = (output / shard).stat().st_size - 8 - output_header_length
        if source_payload != output_payload:
            raise ValueError(f"{shard} raw tensor payload size changed")
    renamed = sum(
        name != rename_routed_expert_key(name)
        for name in source_index["weight_map"]
    )
    return {"shards": len(expected_by_shard), "keys": len(expected_map), "renamed": renamed}


def reexport_checkpoint(source: Path, output: Path) -> dict[str, int]:
    """Create and statically verify a vLLM-compatible MiniMax-M3 checkpoint."""
    source = source.resolve()
    output = output.resolve()
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    source_index = _load_index(index_path)
    shards = sorted(set(source_index["weight_map"].values()))
    missing = [shard for shard in shards if not (source / shard).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source shards: {', '.join(missing)}")

    output.mkdir(parents=True)
    try:
        for path in source.iterdir():
            if path.name == index_path.name or path.name in shards:
                continue
            target = output / path.name
            if path.is_dir():
                shutil.copytree(path, target, symlinks=True)
            else:
                shutil.copy2(path, target, follow_symlinks=False)

        renamed = sum(
            rewrite_safetensors_shard(source / shard, output / shard)
            for shard in shards
        )
        rewritten_index = dict(source_index)
        rewritten_index["weight_map"] = {
            rename_routed_expert_key(name): shard
            for name, shard in source_index["weight_map"].items()
        }
        (output / index_path.name).write_text(
            json.dumps(rewritten_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verified = verify_reexport(source, output)
        if renamed != verified["renamed"]:
            raise ValueError("Shard/index routed-key rename counts disagree")
        return verified
    except Exception:
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-export MiniMax-M3 routed expert keys for stock vLLM."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = reexport_checkpoint(args.source, args.output)
    print(
        "re-export verified: "
        f"{result['shards']} shards, {result['keys']} keys, "
        f"{result['renamed']} routed keys renamed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
