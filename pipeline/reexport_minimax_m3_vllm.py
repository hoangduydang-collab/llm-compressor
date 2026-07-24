#!/usr/bin/env python
"""Create a portable MiniMax-M3 checkpoint with vLLM routed-expert aliases.

MiniMax-M3 checkpoints exported by recent Transformers versions name routed
expert projections ``gate_proj``, ``down_proj``, and ``up_proj``. The current
vLLM M3 loader expects ``w1``, ``w2``, and ``w3`` respectively. This utility
rewrites Safetensors headers and copies their raw payloads unchanged, avoiding
re-quantization and full-shard tensor materialization.

``--fp8-serve-fix`` additionally repairs mixed int4+FP8 (r8-class) checkpoints
for serving; see the block comment above ``_FP8_SERVE_TARGETS``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

_ROUTED_EXPERT_MARKER = ".block_sparse_moe.experts."
_ROUTED_PROJECTION_ALIASES = {
    ".gate_proj.": ".w1.",
    ".down_proj.": ".w2.",
    ".up_proj.": ".w3.",
}
_COPY_BUFFER_BYTES = 64 * 1024 * 1024

# --- r8-class mixed int4+FP8 serve fix -------------------------------------
#
# The r8/r8a recipes quantize attention q/k/v/o, shared experts, and dense
# layers 0-2 to FP8 at quant time (transformers layout:
# ``language_model.layers.N.self_attn.q_proj`` ...). The checkpoint is saved
# in serve layout (``language_model.model.layers.N...``, shared experts under
# ``block_sparse_moe.`` with split gate/up), but the serialized
# ``quantization_config`` still carries quant-layout target regexes AND the
# GPTQ recipe's broad quant-layout ignore regexes (``re:.*self_attn[.].*``,
# ...). vLLM checks ignore FIRST, so every FP8 module served as
# "unquantized": raw fp8 bits were cast into bf16 params without their
# scales -> garbage output (r8 ABI smoke, 2026-07-24).
#
# Additionally, vLLM's M3 plugin fuses q/k/v with the (deliberately-bf16)
# indexer projections into ONE GEMM weight on sparse layers
# (MinimaxM3QKVParallelLinearWithIndexer), so fp8 qkv cannot serve there at
# all. o_proj, shared experts, and dense MLPs have no such fusion conflict.
#
# ``--fp8-serve-fix`` therefore:
#   1. dequantizes attention q/k/v weights back to BF16 (dropping their
#      weight_scale tensors) -- numerically the fp8 values the GEMM would
#      have used, minus only the activation-quant perf win;
#   2. rewrites the float config group's targets to serve-layout regexes
#      (o_proj + shared experts + dense 0-2);
#   3. replaces the broad quant-layout ignore regexes with precise
#      serve-layout entries (q/k/v + indexer projections);
#   4. runs a fail-closed storage-vs-scheme consistency audit using vLLM's
#      own ignore matcher.

_ATTN_QKV_WEIGHT_RE = re.compile(
    r".*language_model\.model\.layers\.\d+\.self_attn\.(q|k|v)_proj\.weight$"
)
# CRITICAL (v2 crash, 2026-07-24): vLLM resolves each module's scheme by
# matching the module's OWN prefix -- and can only expand fused prefixes
# (gate_up_proj -> gate_proj/up_proj) through the model class's
# packed_modules_mapping, which the M3 NVIDIA plugin class does NOT define.
# Targets/ignores written only in checkpoint-shard names therefore match
# nothing: gate_up_proj fell through to the int4 Linear catch-all and its
# fp8 channel scale hit the int4 group-scale param shape assert. Every
# pattern below alternates over BOTH the fused vLLM prefix and the unfused
# shard names so it works with or without a fused mapping.
_FP8_SERVE_TARGETS = [
    "re:.*language_model[.]model[.]layers[.][0-9]+[.]self_attn[.]o_proj$",
    "re:.*language_model[.]model[.]layers[.][0-9]+[.]block_sparse_moe[.]"
    "shared_experts[.](gate_up_proj|gate_proj|up_proj|down_proj)$",
    "re:.*language_model[.]model[.]layers[.][0-2][.]mlp[.]"
    "(gate_up_proj|gate_proj|up_proj|down_proj)$",
]
_IGNORE_REMOVE = {
    "re:.*self_attn[.].*",
    "re:.*mlp[.]shared_experts[.].*",
    "re:.*block_sparse_moe[.]shared_experts[.].*",
    "re:.*layers[.][0-2][.].*",
    # superseded shard-only form from the first --fp8-serve-fix revision
    "re:.*language_model[.]model[.]layers[.][0-9]+[.]self_attn[.](q|k|v)_proj$",
}
_IGNORE_ADD = [
    "re:.*language_model[.]model[.]layers[.][0-9]+[.]self_attn[.]"
    "(qkv_proj|q_proj|k_proj|v_proj)$",
    "re:.*self_attn[.]index_(q|k)_proj$",
]
# Fused-module shard mapping mirroring the M3 plugin's fusions (qkv also
# fuses the indexer projections on sparse layers). The plugin class defines
# no packed_modules_mapping, so vLLM matches fused prefixes DIRECTLY; the
# audit checks both semantics.
_FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
# Unquantized-at-serve Linears must be explicitly ignored because the int4
# config group targets the ``Linear`` class (catch-all). Only these suffixes
# are Linear projections; norms/embeddings never match a Linear scheme.
_QUANTIZABLE_SUFFIX_RE = re.compile(r"(_proj|[.]gate|lm_head)$")


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


def _collect_tensor_metadata(
    source: Path, weight_map: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Read every shard header once; return key -> tensor metadata."""
    metadata: dict[str, dict[str, Any]] = {}
    for shard in sorted(set(weight_map.values())):
        header, _ = _read_safetensors_header(source / shard)
        for key, meta in header.items():
            if key != "__metadata__":
                metadata[key] = meta
    return metadata


def plan_fp8_serve_fix(
    tensor_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], set[str]]:
    """Return (weight -> weight_scale to dequant with, keys to drop)."""
    dequant: dict[str, str] = {}
    drop: set[str] = set()
    for key, meta in tensor_metadata.items():
        if not _ATTN_QKV_WEIGHT_RE.match(key):
            continue
        if meta["dtype"] != "F8_E4M3":
            raise ValueError(
                f"{key}: expected F8_E4M3 attention weight, found {meta['dtype']}"
            )
        scale_key = key + "_scale"
        if scale_key not in tensor_metadata:
            raise ValueError(f"{key}: missing sibling {scale_key}")
        dequant[key] = scale_key
        drop.add(scale_key)
    if not dequant:
        raise ValueError(
            "--fp8-serve-fix found no fp8 attention q/k/v weights to dequantize"
        )
    return dequant, drop


def _build_tensor_reader(
    source: Path, weight_map: dict[str, str]
) -> Callable[[str], tuple[bytes, str, list[int]]]:
    header_cache: dict[str, tuple[dict[str, Any], int]] = {}

    def read(key: str) -> tuple[bytes, str, list[int]]:
        shard = weight_map[key]
        if shard not in header_cache:
            header_cache[shard] = _read_safetensors_header(source / shard)
        header, header_length = header_cache[shard]
        meta = header[key]
        begin, end = meta["data_offsets"]
        with (source / shard).open("rb") as fh:
            fh.seek(8 + header_length + begin)
            raw = fh.read(end - begin)
        if len(raw) != end - begin:
            raise ValueError(f"short read for {key} in {shard}")
        return raw, meta["dtype"], meta["shape"]

    return read


def _dequant_fp8_to_bf16_bytes(
    weight_raw: bytes,
    shape: list[int],
    scale_raw: bytes,
    scale_dtype: str,
    scale_shape: list[int],
) -> bytes:
    import torch

    weight = (
        torch.frombuffer(bytearray(weight_raw), dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .reshape(shape)
    )
    torch_scale_dtype = {"F32": torch.float32, "BF16": torch.bfloat16,
                         "F16": torch.float16}.get(scale_dtype)
    if torch_scale_dtype is None:
        raise ValueError(f"unsupported weight_scale dtype {scale_dtype}")
    scale = torch.frombuffer(bytearray(scale_raw), dtype=torch_scale_dtype)
    if scale.numel() != shape[0]:
        raise ValueError(
            f"only per-output-channel scales supported: weight {shape}, "
            f"scale numel {scale.numel()} (shape {scale_shape})"
        )
    dequant = weight.to(torch.float32) * scale.to(torch.float32).reshape(-1, 1)
    return dequant.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()


def rewrite_safetensors_shard(
    source: Path,
    output: Path,
    *,
    drop_keys: frozenset[str] | set[str] = frozenset(),
    dequant_map: dict[str, str] | None = None,
    tensor_reader: Callable[[str], tuple[bytes, str, list[int]]] | None = None,
) -> int:
    """Rewrite a shard header and copy or transform its tensor payloads.

    Returns the number of renamed routed-expert tensor keys. ``output`` must
    not already exist so a partial or accidental re-export cannot overwrite a
    checkpoint. Keys in ``drop_keys`` are removed; keys in ``dequant_map``
    (fp8 weight -> its weight_scale key) are dequantized to BF16 using
    ``tensor_reader`` to fetch the scale (which may live in another shard).
    """
    dequant_map = dequant_map or {}
    if dequant_map and tensor_reader is None:
        raise ValueError("dequant_map requires a tensor_reader")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing shard: {output}")
    header, header_length = _read_safetensors_header(source)
    entries = sorted(
        ((k, v) for k, v in header.items() if k != "__metadata__"),
        key=lambda kv: kv[1]["data_offsets"][0],
    )
    rewritten: dict[str, Any] = {}
    if "__metadata__" in header:
        rewritten["__metadata__"] = header["__metadata__"]
    plans: list[tuple[str, str, int, int, list[int]]] = []
    cursor = 0
    renamed = 0
    for name, meta in entries:
        if name in drop_keys:
            continue
        target_name = rename_routed_expert_key(name)
        renamed += target_name != name
        if target_name in rewritten:
            raise ValueError(
                f"Key collision while rewriting {source.name}: {name} -> {target_name}"
            )
        begin, end = meta["data_offsets"]
        if name in dequant_map:
            numel = 1
            for dim in meta["shape"]:
                numel *= dim
            new_length = 2 * numel  # BF16
            rewritten[target_name] = {
                "dtype": "BF16",
                "shape": meta["shape"],
                "data_offsets": [cursor, cursor + new_length],
            }
            plans.append(("dequant", name, begin, end - begin, meta["shape"]))
            cursor += new_length
        else:
            length = end - begin
            rewritten[target_name] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [cursor, cursor + length],
            }
            plans.append(("copy", name, begin, length, meta["shape"]))
            cursor += length
    encoded_header = json.dumps(
        rewritten, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")

    with source.open("rb") as src, output.open("xb") as dst:
        dst.write(len(encoded_header).to_bytes(8, "little"))
        dst.write(encoded_header)
        payload_base = 8 + header_length
        for kind, name, begin, length, shape in plans:
            src.seek(payload_base + begin)
            if kind == "copy":
                remaining = length
                while remaining:
                    chunk = src.read(min(_COPY_BUFFER_BYTES, remaining))
                    if not chunk:
                        raise ValueError(f"short payload read for {name}")
                    dst.write(chunk)
                    remaining -= len(chunk)
            else:
                weight_raw = src.read(length)
                if len(weight_raw) != length:
                    raise ValueError(f"short payload read for {name}")
                scale_raw, scale_dtype, scale_shape = tensor_reader(
                    dequant_map[name]
                )
                dst.write(
                    _dequant_fp8_to_bf16_bytes(
                        weight_raw, shape, scale_raw, scale_dtype, scale_shape
                    )
                )
    return renamed


def _load_index(path: Path) -> dict[str, Any]:
    index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"{path} has no Safetensors weight_map")
    return index


def _expected_output_key(name: str, drop_keys: set[str]) -> str | None:
    if name in drop_keys:
        return None
    return rename_routed_expert_key(name)


def verify_reexport(
    source: Path,
    output: Path,
    *,
    drop_keys: frozenset[str] | set[str] = frozenset(),
    dequant_keys: frozenset[str] | set[str] = frozenset(),
) -> dict[str, int]:
    """Statically verify index keys, shard headers, and payload sizes."""
    source_index = _load_index(source / "model.safetensors.index.json")
    output_index = _load_index(output / "model.safetensors.index.json")
    drop_keys = set(drop_keys)
    expected_map = {}
    for name, shard in source_index["weight_map"].items():
        target = _expected_output_key(name, drop_keys)
        if target is not None:
            expected_map[target] = shard
    if expected_map != output_index["weight_map"]:
        raise ValueError("Output Safetensors index does not match renamed source keys")

    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for name, shard in expected_map.items():
        expected_by_shard[shard].add(name)
    for shard in sorted(set(source_index["weight_map"].values())):
        source_header, source_header_length = _read_safetensors_header(
            source / shard
        )
        output_header, output_header_length = _read_safetensors_header(
            output / shard
        )
        source_keys = set(source_header) - {"__metadata__"}
        output_keys = set(output_header) - {"__metadata__"}
        if output_keys != expected_by_shard.get(shard, set()):
            raise ValueError(f"{shard} output header keys do not match its index")

        expected_payload = 0
        for name in source_keys:
            begin, end = source_header[name]["data_offsets"]
            if name in drop_keys:
                continue
            if name in dequant_keys:
                expected_payload += 2 * (end - begin)  # F8_E4M3 -> BF16
            else:
                expected_payload += end - begin
        output_payload = (output / shard).stat().st_size - 8 - output_header_length
        if expected_payload != output_payload:
            raise ValueError(
                f"{shard} raw tensor payload size mismatch: "
                f"expected {expected_payload}, found {output_payload}"
            )
    renamed = sum(
        name != rename_routed_expert_key(name)
        for name in source_index["weight_map"]
        if name not in drop_keys
    )
    return {
        "shards": len(set(source_index["weight_map"].values())),
        "keys": len(expected_map),
        "renamed": renamed,
        "dequantized": len(set(dequant_keys)),
        "dropped": len(drop_keys),
    }


def rewrite_quantization_config_for_serve(config: dict[str, Any]) -> dict[str, Any]:
    """Rewrite quant-layout FP8 targets/ignores to serve layout, in place."""
    qc = config.get("quantization_config")
    if not qc:
        raise ValueError("config.json has no quantization_config")
    groups = qc.get("config_groups") or {}
    float_groups = [
        group
        for group in groups.values()
        if (group.get("weights") or {}).get("type") == "float"
    ]
    if len(float_groups) != 1:
        raise ValueError(
            f"expected exactly one float-weight config group, found "
            f"{len(float_groups)}"
        )
    float_group = float_groups[0]
    old_targets = list(float_group["targets"])
    float_group["targets"] = list(_FP8_SERVE_TARGETS)

    ignore = list(qc.get("ignore", []))
    removed = [
        entry
        for entry in ignore
        if entry in _IGNORE_REMOVE or ".self_attn.indexer." in entry
    ]
    kept = [entry for entry in ignore if entry not in removed]
    added = [entry for entry in _IGNORE_ADD if entry not in kept]
    qc["ignore"] = kept + added
    return {"old_targets": old_targets, "removed_ignores": removed, "added_ignores": added}


def audit_serve_consistency(output: Path) -> dict[str, int]:
    """Fail-closed check that on-disk storage agrees with the serve config.

    Classifies every stored module family (packed int4 / fp8 / plain) and
    asserts, with vLLM's own ignore matcher, that packed and fp8 modules are
    NOT ignored, fp8 modules match a float target, and plain Linear
    projections ARE ignored (the int4 group targets the ``Linear`` class, so
    an unignored plain Linear would be mis-schemed and silently mis-loaded).
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    qc = config["quantization_config"]
    ignore = qc.get("ignore", [])
    float_targets = [
        target
        for group in (qc.get("config_groups") or {}).values()
        if (group.get("weights") or {}).get("type") == "float"
        for target in group["targets"]
    ]

    index = _load_index(output / "model.safetensors.index.json")
    dtypes: dict[str, str] = {}
    weight_map = index["weight_map"]
    for shard in sorted(set(weight_map.values())):
        header, _ = _read_safetensors_header(output / shard)
        for key, meta in header.items():
            if key != "__metadata__":
                dtypes[key] = meta["dtype"]

    families: dict[str, dict[str, str]] = defaultdict(dict)
    for key, dtype in dtypes.items():
        if "." not in key:
            continue
        module, param = key.rsplit(".", 1)
        families[module][param] = dtype

    shard_to_fused = {
        shard: fused for fused, shards in _FUSED_MAPPING.items() for shard in shards
    }

    def fused_prefix(module: str) -> str:
        head, _, last = module.rpartition(".")
        fused = shard_to_fused.get(last)
        return f"{head}.{fused}" if fused else module

    def matches_float(module: str) -> bool:
        return any(
            target.startswith("re:") and re.match(target[3:], module)
            for target in float_targets
        )

    counts = {"packed": 0, "fp8": 0, "plain_linear": 0}
    errors: list[str] = []
    for module, params in sorted(families.items()):
        if module.startswith("mtp.") or ".mtp." in module:
            continue
        if "weight" not in params and "weight_packed" not in params:
            continue
        prefix = fused_prefix(module)
        # The M3 NVIDIA plugin class defines no packed_modules_mapping, so
        # vLLM matches the fused prefix DIRECTLY (empty mapping). Check that
        # semantics first, then the expanded semantics for future-proofing.
        ignored_direct = should_ignore_layer(prefix, ignore=ignore, fused_mapping={})
        ignored_expanded = should_ignore_layer(
            prefix, ignore=ignore, fused_mapping=_FUSED_MAPPING
        )
        if ignored_direct != ignored_expanded:
            errors.append(
                f"{module}: ignore verdict differs between direct "
                f"({ignored_direct}) and fused-expanded ({ignored_expanded}) "
                "matching"
            )
        ignored = ignored_direct
        if "weight_packed" in params:
            counts["packed"] += 1
            if ignored:
                errors.append(f"{module}: packed int4 but ignored at serve")
        elif params.get("weight") == "F8_E4M3":
            counts["fp8"] += 1
            if ignored:
                errors.append(f"{module}: fp8 stored but ignored at serve")
            elif not matches_float(prefix):
                errors.append(
                    f"{module}: fp8 stored but fused prefix {prefix} matches "
                    "no float target (vLLM matches the fused prefix directly)"
                )
            elif not matches_float(module):
                errors.append(f"{module}: fp8 stored but matches no float target")
        elif _QUANTIZABLE_SUFFIX_RE.search(module) and "vision_tower" not in module:
            counts["plain_linear"] += 1
            if not ignored:
                errors.append(
                    f"{module}: unquantized Linear not ignored (would be "
                    "mis-schemed by the int4 Linear catch-all)"
                )
            if matches_float(module) or matches_float(prefix):
                errors.append(f"{module}: plain weights but matches a float target")
    if errors:
        preview = "\n  ".join(errors[:40])
        raise ValueError(
            f"serve-consistency audit FAILED ({len(errors)} problems):\n  {preview}"
        )
    if counts["fp8"] == 0:
        raise ValueError("serve-consistency audit found no fp8-served modules")
    if counts["packed"] == 0:
        raise ValueError("serve-consistency audit found no packed int4 modules")
    return counts


def reexport_checkpoint(
    source: Path, output: Path, *, fp8_serve_fix: bool = False
) -> dict[str, int]:
    """Create and statically verify a vLLM-compatible MiniMax-M3 checkpoint."""
    source = source.resolve()
    output = output.resolve()
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    source_index = _load_index(index_path)
    weight_map = source_index["weight_map"]
    shards = sorted(set(weight_map.values()))
    missing = [shard for shard in shards if not (source / shard).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source shards: {', '.join(missing)}")

    dequant_map: dict[str, str] = {}
    drop_keys: set[str] = set()
    tensor_reader = None
    if fp8_serve_fix:
        tensor_metadata = _collect_tensor_metadata(source, weight_map)
        dequant_map, drop_keys = plan_fp8_serve_fix(tensor_metadata)
        tensor_reader = _build_tensor_reader(source, weight_map)

    output.mkdir(parents=True)
    for path in source.iterdir():
        if path.name == index_path.name or path.name in shards:
            continue
        target = output / path.name
        if path.is_dir():
            shutil.copytree(path, target, symlinks=True)
        else:
            shutil.copy2(path, target, follow_symlinks=False)

    renamed = sum(
        rewrite_safetensors_shard(
            source / shard,
            output / shard,
            drop_keys=drop_keys,
            dequant_map=dequant_map,
            tensor_reader=tensor_reader,
        )
        for shard in shards
    )
    rewritten_index = dict(source_index)
    rewritten_index["weight_map"] = {
        rename_routed_expert_key(name): shard
        for name, shard in weight_map.items()
        if name not in drop_keys
    }
    if fp8_serve_fix and isinstance(rewritten_index.get("metadata"), dict):
        total = 0
        for shard in shards:
            header, header_length = _read_safetensors_header(output / shard)
            total += sum(
                meta["data_offsets"][1] - meta["data_offsets"][0]
                for key, meta in header.items()
                if key != "__metadata__"
            )
        rewritten_index["metadata"] = dict(rewritten_index["metadata"])
        rewritten_index["metadata"]["total_size"] = total
    (output / index_path.name).write_text(
        json.dumps(rewritten_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if fp8_serve_fix:
        config_path = output / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary = rewrite_quantization_config_for_serve(config)
        config_path.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "[fp8-serve-fix] float targets rewritten "
            f"({len(summary['old_targets'])} -> {len(_FP8_SERVE_TARGETS)}), "
            f"ignores: -{len(summary['removed_ignores'])} "
            f"+{len(summary['added_ignores'])}, "
            f"dequantized {len(dequant_map)} qkv weights"
        )

    verified = verify_reexport(
        source,
        output,
        drop_keys=drop_keys,
        dequant_keys=set(dequant_map),
    )
    if renamed != verified["renamed"]:
        raise ValueError("Shard/index routed-key rename counts disagree")
    if fp8_serve_fix:
        counts = audit_serve_consistency(output)
        print(
            "[fp8-serve-fix] consistency audit OK: "
            f"{counts['fp8']} fp8, {counts['packed']} packed int4, "
            f"{counts['plain_linear']} ignored plain Linears"
        )
        verified.update({f"audit_{k}": v for k, v in counts.items()})
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-export MiniMax-M3 routed expert keys for stock vLLM."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--fp8-serve-fix",
        action="store_true",
        help=(
            "repair a mixed int4+FP8 (r8-class) checkpoint for serving: "
            "dequantize attention q/k/v to BF16, rewrite quantization_config "
            "targets/ignores to serve layout, and audit the result"
        ),
    )
    args = parser.parse_args()
    result = reexport_checkpoint(
        args.source, args.output, fp8_serve_fix=args.fp8_serve_fix
    )
    print(
        "re-export verified: "
        f"{result['shards']} shards, {result['keys']} keys, "
        f"{result['renamed']} routed keys renamed, "
        f"{result['dequantized']} dequantized, {result['dropped']} dropped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
