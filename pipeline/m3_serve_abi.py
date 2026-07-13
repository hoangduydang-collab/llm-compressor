"""Static MiniMax-M3 compressed-tensors to vLLM serving ABI validation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from pipeline.m3_checkpoint_diagnostics import classify_module


_SUFFIXES = (
    ".weight_packed",
    ".weight_scale",
    ".weight_shape",
    ".weight_zero_point",
    ".weight",
)
_PLAIN_QUANTIZABLE = {
    "attention",
    "dense_mlp",
    "lm_head",
    "msa_indexer",
    "routers",
    "shared_experts",
    "vision",
}


def _module_name(key: str) -> str | None:
    for suffix in _SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def transformers_alias(runtime_name: str) -> str:
    """Translate known vLLM MiniMax-M3 fused names to Transformers names."""
    name = runtime_name.replace("language_model.model.", "model.language_model.", 1)
    name = name.replace(".block_sparse_moe.shared_experts.", ".mlp.shared_experts.")
    name = name.replace(".block_sparse_moe.gate", ".mlp.gate")
    name = name.replace(".block_sparse_moe.experts.", ".mlp.experts.")
    name = name.replace(".w1", ".gate_proj")
    name = name.replace(".w2", ".down_proj")
    name = name.replace(".w3", ".up_proj")
    name = name.replace(".self_attn.index_q_proj", ".self_attn.indexer.q_proj")
    name = name.replace(".self_attn.index_k_proj", ".self_attn.indexer.k_proj")
    return name


def _matches(pattern: str, module: str) -> bool:
    if pattern.startswith("re:"):
        try:
            return re.match(pattern[3:], module) is not None
        except re.error:
            return False
    return module == pattern or module.endswith(f".{pattern}")


def analyze_serving_abi(config: dict[str, Any], weight_keys: Iterable[str]) -> dict[str, Any]:
    """Return a JSON-serializable, GPU-free serving contract report."""
    quant = config.get("quantization_config") or {}
    ignore = list(quant.get("ignore") or [])
    keys = list(weight_keys)
    quantized = {
        name
        for key in keys
        if key.endswith(".weight_packed") and (name := _module_name(key))
    }
    scales = {
        name
        for key in keys
        if key.endswith(".weight_scale") and (name := _module_name(key))
    }
    plain = {
        name
        for key in keys
        if key.endswith(".weight") and (name := _module_name(key))
    }
    plain_quantizable = {
        name for name in plain if classify_module(name) in _PLAIN_QUANTIZABLE
    }
    runtime_modules = quantized | plain_quantizable
    source_modules = {transformers_alias(name) for name in runtime_modules}
    patterns = [
        {
            "pattern": pattern,
            "source_matches": sum(_matches(pattern, name) for name in source_modules),
            "runtime_matches": sum(_matches(pattern, name) for name in runtime_modules),
        }
        for pattern in ignore
    ]

    errors: list[dict[str, Any]] = []
    for pattern in ignore:
        if pattern.startswith("re:"):
            try:
                re.compile(pattern[3:])
            except re.error as error:
                errors.append(
                    {
                        "code": "invalid_ignore_regex",
                        "pattern": pattern,
                        "error": str(error),
                    }
                )
    groups = quant.get("config_groups") or {}
    for group_name, group in groups.items():
        targets = group.get("targets") or []
        if "Linear" not in targets:
            errors.append(
                {
                    "code": "quantization_group_does_not_target_linear",
                    "group": group_name,
                    "targets": targets,
                }
            )
    for name in sorted(quantized & plain):
        errors.append(
            {"code": "module_has_packed_and_plain_weight", "module": name}
        )
    for name in sorted(plain_quantizable):
        if not any(_matches(pattern, name) for pattern in ignore):
            errors.append(
                {
                    "code": "plain_runtime_module_not_ignored",
                    "module": name,
                    "source_alias": transformers_alias(name),
                    "component": classify_module(name),
                }
            )
    for name in sorted(quantized):
        matched = [pattern for pattern in ignore if _matches(pattern, name)]
        if matched:
            errors.append(
                {
                    "code": "packed_module_is_ignored",
                    "module": name,
                    "patterns": matched,
                }
            )
    for name in sorted(quantized - scales):
        errors.append({"code": "packed_module_missing_scale", "module": name})

    return {
        "schema_version": 1,
        "valid": not errors,
        "format": quant.get("format"),
        "inventory": {
            "quantized_modules": len(quantized),
            "plain_modules": len(plain),
            "plain_quantizable_modules": len(plain_quantizable),
            "runtime_modules_checked": len(runtime_modules),
        },
        "components": {
            component: {
                "quantized": sum(
                    classify_module(name) == component for name in quantized
                ),
                "plain": sum(classify_module(name) == component for name in plain),
                "plain_unignored": sum(
                    classify_module(name) == component
                    and not any(_matches(pattern, name) for pattern in ignore)
                    for name in plain_quantizable
                ),
            }
            for component in sorted(
                {classify_module(name) for name in quantized | plain}
            )
        },
        "patterns": patterns,
        "errors": errors,
    }


def analyze_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Analyze one checkpoint using only config and Safetensors index metadata."""
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if not config.get("quantization_config"):
        return {"schema_version": 1, "valid": True, "exempt": "unquantized"}
    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    return analyze_serving_abi(config, index["weight_map"].keys())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = analyze_checkpoint(args.checkpoint)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
