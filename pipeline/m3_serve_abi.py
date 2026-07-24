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


# ModelOpt / block-scale checkpoints (e.g. quant_method "mxfp8", "nvfp4",
# "modelopt*") do not pack weights: a quantized Linear keeps a low-bit
# ``.weight`` plus a ``.weight_scale_inv`` (MXFP8 microscale) or ``.weight_scale``
# (NVFP4) tensor, and the un-quantized modules are listed under ``ignored_layers``
# / ``exclude_modules`` in Transformers namespace. The compressed-tensors path
# below keys off ``.weight_packed`` + ``ignore`` + ``config_groups`` and would
# read every ModelOpt weight as "plain", so we dispatch that format separately.
_MODELOPT_METHODS = {
    "mxfp8",
    "nvfp4",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "fp8",
}
_MODELOPT_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale")


def _detect_format(quant: dict[str, Any], keys: list[str]) -> str:
    """Classify the on-disk quantization format from config + tensor names."""
    if any(key.endswith(".weight_packed") for key in keys) or quant.get("config_groups"):
        return "compressed-tensors"
    method = str(quant.get("quant_method") or "").lower()
    if (
        method in _MODELOPT_METHODS
        or quant.get("ignored_layers") is not None
        or quant.get("exclude_modules") is not None
        or any(key.endswith(".weight_scale_inv") for key in keys)
    ):
        return "modelopt-scale"
    return "compressed-tensors"


def _matches_modelopt(pattern: str, module: str) -> bool:
    """ModelOpt ignore semantics: exact, subtree prefix, or bare-leaf suffix."""
    if pattern.startswith("re:"):
        try:
            return re.match(pattern[3:], module) is not None
        except re.error:
            return False
    return (
        module == pattern
        or module.startswith(f"{pattern}.")
        or module.endswith(f".{pattern}")
    )


def _analyze_modelopt(quant: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Serving-ABI report for ModelOpt block-scale checkpoints (mxfp8/nvfp4)."""
    ignore = list(quant.get("ignored_layers") or quant.get("exclude_modules") or [])
    quantized = {
        key[: -len(suffix)]
        for key in keys
        for suffix in _MODELOPT_SCALE_SUFFIXES
        if key.endswith(suffix)
    }
    weight_modules = {
        key[: -len(".weight")] for key in keys if key.endswith(".weight")
    }
    plain = weight_modules - quantized
    plain_quantizable = {
        name for name in plain if classify_module(name) in _PLAIN_QUANTIZABLE
    }

    errors: list[dict[str, Any]] = []
    for pattern in ignore:
        if pattern.startswith("re:"):
            try:
                re.compile(pattern[3:])
            except re.error as error:
                errors.append(
                    {"code": "invalid_ignore_regex", "pattern": pattern, "error": str(error)}
                )
    for name in sorted(plain_quantizable):
        if not any(_matches_modelopt(pattern, name) for pattern in ignore):
            errors.append(
                {
                    "code": "plain_runtime_module_not_ignored",
                    "module": name,
                    "component": classify_module(name),
                }
            )
    for name in sorted(quantized):
        matched = [pattern for pattern in ignore if _matches_modelopt(pattern, name)]
        if matched:
            errors.append(
                {"code": "quantized_module_is_ignored", "module": name, "patterns": matched}
            )

    all_named = quantized | plain
    return {
        "schema_version": 1,
        "valid": not errors,
        "format": quant.get("quant_method") or quant.get("format"),
        "inventory": {
            "quantized_modules": len(quantized),
            "plain_modules": len(plain),
            "plain_quantizable_modules": len(plain_quantizable),
            "runtime_modules_checked": len(quantized | plain_quantizable),
        },
        "components": {
            component: {
                "quantized": sum(
                    classify_module(name) == component for name in quantized
                ),
                "plain": sum(classify_module(name) == component for name in plain),
                "plain_unignored": sum(
                    classify_module(name) == component
                    and not any(_matches_modelopt(pattern, name) for pattern in ignore)
                    for name in plain_quantizable
                ),
            }
            for component in sorted({classify_module(name) for name in all_named})
        },
        "patterns": [],
        "errors": errors,
    }


def analyze_serving_abi(config: dict[str, Any], weight_keys: Iterable[str]) -> dict[str, Any]:
    """Return a JSON-serializable, GPU-free serving contract report."""
    quant = config.get("quantization_config") or {}
    keys = list(weight_keys)
    if _detect_format(quant, keys) == "modelopt-scale":
        return _analyze_modelopt(quant, keys)
    ignore = list(quant.get("ignore") or [])
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
    # Mixed int4+fp8 checkpoints (r8/r8a lanes): fp8 Linears keep a plain
    # ``.weight`` (F8_E4M3) + ``.weight_scale`` and are matched by the regex
    # targets of a float-typed config group — targeted, NOT ignored. vLLM
    # checks ignore before targets, so an fp8 module that also matches an
    # ignore pattern serves its raw fp8 bits cast to bf16 (the r8 v1 bug).
    groups = quant.get("config_groups") or {}
    float_target_patterns = [
        target
        for group in groups.values()
        if str((group.get("weights") or {}).get("type") or "").lower() == "float"
        for target in (group.get("targets") or [])
        if isinstance(target, str) and target.startswith("re:")
    ]
    fp8_targeted = {
        name
        for name in plain_quantizable
        if any(_matches(pattern, name) for pattern in float_target_patterns)
    }
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
    for group_name, group in groups.items():
        targets = group.get("targets") or []
        if "Linear" in targets:
            continue
        # A group may forgo the Linear catch-all only if it is a float (fp8)
        # group whose regex targets actually hit runtime modules; anything
        # else is a quant layout vLLM will never apply.
        is_float = str((group.get("weights") or {}).get("type") or "").lower() == "float"
        regex_targets = [
            t for t in targets if isinstance(t, str) and t.startswith("re:")
        ]
        hits_runtime = any(
            _matches(t, name) for t in regex_targets for name in runtime_modules
        )
        if not (is_float and hits_runtime):
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
        matched_ignore = [pattern for pattern in ignore if _matches(pattern, name)]
        if name in fp8_targeted:
            if matched_ignore:
                errors.append(
                    {
                        "code": "fp8_module_is_ignored",
                        "module": name,
                        "patterns": matched_ignore,
                    }
                )
            if name not in scales:
                errors.append(
                    {"code": "fp8_targeted_module_missing_scale", "module": name}
                )
            continue
        if not matched_ignore:
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
            "fp8_targeted_modules": len(fp8_targeted),
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
