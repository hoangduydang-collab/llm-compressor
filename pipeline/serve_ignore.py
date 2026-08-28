"""Keep a checkpoint's ``quantization_config.ignore`` consistent with its tensors.

THE BUG THIS PREVENTS, in the words of the incident that found it
(``reexport_minimax_m3_vllm.py``, r8 ABI smoke 2026-07-24):

    the serialized ``quantization_config`` still carries [...] the GPTQ recipe's
    broad quant-layout ignore regexes (``re:.*self_attn[.].*``, ...). vLLM checks
    ignore FIRST, so every FP8 module served as "unquantized": raw fp8 bits were
    cast into bf16 params without their scales -> garbage output.

Exit code 0. Coherent-looking tokens. Every offline gate green, because none of
them read the ignore list against the tensors.

``_persist_ignore_to_config`` is what introduces the broad patterns. Its purpose is
sound -- llm-compressor prunes ignore entries that never matched a quantized
module, which drops coverage a catch-all ``targets: ["Linear"]`` group still needs
-- but it wrote the recipe's patterns VERBATIM, and a recipe pattern is written to
say "the int4 modifier must not touch this", which is a different statement from
"no loader should treat this as quantized". ``re:.*self_attn[.].*`` means the first
and is fatal as the second, because the FP8 modifier owns those same modules by
explicit target.

So the rule here: a pattern may be persisted only if it shadows nothing the
checkpoint actually quantized. One that does shadow is replaced by the concrete
unquantized modules it matches, which is derivable from the weight index and
preserves the coverage the catch-all group needs.

Model-agnostic on purpose. M3's remedy was a per-model reexport tool with
serve-layout regexes hand-written into it, which is why GLM inherited the defect
with no equivalent: the fix lived downstream of the cause.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Tensor suffixes that mark a module as carrying quantization metadata. A module
# with any of these is quantized in storage, so a loader that treats it as
# unquantized will mis-load it.
QUANT_MARKER_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_inv",
    ".weight_packed",
    ".weight_zero_point",
    ".weight_g_idx",
)
WEIGHT_SUFFIXES = (".weight", ".weight_packed")


def match_name(name: str, target: str) -> bool:
    """Mirror of ``compressed_tensors.utils.match.match_name``.

    ``re.match`` (prefix), not ``re.fullmatch``, and exact equality otherwise.
    vLLM's ``check_equal_or_regex_match`` behaves the same way. A checker using
    different semantics than the loader is worse than no checker.
    """
    if target.startswith("re:"):
        return re.match(target.removeprefix("re:"), name) is not None
    return target == name


def weight_map_of(ckpt: Path) -> dict[str, str]:
    """Tensor name -> shard filename, for sharded AND single-shard checkpoints.

    The single-shard case is not an edge case worth skipping: save_pretrained omits
    the index for anything small enough, which is every subset probe and small
    smoke we use for fast validation. Assuming the index exists has already bitten
    twice -- it made the smooth-fold gate print "skipped" and return (silent), and
    made verify_quant_checkpoint's dequant check raise FileNotFoundError after a
    successful save (loud). Both on the same day.
    """
    ckpt = Path(ckpt)
    index = ckpt / "model.safetensors.index.json"
    if index.exists():
        return dict(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    single = ckpt / "model.safetensors"
    if single.exists():
        from safetensors import safe_open

        with safe_open(single, framework="pt") as handle:
            return {name: single.name for name in handle.keys()}
    raise FileNotFoundError(f"no safetensors index or single shard under {ckpt}")


def _weight_keys(ckpt: Path) -> list[str]:
    return list(weight_map_of(ckpt))


def checkpoint_modules(ckpt: Path) -> tuple[set[str], set[str]]:
    """``(all weight-bearing modules, those quantized in storage)``."""
    keys = _weight_keys(ckpt)
    quantized = {
        key[: -len(suffix)]
        for key in keys
        for suffix in QUANT_MARKER_SUFFIXES
        if key.endswith(suffix)
    }
    modules = {
        key[: -len(suffix)]
        for key in keys
        for suffix in WEIGHT_SUFFIXES
        if key.endswith(suffix)
    }
    return modules | quantized, quantized


def shadowed_by(pattern: str, quantized: set[str]) -> list[str]:
    """Quantized modules this ignore pattern would hide from the loader."""
    return sorted(module for module in quantized if match_name(module, pattern))


def resolve_ignore_patterns(
    patterns: list[str],
    modules: set[str],
    quantized: set[str],
    *,
    max_concrete: int = 4096,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Turn recipe ignore patterns into entries safe to serve with.

    Returns ``(entries, report)``. A pattern that shadows nothing passes through
    unchanged. A pattern that shadows is dropped and replaced by the concrete
    UNQUANTIZED modules it matches -- the coverage a ``targets: ["Linear"]``
    catch-all group genuinely needs -- with the substitution recorded.

    ``max_concrete`` is a guard, not a policy: a partial-scope smoke recipe can
    ignore tens of thousands of modules, and writing all of them into config.json
    is not useful. Over the cap the pattern is dropped with the overflow recorded
    so the caller can refuse rather than silently ship a shadowing pattern.
    """
    unquantized = modules - quantized
    entries: list[str] = []
    report: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        shadowed = shadowed_by(pattern, quantized)
        if not shadowed:
            entries.append(pattern)
            continue
        concrete = sorted(m for m in unquantized if match_name(m, pattern))
        record: dict[str, Any] = {
            "shadowed": shadowed,
            "shadowed_count": len(shadowed),
            "replaced_with_count": len(concrete),
        }
        if len(concrete) > max_concrete:
            record["overflow"] = True
            report[pattern] = record
            continue
        entries.extend(concrete)
        report[pattern] = record
    # de-duplicate, preserving order
    seen: set[str] = set()
    deduped = [e for e in entries if not (e in seen or seen.add(e))]
    return deduped, report


def audit_checkpoint_ignore(ckpt: Path) -> dict[str, Any]:
    """Read a SAVED checkpoint and report ignore entries that shadow its tensors.

    Runs against an artifact rather than a recipe, so it works on checkpoints
    produced before this module existed -- which is the whole point: the r8-class
    defect is only visible by comparing the shipped config against the shipped
    tensors.
    """
    config = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    quant_config = config.get("quantization_config") or {}
    ignore = list(quant_config.get("ignore") or [])
    modules, quantized = checkpoint_modules(ckpt)

    shadowing = {}
    for pattern in ignore:
        hits = shadowed_by(pattern, quantized)
        if hits:
            shadowing[pattern] = hits

    all_shadowed = sorted({m for hits in shadowing.values() for m in hits})
    return {
        "checkpoint": str(ckpt),
        "ignore_entries": len(ignore),
        "quantized_modules": len(quantized),
        "shadowing_patterns": {p: len(h) for p, h in shadowing.items()},
        "shadowed_modules": all_shadowed,
        "shadowed_module_count": len(all_shadowed),
        "ok": not all_shadowed,
    }


def assert_no_ignore_shadowing(ckpt: Path) -> None:
    """Fail-closed gate. A shadowed module will serve as unquantized."""
    report = audit_checkpoint_ignore(Path(ckpt))
    if report["ok"]:
        print(
            "[pipeline] serve-ignore gate OK: no ignore entry shadows any of the "
            f"{report['quantized_modules']} quantized modules"
        )
        return
    examples = report["shadowed_modules"][:8]
    raise RuntimeError(
        "serve-ignore gate FAILED — the saved quantization_config.ignore hides "
        f"{report['shadowed_module_count']} modules that ARE quantized in this "
        "checkpoint. Loaders check ignore before targets, so those modules serve "
        "as unquantized: their quantized bytes get cast into unscaled parameters "
        "and the model emits garbage at exit code 0 (M3 r8 ABI smoke, "
        "2026-07-24). Shadowing patterns: "
        + json.dumps(report["shadowing_patterns"], indent=2)
        + f"\n  e.g. shadowed: {examples}"
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_checkpoint_ignore(args.checkpoint)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"checkpoint         : {report['checkpoint']}")
        print(f"ignore entries     : {report['ignore_entries']}")
        print(f"quantized modules  : {report['quantized_modules']}")
        print(f"shadowed modules   : {report['shadowed_module_count']}")
        for pattern, count in report["shadowing_patterns"].items():
            print(f"  {pattern}  -> shadows {count}")
        for module in report["shadowed_modules"][:12]:
            print(f"    e.g. {module}")
        print("RESULT:", "OK" if report["ok"] else "SHADOWING — WILL SERVE GARBAGE")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
