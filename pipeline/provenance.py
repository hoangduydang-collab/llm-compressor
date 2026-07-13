"""Environment + model provenance logging for MiniMax-M3 quant debugging.

The AWQ sequential-trace collapse (single subgraph, ``completed=0``, decoders
never calibrated) could not be reproduced from the model architecture offline;
it points at a load/environment factor -- most likely ``trust_remote_code``
loading a decoder class that is NOT literally named ``MiniMaxM3VLDecoderLayer``,
so the configured ``sequential_targets`` match zero modules.

This module captures, on the machine that actually loads the model, the two
things that localize that: (1) where the loaded modeling code comes from
(installed ``transformers`` vs a ``transformers_modules.*`` remote-code module),
and (2) whether ``sequential_targets`` actually match any module. Every field is
best-effort; collection never raises into the calling run.
"""

from __future__ import annotations

import inspect
import platform
import sys
from pathlib import Path
from typing import Any


def _version(module_name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(module_name)
    except Exception:
        try:
            mod = __import__(module_name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None


def _all_packages() -> dict[str, str]:
    """Full installed distribution -> version map (pip-freeze equivalent).

    Lets the cluster environment be diffed exactly against a local one, so a
    version skew in transformers / torch / llmcompressor / any transitive dep
    that changes tracing or model loading is visible rather than guessed.
    """
    try:
        import importlib.metadata as md

        packages: dict[str, str] = {}
        for dist in md.distributions():
            try:
                name = dist.metadata["Name"]
            except Exception:
                name = getattr(dist, "name", None)
            if not name:
                continue
            packages[name] = dist.version
        return dict(sorted(packages.items(), key=lambda kv: kv[0].lower()))
    except Exception:
        return {}


def _class_origin(obj: Any) -> dict[str, Any]:
    cls = type(obj)
    origin: dict[str, Any] = {
        "class": cls.__name__,
        "module": cls.__module__,
    }
    try:
        origin["file"] = inspect.getfile(cls)
    except Exception:
        origin["file"] = None
    # A `transformers_modules.*` module (or a file under a HF hub cache) is the
    # signature of trust_remote_code loading the checkpoint's own modeling code.
    origin["is_remote_code"] = str(cls.__module__).startswith("transformers_modules")
    return origin


def _env_block() -> dict[str, Any]:
    block: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "node": platform.node(),
        "executable": sys.executable,
        "versions": {
            name: _version(name)
            for name in (
                "transformers",
                "torch",
                "llmcompressor",
                "compressed_tensors",
                "compressed-tensors",
                "accelerate",
            )
        },
        # Full pip-freeze-equivalent snapshot for exact local-vs-cluster diffing.
        "installed_packages": _all_packages(),
    }
    try:
        import torch

        block["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
            "devices": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception:
        block["cuda"] = None
    return block


def _config_block(model: Any) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {}
    block: dict[str, Any] = {
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
        # auto_map is the definitive trust_remote_code indicator: it maps the
        # Auto* classes to remote-code module paths shipped with the checkpoint.
        "auto_map": getattr(cfg, "auto_map", None),
    }
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        block["text_config"] = {
            "model_type": getattr(text_cfg, "model_type", None),
            "num_hidden_layers": getattr(text_cfg, "num_hidden_layers", None),
        }
    return block


def _decoder_layers_block(model: Any) -> dict[str, Any]:
    """Distinct classes of modules whose name ends with 'DecoderLayer'."""
    by_class: dict[str, dict[str, Any]] = {}
    total = 0
    first_paths: list[str] = []
    try:
        for name, module in model.named_modules():
            if type(module).__name__.endswith("DecoderLayer"):
                total += 1
                if len(first_paths) < 3:
                    first_paths.append(name)
                key = type(module).__name__
                if key not in by_class:
                    entry = _class_origin(module)
                    entry["count"] = 0
                    by_class[key] = entry
                by_class[key]["count"] += 1
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "total_decoderlayer_modules": total,
        "sample_paths": first_paths,
        "distinct_classes": list(by_class.values()),
    }


def _target_match_block(
    model: Any, sequential_targets: Any
) -> dict[str, Any]:
    """Does each configured sequential target actually match any module?

    A zero match count is the direct cause of the single-subgraph collapse.
    """
    if not sequential_targets:
        return {"sequential_targets": sequential_targets, "note": "no targets set"}
    try:
        from compressed_tensors.utils.match import match_named_modules
    except Exception as exc:  # noqa: BLE001
        return {"error": f"match import failed: {type(exc).__name__}: {exc}"}

    targets = (
        [sequential_targets]
        if isinstance(sequential_targets, str)
        else list(sequential_targets)
    )
    per_target: list[dict[str, Any]] = []
    for target in targets:
        try:
            matched = list(match_named_modules(model, [target]))
        except Exception as exc:  # noqa: BLE001
            per_target.append(
                {"target": target, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        per_target.append(
            {
                "target": target,
                "match_count": len(matched),
                "sample_matched_paths": [name for name, _ in matched[:3]],
                "matched_classes": sorted(
                    {type(mod).__name__ for _, mod in matched}
                ),
            }
        )
    return {"per_target": per_target}


def collect_model_provenance(
    model: Any, sequential_targets: Any = None
) -> dict[str, Any]:
    """Gather environment + model-load provenance as a JSON-serializable dict."""
    return {
        "environment": _env_block(),
        "model": _class_origin(model),
        "config": _config_block(model),
        "decoder_layers": _decoder_layers_block(model),
        "target_match": _target_match_block(model, sequential_targets),
    }


def log_model_provenance(
    model: Any,
    sequential_targets: Any = None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Collect provenance, print a compact summary, and (optionally) write JSON.

    Returns the provenance dict. Best-effort: never raises into the caller.
    """
    try:
        prov = collect_model_provenance(model, sequential_targets)
    except Exception as exc:  # noqa: BLE001
        print(f"[provenance] collection failed: {type(exc).__name__}: {exc}")
        return {}

    model_info = prov.get("model", {})
    dl = prov.get("decoder_layers", {})
    tm = prov.get("target_match", {})
    env = prov.get("environment", {})
    versions = env.get("versions", {})
    ct_version = versions.get("compressed_tensors") or versions.get(
        "compressed-tensors"
    )
    n_packages = len(env.get("installed_packages", {}))
    print(
        "[provenance] env: "
        f"python={env.get('python')} "
        f"transformers={versions.get('transformers')} "
        f"torch={versions.get('torch')} "
        f"llmcompressor={versions.get('llmcompressor')} "
        f"compressed_tensors={ct_version} "
        f"({n_packages} packages captured)"
    )
    print(
        "[provenance] model="
        f"{model_info.get('class')} ({model_info.get('module')}) "
        f"remote_code={model_info.get('is_remote_code')}"
    )
    print(
        "[provenance] decoder classes: "
        + ", ".join(
            f"{c.get('class')}x{c.get('count')} "
            f"[{'REMOTE' if c.get('is_remote_code') else 'installed'}]"
            for c in dl.get("distinct_classes", [])
        )
        or "[provenance] decoder classes: none"
    )
    for entry in tm.get("per_target", []):
        if "error" in entry:
            print(f"[provenance] target {entry['target']!r}: ERROR {entry['error']}")
        else:
            flag = (
                "  <-- ZERO MATCH (collapse cause)"
                if entry["match_count"] == 0
                else ""
            )
            print(
                f"[provenance] target {entry['target']!r}: "
                f"match_count={entry['match_count']} "
                f"classes={entry['matched_classes']}{flag}"
            )

    if out_path is not None:
        import json

        out_path = Path(out_path)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(prov, fh, indent=2, default=str)
            print(f"[provenance] wrote {out_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[provenance] failed to write {out_path}: {exc}")
    return prov
