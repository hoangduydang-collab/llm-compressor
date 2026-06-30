"""Versioned artifact directories + reproducible run metadata.

Each run gets a directory ``<output_dir>/<run_slug>/<timestamp>/`` containing:
  - ``checkpoint/``   the saved compressed-tensors model
  - ``config.yaml``   the exact resolved pipeline config
  - ``recipe.json``   the recipe description
  - ``metadata.json`` git SHA, package versions, GPU, timing
  - ``eval_report.json`` (written by the eval gate)
"""

import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pipeline.config import PipelineConfig


def _git_sha(repo_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _gpu_info() -> dict:
    info: dict = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{major}.{minor}"  # 9.0 == Hopper SM90
    except Exception:
        pass
    return info


def create_run_dir(cfg: PipelineConfig) -> Path:
    """Create and return a fresh timestamped run directory."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.output_dir) / cfg.run_slug / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def checkpoint_dir(run_dir: Path) -> Path:
    return run_dir / "checkpoint"


def write_config(run_dir: Path, cfg: PipelineConfig) -> None:
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False)


def write_recipe(run_dir: Path, recipe_desc: dict) -> None:
    with (run_dir / "recipe.json").open("w", encoding="utf-8") as fh:
        json.dump(recipe_desc, fh, indent=2)


def write_metadata(run_dir: Path, cfg: PipelineConfig, extra: dict | None = None) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    meta = {
        "run_slug": cfg.run_slug,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": cfg.model.id,
        "method": cfg.quantization.method,
        "scheme": cfg.quantization.scheme,
        "git_sha": _git_sha(repo_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "llmcompressor": _pkg_version("llmcompressor"),
            "compressed-tensors": _pkg_version("compressed-tensors"),
            "transformers": _pkg_version("transformers"),
            "vllm": _pkg_version("vllm"),
            "lm_eval": _pkg_version("lm_eval"),
        },
        "gpu": _gpu_info(),
    }
    if extra:
        meta.update(extra)
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def write_eval_report(run_dir: Path, report: dict) -> Path:
    path = run_dir / "eval_report.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path
