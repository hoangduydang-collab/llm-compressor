"""NVIDIA gpqa_diamond_aa_v3 client contract (PyPI nvidia-simple-evals==26.3).

Does not implement GPQA prompt or extract regex. Those live in the pinned
wheel's framework.yml. This module only writes the run-config NVIDIA's
``nemo-evaluator run_eval`` consumes, plus GLM (max) decode overrides.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import yaml

PACKAGE_PIN = "nvidia-simple-evals==26.3"
# nvidia-simple-evals requires nemo-evaluator>=0.1.51. Unpinned, pip
# resolves 0.3.0 — a different product whose CLI is `nel`, not
# `nemo-evaluator ls` / `run_eval`. Cap below that rewrite.
NEMO_EVALUATOR_PIN = "nemo-evaluator>=0.1.51,<0.3"
TASK_NAME = "gpqa_diamond_aa_v3"
DEFAULT_VENV = "/mnt/cephfs/hoangduy/venvs/nvidia-simple-evals-26.3"

# AA reasoning / GLM (max). Not NVIDIA's greedy task defaults.
TEMPERATURE = 0.6
TOP_P = 1.0
MAX_NEW_TOKENS = 65536
REQUEST_TIMEOUT = 3600
N_SAMPLES = 5

PathLike = Union[str, Path]


class TaskMissingError(RuntimeError):
    """``nemo-evaluator ls`` did not list the required task."""


def require_task(ls_stdout: str, task: str = TASK_NAME) -> None:
    """Fail closed unless *task* appears as its own token in ls output.

    Substring match on ``gpqa_diamond_aa_v2`` must not satisfy ``..._v3``.
    """
    names = set()
    for raw in ls_stdout.splitlines():
        line = raw.strip().lstrip("*").strip()
        if not line:
            continue
        name = line.split()[0]
        names.add(name)
        # ``gpqa_diamond_aa_v3 (in simple_evals)``
        if "(" in name:
            names.add(name.split("(", 1)[0])
    if task not in names:
        raise TaskMissingError(
            f"{task!r} is not in nemo-evaluator ls (looked at {sorted(names)}). "
            f"Need {PACKAGE_PIN}; do not fall back to gpqa_diamond_aa_v2."
        )


def write_run_config(
    path: PathLike,
    *,
    url: str,
    model_id: str,
    output_dir: str,
    limit_samples: Optional[int] = None,
) -> dict[str, Any]:
    """Write a nemo-evaluator ``--run_config`` YAML. Returns the document."""
    cfg: dict[str, Any] = {
        "config": {
            "type": TASK_NAME,
            "output_dir": output_dir,
            "params": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS,
                "request_timeout": REQUEST_TIMEOUT,
                "limit_samples": limit_samples,
                "extra": {"n_samples": N_SAMPLES},
            },
        },
        "target": {
            "api_endpoint": {
                "url": url,
                "model_id": model_id,
                "type": "chat",
                "adapter_config": {
                    "params_to_add": {
                        "chat_template_kwargs": {"enable_thinking": True}
                    }
                },
            }
        },
    }
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return cfg


def build_run_eval_argv(nemo_evaluator: str, config_path: str) -> list[str]:
    return [nemo_evaluator, "run_eval", "--run_config", config_path]


def write_manifest(
    path: PathLike,
    **fields: Any,
) -> dict[str, Any]:
    man: dict[str, Any] = {
        "package": PACKAGE_PIN,
        "task": TASK_NAME,
        "n_samples": N_SAMPLES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "request_timeout": REQUEST_TIMEOUT,
        "enable_thinking": True,
        "score_is_artificial_analysis": False,
        "honesty": (
            "NVIDIA nvidia-simple-evals==26.3 gpqa_diamond_aa_v3 "
            "(AA methodology clone), not Artificial Analysis's private runner."
        ),
    }
    man.update(fields)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    return man


def parse_limit(raw: Optional[str]) -> Optional[int]:
    """Arm ``LIMIT`` env: empty means formal (no limit_samples)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    n = int(s)
    if n <= 0:
        raise ValueError(f"LIMIT must be a positive int, got {raw!r}")
    return n


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_req = sub.add_parser("require-task", help="gate nemo-evaluator ls stdout")
    p_req.add_argument("--ls-file", required=True)

    p_cfg = sub.add_parser("write-config", help="write --run_config YAML")
    p_cfg.add_argument("--out", required=True)
    p_cfg.add_argument("--url", required=True)
    p_cfg.add_argument("--model-id", required=True)
    p_cfg.add_argument("--output-dir", required=True)
    p_cfg.add_argument("--limit", default="")

    p_man = sub.add_parser("write-manifest")
    p_man.add_argument("--out", required=True)
    p_man.add_argument("--arm", default="")
    p_man.add_argument("--run-id", default="")
    p_man.add_argument("--url", default="")
    p_man.add_argument("--model-id", default="")
    p_man.add_argument("--limit", default="")

    p_argv = sub.add_parser("print-argv")
    p_argv.add_argument("--nemo-evaluator", required=True)
    p_argv.add_argument("--config", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "require-task":
        require_task(Path(a.ls_file).read_text(encoding="utf-8"))
        return 0
    if a.cmd == "write-config":
        write_run_config(
            a.out,
            url=a.url,
            model_id=a.model_id,
            output_dir=a.output_dir,
            limit_samples=parse_limit(a.limit),
        )
        return 0
    if a.cmd == "write-manifest":
        write_manifest(
            a.out,
            arm=a.arm,
            run_id=a.run_id,
            url=a.url,
            model_id=a.model_id,
            limit_samples=parse_limit(a.limit),
        )
        return 0
    if a.cmd == "print-argv":
        print(" ".join(build_run_eval_argv(a.nemo_evaluator, a.config)))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
