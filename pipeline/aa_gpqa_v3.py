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

# AA reasoning / GLM (max). Not NVIDIA's greedy task defaults (16384).
#
# SAMPLING PROVENANCE (read before changing these two numbers).
# AA publishes a default sampling config, temperature 0.6 / top_p 1.0, and
# overrides it with the model creator's recommended config whenever the lab
# publishes one. Z.ai, the creator of GLM-5.3, recommends temperature 1.0 /
# top_p 0.95, so for this model the lab-override branch applies and 1.0/0.95
# is the AA-faithful setting. That is what these constants are.
#
# The Sep-11 formal run (ours 81.82% / phala 79.39%,
# `results-formal-198-c8`) ran on AA's *generic* 0.6/1.0 default, i.e. the
# wrong branch of AA's own rule for this model. Those scores are therefore
# on a different sampling contract than anything produced here: do not pool
# them, and do not report a delta between the two as a model difference.
# Both values are recorded in the manifest (`write_manifest`), and both are
# overridable per run with --temperature / --top-p.
#
# max_new_tokens: Z.ai discloses 128K max output for GLM-5.3; AA's
# reasoning-model rule is "maximum output tokens allowed, as disclosed by
# model creators." Serve --context-length is 164800 (measured FP8 KV pool on
# this 8xH100 arm). Must exceed this plus the prompt (64k ctx + 64k
# max_tokens 400'd).
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 131072
REQUEST_TIMEOUT = 3600
# AA scores GPQA Diamond as pass@1 averaged over 5 repeats of all 198 items.
# Lower it ONLY for cost canaries, and read the result as a length/throughput
# probe, not a score: at n_samples=1 the per-arm stderr is ~2.7 pp, wider than
# any gap we are trying to see. Prefer reducing repeats over `limit_samples`,
# which takes the FIRST n Diamond rows — a subject-ordered, biased subset.
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
    max_new_tokens: Optional[int] = None,
    request_timeout: Optional[int] = None,
    parallelism: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n_samples: Optional[int] = None,
) -> dict[str, Any]:
    """Write a nemo-evaluator ``--run_config`` YAML. Returns the document."""
    params: dict[str, Any] = {
        "temperature": TEMPERATURE if temperature is None else temperature,
        "top_p": TOP_P if top_p is None else top_p,
        "max_new_tokens": MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens,
        "request_timeout": REQUEST_TIMEOUT if request_timeout is None else request_timeout,
        "limit_samples": limit_samples,
        "extra": {"n_samples": N_SAMPLES if n_samples is None else n_samples},
    }
    if parallelism is not None:
        params["parallelism"] = parallelism
    cfg: dict[str, Any] = {
        "config": {
            "type": TASK_NAME,
            "output_dir": output_dir,
            "params": params,
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
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n_samples: Optional[int] = None,
    **fields: Any,
) -> dict[str, Any]:
    temp = TEMPERATURE if temperature is None else temperature
    tp = TOP_P if top_p is None else top_p
    n_s = N_SAMPLES if n_samples is None else n_samples
    man: dict[str, Any] = {
        "package": PACKAGE_PIN,
        "task": TASK_NAME,
        "n_samples": n_s,
        "is_formal_aa_protocol": n_s == N_SAMPLES and fields.get("limit_samples") is None,
        "temperature": temp,
        "top_p": tp,
        "max_new_tokens": MAX_NEW_TOKENS,
        "request_timeout": REQUEST_TIMEOUT,
        "enable_thinking": True,
        "score_is_artificial_analysis": False,
        "sampling_provenance": (
            "Z.ai (GLM-5.3 creator) recommended config, the lab-override "
            "branch of AA's sampling rule"
            if (temp, tp) == (1.0, 0.95)
            else "AA published default (no lab override applied)"
            if (temp, tp) == (0.6, 1.0)
            else "custom: neither AA's default nor Z.ai's recommended config"
        ),
        "comparable_to_sep11_formal_198_c8": (temp, tp) == (0.6, 1.0),
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
    p_cfg.add_argument("--max-new-tokens", type=int, default=None)
    p_cfg.add_argument("--request-timeout", type=int, default=None)
    p_cfg.add_argument("--parallelism", type=int, default=None)
    p_cfg.add_argument("--temperature", type=float, default=None)
    p_cfg.add_argument("--top-p", type=float, default=None)
    p_cfg.add_argument("--n-samples", type=int, default=None)

    p_man = sub.add_parser("write-manifest")
    p_man.add_argument("--out", required=True)
    p_man.add_argument("--arm", default="")
    p_man.add_argument("--run-id", default="")
    p_man.add_argument("--url", default="")
    p_man.add_argument("--model-id", default="")
    p_man.add_argument("--limit", default="")
    p_man.add_argument("--temperature", type=float, default=None)
    p_man.add_argument("--top-p", type=float, default=None)
    p_man.add_argument("--n-samples", type=int, default=None)

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
            max_new_tokens=a.max_new_tokens,
            request_timeout=a.request_timeout,
            parallelism=a.parallelism,
            temperature=a.temperature,
            top_p=a.top_p,
            n_samples=a.n_samples,
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
            temperature=a.temperature,
            top_p=a.top_p,
            n_samples=a.n_samples,
        )
        return 0
    if a.cmd == "print-argv":
        print(" ".join(build_run_eval_argv(a.nemo_evaluator, a.config)))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
