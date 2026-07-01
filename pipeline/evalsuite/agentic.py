"""Agentic evaluation via tau2-bench (reuses benchmarks-repo launcher)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from pipeline.config import AgenticConfig, PipelineConfig


def _agentic_ready(ag: AgenticConfig) -> tuple[bool, str]:
    if not ag.enabled:
        return False, "agentic.enabled is false"
    if ag.harness != "tau2":
        return False, f"unsupported harness {ag.harness!r}"
    if not ag.tau2_dir:
        return False, "agentic.tau2_dir not set"
    tau2_bin = Path(ag.tau2_dir) / ".venv" / "bin" / "tau2"
    if not tau2_bin.exists():
        return False, f"tau2 not found at {tau2_bin}"
    if not ag.user_base or not ag.user_model:
        return False, "agentic user simulator not configured (user_base, user_model)"
    if not ag.user_key_file or not Path(ag.user_key_file).is_file():
        return False, f"agentic.user_key_file not readable: {ag.user_key_file}"
    return True, "ok"


def _default_calibration_script() -> Path | None:
    """Locate benchmarks-repo run_calibration.sh relative to workspace."""
    import os

    here = Path(__file__).resolve()
    work_root = Path(os.environ.get("WORK_ROOT", "/mnt/nfs/hoangduy"))
    candidates = [
        here.parents[2]
        / ".."
        / "benchmarks"
        / "llm-perf-benchmarks"
        / "performance"
        / "calibration"
        / "run_calibration.sh",
        here.parents[3]
        / "benchmarks"
        / "llm-perf-benchmarks"
        / "performance"
        / "calibration"
        / "run_calibration.sh",
        here.parents[3]
        / "projects"
        / "benchmarks"
        / "llm-perf-benchmarks"
        / "performance"
        / "calibration"
        / "run_calibration.sh",
        work_root
        / "projects"
        / "benchmarks"
        / "llm-perf-benchmarks"
        / "performance"
        / "calibration"
        / "run_calibration.sh",
    ]
    for p in candidates:
        p = p.resolve()
        if p.is_file():
            return p
    return None


def _resolve_calibration_script(ag: AgenticConfig) -> Path | None:
    if ag.calibration_script:
        script = Path(ag.calibration_script)
        return script if script.is_file() else None
    return _default_calibration_script()


def _run_via_shell_script(ag: AgenticConfig, env_extra: dict[str, str], script: Path) -> Path:
    script = Path(script)
    env = os.environ.copy()
    env.update(env_extra)
    print(f"[evalsuite] agentic: running {script}")
    subprocess.run(["bash", str(script)], check=True, env=env)
    tau2_dir = Path(ag.tau2_dir)
    return tau2_dir / "data" / "simulations" / ag.save_to


def _run_tau2_direct(ag: AgenticConfig, env_extra: dict[str, str]) -> Path:
    """Invoke tau2 CLI directly (supports num_trials > 1)."""
    tau2_dir = Path(ag.tau2_dir)
    tau2_bin = tau2_dir / ".venv" / "bin" / "tau2"

    key = Path(ag.user_key_file).read_text(encoding="utf-8").strip()
    thinking = ag.thinking.lower() == "on"
    agent_temp = 0.6 if thinking else 0.0

    agent_args = json.dumps(
        {
            "api_base": env_extra["AGENT_BASE"],
            "api_key": "EMPTY",
            "temperature": agent_temp,
            "max_tokens": 32768,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}},
        }
    )
    user_args = json.dumps(
        {
            "api_base": ag.user_base,
            "api_key": key,
            "temperature": 0.0,
        }
    )

    cmd = [
        str(tau2_bin),
        "run",
        "--domain",
        ag.domain,
        "--task-split-name",
        ag.split,
        "--agent-llm",
        f"openai/{env_extra['AGENT_MODEL']}",
        "--agent-llm-args",
        agent_args,
        "--user-llm",
        f"openai/{ag.user_model}",
        "--user-llm-args",
        user_args,
        "--num-trials",
        str(ag.num_trials),
        "--max-concurrency",
        str(ag.max_conc),
        "--max-steps",
        str(ag.max_steps),
        "--timeout",
        str(ag.timeout),
        "--seed",
        str(ag.seed),
        "--save-to",
        ag.save_to,
    ]
    if ag.num_tasks is not None:
        cmd.extend(["--num-tasks", str(ag.num_tasks)])

    print(f"[evalsuite] agentic: {' '.join(cmd[:8])} ...")
    subprocess.run(cmd, check=True, cwd=str(tau2_dir))
    return tau2_dir / "data" / "simulations" / ag.save_to


def _iter_task_records(obj) -> list[dict]:
    """Schema-tolerant walk for per-task reward records in tau2 output."""
    found: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            reward = node.get("reward")
            task_id = node.get("task_id") or node.get("id") or node.get("task")
            if reward is not None and task_id is not None:
                found.append(
                    {
                        "task_id": str(task_id),
                        "reward": float(reward),
                        "domain": node.get("domain"),
                        "trial": node.get("trial"),
                    }
                )
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return found


def _parse_tau2_results(sim_dir: Path, threshold: float) -> tuple[list[dict], dict]:
    results_path = sim_dir / "results.json"
    if not results_path.exists():
        for alt in sim_dir.glob("**/*.json"):
            if alt.name in ("results.json", "simulations.json"):
                results_path = alt
                break

    if not results_path.exists():
        raise FileNotFoundError(f"no tau2 results.json under {sim_dir}")

    data = json.loads(results_path.read_text(encoding="utf-8"))
    records = _iter_task_records(data)

    # Deduplicate: keep mean reward per task_id if multiple trials.
    by_task: dict[str, list[float]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r["reward"])

    rows: list[dict] = []
    for task_id, rewards in sorted(by_task.items()):
        mean_reward = sum(rewards) / len(rewards)
        rows.append(
            {
                "task_id": task_id,
                "reward": mean_reward,
                "success": int(mean_reward >= threshold),
                "correct": int(mean_reward >= threshold),
                "n_trials": len(rewards),
            }
        )

    n = len(rows)
    success_rate = sum(r["success"] for r in rows) / n if n else 0.0
    aggregate = {
        "n_tasks": n,
        "mean_reward": sum(r["reward"] for r in rows) / n if n else None,
        "success_rate": success_rate,
        "reward_threshold": threshold,
        "sim_dir": str(sim_dir),
    }
    return rows, aggregate


def run_agentic_eval(
    cfg: PipelineConfig,
    model_path: str | Path,
    out_dir: str | Path,
    *,
    agent_base: str | None = None,
    agent_model: str | None = None,
) -> dict | None:
    """Run tau2 agentic eval; returns None if prerequisites missing."""
    ag = cfg.agentic
    ok, reason = _agentic_ready(ag)
    if not ok:
        print(f"[evalsuite] agentic skipped: {reason}")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_base = agent_base or ag.agent_base or f"http://localhost:8000/v1"
    agent_model = agent_model or ag.agent_model or cfg.model.id

    env_extra = {
        "TAU2_DIR": str(ag.tau2_dir),
        "AGENT_BASE": agent_base,
        "AGENT_MODEL": agent_model,
        "USER_BASE": ag.user_base,
        "USER_MODEL": ag.user_model,
        "USER_KEY_FILE": ag.user_key_file,
        "DOMAIN": ag.domain,
        "SPLIT": ag.split,
        "MAX_CONC": str(ag.max_conc),
        "THINKING": ag.thinking,
        "SAVE_TO": ag.save_to,
        "SEED": str(ag.seed),
    }
    if ag.num_tasks is not None:
        env_extra["NUM_TASKS"] = str(ag.num_tasks)

    script = _resolve_calibration_script(ag)
    if ag.num_trials == 1 and script is not None:
        sim_dir = _run_via_shell_script(ag, env_extra, script)
    else:
        if ag.num_trials == 1 and script is None:
            print("[evalsuite] agentic: calibration script not found; using tau2 CLI directly")
        sim_dir = _run_tau2_direct(ag, env_extra)

    threshold = cfg.compare.agentic_reward_threshold
    rows, aggregate = _parse_tau2_results(sim_dir, threshold)

    samples_path = out_dir / "agentic_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    with (out_dir / "agentic_aggregate.json").open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2)

    print(
        f"[evalsuite] agentic done: {aggregate['n_tasks']} tasks, "
        f"success_rate={aggregate['success_rate']:.2%}"
    )
    return {"samples_path": samples_path, "aggregate": aggregate, "rows": rows}
