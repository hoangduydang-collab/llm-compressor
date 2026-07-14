"""Focused CPU contracts for the capped MiniMax-M3 Slurm array launcher."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


SUBMIT = Path("pipeline/slurm/submit_m3_quality_eval_array.sh")
ARRAY_ARM = Path("pipeline/slurm/run_m3_quality_eval_array_arm.sh")
MATRIX = Path(
    "pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml"
)


def _bash_path(path: Path) -> str:
    value = path.resolve().as_posix()
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/{value[0].lower()}{value[2:]}"
    return value


def _workspace_tmp(tmp_path: Path, request) -> Path:
    root = Path(".pytest-m3-array") / tmp_path.name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)

    def cleanup():
        shutil.rmtree(root, ignore_errors=True)
        try:
            root.parent.rmdir()
        except OSError:
            pass

    request.addfinalizer(cleanup)
    return root


def test_submission_dry_run_maps_twelve_arms_with_six_node_cap(
    tmp_path, request
):
    root = _workspace_tmp(tmp_path, request)
    (root / "preflight").mkdir()
    gate = root / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": True}), encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(SUBMIT),
            "--matrix",
            str(MATRIX),
            "--run-root",
            _bash_path(root),
            "--smoke-gate",
            _bash_path(gate),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("index=") == 12
    for fragment in (
        "--array=0-11%6",
        "--nodes=1",
        "--ntasks=1",
        "--gpus-per-node=8",
        "--exclusive",
        "--time=08:00:00",
    ):
        assert fragment in result.stdout
    assert not (root / "array_job_id.txt").exists()


def test_array_index_forwards_exact_launch_plan_arm(tmp_path, request):
    root = _workspace_tmp(tmp_path, request)
    preflight = root / "preflight"
    preflight.mkdir()
    (preflight / "resolved_tasks.json").write_text(
        json.dumps({"aliases": {"ifeval": "leaderboard_ifeval"}}),
        encoding="utf-8",
    )
    plan = root / "production_launch_plan.json"
    plan.write_text(
        json.dumps(
            {
                "arms": [
                    {
                        "model_label": "awq",
                        "model_path": "/models/awq",
                        "shard": "gpqa",
                        "nodes": 1,
                        "gpus_per_node": 8,
                        "tensor_parallel_size": 8,
                        "pipeline_parallel_size": 1,
                        "distributed_executor_backend": "mp",
                        "tasks": ["gpqa_diamond"],
                        "distributional_probe": False,
                        "probe_tokens": 0,
                    },
                    {
                        "model_label": "gptq",
                        "model_path": "/models/gptq",
                        "shard": "ifeval",
                        "nodes": 1,
                        "gpus_per_node": 8,
                        "tensor_parallel_size": 8,
                        "pipeline_parallel_size": 1,
                        "distributed_executor_backend": "mp",
                        "tasks": ["ifeval"],
                        "distributional_probe": False,
                        "probe_tokens": 0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    recorder = root / "record-arm.sh"
    recorder.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" >\"$RECORD\"\n",
        encoding="utf-8",
    )
    recorder.chmod(0o755)
    record = root / "args.txt"
    env = os.environ.copy()
    env.update(
        {
            "SLURM_ARRAY_TASK_ID": "1",
            "M3_QUALITY_ARM_RUNNER": _bash_path(recorder),
            "RECORD": _bash_path(record),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(ARRAY_ARM),
            "--plan",
            _bash_path(plan),
            "--run-root",
            _bash_path(root),
            "--matrix",
            str(MATRIX),
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = record.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--model-label") + 1] == "gptq"
    assert args[args.index("--model") + 1] == "/models/gptq"
    assert args[args.index("--shard") + 1] == "ifeval"
    assert args[args.index("--tasks") + 1] == "leaderboard_ifeval"
    assert args[args.index("--tensor-parallel-size") + 1] == "8"
    assert args[args.index("--run-probe") + 1] == "0"


def test_array_scripts_are_valid_bash():
    for script in (SUBMIT, ARRAY_ARM):
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True)
        assert result.returncode == 0, result.stderr.decode()
