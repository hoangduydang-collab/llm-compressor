"""CPU contract tests for the srun quality launcher."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT = Path("pipeline/slurm/run_m3_quality_eval_srun.sh")
MATRIX = Path("pipeline/configs/minimax_m3_quality_matrix.yaml")


def _run(*args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args], text=True, capture_output=True, check=False
    )


def test_smoke_dry_run_has_ray_preflight_and_three_parallel_arms(tmp_path):
    result = _run(
        "--profile", "smoke", "--matrix", str(MATRIX),
        "--run-root", str(tmp_path), "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("test_m3_ray_topology.sh") == 1
    assert result.stdout.count("test_m3_quality_eval_arm.sh") == 3
    assert "--nodes=2" in result.stdout
    assert "total_nodes=4" in result.stdout


def test_production_dry_run_requires_gate_and_has_six_arms(tmp_path):
    failed = _run(
        "--profile", "production", "--matrix", str(MATRIX),
        "--run-root", str(tmp_path), "--dry-run",
    )
    assert failed.returncode != 0
    gate = tmp_path / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": True}))
    result = _run(
        "--profile", "production", "--matrix", str(MATRIX),
        "--run-root", str(tmp_path), "--smoke-gate", str(gate), "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("test_m3_quality_eval_arm.sh") == 6
    assert "total_nodes=8" in result.stdout


def test_runner_scripts_are_valid_bash():
    for script in (
        SCRIPT,
        Path("pipeline/slurm/test_m3_quality_eval_arm.sh"),
        Path("pipeline/slurm/test_m3_ray_topology.sh"),
    ):
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True)
        assert result.returncode == 0, result.stderr.decode()


def test_bf16_arm_requires_ray_gate_before_eval():
    arm = Path("pipeline/slurm/test_m3_quality_eval_arm.sh").read_text()
    assert "test_m3_ray_topology.sh" in arm
    assert "ray_preflight/gate.json" in arm
    assert arm.index("ray_preflight/gate.json") < arm.index("pipeline.evalsuite.cli")
    assert "exec ray start" not in arm


def test_smoke_probe_runs_before_eval_and_failure_skips_eval():
    arm = Path("pipeline/slurm/test_m3_quality_eval_arm.sh").read_text()
    probe = 'python -m pipeline.m3_distributional_probe run'
    evaluate = '"${eval_cmd[@]}"'

    assert arm.index(probe) < arm.index(evaluate)
    assert 'if [[ "$PROFILE" == smoke && "$RUN_PROBE" == 1 ]]; then' in arm
    assert 'if ((rc == 0)); then\n  "${eval_cmd[@]}"' in arm
    assert 'if ((rc == 0 && RUN_PROBE == 1 && probe_ran == 0)); then' in arm
