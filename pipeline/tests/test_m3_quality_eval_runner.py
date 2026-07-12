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


def test_smoke_dry_run_has_four_parallel_arms_and_five_nodes(tmp_path):
    result = _run(
        "--profile", "smoke", "--matrix", str(MATRIX),
        "--run-root", str(tmp_path), "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("srun --exclusive") == 4
    assert "--nodes=2" in result.stdout
    assert "total_nodes=5" in result.stdout


def test_production_dry_run_requires_gate_and_has_eight_arms(tmp_path):
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
    assert result.stdout.count("srun --exclusive") == 8
    assert "total_nodes=10" in result.stdout


def test_runner_scripts_are_valid_bash():
    for script in (SCRIPT, Path("pipeline/slurm/test_m3_quality_eval_arm.sh")):
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True)
        assert result.returncode == 0, result.stderr.decode()
