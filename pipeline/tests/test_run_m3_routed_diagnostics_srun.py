"""CPU dry-run tests for parallel routed diagnostics through srun."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pipeline.m3_routed_diagnostics import EXPECTED_ARMS

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "pipeline/slurm/run_m3_routed_diagnostics_srun.sh"


def test_srun_dry_run_emits_three_exclusive_node_commands():
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", "MATRIX_ID": "diag-fixed"})

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    commands = [line for line in completed.stdout.splitlines() if line.startswith("srun ")]
    assert len(commands) == 3
    for arm in EXPECTED_ARMS:
        matching = [line for line in commands if f"ARM={arm}" in line]
        assert len(matching) == 1
        assert "MATRIX_ID=diag-fixed" in matching[0]
        assert "--exclusive" in matching[0]
        assert "--nodes=1" in matching[0]
        assert "--ntasks=1" in matching[0]
        assert "--gres=gpu:8" in matching[0]
        assert f"diag-fixed-{arm}.log" in matching[0]
    assert "sbatch" not in completed.stdout
