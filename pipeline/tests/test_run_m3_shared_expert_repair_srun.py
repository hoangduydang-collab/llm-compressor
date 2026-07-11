"""CPU dry-run test for the parallel shared-expert repair matrix."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pipeline.m3_shared_expert_repair import EXPECTED_ARMS

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "pipeline/slurm/run_m3_shared_expert_repair_srun.sh"


def test_repair_srun_dry_run_emits_three_exclusive_node_commands():
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env={**os.environ, "DRY_RUN": "1", "MATRIX_ID": "repair-fixed"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    commands = [
        line for line in completed.stdout.splitlines() if line.startswith("srun ")
    ]
    assert len(commands) == 3
    for arm in EXPECTED_ARMS:
        matching = [line for line in commands if f"ARM={arm}" in line]
        assert len(matching) == 1
        command = matching[0]
        assert "MATRIX_ID=repair-fixed" in command
        assert "--exclusive" in command
        assert "--nodes=1" in command
        assert "--ntasks=1" in command
        assert "--gres=gpu:8" in command
        assert f"repair-fixed-{arm}.log" in command
    assert "sbatch" not in completed.stdout.lower()
