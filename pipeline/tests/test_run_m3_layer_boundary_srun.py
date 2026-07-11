"""CPU dry-run test for the eleven-node MiniMax-M3 boundary matrix."""

import os
import subprocess
from pathlib import Path

from pipeline.m3_layer_boundary_diagnostics import EXPECTED_ARMS

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "pipeline/slurm/run_m3_layer_boundary_srun.sh"


def test_boundary_launcher_emits_all_exclusive_srun_arms_concurrently():
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env={**os.environ, "DRY_RUN": "1", "MATRIX_ID": "boundary-fixed"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    commands = [
        line for line in completed.stdout.splitlines() if line.startswith("srun ")
    ]
    assert len(commands) == len(EXPECTED_ARMS) == 11
    for arm in EXPECTED_ARMS:
        matching = [line for line in commands if f"ARM={arm}" in line]
        assert len(matching) == 1
        assert "--exclusive" in matching[0]
        assert "--nodes=1" in matching[0]
        assert "--gres=gpu:8" in matching[0]
    assert "sbatch" not in completed.stdout.lower()
