"""Dry-run test for the parallel MiniMax-M3 AWQ/GPTQ matrix."""

import os
import subprocess
from pathlib import Path

from pipeline.m3_awq_gptq_repair import EXPECTED_ARMS

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "pipeline/slurm/run_m3_awq_gptq_repair_srun.sh"


def test_launcher_emits_eight_serve_arms_and_one_audit():
    completed = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO_ROOT,
        env={**os.environ, "DRY_RUN": "1", "MATRIX_ID": "repair-test"},
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    commands = [line for line in completed.stdout.splitlines() if line.startswith("srun ")]
    assert len(commands) == len(EXPECTED_ARMS) + 1 == 12
    for arm in EXPECTED_ARMS:
        assert sum(f"ARM={arm}" in command for command in commands) == 1
    assert "checkpoint_scale_audit.json" in completed.stdout
    assert "sbatch" not in completed.stdout.lower()
