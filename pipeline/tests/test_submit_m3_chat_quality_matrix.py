"""CPU dry-run test for parallel MiniMax-M3 chat matrix submission."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pipeline.m3_chat_quality import EXPECTED_ARMS

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMIT = REPO_ROOT / "pipeline/slurm/submit_m3_chat_quality_matrix.sh"


def test_submit_dry_run_emits_four_independent_node_commands():
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", "MATRIX_ID": "matrix-fixed"})

    completed = subprocess.run(
        ["bash", str(SUBMIT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    commands = [line for line in completed.stdout.splitlines() if line.startswith("sbatch ")]
    assert len(commands) == 4
    for arm in EXPECTED_ARMS:
        matching = [line for line in commands if f"ARM={arm}" in line]
        assert len(matching) == 1
        assert "MATRIX_ID=matrix-fixed" in matching[0]
        assert "--nodes=1" in matching[0]
        assert "--gres=gpu:8" in matching[0]
        assert f"matrix-fixed-{arm}.out" in matching[0]
        assert f"matrix-fixed-{arm}.err" in matching[0]
