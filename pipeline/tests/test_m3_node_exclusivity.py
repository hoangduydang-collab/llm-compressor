import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]

def test_quality_tmux_wrapper_rejects_nested_slurm_allocation(tmp_path):
    env = {
        **os.environ,
        "SLURM_JOB_ID": "nested-123",
        "DRY_RUN": "0",
        "RUN_ID": "nested-test",
        "SESSION_NAME": "nested-test",
        "RUN_ROOT": str(tmp_path / "run"),
        "RESULT_ROOT": str(tmp_path / "results"),
        "LOG_ROOT": str(tmp_path / "logs"),
        "MATRIX": str(tmp_path / "matrix.yaml"),
        "REPAIRED_GPTQ": str(tmp_path / "gptq"),
    }
    completed = subprocess.run(
        ["bash", str(ROOT / "pipeline/slurm/start_m3_quality_smoke_tmux.sh")], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "outside any Slurm allocation" in completed.stderr


def test_representative_tmux_wrapper_rejects_nested_slurm_allocation(tmp_path):
    env = {
        **os.environ,
        "SLURM_JOB_ID": "nested-123",
        "DRY_RUN": "0",
        "RUN_ID": "nested-test",
        "SESSION_NAME": "nested-test",
        "RUN_ROOT": str(tmp_path / "run"),
        "RESULT_ROOT": str(tmp_path / "results"),
        "LOG_ROOT": str(tmp_path / "logs"),
        "MATRIX": str(tmp_path / "matrix.yaml"),
        "REPAIRED_GPTQ": str(tmp_path / "gptq"),
    }
    completed = subprocess.run(
        ["bash", str(ROOT / "pipeline/slurm/start_m3_awq_representative_tmux.sh")], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "outside any Slurm allocation" in completed.stderr


def test_direct_controllers_reject_nested_slurm_allocation(tmp_path):
    for relative in (
        "pipeline/slurm/run_m3_quality_smoke_srun.sh",
        "pipeline/slurm/run_m3_awq_representative_srun.sh",
    ):
        env = {
            **os.environ,
            "DRY_RUN": "0",
            "SLURM_JOB_ID": "nested-456",
            "RUN_ROOT": str(tmp_path / "run"),
            "RESULT_ROOT": str(tmp_path / "results"),
            "LOG_ROOT": str(tmp_path / "logs"),
            "MATRIX": str(tmp_path / "matrix.yaml"),
            "REPAIRED_GPTQ": str(tmp_path / "gptq"),
        }
        completed = subprocess.run(
            ["bash", str(ROOT / relative)], cwd=ROOT, env=env,
            capture_output=True, text=True,
        )
        assert completed.returncode != 0, relative
        assert "refusing nested srun" in completed.stderr
