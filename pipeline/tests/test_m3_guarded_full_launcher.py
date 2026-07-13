import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "pipeline/slurm/run_m3_guarded_full_srun.sh"
TMUX = ROOT / "pipeline/slurm/start_m3_guarded_full_tmux.sh"


def test_dry_run_smokes_before_three_exclusive_full_nodes():
    env = {**os.environ, "DRY_RUN": "1", "RUN_ID": "pytest-guarded"}
    output = subprocess.check_output(
        ["bash", str(RUNNER)], cwd=ROOT, env=env, text=True
    )
    assert output.count("srun --exclusive --nodes=1") == 4
    assert output.index("pipeline.m3_trace_diagnostic") < output.index(
        "--variant offsetfix"
    )
    for variant in ("offsetfix", "nosmooth", "quant_only"):
        assert f"--variant {variant}" in output
    assert "sbatch" not in output


def test_tmux_dry_run_delegates_to_guarded_runner():
    env = {**os.environ, "DRY_RUN": "1", "RUN_ID": "pytest-guarded"}
    output = subprocess.check_output(["bash", str(TMUX)], cwd=ROOT, env=env, text=True)
    assert "run_m3_guarded_full_srun.sh" in output
    assert output.count("srun --exclusive --nodes=1") == 4


def test_real_launchers_reject_nested_slurm_and_use_tmux_ownership():
    runner = RUNNER.read_text()
    tmux = TMUX.read_text()
    assert "SLURM_JOB_ID" in runner and "SLURM_JOB_ID" in tmux
    assert "tmux new-session -d" in tmux
    assert "tmux has-session" in tmux
