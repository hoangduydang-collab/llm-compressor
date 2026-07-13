import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "pipeline/slurm/run_m3_safe_diagnostic_full_srun.sh"
TMUX = ROOT / "pipeline/slurm/start_m3_safe_diagnostic_full_tmux.sh"
SAFE_WORKER = ROOT / "pipeline/slurm/run_m3_safe_full_lane.sh"


def _dry_run(**extra):
    env = {**os.environ, "DRY_RUN": "1", "RUN_ID": "pytest-safe-diagnostic", **extra}
    return subprocess.check_output(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True)


def test_dry_run_launches_safe_and_diagnostic_lanes_on_exclusive_nodes():
    output = _dry_run()
    assert output.count("srun --exclusive --nodes=1 --ntasks=1") == 6
    assert output.count("python -m pipeline.run") == 3
    assert output.count("python -m pipeline.m3_guarded_full arm") == 2
    for lane in (
        "safe-offsetfix",
        "safe-nosmooth",
        "safe-quant_only",
        "diag-heavy-offsetfix",
        "diag-light-offsetfix",
    ):
        assert lane in output
    assert "--diagnostic-mode heavy" in output
    assert "--diagnostic-mode light" in output
    assert "sbatch" not in output


def test_safe_lane_commands_use_only_production_runner_and_static_checker():
    output = _dry_run()
    safe_lines = [line for line in output.splitlines() if "lane=safe-" in line]
    assert len(safe_lines) == 3
    for line in safe_lines:
        assert "pipeline.run" in line
        assert "--stage quantize" in line
        assert "m3_guarded_full" not in line
    runner = RUNNER.read_text()
    worker = SAFE_WORKER.read_text()
    assert "python -m pipeline.run" in worker
    assert "m3_guarded_full" not in worker
    assert "enable_quantization" not in worker
    assert "register_forward_hook" not in worker
    assert "pipeline.verify_quant_checkpoint" in worker
    assert "--check-tensors" in worker
    assert "refusing non-fresh lane root" in worker
    assert "expected exactly one checkpoint" in worker
    assert ".rc.tmp" in runner


def test_one_lane_smoke_keeps_tiny_overrides_and_one_full_lane_node():
    output = _dry_run(
        LANE_FILTER="safe-quant_only",
        SAFE_NUM_SAMPLES="2",
        SAFE_MAX_SEQ_LENGTH="128",
    )
    assert output.count("srun --exclusive --nodes=1 --ntasks=1") == 2
    assert "--num-samples 2" in output
    assert "--max-seq-length 128" in output
    assert "lane=safe-quant_only" in output
    assert "lane=safe-offsetfix" not in output


def test_tmux_launcher_owns_controller_and_both_refuse_nested_slurm():
    output = subprocess.check_output(
        ["bash", str(TMUX)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", "RUN_ID": "pytest-safe-diagnostic"},
        text=True,
    )
    assert "run_m3_safe_diagnostic_full_srun.sh" in output
    assert output.count("srun --exclusive --nodes=1 --ntasks=1") == 6
    runner = RUNNER.read_text()
    tmux = TMUX.read_text()
    assert "SLURM_JOB_ID" in runner and "SLURM_JOB_ID" in tmux
    assert "tmux new-session -d" in tmux
    assert "tmux has-session" in tmux
