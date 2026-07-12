import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "pipeline/slurm/start_m3_quality_smoke_tmux.sh"
CONTROLLER = ROOT / "pipeline/slurm/run_m3_quality_smoke_srun.sh"


def _base_env(tmp_path):
    return {
        **{k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
        "DRY_RUN": "1",
        "RUN_ID": "quality-123",
        "SESSION_NAME": "m3-quality-123",
        "RUN_ROOT": str(tmp_path / "run root"),
        "MATRIX": str(tmp_path / "matrix.yaml"),
        "REPAIRED_GPTQ": str(tmp_path / "gptq view"),
        "LOG_ROOT": str(tmp_path / "logs"),
    }


def test_quality_controller_dry_run_preserves_four_arm_plan(tmp_path):
    completed = subprocess.run(
        ["bash", str(CONTROLLER)], cwd=ROOT, env=_base_env(tmp_path),
        check=True, capture_output=True, text=True,
    )
    output = completed.stdout
    assert output.count("srun --exclusive") == 4
    assert "inhouse_gptq" in output
    assert "cyankiwi_awq" in output
    assert "test_m3_ray_topology.sh" in output
    assert "ray_preflight" in output
    assert "--stop-after-check" in output
    assert "m3_ray_placement_group" not in output
    assert "--model-label bf16" in output
    assert "timeout --signal=TERM --kill-after=60s 10m" in output


def test_quality_tmux_wrapper_dry_run_prints_monitoring(tmp_path):
    completed = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=_base_env(tmp_path),
        check=True, capture_output=True, text=True,
    )
    output = completed.stdout
    assert "tmux new-session -d" in output
    assert "tmux has-session" in output
    assert "tmux capture-pane" in output
    assert "tmux attach-session" in output
    assert "controller.rc" in output
    assert "run_m3_quality_smoke_srun.sh" in output


def _fake_tmux(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state = tmp_path / "session-state"
    tmux = bin_dir / "tmux"
    tmux.write_text(
        "#!/usr/bin/env bash\nset -eu\nstate=${FAKE_TMUX_STATE:?}\n"
        "case $1 in\n"
        " has-session) test -f \"$state\" ;;;;\n"
        " new-session) touch \"$state\" ;;;;\n"
        " *) exit 2 ;;;;\nesac\n".replace(";;;;", ";;")
    )
    tmux.chmod(0o755)
    return bin_dir, state


def test_quality_tmux_wrapper_creates_verified_session(tmp_path):
    bin_dir, state = _fake_tmux(tmp_path)
    env = _base_env(tmp_path)
    env.update({"DRY_RUN": "0", "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_TMUX_STATE": str(state)})
    completed = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env,
        check=True, capture_output=True, text=True,
    )
    assert "verified detached tmux session" in completed.stdout
    text = (tmp_path / "run root" / "controller.sh").read_text()
    assert "run_m3_quality_smoke_srun.sh" in text
    assert "controller.rc" in text


def test_quality_tmux_wrapper_rejects_existing_session(tmp_path):
    bin_dir, state = _fake_tmux(tmp_path)
    state.touch()
    env = _base_env(tmp_path)
    env.update({"DRY_RUN": "0", "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_TMUX_STATE": str(state)})
    completed = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "already exists" in completed.stderr


def test_quality_tmux_wrapper_rejects_stale_controller_evidence(tmp_path):
    bin_dir, state = _fake_tmux(tmp_path)
    run_root = tmp_path / "run root"
    run_root.mkdir()
    (run_root / "controller.rc").write_text("0\n")
    env = _base_env(tmp_path)
    env.update({"DRY_RUN": "0", "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_TMUX_STATE": str(state)})
    completed = subprocess.run(
        ["bash", str(WRAPPER)], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "fresh RUN_ID/RUN_ROOT" in completed.stderr


def test_quality_tmux_wrapper_forbids_unsafe_detachment():
    text = WRAPPER.read_text()
    assert "nohup" not in text
    assert "setsid" not in text
    assert "screen -" not in text
