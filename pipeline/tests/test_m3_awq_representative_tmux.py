import os
from pathlib import Path
import subprocess


WRAPPER = (
    Path(__file__).resolve().parents[1]
    / "slurm"
    / "start_m3_awq_representative_tmux.sh"
)


def test_tmux_wrapper_dry_run_prints_durable_launch_and_monitoring(tmp_path):
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=WRAPPER.parents[2],
        env={
            **{k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
            "DRY_RUN": "1",
            "RUN_ID": "run-123",
            "SESSION_NAME": "m3-awq-run-123",
            "LOG_ROOT": str(tmp_path / "logs with spaces"),
            "RESULT_ROOT": str(tmp_path / "results with spaces"),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    assert "tmux new-session -d" in output
    assert "m3-awq-run-123" in output
    assert "run_m3_awq_representative_srun.sh" in output
    assert "controller.log" in output
    assert "tmux has-session" in output
    assert "tmux capture-pane" in output
    assert "tmux attach-session" in output
    assert "squeue" in output
    assert output.count("python -m pipeline.m3_awq_representative arm") == 6


def _fake_tmux(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state = tmp_path / "session-state"
    tmux = bin_dir / "tmux"
    tmux.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "state=${FAKE_TMUX_STATE:?}\n"
        "case $1 in\n"
        "  has-session) test -f \"$state\" ;;\n"
        "  new-session) touch \"$state\" ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n"
    )
    tmux.chmod(0o755)
    return bin_dir, state


def test_tmux_wrapper_creates_and_verifies_detached_session(tmp_path):
    bin_dir, state = _fake_tmux(tmp_path)
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=WRAPPER.parents[2],
        env={
            **{k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_TMUX_STATE": str(state),
            "RUN_ID": "verified-run",
            "SESSION_NAME": "verified-session",
            "LOG_ROOT": str(tmp_path / "logs"),
            "RESULT_ROOT": str(tmp_path / "results"),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert state.exists()
    assert "verified detached tmux session: verified-session" in completed.stdout
    controller = tmp_path / "results" / "controller.sh"
    assert controller.is_file()
    text = controller.read_text()
    assert "run_m3_awq_representative_srun.sh" in text
    assert "controller.rc" in text


def test_tmux_wrapper_rejects_existing_session(tmp_path):
    bin_dir, state = _fake_tmux(tmp_path)
    state.touch()
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=WRAPPER.parents[2],
        env={
            **{k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_TMUX_STATE": str(state),
            "RUN_ID": "duplicate-run",
            "SESSION_NAME": "duplicate-session",
            "LOG_ROOT": str(tmp_path / "logs"),
            "RESULT_ROOT": str(tmp_path / "results"),
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "already exists" in completed.stderr


def test_tmux_wrapper_rejects_stale_result_root(tmp_path):
    bin_dir, state = _fake_tmux(tmp_path)
    result_root = tmp_path / "results"
    result_root.mkdir()
    (result_root / "controller.rc").write_text("0\n")
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=WRAPPER.parents[2],
        env={
            **{k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_TMUX_STATE": str(state),
            "RUN_ID": "stale-run",
            "SESSION_NAME": "stale-session",
            "LOG_ROOT": str(tmp_path / "logs"),
            "RESULT_ROOT": str(result_root),
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "fresh RUN_ID" in completed.stderr


def test_tmux_wrapper_forbids_unsafe_detachment_methods():
    text = WRAPPER.read_text()
    assert "nohup" not in text
    assert "setsid" not in text
    assert "screen -" not in text
