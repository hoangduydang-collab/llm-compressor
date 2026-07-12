from pathlib import Path
import os
import subprocess


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "slurm"
    / "run_m3_awq_representative_srun.sh"
)


def launcher_text() -> str:
    return LAUNCHER.read_text()


def test_launcher_declares_the_six_expected_arms():
    text = launcher_text()
    for variant in ("offsetfix", "nosmooth"):
        for layer in (8, 31, 59):
            assert f"{variant}-layer{layer}" in text


def test_launcher_exposes_required_overrides_and_uses_only_srun():
    text = launcher_text()
    for variable in (
        "DRY_RUN",
        "TIME_LIMIT",
        "LOG_ROOT",
        "RESULT_ROOT",
        "ENV_FILE",
        "VENV_ACTIVATE",
        "SRUN_ARGS",
        "RUN_ID",
    ):
        assert f"{variable}=" in text
    assert "srun" in text
    assert "sbatch" not in text
    assert "--exclusive" in text
    assert "--nodes=1" in text
    assert "--ntasks=1" in text
    assert "--gres=gpu:1" in text


def test_launcher_runs_concurrently_and_records_every_return_code():
    text = launcher_text()
    assert "pids+=(" in text
    assert '"${command[@]}"' in text
    assert " &" in text
    assert 'wait "${pids[$index]}"' in text
    assert 'rc_file="$output_dir/rc"' in text
    assert ">\"$LOG_ROOT/" in text
    assert "trap " in text
    assert "143" in text
    assert "129" in text
    assert "SIGKILL" in text


def test_launcher_aggregates_only_after_waiting_for_all_arms():
    text = launcher_text()
    wait_position = text.rindex("wait ")
    aggregate_position = text.rindex("pipeline.m3_awq_representative aggregate")
    assert aggregate_position > wait_position
    assert "matrix.json" in text
    assert "report.md" in text


def test_launcher_dry_run_executes_six_unique_ordered_commands(tmp_path):
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=LAUNCHER.parents[2],
        env={
            **os.environ,
            "DRY_RUN": "1",
            "RUN_ID": "test-run",
            "LOG_ROOT": str(tmp_path / "logs"),
            "RESULT_ROOT": str(tmp_path / "results"),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    assert len(lines) == 6
    expected = [
        ("offsetfix", 8),
        ("offsetfix", 31),
        ("offsetfix", 59),
        ("nosmooth", 8),
        ("nosmooth", 31),
        ("nosmooth", 59),
    ]
    for line, (variant, layer) in zip(lines, expected, strict=True):
        arm = f"{variant}-layer{layer}"
        assert line.startswith("srun ")
        assert f"--layer {layer} --variant {variant}" in line
        assert f"{tmp_path}/results/{arm}" in line
        assert f"{tmp_path}/logs/{arm}.log" in line
    assert len(set(lines)) == 6


def test_launcher_defaults_to_run_specific_roots():
    text = launcher_text()
    assert 'RUN_ID="${RUN_ID:-' in text
    assert '/$RUN_ID}"' in text
    assert "read -r -a EXTRA_SRUN_ARGS" in text
