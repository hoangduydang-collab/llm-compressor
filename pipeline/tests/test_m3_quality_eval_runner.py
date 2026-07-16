"""CPU contract tests for the srun quality launcher."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from pipeline.m3_quality_eval import build_launch_plan, load_matrix

SCRIPT = Path("pipeline/slurm/run_m3_quality_eval_srun.sh")
MATRIX = Path("pipeline/configs/minimax_m3_quality_matrix.yaml")
GROUPED_MATRIX = Path(
    "pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml"
)
REASONING_R4_MATRIX = Path(
    "pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml"
)
BF16_REASONING_R4_MATRIX = Path(
    "pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml"
)


def _run(*args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args], text=True, capture_output=True, check=False
    )


def _bash_path(path: Path) -> str:
    value = path.resolve().as_posix()
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/{value[0].lower()}{value[2:]}"
    return value


def _workspace_tmp(tmp_path: Path, request) -> Path:
    work_dir = Path(".pytest-m3-quality") / tmp_path.name
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)

    def cleanup():
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            work_dir.parent.rmdir()
        except OSError:
            pass

    request.addfinalizer(cleanup)
    return work_dir


def test_smoke_dry_run_has_ray_preflight_and_three_parallel_arms(tmp_path, request):
    run_root = _workspace_tmp(tmp_path, request)
    result = _run(
        "--profile",
        "smoke",
        "--matrix",
        str(MATRIX),
        "--run-root",
        _bash_path(run_root),
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("test_m3_ray_topology.sh") == 1
    assert result.stdout.count("test_m3_quality_eval_arm.sh") == 3
    assert "--nodes=2" in result.stdout
    assert "total_nodes=4" in result.stdout


def test_production_dry_run_requires_gate_and_has_six_arms(tmp_path, request):
    run_root = _workspace_tmp(tmp_path, request)
    failed = _run(
        "--profile",
        "production",
        "--matrix",
        str(MATRIX),
        "--run-root",
        _bash_path(run_root),
        "--dry-run",
    )
    assert failed.returncode != 0
    gate = run_root / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": True}))
    result = _run(
        "--profile",
        "production",
        "--matrix",
        str(MATRIX),
        "--run-root",
        _bash_path(run_root),
        "--smoke-gate",
        _bash_path(gate),
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("test_m3_quality_eval_arm.sh") == 6
    assert "total_nodes=8" in result.stdout
    # BF16 must run TP8xPP2 in production, not TP16xPP1.
    assert "--pipeline-parallel-size 2" in result.stdout
    # Multi-node arms require the two-node Ray topology gate in production too.
    assert "test_m3_ray_topology.sh" in result.stdout


def test_grouped_quality_dry_run_uses_six_independent_srun_arms_and_matrix_time(
    tmp_path, request
):
    run_root = _workspace_tmp(tmp_path, request)
    gate = run_root / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": True}))

    result = _run(
        "--profile",
        "production",
        "--matrix",
        str(GROUPED_MATRIX),
        "--run-root",
        _bash_path(run_root),
        "--smoke-gate",
        _bash_path(gate),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("srun ")]
    assert len(commands) == 6
    assert all("--nodes=1" in line and "--gpus-per-node=8" in line for line in commands)
    assert all("--time 16:00:00" in line for line in commands)
    assert sum("--run-probe 1" in line for line in commands) == 2
    assert sum("--run-probe 0" in line for line in commands) == 4
    assert sum("--probe-tokens 8192" in line for line in commands) == 2
    assert sum("--probe-tokens 0" in line for line in commands) == 4
    assert "sbatch" not in result.stdout
    assert "total_nodes=6" in result.stdout


def test_grouped_quality_rejects_time_override_that_conflicts_with_matrix(
    tmp_path, request
):
    run_root = _workspace_tmp(tmp_path, request)
    gate = run_root / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": True}))
    env = os.environ.copy()
    env["TIME_LIMIT"] = "08:00:00"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--profile",
            "production",
            "--matrix",
            str(GROUPED_MATRIX),
            "--run-root",
            _bash_path(run_root),
            "--smoke-gate",
            _bash_path(gate),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode != 0
    assert "TIME_LIMIT conflicts with matrix arm_time_limit" in result.stderr


def test_r4_dry_run_emits_four_top_level_srun_arms(tmp_path, request):
    run_root = _workspace_tmp(tmp_path, request)
    gate = run_root / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": True}))
    preflight = run_root / "preflight"
    preflight.mkdir()
    (preflight / "resolved_tasks.json").write_text(
        json.dumps(
            {
                "aliases": {
                    "gpqa_diamond": "gpqa_diamond_cot_zeroshot",
                    "mmlu_pro": "mmlu_pro",
                    "gsm8k": "gsm8k_cot",
                    "aime_2025": "aime25",
                }
            }
        ),
        encoding="utf-8",
    )

    result = _run(
        "--profile",
        "production",
        "--matrix",
        str(REASONING_R4_MATRIX),
        "--run-root",
        _bash_path(run_root),
        "--smoke-gate",
        _bash_path(gate),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("srun ")]
    assert len(commands) == 4
    assert all("--nodes=1" in line and "--gpus-per-node=8" in line for line in commands)
    assert all("--time 24:00:00" in line for line in commands)
    assert all("--run-probe 0" in line for line in commands)
    assert sum("--tasks gpqa_diamond_cot_zeroshot" in line for line in commands) == 2
    assert sum(r"--tasks mmlu_pro\,gsm8k_cot\,aime25" in line for line in commands) == 2
    assert "sbatch" not in result.stdout
    assert "total_nodes=4" in result.stdout


def test_bf16_reasoning_r4_smoke_dry_run_uses_tp16_pp1_ray(tmp_path, request):
    spec = load_matrix(BF16_REASONING_R4_MATRIX)
    model = spec.models[0]

    assert len(spec.models) == 1
    assert model.label == "bf16"
    assert model.kind == "bf16"
    assert model.nodes == 2
    assert model.tensor_parallel_size == 16
    assert model.pipeline_parallel_size == 1
    assert model.distributed_executor_backend == "ray"
    assert spec.scheduling.max_parallel_arms == 2
    assert spec.scheduling.arm_time_limit == "12:00:00"

    run_root = _workspace_tmp(tmp_path, request)
    result = _run(
        "--profile",
        "smoke",
        "--matrix",
        str(BF16_REASONING_R4_MATRIX),
        "--run-root",
        _bash_path(run_root),
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    smoke = json.loads((run_root / "smoke_launch_plan.json").read_text())
    assert len(smoke["arms"]) == 1
    assert smoke["total_nodes"] == 2
    arm = smoke["arms"][0]
    assert smoke["profile"] == "smoke"
    assert arm["nodes"] == 2
    assert arm["tensor_parallel_size"] == 16
    assert arm["pipeline_parallel_size"] == 1
    assert arm["distributed_executor_backend"] == "ray"
    assert arm["tasks"] == [
        "gpqa_diamond",
        "mmlu_pro",
        "gsm8k",
        "aime_2025",
    ]
    commands = [
        line
        for line in result.stdout.splitlines()
        if "test_m3_quality_eval_arm.sh" in line
    ]
    assert len(commands) == 1
    assert "--nodes=2" in commands[0]
    assert "--tensor-parallel-size 16" in commands[0]
    assert "--pipeline-parallel-size 1" in commands[0]
    assert "--distributed-executor-backend ray" in commands[0]
    assert "sbatch" not in result.stdout

    gate = tmp_path / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": True}), encoding="utf-8")
    production = build_launch_plan(spec, profile="production", smoke_gate=gate)
    assert production["total_nodes"] == 4
    assert production["max_parallel_arms"] == 2
    assert [arm["shard"] for arm in production["arms"]] == [
        "gpqa",
        "reasoning_suite",
    ]
    assert all(arm["nodes"] == 2 for arm in production["arms"])
    assert all(arm["tensor_parallel_size"] == 16 for arm in production["arms"])
    assert all(arm["pipeline_parallel_size"] == 1 for arm in production["arms"])
    assert all(
        arm["distributed_executor_backend"] == "ray"
        for arm in production["arms"]
    )


def test_runner_scripts_are_valid_bash():
    for script in (
        SCRIPT,
        Path("pipeline/slurm/test_m3_quality_eval_arm.sh"),
        Path("pipeline/slurm/test_m3_ray_topology.sh"),
        Path("pipeline/slurm/test_m3_ray_placement_group.sh"),
    ):
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True)
        assert result.returncode == 0, result.stderr.decode()

    arm_script = Path("pipeline/slurm/test_m3_quality_eval_arm.sh").read_text()
    assert "gpu-monitor-rank-$rank.log" in arm_script
    assert "nvidia-smi --query-gpu=" in arm_script
    assert "ray-logs-rank-$rank.tar.gz" in arm_script
    assert "session_latest/logs" in arm_script
    assert "placement-monitor.log" in arm_script
    assert "M3_PLACEMENT_TIMEOUT_SECONDS" in arm_script
    assert "placement-timeout.json" in arm_script
    assert "M3_MODEL_INIT_TIMEOUT_SECONDS" in arm_script
    assert "model-init-timeout.json" in arm_script


def test_bf16_arm_requires_ray_gate_before_eval():
    arm = Path("pipeline/slurm/test_m3_quality_eval_arm.sh").read_text()
    assert "test_m3_ray_topology.sh" in arm
    assert "ray_preflight/gate.json" in arm
    assert arm.index("ray_preflight/gate.json") < arm.index("pipeline.evalsuite.cli")
    assert "exec ray start" not in arm


def test_smoke_probe_runs_before_eval_and_failure_skips_eval():
    arm = Path("pipeline/slurm/test_m3_quality_eval_arm.sh").read_text()
    probe = "python -m pipeline.m3_distributional_probe run"
    evaluate = '"${eval_cmd[@]}"'

    assert arm.index(probe) < arm.index(evaluate)
    assert 'if [[ "$PROFILE" == smoke && "$RUN_PROBE" == 1 ]]; then' in arm
    assert (
        'if ((rc == 0)); then\n  if [[ -n "$TASKS" ]]; then\n    "${eval_cmd[@]}"'
    ) in arm
    assert "if ((rc == 0 && RUN_PROBE == 1 && probe_ran == 0)); then" in arm


def test_ray_placement_group_diagnostic_is_bounded_and_captures_state():
    script = Path("pipeline/slurm/test_m3_ray_placement_group.sh").read_text()

    assert "EXPECTED_BUNDLES=16" in script
    assert "TIMEOUT_SECONDS=120" in script
    assert 'placement_group([{"GPU": 1}] * expected' in script
    assert "ray.get(group.ready(), timeout=timeout)" in script
    assert "ray list placement-groups" in script
    assert "ray-logs-rank-$rank.tar.gz" in script
    assert "driver-done" in script
    assert "ray stop --force" in script


def test_distributional_probe_receives_distributed_backend():
    arm = Path("pipeline/slurm/test_m3_quality_eval_arm.sh").read_text()
    assert '--distributed-executor-backend "$BACKEND"' in arm


def test_arm_manifest_records_srun_scheduler_identity():
    arm = Path("pipeline/slurm/test_m3_quality_eval_arm.sh").read_text()
    assert 'slurm_job_id=os.environ.get("SLURM_JOB_ID")' in arm
    assert 'slurm_step_id=os.environ.get("SLURM_STEP_ID")' in arm
    assert 'slurm_node_name=os.environ.get("SLURMD_NODENAME")' in arm


def test_multinode_arm_captures_vllm_placement_during_startup():
    arm = Path("pipeline/slurm/test_m3_quality_eval_arm.sh").read_text()
    assert "placement-monitor.log" in arm
    assert "ray list placement-groups --detail" in arm
    assert "placement_monitor_pid" in arm


def test_arm_cleanup_reaps_managed_jobs_and_keeps_ray_archives_exclusive():
    arm = Path("pipeline/slurm/test_m3_quality_eval_arm.sh").read_text()

    assert arm.count("set -m") >= 4
    assert arm.count("set +m") >= 4
    assert 'kill -TERM -- "-$pid"' in arm
    assert "cleanup_done" in arm
    assert "local status=$?" in arm
    assert 'return "$status"' in arm
    for pid in (
        "placement_monitor_pid",
        "gpu_monitor_pid",
        "placement_watchdog_pid",
        "init_watchdog_pid",
    ):
        assert f'stop_managed_job "${pid}"' in arm
    assert 'rm -f "$archive" "$archive.missing"' in arm
    assert 'rm -f "$archive" || true' in arm
    assert 'if ((nodes > 1)) && [[ "$placement_timeout"' in arm


def test_smoke_evidence_counts_tp_times_pp_workers():
    arm = Path("pipeline/slurm/test_m3_quality_eval_arm.sh").read_text()
    assert '"$TP" "$PP" "$rc"' in arm
    assert "'distributed_world_size':tp * pp" in arm


def test_probe_only_arm_skips_evalsuite_and_writes_empty_aggregate(tmp_path, request):
    work_dir = _workspace_tmp(tmp_path, request)
    run_root = work_dir / "run"
    preflight = run_root / "preflight"
    preflight.mkdir(parents=True)
    (run_root / "run_manifest.json").write_text("{}", encoding="utf-8")
    (preflight / "production_sample_manifest.json").write_text("{}", encoding="utf-8")
    (preflight / "resolved_eval_config.yaml").write_text("{}", encoding="utf-8")
    (preflight / "production_probe_corpus.json").write_text("{}", encoding="utf-8")

    fake_bin = work_dir / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "-" ]]; then
  case "$2" in
    *arm_manifest.json)
      printf '{"model_label":"gptq","shard":"distributional_probe"}\n' >"$2"
      ;;
    *arm_complete.json)
      [[ "$3" == "0" ]] && complete=true || complete=false
      printf '{"complete":%s}\n' "$complete" >"$2"
      ;;
    *) echo "unexpected inline writer: $2" >&2; exit 97 ;;
  esac
  exit 0
fi
if [[ "$1 $2" == "-m pipeline.evalsuite.cli" ]]; then
  touch "$FAKE_STATE/evalsuite-called"
  exit 91
fi
if [[ "$1 $2" == "-m pipeline.m3_distributional_probe" ]]; then
  out=""
  while (($#)); do
    [[ "$1" == "--out" ]] && { out=$2; break; }
    shift
  done
  [[ -n "$out" ]] || exit 96
  printf '{"prompt_id":"p","position":1}\n' >"$out"
  exit 0
fi
echo "unexpected python call: $*" >&2
exit 95
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    env["FAKE_STATE"] = _bash_path(work_dir)
    result = subprocess.run(
        [
            "bash",
            "pipeline/slurm/test_m3_quality_eval_arm.sh",
            "--profile",
            "production",
            "--run-root",
            _bash_path(run_root),
            "--matrix",
            "matrix.yaml",
            "--model-label",
            "gptq",
            "--model",
            "/models/gptq",
            "--shard",
            "distributional_probe",
            "--tasks",
            "",
            "--tensor-parallel-size",
            "8",
            "--pipeline-parallel-size",
            "1",
            "--distributed-executor-backend",
            "mp",
            "--run-probe",
            "1",
            "--probe-tokens",
            "8192",
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    arm = run_root / "models" / "gptq" / "shards" / "distributional_probe"
    assert result.returncode == 0, result.stderr
    assert not (work_dir / "evalsuite-called").exists()
    assert json.loads((arm / "aggregate.json").read_text()) == {}
    assert (arm / "distributional_probe.jsonl").is_file()
    assert (arm / "return_code.txt").read_text().strip() == "0"
    assert json.loads((arm / "arm_complete.json").read_text())["complete"] is True
