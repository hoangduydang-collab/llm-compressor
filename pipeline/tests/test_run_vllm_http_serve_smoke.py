"""CPU dry-run tests for the MiniMax-M3 HTTP serving launcher."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "pipeline/slurm/run_vllm_http_serve_smoke.sh"
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.is_file() else "bash"


def _run_launcher(tmp_path, **overrides):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiniMaxM3ForCausalLM"],
                "model_type": "minimax_m3",
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    for name in (
        "EXTRA_VLLM_ARGS",
        "M3_W4A8_BACKEND",
        "VLLM_HUMMING_MOE_GEMM_TYPE",
        "VLLM_HUMMING_USE_F16_ACCUM",
        "HUMMING_CACHE_DIR",
        "HUMMING_M3_W4A8_CACHE_ROOT",
    ):
        env.pop(name, None)
    env.update(
        {
            "PRINT_EFFECTIVE_CONFIG": "1",
            "CKPT": str(checkpoint),
            "MODEL_ID": str(checkpoint),
            "LOG": str(tmp_path / "serve.log"),
            "PID_FILE": str(tmp_path / "serve.pid"),
            **overrides,
        }
    )
    return subprocess.run(
        [BASH, str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_default_backend_preserves_cutlass_command(tmp_path):
    completed = _run_launcher(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "M3_W4A8_BACKEND=cutlass" in completed.stdout
    assert "--quantization humming" not in completed.stdout


def test_humming_backend_adds_structured_quantization_and_policy(tmp_path):
    completed = _run_launcher(tmp_path, M3_W4A8_BACKEND="humming")

    assert completed.returncode == 0, completed.stderr
    assert "M3_W4A8_BACKEND=humming" in completed.stdout
    assert "VLLM_HUMMING_USE_F16_ACCUM=0" in completed.stdout
    assert "VLLM_HUMMING_MOE_GEMM_TYPE=indexed" in completed.stdout
    assert "HUMMING_CACHE_DIR=" in completed.stdout
    assert "cache-m3-gptq-w4a8-v1" in completed.stdout
    effective = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("EFFECTIVE_ARGV:")
    )
    assert effective.count("--quantization humming") == 1


def test_rejects_unknown_backend_before_effective_command(tmp_path):
    completed = _run_launcher(tmp_path, M3_W4A8_BACKEND="unknown")

    assert completed.returncode != 0
    assert "M3_W4A8_BACKEND must be cutlass or humming" in completed.stderr
    assert "EFFECTIVE_ARGV:" not in completed.stdout


def test_rejects_raw_quantization_pair_in_extra_args(tmp_path):
    completed = _run_launcher(
        tmp_path,
        EXTRA_VLLM_ARGS="--quantization marlin",
    )

    assert completed.returncode != 0
    assert "EXTRA_VLLM_ARGS must not set --quantization" in completed.stderr


def test_rejects_raw_quantization_assignment_in_extra_args(tmp_path):
    completed = _run_launcher(
        tmp_path,
        EXTRA_VLLM_ARGS="--quantization=humming",
    )

    assert completed.returncode != 0
    assert "EXTRA_VLLM_ARGS must not set --quantization" in completed.stderr


def test_rejects_humming_fp16_accumulation(tmp_path):
    completed = _run_launcher(
        tmp_path,
        M3_W4A8_BACKEND="humming",
        VLLM_HUMMING_USE_F16_ACCUM="1",
    )

    assert completed.returncode != 0
    assert "requires FP32 accumulation" in completed.stderr


def test_rejects_grouped_humming_gemm_during_first_qualification(tmp_path):
    completed = _run_launcher(
        tmp_path,
        M3_W4A8_BACKEND="humming",
        VLLM_HUMMING_MOE_GEMM_TYPE="grouped_contiguous",
    )

    assert completed.returncode != 0
    assert "requires indexed MoE GEMM" in completed.stderr
