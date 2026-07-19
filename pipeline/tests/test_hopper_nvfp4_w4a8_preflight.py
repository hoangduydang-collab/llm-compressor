from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from pipeline.hopper_nvfp4_w4a8.preflight import (
    PreflightInput,
    evaluate_preflight,
)


@pytest.fixture
def eligible_input() -> PreflightInput:
    return PreflightInput(
        device_capability=(9, 0),
        humming_version="0.1.10",
        patch_status="patched",
        source_hashes={"humming/config/config.py": "abc123"},
        activation_dtype="float8e4m3",
        weight_dtype="float4e2m1",
        scale_dtype="float8e4m3",
        input_group_size=0,
        weight_group_size=16,
        weight_scale_type="GROUP_TENSOR",
        has_zero_point=False,
        global_scale_count=1,
        expected_global_scale_count=1,
        global_scale_uniform=True,
        accumulator_dtype="float32",
        output_dtype="bfloat16",
        checkpoint_bytes=580,
        transformed_bytes=580,
    )


def test_accepts_only_exact_humming_w4a8_g16_policy(eligible_input):
    report = evaluate_preflight(eligible_input)

    assert report.eligible is True
    assert report.backend == "humming_w4a8_g16"
    assert report.reason_code == "ELIGIBLE"
    assert report.runtime_retry is False
    assert report.details["persistent_ratio"] == 1.0
    assert report.details["source_hashes"] == {"humming/config/config.py": "abc123"}


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"device_capability": (8, 0)}, "SM_NOT_90"),
        ({"humming_version": "0.1.9"}, "HUMMING_VERSION_MISMATCH"),
        ({"patch_status": "pristine"}, "PATCH_INCOMPLETE"),
        ({"weight_dtype": "int4"}, "WEIGHT_DTYPE_NOT_E2M1"),
        ({"activation_dtype": "int8"}, "ACTIVATION_DTYPE_NOT_E4M3"),
        ({"scale_dtype": "float16"}, "SCALE_DTYPE_NOT_E4M3"),
        ({"input_group_size": 16}, "INPUT_GROUP_SIZE_NOT_TENSOR"),
        ({"weight_group_size": 32}, "WEIGHT_GROUP_SIZE_NOT_16"),
        ({"has_zero_point": True}, "ZERO_POINT_UNSUPPORTED"),
        ({"weight_scale_type": "GROUP"}, "SCALE_TYPE_NOT_GROUP_TENSOR"),
        ({"global_scale_count": 0}, "GLOBAL_SCALE_MISSING"),
        ({"global_scale_uniform": False}, "GLOBAL_SCALE_NOT_UNIFORM"),
        ({"accumulator_dtype": "float16"}, "F16_ACCUM_UNSUPPORTED"),
        ({"output_dtype": "float16"}, "OUTPUT_DTYPE_NOT_BF16"),
        ({"transformed_bytes": 639}, "PERSISTENT_RATIO_EXCEEDED"),
    ],
)
def test_rejects_to_marlin_without_runtime_retry(eligible_input, changes, reason):
    candidate = dataclasses.replace(eligible_input, **changes)

    report = evaluate_preflight(candidate)

    assert report.eligible is False
    assert report.backend == "marlin_w4a16"
    assert report.reason_code == reason
    assert report.runtime_retry is False


def test_rejects_differing_global_scale_cardinality(eligible_input):
    candidate = dataclasses.replace(
        eligible_input,
        global_scale_count=2,
        expected_global_scale_count=1,
    )

    assert evaluate_preflight(candidate).reason_code == "GLOBAL_SCALE_CARDINALITY"


@pytest.mark.parametrize(
    ("checkpoint_bytes", "transformed_bytes"),
    [(0, 1), (1, 0), (-1, 1)],
)
def test_rejects_invalid_byte_accounting(
    eligible_input, checkpoint_bytes, transformed_bytes
):
    candidate = dataclasses.replace(
        eligible_input,
        checkpoint_bytes=checkpoint_bytes,
        transformed_bytes=transformed_bytes,
    )

    assert evaluate_preflight(candidate).reason_code == "INVALID_BYTE_ACCOUNTING"


def test_installer_is_pinned_scoped_and_keeps_marlin_separate():
    repo_root = Path(__file__).resolve().parents[2]
    script = (repo_root / "pipeline/slurm/install_humming_nvfp4_w4a8.sh").read_text(
        encoding="utf-8"
    )

    assert "/mnt/nfs/hoangduy/venvs/quant/bin/activate" in script
    assert "humming-kernels" in script
    assert "0.1.10" in script
    assert "patch_humming_nvfp4_w4a8.py" in script
    assert "--check" in script
    assert "get_humming_cache_dir" in script
    assert "marlin" in script.lower()
    assert "install_vllm_m3_serve.sh" not in script
    assert "sbatch" not in script
