"""Tests for the distributed shared-CPU offload VMA budget gate.

Regression guard for the 2026-07-17 DDP weights-side VMA exhaustion
(run 20260717T064357Z-m3-ddp-awq-full-r1; see BUGS_AND_FIXES.md).
"""

import json

import pytest

from pipeline.config import ModelConfig, PipelineConfig, QuantizationConfig
from pipeline.distributed import DistributedContext
from pipeline.quantize import (
    _VMA_GUARD_SLACK,
    assert_vma_budget_for_shared_offload,
    estimate_shared_offload_segments,
)


def _write_index(tmp_path, n_tensors: int, total_bytes: int):
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": {f"model.layers.{i}.weight": "model.safetensors" for i in range(n_tensors)},
    }
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps(index))
    return path


def _cfg(tmp_path, max_memory):
    return PipelineConfig(
        name="vma-guard-test",
        model=ModelConfig(
            id=str(tmp_path),
            device_map="auto_offload",
            max_memory=max_memory,
            offload_folder=str(tmp_path / "offload"),
        ),
        quantization=QuantizationConfig(method="awq", scheme="W4AFP8"),
    )


def test_estimate_full_budget_applies_linearization_factor(tmp_path):
    """23,416 index entries produced 63,122 segments in the failed run; the
    3x factor must estimate at or above that observed count."""
    index = _write_index(tmp_path, n_tensors=23416, total_bytes=869_157_697_024)
    planned, n, total = estimate_shared_offload_segments(index, 1e12)
    assert n == 23416 and total == 869_157_697_024
    assert planned >= 63122  # observed segments of run ...-awq-full-r1
    assert planned == int(23416 * 3.0)


def test_estimate_partial_budget_scales_linearly(tmp_path):
    index = _write_index(tmp_path, n_tensors=23416, total_bytes=869_157_697_024)
    planned, _, _ = estimate_shared_offload_segments(index, 32e9)
    assert planned == int(int(23416 * 3.0) * (32e9 / 869_157_697_024))


def test_gate_refuses_m3_scale_shm_plan(tmp_path):
    """The exact failure shape: M3's real index size, whole model to shm."""
    _write_index(tmp_path, n_tensors=23416, total_bytes=869_157_697_024)
    cfg = _cfg(tmp_path, {"cpu": 1e12})
    dist_ctx = DistributedContext(enabled=True, rank=0, world_size=8)
    with pytest.raises(RuntimeError, match="vm.max_map_count"):
        assert_vma_budget_for_shared_offload(cfg, dist_ctx, _max_map_count=65530)


def test_gate_passes_disk_overflow_plan(tmp_path):
    _write_index(tmp_path, n_tensors=23416, total_bytes=869_157_697_024)
    cfg = _cfg(tmp_path, {"cpu": 32e9})
    dist_ctx = DistributedContext(enabled=True, rank=0, world_size=8)
    assert_vma_budget_for_shared_offload(cfg, dist_ctx, _max_map_count=65530)


def test_gate_passes_when_sysctl_raised(tmp_path):
    _write_index(tmp_path, n_tensors=23416, total_bytes=869_157_697_024)
    cfg = _cfg(tmp_path, {"cpu": 1e12})
    dist_ctx = DistributedContext(enabled=True, rank=0, world_size=8)
    assert_vma_budget_for_shared_offload(cfg, dist_ctx, _max_map_count=1048576)


def test_gate_defaults_budget_to_shm_size_when_unset(tmp_path):
    """No explicit max_memory mirrors load.py: whole /dev/shm is the budget."""
    _write_index(tmp_path, n_tensors=23416, total_bytes=869_157_697_024)
    cfg = _cfg(tmp_path, None)
    dist_ctx = DistributedContext(enabled=True, rank=0, world_size=8)
    with pytest.raises(RuntimeError, match="vm.max_map_count"):
        assert_vma_budget_for_shared_offload(
            cfg, dist_ctx, _max_map_count=65530, _shm_total_bytes=1.08e12
        )


def test_gate_noop_single_process(tmp_path):
    """Single-process offload holds weights privately: no per-tensor shm maps."""
    _write_index(tmp_path, n_tensors=23416, total_bytes=869_157_697_024)
    cfg = _cfg(tmp_path, {"cpu": 1e12})
    assert_vma_budget_for_shared_offload(
        cfg, DistributedContext(), _max_map_count=65530
    )


def test_gate_skips_without_local_index(tmp_path):
    cfg = _cfg(tmp_path, {"cpu": 1e12})  # no index.json written
    dist_ctx = DistributedContext(enabled=True, rank=0, world_size=8)
    assert_vma_budget_for_shared_offload(cfg, dist_ctx, _max_map_count=65530)


def test_gate_env_bypass(tmp_path, monkeypatch):
    _write_index(tmp_path, n_tensors=23416, total_bytes=869_157_697_024)
    cfg = _cfg(tmp_path, {"cpu": 1e12})
    dist_ctx = DistributedContext(enabled=True, rank=0, world_size=8)
    monkeypatch.setenv("M3_SKIP_VMA_GUARD", "1")
    assert_vma_budget_for_shared_offload(cfg, dist_ctx, _max_map_count=65530)


def test_slack_leaves_room_for_calibration_vmas():
    """Slack must cover CUDA + allocator + activation-cache maps; the failed
    run's non-shm VMA overhead was ~2.4k (65530 - 63122) at the moment of
    death and still growing, so anything under ~10k would repeat the incident."""
    assert _VMA_GUARD_SLACK >= 10_000
