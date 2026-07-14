import os
from pathlib import Path

import pytest

from pipeline.distributed import DistributedContext


def test_single_process_context_is_noop(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    ctx = DistributedContext.from_environment()

    assert (ctx.enabled, ctx.rank, ctx.world_size, ctx.local_rank) == (
        False,
        0,
        1,
        0,
    )
    assert ctx.is_source is True
    assert ctx.rank_path(Path("run/quant_metrics.jsonl")) == Path(
        "run/quant_metrics.jsonl"
    )


def test_rank_path_suffixes_distributed_evidence():
    ctx = DistributedContext(enabled=True, rank=3, world_size=8, local_rank=3)

    assert ctx.is_source is False
    assert ctx.rank_path(Path("run/quant_metrics.jsonl")) == Path(
        "run/quant_metrics.rank-3.jsonl"
    )
    assert ctx.rank_path(Path("run/model_provenance.json")) == Path(
        "run/model_provenance.rank-3.json"
    )


def test_distributed_environment_requires_torchrun_rank_variables(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    with pytest.raises(RuntimeError, match="RANK.*LOCAL_RANK"):
        DistributedContext.from_environment()


def test_invalid_world_size_is_rejected(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "not-an-int")

    with pytest.raises(RuntimeError, match="WORLD_SIZE"):
        DistributedContext.from_environment()


def test_rank_path_preserves_multi_suffixes():
    ctx = DistributedContext(enabled=True, rank=1, world_size=2, local_rank=1)

    assert ctx.rank_path(Path("run/metrics.tar.jsonl")) == Path(
        "run/metrics.tar.rank-1.jsonl"
    )


def test_environment_snapshot_is_json_serializable(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    snapshot = DistributedContext.from_environment().snapshot()

    assert snapshot == {
        "enabled": False,
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "slurm_job_id": "123",
        "slurm_step_id": None,
        "node": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME"),
    }


class _FakeBroadcastContext:
    def __init__(self, *, is_source, broadcast_result):
        self.is_source = is_source
        self.broadcast_result = Path(broadcast_result)
        self.seen = []

    def broadcast_path(self, path):
        self.seen.append(path)
        return self.broadcast_result


def test_source_rank_creates_then_broadcasts_run_dir(monkeypatch, tmp_path):
    from pipeline import run

    created = tmp_path / "created"
    monkeypatch.setattr(run.versioning, "create_run_dir", lambda cfg: created)
    ctx = _FakeBroadcastContext(is_source=True, broadcast_result=created)

    result = run._create_distributed_run_dir(object(), ctx)

    assert result == created
    assert ctx.seen == [created]


def test_non_source_rank_only_receives_run_dir(monkeypatch, tmp_path):
    from pipeline import run

    def fail_if_called(cfg):
        raise AssertionError("non-source rank created a run directory")

    monkeypatch.setattr(run.versioning, "create_run_dir", fail_if_called)
    received = tmp_path / "source-created"
    ctx = _FakeBroadcastContext(is_source=False, broadcast_result=received)

    result = run._create_distributed_run_dir(object(), ctx)

    assert result == received
    assert ctx.seen == [None]
