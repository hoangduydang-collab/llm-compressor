"""Structured phase evidence preserves nesting, failures, and unavailable counters."""

import json
from pathlib import Path

import pytest

from llmcompressor.logger import configure_logger
from llmcompressor.utils import metric_logging
from llmcompressor.utils.metric_logging import compression_phase
from pipeline.metrics import capture_quant_metrics, summarize_phases


def _records(path):
    return [
        json.loads(line)["record"]["extra"] for line in path.read_text().splitlines()
    ]


def test_nested_phases_survive_logger_reset_and_restore_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(metric_logging, "_phase_snapshot", lambda rank: {})
    path = tmp_path / "phases.jsonl"
    with capture_quant_metrics(path):
        with compression_phase("load") as outer:
            configure_logger()
            with compression_phase("dispatch", module="model.layers.0") as inner:
                pass
        with compression_phase("save"):
            pass
    records = _records(path)
    assert [r["event"] for r in records] == [
        "phase_start",
        "phase_start",
        "phase_end",
        "phase_end",
        "phase_start",
        "phase_end",
    ]
    assert records[1]["parent_span_id"] == outer
    assert records[2]["span_id"] == inner
    assert records[1]["identity"]["module"] == "model.layers.0"
    assert records[4]["parent_span_id"] is None
    for start, end in ((records[0], records[3]), (records[1], records[2])):
        assert end["duration_ns"] == end["timestamp_ns"] - start["timestamp_ns"]
        assert end["status"] == "ok"
    assert all(record["world_size"] == 1 for record in records)
    summary = summarize_phases([path])
    assert summary["paired_complete"] is True
    assert summary["complete"] is False
    assert summary["missing_run_ranks"] == [0]


def test_phase_failure_reraises_original_and_restores_context(tmp_path, monkeypatch):
    monkeypatch.setattr(metric_logging, "_phase_snapshot", lambda rank: {})
    error = ValueError("original")
    path = tmp_path / "phases.jsonl"
    with capture_quant_metrics(path):
        with pytest.raises(ValueError) as caught:
            with compression_phase("solve"):
                raise error
        with compression_phase("after_failure"):
            pass
    assert caught.value is error
    records = _records(path)
    assert records[1]["status"] == "error"
    assert records[1]["error_type"] == "ValueError"
    assert records[2]["parent_span_id"] is None


def test_lightweight_phase_performs_no_snapshot_reads(tmp_path, monkeypatch):
    def forbidden(rank):
        raise AssertionError("snapshot read")

    monkeypatch.setattr(metric_logging, "_safe_phase_snapshot", forbidden)
    monkeypatch.setattr(
        metric_logging.torch.cuda,
        "synchronize",
        lambda *args: (_ for _ in ()).throw(AssertionError("CUDA sync")),
    )
    path = tmp_path / "lightweight.jsonl"
    with capture_quant_metrics(path):
        with compression_phase("lightweight", collect_snapshot=False):
            pass
    records = _records(path)
    assert [record["snapshot"] for record in records] == [{}, {}]


def test_unavailable_platform_counters_are_not_zero(monkeypatch):
    def unavailable(self, *args, **kwargs):
        raise PermissionError("counter unavailable")

    monkeypatch.setattr(Path, "read_text", unavailable)
    monkeypatch.setattr(metric_logging.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    snapshot = metric_logging._phase_snapshot(0)
    assert snapshot["process"]["memory"]["available"] is False
    assert snapshot["process"]["io"]["available"] is False
    assert snapshot["process"]["io"]["source"] == "/proc/self/io"
    assert snapshot["gpu"]["available"] is False
    assert "allocated_bytes" not in snapshot["gpu"]
    assert snapshot["node"]["cgroup"]["available"] is False


def test_node_counters_only_collected_on_local_rank_zero(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(metric_logging.torch.cuda, "is_initialized", lambda: False)
    assert metric_logging._phase_snapshot(1)["node"] == {
        "available": False,
        "reason": "collected by node local rank zero",
    }


def test_snapshot_failure_cannot_replace_work_exception(tmp_path, monkeypatch):
    def broken_snapshot(rank):
        raise RuntimeError("counter read failed")

    monkeypatch.setattr(metric_logging, "_phase_snapshot", broken_snapshot)
    original = ValueError("solver failed")
    path = tmp_path / "phases.jsonl"
    with capture_quant_metrics(path), pytest.raises(ValueError) as caught:
        with compression_phase("solve"):
            raise original
    assert caught.value is original
    assert _records(path)[1]["snapshot"]["available"] is False


def test_run_capture_encloses_load_save_and_final_span(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pipeline import quantize

    monkeypatch.setattr(metric_logging, "_phase_snapshot", lambda rank: {})
    saved_metadata = {}

    def work(*args, **kwargs):
        with compression_phase("model_load_dispatch"):
            configure_logger()
        with compression_phase("checkpoint_save"):
            pass
        return tmp_path / "checkpoint"

    monkeypatch.setattr(quantize, "_run_quantize", work)
    monkeypatch.setattr(
        quantize.versioning,
        "update_metadata",
        lambda path, data: saved_metadata.update(data),
    )
    cfg = SimpleNamespace(model=SimpleNamespace(id="test-model"))
    assert quantize.run_quantize(cfg, tmp_path) == tmp_path / "checkpoint"
    phases = saved_metadata["quant_metrics"]["phases"]
    assert phases["complete"] is True
    assert set(phases["phases"]) == {
        "quantize_run",
        "model_load_dispatch",
        "checkpoint_save",
    }


def test_failed_load_leaves_raw_paired_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pipeline import quantize

    monkeypatch.setattr(metric_logging, "_phase_snapshot", lambda rank: {})

    def work(*args, **kwargs):
        with compression_phase("model_load_dispatch"):
            raise OSError("load failed")

    monkeypatch.setattr(quantize, "_run_quantize", work)
    cfg = SimpleNamespace(model=SimpleNamespace(id="test-model"))
    with pytest.raises(OSError, match="load failed"):
        quantize.run_quantize(cfg, tmp_path)
    phases = summarize_phases([tmp_path / "quant_metrics.jsonl"])
    assert phases["complete"] is True
    assert phases["phases"]["model_load_dispatch"]["failed_spans"] == 1
    assert phases["phases"]["quantize_run"]["failed_spans"] == 1


def test_run_timing_report_uses_declared_rank_paths(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from pipeline import quantize, timing_report
    from pipeline.distributed import DistributedContext

    calls = {}
    monkeypatch.setattr(
        quantize.metrics, "capture_quant_metrics", lambda path: nullcontext()
    )
    monkeypatch.setattr(quantize.metrics, "summarize_quant_metrics", lambda path: {})
    monkeypatch.setattr(quantize.versioning, "update_metadata", lambda *args: None)
    monkeypatch.setattr(
        quantize, "_run_quantize", lambda *args, **kwargs: tmp_path / "checkpoint"
    )
    monkeypatch.setattr(
        timing_report,
        "write_timing_report",
        lambda paths, prefix: calls.update(paths=list(paths), prefix=prefix),
    )
    cfg = SimpleNamespace(model=SimpleNamespace(id="test-model"))
    ctx = DistributedContext(enabled=True, rank=0, world_size=2, local_rank=0)

    assert quantize.run_quantize(cfg, tmp_path, ctx) == tmp_path / "checkpoint"
    assert calls == {
        "paths": [
            str(tmp_path / "quant_metrics.rank-0.jsonl"),
            str(tmp_path / "quant_metrics.rank-1.jsonl"),
        ],
        "prefix": tmp_path / "timing_report",
    }


def test_report_failure_never_changes_work_result_or_exception(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from pipeline import quantize, timing_report

    monkeypatch.setattr(
        quantize.metrics, "capture_quant_metrics", lambda path: nullcontext()
    )
    monkeypatch.setattr(quantize.metrics, "summarize_quant_metrics", lambda path: {})
    monkeypatch.setattr(quantize.versioning, "update_metadata", lambda *args: None)
    monkeypatch.setattr(
        timing_report,
        "write_timing_report",
        lambda *args: (_ for _ in ()).throw(RuntimeError("report failed")),
    )
    cfg = SimpleNamespace(model=SimpleNamespace(id="test-model"))
    expected = tmp_path / "checkpoint"
    monkeypatch.setattr(quantize, "_run_quantize", lambda *args, **kwargs: expected)
    assert quantize.run_quantize(cfg, tmp_path) == expected

    original = OSError("work failed")
    monkeypatch.setattr(
        quantize,
        "_run_quantize",
        lambda *args, **kwargs: (_ for _ in ()).throw(original),
    )
    with pytest.raises(OSError) as caught:
        quantize.run_quantize(cfg, tmp_path)
    assert caught.value is original


def test_lightweight_snapshot_shape_is_accepted_by_peak_cuda_validator(tmp_path):
    from pipeline.validate_glm53_ep_gptq import peak_cuda_bytes

    path = tmp_path / "lightweight.jsonl"
    with capture_quant_metrics(path):
        with compression_phase("lightweight", collect_snapshot=False):
            pass
    assert peak_cuda_bytes([path]) is None
