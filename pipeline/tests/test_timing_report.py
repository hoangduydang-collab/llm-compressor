import json
import subprocess
import sys

from pipeline.timing_report import build_timing_report, write_timing_report


def _boundary(
    event,
    phase,
    span,
    rank,
    timestamp,
    *,
    node="node-a",
    parent=None,
    identity=None,
    status=None,
    snapshot=None,
    world=2,
):
    extra = {
        "event": event,
        "phase": phase,
        "span_id": span,
        "rank": rank,
        "node": node,
        "world_size": world,
        "timestamp_ns": timestamp,
        "parent_span_id": parent,
        "identity": identity or {},
        "status": status or ("running" if event == "phase_start" else "ok"),
        "snapshot": snapshot,
    }
    return json.dumps({"record": {"extra": extra, "message": event}})


def _span(lines, phase, span, rank, lo, hi, **kwargs):
    lines.append(_boundary("phase_start", phase, span, rank, lo, **kwargs))
    lines.append(_boundary("phase_end", phase, span, rank, hi, **kwargs))


def test_report_preserves_identity_rank_skew_and_nested_elapsed(tmp_path):
    paths = []
    for rank, node, offset, scale in (
        (0, "node-a", 0, 1),
        (1, "node-b", 9_000_000_000, 2),
    ):
        lines = []
        _span(
            lines,
            "quantize_run",
            f"root-{rank}",
            rank,
            offset,
            offset + 100 * scale,
            node=node,
        )
        _span(
            lines,
            "calibration_forward",
            f"cal-{rank}",
            rank,
            offset + 10,
            offset + 30 * scale,
            node=node,
            parent=f"root-{rank}",
            identity={"subgraph": 3},
        )
        _span(
            lines,
            "awq_search",
            f"awq-{rank}",
            rank,
            offset + 12,
            offset + 20 * scale,
            node=node,
            parent=f"cal-{rank}",
            identity={"module": "model.layers.7.input_layernorm"},
        )
        path = tmp_path / f"rank-{rank}.jsonl"
        path.write_text("\n".join(lines) + "\n")
        paths.append(path)

    report = build_timing_report(paths)
    assert report["summary"]["critical_path_elapsed_s"] == 0.0000002
    assert report["summary"]["summed_rank_work_s"] == 0.0000003
    awq = next(group for group in report["groups"] if group["phase"] == "awq_search")
    assert awq["decoder_layer"] == 7
    assert awq["identity"] == {"module": "model.layers.7.input_layernorm"}
    assert awq["slowest_rank"] == "node-b:1"
    assert awq["spread_s"] > 0
    assert "collective latency" in report["timing_scope"]


def test_incomplete_failed_truncated_missing_and_duplicate_evidence(tmp_path):
    path = tmp_path / "rank-0.jsonl"
    lines = []
    _span(lines, "quantize_run", "root", 0, 0, 100, world=2, status="error")
    lines.append(
        _boundary("phase_start", "open", "open", 0, 10, parent="root", world=2)
    )
    duplicate = _boundary("phase_start", "dup", "dup", 0, 20, parent="root", world=2)
    lines.extend(
        [
            duplicate,
            duplicate,
            _boundary("phase_end", "dup", "dup", 0, 30, parent="root", world=2),
        ]
    )
    path.write_text("\n".join(lines) + "\n{truncated")

    report = build_timing_report([path, tmp_path / "rank-1.jsonl"])
    assert report["evidence_complete"] is False
    assert report["summary"]["missing_paths"]
    assert report["summary"]["unmatched_starts"] == ["open"]
    assert report["ambiguities"]["duplicate_span_boundaries"] == ["dup"]
    assert report["ambiguities"]["malformed_or_truncated_lines"] == 1
    assert (
        next(g for g in report["groups"] if g["phase"] == "quantize_run")["status"]
        == "failed"
    )


def test_resource_deltas_reset_and_gauges(tmp_path):
    def snap(read, rss, allocated):
        return {
            "process": {
                "scope": "process; I/O is not unique filesystem traffic",
                "io": {"available": True, "counters": {"read_bytes": read}},
                "memory": {"rss_bytes": rss, "peak_rss_bytes": rss + 5},
            },
            "gpu": {"available": True, "allocated_bytes": allocated},
        }

    path = tmp_path / "rank.jsonl"
    path.write_text(
        _boundary(
            "phase_start",
            "quantize_run",
            "root",
            0,
            0,
            snapshot=snap(20, 100, 10),
            world=1,
        )
        + "\n"
        + _boundary(
            "phase_end",
            "quantize_run",
            "root",
            0,
            10,
            snapshot=snap(10, 120, 8),
            world=1,
        )
        + "\n"
    )
    obs = build_timing_report([path])["resource_observations"][0]
    assert obs["process_io_delta"]["read_bytes"] is None
    assert obs["process_io_delta"]["write_bytes"] is None
    assert obs["rss"]["rss_bytes"] == {"start": 100, "end": 120}
    assert obs["gpu_allocator"]["allocated_bytes"] == {"start": 10, "end": 8}


def test_write_and_cli_emit_json_and_markdown(tmp_path):
    log = tmp_path / "rank.jsonl"
    lines = []
    _span(lines, "quantize_run", "root", 0, 0, 10, world=1)
    log.write_text("\n".join(lines) + "\n")
    prefix = tmp_path / "report"
    report = write_timing_report([log], prefix)
    assert report["evidence_complete"] is True
    markdown = prefix.with_suffix(".md").read_text()
    assert "Phase totals" in markdown
    assert "Slowest layer and subgraph groups" in markdown
    assert json.loads(prefix.with_suffix(".json").read_text())["source_paths"] == [
        str(log)
    ]

    cli_prefix = tmp_path / "cli-report"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline.timing_report",
            str(log),
            "--output-prefix",
            str(cli_prefix),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert cli_prefix.with_suffix(".json").exists()


def test_same_phase_and_layer_overlaps_are_unioned_and_parent_mismatch_excluded(
    tmp_path,
):
    lines = []
    _span(lines, "quantize_run", "root", 0, 0, 100, world=1)
    _span(
        lines,
        "awq_search",
        "a",
        0,
        10,
        30,
        parent="root",
        identity={"module": "model.layers.4.a"},
        world=1,
    )
    _span(
        lines,
        "awq_search",
        "b",
        0,
        20,
        40,
        parent="root",
        identity={"module": "model.layers.4.b"},
        world=1,
    )
    lines.append(_boundary("phase_start", "bad", "bad", 0, 50, parent="root", world=1))
    lines.append(_boundary("phase_end", "bad", "bad", 0, 60, parent="other", world=1))
    path = tmp_path / "rank.jsonl"
    path.write_text("\n".join(lines) + "\n")

    report = build_timing_report([path])
    phase = next(item for item in report["phases"] if item["phase"] == "awq_search")
    assert phase["max_duration_s"] == 30 / 1e9
    rollup = report["decoder_layer_rollups"][0]
    assert rollup["decoder_layer"] == 4
    assert rollup["max_duration_s"] == 30 / 1e9
    assert not any(group["phase"] == "bad" for group in report["groups"])
    assert report["summary"]["invalid_spans"] == ["bad"]


def test_automatic_report_defers_without_parsing_over_budget(tmp_path, monkeypatch):
    import pipeline.timing_report as timing_report

    log = tmp_path / "large.jsonl"
    log.write_bytes(b"x" * 11)
    monkeypatch.setattr(
        timing_report,
        "build_timing_report",
        lambda paths: (_ for _ in ()).throw(AssertionError("parsed")),
    )

    prefix = tmp_path / "timing_report"
    report = timing_report.write_automatic_timing_report(
        [log], prefix, max_input_bytes=10
    )
    assert report["status"] == "deferred"
    assert report["input_sizes_bytes"] == {str(log): 11}
    assert "python" in report["offline_command"]
    assert "pipeline.timing_report" in prefix.with_suffix(".md").read_text()
    assert json.loads(prefix.with_suffix(".json").read_text())["status"] == "deferred"
