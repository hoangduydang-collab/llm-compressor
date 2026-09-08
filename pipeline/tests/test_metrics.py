"""Lock the quant-metrics parser against the GPTQ/AWQ log message formats.

If a future llm-compressor version changes how GPTQ logs ``error``/``time`` or
how AWQ logs its ``AWQ per-mapping error metrics: {...}`` line, these tests
fail loudly instead of the metrics silently going empty.

Run: pytest pipeline/tests/test_metrics.py
"""

import json
from types import SimpleNamespace

from pipeline.metrics import (
    _is_captured_message,
    _parse_awq_metrics,
    capture_quant_metrics,
    summarize_quant_metrics,
    summarize_quantized_layers,
)

# --- Sample messages, mirroring the exact strings llm-compressor emits ------

GPTQ_MESSAGES = [
    (
        "INFO",
        "Quantizing language_model.layers.3.mlp.experts.0.down_proj using 8 samples",
    ),
    (
        "INFO",
        "Quantizing language_model.layers.31.mlp.experts.0.down_proj using 8 samples",
    ),
    (
        "INFO",
        "Quantizing language_model.layers.59.mlp.experts.0.down_proj using 8 samples",
    ),
    ("METRIC", "error 1724.52"),
    ("METRIC", "time 0.49s"),
    ("METRIC", "error 129.39"),
    ("METRIC", "time 0.18s"),
    ("METRIC", "Accelerator 0 | usage: 7.88% | total memory: 85.0 Gb"),
]

AWQ_DATA = {
    "quantization_config": {"duo_scaling": True, "n_grid": 20},
    "total_layers": 2,
    "metrics": [
        {
            "layer_name": "model.layers.0.mlp",
            "parent_name": "p0",
            "initial_error": 1.0e-3,
            "best_error": 9.8e-06,
            "reduction": 0.0098,
        },
        {
            "layer_name": "model.layers.1.mlp",
            "parent_name": "p1",
            "initial_error": 2.0e-3,
            "best_error": 1.2e-05,
            "reduction": 0.006,
        },
    ],
}
AWQ_MESSAGE = "AWQ per-mapping error metrics: " + repr(AWQ_DATA)


def test_capture_cleanup_tolerates_sink_removed_by_library(tmp_path):
    from loguru import logger

    with capture_quant_metrics(tmp_path / "quant_metrics.jsonl"):
        # library wipes every handler (including ours) without re-installing;
        # exiting the context must stay silent instead of raising ValueError
        logger.remove()


def test_capture_survives_llmcompressor_logger_reset(tmp_path):
    """Distributed-run regression (r6): `oneshot` calls
    `configure_distributed_logger()` internally, whose `logger.remove()` reset
    used to disconnect the capture sink AFTER it was installed, leaving
    `quant_metrics.rank-*.jsonl` empty while native METRIC records went to
    stdout. The external-sink registry must re-install the sink on reset.
    """
    from llmcompressor.logger import configure_logger, logger

    path = tmp_path / "quant_metrics.jsonl"
    with capture_quant_metrics(path):
        configure_logger()  # the reset that oneshot performs mid-capture
        logger.log("METRIC", "time 0.23s")
        logger.info(
            "Quantizing model.language_model.layers.3.mlp.experts.0.gate_proj "
            "using 8 samples"
        )
    lines = [
        json.loads(line)["record"]["message"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "time 0.23s" in lines
    assert any(message.startswith("Quantizing") for message in lines)

    # the sink must also be gone after the context exits, even after a reset
    logger.log("METRIC", "after-exit record")
    lines_after = path.read_text(encoding="utf-8").splitlines()
    assert not any("after-exit record" in line for line in lines_after)


def _write_jsonl(path, records):
    """records: list of (level_name, message). Mirrors loguru serialize=True."""
    with path.open("w", encoding="utf-8") as fh:
        for level, msg in records:
            fh.write(
                json.dumps(
                    {"text": "x", "record": {"message": msg, "level": {"name": level}}}
                )
                + "\n"
            )


def test_missing_file_returns_unavailable(tmp_path):
    assert summarize_quant_metrics(tmp_path / "nope.jsonl") == {"available": False}


def test_gptq_error_and_time_distributions(tmp_path):
    p = tmp_path / "gptq.jsonl"
    _write_jsonl(p, GPTQ_MESSAGES)
    summary = summarize_quant_metrics(p)

    assert summary["available"] is True
    assert summary["num_module_errors"] == 2
    assert summary["error"]["max"] == 1724.52
    assert summary["error"]["min"] == 129.39
    assert summary["time_s"]["count"] == 2
    assert abs(summary["time_s"]["total"] - 0.67) < 1e-9
    # GPTQ run -> no AWQ block
    assert "awq" not in summary


def test_awq_parses_best_error_and_reduction(tmp_path):
    p = tmp_path / "awq.jsonl"
    _write_jsonl(p, [("DEBUG", AWQ_MESSAGE)])
    summary = summarize_quant_metrics(p)

    assert summary["available"] is True
    assert summary["awq"]["num_layers"] == 2
    assert summary["awq"]["best_error"]["min"] == 9.8e-06
    assert summary["awq"]["best_error"]["max"] == 1.2e-05
    assert summary["awq"]["reduction"]["max"] == 0.0098
    assert summary["awq"]["reduction"]["min"] == 0.006
    # AWQ run -> no GPTQ module-error metrics
    assert summary["num_module_errors"] == 0


def test_parse_awq_metrics_direct():
    metrics = _parse_awq_metrics(AWQ_MESSAGE)
    assert metrics is not None
    assert len(metrics) == 2
    assert metrics[0]["best_error"] == 9.8e-06


def test_malformed_awq_payload_is_skipped(tmp_path):
    p = tmp_path / "bad.jsonl"
    _write_jsonl(p, [("DEBUG", "AWQ per-mapping error metrics: {not: valid, repr}")])
    summary = summarize_quant_metrics(p)
    # No crash, and no awq block produced.
    assert summary["available"] is True
    assert "awq" not in summary
    assert _parse_awq_metrics("AWQ per-mapping error metrics: {bad") is None


def test_non_awq_message_not_parsed():
    assert _parse_awq_metrics("error 1724.52") is None


def test_capture_filter_keeps_native_gptq_work_record():
    record = {
        "level": SimpleNamespace(name="INFO"),
        "message": "Quantizing language_model.layers.3.mlp.down_proj using 8 samples",
    }

    assert _is_captured_message(record) is True


def test_summarize_gptq_quantized_layers_from_native_records(tmp_path):
    p = tmp_path / "gptq.jsonl"
    _write_jsonl(p, GPTQ_MESSAGES)

    summary = summarize_quantized_layers([p], method="gptq")

    assert summary == {
        "method": "gptq",
        "record_count": 3,
        "layers": [3, 31, 59],
        "unresolved_names": [],
    }


def test_summarize_awq_quantized_layers_from_native_records(tmp_path):
    p = tmp_path / "awq.jsonl"
    data = dict(AWQ_DATA)
    data["metrics"] = [
        {"layer_name": f"language_model.layers.{layer}.mlp", "best_error": 0.1}
        for layer in (3, 31, 59)
    ]
    _write_jsonl(p, [("DEBUG", "AWQ per-mapping error metrics: " + repr(data))])

    summary = summarize_quantized_layers([p], method="awq")

    assert summary["record_count"] == 3
    assert summary["layers"] == [3, 31, 59]
    assert summary["unresolved_names"] == []


def test_summarize_quantized_layers_exposes_empty_and_unexpected_records(tmp_path):
    empty = tmp_path / "empty.jsonl"
    _write_jsonl(empty, [("METRIC", "error 1.0")])
    unexpected = tmp_path / "unexpected.jsonl"
    _write_jsonl(
        unexpected,
        [("INFO", "Quantizing language_model.layers.8.mlp.down_proj using 8 samples")],
    )

    assert summarize_quantized_layers([empty], method="gptq")["record_count"] == 0
    assert summarize_quantized_layers([unexpected], method="gptq")["layers"] == [8]


def test_non_language_model_layer_stack_does_not_count_as_decoder_work(tmp_path):
    p = tmp_path / "vision.jsonl"
    name = "vision_tower.layers.3.mlp.down_proj"
    _write_jsonl(p, [("INFO", f"Quantizing {name} using 8 samples")])

    summary = summarize_quantized_layers([p], method="gptq")

    assert summary["layers"] == []
    assert summary["unresolved_names"] == [name]


def test_glm_decoder_root_is_accepted_without_vision_false_positives(tmp_path):
    p = tmp_path / "glm.jsonl"
    names = [
        "model.layers.7.mlp.down_proj",
        "vision_tower.model.layers.8.mlp.down_proj",
        "other.model.layers.9.mlp.down_proj",
    ]
    _write_jsonl(p, [("INFO", f"Quantizing {name} using 8 samples") for name in names])
    summary = summarize_quantized_layers([p], method="gptq")
    assert summary["layers"] == [7]
    assert summary["unresolved_names"] == sorted(names[1:])


def _write_phases(path, spans, *, world_size=None, node="node"):
    if world_size is None:
        world_size = max(span[1] for span in spans) + 1
    with path.open("w") as fh:
        for span, rank, phase, start, stop, parent in spans:
            for event, timestamp in (("phase_start", start), ("phase_end", stop)):
                if timestamp is None:
                    continue
                record = {
                    "extra": {
                        "span_id": span,
                        "rank": rank,
                        "node": node,
                        "world_size": world_size,
                        "phase": phase,
                        "event": event,
                        "timestamp_ns": int(timestamp * 1e9),
                        "parent_span_id": parent,
                        "status": "ok",
                    }
                }
                fh.write(json.dumps({"record": record}) + "\n")


def test_phase_summary_separates_rank_work_from_elapsed_and_nested_work(tmp_path):
    from pipeline.metrics import summarize_phases

    path = tmp_path / "phases.jsonl"
    _write_phases(
        path,
        [
            ("outer0", 0, "quantize_run", 0, 10, None),
            ("nested0", 0, "solve", 1, 4, "outer0"),
            ("overlap0", 0, "prewarm", 2, 7, "outer0"),
            ("outer1", 1, "quantize_run", 0, 9, None),
            ("nested1", 1, "solve", 2, 6, "outer1"),
        ],
    )
    summary = summarize_phases([path])
    assert summary["critical_path_elapsed_s"] == 10
    assert summary["summed_rank_work_s"] == 19
    assert summary["phases"]["solve"]["summed_rank_work_s"] == 7
    assert summary["phases"]["solve"]["max_rank_observed_s"] == 4
    assert summary["complete"] is True


def test_missing_end_retains_incomplete_phase_without_zero_duration(tmp_path):
    from pipeline.metrics import summarize_phases

    path = tmp_path / "phases.jsonl"
    _write_phases(path, [("killed", 0, "load", 1, None, None)])
    summary = summarize_phases([path])
    assert summary["complete"] is False
    assert summary["unmatched_starts"] == ["killed"]
    assert summary["critical_path_elapsed_s"] is None
    assert summary["summed_rank_work_s"] is None
    assert summary["phases"] == {}
    assert summarize_phases([tmp_path / "missing"]) == {"available": False}


def test_phase_summary_reports_missing_rank_logs(tmp_path):
    from pipeline.metrics import summarize_phases

    rank0 = tmp_path / "rank0.jsonl"
    rank1 = tmp_path / "rank1.jsonl"
    _write_phases(rank0, [("run0", 0, "run", 1, 5, None)])
    summary = summarize_phases([rank0, rank1])
    assert summary["complete"] is False
    assert summary["missing_paths"] == [str(rank1)]
    assert summary["critical_path_elapsed_s"] == 4
    assert summary["observed_ranks"] == ["node:0"]


def test_world_size_exposes_entirely_absent_ranks(tmp_path):
    from pipeline.metrics import summarize_phases

    path = tmp_path / "rank0.jsonl"
    _write_phases(path, [("run0", 0, "quantize_run", 0, 10, None)], world_size=8)
    summary = summarize_phases([path])
    assert summary["paired_complete"] is True
    assert summary["complete"] is False
    assert summary["expected_world_size"] == 8
    assert summary["missing_ranks"] == list(range(1, 8))
    assert summary["missing_run_ranks"] == list(range(1, 8))


def test_trace_only_evidence_cannot_claim_run_completeness(tmp_path):
    from pipeline.metrics import summarize_phases

    path = tmp_path / "trace.jsonl"
    _write_phases(path, [("trace", 0, "trace", 0, 1, None)])
    summary = summarize_phases([path])
    assert summary["paired_complete"] is True
    assert summary["complete"] is False
    assert summary["missing_run_ranks"] == [0]
    assert summary["unenclosed_spans"] == ["trace"]


def test_missing_parent_is_incomplete_even_with_a_run_root(tmp_path):
    from pipeline.metrics import summarize_phases

    path = tmp_path / "phases.jsonl"
    _write_phases(
        path,
        [
            ("run", 0, "quantize_run", 0, 5, None),
            ("solve", 0, "solve", 1, 2, "missing_calibration"),
        ],
    )
    summary = summarize_phases([path])
    assert summary["paired_complete"] is True
    assert summary["complete"] is False
    assert summary["missing_parent_spans"] == ["missing_calibration"]
    assert summary["unenclosed_spans"] == ["solve"]


def test_staggered_ranks_use_whole_node_envelope(tmp_path):
    from pipeline.metrics import summarize_phases

    path = tmp_path / "ranks.jsonl"
    _write_phases(
        path,
        [
            ("run0", 0, "quantize_run", 0, 10, None),
            ("run1", 1, "quantize_run", 5, 15, None),
        ],
    )
    summary = summarize_phases([path])
    assert summary["complete"] is True
    assert summary["critical_path_elapsed_s"] == 15
    assert summary["max_rank_elapsed_s"] == 10
    assert summary["summed_rank_work_s"] == 20


def test_elapsed_never_compares_clocks_across_nodes(tmp_path):
    from pipeline.metrics import summarize_phases

    node0, node1 = tmp_path / "node0.jsonl", tmp_path / "node1.jsonl"
    _write_phases(
        node0,
        [("run0", 0, "quantize_run", 0, 10, None)],
        world_size=2,
        node="node0",
    )
    _write_phases(
        node1,
        [("run1", 1, "quantize_run", 1000, 1012, None)],
        world_size=2,
        node="node1",
    )
    summary = summarize_phases([node0, node1])
    assert summary["complete"] is True
    assert summary["critical_path_elapsed_s"] == 12
    assert summary["max_rank_elapsed_s"] == 12
    assert summary["summed_rank_work_s"] == 22
