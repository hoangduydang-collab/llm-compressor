"""Lock the quant-metrics parser against the GPTQ/AWQ log message formats.

If a future llm-compressor version changes how GPTQ logs ``error``/``time`` or
how AWQ logs its ``AWQ per-mapping error metrics: {...}`` line, these tests
fail loudly instead of the metrics silently going empty.

Run: pytest pipeline/tests/test_metrics.py
"""

import json

from pipeline.metrics import _parse_awq_metrics, summarize_quant_metrics

# --- Sample messages, mirroring the exact strings llm-compressor emits ------

GPTQ_MESSAGES = [
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
