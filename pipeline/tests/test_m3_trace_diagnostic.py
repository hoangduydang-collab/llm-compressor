import json

import torch

from pipeline.m3_trace_diagnostic import (
    classify_trace_reports,
    filter_sample_for_model,
    persist_root_artifacts,
    trace_root,
)


class ExplicitInputs(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None):
        return input_ids


def test_filter_sample_for_model_keeps_only_explicit_forward_inputs():
    sample = {
        "input_ids": torch.ones(1, 2),
        "attention_mask": torch.ones(1, 2),
        "labels": torch.ones(1, 2),
    }
    assert list(filter_sample_for_model(ExplicitInputs(), sample)) == [
        "input_ids",
        "attention_mask",
    ]


def test_classify_trace_reports_localizes_structural_boundary():
    reports = {
        "full_wrapper": {"status": "ok", "target_node_count": 0},
        "language_model": {"status": "ok", "target_node_count": 60},
    }
    assert classify_trace_reports(reports) == "multimodal_wrapper_boundary"


def test_persist_root_artifacts_splits_large_graph_evidence(tmp_path):
    report = {
        "status": "ok",
        "graph_code": "def forward():\n    pass\n",
        "nodes": [{"name": "x"}],
    }
    persist_root_artifacts(tmp_path, report)
    compact = json.loads((tmp_path / "report.json").read_text())
    assert "graph_code" not in compact
    assert "nodes" not in compact
    assert (tmp_path / "graph.py").read_text().startswith("def forward")
    assert json.loads((tmp_path / "nodes.json").read_text()) == [{"name": "x"}]


def test_trace_root_persists_exception_and_partial_diagnostics(tmp_path):
    def failing_trace(*args, diagnostics, **kwargs):
        diagnostics["matched_target_count"] = 60
        raise RuntimeError("trace exploded")

    report = trace_root(
        label="full_wrapper",
        model=ExplicitInputs(),
        sample={"input_ids": torch.ones(1, 2)},
        sequential_targets=["DecoderLayer"],
        ignore=[],
        output_dir=tmp_path,
        trace_fn=failing_trace,
    )
    persisted = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "error"
    assert persisted["matched_target_count"] == 60
    assert persisted["error"]["type"] == "RuntimeError"
    assert "trace exploded" in persisted["error"]["traceback"]
