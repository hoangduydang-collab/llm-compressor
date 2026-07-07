"""Unit tests for per-task eval checkpointing (no GPU)."""

import json
from pathlib import Path

from pipeline.config import EvalTask
from pipeline.evalsuite.static import (
    checkpoint_task_result,
    load_aggregate_checkpoint,
    pending_eval_tasks,
)


def test_load_aggregate_checkpoint_missing(tmp_path: Path):
    assert load_aggregate_checkpoint(tmp_path / "aggregate.json") == {}


def test_load_aggregate_checkpoint_roundtrip(tmp_path: Path):
    path = tmp_path / "aggregate.json"
    path.write_text(
        json.dumps({"mmlu": {"acc,none": 0.65, "acc_stderr,none": 0.01}}),
        encoding="utf-8",
    )
    loaded = load_aggregate_checkpoint(path)
    assert loaded == {"mmlu": {"acc,none": 0.65}}


def test_pending_eval_tasks_skips_completed():
    tasks = [
        EvalTask(name="wikitext", metric="word_perplexity,none", higher_is_better=False),
        EvalTask(name="mmlu", metric="acc,none", num_fewshot=5),
        EvalTask(name="gsm8k", metric="exact_match,strict-match", num_fewshot=5),
    ]
    pending = pending_eval_tasks(tasks, {"wikitext", "mmlu"})
    assert [t.name for t in pending] == ["gsm8k"]


def test_checkpoint_task_result_writes_aggregate_and_samples(tmp_path: Path):
    task = EvalTask(name="gsm8k", metric="exact_match,strict-match", num_fewshot=5)
    aggregate: dict[str, dict[str, float]] = {}
    batch = {
        "results": {"gsm8k": {"exact_match,strict-match": 0.42}},
        "samples": {
            "gsm8k": [
                {
                    "doc_id": 0,
                    "target": "42",
                    "exact_match,strict-match": 1.0,
                    "resps": ["42"],
                }
            ]
        },
    }

    rows = checkpoint_task_result(
        task=task,
        batch=batch,
        aggregate=aggregate,
        aggregate_path=tmp_path / "aggregate.json",
        samples_dir=tmp_path / "samples",
        log_samples=True,
    )

    assert aggregate["gsm8k"]["exact_match,strict-match"] == 0.42
    assert json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8")) == aggregate
    assert len(rows) == 1
    assert (tmp_path / "samples" / "gsm8k.jsonl").is_file()
