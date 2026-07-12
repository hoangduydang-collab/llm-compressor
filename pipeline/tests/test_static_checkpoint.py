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


def test_collect_task_samples_merges_group_subtasks():
    from pipeline.evalsuite.static import _collect_task_samples

    batch = {
        "group_subtasks": {
            "mmlu": ["mmlu_abstract_algebra", "mmlu_anatomy"],
        },
        "samples": {
            "mmlu_abstract_algebra": [{"doc_id": 0}],
            "mmlu_anatomy": [{"doc_id": 1}, {"doc_id": 2}],
        },
    }
    assert len(_collect_task_samples(batch, "mmlu")) == 3
    assert len(_collect_task_samples(batch, "gsm8k")) == 0


def test_checkpoint_task_result_mmlu_group_samples(tmp_path: Path):
    task = EvalTask(name="mmlu", metric="acc,none", num_fewshot=5)
    aggregate: dict[str, dict[str, float]] = {}
    batch = {
        "results": {"mmlu": {"acc,none": 0.65}},
        "group_subtasks": {"mmlu": ["mmlu_abstract_algebra"]},
        "samples": {
            "mmlu_abstract_algebra": [
                {"doc_id": 0, "acc,none": 1.0, "target": "A", "resps": ["A"]},
            ],
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

    assert len(rows) == 1
    assert (tmp_path / "samples" / "mmlu.jsonl").is_file()


def test_group_rows_namespace_duplicate_doc_ids_by_subtask(tmp_path: Path):
    task = EvalTask(name="mmlu_pro", metric="acc,none")
    aggregate: dict[str, dict[str, float]] = {}
    batch = {
        "groups": {"mmlu_pro": {"acc,none": 0.5}},
        "group_subtasks": {"mmlu_pro": ["mmlu_pro_math", "mmlu_pro_history"]},
        "samples": {
            "mmlu_pro_math": [
                {"doc_id": 0, "acc,none": 1.0, "target": "A", "resps": ["A"]}
            ],
            "mmlu_pro_history": [
                {"doc_id": 0, "acc,none": 0.0, "target": "B", "resps": ["A"]}
            ],
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

    assert [row["subtask"] for row in rows] == ["mmlu_pro_math", "mmlu_pro_history"]
    assert rows[0]["doc_id"] == rows[1]["doc_id"] == 0
    assert rows[0]["sample_uid"] != rows[1]["sample_uid"]



def test_checkpoint_writes_generation_health_with_periodic_loop(tmp_path: Path):
    task = EvalTask(name="gsm8k", metric="exact_match,strict-match")
    batch = {
        "results": {"gsm8k": {"exact_match,strict-match": 0.0}},
        "samples": {
            "gsm8k": [
                {
                    "doc_id": 0,
                    "target": "42",
                    "exact_match,strict-match": 0.0,
                    "resps": ["one two one two"],
                    "filtered_resps": [None],
                    "response_token_ids": [1, 2] * 8,
                    "max_gen_toks": 16,
                }
            ]
        },
    }

    rows = checkpoint_task_result(
        task=task,
        batch=batch,
        aggregate={},
        aggregate_path=tmp_path / "aggregate.json",
        samples_dir=tmp_path / "samples",
        log_samples=True,
    )

    assert rows[0]["response"] == "one two one two"
    assert rows[0]["health"]["periodic_loop"] is True
    health_path = tmp_path / "generation_health" / "gsm8k.json"
    health = json.loads(health_path.read_text(encoding="utf-8"))
    assert health["periodic_loop_count"] == 1
    assert health["length_cap_hit_count"] == 1



def test_loglikelihood_response_is_not_treated_as_missing_generation(tmp_path: Path):
    task = EvalTask(name="mmlu_pro", metric="acc,none")
    batch = {
        "results": {"mmlu_pro": {"acc,none": 1.0}},
        "samples": {
            "mmlu_pro": [
                {
                    "doc_id": 0,
                    "acc,none": 1.0,
                    "resps": [[(-1.2, True)]],
                }
            ]
        },
    }

    rows = checkpoint_task_result(
        task=task,
        batch=batch,
        aggregate={},
        aggregate_path=tmp_path / "aggregate.json",
        samples_dir=tmp_path / "samples",
        log_samples=True,
    )

    assert rows[0]["health"] == {"applicable": False}
    health = json.loads(
        (tmp_path / "generation_health" / "mmlu_pro.json").read_text()
    )
    assert health["samples"] == 0
    assert health["not_applicable_count"] == 1
