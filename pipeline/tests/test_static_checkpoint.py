"""Unit tests for per-task eval checkpointing (no GPU)."""

import json
from pathlib import Path

import pytest

from pipeline.config import EvalTask, PipelineConfig
from pipeline.evalsuite.static import (
    checkpoint_task_result,
    load_aggregate_checkpoint,
    pending_eval_tasks,
    run_static_eval,
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
        EvalTask(
            name="wikitext", metric="word_perplexity,none", higher_is_better=False
        ),
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
    assert (
        json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
        == aggregate
    )
    assert len(rows) == 1
    assert (tmp_path / "samples" / "gsm8k.jsonl").is_file()


def test_checkpoint_collapses_identical_duplicate_sample_uids(tmp_path: Path):
    task = EvalTask(name="gsm8k", metric="exact_match,strict-match")
    sample = {
        "doc_id": 7,
        "target": "42",
        "exact_match,strict-match": 1.0,
        "resps": ["42"],
    }
    batch = {
        "results": {"gsm8k": {"exact_match,strict-match": 1.0}},
        "samples": {"gsm8k": [sample, dict(sample)]},
    }

    rows = checkpoint_task_result(
        task=task,
        batch=batch,
        aggregate={},
        aggregate_path=tmp_path / "aggregate.json",
        samples_dir=tmp_path / "samples",
        log_samples=True,
    )

    assert len(rows) == 1
    assert len((tmp_path / "samples" / "gsm8k.jsonl").read_text().splitlines()) == 1


def test_checkpoint_rejects_conflicting_duplicate_sample_uids(tmp_path: Path):
    task = EvalTask(name="gsm8k", metric="exact_match,strict-match")
    batch = {
        "results": {"gsm8k": {"exact_match,strict-match": 0.5}},
        "samples": {
            "gsm8k": [
                {"doc_id": 7, "exact_match,strict-match": 1.0, "resps": ["42"]},
                {"doc_id": 7, "exact_match,strict-match": 0.0, "resps": ["41"]},
            ]
        },
    }

    with pytest.raises(ValueError, match="conflicting duplicate sample_uid"):
        checkpoint_task_result(
            task=task,
            batch=batch,
            aggregate={},
            aggregate_path=tmp_path / "aggregate.json",
            samples_dir=tmp_path / "samples",
            log_samples=True,
        )
    assert not (tmp_path / "aggregate.json").exists()


def test_checkpoint_selects_the_configured_lm_eval_filter(tmp_path: Path):
    task = EvalTask(name="gpqa_diamond", metric="exact_match,flexible-extract")
    shared = {
        "doc_id": 7,
        "doc": {"Question": "Q", "answer": "A"},
        "target": "A",
        "resps": [["The answer is A"]],
    }
    batch = {
        "results": {
            "gpqa_diamond": {
                "exact_match,flexible-extract": 1.0,
                "exact_match,strict-match": 0.0,
            }
        },
        # lm-eval logs one row per filter pipeline for the same document.
        "samples": {
            "gpqa_diamond": [
                {
                    **shared,
                    "filter": "strict-match",
                    "filtered_resps": ["The answer is A"],
                    "exact_match,strict-match": 0.0,
                },
                {
                    **shared,
                    "filter": "flexible-extract",
                    "filtered_resps": ["A"],
                    "exact_match,flexible-extract": 1.0,
                },
            ]
        },
    }

    rows = checkpoint_task_result(
        task=task,
        generation_seed=42,
        expected_generation_seeds=[42],
        batch=batch,
        aggregate={},
        aggregate_path=tmp_path / "aggregate.json",
        samples_dir=tmp_path / "samples",
        log_samples=True,
    )

    assert len(rows) == 1
    assert rows[0]["metric"] == "exact_match,flexible-extract"
    assert rows[0]["extracted_answer"] == "A"
    assert rows[0]["correct"] == 1


def test_repeated_checkpoint_preserves_question_and_attempt_identity(tmp_path: Path):
    task = EvalTask(name="gpqa", metric="exact_match,flexible-extract")
    aggregate: dict[str, dict[str, float]] = {}
    for seed, correct in ((42, 1), (1234, 0), (4158, 1)):
        batch = {
            "results": {"gpqa": {"exact_match,flexible-extract": correct}},
            "samples": {
                "gpqa": [
                    {
                        "doc_id": 7,
                        "doc": {"Question": "Q", "answer": "A"},
                        "arguments": [["Q\nAnswer:", {"max_gen_toks": 16384}]],
                        "target": "A",
                        "resps": [["Answer: A"]],
                        "filtered_resps": ["A"],
                        "exact_match,flexible-extract": correct,
                    }
                ]
            },
        }
        checkpoint_task_result(
            task=task,
            generation_seed=seed,
            expected_generation_seeds=[42, 1234, 4158],
            batch=batch,
            aggregate=aggregate,
            aggregate_path=tmp_path / "aggregate.json",
            samples_dir=tmp_path / "samples",
            progress_path=tmp_path / "seed_progress.json",
            log_samples=True,
        )

    rows = [
        json.loads(line)
        for line in (tmp_path / "samples/gpqa.jsonl").read_text().splitlines()
    ]
    assert len({row["sample_uid"] for row in rows}) == 1
    assert len({row["attempt_uid"] for row in rows}) == 3
    assert {row["generation_seed"] for row in rows} == {42, 1234, 4158}
    assert all(row["source_doc"] == {"Question": "Q", "answer": "A"} for row in rows)
    assert aggregate["gpqa"]["pass_at_1_seed_42"] == 1.0
    assert aggregate["gpqa"]["pass_at_1_seed_1234"] == 0.0
    assert aggregate["gpqa"]["mean_pass_at_1"] == pytest.approx(2 / 3)
    progress = json.loads((tmp_path / "seed_progress.json").read_text())
    assert progress["tasks"]["gpqa"] == [42, 1234, 4158]


def test_repeated_checkpoint_survives_unicode_line_separators(tmp_path: Path):
    """Regression: generated text carrying U+2028/U+2029/U+0085 must not break
    the per-seed sample re-read.

    Samples are written with ``json.dumps(ensure_ascii=False) + "\\n"``, which
    emits these Unicode line-break characters literally (they are legal inside
    JSON strings). A re-read that split on ``str.splitlines()`` shredded a single
    record into fragments and raised ``JSONDecodeError: Unterminated string``
    when the next seed's checkpoint merged the existing file. The read must split
    on the literal ``"\\n"`` terminator only.
    """
    task = EvalTask(name="gpqa", metric="exact_match,flexible-extract")
    # U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR, U+0085 NEL — all split by
    # str.splitlines() but never by str.split("\n").
    poisoned = "Step 1.\u2028Step 2.\u2029Done.\u0085Answer: A"
    aggregate: dict[str, dict[str, float]] = {}
    for seed, correct in ((42, 1), (1234, 0)):
        batch = {
            "results": {"gpqa": {"exact_match,flexible-extract": correct}},
            "samples": {
                "gpqa": [
                    {
                        "doc_id": 7,
                        "doc": {"Question": "Q", "answer": "A"},
                        "arguments": [["Q\nAnswer:", {"max_gen_toks": 16384}]],
                        "target": "A",
                        "resps": [[poisoned]],
                        "filtered_resps": ["A"],
                        "exact_match,flexible-extract": correct,
                    }
                ]
            },
        }
        # Before the fix, the seed-1234 call raised JSONDecodeError here while
        # re-reading the seed-42 sample file.
        checkpoint_task_result(
            task=task,
            generation_seed=seed,
            expected_generation_seeds=[42, 1234, 4158],
            batch=batch,
            aggregate=aggregate,
            aggregate_path=tmp_path / "aggregate.json",
            samples_dir=tmp_path / "samples",
            progress_path=tmp_path / "seed_progress.json",
            log_samples=True,
        )

    raw = (tmp_path / "samples/gpqa.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.split("\n") if line.strip()]
    assert {row["generation_seed"] for row in rows} == {42, 1234}
    assert any("\u2028" in json.dumps(row, ensure_ascii=False) for row in rows)
    progress = json.loads((tmp_path / "seed_progress.json").read_text())
    assert progress["tasks"]["gpqa"] == [42, 1234]


def test_repeated_checkpoint_rejects_unexpected_seed(tmp_path: Path):
    task = EvalTask(name="gpqa", metric="exact_match,flexible-extract")
    with pytest.raises(ValueError, match="unexpected generation seed"):
        checkpoint_task_result(
            task=task,
            generation_seed=99,
            expected_generation_seeds=[42, 1234, 4158],
            batch={
                "results": {"gpqa": {"exact_match,flexible-extract": 1.0}},
                "samples": {"gpqa": []},
            },
            aggregate={},
            aggregate_path=tmp_path / "aggregate.json",
            samples_dir=tmp_path / "samples",
            progress_path=tmp_path / "seed_progress.json",
            log_samples=True,
        )


def test_run_static_eval_resumes_completed_generation_seeds(monkeypatch, tmp_path):
    cfg = PipelineConfig()
    cfg.eval.tasks = [
        EvalTask(name="gpqa", metric="exact_match,flexible-extract", limit=None)
    ]
    cfg.eval.generation_seeds = [42, 1234]
    seen_completed = []

    def fake_evaluate(
        model_path,
        config,
        tasks,
        *,
        log_samples,
        completed_task_seeds,
        on_task_complete,
    ):
        seen_completed.append(set(completed_task_seeds))
        for seed in config.eval.generation_seeds:
            if ("gpqa", seed) in completed_task_seeds:
                continue
            on_task_complete(
                tasks[0],
                seed,
                {
                    "results": {"gpqa": {"exact_match,flexible-extract": 1.0}},
                    "samples": {
                        "gpqa": [
                            {
                                "doc_id": 0,
                                "exact_match,flexible-extract": 1.0,
                                "resps": ["A"],
                                "filtered_resps": ["A"],
                            }
                        ]
                    },
                },
            )
            break
        return {}

    monkeypatch.setattr("pipeline.evalsuite.static.evaluate_tasks", fake_evaluate)

    first = run_static_eval(cfg, "/model", tmp_path)
    second = run_static_eval(cfg, "/model", tmp_path)

    assert seen_completed == [set(), {("gpqa", 42)}]
    assert first["aggregate"]["gpqa"]["pass_at_1_seed_42"] == 1.0
    assert second["sample_counts"] == {"gpqa": 2}


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
    health = json.loads((tmp_path / "generation_health" / "mmlu_pro.json").read_text())
    assert health["samples"] == 0
    assert health["not_applicable_count"] == 1


def test_nested_singleton_text_response_is_generation(tmp_path: Path):
    task = EvalTask(name="ifeval", metric="exact_match,strict-match")
    batch = {
        "results": {"ifeval": {"exact_match,strict-match": 0.0}},
        "samples": {
            "ifeval": [
                {
                    "doc_id": 0,
                    "exact_match,strict-match": 0.0,
                    "resps": [["coherent generated text"]],
                    "filtered_resps": [[None]],
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

    assert rows[0]["response"] == "coherent generated text"
    assert rows[0]["health"]["applicable"] is True
