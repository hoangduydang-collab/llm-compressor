"""CPU-only tests for the exact MiniMax-M3 empty-output replay contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.m3_empty_output_replay import (
    REPLAY_CAPS,
    load_replay_attempt,
    postprocess_stages,
    run_controls,
)


ROW = {
    "attempt_uid": "8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878",
    "task": "mmlu_pro",
    "subtask": "mmlu_pro_economics",
    "doc_id": 45,
    "generation_seed": 1234,
    "response": "",
    "generation_arguments": [[
        "rendered prompt",
        {
            "until": ["Question:"],
            "max_gen_toks": 256,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 1234,
        },
    ]],
}


def _write_rows(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_replay_attempt_returns_exact_normalized_request(tmp_path: Path):
    path = tmp_path / "attempts.jsonl"
    _write_rows(path, [ROW])

    attempt = load_replay_attempt(path, ROW["attempt_uid"])

    assert attempt.prompt == "rendered prompt"
    assert attempt.prompt_sha256 == hashlib.sha256(b"rendered prompt").hexdigest()
    assert attempt.generation_kwargs["max_gen_toks"] == 256
    assert attempt.source_row == ROW


@pytest.mark.parametrize("matching_rows", [0, 2])
def test_load_replay_attempt_requires_exactly_one_matching_uid(
    tmp_path: Path,
    matching_rows: int,
):
    path = tmp_path / "attempts.jsonl"
    _write_rows(path, [ROW] * matching_rows)

    with pytest.raises(ValueError, match="exactly one"):
        load_replay_attempt(path, ROW["attempt_uid"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "mmlu"),
        ("subtask", "mmlu_pro_history"),
        ("doc_id", 46),
        ("generation_seed", 4321),
    ],
)
def test_load_replay_attempt_rejects_wrong_attempt_identity(
    tmp_path: Path,
    field: str,
    value: object,
):
    path = tmp_path / "attempts.jsonl"
    row = copy.deepcopy(ROW)
    row[field] = value
    _write_rows(path, [row])

    with pytest.raises(ValueError, match=field):
        load_replay_attempt(path, ROW["attempt_uid"])


def test_load_replay_attempt_requires_original_empty_response(tmp_path: Path):
    path = tmp_path / "attempts.jsonl"
    row = copy.deepcopy(ROW)
    row["response"] = "generated text"
    _write_rows(path, [row])

    with pytest.raises(ValueError, match="response"):
        load_replay_attempt(path, ROW["attempt_uid"])


@pytest.mark.parametrize(
    "generation_arguments",
    [
        None,
        [],
        [["rendered prompt", ROW["generation_arguments"][0][1]]] * 2,
        [["rendered prompt"]],
        [["rendered prompt", ROW["generation_arguments"][0][1], "extra"]],
        [[123, ROW["generation_arguments"][0][1]]],
        [["rendered prompt", None]],
    ],
)
def test_load_replay_attempt_rejects_malformed_generation_arguments(
    tmp_path: Path,
    generation_arguments: object,
):
    path = tmp_path / "attempts.jsonl"
    row = copy.deepcopy(ROW)
    row["generation_arguments"] = generation_arguments
    _write_rows(path, [row])

    with pytest.raises(ValueError, match="generation_arguments"):
        load_replay_attempt(path, ROW["attempt_uid"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("until", ["Answer:"]),
        ("max_gen_toks", 255),
        ("do_sample", False),
        ("temperature", 0.9),
        ("top_p", 1.0),
        ("seed", 1235),
    ],
)
def test_load_replay_attempt_rejects_changed_pinned_generation_setting(
    tmp_path: Path,
    field: str,
    value: object,
):
    path = tmp_path / "attempts.jsonl"
    row = copy.deepcopy(ROW)
    row["generation_arguments"][0][1][field] = value
    _write_rows(path, [row])

    with pytest.raises(ValueError, match="generation"):
        load_replay_attempt(path, ROW["attempt_uid"])


def test_postprocess_stages_records_thinking_and_task_stop_boundaries():
    stages = postprocess_stages(
        "<mm:think>reasoning</mm:think>Question: hidden",
        think_end_token="</mm:think>",
        until=["Question:"],
    )

    assert stages == {
        "raw_text": "<mm:think>reasoning</mm:think>Question: hidden",
        "after_thinking": "Question: hidden",
        "after_task_stops": "",
        "thinking_marker_present": True,
        "matched_stop": "Question:",
    }


def test_postprocess_stages_uses_last_marker_lstrip_and_sequential_stops():
    stages = postprocess_stages(
        "old</mm:think>discard</mm:think>  answer END ignored STOP tail",
        think_end_token="</mm:think>",
        until=["", "STOP", "END"],
    )

    assert stages["after_thinking"] == "answer END ignored STOP tail"
    assert stages["after_task_stops"] == "answer "
    assert stages["matched_stop"] == "END"
    assert stages["thinking_marker_present"] is True


def test_postprocess_stages_preserves_text_when_no_markers_match():
    stages = postprocess_stages(
        "  plain answer",
        think_end_token="</mm:think>",
        until=["Question:"],
    )

    assert stages["after_thinking"] == "plain answer"
    assert stages["after_task_stops"] == "plain answer"
    assert stages["thinking_marker_present"] is False
    assert stages["matched_stop"] is None


def test_run_controls_uses_fixed_cap_order_and_reports_postprocessing(
    tmp_path: Path,
):
    path = tmp_path / "attempts.jsonl"
    _write_rows(path, [ROW])
    attempt = load_replay_attempt(path, ROW["attempt_uid"])
    calls = []

    def generate(replay_attempt, cap):
        calls.append((replay_attempt, cap))
        return {"raw_text": f"answer {cap}Question: hidden", "tokens": cap // 2}

    controls = run_controls(attempt, generate)

    assert [cap for _, cap in calls] == list(REPLAY_CAPS) == [256, 16384]
    assert all(replay_attempt is attempt for replay_attempt, _ in calls)
    assert [control["max_gen_toks"] for control in controls] == [256, 16384]
    assert [control["tokens"] for control in controls] == [128, 8192]
    assert [
        control["postprocessing"]["after_task_stops"] for control in controls
    ] == ["answer 256", "answer 16384"]
