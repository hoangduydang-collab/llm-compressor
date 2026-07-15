"""Pure contracts for replaying one pinned MiniMax-M3 empty output."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPLAY_CAPS = (256, 16384)
EXPECTED_ATTEMPT = {
    "task": "mmlu_pro",
    "subtask": "mmlu_pro_economics",
    "doc_id": 45,
    "generation_seed": 1234,
}
EXPECTED_GENERATION = {
    "until": ["Question:"],
    "max_gen_toks": 256,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "seed": 1234,
}


@dataclass(frozen=True)
class ReplayAttempt:
    """The validated source request for the exact empty-output replay."""

    attempt_uid: str
    prompt: str
    prompt_sha256: str
    generation_kwargs: dict[str, Any]
    source_row: dict[str, Any]


def _same_typed_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_typed_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_value(item, expected_item)
            for item, expected_item in zip(actual, expected)
        )
    return actual == expected


def load_replay_attempt(path: Path, attempt_uid: str) -> ReplayAttempt:
    """Load and validate exactly one row matching the pinned replay request."""

    matches: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            if row.get("attempt_uid") == attempt_uid:
                matches.append(row)

    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one row for attempt_uid={attempt_uid!r}; "
            f"found {len(matches)}"
        )
    row = matches[0]

    for field, expected in EXPECTED_ATTEMPT.items():
        if not _same_typed_value(row.get(field), expected):
            raise ValueError(
                f"unexpected {field}: expected {expected!r}, got {row.get(field)!r}"
            )
    if row.get("response") != "":
        raise ValueError("replay source response must be exactly empty")

    arguments = row.get("generation_arguments")
    if not isinstance(arguments, list) or len(arguments) != 1:
        raise ValueError("generation_arguments must contain exactly one request")
    request = arguments[0]
    if not isinstance(request, list) or len(request) != 2:
        raise ValueError("generation_arguments request must be [prompt, kwargs]")
    prompt, generation_kwargs = request
    if not isinstance(prompt, str) or not isinstance(generation_kwargs, dict):
        raise ValueError("generation_arguments request must be [str, dict]")
    if not _same_typed_value(generation_kwargs, EXPECTED_GENERATION):
        raise ValueError(
            "generation kwargs do not match the pinned r4.5 smoke settings"
        )

    return ReplayAttempt(
        attempt_uid=attempt_uid,
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        generation_kwargs=generation_kwargs,
        source_row=row,
    )


def postprocess_stages(
    raw_text: str,
    *,
    think_end_token: str,
    until: list[str],
) -> dict[str, Any]:
    """Expose the thinking-removal and sequential task-stop stages."""

    thinking_marker_present = bool(think_end_token) and think_end_token in raw_text
    after_thinking = (
        raw_text.split(think_end_token)[-1] if think_end_token else raw_text
    ).lstrip()
    after_task_stops = after_thinking
    matched_stop = None
    for stop in until:
        if stop and stop in after_task_stops:
            after_task_stops = after_task_stops.split(stop)[0]
            matched_stop = stop
    return {
        "raw_text": raw_text,
        "after_thinking": after_thinking,
        "after_task_stops": after_task_stops,
        "thinking_marker_present": thinking_marker_present,
        "matched_stop": matched_stop,
    }


def run_controls(
    attempt: ReplayAttempt,
    generate: Callable[[ReplayAttempt, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate both fixed-cap controls and retain each processing stage."""

    controls = []
    for cap in REPLAY_CAPS:
        completion = generate(attempt, cap)
        controls.append(
            {
                "max_gen_toks": cap,
                **completion,
                "postprocessing": postprocess_stages(
                    completion["raw_text"],
                    think_end_token="</mm:think>",
                    until=attempt.generation_kwargs["until"],
                ),
            }
        )
    return controls
