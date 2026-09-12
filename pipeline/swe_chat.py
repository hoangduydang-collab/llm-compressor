"""Strict adapter for the published SALT-NLP/SWE-chat conversations table."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class NormalizedSweSession:
    messages: list[dict[str, Any]]
    message_turns: list[int]
    anchor_message_indices: list[int]


def _text(value: Any, field: str, turn: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"SWE-chat turn {turn} {field} must be a string")
    return value


def _append_text(message: dict[str, Any], field: str, content: str) -> None:
    if not content:
        return
    current = message.get(field)
    message[field] = f"{current}\n{content}" if current else content


def _is_substantive_assistant(message: dict[str, Any]) -> bool:
    if message.get("tool_calls"):
        return True
    text = "\n".join(
        str(message.get(field) or "") for field in ("reasoning_content", "content")
    ).strip()
    return len(text) >= 32 or "```" in text or "\n" in text


def normalize_swe_chat_session(rows: Iterable[dict[str, Any]]) -> NormalizedSweSession:
    """Convert one ordered normalized SWE-chat session to tokenizer messages."""
    ordered = sorted(rows, key=lambda row: row.get("turn_number", -1))
    messages: list[dict[str, Any]] = []
    message_turns: list[int] = []
    seen_turns: set[int] = set()
    open_calls: set[str] = set()

    for row in ordered:
        turn = row.get("turn_number")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
            raise ValueError("SWE-chat turn_number must be a non-negative integer")
        if turn in seen_turns:
            raise ValueError(f"SWE-chat session has duplicate turn_number {turn}")
        seen_turns.add(turn)
        role = row.get("role")

        if role == "metadata":
            continue
        if role == "user":
            content = _text(row.get("content"), "content", turn)
            messages.append({"role": "user", "content": content})
            message_turns.append(turn)
            continue
        if role == "assistant":
            turn_type = row.get("turn_type")
            if turn_type not in {"assistant_thinking", "assistant_response"}:
                raise ValueError(
                    f"SWE-chat assistant turn {turn} has unsupported "
                    f"turn_type {turn_type!r}"
                )
            content = _text(row.get("content"), "content", turn)
            if messages and messages[-1].get("role") == "assistant":
                message = messages[-1]
                message_turns[-1] = turn
            else:
                message = {"role": "assistant", "content": ""}
                messages.append(message)
                message_turns.append(turn)
            field = (
                "reasoning_content" if turn_type == "assistant_thinking" else "content"
            )
            _append_text(message, field, content)
            continue
        if role == "tool_use":
            tool_name = row.get("tool_name")
            call_id = row.get("tool_call_id")
            raw_arguments = row.get("tool_input_json")
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError(f"SWE-chat tool_use turn {turn} requires tool_name")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"SWE-chat tool_use turn {turn} requires tool_call_id")
            if call_id in open_calls:
                raise ValueError(f"SWE-chat duplicate tool_call_id {call_id!r}")
            if not isinstance(raw_arguments, str):
                raise ValueError(
                    f"SWE-chat tool_use turn {turn} tool_input_json must "
                    "be a JSON string"
                )
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"SWE-chat tool_use turn {turn} has malformed tool_input_json"
                ) from exc
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"SWE-chat tool_use turn {turn} arguments must decode "
                    "to a dictionary"
                )
            if messages and messages[-1].get("role") == "assistant":
                message = messages[-1]
                message_turns[-1] = turn
            else:
                message = {"role": "assistant", "content": ""}
                messages.append(message)
                message_turns.append(turn)
            message.setdefault("tool_calls", []).append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }
            )
            open_calls.add(call_id)
            continue
        if role == "tool_result":
            call_id = row.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(
                    f"SWE-chat tool_result turn {turn} requires tool_call_id"
                )
            if call_id not in open_calls:
                raise ValueError(
                    f"SWE-chat tool_result turn {turn} has no matching "
                    f"tool_use for {call_id!r}"
                )
            messages.append(
                {
                    "role": "tool",
                    "content": _text(row.get("content"), "content", turn),
                    "tool_call_id": call_id,
                }
            )
            message_turns.append(turn)
            open_calls.remove(call_id)
            continue
        raise ValueError(f"SWE-chat turn {turn} has unknown role {role!r}")

    if open_calls:
        raise ValueError(
            f"SWE-chat session has tool calls without results: {sorted(open_calls)}"
        )
    anchors = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" and _is_substantive_assistant(message)
    ]
    return NormalizedSweSession(messages, message_turns, anchors)
