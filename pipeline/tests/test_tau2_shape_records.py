"""Unit tests for the tau2 -> TurnRecord JSONL shape converter.

Every mapping decision that changes the derived shape numbers is pinned here.
"""

import json

import pytest

from pipeline.tau2_shape_records import convert, session_turns

GREETING = {
    "role": "assistant",
    "content": "Hi! How can I help you today?",
    "turn_idx": 0,
    "timestamp": "2026-07-22T09:17:20.000000",
    "usage": None,
    "tool_calls": None,
}


def _sim():
    """One session: greeting, a tool-calling turn, a tool result, a user-facing turn."""
    return {
        "id": "sess-1",
        "messages": [
            GREETING,
            {"role": "assistant", "turn_idx": 1, "timestamp": "2026-07-22T09:17:22.000000",
             "usage": {"prompt_tokens": 7347, "completion_tokens": 81},
             "generation_time_seconds": 1.5,
             "tool_calls": [{"id": "c1", "name": "get_customer_by_phone", "arguments": {}}]},
            {"role": "tool", "id": "c1", "turn_idx": 1,
             "timestamp": "2026-07-22T09:17:22.250000"},
            {"role": "assistant", "turn_idx": 2, "timestamp": "2026-07-22T09:17:24.000000",
             "usage": {"prompt_tokens": 7500, "completion_tokens": 40},
             "tool_calls": None},
            {"role": "user", "turn_idx": 2, "timestamp": "2026-07-22T09:17:30.000000"},
        ],
    }


def test_scripted_greeting_is_not_a_turn():
    """The greeting has no usage — it is a seeded message, not a model call.

    Counting it would inflate turns-per-session by exactly one in every session.
    """
    rows = list(session_turns(_sim()))
    assert len(rows) == 2
    assert [r["turn_index"] for r in rows] == [0, 1]
    assert all(r["input_tokens"] is not None for r in rows)


def test_tokens_and_latency_mapped():
    rows = list(session_turns(_sim()))
    assert (rows[0]["input_tokens"], rows[0]["output_tokens"]) == (7347, 81)
    assert rows[0]["latency_ms"] == pytest.approx(1500.0)
    # tau2 output carries no TTFT, and vLLM 0.24 reports no cached prompt tokens
    assert rows[0]["ttft_ms"] is None
    assert rows[0]["cached_input_tokens"] is None
    # generation_time_seconds absent on the second turn
    assert rows[1]["latency_ms"] is None


def test_tool_exec_gap_counted():
    rows = list(session_turns(_sim()))
    assert rows[0]["post_delay_ms"] == pytest.approx(250.0)
    assert rows[0]["tools"] == ["get_customer_by_phone"]


def test_user_simulator_gap_not_counted():
    """The 6s gap before the user message is LLM-simulator latency, not human think
    time — counting it would inflate inter-turn delay with something no production
    workload has."""
    rows = list(session_turns(_sim()))
    assert rows[1]["post_delay_ms"] is None
    assert rows[1]["tools"] == []


def test_negative_and_unparseable_timestamps_drop_to_none():
    sim = _sim()
    sim["messages"][2]["timestamp"] = "2026-07-22T09:17:21.000000"   # before the assistant
    assert list(session_turns(sim))[0]["post_delay_ms"] is None
    sim["messages"][2]["timestamp"] = "not-a-timestamp"
    assert list(session_turns(sim))[0]["post_delay_ms"] is None


def test_convert_reads_all_simulations(tmp_path):
    doc = {"simulations": [_sim(), dict(_sim(), id="sess-2")]}
    p = tmp_path / "results.json"
    p.write_text(json.dumps(doc))
    rows = convert(str(p))
    assert len(rows) == 4
    assert {r["session_id"] for r in rows} == {"sess-1", "sess-2"}
    assert {r["archetype"] for r in rows} == {"tau2-telecom"}


def test_no_run_info_is_emitted(tmp_path):
    """The run's info block embeds the gateway API key — it must never reach output."""
    doc = {"info": {"user_info": {"llm_args": {"api_key": "SECRET-DO-NOT-EMIT"}}},
           "simulations": [_sim()]}
    p = tmp_path / "results.json"
    p.write_text(json.dumps(doc))
    assert "SECRET-DO-NOT-EMIT" not in json.dumps(convert(str(p)))
