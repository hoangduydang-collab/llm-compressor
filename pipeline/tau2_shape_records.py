#!/usr/bin/env python3
"""tau2-bench results.json -> generic TurnRecord JSONL (benchmarks shape-capture input).

WHY THIS LIVES HERE
-------------------
The benchmarks repo owns the shape-capture seam (`performance/shapes/`) and supports
four sources: Inspect `.eval`, mini-swe-agent, Harbor ATIF, and a **generic TurnRecord
JSONL** escape hatch. tau2-bench is not one of the four, but its run output already
carries everything a shape needs — so no new adapter belongs in that repo. This script
converts OUR calibration artifacts into the generic JSONL it already accepts:

    python -m pipeline.tau2_shape_records \
        --results /mnt/nfs/hoangduy/tau2-bench/data/simulations/<run>/results.json \
        --out /tmp/m3_tau2_turns.jsonl
    # then, in the benchmarks repo:
    python -m performance.shapes.extract --turns /tmp/m3_tau2_turns.jsonl \
        --archetype tau2-telecom

It stays on this side because it reads quant-run evidence we own; only the resulting
FROZEN shape is committed to the benchmarks repo.

MAPPING DECISIONS (documented because they change the numbers)
-------------------------------------------------------------
* **A turn is an assistant message that has `usage`.** Every tau2 session opens with a
  scripted greeting ("Hi! How can I help you today?") that is identical across sessions,
  makes no tool calls and has no `usage` — it is a seeded message, not a model call.
  Including it would inflate the turn-count distribution by exactly one per session.
* **`post_delay_ms` is the tool-execution gap only** — `ts(next tool message) -
  ts(this assistant)` — mirroring the mini-swe-agent adapter's convention in the
  benchmarks repo. Gaps before a *user* message are deliberately NOT counted: tau2's
  user is an LLM simulator (DeepSeek-V4-Pro over a gateway), so that gap measures
  simulator latency, not how long a human takes to reply. Counting it would inflate
  inter-turn delay with something no production workload has.
* **`cached_input_tokens` is always None.** vLLM 0.24 does not report
  `prompt_tokens_details.cached_tokens`, so prefix reuse has to be inferred downstream
  from input-token structure instead of read directly.
* `latency_ms` comes from `generation_time_seconds` when present (it often is not);
  `ttft_ms` is never available from tau2 output.

Nothing from the run's `info` block is emitted — it embeds the gateway API key used for
the user simulator.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

ARCHETYPE_DEFAULT = "tau2-telecom"


def _parse_ts(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _tool_names(msg: Dict[str, Any]) -> List[str]:
    out = []
    for tc in msg.get("tool_calls") or []:
        name = tc.get("name") or (tc.get("function") or {}).get("name")
        if name:
            out.append(str(name))
    return out


def _post_delay_ms(messages: List[Dict[str, Any]], idx: int) -> Optional[float]:
    """Tool-exec gap after the assistant message at `idx`, in ms.

    Only counted when the very next message is a tool result — see MAPPING DECISIONS.
    """
    if idx + 1 >= len(messages):
        return None
    nxt = messages[idx + 1]
    if nxt.get("role") != "tool":
        return None
    a, b = _parse_ts(messages[idx].get("timestamp")), _parse_ts(nxt.get("timestamp"))
    if a is None or b is None:
        return None
    delta = (b - a).total_seconds() * 1000.0
    return delta if delta >= 0 else None


def session_turns(sim: Dict[str, Any], archetype: str = ARCHETYPE_DEFAULT) -> Iterator[Dict[str, Any]]:
    """Yield TurnRecord-shaped dicts for one tau2 simulation."""
    session_id = str(sim.get("id") or sim.get("task_id") or "")
    messages = sim.get("messages") or []
    turn_index = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        usage = msg.get("usage")
        if not usage:                      # scripted greeting, not a model call
            continue
        gen_s = msg.get("generation_time_seconds")
        yield {
            "session_id": session_id,
            "turn_index": turn_index,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "cached_input_tokens": None,
            "ttft_ms": None,
            "latency_ms": float(gen_s) * 1000.0 if isinstance(gen_s, (int, float)) else None,
            "post_delay_ms": _post_delay_ms(messages, i),
            "tools": _tool_names(msg),
            "archetype": archetype,
        }
        turn_index += 1


def convert(results_path: str, archetype: str = ARCHETYPE_DEFAULT) -> List[Dict[str, Any]]:
    with open(results_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    rows: List[Dict[str, Any]] = []
    for sim in doc.get("simulations") or []:
        rows.extend(session_turns(sim, archetype))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="tau2-bench results.json")
    ap.add_argument("--out", required=True, help="TurnRecord JSONL to write")
    ap.add_argument("--archetype", default=ARCHETYPE_DEFAULT)
    args = ap.parse_args(argv)

    rows = convert(args.results, args.archetype)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    sessions = len({r["session_id"] for r in rows})
    with_delay = sum(1 for r in rows if r["post_delay_ms"] is not None)
    print("[tau2-shape] wrote %s" % args.out)
    print("[tau2-shape] %d turns across %d sessions; tool-exec delay on %d turns"
          % (len(rows), sessions, with_delay))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
