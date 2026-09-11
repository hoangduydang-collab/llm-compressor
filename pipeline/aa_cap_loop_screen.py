#!/usr/bin/env python3
"""Zero-GPU loop screen for AA GPQA cap-hit traces in nemo-evaluator cache.db.

Classifies each completion that hit the output cap as degenerate loop vs
genuine unfinished reasoning.

Hard gate (label ``loop`` iff any fire):

* zlib of the last 8k chars ≤ 0.08 — ``sampling_probe`` loops compress to
  ~0.004; the 2026-09-11 GLM-5.3 cap-hit median is 0.31.
* ``pipeline.evalsuite.health._periodic_suffix`` on the tail's whitespace
  tokens and last 4096 chars (period 1–16, ≥4 repeats, ≥16 tokens).

``sample_output_check.judge`` (distinct-4gram < 0.30) is **reported** as
``judge_flags`` but is not the label. That threshold was measured on
~100-char M3 collapses and over-fires on 131k-token GPQA CoT.

Usage::

    python -m pipeline.aa_cap_loop_screen \\
        --json-out screen.json \\
        --arm ours:131072:/path/to/cache.db \\
        --arm phala:131072:/path/to/cache.db
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

CAP_AA = 131_072
TAIL_CHARS = 32_000
CHAR_PERIOD_WINDOW = 4_096
TAIL_PREVIEW = 240

try:
    from pipeline.sample_output_check import degeneracy_report, judge
except ImportError:  # reconstructed package on the cluster job
    from sample_output_check import degeneracy_report, judge  # type: ignore


def _health_mod():
    path = Path(__file__).resolve().parent / "evalsuite" / "health.py"
    spec = importlib.util.spec_from_file_location("_aa_cap_health", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load health detectors from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_HEALTH = None


def _health():
    global _HEALTH
    if _HEALTH is None:
        _HEALTH = _health_mod()
    return _HEALTH


def assistant_text(convo: list) -> str:
    """Visible + thinking text of the assistant turn, without double-counting."""
    if not convo or len(convo) < 2 or not isinstance(convo[1], dict):
        return ""
    msg = convo[1]
    reasoning = str(msg.get("reasoning_content") or msg.get("reasoning") or "")
    content = str(msg.get("content") or "")
    if reasoning and content:
        if content in reasoning:
            return reasoning
        if reasoning in content:
            return content
        return reasoning + "\n" + content
    return reasoning or content


def _zlib_ratio(text: str) -> float | None:
    """sampling_probe._rep_ratio: loops compress to ~0.004."""
    if len(text) < 200:
        return None
    raw = text.encode("utf-8", "ignore")
    return round(len(zlib.compress(raw, 6)) / max(1, len(raw)), 4)


def _repetition_problems(report: dict) -> list[str]:
    return [p for p in judge(report) if "too short" not in p]


def classify_trace(text: str) -> dict[str, Any]:
    """Label one completion. Never returns the full text (traces are huge)."""
    health = _health()
    full_report = degeneracy_report(text)
    tail = text[-TAIL_CHARS:] if text else ""
    tail_report = degeneracy_report(tail)
    full_problems = _repetition_problems(full_report)
    tail_problems = _repetition_problems(tail_report)

    words = tail.split()
    periodic_word, period_word, repeated_word = (
        health._periodic_suffix(words) if words else (False, None, 0)
    )
    chars = list(tail[-CHAR_PERIOD_WINDOW:])
    periodic_char, period_char, repeated_char = (
        health._periodic_suffix(chars) if len(chars) >= 16 else (False, None, 0)
    )

    zlib_8k = _zlib_ratio(text[-8000:] if text else "")
    # Hard loop gate: sampling_probe loops compress to ~0.004; health.py
    # periodic_suffix is the M3 token-period detector. Do NOT use
    # distinct-4gram < 0.30 as the label — that threshold was measured on
    # ~100-char collapses and over-fires on 131k-token GPQA CoT that reuses
    # notation (formal-run median zlib 0.31, periodic_suffix 0/120).
    tight_zlib = zlib_8k is not None and zlib_8k <= 0.08
    tight_loop = bool(tight_zlib or periodic_word or periodic_char)

    reasons: list[str] = []
    if tight_zlib:
        reasons.append(f"zlib_tail_8k={zlib_8k} <= 0.08 (sampling_probe loop)")
    if periodic_word:
        reasons.append(
            f"tail periodic word period={period_word} repeated_tokens={repeated_word}"
        )
    if periodic_char:
        reasons.append(
            f"tail periodic char period={period_char} repeated_tokens={repeated_char}"
        )
    judge_flags = list(full_problems)
    for problem in tail_problems:
        tagged = f"tail: {problem}"
        if tagged not in judge_flags and problem not in judge_flags:
            judge_flags.append(tagged)

    word_ids = words
    rep4 = health._repeated_ngram_fraction(word_ids, 4) if word_ids else None
    return {
        "label": "loop" if tight_loop else "unfinished",
        "reasons": reasons,
        "judge_flags": judge_flags,
        "tail_fired": bool(tail_problems or periodic_word or periodic_char),
        "full_distinct_4gram": full_report["distinct_4gram_ratio"],
        "tail_distinct_4gram": tail_report["distinct_4gram_ratio"],
        "full_unique_word_ratio": full_report["unique_word_ratio"],
        "tail_unique_word_ratio": tail_report["unique_word_ratio"],
        "tail_repeated_4gram_fraction": rep4,
        "zlib_tail_8k": zlib_8k,
        "periodic_word": periodic_word,
        "periodic_char": periodic_char,
        "loop_period_word": period_word,
        "loop_period_char": period_char,
        "tail_preview": (tail[-TAIL_PREVIEW:] if tail else ""),
        "n_chars": len(text),
        "n_words": full_report["words"],
    }


def load_cache_records(path: str | Path) -> list[dict]:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        tables = [
            row[0]
            for row in con.execute(
                "select name from sqlite_master where type='table'"
            )
        ]
        if "Cache" not in tables:
            return []
        records = []
        for (value,) in con.execute("select value from Cache"):
            try:
                data = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and "convo" in data and "score" in data:
                records.append(data)
        return records
    finally:
        con.close()


def _prompt_key(convo: list) -> str:
    if not convo or not isinstance(convo[0], dict):
        return "none"
    prompt = str(convo[0].get("content", ""))
    return hashlib.sha256(prompt.encode("utf-8", "ignore")).hexdigest()[:12]


def _completion_tokens(convo: list) -> int:
    if len(convo) < 2 or not isinstance(convo[1], dict):
        return -1
    usage = convo[1].get("usage") or {}
    try:
        return int(usage.get("completion_tokens", -1))
    except (TypeError, ValueError):
        return -1


def screen_records(records: list[dict], cap: int = CAP_AA) -> list[dict]:
    rows = []
    for data in records:
        convo = data.get("convo") or []
        tokens = _completion_tokens(convo)
        row: dict[str, Any] = {
            "score": float(data.get("score") or 0.0),
            "completion_tokens": tokens,
            "question_key": _prompt_key(convo),
            "capped": tokens >= cap,
        }
        if tokens >= cap:
            row.update(classify_trace(assistant_text(convo)))
        rows.append(row)
    return rows


def summarize_screen(rows: list[dict]) -> dict[str, Any]:
    capped = [row for row in rows if row.get("capped")]
    loops = [row for row in capped if row.get("label") == "loop"]
    unfinished = [row for row in capped if row.get("label") == "unfinished"]
    cap_questions = {row["question_key"] for row in capped}
    loop_questions = {row["question_key"] for row in loops}
    unfinished_questions = {row["question_key"] for row in unfinished}
    return {
        "n_loaded": len(rows),
        "n_capped": len(capped),
        "n_loop": len(loops),
        "n_unfinished": len(unfinished),
        "capped_unique_questions": len(cap_questions),
        "loop_unique_questions": len(loop_questions),
        "unfinished_unique_questions": len(unfinished_questions),
        "loop_rate_among_capped": (
            len(loops) / len(capped) if capped else None
        ),
        "unfinished_question_keys": sorted(unfinished_questions),
        "caps_per_question": sorted(
            Counter(row["question_key"] for row in capped).values()
        ),
    }


def screen_arm(name: str, cap: int, db_path: str | Path) -> dict[str, Any]:
    records = load_cache_records(db_path)
    rows = screen_records(records, cap=cap)
    summary = summarize_screen(rows)
    capped_rows = [
        {k: v for k, v in row.items() if k != "text"}
        for row in rows
        if row.get("capped")
    ]
    return {
        "name": name,
        "cap": cap,
        "db": str(db_path),
        "summary": summary,
        "capped_rows": capped_rows,
    }


def _parse_arm(spec: str) -> tuple[str, int, str]:
    name, cap_s, path = spec.split(":", 2)
    return name, int(cap_s), path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME:CAP:PATH",
        help="e.g. ours:131072:/mnt/.../cache.db (repeatable)",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.arm:
        parser.error("need at least one --arm NAME:CAP:PATH")

    report = {"arms": []}
    for spec in args.arm:
        name, cap, path = _parse_arm(spec)
        print(f"=== arm={name} cap={cap} db={path}", flush=True)
        if not Path(path).is_file():
            print(f"FATAL: missing {path}", flush=True)
            return 10
        arm = screen_arm(name, cap, path)
        report["arms"].append(arm)
        s = arm["summary"]
        print(
            f"VERDICT {name}: loaded={s['n_loaded']} capped={s['n_capped']} "
            f"loop={s['n_loop']} unfinished={s['n_unfinished']} "
            f"loop_rate={s['loop_rate_among_capped']} "
            f"unique_q_capped={s['capped_unique_questions']} "
            f"unique_q_unfinished={s['unfinished_unique_questions']}",
            flush=True,
        )
        if s["unfinished_question_keys"]:
            print(
                f"  unfinished_question_keys={s['unfinished_question_keys']}",
                flush=True,
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"==> wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
