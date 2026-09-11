#!/usr/bin/env python3
"""Overthinking errors on AA GPQA, identical to Lotfi et al. 2026.

Paper: arXiv:2606.00206 §4.2 and Table 12 (GPT-5 judge prompt).

On an *incorrect* attempt, assign exactly one category. The judge is
instructed to choose overthinking whenever the gold answer appears in the
chain-of-thought but is not produced as the final answer:

    If at any point in the chain-of-thought, the model finds the correct
    (gold) answer but does not commit to it as the final answer, choose
    overthinking (A).

For GPQA Diamond the gold answer is a letter A–D under that attempt's
shuffle. "Finds the gold answer" is operationalized with the *same*
high-precision committed-answer regexes NVIDIA ``gpqa_diamond_aa_v3`` uses
for the final extract (``Answer: X``, ``\\boxed{X}``, ``answer is X``),
applied to the full trace in left-to-right order — not a raw letter mention
inside option restatement.

The other three paper labels (logical / arithmetic / formatting) require
the LLM judge on the residual failures. This module labels those
``not_overthinking`` rather than inventing a second taxonomy.

Gold choice text is taken from official GPQA Diamond (``Idavidrein/gpqa``,
config ``gpqa_diamond``): ``Question`` → ``Correct Answer``, then mapped onto
that attempt's shuffled A/B/C/D. Matching is by normalized question stem,
then by the unordered set of four choice strings if the stem differs. Sibling
inference is only a fallback when the official row cannot be joined.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline.aa_cap_loop_screen import assistant_text, load_cache_records

CAP_AA = 131_072

# AA gpqa_diamond_aa_v3 high-precision extractors (not the loose "C)" /
# trailing-letter fallbacks, which fire on option restatement).
_COMMITTED = [
    re.compile(
        r"(?i)[\*_]{0,2}Answer[\*_]{0,2}\s*:[\s*_]{0,2}\s*([A-D])(?![a-zA-Z0-9])"
    ),
    re.compile(r"\\boxed\{[^}]*([A-D])[^}]*\}"),
    re.compile(r"(?i)answer is \(([A-D])\)"),
    re.compile(r"(?i)(?:the )?answer is ([A-D])(?![a-zA-Z])"),
    re.compile(r"(?i)([A-D])\s+is\s+the\s+correct\s+answer"),
]

_STEM = re.compile(
    r"Answer the following multiple choice question\..*?\n\n(.+?)\n\nA\)",
    re.S,
)
_CHOICES = re.compile(
    r"\nA\)\s*(.*?)\nB\)\s*(.*?)\nC\)\s*(.*?)\nD\)\s*(.*?)\s*$",
    re.S,
)


def parse_mcq_prompt(prompt: str) -> tuple[str, dict[str, str]]:
    stem_m = _STEM.search(prompt)
    choice_m = _CHOICES.search(prompt)
    if not stem_m or not choice_m:
        raise ValueError("prompt is not gpqa_diamond_aa_v3 shaped")
    stem = normalize_text(stem_m.group(1))
    choices = {
        letter: normalize_text(text)
        for letter, text in zip("ABCD", choice_m.groups())
    }
    return stem, choices


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def stem_key(stem: str) -> str:
    return hashlib.sha256(stem.encode("utf-8", "ignore")).hexdigest()[:12]


def committed_letters(text: str) -> list[str]:
    """Gold-finding events, left to right, as the paper's judge would scan CoT."""
    hits: list[tuple[int, str]] = []
    for pattern in _COMMITTED:
        for match in pattern.finditer(text or ""):
            hits.append((match.start(), match.group(1).upper()))
    hits.sort(key=lambda item: item[0])
    return [letter for _, letter in hits]


_UNSET = object()


def categorize_failure(
    *,
    text: str,
    score: float,
    gold_letter: str | None,
    final_letter: object = _UNSET,
    capped: bool = False,
) -> dict[str, Any]:
    """Table 12 priority rule. ``category`` is None when the attempt scored."""
    letters = committed_letters(text)
    final = (
        (letters[-1] if letters else None)
        if final_letter is _UNSET
        else final_letter
    )
    if score >= 1.0:
        return {
            "category": None,
            "committed": letters,
            "final_letter": final,
            "gold_letter": gold_letter,
        }
    gold_seen = bool(gold_letter and gold_letter in letters)
    abandoned = gold_seen and final != gold_letter
    # Stated gold then kept generating into the cap: did not commit as final.
    stated_then_ran_on = gold_seen and capped and score < 1.0
    if abandoned or stated_then_ran_on:
        return {
            "category": "overthinking",
            "committed": letters,
            "final_letter": final,
            "gold_letter": gold_letter,
            "gold_first_index": letters.index(gold_letter) if gold_letter else None,
            "n_committed": len(letters),
        }
    return {
        "category": "not_overthinking",
        "committed": letters,
        "final_letter": final,
        "gold_letter": gold_letter,
    }


def infer_gold_choice_text(records: list[dict]) -> str | None:
    """Gold is the choice text of a score=1 sibling of the same stem."""
    votes: Counter[str] = Counter()
    for rec in records:
        if float(rec.get("score") or 0) < 1.0:
            continue
        letter = (committed_letters(rec.get("text") or "") or [None])[-1]
        if not letter:
            continue
        text = (rec.get("choices") or {}).get(letter)
        if text:
            votes[text] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def index_gpqa_diamond(rows: list[dict]) -> tuple[dict[str, str], dict[frozenset[str], str]]:
    """Official gold text keyed by question stem and by the 4-choice set."""
    by_stem: dict[str, str] = {}
    by_choices: dict[frozenset[str], str] = {}
    for row in rows:
        gold = normalize_text(row["Correct Answer"])
        by_stem[normalize_text(row["Question"])] = gold
        keys = frozenset(
            normalize_text(row[field])
            for field in (
                "Correct Answer",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
            )
        )
        by_choices[keys] = gold
    return by_stem, by_choices


def load_gpqa_diamond_rows(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lookup_official_gold(
    stem: str,
    choices: dict[str, str],
    by_stem: dict[str, str],
    by_choices: dict[frozenset[str], str],
) -> tuple[str | None, str | None]:
    gold = by_stem.get(normalize_text(stem))
    if gold:
        return gold, "gpqa_question"
    gold = by_choices.get(frozenset(normalize_text(v) for v in choices.values()))
    if gold:
        return gold, "gpqa_choice_set"
    return None, None


def _gold_letter(choices: dict[str, str], gold_text: str | None) -> str | None:
    if not gold_text:
        return None
    want = normalize_text(gold_text)
    for letter, text in choices.items():
        if normalize_text(text) == want:
            return letter
    want_cf = want.casefold()
    for letter, text in choices.items():
        if normalize_text(text).casefold() == want_cf:
            return letter
    return None


def _completion_tokens(convo: list) -> int:
    if len(convo) < 2 or not isinstance(convo[1], dict):
        return -1
    try:
        return int((convo[1].get("usage") or {}).get("completion_tokens", -1))
    except (TypeError, ValueError):
        return -1


def records_from_cache(path: str | Path) -> list[dict]:
    out = []
    for data in load_cache_records(path):
        convo = data.get("convo") or []
        prompt = str(convo[0].get("content", "")) if convo else ""
        try:
            stem, choices = parse_mcq_prompt(prompt)
        except ValueError:
            continue
        out.append(
            {
                "stem": stem,
                "stem_key": stem_key(stem),
                "choices": choices,
                "score": float(data.get("score") or 0.0),
                "text": assistant_text(convo),
                "completion_tokens": _completion_tokens(convo),
            }
        )
    return out


def screen_records(
    records: list[dict],
    gpqa_rows: list[dict] | None = None,
) -> dict[str, Any]:
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_stem[rec["stem_key"]].append(rec)

    official_stem: dict[str, str] = {}
    official_choices: dict[frozenset[str], str] = {}
    if gpqa_rows:
        official_stem, official_choices = index_gpqa_diamond(gpqa_rows)

    sibling_gold = {sk: infer_gold_choice_text(recs) for sk, recs in by_stem.items()}
    rows = []
    source_counts: Counter[str] = Counter()
    for rec in records:
        gold_text, source = lookup_official_gold(
            rec["stem"], rec["choices"], official_stem, official_choices
        )
        letter = _gold_letter(rec["choices"], gold_text) if gold_text else None
        if not letter:
            gold_text = sibling_gold[rec["stem_key"]]
            source = "correct_sibling" if gold_text else None
            letter = _gold_letter(rec["choices"], gold_text)
        source_counts[source or "unmatched"] += 1
        cat = categorize_failure(
            text=rec["text"],
            score=rec["score"],
            gold_letter=letter,
            capped=rec["completion_tokens"] >= CAP_AA,
        )
        rec = dict(rec)
        rec.pop("text", None)
        rec["gold_choice_text"] = gold_text
        rec["gold_letter"] = letter
        rec["gold_source"] = source if letter else None
        rec.update(cat)
        rec["capped"] = rec["completion_tokens"] >= CAP_AA
        rows.append(rec)

    fails = [r for r in rows if r["score"] < 1.0]
    labeled = [r for r in fails if r.get("gold_letter")]
    unlabeled = [r for r in fails if not r.get("gold_letter")]
    over = [r for r in labeled if r["category"] == "overthinking"]
    not_over = [r for r in labeled if r["category"] == "not_overthinking"]

    def frac(n, d):
        return round(n / d, 4) if d else None

    return {
        "n_loaded": len(rows),
        "n_fail": len(fails),
        "n_fail_gold_known": len(labeled),
        "n_fail_gold_unknown": len(unlabeled),
        "n_overthinking": len(over),
        "n_not_overthinking": len(not_over),
        "overthinking_rate_among_labeled_fails": frac(len(over), len(labeled)),
        "overthinking_among_capped_fails": sum(
            1 for r in over if r["capped"]
        ),
        "capped_fails_gold_known": sum(1 for r in labeled if r["capped"]),
        "overthinking_among_uncapped_fails": sum(
            1 for r in over if not r["capped"]
        ),
        "uncapped_fails_gold_known": sum(1 for r in labeled if not r["capped"]),
        "gold_source_counts": dict(source_counts),
        "stems_without_gold": sorted({r["stem_key"] for r in unlabeled}),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help="e.g. ours:/mnt/.../cache.db (repeatable)",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--gpqa-gold",
        type=Path,
        default=None,
        help="JSON list of official gpqa_diamond rows "
        "(Question, Correct Answer, Incorrect Answer 1-3)",
    )
    args = parser.parse_args(argv)
    if not args.arm:
        parser.error("need at least one --arm NAME:PATH")

    gpqa_rows = load_gpqa_diamond_rows(args.gpqa_gold) if args.gpqa_gold else None
    report = {
        "source": "arXiv:2606.00206 Table 12 priority rule",
        "gpqa_gold": str(args.gpqa_gold) if args.gpqa_gold else None,
        "n_gpqa_rows": len(gpqa_rows) if gpqa_rows is not None else 0,
        "arms": [],
    }
    for spec in args.arm:
        name, path = spec.split(":", 1)
        print(f"=== arm={name} db={path}", flush=True)
        recs = records_from_cache(path)
        summary = screen_records(recs, gpqa_rows=gpqa_rows)
        compact = {k: v for k, v in summary.items() if k != "rows"}
        compact["name"] = name
        compact["db"] = path
        over_rows = [
            {
                "stem_key": r["stem_key"],
                "capped": r["capped"],
                "completion_tokens": r["completion_tokens"],
                "gold_letter": r["gold_letter"],
                "gold_source": r.get("gold_source"),
                "final_letter": r["final_letter"],
                "committed": r["committed"],
                "gold_first_index": r.get("gold_first_index"),
            }
            for r in summary["rows"]
            if r.get("category") == "overthinking"
        ]
        report["arms"].append({**compact, "overthinking_rows": over_rows})
        print(
            f"VERDICT {name}: loaded={compact['n_loaded']} fail={compact['n_fail']} "
            f"gold_known={compact['n_fail_gold_known']} "
            f"overthinking={compact['n_overthinking']} "
            f"rate={compact['overthinking_rate_among_labeled_fails']} "
            f"over_capped={compact['overthinking_among_capped_fails']}/"
            f"{compact['capped_fails_gold_known']} "
            f"over_uncapped={compact['overthinking_among_uncapped_fails']}/"
            f"{compact['uncapped_fails_gold_known']} "
            f"gold_source={compact.get('gold_source_counts')}",
            flush=True,
        )
        if compact["stems_without_gold"]:
            print(
                f"  unlabeled_stems={len(compact['stems_without_gold'])} "
                f"{compact['stems_without_gold']}",
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
