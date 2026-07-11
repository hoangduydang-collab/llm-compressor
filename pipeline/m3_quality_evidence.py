"""MiniMax-M3 smoke-quality assessment and paired evidence decisions.

This module intentionally imports only the Python standard library so reports
can be classified on a CPU login node without importing Torch or vLLM.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QualityCase:
    """One deterministic smoke prompt and its accepted answer fragments."""

    case_id: str
    prompt: str
    expected_any: tuple[str, ...]


M3_QUALITY_CASES = (
    QualityCase("capital_france", "The capital of France is", ("paris",)),
    QualityCase(
        "arithmetic_2_plus_2",
        "What is 2 + 2? Answer with only the number.",
        ("4", "four"),
    ),
)


def _has_consecutive_character_chunk(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.casefold())
    if len(compact) < 6:
        return False
    max_chunk = min(32, len(compact) // 3)
    for size in range(2, max_chunk + 1):
        for start in range(0, len(compact) - (size * 3) + 1):
            chunk = compact[start : start + size]
            repeats = 1
            cursor = start + size
            while compact[cursor : cursor + size] == chunk:
                repeats += 1
                cursor += size
            if repeats >= 3 and repeats * size >= max(12, len(compact) // 2):
                return True
    return False


def _has_dominant_token(text: str) -> bool:
    tokens = re.findall(r"[\w']+", text.casefold(), flags=re.UNICODE)
    if len(tokens) < 4:
        return False
    _, count = Counter(tokens).most_common(1)[0]
    return count >= 4 and count / len(tokens) >= 0.6


def assess_output(text: str, expected_any: tuple[str, ...]) -> dict[str, Any]:
    """Assess one raw generation without discarding evidence."""

    raw = text if isinstance(text, str) else str(text)
    normalized = " ".join(raw.casefold().split())
    reasons: list[str] = []
    if _has_dominant_token(raw):
        reasons.append("dominant_token")
    if _has_consecutive_character_chunk(raw):
        reasons.append("character_chunk")
    expected_match = any(
        expected.casefold() in normalized for expected in expected_any
    )
    nonempty = bool(normalized)
    return {
        "text": raw,
        "normalized": normalized,
        "expected_any": list(expected_any),
        "expected_match": expected_match,
        "nonempty": nonempty,
        "repetitive": bool(reasons),
        "repetition_reasons": reasons,
        "passed": nonempty and expected_match and not reasons,
    }


def assess_quality_outputs(outputs: list[str]) -> dict[str, Any]:
    """Assess outputs in the fixed order of :data:`M3_QUALITY_CASES`."""

    errors: list[str] = []
    complete = len(outputs) == len(M3_QUALITY_CASES)
    if not complete:
        errors.append(
            f"output_count={len(outputs)} expected={len(M3_QUALITY_CASES)}"
        )
    cases: list[dict[str, Any]] = []
    for quality_case, output in zip(M3_QUALITY_CASES, outputs):
        assessed = assess_output(output, quality_case.expected_any)
        cases.append({**asdict(quality_case), **assessed})
    quality_ok = complete and all(case["passed"] for case in cases)
    return {
        "complete": complete,
        "quality_ok": quality_ok,
        "quality_cases": cases,
        "errors": errors,
    }


def classify_pair(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Select the next quality boundary without guessing past missing evidence."""

    if reference.get("quality_ok") is not True:
        return {
            "verdict": "invalid_reference",
            "next": "repair the reference serving baseline before candidate analysis",
        }
    if candidate.get("quality_ok") is True:
        return {
            "verdict": "candidate_quality_pass",
            "next": "confirm with the broader quality evaluation",
        }
    if evidence.get("required_complete") is not True:
        return {
            "verdict": "inconclusive_missing_evidence",
            "missing": list(evidence.get("missing") or []),
            "next": "collect only the missing paired evidence",
        }
    if evidence.get("lm_head_bad") is True:
        return {
            "verdict": "lm_head_boundary",
            "next": "isolate the MiniMax-M3 lm_head loader mapping",
        }
    if evidence.get("shared_expert_bad") is True:
        return {
            "verdict": "shared_expert_boundary",
            "next": "isolate shared-expert construction, loading, and contribution",
        }
    return {
        "verdict": "attention_indexer_boundary",
        "next": "compare q/k/v and MSA-indexer construction and loading",
    }
