"""CPU-only tests for MiniMax-M3 routed-expert diagnostics."""

from __future__ import annotations

from pipeline.m3_routed_diagnostics import (
    EXPECTED_ARMS,
    classify_diagnostics,
    compare_first_moe_inputs,
    compare_unquantized_fingerprints,
)


def _fps(suffix: str = "same") -> list[dict]:
    return [
        {
            "scope": "MiniMaxM3SparseForConditionalGeneration",
            "rank": rank,
            "name": f"model.{category}",
            "category": category,
            "sample_sha256": f"{category}-{rank}-{suffix}",
        }
        for rank in range(8)
        for category in (
            "lm_head",
            "shared_expert",
            "attention_qkv",
            "msa_indexer",
            "routed_expert",
        )
    ]


def _probes(input_digest: str = "same", output_digest: str = "out") -> list[dict]:
    return [
        {
            "rank": rank,
            "probe_index": 1,
            "tokens": 168,
            "input_sample_sha256": f"{input_digest}-{rank}",
            "output_sample_sha256": f"{output_digest}-{rank}",
            "routed_sample_sha256": f"routed-{output_digest}-{rank}",
            "shared_present": True,
            "shared_norm": 2.0,
            "dropped": False,
        }
        for rank in range(8)
    ]


def _arm(quality_ok: bool, *, digest: str = "same", fp_suffix: str = "same") -> dict:
    return {
        "infrastructure_ok": True,
        "quality_ok": quality_ok,
        "fingerprints": _fps(fp_suffix),
        "moe_probe_records": _probes(digest),
        "loader_audit_lines": [
            f"rank={rank} M3_LOAD_AUDIT# matched" for rank in range(8)
        ],
    }


def test_unquantized_fingerprints_match_by_scope_rank_name():
    comparison = compare_unquantized_fingerprints(_fps(), _fps())

    # Only tensors stored unquantized in both checkpoint schemes are exact
    # controls. Reference attention is W4A16 while candidate attention is BF16.
    assert comparison["compared"] == 16
    assert comparison["mismatched"] == []
    assert comparison["missing"] == []


def test_first_moe_input_comparison_is_rank_aligned():
    comparison = compare_first_moe_inputs(_probes("input"), _probes("input"))

    assert comparison == {
        "compared_ranks": list(range(8)),
        "mismatched_ranks": [],
        "missing_ranks": [],
    }


def test_missing_arm_is_inconclusive():
    result = classify_diagnostics({"reference_w4a16": _arm(True)})

    assert result["verdict"] == "inconclusive_missing_arms"
    assert set(result["missing_arms"]) == set(EXPECTED_ARMS) - {"reference_w4a16"}


def test_invalid_reference_stops_candidate_diagnosis():
    arms = {
        "reference_w4a16": _arm(False),
        "candidate_w4a8": _arm(False),
        "candidate_w4a16": _arm(False),
    }

    assert classify_diagnostics(arms)["verdict"] == "invalid_reference"


def test_missing_rank_or_category_is_inconclusive():
    arms = {
        "reference_w4a16": _arm(True),
        "candidate_w4a8": _arm(False),
        "candidate_w4a16": _arm(False),
    }
    arms["candidate_w4a8"]["moe_probe_records"] = _probes()[:-1]

    result = classify_diagnostics(arms)

    assert result["verdict"] == "inconclusive_missing_diagnostics"
    assert "candidate_w4a8" in result["failed_arms"]


def test_w4a16_recovery_selects_activation_boundary():
    arms = {
        "reference_w4a16": _arm(True),
        "candidate_w4a8": _arm(False),
        "candidate_w4a16": _arm(True),
    }

    assert classify_diagnostics(arms)["verdict"] == "w4a8_activation_boundary"


def test_both_candidate_failures_select_routed_weight_or_loader_boundary():
    arms = {
        "reference_w4a16": _arm(True),
        "candidate_w4a8": _arm(False),
        "candidate_w4a16": _arm(False),
    }

    result = classify_diagnostics(arms)

    assert result["verdict"] == "routed_weight_or_loader_boundary"
    assert result["first_moe_inputs"]["candidate_w4a8_vs_w4a16"]["mismatched_ranks"] == []


def test_unquantized_mismatch_has_priority_over_routed_experts():
    arms = {
        "reference_w4a16": _arm(True),
        "candidate_w4a8": _arm(False, fp_suffix="different"),
        "candidate_w4a16": _arm(False, fp_suffix="different"),
    }

    assert classify_diagnostics(arms)["verdict"] == "unquantized_load_boundary"


def test_candidate_overlay_input_mismatch_is_inconclusive():
    arms = {
        "reference_w4a16": _arm(True),
        "candidate_w4a8": _arm(False, digest="candidate"),
        "candidate_w4a16": _arm(False, digest="overlay-different"),
    }

    assert classify_diagnostics(arms)["verdict"] == "overlay_pre_moe_divergence"


def test_reference_input_digest_difference_does_not_override_activation_result():
    arms = {
        "reference_w4a16": _arm(True, digest="reference-quantized-attention"),
        "candidate_w4a8": _arm(False, digest="candidate"),
        "candidate_w4a16": _arm(True, digest="candidate"),
    }

    assert classify_diagnostics(arms)["verdict"] == "w4a8_activation_boundary"
