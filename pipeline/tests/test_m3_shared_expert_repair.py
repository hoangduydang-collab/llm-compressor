"""CPU tests for MiniMax-M3 shared-expert repair evidence."""

from __future__ import annotations

from pipeline.m3_shared_expert_repair import EXPECTED_ARMS, classify_repair


def _offline(quality_ok: bool = True) -> dict:
    return {
        "infrastructure_ok": True,
        "quality_ok": quality_ok,
        "loader_audit_lines": [
            f"rank={rank} M3_LOAD_AUDIT# matched=65835 "
            "unmatched_this_scope=0 shared_seen=171"
            for rank in range(8)
        ],
        "fingerprints": [
            {
                "rank": rank,
                "category": "shared_expert",
                "name": f"layers.3.shared_experts.{projection}.weight",
                "dtype": "torch.bfloat16",
                "sample_abs_max": 0.125,
            }
            for rank in range(8)
            for projection in ("gate_up_proj", "down_proj")
        ],
        "moe_probe_records": [
            {
                "rank": rank,
                "probe_index": 1,
                "shared_present": True,
                "shared_norm": 50.0,
                "dropped": False,
            }
            for rank in range(8)
        ],
    }


def _http(quality_ok: bool = True) -> dict:
    return {"infrastructure_ok": True, "quality_ok": quality_ok}


def _arms() -> dict[str, dict]:
    return {
        "repaired_w4a8_offline": _offline(),
        "repaired_w4a16_offline": _offline(),
        "repaired_w4a8_http": _http(),
    }


def test_missing_arm_is_inconclusive():
    result = classify_repair({"repaired_w4a8_offline": _offline()})
    assert result["verdict"] == "inconclusive_missing_arms"
    assert set(result["missing_arms"]) == set(EXPECTED_ARMS) - {
        "repaired_w4a8_offline"
    }


def test_all_repaired_arms_pass():
    assert classify_repair(_arms())["verdict"] == "quality_repair_pass"


def test_unmatched_shared_tensors_fail_repair():
    arms = _arms()
    arms["repaired_w4a8_offline"]["loader_audit_lines"][0] = (
        "M3_LOAD_AUDIT# unmatched_this_scope=171 shared_seen=171"
    )
    assert classify_repair(arms)["verdict"] == "shared_ignore_repair_failed"


def test_packed_or_zero_shared_runtime_fails_repair():
    arms = _arms()
    fingerprint = arms["repaired_w4a8_offline"]["fingerprints"][0]
    fingerprint["name"] = fingerprint["name"].replace("weight", "weight_packed")
    fingerprint["dtype"] = "torch.int32"
    fingerprint["sample_abs_max"] = 0.0
    assert classify_repair(arms)["verdict"] == "shared_ignore_repair_failed"


def test_zero_shared_probe_fails_repair():
    arms = _arms()
    arms["repaired_w4a16_offline"]["moe_probe_records"][0]["shared_norm"] = 0.0
    assert classify_repair(arms)["verdict"] == "shared_ignore_repair_failed"


def test_missing_rank_diagnostics_are_inconclusive():
    arms = _arms()
    arms["repaired_w4a8_offline"]["moe_probe_records"].pop()
    result = classify_repair(arms)
    assert result["verdict"] == "inconclusive_missing_diagnostics"


def test_w4a16_only_recovery_selects_activation_boundary():
    arms = _arms()
    arms["repaired_w4a8_offline"]["quality_ok"] = False
    arms["repaired_w4a8_http"]["quality_ok"] = False
    assert (
        classify_repair(arms)["verdict"]
        == "activation_boundary_after_shared_repair"
    )


def test_offline_http_disagreement_is_explicit():
    arms = _arms()
    arms["repaired_w4a8_http"]["quality_ok"] = False
    assert classify_repair(arms)["verdict"] == "candidate_interface_disagreement"


def test_both_schemes_fail_after_healthy_shared_loading():
    arms = _arms()
    arms["repaired_w4a8_offline"]["quality_ok"] = False
    arms["repaired_w4a8_http"]["quality_ok"] = False
    arms["repaired_w4a16_offline"]["quality_ok"] = False
    assert classify_repair(arms)["verdict"] == "post_shared_routed_boundary"


def test_w4a16_overlay_regression_is_explicit():
    arms = _arms()
    arms["repaired_w4a16_offline"]["quality_ok"] = False
    assert classify_repair(arms)["verdict"] == "w4a16_overlay_backend_regression"


def test_infrastructure_failure_precedes_quality_classification():
    arms = _arms()
    arms["repaired_w4a8_http"]["infrastructure_ok"] = False
    result = classify_repair(arms)
    assert result["verdict"] == "infrastructure_failure"
    assert result["failed_arms"] == ["repaired_w4a8_http"]
