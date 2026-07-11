"""CPU tests for the parallel MiniMax-M3 layer-boundary matrix."""

from pipeline.m3_layer_boundary_diagnostics import EXPECTED_ARMS, classify_matrix


def _boundaries(explosive_boundary: str | None = None) -> list[dict]:
    order = (
        "decoder_input_hidden",
        "attention_input",
        "attention_output",
        "moe_input",
        "moe_output",
        "decoder_output_hidden",
        "decoder_output_residual",
    )
    records = []
    for rank in range(8):
        for boundary in order:
            records.append(
                {
                    "rank": rank,
                    "layer": 8,
                    "boundary": boundary,
                    "tokens": 168,
                    "norm": 20000.0 if boundary == explosive_boundary else 200.0,
                    "finite_fraction": 1.0,
                }
            )
    return records


def _arm(quality: bool, *, explosive: str | None = None) -> dict:
    return {
        "infrastructure_ok": True,
        "quality_ok": quality,
        "layer_boundary_records": _boundaries(explosive),
        "fingerprints": [
            {
                "rank": rank,
                "category": "moe_router",
                "dtype": "torch.float32",
                "sample_abs_max": 0.2,
            }
            for rank in range(8)
        ],
    }


def _arms() -> dict[str, dict]:
    arms = {name: _arm(False, explosive="moe_input") for name in EXPECTED_ARMS}
    arms["reference_w4a16_ep_fp8kv"] = _arm(True)
    arms["reference_w4a16_tp_fp8kv"] = _arm(True)
    return arms


def test_missing_arms_are_inconclusive():
    result = classify_matrix({"reference_w4a16_ep_fp8kv": _arm(True)})
    assert result["verdict"] == "inconclusive_missing_arms"


def test_router_alias_recovery_has_priority():
    arms = _arms()
    arms["candidate_w4a8_router_alias"]["quality_ok"] = True
    result = classify_matrix(arms)
    assert result["verdict"] == "router_alias_boundary"


def test_expert_parallel_recovery_is_classified():
    arms = _arms()
    arms["candidate_w4a16_tp_fp8kv"]["quality_ok"] = True
    result = classify_matrix(arms)
    assert result["verdict"] == "expert_parallel_boundary"


def test_kv_cache_recovery_is_classified():
    arms = _arms()
    arms["candidate_w4a8_ep_autokv"]["quality_ok"] = True
    result = classify_matrix(arms)
    assert result["verdict"] == "kv_cache_boundary"


def test_first_explosive_moe_input_selects_attention_residual_boundary():
    result = classify_matrix(_arms())
    assert result["verdict"] == "attention_residual_boundary"
    assert result["first_explosive_boundary"]["boundary"] == "moe_input"


def test_first_explosive_moe_output_selects_routed_moe_boundary():
    arms = _arms()
    for name in ("candidate_w4a8_ep_fp8kv", "candidate_w4a16_ep_fp8kv"):
        arms[name]["layer_boundary_records"] = _boundaries("moe_output")
    assert classify_matrix(arms)["verdict"] == "routed_moe_boundary"


def test_invalid_reference_stops_diagnosis():
    arms = _arms()
    arms["reference_w4a16_ep_fp8kv"]["quality_ok"] = False
    assert classify_matrix(arms)["verdict"] == "invalid_reference"
