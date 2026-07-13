"""CPU classifier tests for the MiniMax-M3 AWQ/GPTQ repair matrix."""

from pipeline.m3_awq_gptq_repair import (
    EARLY_ARMS,
    EXPECTED_ARMS,
    FINISH_ARMS,
    classify_early_matrix,
    classify_matrix,
)


def _records(explosive: bool) -> list[dict]:
    return [
        {
            "rank": rank,
            "layer": 8,
            "boundary": "moe_input",
            "norm": 176000.0 if explosive else 177.0,
            "finite_fraction": 1.0,
        }
        for rank in range(8)
    ]


def _arm(quality: bool, explosive: bool = True) -> dict:
    return {
        "infrastructure_ok": True,
        "quality_ok": quality,
        "layer_boundary_records": _records(explosive),
    }


def _arms() -> dict[str, dict]:
    arms = {name: _arm(False) for name in EXPECTED_ARMS}
    arms["reference_w4a16"] = _arm(True, False)
    return arms


def test_no_smoothing_recovery_is_root_cause():
    arms = _arms()
    for name in ("awq_nosmooth_w4a8", "awq_nosmooth_w4a16", "awq_nosmooth_http"):
        arms[name] = _arm(True, False)
    assert classify_matrix(arms)["verdict"] == "awq_mlp_input_smoothing_root_cause"


def test_offset_norm_recovery_is_root_cause():
    arms = _arms()
    for name in ("awq_offsetfix_w4a8", "awq_offsetfix_w4a16", "awq_offsetfix_http"):
        arms[name] = _arm(True, False)
    assert classify_matrix(arms)["verdict"] == "minimax_offset_norm_root_cause"


def test_repaired_gptq_passes_isolates_awq():
    arms = _arms()
    for name in ("gptq_w4a8", "gptq_w4a16", "gptq_http"):
        arms[name] = _arm(True, False)
    assert classify_matrix(arms)["verdict"] == "awq_specific_unresolved"


def test_same_gptq_boundary_points_to_shared_logic():
    assert classify_matrix(_arms())["verdict"] == "shared_compression_export_boundary"


def test_staged_arm_sets_partition_the_full_matrix():
    assert set(EARLY_ARMS).isdisjoint(FINISH_ARMS)
    assert set(EARLY_ARMS) | set(FINISH_ARMS) == set(EXPECTED_ARMS)


def test_early_gptq_pass_isolates_awq_without_fresh_checkpoints():
    arms = {name: _arm(False) for name in EARLY_ARMS}
    arms["reference_w4a16"] = _arm(True, False)
    for name in ("gptq_w4a8", "gptq_w4a16", "gptq_http"):
        arms[name] = _arm(True, False)

    assert classify_early_matrix(arms)["verdict"] == "gptq_pass_awq_specific"
