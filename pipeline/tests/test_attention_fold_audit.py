"""The fold audit must cover the ATTENTION-side norms, not just the MoE input.

Until 2026-08-28 ``m3_checkpoint_scale_audit`` looked only at
``post_attention_layernorm`` (plus the router and shared experts that consume it),
so the entire attention block's fold was unchecked. That is the gap that would
have hidden GLM's DSA indexer: ``GlmMoeDsaDecoderLayer.forward`` feeds
``input_layernorm``'s output verbatim to the indexer, whose ``wk`` and
``weights_proj`` consume it, and ``wq_b`` consumes ``q_a_layernorm(q_a_proj(x))``
-- none of the three was an AWQ balance layer, and no gate looked.

The trap is latent rather than shipped (every attention-side balance layer is FP8
today, and AWQ skips a mapping with no int-quantized balance layer), so these
tests are about keeping it that way: an uncompensated attention consumer must be
detectable, and a MISSING one must fail rather than pass quietly.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pipeline.m3_checkpoint_scale_audit import (
    _ATTENTION_FOLD_GROUPS,
    _component_suffixes,
    audit_attention_fold,
    audit_checkpoint,
)

HIDDEN = 4
QLORA = 2
LAYER = 7


def _write(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, path / shard)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )


def _tensors(
    *,
    input_scale: torch.Tensor,
    qa_scale: torch.Tensor,
    compensate: tuple[str, ...],
    with_indexer: bool = True,
) -> dict[str, torch.Tensor]:
    """A GLM-shaped layer, folded by ``input_scale`` / ``qa_scale``.

    Only the consumers named in ``compensate`` receive the matching multiply, so a
    test can build the exact half-applied fold that the real defect would produce.
    """
    p = f"model.layers.{LAYER}."
    base_input_norm = torch.tensor([1.0, 0.5, 2.0, 0.25])
    base_qa_norm = torch.tensor([1.0, 0.5])

    def maybe(name: str, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return weight * scale.reshape(1, -1) if name in compensate else weight

    torch.manual_seed(0)
    hidden_consumers = {
        "attn_q": (p + "self_attn.q_a_proj.weight", (QLORA, HIDDEN)),
        "attn_kv_a": (p + "self_attn.kv_a_proj_with_mqa.weight", (3, HIDDEN)),
    }
    qa_consumers = {"attn_q_b": (p + "self_attn.q_b_proj.weight", (5, QLORA))}
    if with_indexer:
        hidden_consumers["indexer_wk"] = (p + "self_attn.indexer.wk.weight", (2, HIDDEN))
        hidden_consumers["indexer_weights_proj"] = (
            p + "self_attn.indexer.weights_proj.weight",
            (1, HIDDEN),
        )
        qa_consumers["indexer_wq_b"] = (p + "self_attn.indexer.wq_b.weight", (4, QLORA))

    tensors = {
        # plain-form norm (GlmMoeDsaRMSNorm): gain is the weight itself, so the
        # inverse fold divides it.
        p + "input_layernorm.weight": base_input_norm / input_scale,
        p + "self_attn.q_a_layernorm.weight": base_qa_norm / qa_scale,
        # the MoE side, so audit_checkpoint's existing components resolve
        p + "post_attention_layernorm.weight": torch.ones(HIDDEN),
        p + "mlp.gate.weight": torch.ones(2, HIDDEN),
        p + "mlp.shared_experts.gate_proj.weight": torch.ones(2, HIDDEN),
    }
    for name, (key, shape) in hidden_consumers.items():
        tensors[key] = maybe(name, torch.full(shape, 1.0) + torch.arange(
            shape[0] * shape[1], dtype=torch.float32
        ).reshape(shape), input_scale)
    for name, (key, shape) in qa_consumers.items():
        tensors[key] = maybe(name, torch.full(shape, 2.0) + torch.arange(
            shape[0] * shape[1], dtype=torch.float32
        ).reshape(shape), qa_scale)
    return tensors


ALL_CONSUMERS = ("attn_q", "attn_kv_a", "indexer_wk", "indexer_weights_proj",
                 "attn_q_b", "indexer_wq_b")


def _pair(tmp_path: Path, **kwargs) -> tuple[Path, Path]:
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    _write(base, _tensors(
        input_scale=torch.ones(HIDDEN), qa_scale=torch.ones(QLORA),
        compensate=ALL_CONSUMERS, with_indexer=kwargs.get("with_indexer", True),
    ))
    _write(cand, _tensors(**kwargs))
    return base, cand


def _audit(base: Path, cand: Path) -> dict:
    return audit_attention_fold(base, cand, LAYER, norm_gain_offset=0.0)


# --- the components must exist at all ----------------------------------------


def test_attention_components_are_registered():
    """The gap was that these names had no entry, so nothing could ask for them."""
    for component in ALL_CONSUMERS + ("attn_input_norm", "qa_norm"):
        suffixes = _component_suffixes(LAYER, component)
        assert suffixes, component


def test_attn_q_covers_both_mla_and_dense_qkv_spellings():
    """GLM-5.2/5.3 spell it q_a_proj; MiniMax-M3 spells it q_proj. Auditing only
    one silently skips the other family."""
    suffixes = _component_suffixes(LAYER, "attn_q")
    assert any(s.endswith("self_attn.q_a_proj.weight") for s in suffixes)
    assert any(s.endswith("self_attn.q_proj.weight") for s in suffixes)


def test_every_group_consumer_has_a_suffix_entry():
    for norm_component, consumers in _ATTENTION_FOLD_GROUPS:
        _component_suffixes(LAYER, norm_component)
        for consumer in consumers:
            _component_suffixes(LAYER, consumer)


# --- a consistent fold is recognised -----------------------------------------


def test_fully_compensated_fold_is_clean(tmp_path: Path):
    base, cand = _pair(
        tmp_path,
        input_scale=torch.tensor([0.25, 4.0, 1.0, 2.0]),
        qa_scale=torch.tensor([0.5, 2.0]),
        compensate=ALL_CONSUMERS,
    )
    report = _audit(base, cand)
    for norm_component, consumers in _ATTENTION_FOLD_GROUPS:
        for consumer in consumers:
            entry = report[norm_component]["consumers"][consumer]
            assert entry["status"] == "checked", (norm_component, consumer)
            assert entry["relative_l2_error"] == pytest.approx(0.0, abs=1e-6)


def test_unsmoothed_layer_audits_as_scale_one(tmp_path: Path):
    """Today's real case: the attention mappings are SKIPPED (all balance layers
    are FP8), so the norms are untouched and every consumer must audit clean at
    scale 1. The gate has to pass in that regime or it is unusable now."""
    base, cand = _pair(
        tmp_path,
        input_scale=torch.ones(HIDDEN),
        qa_scale=torch.ones(QLORA),
        compensate=ALL_CONSUMERS,
    )
    report = _audit(base, cand)
    entry = report["attn_input_norm"]["consumers"]["indexer_wk"]
    assert entry["relative_l2_error"] == pytest.approx(0.0, abs=1e-6)
    assert entry["scale"]["mean"] == pytest.approx(1.0)


# --- the defect this exists to catch -----------------------------------------


@pytest.mark.parametrize(
    "uncompensated,norm_component",
    [
        ("indexer_wk", "attn_input_norm"),
        ("indexer_weights_proj", "attn_input_norm"),
        ("indexer_wq_b", "qa_norm"),
        ("attn_q", "attn_input_norm"),
        ("attn_q_b", "qa_norm"),
    ],
)
def test_one_uncompensated_consumer_is_detected(
    tmp_path: Path, uncompensated: str, norm_component: str
):
    """The exact shape of the real defect: the norm is divided by s and every
    consumer BUT ONE is multiplied by s. The one left behind must show a large
    residual while its siblings stay clean."""
    compensate = tuple(c for c in ALL_CONSUMERS if c != uncompensated)
    base, cand = _pair(
        tmp_path,
        input_scale=torch.tensor([0.25, 4.0, 1.0, 2.0]),
        qa_scale=torch.tensor([0.5, 2.0]),
        compensate=compensate,
    )
    report = _audit(base, cand)
    guilty = report[norm_component]["consumers"][uncompensated]
    assert guilty["status"] == "checked"
    assert guilty["relative_l2_error"] > 0.1, guilty["relative_l2_error"]
    for other in compensate:
        group = next(
            n for n, cs in _ATTENTION_FOLD_GROUPS if other in cs
        )
        assert report[group]["consumers"][other]["relative_l2_error"] == pytest.approx(
            0.0, abs=1e-6
        )


# --- absence must be explicit, and the two kinds must be distinguished -------


def test_layer_without_an_indexer_records_absent_not_failure(tmp_path: Path):
    """57 of GLM's 78 layers legitimately have no indexer. Those must report
    `absent`, not an error and not silence."""
    base, cand = _pair(
        tmp_path,
        input_scale=torch.ones(HIDDEN),
        qa_scale=torch.ones(QLORA),
        compensate=ALL_CONSUMERS,
        with_indexer=False,
    )
    report = _audit(base, cand)
    assert report["attn_input_norm"]["consumers"]["indexer_wk"]["status"] == "absent"
    assert report["qa_norm"]["consumers"]["indexer_wq_b"]["status"] == "absent"
    assert report["attn_input_norm"]["consumers"]["attn_q"]["status"] == "checked"


def test_tensor_dropped_by_quantization_is_flagged_not_skipped(tmp_path: Path):
    """A weight the base has and the saved checkpoint does not is a dropped
    tensor. Reporting that as `absent` would make the MTP-style silent-drop class
    of bug invisible, so it gets its own status."""
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    _write(base, _tensors(
        input_scale=torch.ones(HIDDEN), qa_scale=torch.ones(QLORA),
        compensate=ALL_CONSUMERS,
    ))
    dropped = _tensors(
        input_scale=torch.ones(HIDDEN), qa_scale=torch.ones(QLORA),
        compensate=ALL_CONSUMERS,
    )
    del dropped[f"model.layers.{LAYER}.self_attn.indexer.wk.weight"]
    _write(cand, dropped)
    report = _audit(base, cand)
    entry = report["attn_input_norm"]["consumers"]["indexer_wk"]
    assert entry["status"] == "missing_from_candidate"


def test_missing_norm_is_recorded_rather_than_crashing(tmp_path: Path):
    """M3 has no q_a_layernorm. That must degrade to a recorded status, because a
    crash here would take out the whole post-save gate on a healthy run."""
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    for path in (base, cand):
        tensors = _tensors(
            input_scale=torch.ones(HIDDEN), qa_scale=torch.ones(QLORA),
            compensate=ALL_CONSUMERS,
        )
        del tensors[f"model.layers.{LAYER}.self_attn.q_a_layernorm.weight"]
        _write(path, tensors)
    report = _audit(base, cand)
    assert report["qa_norm"]["status"] == "norm_absent"
    assert report["attn_input_norm"]["status"] == "checked"


# --- it has to be reachable from the real entry point ------------------------


def test_audit_checkpoint_includes_the_attention_fold(tmp_path: Path):
    """Wiring test: the gate reads audit_checkpoint's output, so a perfect
    audit_attention_fold that nothing calls closes no gap."""
    base, cand = _pair(
        tmp_path,
        input_scale=torch.tensor([0.25, 4.0, 1.0, 2.0]),
        qa_scale=torch.tensor([0.5, 2.0]),
        compensate=tuple(c for c in ALL_CONSUMERS if c != "indexer_wk"),
    )
    report = audit_checkpoint(base, cand, [LAYER], norm_gain_offset=0.0)
    fold = report["layers"][str(LAYER)]["attention_fold"]
    assert fold["attn_input_norm"]["consumers"]["indexer_wk"][
        "relative_l2_error"
    ] > 0.1


# --- the GATE, end to end ----------------------------------------------------
#
# The audit reporting a residual closes nothing on its own; the run only stops if
# assert_smooth_fold_consistency turns it into a failure. That function had no
# test of any kind before 2026-08-28, which is its own instance of the pattern
# these files keep documenting: the check existed, the check on the check did not.


def _gate(ckpt: Path, base: Path):
    from pipeline.quantize import assert_smooth_fold_consistency

    return assert_smooth_fold_consistency(ckpt, base, [LAYER], norm_gain_offset=0.0)


def test_gate_passes_a_fully_compensated_fold(tmp_path: Path):
    base, cand = _pair(
        tmp_path,
        input_scale=torch.tensor([0.25, 4.0, 1.0, 2.0]),
        qa_scale=torch.tensor([0.5, 2.0]),
        compensate=ALL_CONSUMERS,
    )
    _gate(cand, base)  # must not raise


@pytest.mark.parametrize(
    "uncompensated", ["indexer_wk", "indexer_weights_proj", "indexer_wq_b"]
)
def test_gate_fails_an_uncompensated_indexer(tmp_path: Path, uncompensated: str):
    base, cand = _pair(
        tmp_path,
        input_scale=torch.tensor([0.25, 4.0, 1.0, 2.0]),
        qa_scale=torch.tensor([0.5, 2.0]),
        compensate=tuple(c for c in ALL_CONSUMERS if c != uncompensated),
    )
    with pytest.raises(RuntimeError, match="smooth-fold consistency gate FAILED"):
        _gate(cand, base)


def test_gate_names_the_guilty_consumer(tmp_path: Path):
    """A gate that fails without saying which tensor moved sends the reader back
    to a 59k-module tree."""
    base, cand = _pair(
        tmp_path,
        input_scale=torch.tensor([0.25, 4.0, 1.0, 2.0]),
        qa_scale=torch.ones(QLORA),
        compensate=tuple(c for c in ALL_CONSUMERS if c != "indexer_wk"),
    )
    with pytest.raises(RuntimeError) as excinfo:
        _gate(cand, base)
    assert "attn_input_norm/indexer_wk" in str(excinfo.value)


def test_gate_fails_a_dropped_attention_tensor(tmp_path: Path):
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    _write(base, _tensors(
        input_scale=torch.ones(HIDDEN), qa_scale=torch.ones(QLORA),
        compensate=ALL_CONSUMERS,
    ))
    dropped = _tensors(
        input_scale=torch.ones(HIDDEN), qa_scale=torch.ones(QLORA),
        compensate=ALL_CONSUMERS,
    )
    del dropped[f"model.layers.{LAYER}.self_attn.indexer.wq_b.weight"]
    _write(cand, dropped)
    with pytest.raises(RuntimeError, match="missing_from_candidate"):
        _gate(cand, base)


def test_gate_reports_attention_coverage_counts(capsys, tmp_path: Path):
    """`the gate passed` must not be readable as `the attention block was
    audited`. On a layer with an indexer, 6 consumers are checked; without one,
    3 are checked and 3 are recorded absent."""
    base, cand = _pair(
        tmp_path,
        input_scale=torch.ones(HIDDEN),
        qa_scale=torch.ones(QLORA),
        compensate=ALL_CONSUMERS,
    )
    _gate(cand, base)
    assert "attention-side consumers checked=6, absent=0" in capsys.readouterr().out

    plain = tmp_path / "plain"
    plain_base = tmp_path / "plain_base"
    for path in (plain, plain_base):
        _write(path, _tensors(
            input_scale=torch.ones(HIDDEN), qa_scale=torch.ones(QLORA),
            compensate=ALL_CONSUMERS, with_indexer=False,
        ))
    _gate(plain, plain_base)
    assert "checked=3, absent=3" in capsys.readouterr().out
