"""The scale audit must be told the architecture's norm gain form.

`compensation_error` re-derives the smoothing scale implied by the norm weights
and checks it explains the change in the balance layers. That derivation depends
on whether the norm applies ``output * weight`` or ``output * (1 + weight)``, and
the function originally hardcoded the second (MiniMaxM3VLRMSNorm is Gemma-style).

GLM-5.2's GlmMoeDsaRMSNorm is the first: plain ``self.weight * hidden_states``,
asserted in KNOWN_ORDINARY_NORM_CLASSES. Auditing it with the offset form does
not error -- it computes a wrong implied scale, so a perfectly consistent fold
reports a large relative L2 and the post-save gate fails a run that was fine, at
the very end, after all the calibration is paid for.

test_wrong_form_reports_a_large_error_on_a_healthy_fold is the regression: it
demonstrates the false failure directly.
"""

import types

import pytest
import torch

from pipeline.m3_checkpoint_scale_audit import compensation_error


def consistent_fold(offset, channels=16, seed=0):
    """A correctly folded (norm, balance) pair for a given gain form.

    AWQ divides the norm's effective gain by ``scale`` and multiplies the
    consumer's input channels by the same ``scale``, leaving the product
    unchanged. Returns tensors as the checkpoints would hold them, i.e. the RAW
    norm parameter, not the effective gain.
    """
    generator = torch.Generator().manual_seed(seed)
    base_gain = torch.rand(channels, generator=generator) + 0.5
    scale = torch.rand(channels, generator=generator) + 0.5
    base_weight = torch.randn(8, channels, generator=generator)

    cand_gain = base_gain / scale
    cand_weight = base_weight * scale.reshape(1, -1)
    # stored parameter = effective gain - offset
    return base_gain - offset, cand_gain - offset, base_weight, cand_weight, scale


# --------------------------------------------------------------------------
# each form, audited with the matching offset
# --------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_consistent_fold_audits_clean_with_the_matching_form(offset):
    base_norm, cand_norm, base_w, cand_w, _ = consistent_fold(offset)
    result = compensation_error(
        base_norm, cand_norm, base_w, cand_w, norm_gain_offset=offset
    )
    assert result["relative_l2_error"] < 1e-5


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_recovers_the_true_scale(offset):
    base_norm, cand_norm, base_w, cand_w, scale = consistent_fold(offset)
    result = compensation_error(
        base_norm, cand_norm, base_w, cand_w, norm_gain_offset=offset
    )
    assert result["scale"]["mean"] == pytest.approx(float(scale.mean()), rel=1e-4)


def test_unsmoothed_layer_audits_as_scale_one():
    """Partial-layer smokes audit every layer; untouched ones must pass."""
    weight = torch.randn(8, 16)
    norm = torch.rand(16) + 0.5
    for offset in (0.0, 1.0):
        result = compensation_error(
            norm, norm.clone(), weight, weight.clone(), norm_gain_offset=offset
        )
        assert result["relative_l2_error"] < 1e-6
        assert result["scale"]["mean"] == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------
# the regression this exists for
# --------------------------------------------------------------------------


def test_wrong_form_reports_a_large_error_on_a_healthy_fold():
    """GLM's exact hazard: a plain-norm fold audited as if it were offset.

    Reference magnitudes from the M3 record: a consistent fold is ~3e-3 and a
    LOST fold is 0.09-0.27, with the gate threshold at 0.02. So the wrong form
    has to land above 0.02 to be a false failure -- assert that explicitly
    rather than just 'bigger'.
    """
    base_norm, cand_norm, base_w, cand_w, _ = consistent_fold(0.0)

    right = compensation_error(
        base_norm, cand_norm, base_w, cand_w, norm_gain_offset=0.0
    )
    wrong = compensation_error(
        base_norm, cand_norm, base_w, cand_w, norm_gain_offset=1.0
    )

    assert right["relative_l2_error"] < 1e-5
    assert wrong["relative_l2_error"] > 0.02, (
        "auditing a plain norm with the offset form must not silently pass; "
        f"got {wrong['relative_l2_error']:.3e}"
    )


def test_wrong_form_in_the_other_direction_also_shows_up():
    base_norm, cand_norm, base_w, cand_w, _ = consistent_fold(1.0)
    wrong = compensation_error(
        base_norm, cand_norm, base_w, cand_w, norm_gain_offset=0.0
    )
    assert wrong["relative_l2_error"] > 0.02


# --------------------------------------------------------------------------
# fail-closed API
# --------------------------------------------------------------------------


def test_norm_gain_offset_is_required():
    """No default: a caller for a new family must state the form."""
    base_norm, cand_norm, base_w, cand_w, _ = consistent_fold(0.0)
    with pytest.raises(TypeError):
        compensation_error(base_norm, cand_norm, base_w, cand_w)


def test_offset_is_recorded_in_the_result_for_provenance():
    from pipeline.m3_checkpoint_scale_audit import audit_checkpoint  # noqa: F401

    # compensation_error itself does not echo the offset; audit_checkpoint does.
    # Assert the signature carries it so the JSON can record it.
    import inspect

    sig = inspect.signature(audit_checkpoint)
    assert "norm_gain_offset" in sig.parameters
    assert sig.parameters["norm_gain_offset"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------
# dead channels, per form
# --------------------------------------------------------------------------


def test_dead_channel_both_zero_is_pinned_to_scale_one():
    """0/0 must not become NaN and fail a layer that is consistent.

    Offset form: a base parameter of exactly -1 gives gain 0 (M3 layers
    8/10-13). Plain form: a parameter of exactly 0 does.
    """
    for offset, dead_value in ((1.0, -1.0), (0.0, 0.0)):
        norm = torch.tensor([dead_value, 0.5, 0.75])
        weight = torch.randn(4, 3)
        result = compensation_error(
            norm, norm.clone(), weight, weight.clone(), norm_gain_offset=offset
        )
        assert result["relative_l2_error"] < 1e-6
        assert not torch.isnan(torch.tensor(result["scale"]["mean"]))


# --------------------------------------------------------------------------
# resolve_norm_gain_offset: derive the form from the live model
# --------------------------------------------------------------------------


def _resolver(monkeypatch, offset_classes, ordinary_classes):
    """Import resolve_norm_gain_offset with stubbed registries.

    The real registries live in llmcompressor.preflight.quantization, which is
    not importable in this environment; the logic under test is the set
    reduction, so stub the authority and exercise that.
    """
    import sys

    module = types.ModuleType("llmcompressor.preflight.quantization")
    module.KNOWN_OFFSET_NORM_CLASSES = frozenset(offset_classes)
    module.KNOWN_ORDINARY_NORM_CLASSES = frozenset(ordinary_classes)
    for name in ("llmcompressor", "llmcompressor.preflight"):
        pkg = sys.modules.get(name) or types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, "llmcompressor.preflight.quantization", module)

    from pipeline.quantize import resolve_norm_gain_offset

    return resolve_norm_gain_offset


def _model_with(*class_names):
    model = torch.nn.Module()
    children = torch.nn.ModuleList()
    for index, class_name in enumerate(class_names):
        cls = type(class_name, (torch.nn.Module,), {})
        children.append(cls())
    model.norms = children
    return model


def test_resolver_returns_one_for_offset_norms(monkeypatch):
    resolve = _resolver(monkeypatch, {"MiniMaxM3VLRMSNorm"}, {"GlmMoeDsaRMSNorm"})
    assert resolve(_model_with("MiniMaxM3VLRMSNorm")) == 1.0


def test_resolver_returns_zero_for_ordinary_norms(monkeypatch):
    resolve = _resolver(monkeypatch, {"MiniMaxM3VLRMSNorm"}, {"GlmMoeDsaRMSNorm"})
    assert resolve(_model_with("GlmMoeDsaRMSNorm")) == 0.0


def test_resolver_returns_none_for_an_unclassified_norm(monkeypatch):
    """A norm nobody has read must not be guessed at."""
    resolve = _resolver(monkeypatch, {"MiniMaxM3VLRMSNorm"}, {"GlmMoeDsaRMSNorm"})
    assert resolve(_model_with("SomeBrandNewRMSNorm")) is None


def test_resolver_returns_none_when_the_model_mixes_both_forms(monkeypatch):
    """One offset per checkpoint; a mixed model cannot be audited this way."""
    resolve = _resolver(monkeypatch, {"MiniMaxM3VLRMSNorm"}, {"GlmMoeDsaRMSNorm"})
    model = _model_with("MiniMaxM3VLRMSNorm", "GlmMoeDsaRMSNorm")
    assert resolve(model) is None


def test_resolver_returns_none_for_a_model_with_no_norms(monkeypatch):
    resolve = _resolver(monkeypatch, {"MiniMaxM3VLRMSNorm"}, {"GlmMoeDsaRMSNorm"})
    assert resolve(torch.nn.Linear(4, 4)) is None
