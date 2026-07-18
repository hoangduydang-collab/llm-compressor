"""Offloaded-save tie-detection compat: transformers 5.12.1 resolves meta
state-dict entries via model.get_parameter(), which raises for registered
buffers (M3 router e_score_correction_bias) and killed the r11 disk-offload
save (2026-07-18). Upstream main fixed the branch to get_parameter_or_buffer;
the shim backports that until we upgrade."""

import inspect

import pytest
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import remove_tied_weights_from_state_dict

from pipeline.quantize import _tied_weights_meta_buffer_compat

# same self-disable condition the shim uses: transformers >=5.13.1 resolves
# buffers via get_parameter_or_buffer and the crash (plus the shim) disappears
_FIXED_UPSTREAM = "get_parameter_or_buffer" in inspect.getsource(
    remove_tied_weights_from_state_dict
)


class _TinyConfig(PretrainedConfig):
    model_type = "tiny-compat-test"


class _TinyModel(PreTrainedModel):
    config_class = _TinyConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(2, 2, bias=False)
        # mirrors MiniMax-M3's router: a *buffer* that lands in the state dict
        self.register_buffer("e_score_correction_bias", torch.zeros(2))


def _meta_state_dict(model):
    # disk offload leaves offloaded entries as meta tensors in the state dict
    return {
        name: torch.empty_like(tensor, device="meta")
        for name, tensor in model.state_dict().items()
    }


def test_offloaded_buffer_crashes_without_shim_and_saves_with_it():
    model = _TinyModel(_TinyConfig())
    state_dict = _meta_state_dict(model)

    # the r11 failure mode, pinned; disappears once upstream resolves buffers
    if not _FIXED_UPSTREAM:
        with pytest.raises(AttributeError, match="e_score_correction_bias"):
            remove_tied_weights_from_state_dict(state_dict, model)

    with _tied_weights_meta_buffer_compat(model):
        cleaned = remove_tied_weights_from_state_dict(state_dict, model)
    assert "e_score_correction_bias" in cleaned
    assert "linear.weight" in cleaned


@pytest.mark.skipif(
    _FIXED_UPSTREAM,
    reason="transformers resolves buffers upstream; the shim is inert by design",
)
def test_shim_restores_original_get_parameter():
    model = _TinyModel(_TinyConfig())

    with _tied_weights_meta_buffer_compat(model):
        # inside: buffers resolve like upstream's get_parameter_or_buffer
        assert torch.equal(
            model.get_parameter("e_score_correction_bias"),
            model.get_buffer("e_score_correction_bias"),
        )

    # outside: stock behavior (raises for buffers) is restored
    with pytest.raises(AttributeError):
        model.get_parameter("e_score_correction_bias")
    assert isinstance(model.get_parameter("linear.weight"), torch.nn.Parameter)


def test_shim_noop_when_model_lacks_helper():
    class _Bare:
        pass

    bare = _Bare()
    with _tied_weights_meta_buffer_compat(bare):
        pass
    assert "get_parameter" not in vars(bare)
