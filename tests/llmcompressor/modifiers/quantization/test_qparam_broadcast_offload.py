"""Regression test: distributed qparam broadcasts must survive disk offload.

The full-calibration AWQ run r3 (2026-07-19) saved uninitialized
``weight_scale`` tensors for ~7/8 of quantized modules. Root cause: in the
distributed weight-qparam path each rank observes only its greedy-binned
subset of modules and the results are ``dist.broadcast`` from the owner rank.
With compressed-tensors dict/disk offload, ``getattr(module, name)`` mints a
FRESH onload tensor on every call, so the broadcast filled a temporary that
``save_pretrained`` never sees — non-owner ranks (including the saving rank)
kept whatever uninitialized bytes lived in offloaded storage. r9 (pure CPU
placement) persisted fine, which is why the bug only appeared once disk
offload entered the pipeline. Same class as the offset-norm fold loss
(``test_offset_norm_offload.py``), different write site.

The fix writes received values back through ``update_offload_parameter``.
GPTQ's ``_broadcast_quantized_params`` shares the pattern and the fix.
"""

import torch

from llmcompressor.modifiers.gptq.base import GPTQModifier
from llmcompressor.modifiers.quantization.quantization import base as q_base
from llmcompressor.modifiers.quantization.quantization.base import (
    QuantizationModifier,
)

_MARKER = 7.5


class _FakeWork:
    def wait(self):
        return None


def _fake_broadcast(tensor, src=None, async_op=False, **kwargs):
    """Simulate a receive: the collective fills the passed tensor in place."""
    with torch.no_grad():
        tensor.fill_(_MARKER)
    return _FakeWork()


def _offloaded_linear_with_scale(tmp_path) -> torch.nn.Module:
    from compressed_tensors.offload import offload_module

    module = torch.nn.Linear(8, 4, bias=False)
    module.weight_scale = torch.nn.Parameter(torch.zeros(4, 1))
    offload_module(module, "cpu", "disk", offload_dir=str(tmp_path))
    return module


def _disk_copy(module: torch.nn.Module, name: str) -> torch.Tensor:
    """Fresh cache read = the offloaded (disk) bytes save_pretrained reads."""
    return module._parameters[name].detach().clone()


def test_qparam_broadcast_persists_through_disk_offload(tmp_path, monkeypatch):
    module = _offloaded_linear_with_scale(tmp_path)

    monkeypatch.setattr(q_base.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(q_base.dist, "broadcast", _fake_broadcast)
    # disk-offloaded modules report cpu execution here; the real runs execute
    # on cuda, which is the branch under test
    monkeypatch.setattr(
        q_base, "get_execution_device", lambda m: torch.device("cuda")
    )

    QuantizationModifier._broadcast_qparam_onloads(
        None, [module], {module: 0}
    )

    expected = torch.full((4, 1), _MARKER)
    assert torch.allclose(_disk_copy(module, "weight_scale"), expected)


def test_gptq_broadcast_persists_through_disk_offload(tmp_path, monkeypatch):
    from llmcompressor.modifiers.gptq import base as gptq_base

    module = _offloaded_linear_with_scale(tmp_path)

    monkeypatch.setattr(gptq_base.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(gptq_base.dist, "broadcast", _fake_broadcast)

    GPTQModifier._broadcast_quantized_params(None, [module], {module: 0})

    expected_scale = torch.full((4, 1), _MARKER)
    expected_weight = torch.full((4, 8), _MARKER)
    assert torch.allclose(_disk_copy(module, "weight_scale"), expected_scale)
    assert torch.allclose(_disk_copy(module, "weight"), expected_weight)


def test_owner_rank_does_not_rewrite_offload(tmp_path, monkeypatch):
    """The owner rank already persisted via update_qparams; the broadcast
    write-back must only target modules received from other ranks (here the
    fake broadcast mutates the onload temp, and on the owner rank that temp
    must NOT be pushed back to disk)."""
    module = _offloaded_linear_with_scale(tmp_path)
    before = _disk_copy(module, "weight_scale")

    monkeypatch.setattr(q_base.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(q_base.dist, "broadcast", _fake_broadcast)
    monkeypatch.setattr(
        q_base, "get_execution_device", lambda m: torch.device("cuda")
    )

    QuantizationModifier._broadcast_qparam_onloads(
        None, [module], {module: 0}
    )

    assert torch.allclose(_disk_copy(module, "weight_scale"), before)
