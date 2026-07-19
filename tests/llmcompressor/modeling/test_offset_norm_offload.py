"""Regression test: norm smoothing folds must survive disk offload.

The full-calibration AWQ run r2 (2026-07-18) produced a numerically
inconsistent checkpoint: AWQ balance layers (router / shared / routed experts)
kept their smoothing-scale multiplies, but the smooth-side fold into the
offset norm was silently lost. Root cause: ``CalibrationOffsetNorm.restore``
wrote the folded weight with a raw ``original.weight.data = ...`` assignment.
With compressed-tensors offload, ``module._parameters`` is an ``OffloadCache``
and only writes through the cache (``update_offload_parameter``) reach the
disk copy that ``save_pretrained`` reads — raw ``.data`` writes only mutate
the onloaded view. r9 (pure CPU placement, no offload cache) persisted fine,
which is why the bug only appeared once disk offload entered the pipeline.
"""

import torch

from llmcompressor.modeling.offset_norm import norm_calibration_context


class GemmaRMSNorm(torch.nn.Module):
    """Offset-norm stand-in; the class NAME drives registry lookup."""

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * (1.0 + self.weight)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = GemmaRMSNorm(8)
        self.config = None  # norm_calibration_context reads model.config


def _offload_norm_to_disk(module: torch.nn.Module, tmp_path) -> None:
    from compressed_tensors.offload import offload_module

    offload_module(module, "cpu", "disk", offload_dir=str(tmp_path))


def _disk_copy(module: torch.nn.Module, name: str) -> torch.Tensor:
    """Read the offloaded (disk) copy. OffloadCache.__getitem__ always onloads
    from offloaded storage (no persistent onload view outside
    ``disable_offloading``), so a fresh cache read reflects the disk bytes."""
    return module._parameters[name].detach().clone()


def test_restore_persists_fold_through_disk_offload(tmp_path):
    model = TinyModel()
    base = torch.arange(8, dtype=torch.float32) / 10.0
    with torch.no_grad():
        model.norm.weight.copy_(base)
    _offload_norm_to_disk(model.norm, tmp_path)

    scales = torch.full((8,), 2.0)
    with norm_calibration_context(model):
        calib_norm = model.norm
        assert not isinstance(calib_norm, GemmaRMSNorm)
        # effective weight seen by modifiers is 1 + base
        assert torch.allclose(calib_norm.weight, 1.0 + base)
        # simulate AWQ's smooth-side fold on the calibration module
        with torch.no_grad():
            calib_norm.weight.div_(scales)

    assert isinstance(model.norm, GemmaRMSNorm)
    expected = (1.0 + base) / scales - 1.0
    # the onloaded view must be folded ...
    assert torch.allclose(model.norm.weight, expected, atol=1e-6)
    # ... and, critically, so must the DISK copy that save_pretrained reads
    assert torch.allclose(_disk_copy(model.norm, "weight"), expected, atol=1e-6)


def test_raw_data_write_does_not_persist_to_disk(tmp_path):
    """Environment pin for the root cause: raw ``.data`` assignments bypass
    the OffloadCache disk copy. If this ever starts persisting, the write
    semantics changed upstream and restore() can be simplified."""
    module = GemmaRMSNorm(8)
    base = module.weight.detach().clone()
    _offload_norm_to_disk(module, tmp_path)

    module.weight.data = torch.ones(8)
    assert torch.allclose(_disk_copy(module, "weight"), base, atol=1e-6)
