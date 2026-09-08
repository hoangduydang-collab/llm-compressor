"""Packed-cache parameter normalization must retain storage and disk identities."""

import pytest
import torch
from compressed_tensors.offload import (
    disable_onloading,
    from_accelerate,
    offload_module,
    to_accelerate,
)
from compressed_tensors.offload.cache.disk import DiskCache

from llmcompressor.transformers.compression.compressed_tensors_utils import (
    _contiguous_disk_serialization,
    _normalize_offloaded_parameter_types,
)


@pytest.mark.parametrize("backend", ["cpu", "disk"])
def test_packed_parameter_roundtrip_preserves_storage_and_values(tmp_path, backend):
    model = torch.nn.ModuleDict({"packed": torch.nn.Module()})
    layer = model["packed"]
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.int32), requires_grad=False),
    )
    kwargs = {"offload_dir": str(tmp_path)} if backend == "disk" else {}
    offload_module(layer, "cpu", backend, **kwargs)
    expected = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    # Compression writes Tensor entries directly through the existing cache.
    layer._parameters["weight_packed"] = expected
    with disable_onloading():
        raw = layer._parameters["weight_packed"]
        raw.__class__ = torch.Tensor
        identity = id(raw)
        pointer = raw.data_ptr() if backend == "cpu" else None
        index = dict(DiskCache.index[raw]) if backend == "disk" else None
    _normalize_offloaded_parameter_types(model)
    with disable_onloading():
        normalized = layer._parameters["weight_packed"]
        assert isinstance(normalized, torch.nn.Parameter)
        assert id(normalized) == identity
        if backend == "cpu":
            assert normalized.data_ptr() == pointer
        else:
            assert DiskCache.index[normalized] == index
    to_accelerate(model)
    from_accelerate(model)
    torch.testing.assert_close(model["packed"].weight_packed, expected, rtol=0, atol=0)


def test_sliced_packed_disk_write_preserves_values_and_restores_writer(tmp_path):
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    from compressed_tensors.offload.cache import disk

    packed = pack_to_int32(torch.zeros(4, 16, dtype=torch.int8), 4)
    assert not packed.is_contiguous()
    writer = disk.save_file
    cache = DiskCache("cpu", offload_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="sentinel"):
        with _contiguous_disk_serialization():
            stored = cache.offload(packed)
            torch.testing.assert_close(cache.onload(stored), packed, rtol=0, atol=0)
            raise RuntimeError("sentinel")
    assert disk.save_file is writer
    assert not packed.is_contiguous(), "serialization must not mutate the source view"
