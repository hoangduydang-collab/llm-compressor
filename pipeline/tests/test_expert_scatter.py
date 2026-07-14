"""CPU bit-parity + orchestration tests for expert-scatter.

The property that makes expert-scatter safe: it must produce, per expert, the
identical result the serial GPTQ loop would, regardless of how many workers run
or which device each expert lands on. These tests pin that with a deterministic
mock quantize_fn (real GPTQ numerics are a separate GPU-gated check).
"""
import torch

from pipeline.expert_scatter import (
    ScatterItem,
    assign_devices,
    scatter_quantize,
    serial_quantize,
)


def _mock_quantize_fn(item, device):
    # Deterministic function of THIS expert's own inputs only. If scatter ever
    # crossed inputs between experts or dropped a device move, the result would
    # differ from the serial computation.
    assert item.weight.device == torch.device("cpu")  # inputs start on cpu
    w = item.weight.to(device)
    h = item.hessian.to(device)
    return {
        "name": item.name,
        "device": str(device),
        "checksum": float(w.sum().item()) + float(torch.diag(h).sum().item()),
    }


def _make_items(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(n):
        dim = 8 + i  # distinct Hessian sizes so assignment/balance is exercised
        items.append(
            ScatterItem(
                name=f"expert.{i}",
                weight=torch.randn(4, dim, generator=g),
                hessian=torch.randn(dim, dim, generator=g),
            )
        )
    return items


def test_scatter_matches_serial_bit_for_bit():
    items = _make_items(16)
    reference = serial_quantize(items, _mock_quantize_fn)
    for devices in (["cpu"], ["cpu"] * 4, ["cpu"] * 8):
        scattered = scatter_quantize(items, devices, _mock_quantize_fn)
        assert set(scattered) == set(reference)
        for name in reference:
            assert scattered[name]["checksum"] == reference[name]["checksum"], name


def test_no_cross_expert_contamination():
    items = _make_items(12, seed=3)
    scattered = scatter_quantize(items, ["cpu"] * 4, _mock_quantize_fn)
    for item in items:
        expected = float(item.weight.sum()) + float(torch.diag(item.hessian).sum())
        assert scattered[item.name]["checksum"] == expected, item.name


def test_every_item_assigned_and_load_balanced():
    items = _make_items(128)
    devices = [f"cuda:{d}" for d in range(8)]
    assignment = assign_devices(items, devices)
    assert len(assignment) == len(items)
    counts = {}
    for dev in assignment:
        counts[str(dev)] = counts.get(str(dev), 0) + 1
    assert set(counts) == set(devices)  # all 8 devices used
    assert max(counts.values()) - min(counts.values()) <= 1  # even count split


def test_assignment_is_deterministic():
    items = _make_items(40, seed=7)
    a = assign_devices(items, [f"cuda:{d}" for d in range(8)])
    b = assign_devices(items, [f"cuda:{d}" for d in range(8)])
    assert [str(x) for x in a] == [str(x) for x in b]


def test_empty_and_single_device_edges():
    assert scatter_quantize([], ["cpu"], _mock_quantize_fn) == {}
    items = _make_items(3)
    one = scatter_quantize(items, ["cpu"], _mock_quantize_fn)
    ref = serial_quantize(items, _mock_quantize_fn)
    assert {k: one[k]["checksum"] for k in one} == {k: ref[k]["checksum"] for k in ref}
