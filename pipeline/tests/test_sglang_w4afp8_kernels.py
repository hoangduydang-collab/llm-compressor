"""Round-trip and property tests for the SGLang w4afp8 conversion kernels.

Both kernels produce well-formed output for wrong input and neither failure mode
raises, so identities are the only real defence. In particular:

  * a reciprocal-vs-multiplier mix-up in the block scale still yields a
    loadable checkpoint that serves noise, and
  * a swapped nibble order still yields the right FILE SIZE and a plausible
    value histogram -- it was only distinguishable on PhalaCloud's checkpoint by
    a least-squares refit.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pipeline.sglang_w4afp8_kernels import (  # noqa: E402
    E4M3_MAX,
    dequantize_block_fp8,
    pack_nibbles_int8,
    quantize_block_fp8,
    repack_int32_to_int8,
    unpack_nibbles_int8,
)


# --------------------------------------------------------------------------
# block fp8
# --------------------------------------------------------------------------


def test_block_fp8_shapes_follow_the_loader_formula():
    w = torch.randn(256, 512)
    q, s = quantize_block_fp8(w)
    assert q.shape == (256, 512)
    assert q.dtype == torch.float8_e4m3fn
    # Fp8LinearMethod.create_weights registers exactly this shape.
    assert s.shape == (256 // 128, 512 // 128)
    assert s.dtype == torch.float32


def test_block_fp8_round_trip_is_within_e4m3_resolution():
    torch.manual_seed(0)
    w = torch.randn(384, 256)
    q, s = quantize_block_fp8(w)
    back = dequantize_block_fp8(q, s)
    resid = (back - w).norm() / w.norm()
    # Expected ~0.0265, measured identically (to 3 decimals) on gaussian,
    # narrow-gaussian and heavy-tailed weights -- e4m3's error is RELATIVE, so
    # unlike int4 it does not track max|w|/rms|w| at all.
    #
    # NOT 0.036. That figure, which this comment previously cited and which
    # appears elsewhere in this repo as "the analytic e4m3 floor", is the
    # WORST-CASE binade: 3 mantissa bits give absolute spacing 2^-3 for values in
    # [1,2), so relative spacing is 0.125/x and 0.125/sqrt(12) only holds at
    # x = 1, the bottom of the binade where resolution is poorest. Averaging over
    # the binade instead -- ||e||/||w|| = sqrt(E[e^2]/E[x^2]) with e ~ U(+-2^-4)
    # and x ~ U(1,2) -- gives 0.024, and real weights spanning many binades
    # measure 0.0265. The distinction matters: against a 0.036 "floor" a genuine
    # 1.4x degradation to 0.036 would read as healthy.
    assert resid < 0.032, resid


def test_block_fp8_scale_is_a_multiplier_not_a_reciprocal():
    """The single most damaging possible error, asserted directly."""
    w = torch.full((128, 128), 10.0)
    q, s = quantize_block_fp8(w)
    # amax is 10, so the multiplier must be 10/448 (small), NOT 44.8 (large).
    assert s.item() == pytest.approx(10.0 / E4M3_MAX, rel=1e-6)
    assert dequantize_block_fp8(q, s).mean().item() == pytest.approx(10.0, rel=1e-2)


def test_block_fp8_uses_the_full_grid():
    """The tile maximum must land on e4m3's largest finite value, not short of
    it (wasted range) and not on inf (saturated)."""
    torch.manual_seed(1)
    w = torch.randn(128, 128) * 3.0
    q, s = quantize_block_fp8(w)
    assert torch.isfinite(q.float()).all()
    assert q.float().abs().max().item() == pytest.approx(E4M3_MAX, rel=1e-6)


def test_block_fp8_all_zero_tile_stays_exactly_zero():
    w = torch.zeros(128, 256)
    w[:, 128:] = 1.0
    q, s = quantize_block_fp8(w)
    back = dequantize_block_fp8(q, s)
    assert torch.equal(back[:, :128], torch.zeros(128, 128))
    assert s[0, 0].item() == 1.0  # finite, not denormal
    assert back[:, 128:].mean().item() == pytest.approx(1.0, rel=1e-3)


def test_block_fp8_pads_and_crops_a_ragged_shape():
    w = torch.randn(130, 300)
    q, s = quantize_block_fp8(w)
    assert q.shape == (130, 300)
    assert s.shape == (2, 3)  # ceil(130/128), ceil(300/128)
    resid = (dequantize_block_fp8(q, s) - w).norm() / w.norm()
    assert resid < 0.05, resid


def test_block_fp8_tiles_independently():
    """A large-magnitude tile must not steal resolution from a small one -- that
    is the entire reason block beats per-tensor."""
    w = torch.zeros(128, 256)
    w[:, :128] = 1e-3
    w[:, 128:] = 1e3
    q, s = quantize_block_fp8(w)
    back = dequantize_block_fp8(q, s)
    assert back[:, :128].mean().item() == pytest.approx(1e-3, rel=1e-2)
    assert back[:, 128:].mean().item() == pytest.approx(1e3, rel=1e-2)


def test_block_fp8_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        quantize_block_fp8(torch.randn(4, 4, 4))


# --------------------------------------------------------------------------
# int4 nibble packing
# --------------------------------------------------------------------------


def test_nibble_round_trip_over_every_value_and_position():
    """All 15 legal values in both nibble positions, so an off-by-one in the
    shift or a missing sign-extension cannot pass."""
    values = torch.arange(-7, 8, dtype=torch.int8).repeat(2, 8)[:, :16]
    packed = pack_nibbles_int8(values)
    assert packed.shape == (2, 8)
    assert packed.dtype == torch.int8
    assert torch.equal(unpack_nibbles_int8(packed), values)


def test_nibble_low_nibble_is_the_even_column():
    """Order is not symmetric: swapping it yields the same file size and a
    plausible histogram, so assert the convention explicitly."""
    values = torch.tensor([[1, 2]], dtype=torch.int8)
    packed = pack_nibbles_int8(values)
    byte = int(packed.view(torch.uint8)[0, 0])
    assert byte & 0x0F == 1, "even column must occupy the low nibble"
    assert (byte >> 4) & 0x0F == 2, "odd column must occupy the high nibble"
    assert byte == 0x21


def test_nibble_negative_values_sign_extend():
    values = torch.tensor([[-1, -7, 0, 7]], dtype=torch.int8)
    packed = pack_nibbles_int8(values)
    assert torch.equal(unpack_nibbles_int8(packed), values)
    # -1 is 0b1111, so the first byte's low nibble must be 0xF.
    assert int(packed.view(torch.uint8)[0, 0]) & 0x0F == 0x0F


def test_nibble_halves_the_width():
    values = torch.randint(-7, 8, (32, 6144), dtype=torch.int8)
    packed = pack_nibbles_int8(values)
    assert packed.shape == (32, 3072)
    assert torch.equal(unpack_nibbles_int8(packed), values)


def test_nibble_rejects_odd_width():
    with pytest.raises(ValueError, match="odd"):
        pack_nibbles_int8(torch.zeros(2, 5, dtype=torch.int8))


def test_nibble_rejects_out_of_range():
    """A value of 8 would wrap to -8 silently, which is exactly the class of bug
    that survives every structural check."""
    with pytest.raises(ValueError, match="4-bit signed range"):
        pack_nibbles_int8(torch.tensor([[8, 0]], dtype=torch.int16))
    with pytest.raises(ValueError, match="4-bit signed range"):
        pack_nibbles_int8(torch.tensor([[-9, 0]], dtype=torch.int16))


def test_nibble_scale_semantics_survive_a_full_expert_round_trip():
    """End-to-end on realistic geometry: group-128 int4 weights repacked and
    recovered bit-exactly, with the group scale applied as a multiplier."""
    torch.manual_seed(2)
    rows, cols, group = 64, 512, 128
    q = torch.randint(-7, 8, (rows, cols), dtype=torch.int8)
    scale = torch.rand(rows, cols // group) + 0.1

    packed = pack_nibbles_int8(q)
    assert packed.shape == (rows, cols // 2)
    recovered = unpack_nibbles_int8(packed)
    assert torch.equal(recovered, q)

    expected = q.float() * scale.repeat_interleave(group, dim=1)
    actual = recovered.float() * scale.repeat_interleave(group, dim=1)
    assert torch.equal(actual, expected)


# --------------------------------------------------------------------------
# int32 -> int8 reinterpret (the conversion fast path)
# --------------------------------------------------------------------------


def _pack_int32_reference(values):
    """Column i at bits [4i, 4i+4) -- compressed-tensors' documented layout."""
    rows, cols = values.shape
    v = (values.to(torch.int32) & 0xF).reshape(rows, cols // 8, 8)
    out = torch.zeros(rows, cols // 8, dtype=torch.int32)
    for i in range(8):
        out |= v[:, :, i] << (4 * i)
    return out


def test_reinterpret_equals_unpack_then_repack():
    """The whole justification for the fast path: same bytes, no compute."""
    torch.manual_seed(11)
    q = torch.randint(-8, 8, (128, 512), dtype=torch.int8)
    packed = _pack_int32_reference(q)
    assert torch.equal(
        repack_int32_to_int8(packed, q.shape[1]), pack_nibbles_int8(q)
    )


def test_reinterpret_round_trips_every_value():
    values = torch.arange(-8, 8, dtype=torch.int8).repeat(4, 4)[:, :64]
    packed = _pack_int32_reference(values)
    out = repack_int32_to_int8(packed, values.shape[1])
    assert torch.equal(unpack_nibbles_int8(out), values)


def test_reinterpret_crops_int32_padding():
    """int32 packing rounds up to a multiple of 8; the tail nibbles are padding
    and must not appear in the output."""
    q = torch.randint(-8, 8, (8, 24), dtype=torch.int8)
    packed = _pack_int32_reference(q)
    assert packed.shape == (8, 3)
    out = repack_int32_to_int8(packed, 20)  # pretend only 20 columns are real
    assert out.shape == (8, 10)
    assert torch.equal(unpack_nibbles_int8(out), q[:, :20])


def test_reinterpret_rejects_wrong_dtype_and_odd_width():
    with pytest.raises(ValueError, match="int32"):
        repack_int32_to_int8(torch.zeros(2, 2, dtype=torch.int8), 4)
    with pytest.raises(ValueError, match="odd"):
        repack_int32_to_int8(torch.zeros(2, 2, dtype=torch.int32), 5)

