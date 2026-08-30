"""Numeric kernels for converting a compressed-tensors W4AFP8 checkpoint into
SGLang's ``quant_method: w4afp8`` layout.

Separated from the file-shuffling plumbing because these two functions are where
a silent defect would live: both produce well-formed output for wrong input, and
neither failure mode raises. They are unit-tested against round-trip identities
in ``pipeline/tests/test_sglang_w4afp8_kernels.py``.

FORMAT FACTS, each established from a primary source rather than assumed:

* ``weight_scale_inv`` is a MULTIPLIER, not a reciprocal, despite the name.
  sglang/srt/layers/quantization/fp8_utils.py::block_quant_dequant computes
  ``x_q_block.to(float32) * x_scale_repeat``, where the scale is expanded with
  ``repeat_interleave(block_n, -2).repeat_interleave(block_k, -1)``. Getting
  this backwards would produce a loadable checkpoint that serves noise.
* Non-expert block scales are fp32 and shaped
  ``[ceil(out/128), ceil(in/128)]``. ``W4AFp8Config`` HARDCODES
  ``weight_block_size = [128, 128]``, so ``Fp8LinearMethod.block_quant`` is
  always True on this path and there is no per-channel fallback -- a per-channel
  ``weight_scale`` simply fails to load.
* e4m3 max finite magnitude is 448.0.
* Expert int4 values are stored two-per-byte as signed nibbles in [-7, +7]
  (-8 unused), low nibble in the EVEN column. Determined empirically from
  PhalaCloud/GLM-5.2-W4AFP8 by testing eight orderings and confirming with an
  independent least-squares refit and a value histogram.
"""

from __future__ import annotations

import torch

E4M3_MAX = 448.0
DEFAULT_BLOCK = (128, 128)


def quantize_block_fp8(
    weight: torch.Tensor, block: tuple[int, int] = DEFAULT_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize ``weight`` [out, in] to e4m3 plus an fp32 block multiplier.

    Returns ``(qweight, scale_inv)`` with ``qweight`` float8_e4m3fn of the same
    shape and ``scale_inv`` fp32 of shape ``[ceil(out/bn), ceil(in/bk)]``, such
    that ``qweight.float() * expand(scale_inv) ~= weight``.

    Padding: a dimension that is not a multiple of its block size is zero-padded
    to compute tile maxima and then cropped, which is what the loader's
    ``(out + bn - 1) // bn`` scale shape implies. GLM's non-expert shapes are all
    multiples of 128, so this path is defensive rather than exercised.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    block_n, block_k = block
    n_out = (out_features + block_n - 1) // block_n
    n_in = (in_features + block_k - 1) // block_k

    work = weight.float()
    pad_out = n_out * block_n - out_features
    pad_in = n_in * block_k - in_features
    if pad_out or pad_in:
        work = torch.nn.functional.pad(work, (0, pad_in, 0, pad_out))

    tiles = work.view(n_out, block_n, n_in, block_k)
    amax = tiles.abs().amax(dim=(1, 3))
    # An all-zero tile has amax 0 and any positive scale reproduces it exactly;
    # 1.0 keeps the stored scale finite and the dequantized tile exactly zero.
    # Using a tiny epsilon instead would make the scale denormal and the
    # round-trip lossy for a tile that should be exact.
    scale_inv = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))

    q = tiles / scale_inv[:, None, :, None]
    # Clamp before the cast: division by amax/448 lands the tile maximum on
    # exactly 448, but fp32 rounding can overshoot it, and float8_e4m3fn
    # saturates overshoot to inf rather than to 448.
    q = q.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

    qweight = q.reshape(n_out * block_n, n_in * block_k)[:out_features, :in_features]
    return qweight.contiguous(), scale_inv.contiguous()


def dequantize_block_fp8(
    qweight: torch.Tensor,
    scale_inv: torch.Tensor,
    block: tuple[int, int] = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Inverse of :func:`quantize_block_fp8`, mirroring SGLang's
    ``block_quant_dequant`` exactly (multiply, then crop)."""
    block_n, block_k = block
    out_features, in_features = qweight.shape
    expanded = scale_inv.repeat_interleave(block_n, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    return qweight.float() * expanded[:out_features, :in_features]


def pack_nibbles_int8(values: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit ``values`` [out, in] into int8 [out, in // 2].

    Adjacent columns share a byte, with the EVEN column in the low nibble. Input
    must already be integral and within [-8, 7]; the caller is responsible for
    that because a silent wrap here is indistinguishable from valid data.
    """
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D tensor, got shape {tuple(values.shape)}")
    if values.shape[1] % 2:
        raise ValueError(
            f"input width {values.shape[1]} is odd, so nibbles cannot be paired"
        )
    work = values.to(torch.int16)
    if int(work.min()) < -8 or int(work.max()) > 7:
        raise ValueError(
            f"values outside 4-bit signed range: min={int(work.min())} "
            f"max={int(work.max())}"
        )

    low = work[:, 0::2] & 0x0F
    high = work[:, 1::2] & 0x0F
    packed = (low | (high << 4)).to(torch.uint8)
    return packed.view(torch.int8).contiguous()


def unpack_nibbles_int8(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_nibbles_int8`; returns int8 [out, 2 * in]."""
    raw = packed.view(torch.uint8).to(torch.int16)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    # Sign-extend 4 bits: 8..15 represent -8..-1.
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)
    interleaved = torch.stack([low, high], dim=-1)
    return interleaved.reshape(packed.shape[0], -1).to(torch.int8).contiguous()
