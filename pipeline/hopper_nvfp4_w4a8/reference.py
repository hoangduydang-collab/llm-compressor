"""Bit-exact scalar reference for the Hopper NVFP4 W4A8 contract.

The runtime kernel converts packed E2M1 values to E4M3FN registers after
applying one E4M3FN scale per K16 group.  These helpers intentionally use
``Fraction`` throughout so CPU tests do not inherit host floating-point
rounding behavior.
"""

from fractions import Fraction
from typing import Iterable

_E2M1_VALUES = (
    Fraction(0),
    Fraction(1, 2),
    Fraction(1),
    Fraction(3, 2),
    Fraction(2),
    Fraction(3),
    Fraction(4),
    Fraction(6),
    Fraction(0),
    Fraction(-1, 2),
    Fraction(-1),
    Fraction(-3, 2),
    Fraction(-2),
    Fraction(-3),
    Fraction(-4),
    Fraction(-6),
)


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent)
    return Fraction(1, 1 << -exponent)


def decode_e2m1(code: int) -> Fraction:
    """Decode one unsigned nibble containing an NVIDIA E2M1 value."""

    if not isinstance(code, int) or not 0 <= code < len(_E2M1_VALUES):
        raise ValueError(f"E2M1 code must be an integer in [0, 15], got {code!r}")
    return _E2M1_VALUES[code]


def decode_e4m3fn(code: int) -> Fraction | None:
    """Decode one E4M3FN byte, returning ``None`` for either NaN code."""

    if not isinstance(code, int) or not 0 <= code <= 0xFF:
        raise ValueError(f"E4M3FN code must be an integer byte, got {code!r}")

    sign = -1 if code & 0x80 else 1
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07

    if exponent == 0x0F and mantissa == 0x07:
        return None
    if exponent == 0:
        magnitude = Fraction(mantissa, 8) * _power_of_two(-6)
    else:
        magnitude = Fraction(8 + mantissa, 8) * _power_of_two(exponent - 7)
    return sign * magnitude


def _finite_codes_for(value: Fraction) -> tuple[tuple[int, Fraction], ...]:
    sign_bit = 0x80 if value < 0 else 0
    candidates = []
    for code in range(sign_bit, sign_bit + 0x80):
        decoded = decode_e4m3fn(code)
        if decoded is not None:
            candidates.append((code, decoded))
    return tuple(candidates)


def encode_e4m3fn(value: Fraction) -> int:
    """Round an exact finite value to E4M3FN using nearest-even semantics."""

    if not isinstance(value, Fraction):
        raise TypeError(f"value must be Fraction, got {type(value).__name__}")
    if value == 0:
        return 0x00

    candidates = _finite_codes_for(value)
    code, _ = min(
        candidates,
        key=lambda item: (
            abs(item[1] - value),
            item[0] & 0x01,
            item[0],
        ),
    )
    return code


def convert_e2m1_group_to_e4m3(
    codes: Iterable[int], scale_code: int
) -> tuple[int, ...]:
    """Apply one E4M3FN scale to exactly one K16 E2M1 group.

    The division by eight is paired with the single global-scale compensation
    in the Humming overlay: ``effective_global_scale = checkpoint_scale * 8``.
    """

    group = tuple(codes)
    if len(group) != 16:
        raise ValueError(f"group must contain exactly 16 E2M1 codes, got {len(group)}")
    scale = decode_e4m3fn(scale_code)
    if scale is None:
        raise ValueError("group scale must be a finite E4M3FN value")
    return tuple(encode_e4m3fn(decode_e2m1(code) * scale / 8) for code in group)


def interleave_k16_scale_rows(
    first_k16: Iterable[int], second_k16: Iterable[int]
) -> tuple[int, ...]:
    """Model the S2R byte layout consumed by one WGMMA B fragment.

    Each N16 subfragment owns two scale bytes from each adjacent K16 row.  The
    dequantizer emits B registers in operand order, so the loader presents the
    bytes as ``[K0, K0, K1, K1]`` for every N16 subfragment.
    """

    first = tuple(first_k16)
    second = tuple(second_k16)
    if len(first) != len(second) or len(first) == 0 or len(first) % 2:
        raise ValueError("K16 scale rows must have equal, even, non-zero lengths")
    if any(
        not isinstance(code, int) or not 0 <= code <= 0xFF for code in first + second
    ):
        raise ValueError("K16 scale rows must contain integer bytes")
    result: list[int] = []
    for offset in range(0, len(first), 2):
        result.extend(first[offset : offset + 2])
        result.extend(second[offset : offset + 2])
    return tuple(result)


def wgmma_b_register_k16_groups() -> tuple[int, int, int, int]:
    """Derive the K16 group of each E4M3 WGMMA B register.

    Pinned Humming's ``dequant_b1248<E2M1, E4M3>`` reads source uint32 words
    ``[0, 0, 1, 1]`` for its four outputs and reverses only within each pair,
    placing them at output register indices ``[1, 0, 3, 2]``.  The pair-local
    reversal therefore cannot cross the K16 boundary: output registers 0/1 use
    the first K16 scale and 2/3 use the adjacent K16 scale.
    """

    source_word_for_iteration = (0, 0, 1, 1)
    output_register_for_iteration = (1, 0, 3, 2)
    result = [-1] * 4
    for source_word, output_register in zip(
        source_word_for_iteration, output_register_for_iteration, strict=True
    ):
        result[output_register] = source_word
    return tuple(result)  # type: ignore[return-value]


def persistent_byte_report(
    n: int, k: int, num_global_scales: int = 1
) -> dict[str, int | float]:
    """Compare checkpoint and transformed persistent packed-storage bytes."""

    if not isinstance(n, int) or not isinstance(k, int):
        raise ValueError("n and k must be integers")
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    if k % 16:
        raise ValueError("k must be divisible by the K16 scale group size")
    if not isinstance(num_global_scales, int) or num_global_scales <= 0:
        raise ValueError("num_global_scales must be a positive integer")

    checkpoint = n * k // 2 + n * k // 16 + 4 * num_global_scales
    transformed = n * k // 2 + n * k // 16 + 4 * num_global_scales
    return {
        "checkpoint": checkpoint,
        "transformed": transformed,
        "ratio": transformed / checkpoint,
    }
