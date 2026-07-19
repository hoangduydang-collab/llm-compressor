from fractions import Fraction

import pytest

from pipeline.hopper_nvfp4_w4a8.reference import (
    convert_e2m1_group_to_e4m3,
    decode_e2m1,
    decode_e4m3fn,
    encode_e4m3fn,
    interleave_k16_scale_rows,
    persistent_byte_report,
    wgmma_b_register_k16_groups,
)


@pytest.mark.parametrize(
    ("code", "expected"),
    enumerate(
        (
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
    ),
)
def test_decode_e2m1_all_codepoints(code, expected):
    assert decode_e2m1(code) == expected


@pytest.mark.parametrize("code", (-1, 16))
def test_decode_e2m1_rejects_non_codepoints(code):
    with pytest.raises(ValueError, match="E2M1 code"):
        decode_e2m1(code)


@pytest.mark.parametrize(
    ("code", "expected"),
    (
        (0x00, Fraction(0)),
        (0x80, Fraction(0)),
        (0x01, Fraction(1, 512)),
        (0x08, Fraction(1, 64)),
        (0x38, Fraction(1)),
        (0x7E, Fraction(448)),
        (0xFE, Fraction(-448)),
        (0x7F, None),
        (0xFF, None),
    ),
)
def test_decode_e4m3fn_special_and_boundary_values(code, expected):
    assert decode_e4m3fn(code) == expected


@pytest.mark.parametrize("code", (-1, 256))
def test_decode_e4m3fn_rejects_non_bytes(code):
    with pytest.raises(ValueError, match="E4M3FN code"):
        decode_e4m3fn(code)


@pytest.mark.parametrize(
    ("value", "expected_code"),
    (
        (Fraction(0), 0x00),
        (Fraction(1, 512), 0x01),
        (Fraction(1), 0x38),
        (Fraction(-1), 0xB8),
        (Fraction(448), 0x7E),
        (Fraction(-448), 0xFE),
        (Fraction(1000), 0x7E),
        (Fraction(-1000), 0xFE),
    ),
)
def test_encode_e4m3fn_exact_values_and_saturation(value, expected_code):
    assert encode_e4m3fn(value) == expected_code


@pytest.mark.parametrize(
    ("midpoint", "expected_code"),
    (
        (Fraction(1, 1024), 0x00),
        (Fraction(17, 16), 0x38),
        (Fraction(19, 16), 0x3A),
        (Fraction(-17, 16), 0xB8),
        (Fraction(-19, 16), 0xBA),
    ),
)
def test_encode_e4m3fn_rounds_midpoints_to_even_mantissa(midpoint, expected_code):
    assert encode_e4m3fn(midpoint) == expected_code


def test_k16_scales_are_applied_to_only_their_own_half():
    codes = (0x02,) * 16
    scale_one = encode_e4m3fn(Fraction(1))
    scale_two = encode_e4m3fn(Fraction(2))

    first_then_second = convert_e2m1_group_to_e4m3(
        codes, scale_one
    ) + convert_e2m1_group_to_e4m3(codes, scale_two)
    second_then_first = convert_e2m1_group_to_e4m3(
        codes, scale_two
    ) + convert_e2m1_group_to_e4m3(codes, scale_one)

    expected_first = (encode_e4m3fn(Fraction(1, 8)),) * 16
    expected_second = (encode_e4m3fn(Fraction(1, 4)),) * 16
    assert first_then_second[:16] == expected_first
    assert first_then_second[16:] == expected_second
    assert second_then_first[:16] == expected_second
    assert second_then_first[16:] == expected_first


def test_group_conversion_rejects_bad_group_and_scale():
    with pytest.raises(ValueError, match="exactly 16"):
        convert_e2m1_group_to_e4m3((0x02,) * 15, 0x38)
    with pytest.raises(ValueError, match="finite"):
        convert_e2m1_group_to_e4m3((0x02,) * 16, 0x7F)


def test_s2r_scale_rows_are_interleaved_per_n16_fragment():
    assert interleave_k16_scale_rows(
        (0x10, 0x11, 0x12, 0x13), (0x20, 0x21, 0x22, 0x23)
    ) == (
        0x10,
        0x11,
        0x20,
        0x21,
        0x12,
        0x13,
        0x22,
        0x23,
    )


def test_dequant_pair_reversal_does_not_cross_k16_scale_boundary():
    assert wgmma_b_register_k16_groups() == (0, 0, 1, 1)


@pytest.mark.parametrize("row0,row1", (((1,), (2,)), ((1, 2), (3, 4, 5))))
def test_s2r_scale_row_interleave_rejects_invalid_rows(row0, row1):
    with pytest.raises(ValueError, match="equal, even"):
        interleave_k16_scale_rows(row0, row1)


def test_persistent_byte_report_preserves_packed_checkpoint_size():
    report = persistent_byte_report(128, 128)

    assert report == {"checkpoint": 9220, "transformed": 9220, "ratio": 1.0}


@pytest.mark.parametrize(
    ("n", "k", "num_global_scales"),
    ((0, 16, 1), (1, 15, 1), (1, 16, 0)),
)
def test_persistent_byte_report_rejects_invalid_shapes(n, k, num_global_scales):
    with pytest.raises(ValueError):
        persistent_byte_report(n, k, num_global_scales)
