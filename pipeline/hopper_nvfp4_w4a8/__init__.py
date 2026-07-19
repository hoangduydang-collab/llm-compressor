"""Hopper NVFP4-to-W4A8 fallback tooling."""

from pipeline.hopper_nvfp4_w4a8.reference import (
    convert_e2m1_group_to_e4m3,
    decode_e2m1,
    decode_e4m3fn,
    encode_e4m3fn,
    persistent_byte_report,
)

__all__ = [
    "convert_e2m1_group_to_e4m3",
    "decode_e2m1",
    "decode_e4m3fn",
    "encode_e4m3fn",
    "persistent_byte_report",
]
