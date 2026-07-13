"""Fail-fast compatibility checks that do not require calibration data."""

from .quantization import (  # noqa: F401
    QuantizationCompatibilityReport,
    analyze_quantization_compatibility,
)
