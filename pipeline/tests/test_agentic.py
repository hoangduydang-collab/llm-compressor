"""Unit tests for agentic eval wiring (no GPU / no tau2 run)."""

from pipeline.config import AgenticConfig
from pipeline.evalsuite.agentic import _agentic_ready, _resolve_calibration_script


def test_agentic_ready_when_disabled():
    ok, reason = _agentic_ready(AgenticConfig(enabled=False))
    assert ok is False
    assert "false" in reason


def test_agentic_ready_requires_tau2_dir():
    ag = AgenticConfig(
        enabled=True,
        user_base="https://example.com/v1",
        user_model="user",
        user_key_file=__file__,
    )
    ok, reason = _agentic_ready(ag)
    assert ok is False
    assert "tau2_dir" in reason


def test_resolve_calibration_script_explicit_missing():
    ag = AgenticConfig(calibration_script="/no/such/run_calibration.sh")
    assert _resolve_calibration_script(ag) is None
