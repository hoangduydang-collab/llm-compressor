"""Unit tests for MiniMax-M3-only serve env defaults (no GPU / no vLLM)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.serve_verify import (
    _install_minimax_m3_site_diagnostics,
    apply_minimax_m3_serve_env,
)
from pipeline import vllm_m3_patches


def _write_ckpt(tmp_path: Path, *, model_type: str, architectures: list[str] | None = None) -> Path:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    cfg: dict = {"model_type": model_type}
    if architectures is not None:
        cfg["architectures"] = architectures
    (ckpt / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return ckpt


def test_apply_minimax_m3_serve_env_sets_stream_disable(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", raising=False)
    ckpt = _write_ckpt(tmp_path, model_type="minimax_m3_vl")

    applied = apply_minimax_m3_serve_env(ckpt)

    assert applied == ["VLLM_DISABLE_SHARED_EXPERTS_STREAM=1"]
    assert os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "1"


def test_apply_minimax_m3_serve_env_preserves_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0")
    ckpt = _write_ckpt(
        tmp_path,
        model_type="minimax_m3_vl",
        architectures=["MiniMaxM3ForCausalLM"],
    )

    applied = apply_minimax_m3_serve_env(ckpt)

    assert applied == []
    assert os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "0"


def test_apply_minimax_m3_serve_env_skips_non_m3(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", raising=False)
    ckpt = _write_ckpt(
        tmp_path,
        model_type="qwen3_moe",
        architectures=["Qwen3MoeForCausalLM"],
    )

    applied = apply_minimax_m3_serve_env(ckpt)

    assert applied == []
    assert "VLLM_DISABLE_SHARED_EXPERTS_STREAM" not in os.environ


def test_dead_runtime_cudagraph_helpers_removed():
    """Persistent site-packages patches remain; unwired runtime duplicates do not."""
    assert not hasattr(vllm_m3_patches, "patch_vllm_m3_fused_ar_for_cudagraph")
    assert not hasattr(vllm_m3_patches, "patch_vllm_m3_moe_router_for_cudagraph")
    assert hasattr(vllm_m3_patches, "patch_vllm_w4a8_swigluoai_uninterleave")


def _diagnostic_install_calls(quantization_config: dict) -> tuple[list[str], dict]:
    with TemporaryDirectory() as raw_tmp:
        ckpt = Path(raw_tmp)
        (ckpt / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "minimax_m3_vl",
                    "quantization_config": quantization_config,
                }
            ),
            encoding="utf-8",
        )
        calls: list[str] = []
        status = _install_minimax_m3_site_diagnostics(
            ckpt,
            diagnostic_installer=lambda: calls.append("quality_diagnostics")
            or "diagnostics installed",
            w4a8_patch_installer=lambda: calls.append("w4a8_patches"),
        )
        return calls, status


def test_m3_w4a16_installs_diagnostics_without_w4a8_patches():
    calls, status = _diagnostic_install_calls(
        {
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4},
                    "input_activations": None,
                }
            }
        }
    )

    assert calls == ["quality_diagnostics"]
    assert status["diagnostics"] == "diagnostics installed"
    assert status["w4a8_patches"] is False


def test_m3_w4a8_installs_diagnostics_and_required_patches():
    calls, status = _diagnostic_install_calls(
        {
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4},
                    "input_activations": {"num_bits": 8},
                }
            }
        }
    )

    assert calls == ["quality_diagnostics", "w4a8_patches"]
    assert status["w4a8_patches"] is True
