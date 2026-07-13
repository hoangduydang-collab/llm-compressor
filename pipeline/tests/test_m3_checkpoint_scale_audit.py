"""CPU tests for MiniMax-M3 checkpoint smoothing audits."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from pipeline.m3_checkpoint_scale_audit import (
    _component_suffixes,
    audit_checkpoints,
    resolve_suffix,
)


def _checkpoint(path: Path, *, scale: torch.Tensor) -> None:
    path.mkdir()
    prefix = "model.language_model.layers.8."
    base_norm = torch.tensor([0.0, 0.5])
    base_router = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    base_shared = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    tensors = {
        prefix + "post_attention_layernorm.weight": (1.0 + base_norm) / scale - 1.0,
        prefix + "mlp.gate.weight": base_router * scale.reshape(1, -1),
        prefix + "mlp.shared_experts.gate_up_proj.weight": (
            base_shared * scale.reshape(1, -1)
        ),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, path / shard)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )


def test_suffix_resolution_requires_one_match():
    assert resolve_suffix({"a.x": "one"}, "x") == "a.x"


def test_component_suffixes_cover_transformers_and_reference_names():
    router = _component_suffixes(8, "router")
    assert "language_model.layers.8.mlp.gate.weight" in router
    assert "model.layers.8.block_sparse_moe.gate.weight" in router


def test_audit_recovers_exact_compensation(tmp_path: Path):
    base = tmp_path / "base"
    reference = tmp_path / "reference"
    awq = tmp_path / "awq"
    gptq = tmp_path / "gptq"
    _checkpoint(base, scale=torch.ones(2))
    _checkpoint(reference, scale=torch.ones(2))
    _checkpoint(awq, scale=torch.tensor([0.25, 4.0]))
    _checkpoint(gptq, scale=torch.ones(2))

    result = audit_checkpoints(base, reference, awq, gptq, [8])

    layer = result["awq"]["layers"]["8"]
    assert layer["router_compensation"]["relative_l2_error"] == 0.0
    assert layer["shared_gate_up_compensation"]["relative_l2_error"] == 0.0
    assert result["gptq"]["layers"]["8"]["normalization"]["candidate"][
        "norm"
    ] == result["gptq"]["layers"]["8"]["normalization"]["base"]["norm"]
