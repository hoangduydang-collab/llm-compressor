"""Tests for portable MiniMax-M3 routed-expert checkpoint key exports."""

import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from pipeline.reexport_minimax_m3_vllm import (
    rename_routed_expert_key,
    rewrite_safetensors_shard,
)


def test_rename_routed_expert_key_uses_vllm_w123_layout_only_for_routed_experts():
    routed = (
        "language_model.model.layers.10.block_sparse_moe."
        "experts.7.gate_proj.weight_packed"
    )
    shared = (
        "language_model.model.layers.10.block_sparse_moe."
        "shared_experts.gate_proj.weight"
    )

    assert rename_routed_expert_key(routed) == (
        "language_model.model.layers.10.block_sparse_moe."
        "experts.7.w1.weight_packed"
    )
    assert rename_routed_expert_key(
        routed.replace("gate_proj", "up_proj")
    ).endswith("experts.7.w3.weight_packed")
    assert rename_routed_expert_key(
        routed.replace("gate_proj", "down_proj")
    ).endswith("experts.7.w2.weight_packed")
    assert rename_routed_expert_key(shared) == shared


def test_rewrite_safetensors_shard_renames_header_without_changing_tensor_data(
    tmp_path,
):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    routed_key = (
        "language_model.model.layers.3.block_sparse_moe."
        "experts.0.gate_proj.weight_packed"
    )
    shared_key = (
        "language_model.model.layers.3.block_sparse_moe."
        "shared_experts.gate_proj.weight"
    )
    tensors = {
        routed_key: torch.arange(8, dtype=torch.uint8),
        shared_key: torch.arange(4, dtype=torch.bfloat16),
    }
    save_file(tensors, str(source), metadata={"format": "pt"})

    renamed = rewrite_safetensors_shard(source, output)

    assert renamed == 1
    with safe_open(str(output), framework="pt") as handle:
        assert set(handle.keys()) == {
            routed_key.replace("gate_proj", "w1"),
            shared_key,
        }
        assert torch.equal(
            handle.get_tensor(routed_key.replace("gate_proj", "w1")),
            tensors[routed_key],
        )
        assert torch.equal(handle.get_tensor(shared_key), tensors[shared_key])

    header_length = int.from_bytes(output.read_bytes()[:8], "little")
    header = json.loads(output.read_bytes()[8 : 8 + header_length])
    assert routed_key.replace("gate_proj", "w1") in header
