"""Tests for the r8-class mixed int4+FP8 serve fix in the M3 re-exporter.

Covers the 2026-07-24 r8 garbage-serving root cause: the saved
``quantization_config`` carried quant-layout FP8 targets and the GPTQ
recipe's broad quant-layout ignore regexes, so vLLM (ignore checked first)
served every FP8 module as unquantized, casting raw fp8 bits into bf16
params. Also covers the indexer-fusion constraint: q/k/v must be dequantized
to BF16 because the M3 plugin fuses them with the bf16 indexer projections.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from pipeline.reexport_minimax_m3_vllm import (
    _FP8_SERVE_TARGETS,
    audit_serve_consistency,
    reexport_checkpoint,
)

L = "language_model.model.layers"

QUANT_LAYOUT_FLOAT_TARGETS = [
    "re:.*language_model[.]layers[.][0-9]+[.]self_attn[.](q|k|v|o)_proj$",
    "re:.*language_model[.]layers[.][0-9]+[.]mlp[.]shared_experts[.]"
    "(gate_up_proj|down_proj)$",
    "re:.*language_model[.]layers[.][0-2][.]mlp[.](gate_up_proj|down_proj)$",
]
BROKEN_IGNORES = [
    f"{L}.3.self_attn.indexer.q_proj",
    f"{L}.3.self_attn.indexer.k_proj",
    "language_model.lm_head",
    "lm_head",
    "re:.*vision_tower.*",
    "re:.*multi_modal_projector.*",
    "re:.*mlp[.]gate$",
    "re:.*block_sparse_moe[.]gate$",
    "re:.*mlp[.]shared_experts[.].*",
    "re:.*block_sparse_moe[.]shared_experts[.].*",
    "re:.*self_attn[.].*",
    "re:.*layers[.][0-2][.].*",
]


def _fp8(shape, seed):
    torch.manual_seed(seed)
    return (torch.randn(shape) * 0.05).to(torch.float8_e4m3fn)


def _scale(out_ch, seed):
    torch.manual_seed(seed)
    return (torch.rand(out_ch, 1) * 0.02 + 0.001).to(torch.float32)


@pytest.fixture
def checkpoint(tmp_path):
    src = tmp_path / "checkpoint"
    src.mkdir()

    # Shard 1: attention (fp8 qkv+o), dense mlp fp8, indexer bf16.
    shard1 = {
        f"{L}.0.self_attn.q_proj.weight": _fp8((4, 8), 0),
        f"{L}.0.self_attn.o_proj.weight": _fp8((8, 4), 1),
        f"{L}.0.self_attn.o_proj.weight_scale": _scale(8, 1),
        f"{L}.0.mlp.gate_proj.weight": _fp8((6, 8), 2),
        f"{L}.0.mlp.gate_proj.weight_scale": _scale(6, 2),
        f"{L}.3.self_attn.index_q_proj.weight": torch.randn(4, 8).bfloat16(),
    }
    # Shard 2: q_proj scale lives in a DIFFERENT shard than its weight;
    # routed expert packed int4; shared experts fp8; router bf16.
    shard2 = {
        f"{L}.0.self_attn.q_proj.weight_scale": _scale(4, 0),
        f"{L}.3.block_sparse_moe.experts.0.gate_proj.weight_packed": torch.ones(
            (16, 2), dtype=torch.int32
        ),
        f"{L}.3.block_sparse_moe.experts.0.gate_proj.weight_scale": _scale(16, 3),
        f"{L}.3.block_sparse_moe.experts.0.gate_proj.weight_shape": torch.tensor(
            [16, 16], dtype=torch.int64
        ),
        f"{L}.3.block_sparse_moe.shared_experts.gate_proj.weight": _fp8((6, 8), 4),
        f"{L}.3.block_sparse_moe.shared_experts.gate_proj.weight_scale": _scale(6, 4),
        f"{L}.3.block_sparse_moe.gate.weight": torch.randn(2, 8).bfloat16(),
    }
    weight_map = {}
    for name, tensors in (("model-00001-of-00002.safetensors", shard1),
                          ("model-00002-of-00002.safetensors", shard2)):
        save_file(tensors, src / name)
        weight_map.update({key: name for key in tensors})
    total = sum(
        t.numel() * t.element_size() for t in {**shard1, **shard2}.values()
    )
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )
    config = {
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "format": "float-quantized",
                    "targets": list(QUANT_LAYOUT_FLOAT_TARGETS),
                    "weights": {"type": "float", "num_bits": 8},
                },
                "group_1": {
                    "format": "pack-quantized",
                    "targets": ["Linear"],
                    "weights": {"type": "int", "num_bits": 4},
                },
            },
            "format": "mixed-precision",
            "ignore": list(BROKEN_IGNORES),
        },
    }
    (src / "config.json").write_text(json.dumps(config))
    return src


def test_fp8_serve_fix_end_to_end(checkpoint, tmp_path):
    out = tmp_path / "out"
    result = reexport_checkpoint(checkpoint, out, fp8_serve_fix=True)
    assert result["dequantized"] == 1  # the lone q_proj
    assert result["dropped"] == 1
    assert result["renamed"] == 3  # expert gate_proj packed/scale/shape -> w1

    from safetensors import safe_open

    # q_proj dequantized to BF16 with correct numerics; scale dropped.
    with safe_open(out / "model-00001-of-00002.safetensors", "pt") as fh:
        keys = set(fh.keys())
        deq = fh.get_tensor(f"{L}.0.self_attn.q_proj.weight")
        o_w = fh.get_tensor(f"{L}.0.self_attn.o_proj.weight")
    assert deq.dtype == torch.bfloat16
    expected = (
        _fp8((4, 8), 0).to(torch.float32) * _scale(4, 0)
    ).to(torch.bfloat16)
    assert torch.equal(deq, expected)
    assert o_w.dtype == torch.float8_e4m3fn  # o_proj untouched
    with safe_open(out / "model-00002-of-00002.safetensors", "pt") as fh:
        keys2 = set(fh.keys())
    assert f"{L}.0.self_attn.q_proj.weight_scale" not in keys2
    assert f"{L}.3.block_sparse_moe.experts.0.w1.weight_packed" in keys2

    # Index consistent with shards; total_size recomputed.
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == keys | keys2
    fp8_numel = 4 * 8
    scale_bytes = 4 * 1 * 4
    old_total = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text()
    )["metadata"]["total_size"]
    assert index["metadata"]["total_size"] == old_total + fp8_numel - scale_bytes

    # Config rewritten to serve layout.
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    assert qc["config_groups"]["group_0"]["targets"] == _FP8_SERVE_TARGETS
    ignore = qc["ignore"]
    assert "re:.*self_attn[.].*" not in ignore
    assert "re:.*layers[.][0-2][.].*" not in ignore
    assert not any(".self_attn.indexer." in e for e in ignore)
    assert (
        "re:.*language_model[.]model[.]layers[.][0-9]+[.]self_attn[.]"
        "(qkv_proj|q_proj|k_proj|v_proj)$" in ignore
    )
    assert "re:.*self_attn[.]index_(q|k)_proj$" in ignore
    assert "re:.*block_sparse_moe[.]gate$" in ignore  # router still ignored


def test_fused_prefixes_match_directly(checkpoint, tmp_path):
    """Regression for the v2 serve crash: the M3 NVIDIA plugin class has no
    packed_modules_mapping, so vLLM matches each fused module prefix
    DIRECTLY against targets/ignores — shard-name-only patterns match
    nothing and the module falls to the int4 Linear catch-all."""
    import re as _re

    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    out = tmp_path / "out"
    reexport_checkpoint(checkpoint, out, fp8_serve_fix=True)
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    targets = qc["config_groups"]["group_0"]["targets"]

    def matches(name):
        return any(_re.match(t[3:], name) for t in targets if t.startswith("re:"))

    # fp8 fused modules must match the float group by their fused prefix
    assert matches(f"{L}.3.block_sparse_moe.shared_experts.gate_up_proj")
    assert matches(f"{L}.1.mlp.gate_up_proj")
    assert matches(f"{L}.3.self_attn.o_proj")
    # the (bf16) fused qkv must be ignored by its fused prefix, no mapping
    assert should_ignore_layer(
        f"{L}.3.self_attn.qkv_proj", ignore=qc["ignore"], fused_mapping={}
    )
    # and the int4 experts must match neither ignore nor the float group
    assert not should_ignore_layer(
        f"{L}.3.block_sparse_moe.experts.0.w1",
        ignore=qc["ignore"],
        fused_mapping={},
    )
    assert not matches(f"{L}.3.block_sparse_moe.experts.0.w1")


def test_audit_rejects_ignored_fp8(checkpoint, tmp_path):
    out = tmp_path / "out"
    reexport_checkpoint(checkpoint, out, fp8_serve_fix=True)
    cfg_path = out / "config.json"
    config = json.loads(cfg_path.read_text())
    config["quantization_config"]["ignore"].append("re:.*self_attn[.]o_proj$")
    cfg_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fp8 stored but ignored"):
        audit_serve_consistency(out)


def test_audit_rejects_unignored_plain_linear(checkpoint, tmp_path):
    out = tmp_path / "out"
    reexport_checkpoint(checkpoint, out, fp8_serve_fix=True)
    cfg_path = out / "config.json"
    config = json.loads(cfg_path.read_text())
    config["quantization_config"]["ignore"].remove(
        "re:.*block_sparse_moe[.]gate$"
    )
    config["quantization_config"]["ignore"].remove("re:.*mlp[.]gate$")
    cfg_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="not ignored"):
        audit_serve_consistency(out)


def test_plain_reexport_unchanged_without_flag(checkpoint, tmp_path):
    out = tmp_path / "out"
    result = reexport_checkpoint(checkpoint, out)
    assert result["dequantized"] == 0
    assert result["dropped"] == 0
    from safetensors import safe_open

    with safe_open(out / "model-00001-of-00002.safetensors", "pt") as fh:
        assert fh.get_tensor(
            f"{L}.0.self_attn.q_proj.weight"
        ).dtype == torch.float8_e4m3fn
    with safe_open(out / "model-00002-of-00002.safetensors", "pt") as fh:
        assert f"{L}.0.self_attn.q_proj.weight_scale" in set(fh.keys())
    qc = json.loads((out / "config.json").read_text())["quantization_config"]
    assert qc["ignore"] == BROKEN_IGNORES  # untouched
