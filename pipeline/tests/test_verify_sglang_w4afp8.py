"""Tests for the converted-checkpoint verifier.

A verifier that cannot fail is decoration, so most of these corrupt one thing
and require a non-zero exit. The corruptions are the real failure modes: a
mis-encoded expert nibble, a dropped AWQ fold, a leftover source tensor, a
recomputed rather than renamed scale.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from pipeline.sglang_w4afp8_kernels import (  # noqa: E402
    pack_nibbles_int8,
    quantize_block_fp8,
)
from pipeline.verify_sglang_w4afp8 import verify  # noqa: E402

HIDDEN = 256
INTER = 128
GROUP = 128


def _pack_int32(values):
    rows, cols = values.shape
    v = (values.to(torch.int32) & 0xF).reshape(rows, cols // 8, 8)
    out = torch.zeros(rows, cols // 8, dtype=torch.int32)
    for i in range(8):
        out |= v[:, :, i] << (4 * i)
    return out


def _unpack_int32(packed, shape):
    rows = packed.shape[0]
    words = packed.to(torch.int64)
    nib = torch.stack([(words >> (4 * i)) & 0xF for i in range(8)], dim=-1)
    flat = nib.reshape(rows, -1)[:, : shape[1]]
    return torch.where(flat > 7, flat - 16, flat).to(torch.int8)


@pytest.fixture(autouse=True)
def _use_matching_unpacker(monkeypatch):
    """The verifier delegates to compressed-tensors, which is not installed
    locally. Inject the packer's own inverse so these tests exercise the
    verifier's LOGIC; the real int32 convention is covered on real tensors by
    the converter's --conformance-only path."""
    import pipeline.verify_sglang_w4afp8 as mod

    monkeypatch.setattr(mod, "_source_unpacker", lambda warnings: _unpack_int32)


def _write(path, tensors, extra=None):
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "model-00001-of-00001.safetensors"),
              metadata={"format": "pt"})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    (path / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": total},
            "weight_map": {k: "model-00001-of-00001.safetensors"
                           for k in tensors},
        }),
        encoding="utf-8",
    )
    config = {"architectures": ["GlmMoeDsaForCausalLM"]}
    config.update(extra or {})
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


@pytest.fixture
def pair(tmp_path):
    """A source compressed-tensors checkpoint and a correct conversion of it."""
    torch.manual_seed(21)
    src_dir, dst_dir = tmp_path / "src", tmp_path / "dst"

    q = torch.randint(-8, 8, (INTER, HIDDEN), dtype=torch.int8)
    scale = (torch.rand(INTER, HIDDEN // GROUP) + 0.1).bfloat16()
    fp8_w = torch.randn(INTER, HIDDEN)
    per_ch = fp8_w.abs().amax(dim=1, keepdim=True) / 448.0
    norm = torch.rand(HIDDEN).bfloat16()

    ep = "model.layers.3.mlp.experts.0.gate_proj"
    fp = "model.layers.3.self_attn.o_proj"
    src = {
        f"{ep}.weight_packed": _pack_int32(q),
        f"{ep}.weight_scale": scale,
        f"{ep}.weight_shape": torch.tensor([INTER, HIDDEN], dtype=torch.int64),
        f"{fp}.weight": (fp8_w / per_ch).to(torch.float8_e4m3fn),
        f"{fp}.weight_scale": per_ch.bfloat16(),
        "model.layers.3.input_layernorm.weight": norm,
        "model.layers.3.mlp.gate.weight": torch.randn(8, HIDDEN).bfloat16(),
    }
    _write(src_dir, src, {"quantization_config": {
        "quant_method": "compressed-tensors", "ignore": ["lm_head"]}})

    bq, bs = quantize_block_fp8(fp8_w, (128, 128))
    dst = {
        f"{ep}.weight": pack_nibbles_int8(q),
        f"{ep}.weight_scale_inv": scale,
        f"{ep}.input_scale": torch.ones(1, dtype=torch.bfloat16),
        f"{fp}.weight": bq,
        f"{fp}.weight_scale_inv": bs,
        "model.layers.3.input_layernorm.weight": norm,
        "model.layers.3.mlp.gate.weight": src["model.layers.3.mlp.gate.weight"],
    }
    _write(dst_dir, dst, {"quantization_config": {
        "quant_method": "w4afp8", "group_size": 128, "ignored_layers": ["lm_head"]}})
    return src_dir, dst_dir, dst


def _rewrite(dst_dir, tensors):
    _write(dst_dir, tensors, {"quantization_config": {
        "quant_method": "w4afp8", "group_size": 128, "ignored_layers": []}})


def test_a_correct_conversion_passes(pair):
    src, dst, _ = pair
    assert verify(src, dst, samples=10) == 0


def test_mis_encoded_expert_nibbles_fail(pair):
    """The headline failure mode: a checkpoint that loads and serves noise."""
    src, dst, tensors = pair
    ep = "model.layers.3.mlp.experts.0.gate_proj"
    with safe_open(str(dst / "model-00001-of-00001.safetensors"),
                   framework="pt") as h:
        good = h.get_tensor(f"{ep}.weight")
    corrupt = good.clone()
    corrupt[0, 0] = (int(corrupt[0, 0]) + 1) % 127  # one byte, one nibble pair
    tensors[f"{ep}.weight"] = corrupt
    _rewrite(dst, tensors)
    assert verify(src, dst, samples=10) == 1


def test_recomputed_rather_than_renamed_expert_scale_fails(pair):
    """The scale must be bitwise identical; casting it is not harmless."""
    src, dst, tensors = pair
    ep = "model.layers.3.mlp.experts.0.gate_proj"
    tensors[f"{ep}.weight_scale_inv"] = \
        tensors[f"{ep}.weight_scale_inv"].float()
    _rewrite(dst, tensors)
    assert verify(src, dst, samples=10) == 1


def test_dropped_fold_on_the_fp8_path_fails(pair):
    """Simulated by block-quantizing a differently-scaled weight, which is what
    forgetting to re-apply s looks like."""
    src, dst, tensors = pair
    fp = "model.layers.3.self_attn.o_proj"
    with safe_open(str(dst / "model-00001-of-00001.safetensors"),
                   framework="pt") as h:
        w = h.get_tensor(f"{fp}.weight").float() * \
            h.get_tensor(f"{fp}.weight_scale_inv").repeat_interleave(
                128, -2).repeat_interleave(128, -1)
    bq, bs = quantize_block_fp8(w * 1.25, (128, 128))  # a 25% fold, dropped
    tensors[f"{fp}.weight"], tensors[f"{fp}.weight_scale_inv"] = bq, bs
    _rewrite(dst, tensors)
    assert verify(src, dst, samples=10) == 1


def test_leftover_source_encoding_fails(pair):
    src, dst, tensors = pair
    tensors["model.layers.3.mlp.experts.0.gate_proj.weight_packed"] = \
        torch.zeros(4, 4, dtype=torch.int32)
    _rewrite(dst, tensors)
    assert verify(src, dst, samples=10) == 1


def test_missing_expert_module_fails(pair):
    src, dst, tensors = pair
    del tensors["model.layers.3.mlp.experts.0.gate_proj.weight"]
    _rewrite(dst, tensors)
    assert verify(src, dst, samples=10) == 1


def test_changed_passthrough_tensor_fails(pair):
    src, dst, tensors = pair
    tensors["model.layers.3.input_layernorm.weight"] = \
        tensors["model.layers.3.input_layernorm.weight"] * 2
    _rewrite(dst, tensors)
    assert verify(src, dst, samples=10) == 1


def test_wrong_quant_method_fails(pair):
    src, dst, tensors = pair
    _write(dst, tensors, {"quantization_config": {
        "quant_method": "compressed-tensors"}})
    assert verify(src, dst, samples=10) == 1


def test_unresolved_regex_in_ignored_layers_fails(pair):
    """is_layer_skipped does prefix matching, so a `re:` entry silently
    un-ignores whatever it was meant to protect."""
    src, dst, tensors = pair
    _write(dst, tensors, {"quantization_config": {
        "quant_method": "w4afp8", "ignored_layers": ["re:.*mlp[.]gate$"]}})
    assert verify(src, dst, samples=10) == 1


def test_index_total_size_mismatch_fails(pair):
    src, dst, _ = pair
    path = dst / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["metadata"]["total_size"] += 4096
    path.write_text(json.dumps(index), encoding="utf-8")
    assert verify(src, dst, samples=10) == 1


def test_missing_shard_fails(pair):
    src, dst, _ = pair
    (dst / "model-00001-of-00001.safetensors").unlink()
    assert verify(src, dst, samples=10) == 1
