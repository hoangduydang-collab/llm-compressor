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
        # w1, NOT gate_proj. The WEIGHT is named by projection, but the
        # input_scale is named by SGLang shard id: its lookup is built as
        # f"experts.{id}.{shard_id}." with shard_id in (w1, w2, w3), and
        # _EXPERT_SHARD_ID maps gate_proj -> w1. An earlier version of this
        # fixture used gate_proj here, which made the "correct conversion"
        # baseline reproduce the very drift the artifact had.
        f"{ep.rpartition('.')[0]}.w1.input_scale":
            torch.ones(1, dtype=torch.bfloat16),
        f"{fp}.weight": bq,
        f"{fp}.weight_scale_inv": bs,
        "model.layers.3.input_layernorm.weight": norm,
        "model.layers.3.mlp.gate.weight": src["model.layers.3.mlp.gate.weight"],
    }
    _write(dst_dir, dst, {"quantization_config": {
        "quant_method": "w4afp8", "group_size": 128,
        "ignored_layers": _IGNORED}})
    return src_dir, dst_dir, dst


# mlp.gate is a Linear the router uses and it stays BF16, so the loader must be
# told to skip it. A conversion that omits it is not a correct conversion, which
# is why it belongs in the baseline fixture rather than only in a failure case.
_IGNORED = ["lm_head", "model.layers.3.mlp.gate"]


def _rewrite(dst_dir, tensors, ignored=None):
    _write(dst_dir, tensors, {"quantization_config": {
        "quant_method": "w4afp8", "group_size": 128,
        "ignored_layers": _IGNORED if ignored is None else ignored}})


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


def test_unquantized_linear_missing_from_ignored_layers_fails(pair):
    """The loud direction: the loader hands mlp.gate Fp8LinearMethod, which
    demands a weight_scale_inv nobody wrote.

    The engine probe would also catch this -- but only for layers inside the
    4-layer slice it can afford on one GPU. GLM-5.3's first per-layer DSA
    indexer is at layer 6, so for those modules this check is the only one that
    runs before an 8-GPU serve.
    """
    src, dst, tensors = pair
    _rewrite(dst, tensors, ignored=["lm_head"])
    assert verify(src, dst, samples=10) == 1


def test_quantized_module_listed_in_ignored_layers_fails(pair):
    """The silent direction, and the worse one.

    Ignoring a quantized module makes the loader read its int8 nibbles as BF16.
    Nothing raises: the engine starts and serves noise. Structural verification
    is the only place this is catchable cheaply.
    """
    src, dst, tensors = pair
    _rewrite(dst, tensors,
             ignored=_IGNORED + ["model.layers.3.mlp.experts.0.gate_proj"])
    assert verify(src, dst, samples=10) == 1


def test_norms_and_embeddings_need_no_ignore_entry(pair):
    """Only Linear modules are at risk. get_quant_method returns None for
    embeddings and is never called for norms, so requiring entries for them
    would make every real checkpoint fail this check."""
    src, dst, tensors = pair
    extra = dict(tensors)
    extra["model.embed_tokens.weight"] = torch.randn(32, HIDDEN).bfloat16()
    extra["model.norm.weight"] = torch.rand(HIDDEN).bfloat16()
    _rewrite(dst, extra)
    assert verify(src, dst, samples=10) == 0


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


def test_norm_entries_in_ignored_layers_are_not_reported_absent(pair, capsys):
    """A warning that fires on every healthy run trains you to ignore it.

    Norms legitimately appear in ignored_layers (the AWQ recipe lists them) and
    are present in the checkpoint as 1-D tensors. Classifying "present" as only
    the 2-D Linears reported all 184 of them as absent on the real artifact.
    """
    src, dst, tensors = pair
    _rewrite(dst, tensors,
             ignored=_IGNORED + ["model.layers.3.input_layernorm"])
    assert verify(src, dst, samples=10) == 0
    out = capsys.readouterr().out
    assert "model.layers.3.input_layernorm" not in out.split("== summary ==")[-1]


def test_genuinely_absent_entries_still_warn(pair, capsys):
    src, dst, tensors = pair
    _rewrite(dst, tensors, ignored=_IGNORED + ["model.layers.99.mlp.gate"])
    assert verify(src, dst, samples=10) == 0
    summary = capsys.readouterr().out.split("== summary ==")[-1]
    assert "model.layers.99.mlp.gate" in summary


def _add_forced_fp8(src_tensors, dst_tensors, bad_scale=False, leave_bf16=False):
    """The DSA indexer shape: BF16 in the source, block-FP8 in the output."""
    key = "model.layers.3.self_attn.indexer.wk"
    torch.manual_seed(5)
    w = torch.randn(128, HIDDEN).bfloat16()
    src_tensors[f"{key}.weight"] = w
    if leave_bf16:
        dst_tensors[f"{key}.weight"] = w
        dst_tensors[f"{key}.weight_scale_inv"] = torch.ones(
            1, HIDDEN // 128, dtype=torch.float32)
        return key
    q, s = quantize_block_fp8(w.float(), (128, 128))
    dst_tensors[f"{key}.weight"] = q
    dst_tensors[f"{key}.weight_scale_inv"] = s * (4.0 if bad_scale else 1.0)
    return key


def test_engine_forced_fp8_indexer_passes_when_re_encoded(pair, capsys):
    """BF16 in the source and e4m3 in the output is CORRECT here, so the
    passthrough check must not demand bit-exactness -- but something must still
    check it, since a missing indexer scale is what made the first artifact
    unloadable."""
    src, dst, tensors = pair
    src_t = dict(_read(src))
    _add_forced_fp8(src_t, tensors)
    _write(src, src_t, {"quantization_config": {
        "quant_method": "compressed-tensors", "ignore": ["lm_head"]}})
    _rewrite(dst, tensors)
    assert verify(src, dst, samples=10) == 0
    assert "engine-forced fp8 module(s) are e4m3" in capsys.readouterr().out


def test_an_indexer_left_in_bf16_fails(pair):
    """The original bug: a scale present but the weight never re-encoded."""
    src, dst, tensors = pair
    src_t = dict(_read(src))
    _add_forced_fp8(src_t, tensors, leave_bf16=True)
    _write(src, src_t, {"quantization_config": {
        "quant_method": "compressed-tensors", "ignore": ["lm_head"]}})
    _rewrite(dst, tensors)
    assert verify(src, dst, samples=10) == 1


def test_a_wrong_indexer_scale_fails(pair):
    src, dst, tensors = pair
    src_t = dict(_read(src))
    _add_forced_fp8(src_t, tensors, bad_scale=True)
    _write(src, src_t, {"quantization_config": {
        "quant_method": "compressed-tensors", "ignore": ["lm_head"]}})
    _rewrite(dst, tensors)
    assert verify(src, dst, samples=10) == 1


def test_a_checkpoint_with_no_forced_fp8_warns(pair, capsys):
    """On GLM-5.3 their absence means the artifact cannot load, so silence is
    the wrong response."""
    src, dst, _ = pair
    assert verify(src, dst, samples=10) == 0
    assert "no engine-forced fp8 modules found" in capsys.readouterr().out


def _read(path):
    with safe_open(str(path / "model-00001-of-00001.safetensors"),
                   framework="pt") as handle:
        return {k: handle.get_tensor(k) for k in handle.keys()}


def test_a_grafted_layer_with_a_bf16_indexer_fails(pair):
    """The blind spot that let the MTP graft ship a BF16 layer-78 indexer.

    Layer 78 exists only in the OUTPUT -- the AWQ source has no MTP layer at all
    -- so every source-relative check skips it, the coverage check calls it "a
    BF16 Linear correctly listed in ignored_layers", and no slice loads it
    because the slicer forces num_nextn_predict_layers=0. The engine-required
    check is source-blind precisely so this cannot hide.
    """
    src, dst, tensors = pair
    key = "model.layers.78.self_attn.indexer.wk"
    tensors[f"{key}.weight"] = torch.randn(128, HIDDEN).bfloat16()
    _rewrite(dst, tensors, ignored=_IGNORED + [key])
    assert verify(src, dst, samples=10) == 1


def test_a_grafted_layer_with_an_fp8_indexer_passes(pair):
    src, dst, tensors = pair
    key = "model.layers.78.self_attn.indexer.wk"
    q, s = quantize_block_fp8(torch.randn(128, HIDDEN).float(), (128, 128))
    tensors[f"{key}.weight"] = q
    tensors[f"{key}.weight_scale_inv"] = s
    _rewrite(dst, tensors)
    assert verify(src, dst, samples=10) == 0


# ---- expert input_scale names -------------------------------------------- #
# SGLang resolves expert activation scales by shard id, not by projection name
# (make_expert_input_scale_params_mapping, fused_moe_triton/layer.py:1603), and
# layer.py:285 drops an unmatched mlp.experts.* tensor with a bare `continue`.
# So a misnamed scale is invisible: no warning, no log line, param left at the
# torch.ones the loader registered. These two tests pin the resulting policy --
# tolerated while the values ARE ones, refused the moment they are not.

_UNRESOLVABLE = "model.layers.3.mlp.experts.0.gate_proj.input_scale"
_RESOLVABLE = "model.layers.3.mlp.experts.0.w1.input_scale"


def test_unresolvable_input_scale_name_warns_while_the_value_is_unit(pair, capsys):
    src, dst, tensors = pair
    tensors.pop(_RESOLVABLE)
    tensors[_UNRESOLVABLE] = torch.ones(1, dtype=torch.bfloat16)
    _rewrite(dst, tensors)
    # Still 0: ones is exactly what the loader defaults to, so skipping the
    # tensor changes no numerics. The report must say so out loud anyway.
    assert verify(src, dst, samples=10) == 0
    out = capsys.readouterr().out
    assert "cannot resolve" in out
    assert "exactly 1.0" in out


def test_non_unit_input_scale_under_an_unresolvable_name_fails(pair):
    src, dst, tensors = pair
    tensors.pop(_RESOLVABLE)
    tensors[_UNRESOLVABLE] = torch.full((1,), 0.5, dtype=torch.bfloat16)
    _rewrite(dst, tensors)
    # A real scale that the engine will silently discard: the served model would
    # quantize activations with 1.0 while the checkpoint claims 0.5.
    assert verify(src, dst, samples=10) == 1


def test_resolvable_input_scale_names_pass_without_a_warning(pair, capsys):
    src, dst, _ = pair
    assert verify(src, dst, samples=10) == 0
    out = capsys.readouterr().out
    assert "cannot resolve" not in out
    assert "use SGLang shard ids" in out
