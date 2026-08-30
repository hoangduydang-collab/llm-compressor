"""Tests for the MTP layer-78 graft.

The graft mutates an existing checkpoint's index in place, so the tests that
matter are the ones proving it (a) uses the same int4 scale convention as the
main model, (b) never damages the target on failure, and (c) cannot
double-append.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from pipeline.sglang_w4afp8_kernels import unpack_nibbles_int8  # noqa: E402
from pipeline.graft_mtp_w4afp8 import (  # noqa: E402
    GROUP,
    INT4_LEVELS,
    classify,
    dequantize_int4_group,
    graft,
    quantize_int4_group_rtn,
)

HIDDEN = 256
INTER = 128
LAYER = 78


# --------------------------------------------------------------------------
# int4 RTN convention
# --------------------------------------------------------------------------


def test_int4_scale_is_max_over_8_not_over_7():
    """The whole point. A /7 scale would make layer 78's scales systematically
    14% off relative to every other layer in the artifact.

    Uses a NEGATIVE extreme, which is what discriminates the two conventions:
    with scale = max/8 the extreme lands exactly on -8, whereas scale = max/7
    would give 8/7 = 1.143 and land it on -7. A positive extreme cannot tell
    them apart, because +8 is unrepresentable and saturates to +7 either way --
    see the clamping test below.
    """
    w = torch.zeros(1, GROUP)
    w[0, 0] = -8.0
    values, scale = quantize_int4_group_rtn(w)
    assert scale.float().item() == pytest.approx(8.0 / INT4_LEVELS, rel=1e-3)
    assert int(values[0, 0]) == -8


def test_int4_uses_the_full_negative_grid_and_clamps_the_positive():
    """scale = max/8 means a group whose extreme is negative reaches -8, while a
    positive extreme wants +8 and must saturate to +7. That asymmetry is
    inherent to the convention, not a bug, and is measured at 0.56-0.62% of
    elements on the main model."""
    neg = torch.zeros(1, GROUP)
    neg[0, 0] = -4.0
    values, _ = quantize_int4_group_rtn(neg)
    assert int(values[0, 0]) == -8

    pos = torch.zeros(1, GROUP)
    pos[0, 0] = 4.0
    values, _ = quantize_int4_group_rtn(pos)
    assert int(values[0, 0]) == 7, "positive extreme must saturate, not wrap"


def test_int4_values_stay_in_range_and_survive_nibble_packing():
    torch.manual_seed(5)
    w = torch.randn(64, 512)
    values, scale = quantize_int4_group_rtn(w)
    assert int(values.min()) >= -8 and int(values.max()) <= 7
    assert scale.shape == (64, 512 // GROUP)
    assert scale.dtype == torch.bfloat16
    from pipeline.sglang_w4afp8_kernels import pack_nibbles_int8

    packed = pack_nibbles_int8(values)
    assert torch.equal(unpack_nibbles_int8(packed), values)


def test_int4_rtn_residual_is_near_the_grid_floor():
    torch.manual_seed(6)
    w = torch.randn(128, 1024)
    values, scale = quantize_int4_group_rtn(w)
    resid = ((dequantize_int4_group(values, scale) - w).norm() / w.norm()).item()
    # ratio/(8*sqrt(12)) with a gaussian group ratio ~3 gives ~0.11; RTN with no
    # smoothing sits a little above the AWQ layers.
    assert 0.05 < resid < 0.20, resid


def test_int4_zero_group_is_exact():
    w = torch.zeros(2, GROUP)
    values, scale = quantize_int4_group_rtn(w)
    assert torch.equal(dequantize_int4_group(values, scale), w)
    assert scale.float().flatten()[0].item() == 1.0


def test_int4_rejects_ragged_group():
    with pytest.raises(ValueError, match="multiple of the group size"):
        quantize_int4_group_rtn(torch.randn(4, GROUP + 1))


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        (f"model.layers.{LAYER}.mlp.experts.0.gate_proj.weight", "expert"),
        (f"model.layers.{LAYER}.mlp.experts.255.down_proj.weight", "expert"),
        (f"model.layers.{LAYER}.self_attn.o_proj.weight", "fp8"),
        (f"model.layers.{LAYER}.self_attn.kv_a_proj_with_mqa.weight", "fp8"),
        (f"model.layers.{LAYER}.mlp.shared_experts.up_proj.weight", "fp8"),
        # Divergences from the vendor, asserted so they cannot drift: the DSA
        # indexer and eh_proj stay BF16 because that is how the other 78 layers
        # of this artifact are built.
        (f"model.layers.{LAYER}.self_attn.indexer.wk.weight", "copy"),
        (f"model.layers.{LAYER}.self_attn.indexer.wq_b.weight", "copy"),
        (f"model.layers.{LAYER}.eh_proj.weight", "copy"),
        (f"model.layers.{LAYER}.enorm.weight", "copy"),
        (f"model.layers.{LAYER}.hnorm.weight", "copy"),
        (f"model.layers.{LAYER}.mlp.gate.weight", "copy"),
        (f"model.layers.{LAYER}.mlp.gate.e_score_correction_bias", "copy"),
        (f"model.layers.{LAYER}.shared_head.norm.weight", "copy"),
        (f"model.layers.{LAYER}.input_layernorm.weight", "copy"),
    ],
)
def test_classification(name, expected):
    assert classify(name, LAYER) == expected


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


@pytest.fixture
def scene(tmp_path):
    """A BF16 source carrying layer 78, and a target w4afp8 checkpoint."""
    torch.manual_seed(9)
    base = tmp_path / "base"
    out = tmp_path / "out"
    base.mkdir()
    out.mkdir()

    src: dict[str, torch.Tensor] = {}
    pref = f"model.layers.{LAYER}"
    for expert in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            src[f"{pref}.mlp.experts.{expert}.{proj}.weight"] = torch.randn(
                INTER, HIDDEN
            ).bfloat16()
    for suffix in ("self_attn.o_proj", "self_attn.q_a_proj",
                   "mlp.shared_experts.gate_proj"):
        src[f"{pref}.{suffix}.weight"] = torch.randn(INTER, HIDDEN).bfloat16()
    src[f"{pref}.enorm.weight"] = torch.rand(HIDDEN).bfloat16()
    src[f"{pref}.hnorm.weight"] = torch.rand(HIDDEN).bfloat16()
    src[f"{pref}.eh_proj.weight"] = torch.randn(HIDDEN, 2 * HIDDEN).bfloat16()
    src[f"{pref}.mlp.gate.weight"] = torch.randn(8, HIDDEN).bfloat16()
    src[f"{pref}.self_attn.indexer.wk.weight"] = torch.randn(64, HIDDEN).bfloat16()
    src[f"{pref}.shared_head.norm.weight"] = torch.rand(HIDDEN).bfloat16()
    # A layer the graft must ignore entirely.
    src["model.layers.5.mlp.gate.weight"] = torch.randn(8, HIDDEN).bfloat16()
    save_file(src, str(base / "model.safetensors"), metadata={"format": "pt"})

    target = {"model.layers.0.input_layernorm.weight": torch.ones(HIDDEN).bfloat16()}
    save_file(target, str(out / "model-00001-of-00001.safetensors"),
              metadata={"format": "pt"})
    (out / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": HIDDEN * 2},
            "weight_map": {
                "model.layers.0.input_layernorm.weight":
                    "model-00001-of-00001.safetensors"
            },
        }),
        encoding="utf-8",
    )
    (out / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "w4afp8"}}),
        encoding="utf-8",
    )
    return base, out


def test_graft_appends_without_touching_existing_shards(scene):
    base, out = scene
    before = (out / "model-00001-of-00001.safetensors").read_bytes()
    assert graft(base, out, layer=LAYER, shard_bytes=10**6) == 0
    assert (out / "model-00001-of-00001.safetensors").read_bytes() == before

    index = json.loads((out / "model.safetensors.index.json").read_text())
    keys = index["weight_map"]
    assert "model.layers.0.input_layernorm.weight" in keys  # preserved
    assert f"model.layers.{LAYER}.mlp.experts.0.gate_proj.weight" in keys
    assert f"model.layers.{LAYER}.mlp.experts.0.gate_proj.weight_scale_inv" in keys
    assert f"model.layers.{LAYER}.self_attn.o_proj.weight_scale_inv" in keys
    # BF16 passthrough keeps its plain name and gains no scale
    assert f"model.layers.{LAYER}.enorm.weight" in keys
    assert f"model.layers.{LAYER}.eh_proj.weight_scale_inv" not in keys
    assert f"model.layers.{LAYER}.self_attn.indexer.wk.weight_scale_inv" not in keys
    # unrelated layers are not dragged in
    assert not any(".layers.5." in k for k in keys)

    config = json.loads((out / "config.json").read_text())
    assert config["num_nextn_predict_layers"] == 1
    assert config["quantization_config"]["quant_method"] == "w4afp8"


def test_graft_dtypes_match_what_the_loader_registers(scene):
    from safetensors import safe_open

    base, out = scene
    assert graft(base, out, layer=LAYER, shard_bytes=10**6) == 0
    index = json.loads((out / "model.safetensors.index.json").read_text())

    def fetch(key):
        with safe_open(str(out / index["weight_map"][key]), framework="pt") as h:
            return h.get_tensor(key)

    pref = f"model.layers.{LAYER}"
    assert fetch(f"{pref}.mlp.experts.0.gate_proj.weight").dtype == torch.int8
    assert fetch(f"{pref}.mlp.experts.0.gate_proj.weight_scale_inv").dtype == \
        torch.bfloat16
    assert fetch(f"{pref}.self_attn.o_proj.weight").dtype == torch.float8_e4m3fn
    assert fetch(f"{pref}.self_attn.o_proj.weight_scale_inv").dtype == torch.float32
    assert fetch(f"{pref}.enorm.weight").dtype == torch.bfloat16


def test_graft_index_total_size_tracks_the_added_bytes(scene):
    from safetensors import safe_open

    base, out = scene
    assert graft(base, out, layer=LAYER, shard_bytes=10**6) == 0
    index = json.loads((out / "model.safetensors.index.json").read_text())
    actual = 0
    for shard in sorted({v for v in index["weight_map"].values()}):
        with safe_open(str(out / shard), framework="pt") as h:
            for key in h.keys():
                t = h.get_tensor(key)
                actual += t.numel() * t.element_size()
    assert index["metadata"]["total_size"] == actual


def test_graft_refuses_to_double_append(scene):
    base, out = scene
    assert graft(base, out, layer=LAYER, shard_bytes=10**6) == 0
    index_before = (out / "model.safetensors.index.json").read_bytes()
    assert graft(base, out, layer=LAYER, shard_bytes=10**6) == 2
    assert (out / "model.safetensors.index.json").read_bytes() == index_before


def test_graft_cleans_up_and_leaves_target_intact_on_failure(scene, monkeypatch):
    """A failed graft must not leave orphan shards or a half-written index --
    the target is a 394 GB artifact that took 20 hours to produce."""
    import pipeline.graft_mtp_w4afp8 as mod

    base, out = scene
    index_before = (out / "model.safetensors.index.json").read_bytes()
    shards_before = {p.name for p in out.glob("*.safetensors")}

    # Break the int4 scale so the residual gate fires.
    monkeypatch.setattr(
        mod, "quantize_int4_group_rtn",
        lambda w: (torch.zeros_like(w, dtype=torch.int8),
                   torch.ones(w.shape[0], w.shape[1] // GROUP,
                              dtype=torch.bfloat16)),
    )
    assert mod.graft(base, out, layer=LAYER, shard_bytes=10**6) == 1
    assert (out / "model.safetensors.index.json").read_bytes() == index_before
    assert {p.name for p in out.glob("*.safetensors")} == shards_before


def test_graft_dry_run_writes_nothing(scene):
    base, out = scene
    index_before = (out / "model.safetensors.index.json").read_bytes()
    shards_before = {p.name for p in out.glob("*.safetensors")}
    assert graft(base, out, layer=LAYER, dry_run=True) == 0
    assert (out / "model.safetensors.index.json").read_bytes() == index_before
    assert {p.name for p in out.glob("*.safetensors")} == shards_before


def test_graft_requires_an_index_and_a_source_layer(scene, tmp_path):
    base, out = scene
    empty = tmp_path / "empty"
    empty.mkdir()
    assert graft(base, empty, layer=LAYER) == 2      # no index
    assert graft(base, out, layer=77) == 2           # no such layer in source
