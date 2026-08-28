"""Tests for the depth-truncating subset builder.

The failure that matters is a SILENTLY incomplete subset: a missing tensor does
not surface here, it surfaces ~1 minute later inside model construction as an
opaque missing-weight error, or worse, as a model that loads with garbage. So the
builder's fail-closed checks are tested as carefully as its happy path.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file, save_file  # noqa: E402

from pipeline.build_subset_checkpoint import (  # noqa: E402
    build,
    layer_of,
    main,
    patch_config,
    plan_shards,
    select_keys,
    tensor_sizes,
)

# Mirrors GLM-5.2's real structure: 3 dense layers (first_k_dense_replace=3),
# then MoE layers, an MTP layer at index num_hidden_layers, and 3 non-layer
# tensors (embed_tokens / norm / lm_head, the last real because
# tie_word_embeddings is False).
NUM_LAYERS = 6
MTP = NUM_LAYERS  # index 6


def _fake_snapshot(tmp_path, num_layers=NUM_LAYERS, experts=2):
    snap = tmp_path / "snap"
    snap.mkdir()
    shards: dict[str, dict] = {}

    def put(shard, key, tensor):
        shards.setdefault(shard, {})[key] = tensor

    put("model-00001.safetensors", "model.embed_tokens.weight", torch.zeros(8, 4))
    put("model-00001.safetensors", "model.norm.weight", torch.zeros(4))
    put("model-00001.safetensors", "lm_head.weight", torch.zeros(8, 4))
    for layer in list(range(num_layers)) + [MTP]:
        shard = f"model-{layer + 2:05d}.safetensors"
        put(shard, f"model.layers.{layer}.input_layernorm.weight", torch.zeros(4))
        put(shard, f"model.layers.{layer}.self_attn.o_proj.weight", torch.zeros(4, 4))
        if layer >= 3:  # MoE layers carry experts
            for e in range(experts):
                put(shard, f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight",
                    torch.zeros(4, 4))
        else:
            put(shard, f"model.layers.{layer}.mlp.gate_proj.weight", torch.zeros(4, 4))

    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, str(snap / shard), metadata={"format": "pt"})
        for key in tensors:
            weight_map[key] = shard
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    (snap / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": num_layers,
                "first_k_dense_replace": 3,
                "num_nextn_predict_layers": 1,
                "tie_word_embeddings": False,
                "_name_or_path": "zai-org/GLM-5.2",
            }
        )
    )
    (snap / "tokenizer_config.json").write_text("{}")
    (snap / "modeling_glm.py").write_text("# remote code\n")
    return snap, weight_map


# --- key selection ----------------------------------------------------------

def test_layer_of():
    assert layer_of("model.layers.42.mlp.gate_proj.weight") == 42
    assert layer_of("model.embed_tokens.weight") is None
    assert layer_of("lm_head.weight") is None


def test_select_keeps_nonlayer_and_drops_at_cut(tmp_path):
    _, weight_map = _fake_snapshot(tmp_path)
    keys = select_keys(weight_map, 4)
    assert "model.embed_tokens.weight" in keys
    assert "lm_head.weight" in keys, "tie_word_embeddings=False makes lm_head required"
    assert "model.norm.weight" in keys
    assert any(".layers.3." in k for k in keys)
    assert not any(".layers.4." in k for k in keys)
    assert not any(f".layers.{MTP}." in k for k in keys), "MTP layer must be dropped"


def test_select_keeps_the_dense_prefix_that_feeds_layer_3(tmp_path):
    """Layer 3's calibration inputs come from 0-2; dropping them would change
    what the probe measures."""
    _, weight_map = _fake_snapshot(tmp_path)
    keys = select_keys(weight_map, 4)
    for layer in (0, 1, 2):
        assert any(f".layers.{layer}." in k for k in keys), layer


def test_select_rejects_zero_layers(tmp_path):
    _, weight_map = _fake_snapshot(tmp_path)
    with pytest.raises(ValueError):
        select_keys(weight_map, 0)


# --- shard planning ---------------------------------------------------------

def test_plan_shards_respects_the_byte_cap(tmp_path):
    snap, weight_map = _fake_snapshot(tmp_path)
    keys = select_keys(weight_map, 4)
    sizes = tensor_sizes(snap, weight_map)
    groups = plan_shards(keys, weight_map, sizes, shard_max_bytes=100)
    assert len(groups) > 1
    for group in groups:
        # A single oversized tensor is allowed to exceed the cap alone.
        if len(group) > 1:
            assert sum(sizes[k] for k in group) <= 100 + max(sizes[k] for k in group)
    assert sorted(k for g in groups for k in g) == sorted(keys)


def test_plan_shards_partitions_exactly_once(tmp_path):
    snap, weight_map = _fake_snapshot(tmp_path)
    keys = select_keys(weight_map, 5)
    sizes = tensor_sizes(snap, weight_map)
    flat = [k for g in plan_shards(keys, weight_map, sizes) for k in g]
    assert sorted(flat) == sorted(keys)
    assert len(flat) == len(set(flat)), "a tensor was assigned to two shards"


# --- config patching --------------------------------------------------------

def test_patch_config_truncates_depth_and_disables_mtp():
    patched = patch_config(
        {"num_hidden_layers": 78, "num_nextn_predict_layers": 1,
         "_name_or_path": "zai-org/GLM-5.2"}, 4)
    assert patched["num_hidden_layers"] == 4
    assert patched["num_nextn_predict_layers"] == 0, \
        "MTP weights live at index num_hidden_layers and are not in the subset"
    assert patched["_subset_num_layers"] == 4
    assert "never serve" in patched["_subset_warning"].lower()


def test_patch_config_leaves_absent_mtp_alone():
    patched = patch_config({"num_hidden_layers": 32}, 2)
    assert "num_nextn_predict_layers" not in patched


def test_patch_config_does_not_mutate_input():
    original = {"num_hidden_layers": 78, "num_nextn_predict_layers": 1}
    patch_config(original, 4)
    assert original == {"num_hidden_layers": 78, "num_nextn_predict_layers": 1}


# --- end to end -------------------------------------------------------------

def test_build_roundtrip(tmp_path):
    snap, weight_map = _fake_snapshot(tmp_path)
    out = tmp_path / "subset"
    summary = build(snap, out, 4)

    index = json.loads((out / "model.safetensors.index.json").read_text())
    new_map = index["weight_map"]
    assert set(new_map) == set(select_keys(weight_map, 4))
    assert summary["tensors"] == len(new_map)

    # every promised shard exists and holds exactly its promised tensors
    for shard in set(new_map.values()):
        loaded = load_file(str(out / shard))
        assert set(loaded) == {k for k, s in new_map.items() if s == shard}

    config = json.loads((out / "config.json").read_text())
    assert config["num_hidden_layers"] == 4
    assert config["num_nextn_predict_layers"] == 0

    # auxiliary files carried over, source index/config NOT copied verbatim
    assert (out / "tokenizer_config.json").exists()
    assert (out / "modeling_glm.py").exists()
    assert index["metadata"]["total_size"] > 0


def test_build_copies_real_bytes_not_symlinks(tmp_path):
    """HF snapshot entries are symlinks into the blob store; the subset must be
    self-contained so it survives on node-local disk with no cephfs mount."""
    snap, _ = _fake_snapshot(tmp_path)
    out = tmp_path / "subset"
    build(snap, out, 4)
    aux = out / "tokenizer_config.json"
    assert not aux.is_symlink()
    assert aux.read_text() == "{}"


def test_build_values_match_source(tmp_path):
    snap, weight_map = _fake_snapshot(tmp_path)
    # give one tensor a recognizable value
    shard = weight_map["model.layers.3.self_attn.o_proj.weight"]
    tensors = load_file(str(snap / shard))
    tensors["model.layers.3.self_attn.o_proj.weight"] = torch.full((4, 4), 7.5)
    save_file(tensors, str(snap / shard), metadata={"format": "pt"})

    out = tmp_path / "subset"
    build(snap, out, 4)
    index = json.loads((out / "model.safetensors.index.json").read_text())
    got = load_file(str(out / index["weight_map"]["model.layers.3.self_attn.o_proj.weight"]))
    assert torch.equal(got["model.layers.3.self_attn.o_proj.weight"],
                       torch.full((4, 4), 7.5))


def test_build_rejects_missing_index(tmp_path):
    (tmp_path / "empty").mkdir()
    assert main(["--snapshot", str(tmp_path / "empty"),
                 "--out", str(tmp_path / "o"), "--layers", "4"]) == 2


def test_cli_builds(tmp_path):
    snap, _ = _fake_snapshot(tmp_path)
    out = tmp_path / "subset"
    assert main(["--snapshot", str(snap), "--out", str(out), "--layers", "4"]) == 0
    assert (out / "model.safetensors.index.json").exists()


# --- per-layer list truncation (the 131426z failure) ------------------------

def test_patch_config_truncates_per_layer_lists():
    """The real failure: GLM-5.2 ships mlp_layer_types with one entry per layer,
    and transformers validates its length against num_hidden_layers."""
    patched = patch_config(
        {
            "num_hidden_layers": 78,
            "num_nextn_predict_layers": 1,
            "mlp_layer_types": ["dense"] * 3 + ["moe"] * 75,
        },
        4,
    )
    assert len(patched["mlp_layer_types"]) == 4
    assert patched["mlp_layer_types"] == ["dense", "dense", "dense", "moe"], \
        "truncation must keep the FIRST n entries so layer 3 stays MoE"
    assert patched["_subset_truncated_lists"] == ["mlp_layer_types"]


def test_patch_config_truncates_lists_sized_with_mtp():
    """Some per-layer lists count the MTP layer, giving length depth+mtp."""
    patched = patch_config(
        {"num_hidden_layers": 78, "num_nextn_predict_layers": 1,
         "layer_types": ["full"] * 79},
        4,
    )
    assert len(patched["layer_types"]) == 4


def test_patch_config_truncates_several_lists():
    patched = patch_config(
        {"num_hidden_layers": 8, "mlp_layer_types": ["a"] * 8,
         "attn_layer_types": ["b"] * 8, "n_experts_per_layer": list(range(8))},
        3,
    )
    assert patched["_subset_truncated_lists"] == [
        "attn_layer_types", "mlp_layer_types", "n_experts_per_layer"]
    for key in ("mlp_layer_types", "attn_layer_types", "n_experts_per_layer"):
        assert len(patched[key]) == 3


def test_patch_config_leaves_unrelated_lists_alone():
    """A list whose length does not match depth is not per-layer data."""
    patched = patch_config(
        {"num_hidden_layers": 78, "architectures": ["GlmMoeDsaForCausalLM"],
         "eos_token_id": [1, 2, 3]},
        4,
    )
    assert patched["architectures"] == ["GlmMoeDsaForCausalLM"]
    assert patched["eos_token_id"] == [1, 2, 3]
    assert "_subset_truncated_lists" not in patched


def test_patch_config_no_lists_no_marker():
    patched = patch_config({"num_hidden_layers": 78}, 4)
    assert "_subset_truncated_lists" not in patched
