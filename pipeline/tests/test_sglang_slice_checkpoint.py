"""Tests for the truncated-checkpoint slicer used for engine load testing."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from pipeline.sglang_slice_checkpoint import (  # noqa: E402
    keep,
    layer_of,
    slice_checkpoint,
    truncate_config,
)

DEPTH = 8
DENSE = 3


def _can_symlink(tmp) -> bool:
    """Probe rather than check the platform: an unelevated Windows process
    cannot create symlinks (WinError 1314), but an elevated one can, and WSL
    paths behave differently again."""
    import os

    target = tmp / "_probe_target"
    target.write_text("x", encoding="utf-8")
    try:
        os.symlink(target, tmp / "_probe_link")
    except OSError:
        return False
    return True


@pytest.fixture
def full(tmp_path):
    """A miniature 8-layer checkpoint spread over two shards."""
    ckpt = tmp_path / "full"
    ckpt.mkdir()

    shard_a, shard_b = {}, {}
    for layer in range(DEPTH):
        target = shard_a if layer < 4 else shard_b
        pref = f"model.layers.{layer}"
        target[f"{pref}.input_layernorm.weight"] = torch.ones(8)
        target[f"{pref}.self_attn.o_proj.weight"] = torch.ones(8, 8)
        if layer >= DENSE:
            target[f"{pref}.mlp.experts.0.gate_proj.weight"] = torch.ones(
                4, 8, dtype=torch.int8
            )
    shard_a["model.embed_tokens.weight"] = torch.ones(16, 8)
    shard_b["model.norm.weight"] = torch.ones(8)
    shard_b["lm_head.weight"] = torch.ones(16, 8)

    save_file(shard_a, str(ckpt / "model-00001-of-00002.safetensors"),
              metadata={"format": "pt"})
    save_file(shard_b, str(ckpt / "model-00002-of-00002.safetensors"),
              metadata={"format": "pt"})

    weight_map = {}
    for name, shard in (("model-00001-of-00002.safetensors", shard_a),
                        ("model-00002-of-00002.safetensors", shard_b)):
        for key in shard:
            weight_map[key] = name
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (ckpt / "config.json").write_text(
        json.dumps({
            "architectures": ["GlmMoeDsaForCausalLM"],
            "num_hidden_layers": DEPTH,
            "first_k_dense_replace": DENSE,
            "num_nextn_predict_layers": 1,
            "indexer_types": ["full"] * 3 + ["shared"] * (DEPTH - 3),
            "unrelated_list": [1, 2],
            "quantization_config": {"quant_method": "w4afp8"},
        }),
        encoding="utf-8",
    )
    (ckpt / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return ckpt, tmp_path / "slice"


def test_layer_of_and_keep():
    assert layer_of("model.layers.12.mlp.gate.weight") == 12
    assert layer_of("model.embed_tokens.weight") is None
    assert keep("model.layers.2.self_attn.o_proj.weight", 4)
    assert not keep("model.layers.4.self_attn.o_proj.weight", 4)
    # embeddings / norm / head survive any truncation
    assert keep("model.embed_tokens.weight", 4)
    assert keep("model.norm.weight", 4)
    assert keep("lm_head.weight", 4)


def test_truncate_config_cuts_per_layer_lists():
    config = {
        "num_hidden_layers": 8,
        "indexer_types": ["full"] * 3 + ["shared"] * 5,
        "num_nextn_predict_layers": 1,
        "unrelated_list": [1, 2],
        "scalar": 5,
    }
    out = truncate_config(config, 4)
    assert out["num_hidden_layers"] == 4
    assert out["indexer_types"] == ["full", "full", "full", "shared"]
    # MTP disabled: its layer index would point past the sliced model.
    assert out["num_nextn_predict_layers"] == 0
    # Lists whose length is not the model depth are coincidences, not per-layer.
    assert out["unrelated_list"] == [1, 2]
    assert out["scalar"] == 5


def test_slice_produces_a_loadable_looking_directory(full):
    ckpt, out = full
    assert slice_checkpoint(ckpt, out, layers=4) == 0

    index = json.loads((out / "model.safetensors.index.json").read_text())
    keys = set(index["weight_map"])
    assert "model.layers.3.mlp.experts.0.gate_proj.weight" in keys
    assert not any(".layers.4." in k for k in keys)
    assert not any(".layers.7." in k for k in keys)
    assert {"model.embed_tokens.weight", "model.norm.weight",
            "lm_head.weight"} <= keys

    config = json.loads((out / "config.json").read_text())
    assert config["num_hidden_layers"] == 4
    assert config["num_nextn_predict_layers"] == 0
    assert len(config["indexer_types"]) == 4
    assert config["quantization_config"]["quant_method"] == "w4afp8"
    assert (out / "tokenizer_config.json").is_file()


def test_slice_links_rather_than_copies(full):
    """Zero-copy is the point: the real shards are 10 GB each on a shared
    volume. A hard link and a symlink are both acceptable; a copy is not."""
    ckpt, out = full
    assert slice_checkpoint(ckpt, out, layers=4) == 0
    shards = list(out.glob("*.safetensors"))
    assert shards
    for shard in shards:
        source = ckpt / shard.name
        linked = shard.is_symlink() or (
            shard.stat().st_ino == source.stat().st_ino
            and shard.stat().st_dev == source.stat().st_dev
        )
        assert linked, f"{shard} is a copy, not a link"


def test_slice_includes_at_least_one_moe_layer(full):
    """A slice at or below first_k_dense_replace never touches the expert
    loader, which is the entire reason the test exists."""
    ckpt, out = full
    assert slice_checkpoint(ckpt, out, layers=DENSE) == 2
    assert slice_checkpoint(ckpt, out, layers=DENSE + 1) == 0


def test_slice_rejects_more_layers_than_the_model_has(full):
    ckpt, out = full
    assert slice_checkpoint(ckpt, out, layers=DEPTH + 1) == 2


def test_slice_total_size_counts_kept_tensors_not_symlinked_files(full):
    """The symlinked shards still contain excluded layers, so total_size must be
    computed from the kept set or it overstates by the whole tail."""
    ckpt, out = full
    assert slice_checkpoint(ckpt, out, layers=4) == 0
    index = json.loads((out / "model.safetensors.index.json").read_text())

    from safetensors import safe_open

    expected = 0
    for shard in sorted({v for v in index["weight_map"].values()}):
        with safe_open(str(ckpt / shard), framework="pt") as handle:
            for key in handle.keys():
                if key in index["weight_map"]:
                    t = handle.get_tensor(key)
                    expected += t.numel() * t.element_size()
    assert index["metadata"]["total_size"] == expected

    on_disk = sum((ckpt / s).stat().st_size
                  for s in {v for v in index["weight_map"].values()})
    assert index["metadata"]["total_size"] < on_disk


def test_slice_is_rerunnable(full):
    ckpt, out = full
    assert slice_checkpoint(ckpt, out, layers=4) == 0
    assert slice_checkpoint(ckpt, out, layers=4) == 0  # symlink replace, no crash
    assert slice_checkpoint(ckpt, out, layers=5) == 0
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert any(".layers.4." in k for k in index["weight_map"])


def test_slice_resolves_hf_cache_style_relative_symlinks(tmp_path):
    """The bug that broke the first real slice.

    Skipped where symlinks cannot be CREATED (unelevated Windows); it runs in
    the Linux container where the conversion and slicing actually happen, which
    is the environment the bug occurred in.

    A HuggingFace cache snapshot is a symlink farm: snapshots/<rev>/x.safetensors
    -> ../../blobs/<sha>. os.link does NOT follow symlinks on Linux, so linking
    such an entry into a directory at a different depth reproduces the RELATIVE
    target, which then resolves to nowhere. The slice ends up full of dangling
    symlinks that `ls` displays happily and the engine reports as
    "incomplete download?".
    """
    import os

    if not _can_symlink(tmp_path):
        pytest.skip("cannot create symlinks on this platform/privilege level")

    root = tmp_path / "cache"
    blobs = root / "blobs"
    snap = root / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)

    tensors = {
        "model.embed_tokens.weight": torch.ones(4, 8),
        "model.layers.0.self_attn.o_proj.weight": torch.ones(8, 8),
        "model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones(
            4, 8, dtype=torch.int8
        ),
        "model.norm.weight": torch.ones(8),
        "lm_head.weight": torch.ones(4, 8),
    }
    blob = blobs / "deadbeef"
    save_file(tensors, str(blob), metadata={"format": "pt"})
    shard = "model-00001-of-00001.safetensors"
    # Relative target, exactly as huggingface_hub writes it.
    os.symlink(os.path.join("..", "..", "blobs", "deadbeef"), snap / shard)

    (snap / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": 0},
            "weight_map": {k: shard for k in tensors},
        }),
        encoding="utf-8",
    )
    (snap / "config.json").write_text(
        json.dumps({
            "architectures": ["GlmMoeDsaForCausalLM"],
            "num_hidden_layers": 8,
            "first_k_dense_replace": 3,
        }),
        encoding="utf-8",
    )

    # A directory at a different depth, so a relative target cannot survive.
    out = tmp_path / "a" / "b" / "slice"
    assert slice_checkpoint(snap, out, layers=4) == 0

    linked = out / shard
    assert linked.is_file(), "slice entry does not resolve to a real file"
    from safetensors import safe_open

    with safe_open(str(linked), framework="pt") as handle:
        assert "model.embed_tokens.weight" in handle.keys()
