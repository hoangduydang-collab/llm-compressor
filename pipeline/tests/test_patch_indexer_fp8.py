"""Tests for the in-place-equivalent indexer FP8 retrofit.

The tool exists to avoid a 2h26m re-conversion, so the tests that matter are the
ones proving it is SAFE to prefer over one: the source is never mutated,
unaffected shards are linked rather than copied, the superseded BF16 bytes are
actually gone (not merely unindexed), and a bad quantization refuses to leave a
loadable artifact behind.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from pipeline.patch_indexer_fp8 import find_targets, patch  # noqa: E402

HIDDEN = 256


def _write_index(path, mapping, total=0):
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": mapping}),
        encoding="utf-8",
    )


@pytest.fixture
def converted(tmp_path):
    """Two shards: one holding an indexer, one clean."""
    torch.manual_seed(11)
    src = tmp_path / "src"
    src.mkdir()

    a = {
        "model.layers.0.self_attn.indexer.wk.weight":
            torch.randn(128, HIDDEN).bfloat16(),
        "model.layers.0.self_attn.indexer.wq_b.weight":
            torch.randn(256, HIDDEN).bfloat16(),
        "model.layers.0.self_attn.indexer.weights_proj.weight":
            torch.randn(32, HIDDEN).bfloat16(),
        "model.layers.0.input_layernorm.weight": torch.rand(HIDDEN).bfloat16(),
    }
    b = {
        "model.layers.1.mlp.experts.0.gate_proj.weight":
            torch.randint(-8, 8, (128, HIDDEN // 2), dtype=torch.int8),
        "model.norm.weight": torch.rand(HIDDEN).bfloat16(),
    }
    save_file(a, str(src / "model-00001-of-00002.safetensors"),
              metadata={"format": "pt"})
    save_file(b, str(src / "model-00002-of-00002.safetensors"),
              metadata={"format": "pt"})

    mapping = {}
    for shard, tensors in (("model-00001-of-00002.safetensors", a),
                           ("model-00002-of-00002.safetensors", b)):
        for key in tensors:
            mapping[key] = shard
    total = sum(t.numel() * t.element_size() for t in {**a, **b}.values())
    _write_index(src, mapping, total)
    (src / "config.json").write_text(
        json.dumps({
            "architectures": ["GlmMoeDsaForCausalLM"],
            "quantization_config": {
                "quant_method": "w4afp8",
                "ignored_layers": [
                    "lm_head",
                    "model.layers.0.self_attn.indexer.wk",
                    "model.layers.0.self_attn.indexer.wq_b",
                    "model.layers.0.self_attn.indexer.weights_proj",
                ],
            },
        }),
        encoding="utf-8",
    )
    (src / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return src, tmp_path / "out"


def test_indexer_becomes_e4m3_with_a_block_scale(converted):
    src, out = converted
    assert patch(src, out) == 0

    mapping = json.loads(
        (out / "model.safetensors.index.json").read_text())["weight_map"]
    for name, rows in (("wk", 128), ("wq_b", 256)):
        stem = f"model.layers.0.self_attn.indexer.{name}"
        assert f"{stem}.weight_scale_inv" in mapping
        with safe_open(str(out / mapping[f"{stem}.weight"]),
                       framework="pt") as handle:
            assert handle.get_slice(f"{stem}.weight").get_dtype() == "F8_E4M3"
            scale = handle.get_slice(f"{stem}.weight_scale_inv")
            assert scale.get_dtype() == "F32"
            # ceil(rows/128) x ceil(HIDDEN/128), the loader's block formula
            assert scale.get_shape() == [(rows + 127) // 128, HIDDEN // 128]

    # weights_proj is built without a quant_config, so it must stay BF16
    plain = "model.layers.0.self_attn.indexer.weights_proj.weight"
    with safe_open(str(out / mapping[plain]), framework="pt") as handle:
        assert handle.get_slice(plain).get_dtype() == "BF16"


def test_superseded_bf16_bytes_are_gone_not_merely_unindexed(converted):
    """The reason appending cannot work.

    SGLang globs *.safetensors and filters by FILE, then yields every tensor in
    each surviving file. A leftover BF16 weight of the same name would be handed
    to the e4m3 parameter alongside its replacement.
    """
    src, out = converted
    assert patch(src, out) == 0
    key = "model.layers.0.self_attn.indexer.wk.weight"
    shard = json.loads(
        (out / "model.safetensors.index.json").read_text())["weight_map"][key]
    with safe_open(str(out / shard), framework="pt") as handle:
        names = list(handle.keys())
    # exactly one tensor with that name, and it is the converted one
    assert names.count(key) == 1
    with safe_open(str(out / shard), framework="pt") as handle:
        assert handle.get_slice(key).get_dtype() == "F8_E4M3"


def test_source_is_never_mutated(converted):
    src, out = converted
    before = {
        p.name: p.read_bytes() for p in sorted(src.glob("*.safetensors"))
    }
    before_index = (src / "model.safetensors.index.json").read_bytes()
    assert patch(src, out) == 0
    for name, blob in before.items():
        assert (src / name).read_bytes() == blob
    assert (src / "model.safetensors.index.json").read_bytes() == before_index


def test_unaffected_shards_are_linked_not_copied(converted):
    """The whole point: 21 of 40 shards must cost nothing."""
    src, out = converted
    assert patch(src, out) == 0
    clean = "model-00002-of-00002.safetensors"
    a, b = (out / clean).stat(), (src / clean).stat()
    assert (out / clean).is_symlink() or (a.st_ino == b.st_ino
                                         and a.st_dev == b.st_dev)
    # ...while a rewritten shard is a genuinely new file
    dirty = "model-00001-of-00002.safetensors"
    c, d = (out / dirty).stat(), (src / dirty).stat()
    assert c.st_ino != d.st_ino


def test_total_size_matches_the_new_encoding(converted):
    """BF16 -> e4m3 halves the weight and adds a small fp32 scale, so a copied
    total_size would be wrong in a way no shape check would notice."""
    src, out = converted
    assert patch(src, out) == 0
    index = json.loads((out / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]

    expected = 0
    for shard in sorted(set(mapping.values())):
        with safe_open(str(out / shard), framework="pt") as handle:
            for key in handle.keys():
                if key in mapping:
                    t = handle.get_tensor(key)
                    expected += t.numel() * t.element_size()
    assert index["metadata"]["total_size"] == expected

    old = json.loads((src / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] != old["metadata"]["total_size"]


def test_running_twice_is_a_no_op(converted):
    src, out = converted
    assert patch(src, out) == 0
    again = out.parent / "out2"
    assert patch(out, again) == 0
    # second run found nothing to do, so it wrote nothing
    assert not again.exists() or not list(again.glob("*.safetensors"))


def test_now_quantized_modules_leave_ignored_layers(converted):
    """SGLang never reads the field, but vLLM reads it FIRST -- the
    serve_ignore.py incident -- so leaving a lie there is not harmless."""
    src, out = converted
    assert patch(src, out) == 0
    ignored = json.loads(
        (out / "config.json").read_text())["quantization_config"]["ignored_layers"]
    assert "model.layers.0.self_attn.indexer.wk" not in ignored
    assert "model.layers.0.self_attn.indexer.wq_b" not in ignored
    # weights_proj is still BF16, so its entry must survive
    assert "model.layers.0.self_attn.indexer.weights_proj" in ignored
    assert "lm_head" in ignored


def test_a_bad_quantization_refuses_to_write_an_index(converted, monkeypatch):
    """Fail closed: without an index the output is not loadable, which is the
    correct state for an artifact we could not verify."""
    import pipeline.patch_indexer_fp8 as mod

    # Capture the original BEFORE patching; calling mod.quantize_block_fp8 from
    # inside the replacement resolves to the replacement itself.
    original = mod.quantize_block_fp8

    def wrecked(weight, block):
        q, s = original(weight, block)
        return q, s * 4.0  # scale no longer inverts the quantization

    src, out = converted
    monkeypatch.setattr(mod, "quantize_block_fp8", wrecked)
    assert mod.patch(src, out) == 1
    assert not (out / "model.safetensors.index.json").exists()


def test_a_checkpoint_without_an_indexer_is_refused(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    save_file({"model.norm.weight": torch.rand(8).bfloat16()},
              str(src / "model-00001-of-00001.safetensors"),
              metadata={"format": "pt"})
    _write_index(src, {"model.norm.weight": "model-00001-of-00001.safetensors"})
    assert patch(src, tmp_path / "out") == 2


def test_base_crosscheck_catches_a_non_passthrough_indexer(converted, tmp_path):
    """If the recorded indexer is not identical to the BF16 source, something
    folded it, and quantizing the copy would bake that fold in permanently."""
    src, out = converted
    base = tmp_path / "base"
    base.mkdir()
    key = "model.layers.0.self_attn.indexer.wk.weight"
    with safe_open(str(src / "model-00001-of-00002.safetensors"),
                   framework="pt") as handle:
        wk = handle.get_tensor(key)
        wq = handle.get_tensor("model.layers.0.self_attn.indexer.wq_b.weight")
    tampered = {key: (wk.float() * 1.5).bfloat16(),
                "model.layers.0.self_attn.indexer.wq_b.weight": wq}
    save_file(tampered, str(base / "model.safetensors"), metadata={"format": "pt"})
    _write_index(base, {k: "model.safetensors" for k in tampered})
    assert patch(src, out, base=base) == 2


def test_find_targets_separates_done_from_todo(converted):
    src, out = converted
    mapping = json.loads(
        (src / "model.safetensors.index.json").read_text())["weight_map"]
    todo, done = find_targets(src, mapping)
    assert sorted(n.rsplit(".", 2)[-2] for n in todo) == ["wk", "wq_b"]
    assert done == []

    assert patch(src, out) == 0
    mapping2 = json.loads(
        (out / "model.safetensors.index.json").read_text())["weight_map"]
    todo2, done2 = find_targets(out, mapping2)
    assert todo2 == []
    assert len(done2) == 2


def test_dry_run_writes_nothing(converted):
    src, out = converted
    assert patch(src, out, dry_run=True) == 0
    assert not out.exists() or not list(out.iterdir())
