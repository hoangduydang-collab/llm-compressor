"""Tests for pipeline.graft_mtp_head against synthetic Safetensors checkpoints.

The real inputs are ~1.35 TiB, so correctness is established on miniature
checkpoints that reproduce the structure that matters: a body of layers 0..N-1, a
head at layer N carrying mixed dtypes (BF16 norms beside FP8 and int-packed
expert weights), tensors spread across multiple source shards, and an index whose
metadata tracks total_size.
"""

import json
import struct
from pathlib import Path

import pytest

from pipeline.graft_mtp_head import (
    graft,
    layer_indices,
    mtp_key_pattern,
    plan_graft,
    verify_graft,
)

INDEX_NAME = "model.safetensors.index.json"


def write_shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """tensors: name -> (dtype, shape, raw payload)."""
    header = {}
    cursor = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(raw)],
        }
        cursor += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for _, (_, _, raw) in tensors.items():
            fh.write(raw)


def make_ckpt(
    root: Path,
    shards: dict[str, dict[str, tuple[str, list[int], bytes]]],
    *,
    total_size: int | None = None,
    config: dict | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for shard, tensors in shards.items():
        write_shard(root / shard, tensors)
        for name in tensors:
            weight_map[name] = shard
    index: dict = {"weight_map": weight_map}
    if total_size is not None:
        index["metadata"] = {"total_size": total_size}
    (root / INDEX_NAME).write_text(json.dumps(index, indent=2))
    if config is not None:
        (root / "config.json").write_text(json.dumps(config, indent=2))
    return root


def body(n_layers: int) -> dict[str, tuple[str, list[int], bytes]]:
    out = {}
    for i in range(n_layers):
        out[f"model.layers.{i}.self_attn.o_proj.weight"] = (
            "F8_E4M3", [4, 4], bytes(range(16))
        )
        out[f"model.layers.{i}.input_layernorm.weight"] = ("BF16", [4], b"\x01" * 8)
    return out


def head(layer: int) -> dict[str, tuple[str, list[int], bytes]]:
    """Mixed dtypes, mirroring a real MTP head."""
    return {
        f"model.layers.{layer}.eh_proj.weight": ("BF16", [4, 4], b"\x02" * 32),
        f"model.layers.{layer}.enorm.weight": ("BF16", [4], b"\x03" * 8),
        f"model.layers.{layer}.hnorm.weight": ("BF16", [4], b"\x04" * 8),
        f"model.layers.{layer}.shared_head.norm.weight": ("BF16", [4], b"\x05" * 8),
        f"model.layers.{layer}.mlp.experts.0.gate_proj.weight": (
            "I32", [2, 2], b"\x06" * 16
        ),
        f"model.layers.{layer}.mlp.experts.0.gate_proj.weight_scale_inv": (
            "F8_E4M3", [2], b"\x07" * 2
        ),
    }


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """Head deliberately split across two shards, as in a real release."""
    h = head(3)
    keys = sorted(h)
    return make_ckpt(
        tmp_path / "source",
        {
            "model-00001-of-00003.safetensors": body(3),
            "model-00002-of-00003.safetensors": {k: h[k] for k in keys[:3]},
            "model-00003-of-00003.safetensors": {k: h[k] for k in keys[3:]},
        },
        total_size=9999,
        config={"num_hidden_layers": 3, "num_nextn_predict_layers": 1},
    )


@pytest.fixture
def target(tmp_path: Path) -> Path:
    """Our quantized output: layers 0..2, no head."""
    return make_ckpt(
        tmp_path / "target",
        {"model-00001-of-00001.safetensors": body(3)},
        total_size=1000,
        config={"num_hidden_layers": 3},
    )


def test_layer_indices_and_pattern():
    assert layer_indices(["model.layers.7.x", "layers.12.y", "lm_head.weight"]) == {
        7,
        12,
    }
    pat = mtp_key_pattern(78)
    assert pat.match("model.layers.78.eh_proj.weight")
    assert pat.match("layers.78.enorm.weight")
    # must not match layer 7 or 780
    assert not pat.match("model.layers.7.eh_proj.weight")
    assert not pat.match("model.layers.780.eh_proj.weight")


def test_graft_copies_head_faithfully(target: Path, source: Path):
    result = graft(target, source, 3)

    assert result["tensors"] == 6
    assert result["verified"] is True
    # dtypes preserved exactly — no reinterpretation
    assert set(result["dtypes"]) == {"BF16", "F8_E4M3", "I32"}
    assert result["num_nextn_predict_layers"] == 1

    # payloads are byte-identical to the source
    from pipeline.reexport_minimax_m3_vllm import _build_tensor_reader, _load_index

    src_map = _load_index(source / INDEX_NAME)["weight_map"]
    tgt_map = _load_index(target / INDEX_NAME)["weight_map"]
    src_read = _build_tensor_reader(source, src_map)
    tgt_read = _build_tensor_reader(target, tgt_map)
    for key in head(3):
        assert src_read(key) == tgt_read(key), key


def test_graft_updates_total_size(target: Path, source: Path):
    before = json.loads((target / INDEX_NAME).read_text())["metadata"]["total_size"]
    result = graft(target, source, 3)
    after = json.loads((target / INDEX_NAME).read_text())["metadata"]["total_size"]
    assert after == before + result["written_bytes"]


def test_graft_is_not_repeatable(target: Path, source: Path):
    graft(target, source, 3)
    with pytest.raises(ValueError, match="refusing to graft twice"):
        graft(target, source, 3)


def test_graft_refuses_on_layer_gap(tmp_path: Path, source: Path):
    """Target with only layers 0..1 would leave a hole at layer 2."""
    short = make_ckpt(
        tmp_path / "short",
        {"model-00001-of-00001.safetensors": body(2)},
        total_size=500,
    )
    with pytest.raises(ValueError, match="highest layer is 1, expected 2"):
        graft(short, source, 3)


def test_graft_refuses_when_source_lacks_the_layer(target: Path, source: Path):
    with pytest.raises(ValueError, match="no tensors for layer 9"):
        graft(target, source, 9)


def test_dry_run_writes_nothing(target: Path, source: Path):
    before = sorted(p.name for p in target.iterdir())
    result = graft(target, source, 3, dry_run=True)
    assert result["dry_run"] is True
    assert result["tensors"] == 6
    assert result["bytes"] == sum(len(v[2]) for v in head(3).values())
    assert sorted(p.name for p in target.iterdir()) == before


def test_shard_splitting(target: Path, source: Path):
    """A small cap must split the head across several shards, still verifying."""
    result = graft(target, source, 3, max_shard_bytes=16)
    assert result["shards"] > 1
    assert len(result["shard_names"]) > 1
    for name in result["shard_names"]:
        assert (target / name).exists()
    assert verify_graft(target, source, 3)["verified"] is True


def test_verify_detects_a_missing_shard(target: Path, source: Path):
    graft(target, source, 3)
    index = json.loads((target / INDEX_NAME).read_text())
    shard = next(
        s for k, s in index["weight_map"].items() if k.startswith("model.layers.3.")
    )
    (target / shard).unlink()
    with pytest.raises(ValueError, match="index names a missing"):
        verify_graft(target, source, 3)


def test_verify_detects_a_dtype_mismatch(target: Path, source: Path):
    """Corrupt one grafted tensor's dtype; verification must reject it."""
    graft(target, source, 3)
    index = json.loads((target / INDEX_NAME).read_text())
    key = "model.layers.3.enorm.weight"
    shard = target / index["weight_map"][key]
    raw = shard.read_bytes()
    hdr_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + hdr_len])
    header[key]["dtype"] = "F32"
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    shard.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + hdr_len :])
    with pytest.raises(ValueError, match="in target"):
        verify_graft(target, source, 3)


def test_source_is_never_modified(target: Path, source: Path):
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    graft(target, source, 3)
    after = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert before == after


def test_plan_graft_reports_source_layers_when_empty(target: Path, source: Path):
    tgt_index = json.loads((target / INDEX_NAME).read_text())
    src_index = json.loads((source / INDEX_NAME).read_text())
    with pytest.raises(ValueError, match=r"Source layers present"):
        plan_graft(tgt_index, src_index, 42)
