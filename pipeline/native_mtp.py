"""Validated, transactional assembly of one GLM draft layer from its BF16 source.

The main checkpoint is already native. Only new draft shards and small metadata
are written here; quantization is delegated to the existing graft kernels.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.graft_mtp_head import layer_indices, mtp_key_pattern, plan_graft
from pipeline.graft_mtp_w4afp8 import classify, quantize_int4_group_rtn
from pipeline.serve_ignore import weight_map_of
from pipeline.sglang_w4afp8_kernels import pack_nibbles_int8, quantize_block_fp8
from pipeline.to_sglang_w4afp8 import _EXPERT_SHARD_ID

INDEX = "model.safetensors.index.json"
MANIFEST = "native_sglang_manifest.json"
MARKER = ".native_mtp_incomplete.json"
MAX_HEADER_BYTES = 64 * 1024**2
MAX_METADATA_BYTES = 128 * 1024**2
DEFAULT_SHARD_BYTES = 1024**3
_GEOMETRY = (
    "num_hidden_layers",
    "hidden_size",
    "moe_intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "num_attention_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "index_head_dim",
    "index_n_heads",
    "vocab_size",
)


@dataclass(frozen=True)
class NativeMTPPlan:
    source: Path
    source_id: str
    revision: str | None
    config: dict[str, Any]
    weight_map: dict[str, str]
    metadata: dict[str, dict[str, Any]]
    identities: dict[str, tuple[int, ...]]
    fingerprints: dict[str, str]

    @property
    def layer(self) -> int:
        return self.config["num_hidden_layers"]

    def provenance(self) -> dict[str, Any]:
        return {
            "id": self.source_id,
            "snapshot": str(self.source),
            "revision": self.revision,
            "source_tensors": len(self.weight_map),
            "native_tensors": len(native_inventory(self.config)),
            "config_sha256": self.fingerprints["config.json"],
            "index_sha256": self.fingerprints[INDEX],
            "shard_header_sha256": {
                name: digest
                for name, digest in self.fingerprints.items()
                if name not in {"config.json", INDEX}
            },
        }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_bytes(raw: bytes):
    return json.loads(raw, object_pairs_hook=_unique_object)


def _identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _metadata_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError(f"oversized metadata: {path}")
    return raw


def _shard_path(root: Path, name: str) -> Path:
    # Snapshot files can be cache symlinks; never resolve them to the blobs parent.
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError(f"invalid shard filename: {name!r}")
    return root / name


def _header(path: Path):
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        length = int.from_bytes(length_bytes, "little")
        if not 0 < length <= MAX_HEADER_BYTES:
            raise ValueError(f"invalid or oversized safetensors header: {path}")
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError(f"truncated safetensors header: {path}")
    header = _json_bytes(raw)
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header: {path}")
    return header, 8 + length, hashlib.sha256(raw).hexdigest()


def _config_dict(config) -> dict[str, Any]:
    return dict(config) if isinstance(config, dict) else config.to_dict()


def source_inventory(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Closed GLM MTP inventory derived from the architecture's dimensions."""
    if config.get("model_type") != "glm_moe_dsa" or config.get("architectures") != [
        "GlmMoeDsaForCausalLM"
    ]:
        raise ValueError("MTP source requires GlmMoeDsaForCausalLM architecture")
    for field in _GEOMETRY:
        if (
            not isinstance(config.get(field), int)
            or isinstance(config[field], bool)
            or config[field] <= 0
        ):
            raise ValueError(f"MTP source requires positive integer {field}")
    if config["num_hidden_layers"] < 2:
        raise ValueError(
            "MTP source requires depth >= 2; depth 1 has legacy NextN mapping"
        )
    if config.get("tie_word_embeddings", False):
        raise ValueError("MTP source requires untied embeddings")
    h, m = config["hidden_size"], config["moe_intermediate_size"]
    if h % 128 or m % 128:
        raise ValueError("MTP expert input dimensions require INT4 groups of 128")
    n, s = config["n_routed_experts"], config["n_shared_experts"] * m
    heads = config["num_attention_heads"]
    q, kv = config["q_lora_rank"], config["kv_lora_rank"]
    rope, nope, v = (
        config[k] for k in ("qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim")
    )
    ih, nh = config["index_head_dim"], config["index_n_heads"]
    shapes = {
        "eh_proj.weight": [h, 2 * h],
        "enorm.weight": [h],
        "hnorm.weight": [h],
        "input_layernorm.weight": [h],
        "post_attention_layernorm.weight": [h],
        "shared_head.norm.weight": [h],
        "mlp.gate.weight": [n, h],
        "mlp.gate.e_score_correction_bias": [n],
        "mlp.shared_experts.gate_proj.weight": [s, h],
        "mlp.shared_experts.up_proj.weight": [s, h],
        "mlp.shared_experts.down_proj.weight": [h, s],
        "self_attn.q_a_proj.weight": [q, h],
        "self_attn.q_b_proj.weight": [heads * (nope + rope), q],
        "self_attn.kv_a_proj_with_mqa.weight": [kv + rope, h],
        "self_attn.kv_b_proj.weight": [heads * (nope + v), kv],
        "self_attn.o_proj.weight": [h, heads * v],
        "self_attn.q_a_layernorm.weight": [q],
        "self_attn.kv_a_layernorm.weight": [kv],
        "self_attn.indexer.wk.weight": [ih, h],
        "self_attn.indexer.wq_b.weight": [nh * ih, q],
        "self_attn.indexer.weights_proj.weight": [nh, h],
        "self_attn.indexer.k_norm.weight": [ih],
        "self_attn.indexer.k_norm.bias": [ih],
    }
    for expert in range(n):
        for projection, shape in (
            ("gate_proj", [m, h]),
            ("up_proj", [m, h]),
            ("down_proj", [h, m]),
        ):
            shapes[f"mlp.experts.{expert}.{projection}.weight"] = shape
    prefix = f"model.layers.{config['num_hidden_layers']}."
    return {
        prefix + name: {
            "shape": shape,
            "dtype": "F32" if name.endswith("e_score_correction_bias") else "BF16",
        }
        for name, shape in shapes.items()
    }


def native_inventory(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Expected names, shapes and dtypes, independent of an assembly manifest."""
    inventory = {}
    layer = config["num_hidden_layers"]
    for name, spec in source_inventory(config).items():
        kind = classify(name, layer)
        shape = spec["shape"]
        if kind == "copy":
            inventory[name] = {
                "shape": shape,
                "dtype": "torch.float32"
                if spec["dtype"] == "F32"
                else "torch.bfloat16",
            }
            continue
        module = name.removesuffix(".weight")
        out, width = shape
        inventory[name] = {
            "shape": [out, width // 2] if kind == "expert" else shape,
            "dtype": "torch.int8" if kind == "expert" else "torch.float8_e4m3fn",
        }
        inventory[module + ".weight_scale_inv"] = {
            "shape": [out, width // 128]
            if kind == "expert"
            else [(out + 127) // 128, (width + 127) // 128],
            "dtype": "torch.bfloat16" if kind == "expert" else "torch.float32",
        }
        if kind == "expert":
            parent, _, projection = module.rpartition(".")
            inventory[f"{parent}.{_EXPERT_SHARD_ID[projection]}.input_scale"] = {
                "shape": [1],
                "dtype": "torch.bfloat16",
            }
    return inventory


def assert_native_mtp_destination(checkpoint: str | Path) -> None:
    """Reject previous draft publication/staging before costly calibration."""
    checkpoint = Path(checkpoint)
    marker = checkpoint / MARKER
    if marker.exists() or marker.is_symlink():
        raise ValueError("incomplete native MTP assembly marker exists")
    if any(checkpoint.glob("model-mtp*.safetensors")) or any(
        checkpoint.glob(".native-mtp-stage-*")
    ):
        raise FileExistsError("MTP destination contains existing draft/staging files")


def preflight_native_mtp(source: str | Path, model_config) -> NativeMTPPlan:
    """Validate cached source metadata before oneshot, without reading weights.

    Hub IDs must resolve at the loaded config's immutable commit, never at main.
    Local directories retain their snapshot paths, including cache file symlinks.
    """
    config = _config_dict(model_config)
    revision = getattr(model_config, "_commit_hash", None) or config.get("_commit_hash")
    source_id = str(source)
    root = Path(source)
    if root.is_dir():
        root = root.absolute()
    else:
        if not revision:
            raise ValueError("MTP Hub source requires loaded config._commit_hash")
        from pipeline.quantize import _resolve_weight_index

        index_path = _resolve_weight_index(source_id, revision=revision)
        if index_path is None:
            raise ValueError(
                f"MTP source snapshot {source_id}@{revision} is not cached"
            )
        root = index_path.parent
    identities, fingerprints = {}, {}
    documents = {}
    for filename in ("config.json", INDEX):
        path = root / filename
        before = _identity(path)
        raw = _metadata_bytes(path)
        identities[filename] = _identity(path)
        if before != identities[filename]:
            raise ValueError(f"MTP source changed during preflight: {filename}")
        fingerprints[filename] = hashlib.sha256(raw).hexdigest()
        documents[filename] = _json_bytes(raw)
    source_config = documents["config.json"]
    expected = source_inventory(source_config)
    if source_config.get("num_nextn_predict_layers") != 1:
        raise ValueError("MTP source must advertise exactly one draft layer")
    if source_config.get("quantization_config"):
        raise ValueError("MTP source must be the original BF16 checkpoint")
    for field in (*_GEOMETRY, "model_type", "architectures", "tie_word_embeddings"):
        if config.get(field) != source_config.get(field):
            raise ValueError(f"MTP source/main config mismatch: {field}")
    if config.get("num_nextn_predict_layers") != 1:
        raise ValueError("loaded source config must advertise one draft layer")
    weight_map = documents[INDEX].get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("MTP source index requires weight_map")
    layer = source_config["num_hidden_layers"]
    if layer_indices(weight_map) != set(range(layer + 1)):
        raise ValueError("MTP source index depth does not match main and draft layers")
    pattern = mtp_key_pattern(layer)
    names = {name for name in weight_map if pattern.match(name)}
    if names != expected.keys():
        raise ValueError(
            "MTP source inventory incomplete or unexpected: "
            f"missing={sorted(expected.keys() - names)[:4]}, "
            f"extra={sorted(names - expected.keys())[:4]}"
        )
    mtp_map = {name: weight_map[name] for name in sorted(names)}
    metadata = {}
    for filename in sorted(set(mtp_map.values())):
        path = _shard_path(root, filename)
        before = _identity(path)
        header, start, digest = _header(path)
        identities[filename] = _identity(path)
        if before != identities[filename]:
            raise ValueError(f"MTP source changed during preflight: {filename}")
        fingerprints[filename] = digest
        header_names = {name for name in header if pattern.match(name)}
        indexed = {name for name, shard in mtp_map.items() if shard == filename}
        if header_names != indexed:
            raise ValueError(f"MTP source shard/index inventory mismatch: {filename}")
        for name in indexed:
            spec = header[name]
            wanted = expected[name]
            if any(spec.get(k) != wanted[k] for k in ("shape", "dtype")):
                raise ValueError(f"MTP source dtype/geometry mismatch: {name}")
            offsets = spec.get("data_offsets")
            size = 4 if wanted["dtype"] == "F32" else 2
            for dim in wanted["shape"]:
                size *= dim
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(x, int) or isinstance(x, bool) for x in offsets)
                or offsets[0] < 0
                or offsets[1] - offsets[0] != size
                or start + offsets[1] > identities[filename][2]
            ):
                raise ValueError(f"MTP source invalid payload bounds: {name}")
            metadata[name] = spec
        # safetensors validates the complete header (including overlap and holes)
        # without loading payloads. Our bounded reader rejects abusive headers first.
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu"):
            pass
    return NativeMTPPlan(
        root,
        source_id,
        revision,
        source_config,
        mtp_map,
        metadata,
        identities,
        fingerprints,
    )


def _check_source(plan: NativeMTPPlan) -> None:
    for name, identity in plan.identities.items():
        path = plan.source / name
        if _identity(path) != identity:
            raise ValueError(f"MTP source changed since preflight: {name}")
        digest = (
            hashlib.sha256(_metadata_bytes(path)).hexdigest()
            if name in {"config.json", INDEX}
            else _header(path)[2]
        )
        if digest != plan.fingerprints[name] or _identity(path) != identity:
            raise ValueError(f"MTP source changed since preflight: {name}")


def _write_json(path: Path, document) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_additions(stage: Path, weight_map, expected, hashes) -> None:
    from safetensors import safe_open

    from pipeline.native_sglang_save import _tensor_hash

    if weight_map.keys() != expected.keys() or hashes.keys() != expected.keys():
        raise ValueError("staged MTP inventory/hash coverage is incomplete")
    for filename in sorted(set(weight_map.values())):
        with safe_open(stage / filename, framework="pt", device="cpu") as handle:
            names = {name for name, shard in weight_map.items() if shard == filename}
            if set(handle.keys()) != names:
                raise ValueError("staged MTP shard inventory mismatch")
            for name in names:
                tensor = handle.get_tensor(name)
                spec = expected[name]
                if (
                    list(tensor.shape) != spec["shape"]
                    or str(tensor.dtype) != spec["dtype"]
                ):
                    raise ValueError(f"staged MTP dtype/geometry mismatch: {name}")
                if _tensor_hash(tensor) != hashes[name]:
                    raise ValueError(f"staged MTP tensor bytes changed: {name}")
                del tensor


def assemble_native_mtp(
    checkpoint: str | Path,
    plan: NativeMTPPlan,
    *,
    max_shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> dict[str, Any]:
    """Append a verified RTN draft layer after the last collective save barrier.

    Ordinary errors restore original metadata and remove this attempt's files.
    Process interruption leaves a marker, which the standalone verifier rejects.
    Main shards are never rewritten or hashed here; the final native gate reads
    the complete artifact once after this function returns.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from pipeline.native_sglang_save import _tensor_hash

    checkpoint = Path(checkpoint)
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    marker = checkpoint / MARKER
    if marker.exists() or marker.is_symlink():
        raise ValueError("incomplete native MTP assembly marker exists")
    _check_source(plan)
    config = _json_bytes(_metadata_bytes(checkpoint / "config.json"))
    manifest = _json_bytes(_metadata_bytes(checkpoint / MANIFEST))
    if (
        manifest.get("version") != 1
        or manifest.get("format") != "sglang-w4afp8"
        or manifest.get("mtp_present") is not False
        or config.get("num_nextn_predict_layers") != 0
        or config.get("quantization_config", {}).get("quant_method") != "w4afp8"
    ):
        raise ValueError("MTP assembly requires a native checkpoint with absent MTP")
    for field in (*_GEOMETRY, "model_type", "architectures"):
        if config.get(field) != plan.config.get(field):
            raise ValueError(f"MTP target/source config mismatch: {field}")
    main_map = weight_map_of(checkpoint)
    if main_map.keys() != manifest["tensors"].keys():
        raise ValueError("MTP target manifest/index inventory mismatch")
    names = plan_graft(
        {"weight_map": main_map}, {"weight_map": plan.weight_map}, plan.layer
    )
    # Header-only validation of the existing checkpoint, never a second main scan.
    main_bytes = 0
    dtype_names = {
        "I8": "torch.int8",
        "BF16": "torch.bfloat16",
        "F32": "torch.float32",
        "F8_E4M3": "torch.float8_e4m3fn",
        "F16": "torch.float16",
    }
    for filename in sorted(set(main_map.values())):
        header, _, _ = _header(_shard_path(checkpoint, filename))
        indexed = {name for name, shard in main_map.items() if shard == filename}
        if set(header) - {"__metadata__"} != indexed:
            raise ValueError("MTP target shard/index inventory mismatch")
        for name in indexed:
            spec = header[name]
            saved = manifest["tensors"][name]
            if (
                spec["shape"] != saved["shape"]
                or dtype_names.get(spec["dtype"]) != saved["dtype"]
            ):
                raise ValueError(
                    f"MTP target header differs from native manifest: {name}"
                )
            main_bytes += spec["data_offsets"][1] - spec["data_offsets"][0]
    expected = native_inventory(plan.config)
    original = {
        name: (checkpoint / name).read_bytes() if (checkpoint / name).exists() else None
        for name in ("config.json", INDEX, MANIFEST)
    }
    created = []
    published_metadata = []
    stage = None
    # Exclusive creation also prevents concurrent assemblers from both publishing.
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({"policy": "source-rtn", "layer": plan.layer}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        stage = Path(tempfile.mkdtemp(prefix=".native-mtp-stage-", dir=checkpoint))
        additions, hashes, buffer = {}, {}, {}
        buffer_bytes = added_bytes = 0
        shard_count = 0

        def flush():
            nonlocal buffer, buffer_bytes, shard_count
            if not buffer:
                return
            shard_count += 1
            filename = f"model-mtp-{shard_count:05d}.safetensors"
            if (checkpoint / filename).exists() or (checkpoint / filename).is_symlink():
                raise FileExistsError(f"refusing to overwrite MTP shard: {filename}")
            save_file(buffer, str(stage / filename), metadata={"format": "pt"})
            additions.update({name: filename for name in buffer})
            buffer, buffer_bytes = {}, 0

        def emit(name, tensor):
            nonlocal buffer_bytes, added_bytes
            if name.endswith((".weight_scale_inv", ".input_scale")) and (
                not torch.isfinite(tensor).all() or not (tensor > 0).all()
            ):
                raise ValueError(
                    f"MTP quantization produced nonfinite/nonpositive scale: {name}"
                )
            size = tensor.numel() * tensor.element_size()
            if buffer and buffer_bytes + size > max_shard_bytes:
                flush()
            buffer[name] = tensor.contiguous()
            hashes[name] = _tensor_hash(tensor)
            buffer_bytes += size
            added_bytes += size
            if buffer_bytes >= max_shard_bytes:
                flush()

        for filename in sorted(set(plan.weight_map.values())):
            path = plan.source / filename
            if _identity(path) != plan.identities[filename]:
                raise ValueError(f"MTP source changed before reading: {filename}")
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in (
                    name for name in names if plan.weight_map[name] == filename
                ):
                    if _identity(path) != plan.identities[filename]:
                        raise ValueError(
                            f"MTP source changed while reading: {filename}"
                        )
                    weight = handle.get_tensor(name)
                    if not torch.isfinite(weight).all():
                        raise ValueError(f"nonfinite MTP source tensor: {name}")
                    kind = classify(name, plan.layer)
                    module = name.removesuffix(".weight")
                    if kind == "expert":
                        values, scale = quantize_int4_group_rtn(weight)
                        emit(name, pack_nibbles_int8(values))
                        emit(module + ".weight_scale_inv", scale)
                        parent, _, projection = module.rpartition(".")
                        emit(
                            f"{parent}.{_EXPERT_SHARD_ID[projection]}.input_scale",
                            torch.ones(1, dtype=torch.bfloat16),
                        )
                        del values, scale
                    elif kind == "fp8":
                        values, scale = quantize_block_fp8(weight)
                        emit(name, values)
                        emit(module + ".weight_scale_inv", scale)
                        del values, scale
                    else:
                        # Clone copied tensors to keep no source mmap alive in shards.
                        emit(name, weight.clone())
                    del weight
                    if _identity(path) != plan.identities[filename]:
                        raise ValueError(
                            f"MTP source changed while reading: {filename}"
                        )
        flush()
        _check_source(plan)
        _verify_additions(stage, additions, expected, hashes)
        evidence = {
            "policy": "source-rtn",
            "layer": plan.layer,
            "expert_quantization": "rtn",
            "moe_input_scale_policy": "fixed_unit",
            "source": plan.provenance(),
            "source_tensors": len(names),
            "tensors": len(expected),
            "shards": sorted(set(additions.values())),
        }
        manifest["mtp_present"] = True
        manifest["mtp"] = evidence
        manifest["tensors"].update(expected)
        manifest["sha256"].update(hashes)
        config["num_nextn_predict_layers"] = 1
        config["quantization_config"]["mtp_policy"] = "source-rtn"
        ignored = config["quantization_config"].setdefault("ignored_layers", [])
        for name, spec in plan.metadata.items():
            if (
                classify(name, plan.layer) == "copy"
                and name.endswith(".weight")
                and len(spec["shape"]) == 2
            ):
                module = name.removesuffix(".weight")
                if module not in ignored:
                    ignored.append(module)
        index = _json_bytes(original[INDEX]) if original[INDEX] else {"metadata": {}}
        index["weight_map"] = main_map | additions
        index.setdefault("metadata", {})["total_size"] = main_bytes + added_bytes
        for filename, document in (
            ("config.json", config),
            (INDEX, index),
            (MANIFEST, manifest),
        ):
            _write_json(stage / filename, document)
        # Hard links publish without overwriting any colliding path. Only our own
        # successfully linked filenames enter rollback's removal list.
        for filename in evidence["shards"]:
            os.link(stage / filename, checkpoint / filename)
            created.append(filename)
        for filename in (INDEX, "config.json", MANIFEST):
            os.replace(stage / filename, checkpoint / filename)
            published_metadata.append(filename)
        shutil.rmtree(stage)
        stage = None
        marker.unlink()
        return evidence
    except Exception:
        # Keep the marker if rollback itself fails: the artifact must fail closed.
        if published_metadata and stage is None:
            stage = Path(tempfile.mkdtemp(prefix=".native-mtp-stage-", dir=checkpoint))
        for filename in published_metadata:
            raw = original[filename]
            path = checkpoint / filename
            if raw is None:
                path.unlink(missing_ok=True)
            else:
                temporary = stage / f"{filename}.rollback"
                temporary.write_bytes(raw)
                os.replace(temporary, path)
        for filename in created:
            (checkpoint / filename).unlink()
        if stage is not None:
            shutil.rmtree(stage)
        marker.unlink()
        raise


def validate_mtp_manifest(config: dict, manifest: dict) -> set[str]:
    """Fail closed on absent/complete policy and return all required draft hashes."""
    present = manifest.get("mtp_present")
    depth = config.get("num_hidden_layers")
    quant = config.get("quantization_config", {})
    tensors = manifest["tensors"]
    if present is False:
        if (
            config.get("num_nextn_predict_layers") != 0
            or "mtp" in manifest
            or quant.get("mtp_policy", "absent") != "absent"
        ):
            raise ValueError("native absent MTP metadata disagrees")
        if isinstance(depth, int) and any(
            layer >= depth for layer in layer_indices(tensors)
        ):
            raise ValueError("native absent MTP inventory contains draft tensors")
        return set()
    if present is not True or config.get("num_nextn_predict_layers") != 1:
        raise ValueError("native MTP presence/config contract changed")
    expected = native_inventory(config)
    pattern = mtp_key_pattern(depth)
    actual = {name: spec for name, spec in tensors.items() if pattern.match(name)}
    if actual != expected or any(layer > depth for layer in layer_indices(tensors)):
        raise ValueError("native MTP inventory incomplete or dtype/geometry changed")
    evidence = manifest.get("mtp", {})
    if (
        quant.get("mtp_policy") != "source-rtn"
        or evidence.get("policy") != "source-rtn"
        or evidence.get("layer") != depth
        or evidence.get("expert_quantization") != "rtn"
        or evidence.get("moe_input_scale_policy") != "fixed_unit"
        or evidence.get("source_tensors") != len(source_inventory(config))
        or evidence.get("tensors") != len(expected)
    ):
        raise ValueError("native MTP RTN/presence/inventory policy changed")
    source = evidence.get("source", {})
    if not isinstance(source, dict) or not all(
        isinstance(source.get(key), str) and source[key] for key in ("id", "snapshot")
    ):
        raise ValueError("native MTP source provenance missing")
    digests = [source.get("config_sha256"), source.get("index_sha256")]
    headers = source.get("shard_header_sha256")
    if not isinstance(headers, dict) or not headers:
        raise ValueError("native MTP source header provenance missing")
    for filename in headers:
        _shard_path(Path("."), filename)
    digests.extend(headers.values())
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        for digest in digests
    ):
        raise ValueError("native MTP source fingerprints invalid")
    shards = evidence.get("shards")
    if not isinstance(shards, list) or not shards or len(set(shards)) != len(shards):
        raise ValueError("native MTP shard provenance missing")
    for filename in shards:
        _shard_path(Path("."), filename)
    ignored = quant.get("ignored_layers", [])
    for name, spec in source_inventory(config).items():
        if (
            classify(name, depth) == "copy"
            and name.endswith(".weight")
            and len(spec["shape"]) == 2
            and name.removesuffix(".weight") not in ignored
        ):
            raise ValueError(f"native MTP copy missing export ignore metadata: {name}")
    return set(expected)
