"""First-write GLM W4AFP8 export through the existing collective CT save path.

Only the compressor factory used by llm-compressor's save wrapper is scoped.
CT still owns module distribution/offload and Transformers owns sharding/writing.
Native parameters are restored losslessly to CT compressed state on exit; no
second checkpoint, original-weight snapshot, or FP8 requantizer is involved.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch
from compressed_tensors import ModelCompressor
from compressed_tensors.compressors.naive_quantized import FloatQuantizationCompressor
from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
from compressed_tensors.config import CompressionFormat
from compressed_tensors.distributed import replace_module_parallel
from compressed_tensors.offload import (
    disable_onloading,
    from_accelerate,
    is_distributed,
)
from compressed_tensors.quantization import QuantizationScheme, QuantizationStatus
from compressed_tensors.utils import get_direct_state_dict, replace_direct_state_dict
from compressed_tensors.utils.helpers import patch_attr

from pipeline.serve_ignore import weight_map_of
from pipeline.sglang_w4afp8_kernels import pack_nibbles_int8, unpack_nibbles_int8
from pipeline.to_sglang_w4afp8 import (
    _EXPERT_SHARD_ID,
    build_config,
    default_unpacker,
    source_fp8_layout,
)

_EXPERT = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
)
_FP8 = re.compile(
    r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|q_a_proj|q_b_proj|"
    r"kv_a_proj_with_mqa|kv_b_proj|o_proj|indexer\.(?:wk|wq_b))|"
    r"mlp\.(?:shared_experts\.)?(?:gate_proj|up_proj|down_proj))$"
)
_UNQUANTIZED = re.compile(
    r"^(?:lm_head|model\.embed_tokens|model\.layers\.\d+\.(?:mlp\.gate|"
    r"self_attn\.indexer\.weights_proj))$"
)
_ACTIVE = False
_MISSING = object()


def assert_native_sglang_preflight(
    model: torch.nn.Module,
    schemes: Mapping[str, QuantizationScheme] | None = None,
) -> dict[str, int]:
    """Validate actual GLM inventory without reading offloaded weights.

    Before calibration, supply resolved exact module names -> effective schemes.
    At save time omit ``schemes`` to inspect attached quantization schemes.
    This intentionally rejects partial recipes over a full model, fused expert
    tensors, group activation ordering and non-native FP8 layouts.
    """
    if getattr(model.config, "model_type", None) != "glm_moe_dsa":
        raise ValueError(
            "native SGLang export requires the glm_moe_dsa module contract"
        )
    if getattr(model.config, "tie_word_embeddings", False):
        raise ValueError("native GLM save requires untied word embeddings")
    counts = {"expert": 0, "fp8": 0, "unquantized": 0}
    known = set()
    with disable_onloading():
        for name, module in model.named_modules():
            known.add(name)
            scheme = (
                schemes.get(name)
                if schemes is not None
                else getattr(module, "quantization_scheme", None)
            )
            state = get_direct_state_dict(module)
            weight = state.get("weight", state.get("weight_packed"))
            if weight is None:
                if scheme is not None:
                    raise ValueError(f"{name}: quantized module has no weight")
                # Fused experts use gate_up_proj/down_proj instead of weight.
                if any(t is not None and t.ndim >= 2 for t in state.values()):
                    raise ValueError(
                        f"{name}: unsupported fused or custom tensor inventory"
                    )
                continue
            expert = bool(_EXPERT.fullmatch(name))
            fp8 = bool(_FP8.fullmatch(name))
            if not expert and not fp8:
                if scheme is not None:
                    raise ValueError(f"{name}: SGLang requires this module unquantized")
                if weight.ndim >= 2 and not _UNQUANTIZED.fullmatch(name):
                    raise ValueError(f"{name}: unsupported GLM weight module")
                counts["unquantized"] += 1
                continue
            if not isinstance(module, torch.nn.Linear) or weight.ndim != 2:
                raise ValueError(
                    f"{name}: native export requires unfused 2-D Linear weights"
                )
            if not isinstance(scheme, QuantizationScheme) or scheme.weights is None:
                raise ValueError(
                    f"{name}: missing required "
                    f"{'INT4' if expert else 'FP8_BLOCK'} scheme"
                )
            args = scheme.weights
            if not args.symmetric or args.dynamic:
                raise ValueError(
                    f"{name}: native export requires symmetric static weights"
                )
            if args.actorder == "group" or "weight_g_idx" in state:
                raise ValueError(
                    f"{name}: group actorder/weight_g_idx cannot be represented"
                )
            if scheme.output_activations is not None:
                raise ValueError(
                    f"{name}: output activation quantization is unsupported"
                )
            activation = scheme.input_activations
            if activation is not None and not (
                activation.type == "float"
                and activation.num_bits == 8
                and activation.symmetric
                and activation.dynamic
                and (
                    activation.strategy in ("token", "tensor")
                    or (
                        not expert
                        and activation.strategy == "group"
                        and activation.group_size == 128
                    )
                )
            ):
                raise ValueError(
                    f"{name}: only absent or dynamic FP8 "
                    "calibration activations are supported"
                )
            if any(k.startswith(("input_", "output_")) for k in state):
                raise ValueError(
                    f"{name}: persisted activation parameters "
                    "conflict with fixed-unit policy"
                )
            if expert:
                if not (
                    args.type == "int"
                    and args.num_bits == 4
                    and args.strategy == "group"
                    and args.group_size == 128
                ):
                    raise ValueError(
                        f"{name}: routed experts require symmetric INT4 group 128"
                    )
                if module.in_features % 128 or module.out_features % 128:
                    raise ValueError(
                        f"{name}: expert dimensions must be divisible by 128"
                    )
                if state.get("bias") is not None:
                    raise ValueError(f"{name}: routed expert bias is unsupported")
                parent, _, projection = name.rpartition(".")
                if hasattr(model.get_submodule(parent), _EXPERT_SHARD_ID[projection]):
                    raise ValueError(
                        f"{name}: native input-scale module name already exists"
                    )
            elif not (
                args.type == "float"
                and args.num_bits == 8
                and args.strategy == "block"
                and args.block_structure == [128, 128]
            ):
                raise ValueError(
                    f"{name}: SGLang requires symmetric static FP8 128x128 blocks"
                )
            counts["expert" if expert else "fp8"] += 1
    if schemes is not None and set(schemes) - known:
        raise ValueError(
            "native scheme map contains unknown modules: "
            f"{sorted(set(schemes) - known)[:5]}"
        )
    if not counts["expert"] or not counts["fp8"]:
        raise ValueError(
            "native W4AFP8 export requires both routed INT4 and block FP8 modules"
        )
    # The main Transformers class does not instantiate a draft layer. Never
    # inherit source metadata that claims a missing MTP head is present.
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is not None:
        for name in known:
            match = re.match(r"model\.layers\.(\d+)(?:\.|$)", name)
            if match and int(match[1]) >= num_layers:
                raise ValueError("native save currently supports absent MTP only")
    return counts


def _apply_modules(modules, fn, desc):
    if is_distributed():
        replace_module_parallel(modules, fn, desc=desc)
    else:
        for module in modules:
            fn(module)


def _validate_native_qparams(entries):
    """Agree on scale/layout errors while CT can still recouple original state.

    CT's replacement callback must not raise on one owner before peers recouple.
    This round never changes a parameter's shape/name and catches owner errors.
    Production offload reads only qparams; weight layout comes from meta tensors.
    """
    errors = []

    def validate(module):
        name = entries[module]
        try:
            with disable_onloading():
                state = get_direct_state_dict(module)
            compressed = module.quantization_status == QuantizationStatus.COMPRESSED
            expert = bool(_EXPERT.fullmatch(name))
            key = "weight_packed" if expert and compressed else "weight"
            weight = state[key]
            expected_shape = (module.out_features, module.in_features)
            if expert and compressed:
                expected_shape = (module.out_features, module.in_features // 8)
            if tuple(weight.shape) != expected_shape:
                raise ValueError(f"invalid {key} geometry")
            expected_dtype = torch.int32 if expert else torch.float8_e4m3fn
            if compressed and weight.dtype != expected_dtype:
                raise ValueError(f"invalid compressed {key} dtype")
            if not compressed and weight.dtype not in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ):
                raise ValueError("working weights must be BF16/FP16/FP32")
            scale = module.weight_scale
            expected_scale = (
                (module.out_features, module.in_features // 128)
                if expert
                else (
                    (module.out_features + 127) // 128,
                    (module.in_features + 127) // 128,
                )
            )
            if tuple(scale.shape) != expected_scale:
                raise ValueError("invalid weight scale geometry")
            if scale.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise ValueError("weight scales must be BF16/FP16/FP32")
            if scale.device.type != "meta" and (
                not torch.isfinite(scale).all() or not (scale > 0).all()
            ):
                raise ValueError("weight scales must be finite and positive")
            zero_point = getattr(module, "weight_zero_point", None)
            if zero_point is not None:
                if tuple(zero_point.shape) != expected_scale:
                    raise ValueError("invalid weight zero-point geometry")
                if zero_point.device.type != "meta" and (zero_point.float() != 0).any():
                    raise ValueError(
                        "symmetric native weights require zero zero-points"
                    )
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    _apply_modules(list(entries), validate, "Validating native SGLang parameters")
    if is_distributed():
        import torch.distributed as dist

        errors_by_rank = [None] * dist.get_world_size()
        dist.all_gather_object(errors_by_rank, errors)
        errors = [error for rank_errors in errors_by_rank for error in rank_errors]
    if errors:
        raise ValueError(
            "native SGLang parameter preflight failed: " + "; ".join(errors[:8])
        )


@contextmanager
def native_sglang_save(model: torch.nn.Module):
    """Wrap collective ``model.save_pretrained(..., save_compressed=True)``.

    Call on every rank. On successful exit the working model retains ordinary
    CT compressed parameters and its CT decompression hook, including original
    FP8 scale dtype/values. A later forward therefore uses CT as usual.
    This process-scoped compressor-factory adapter forbids nested contexts;
    concurrent saves in another thread are unsupported.
    """
    global _ACTIVE
    from llmcompressor.transformers.compression import (
        compressed_tensors_utils as save_api,
    )

    if _ACTIVE:
        raise RuntimeError("native SGLang save contexts cannot overlap")
    counts = assert_native_sglang_preflight(model)
    if not getattr(model.save_pretrained, "_overridden", False):
        raise ValueError(
            "native save requires llm-compressor's collective save_pretrained wrapper"
        )
    entries = {
        m: n
        for n, m in model.named_modules()
        if getattr(m, "quantization_scheme", None) is not None
    }
    with disable_onloading():
        for module, name in entries.items():
            if module.quantization_status not in (
                QuantizationStatus.FROZEN,
                QuantizationStatus.COMPRESSED,
                QuantizationStatus.DECOMPRESSED,
            ):
                raise ValueError(f"{name}: finalize calibration before native save")
        scale_dtypes = {m: m.weight_scale.dtype for m in entries}
    source_config = model.config.to_dict()
    source_config["quantization_config"] = {
        "ignore": [
            name
            for name, m in model.named_modules()
            if m not in entries and _UNQUANTIZED.fullmatch(name)
        ]
    }
    native_config = build_config(source_config, module_names=entries.values())
    native_config["num_nextn_predict_layers"] = 0
    native_config["quantization_config"].update(
        {
            "weight_block_size": [128, 128],
            "linear_activation_scheme": "dynamic",
            "moe_activation_scheme": "static",
            "moe_input_scale_policy": "fixed_unit",
        }
    )
    config_previous = {
        key: getattr(model.config, key, _MISSING)
        for key in ("quantization_config", "num_nextn_predict_layers")
    }
    native_modules = set()
    tensor_hashes = {}
    manifest = {}
    scale_modules = []
    original_factory = save_api.ModelCompressor
    original_save = model.save_pretrained

    @wraps(original_save)
    def checked_save(save_directory, **kwargs):
        if kwargs.get("save_compressed", True) is not True:
            raise ValueError("native save requires save_compressed=True")
        if kwargs.get("state_dict") is not None or kwargs.get("variant") is not None:
            raise ValueError(
                "native save requires its own unmodified model state and shard names"
            )
        # This contract is defined over the unfused GLM module inventory. A
        # Transformers checkpoint conversion must not rename/re-fuse those keys.
        kwargs["save_original_format"] = False
        return original_save(save_directory, **kwargs)

    def compress(module):
        name = entries[module]
        scheme = module.quantization_scheme
        expert = bool(_EXPERT.fullmatch(name))
        scheme.format = (
            CompressionFormat.pack_quantized
            if expert
            else CompressionFormat.float_quantized
        )
        compressor = (
            PackedQuantizationCompressor if expert else FloatQuantizationCompressor
        )
        state = get_direct_state_dict(module)
        if module.quantization_status != QuantizationStatus.COMPRESSED:
            state = compressor.compress(state, scheme)
        # Keep all arithmetic in the installed CT compressor and the existing
        # independently checked native integer repacker.
        if expert:
            packed = state.pop("weight_packed")
            shape = (module.out_features, module.in_features)
            state.pop("weight_shape")
            if packed.device.type == "meta":
                state["weight"] = torch.empty(
                    (shape[0], shape[1] // 2), dtype=torch.int8, device="meta"
                )
            else:
                state["weight"] = pack_nibbles_int8(
                    default_unpacker(packed, torch.Size(shape))
                )
            scale = state.pop("weight_scale")
            if scale.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise ValueError(f"{name}: INT4 scales must be BF16/FP16/FP32")
            if tuple(scale.shape) != (shape[0], shape[1] // 128):
                raise ValueError(f"{name}: invalid INT4 scale geometry")
            if scale.device.type != "meta" and (
                not torch.isfinite(scale).all() or not (scale > 0).all()
            ):
                raise ValueError(f"{name}: INT4 scales must be finite and positive")
            state["weight_scale_inv"] = scale
        else:
            scale = state.pop("weight_scale")
            if state["weight"].device.type != "meta":
                config = {
                    "quantization_config": {
                        "config_groups": {
                            "fp8": scheme.model_copy(
                                update={"targets": [name]}
                            ).model_dump(mode="json")
                        }
                    }
                }
                if source_fp8_layout(name, state["weight"], scale, config) != "block":
                    raise ValueError(f"{name}: native block FP8 required")
            state["weight_scale_inv"] = scale.float()
        for key, value in state.items():
            if value is not None and value.device.type != "meta":
                tensor_hashes[f"{name}.{key}"] = _tensor_hash(value)
        replace_direct_state_dict(
            module,
            {
                key: value.contiguous() if value is not None else None
                for key, value in state.items()
            },
        )
        module.quantization_status = QuantizationStatus.COMPRESSED
        native_modules.add(module)

    def restore(module):
        if module not in native_modules:
            return
        state = get_direct_state_dict(module)
        state["weight_scale"] = state.pop("weight_scale_inv").to(scale_dtypes[module])
        if _EXPERT.fullmatch(entries[module]):
            weight = state.pop("weight")
            if weight.device.type == "meta":
                packed = torch.empty(
                    (module.out_features, module.in_features // 8),
                    dtype=torch.int32,
                    device="meta",
                )
            else:
                packed = pack_to_int32(unpack_nibbles_int8(weight), 4)
            state["weight_packed"] = packed.contiguous()
            state["weight_shape"] = torch.tensor(
                [module.out_features, module.in_features]
            )
        replace_direct_state_dict(module, state)

    class NativeCompressor(ModelCompressor):
        @classmethod
        def from_pretrained_model(cls, candidate, *args, **kwargs):
            if candidate is not model:
                return original_factory.from_pretrained_model(
                    candidate, *args, **kwargs
                )
            if kwargs.get("quantization_format") is not None:
                raise ValueError(
                    "native mixed save must not force a global quantization format"
                )
            return super().from_pretrained_model(candidate, *args, **kwargs)

        def compress_model(self, candidate):
            if native_modules:
                raise RuntimeError("only one save is supported per native context")
            _validate_native_qparams(entries)
            _apply_modules(list(entries), compress, "Compressing native SGLang model")
            if is_distributed():
                import torch.distributed as dist

                hashes_by_rank = [None] * dist.get_world_size()
                dist.all_gather_object(hashes_by_rank, tensor_hashes)
                for hashes in hashes_by_rank:
                    tensor_hashes.update(hashes)
            self.quantization_config.quantization_status = QuantizationStatus.COMPRESSED
            self.remove_decompression_hook(candidate)
            self.add_decompress_hook(candidate)
            for module, name in entries.items():
                if not _EXPERT.fullmatch(name):
                    continue
                parent_name, _, projection = name.rpartition(".")
                parent = model.get_submodule(parent_name)
                suffix = _EXPERT_SHARD_ID[projection]
                holder = torch.nn.Module()
                holder.register_buffer(
                    "input_scale", torch.ones(1, dtype=torch.bfloat16)
                )
                parent.add_module(suffix, holder)
                scale_modules.append((parent, suffix))
                tensor_hashes[f"{parent_name}.{suffix}.input_scale"] = _tensor_hash(
                    holder.input_scale
                )
            for key in config_previous:
                setattr(model.config, key, native_config[key])
            with disable_onloading():
                inventory = {
                    name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                    for name, tensor in model.state_dict().items()
                }
            manifest.update(
                {
                    "version": 1,
                    "format": "sglang-w4afp8",
                    "mtp_present": False,
                    "moe_input_scale_policy": "fixed_unit",
                    "tensors": inventory,
                    "sha256": tensor_hashes,
                }
            )

        def update_config(self, save_directory):
            if not native_modules:
                raise ValueError("native save requires save_compressed=True")
            save_directory = Path(save_directory)
            serialized = _serialized_tensor_inventory(save_directory)
            expected = manifest["tensors"]
            if set(serialized) != set(expected):
                raise ValueError(
                    "serialized tensor inventory differs from resident save state"
                )
            for name, spec in serialized.items():
                if spec["shape"] != expected[name]["shape"]:
                    raise ValueError(f"{name}: serialized tensor shape changed")
                if name in tensor_hashes and spec["dtype"] != expected[name]["dtype"]:
                    raise ValueError(
                        f"{name}: serialized quantized tensor dtype changed"
                    )
            # Offloaded modules can retain FP32 meta placeholders while their
            # materialized save tensors use the requested BF16 model dtype.
            # The manifest must describe the bytes Transformers actually wrote.
            manifest["tensors"] = serialized

            path = save_directory / "config.json"
            config = json.loads(path.read_text())
            config["quantization_config"] = native_config["quantization_config"]
            config["num_nextn_predict_layers"] = 0
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            (save_directory / "native_sglang_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )

    _ACTIVE = True
    try:
        with (
            patch_attr(save_api, "ModelCompressor", NativeCompressor),
            patch_attr(model, "save_pretrained", checked_save),
        ):
            yield {
                **counts,
                "mtp_present": False,
                "moe_input_scale_policy": "fixed_unit",
            }
    finally:
        try:
            if native_modules:
                # The CT save wrapper normally restores offloading itself. If
                # the writer raised, its final from_accelerate was skipped.
                from_accelerate(model)
                with save_api._contiguous_disk_serialization():
                    _apply_modules(list(entries), restore, "Restoring CT model layout")
        finally:
            for parent, suffix in scale_modules:
                delattr(parent, suffix)
            for key, previous in config_previous.items():
                if previous is _MISSING:
                    if hasattr(model.config, key):
                        delattr(model.config, key)
                else:
                    setattr(model.config, key, previous)
            _ACTIVE = False


def _tensor_hash(tensor: torch.Tensor) -> str:
    """Hash raw storage without materializing a Python bytes copy."""
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _serialized_tensor_inventory(checkpoint: Path) -> dict[str, dict[str, object]]:
    """Read shape/dtype metadata from written shard headers without loading data."""
    from safetensors import safe_open
    from safetensors.torch import _getdtype

    weight_map = weight_map_of(checkpoint)
    inventory = {}
    for filename in sorted(set(weight_map.values())):
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor_slice = handle.get_slice(name)
                inventory[name] = {
                    "shape": list(tensor_slice.get_shape()),
                    "dtype": str(_getdtype(tensor_slice.get_dtype())),
                }
    if set(inventory) != set(weight_map):
        raise ValueError("written shard/index tensor inventory mismatch")
    return inventory


def verify_native_sglang_checkpoint(checkpoint: str | Path) -> dict[str, int | bool]:
    """Source-only exact serialization gate; never onloads distributed model state.

    Validate full key/shape/dtype inventory, and every native quantized tensor's
    bytes against its resident save-time hash. Read at most one tensor at a time.
    Main unquantized tensors have structural checks only; assembled MTP tensors
    all require hashes, including copied BF16/FP32 tensors. This certifies
    serialization, not calibration quality or an SGLang GPU execution path.
    """
    from safetensors import safe_open

    from pipeline.native_mtp import MARKER, validate_mtp_manifest

    checkpoint = Path(checkpoint)
    if (checkpoint / MARKER).exists() or (checkpoint / MARKER).is_symlink():
        raise ValueError("incomplete native MTP assembly marker exists")
    manifest = json.loads((checkpoint / "native_sglang_manifest.json").read_text())
    if manifest.get("version") != 1 or manifest.get("format") != "sglang-w4afp8":
        raise ValueError("unsupported native SGLang manifest")
    config = json.loads((checkpoint / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if (
        quant.get("quant_method") != "w4afp8"
        or quant.get("group_size") != 128
        or quant.get("weight_block_size") != [128, 128]
        or quant.get("moe_input_scale_policy") != "fixed_unit"
        or quant.get("moe_activation_scheme") != "static"
        or quant.get("linear_activation_scheme") != "dynamic"
    ):
        raise ValueError("native SGLang configuration/activation/MTP contract changed")
    mtp_hashes = validate_mtp_manifest(config, manifest)
    expected = manifest["tensors"]
    hashes = manifest["sha256"]
    required_hashes = {
        name
        for name in expected
        if name.endswith((".weight_scale_inv", ".input_scale"))
    }
    required_hashes.update(
        name.removesuffix("weight_scale_inv") + "weight"
        for name in required_hashes.copy()
        if name.endswith(".weight_scale_inv")
    )
    required_hashes.update(mtp_hashes)
    if not required_hashes <= hashes.keys() or not hashes.keys() <= expected.keys():
        raise ValueError("native manifest is missing quantized tensor hashes")
    weight_map = weight_map_of(checkpoint)
    if set(weight_map) != set(expected):
        raise ValueError(
            "native checkpoint tensor inventory differs from save-time inventory"
        )
    if mtp_hashes:
        mtp_shards = {weight_map[name] for name in mtp_hashes}
        if mtp_shards != set(manifest["mtp"]["shards"]) or any(
            shard in mtp_shards for name, shard in weight_map.items()
            if name not in mtp_hashes
        ):
            raise ValueError("native MTP shard provenance/inventory changed")
    checked = set()
    for filename in sorted(set(weight_map.values())):
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
            indexed_keys = {
                key for key, value in weight_map.items() if value == filename
            }
            if shard_keys != indexed_keys:
                raise ValueError(f"{filename}: shard/index tensor inventory mismatch")
            for name in sorted(shard_keys):
                tensor = handle.get_tensor(name)
                spec = expected[name]
                if (
                    list(tensor.shape) != spec["shape"]
                    or str(tensor.dtype) != spec["dtype"]
                ):
                    raise ValueError(f"{name}: native shape/dtype changed")
                if name in hashes and _tensor_hash(tensor) != hashes[name]:
                    raise ValueError(
                        f"{name}: native tensor bytes changed after compression"
                    )
                checked.add(name)
                del tensor
    return {"ok": True, "tensors": len(checked), "hashed_tensors": len(hashes)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify a direct native SGLang checkpoint"
    )
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_native_sglang_checkpoint(args.checkpoint), sort_keys=True))
