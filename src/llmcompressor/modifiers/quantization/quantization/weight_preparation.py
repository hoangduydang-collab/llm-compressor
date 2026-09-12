"""Prepare FP8 working weights before downstream GPTQ calibration.

Reuse MoE-Quant's FP8-dequantized working-weight ordering, with this project's
CT observers, quantizer and persistent distributed publication. No activation
quantization or new rounding arithmetic is introduced here.
"""

import torch
from compressed_tensors.quantization import fake_quantize
from compressed_tensors.utils import match_named_modules, update_offload_parameter
from torch import distributed as dist

from llmcompressor.modifiers.quantization.calibration import observe
from llmcompressor.utils.metric_logging import compression_phase


def _validate_weight_preparation(model, modifiers, propagate_error):
    """Resolve disjoint FP8 targets before any calibration forward or mutation."""
    preparers = [
        m for m in modifiers if getattr(m, "quantize_weights_before_calibration", False)
    ]
    if not preparers:
        return []
    if not propagate_error:
        raise ValueError("FP8 weight preparation requires propagate_error=True")
    if sum(type(m).__name__ == "GPTQModifier" for m in modifiers) != 1 or any(
        type(m).__name__ not in ("GPTQModifier", "QuantizationModifier")
        for m in modifiers
    ):
        raise ValueError("FP8 weight preparation requires one GPTQ and RTN modifiers")
    targets = [
        dict(
            (module, name)
            for name, module in match_named_modules(model, m.resolved_targets, m.ignore)
        )
        for m in modifiers
    ]
    for index, modifier in enumerate(modifiers):
        if not getattr(modifier, "quantize_weights_before_calibration", False):
            continue
        selected = targets[index]
        if not selected:
            raise ValueError("FP8 weight preparation matched no modules")
        others = set().union(*(set(t) for i, t in enumerate(targets) if i != index))
        overlap = set(selected) & others
        if overlap:
            raise ValueError(
                "FP8 weight preparation requires disjoint targets: "
                + ", ".join(sorted(selected[m] for m in overlap))
            )
        for module, name in selected.items():
            scheme = getattr(module, "quantization_scheme", None)
            args = getattr(scheme, "weights", None)
            if not (
                isinstance(module, torch.nn.Linear)
                and args is not None
                and args.type == "float"
                and args.num_bits == 8
                and args.strategy == "block"
                and args.block_structure == [128, 128]
                and args.symmetric
                and not args.dynamic
                and args.actorder is None
            ):
                raise ValueError(
                    f"{name}: preparation requires symmetric FP8_BLOCK weights"
                )
            if any(
                a is not None and (not a.dynamic or a.observer is not None)
                for a in (scheme.input_activations, scheme.output_activations)
            ):
                raise ValueError(
                    f"{name}: preparation requires observer-free dynamic activations"
                )
        modifier._weight_preparation_targets = selected
    return preparers


def validate_weight_preparation(model, modifiers, propagate_error=True):
    """Make preparation preflight failures collective before any rank mutates."""
    distributed = dist.is_available() and dist.is_initialized()
    if not distributed:
        return _validate_weight_preparation(model, modifiers, propagate_error)
    from llmcompressor.modifiers.gptq.distributed import agree_ep_manifest

    flags = [
        bool(getattr(m, "quantize_weights_before_calibration", False))
        for m in modifiers
    ]
    agree_ep_manifest({"fp8_weight_preparation": flags})
    if not any(flags):
        return []
    from compressed_tensors.offload.cache.dist_disk import DistributedDiskCache

    from llmcompressor.modifiers.gptq.distributed import (
        gather_ep_records,
        parameter_storage,
    )

    manifest = []
    preparers, error = [], None
    try:
        preparers = _validate_weight_preparation(model, modifiers, propagate_error)
        for modifier in preparers:
            for module, name in sorted(
                modifier._weight_preparation_targets.items(), key=lambda item: item[1]
            ):
                parameters = []
                for attribute in ("weight", "weight_scale", "weight_zero_point"):
                    cache, value = parameter_storage(module, attribute)
                    if isinstance(
                        cache, DistributedDiskCache
                    ) and "update_offload" not in vars(type(cache)):
                        raise ValueError(
                            f"{name}: distributed-safe disk update patch is required"
                        )
                    parameters.append(
                        (
                            attribute,
                            type(cache).__name__,
                            tuple(value.shape) if value is not None else None,
                            str(value.dtype) if value is not None else None,
                        )
                    )
                manifest.append(
                    (
                        name,
                        module.quantization_scheme.model_dump(mode="json"),
                        parameters,
                    )
                )
    except ValueError as exc:
        error = str(exc)
    errors = gather_ep_records(error)
    if any(errors):
        raise ValueError(f"FP8 weight preparation preflight failed: {errors}")
    agree_ep_manifest({"fp8_weight_targets": manifest})
    return preparers


def _quantized_parameters(module):
    # Use the original working weight exactly once to choose the FP8 scales.
    observe(module, "weight")
    qparams = module.weight_observer.get_qparams()
    scale = qparams["scale"]
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("FP8 preparation produced invalid scales")
    # Match the dtype stored by CT's initialized parameters and later exporter.
    values = {
        f"weight_{name}": value.to(getattr(module, f"weight_{name}").dtype)
        for name, value in qparams.items()
        if value is not None and hasattr(module, f"weight_{name}")
    }
    values["weight"] = fake_quantize(
        module.weight,
        values["weight_scale"],
        values.get("weight_zero_point"),
        module.quantization_scheme.weights,
    )
    if not torch.isfinite(values["weight"]).all():
        raise ValueError("FP8 preparation produced non-finite weights")
    return values


@torch.no_grad()
def prepare_subgraph_weights(preparers, modules):
    """Publish rounded weights while this subgraph's offload cache is resident."""
    distributed = dist.is_available() and dist.is_initialized()
    for modifier in preparers:
        selected = modifier._weight_preparation_targets
        for module in sorted(set(modules) & set(selected), key=selected.get):
            if module in modifier._prepared_weight_modules:
                continue
            with compression_phase("fp8_weight_preparation", module=selected[module]):
                if distributed:
                    from llmcompressor.modifiers.gptq.distributed import (
                        gather_ep_records,
                        publish_gptq_result,
                    )

                    values, error = None, None
                    try:
                        if dist.get_rank() == 0:
                            values = _quantized_parameters(module)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    errors = gather_ep_records(error)
                    if any(errors):
                        raise ValueError(f"FP8 weight preparation failed: {errors}")
                    publish_gptq_result(module, 0, values)
                else:
                    for name, value in _quantized_parameters(module).items():
                        update_offload_parameter(module, name, value)
                modifier._prepared_weight_modules.add(module)
