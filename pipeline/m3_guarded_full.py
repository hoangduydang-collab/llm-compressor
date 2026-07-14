"""Guarded full-model MiniMax-M3 quantization diagnostics and runner."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

VARIANTS = ("offsetfix", "nosmooth", "quant_only")
DIAGNOSTIC_MODES = ("heavy", "light")
BOUNDARIES = ("layer_input", "moe_input", "moe_output", "layer_output")
DEFAULT_CONFIG = Path("pipeline/configs/minimax_m3_full_calib.yaml")
DEFAULT_SKETCH_VALUES = 4096
_LAYER_RE = re.compile(r"(?:^|[.])language_model[.]layers[.](\d+)(?:[.]|$)")


class GuardedRunAbort(RuntimeError):
    """Raised only after a failing layer record and abort report are durable."""


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _as_tensor(item)
            except TypeError:
                pass
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _as_tensor(item)
            except TypeError:
                pass
    raise TypeError(f"no tensor in {type(value).__name__}")


def _evenly_spaced_indices(numel: int, count: int, device: Any) -> torch.Tensor:
    """Evenly spaced [0, numel-1] index tensor, both endpoints included.

    Uses int64 arithmetic instead of ``torch.linspace``: linspace defaults to
    float32, which cannot represent integers above 2**24 exactly, so for large
    tensors (e.g. MiniMax-M3 per-expert weights of 18.9M-37.7M elements) the
    rounded endpoint overshoots to ``numel`` -- one past the end -- and the
    downstream ``index_select`` raises a CUDA device-side assert. This is why
    the diagnostic died at layer 3 (the first MoE layer) after the dense layers
    0-2 (whose smooth weights are <2**24) passed. Integer math + clamp is exact.
    """
    if count <= 1:
        return torch.zeros(max(count, 0), dtype=torch.long, device=device)
    steps = torch.arange(count, device=device, dtype=torch.long)
    # round(step * (numel-1) / (count-1)) with integer round-to-nearest.
    indices = (steps * (numel - 1) + (count - 1) // 2) // (count - 1)
    return indices.clamp_(0, numel - 1)


def deterministic_sketch(
    value: Any, *, max_values: int = DEFAULT_SKETCH_VALUES
) -> torch.Tensor:
    """Return a bounded deterministic flat FP32 sample including both endpoints."""
    tensor = _as_tensor(value).reshape(-1)
    if tensor.numel() == 0:
        raise ValueError("cannot sketch an empty tensor")
    count = min(int(max_values), tensor.numel())
    indices = _evenly_spaced_indices(tensor.numel(), count, tensor.device)
    return tensor.index_select(0, indices).float().cpu()


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def _tensor_shape_descriptor(value: Any) -> dict[str, Any] | None:
    """Shape/dtype/device of a tensor attribute, or None if it is absent."""
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _fake_quant_input_descriptor(module: Any, scheme: Any) -> dict[str, Any]:
    """Capture the operands and qparams a fake-quant kernel will index.

    A device-side assert in W4 grouped fake-quant is almost always an
    out-of-bounds group index, i.e. the weight's ``in_features`` is not
    consistent with ``scale`` columns for the configured ``group_size``. Recording
    these up front makes that mismatch legible from the crashed run's artifacts.
    """
    weight = getattr(module, "weight", None)
    scale = getattr(module, "weight_scale", None)
    zero_point = getattr(module, "weight_zero_point", None)
    descriptor: dict[str, Any] = {
        "weight": _tensor_shape_descriptor(weight),
        "weight_scale": _tensor_shape_descriptor(scale),
        "weight_zero_point": _tensor_shape_descriptor(zero_point),
        "scheme": {
            "num_bits": getattr(scheme, "num_bits", None),
            "type": str(getattr(scheme, "type", None)),
            "group_size": getattr(scheme, "group_size", None),
            "strategy": str(getattr(scheme, "strategy", None)),
            "symmetric": getattr(scheme, "symmetric", None),
        },
    }
    # Flag the classic grouped-quant inconsistency directly: for group_size G, a
    # weight of shape [out, in] expects scale columns == ceil(in / G).
    try:
        group_size = getattr(scheme, "group_size", None)
        if (
            isinstance(weight, torch.Tensor)
            and isinstance(scale, torch.Tensor)
            and group_size
            and group_size > 0
            and weight.dim() == 2
            and scale.dim() == 2
        ):
            in_features = weight.shape[1]
            expected_groups = -(-in_features // group_size)  # ceil div
            descriptor["expected_scale_groups"] = expected_groups
            descriptor["actual_scale_groups"] = scale.shape[1]
            descriptor["group_geometry_consistent"] = (
                expected_groups == scale.shape[1]
            )
    except Exception:
        pass
    return descriptor


def tensor_summary(value: Any) -> dict[str, float | int | None]:
    """Summarize a tensor without allowing non-finite values to poison statistics."""
    tensor = _as_tensor(value).reshape(-1).float().cpu()
    if tensor.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    result: dict[str, float | int | None] = {
        "count": tensor.numel(),
        "finite_fraction": _safe_float(finite.float().mean()),
        "zero_fraction": _safe_float((tensor == 0).float().mean()),
        "min": None,
        "max": None,
        "mean": None,
        "p01": None,
        "p50": None,
        "p99": None,
    }
    if finite_values.numel():
        quantiles = torch.quantile(finite_values, torch.tensor([0.01, 0.5, 0.99]))
        result.update(
            min=_safe_float(finite_values.min()),
            max=_safe_float(finite_values.max()),
            mean=_safe_float(finite_values.mean()),
            p01=_safe_float(quantiles[0]),
            p50=_safe_float(quantiles[1]),
            p99=_safe_float(quantiles[2]),
        )
    return result


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return 1.0 if numerator == 0 else None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def compare_sketches(reference: Any, candidate: Any) -> dict[str, float | None]:
    """Compute bounded activation/weight fidelity, including sign flips."""
    ref = _as_tensor(reference).reshape(-1).double().cpu()
    cand = _as_tensor(candidate).reshape(-1).double().cpu()
    if ref.shape != cand.shape or ref.numel() == 0:
        raise ValueError(f"sketch shape mismatch or empty: {ref.shape} != {cand.shape}")
    finite_fraction = _safe_float(torch.isfinite(cand).float().mean())
    result: dict[str, float | None] = {
        "finite_fraction": finite_fraction,
        "reference_l2": None,
        "candidate_l2": None,
        "norm_ratio": None,
        "cosine_similarity": None,
        "relative_rmse": None,
        "max_abs_error": None,
        "sign_flip_ratio": None,
    }
    if not bool(torch.isfinite(ref).all()) or finite_fraction != 1.0:
        return result
    ref_l2 = _safe_float(torch.linalg.vector_norm(ref))
    cand_l2 = _safe_float(torch.linalg.vector_norm(cand))
    diff = cand - ref
    ref_rms = _safe_float(torch.sqrt(torch.mean(ref.square())))
    diff_rms = _safe_float(torch.sqrt(torch.mean(diff.square())))
    denominator = ref_l2 * cand_l2
    cosine = 1.0 if denominator == 0 and ref_l2 == cand_l2 else None
    if denominator:
        cosine = max(-1.0, min(1.0, _safe_float(torch.dot(ref, cand)) / denominator))
    result.update(
        reference_l2=ref_l2,
        candidate_l2=cand_l2,
        norm_ratio=_ratio(cand_l2, ref_l2),
        cosine_similarity=cosine,
        relative_rmse=_ratio(diff_rms, ref_rms),
        max_abs_error=_safe_float(diff.abs().max()),
        sign_flip_ratio=_safe_float(((ref < 0) != (cand < 0)).float().mean()),
    )
    return result


def _violation(
    check: str, observed: Any, threshold: str, detail: str
) -> dict[str, Any]:
    return {
        "check": check,
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
    }


def evaluate_layer_record(record: dict[str, Any]) -> dict[str, Any]:
    """Apply variant-aware fail-fast checks while retaining every observed value."""
    variant = record["variant"]
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    layer = int(record["layer"])
    violations: list[dict[str, Any]] = []
    lifecycle = record.get("mapping_lifecycle", {})
    if layer >= 3 and variant != "quant_only":
        expected = int(lifecycle.get("expected_count", 0))
        resolved = int(lifecycle.get("resolved_count", 0))
        completed = int(lifecycle.get("completed_count", 0))
        unprocessed = int(lifecycle.get("unprocessed_count", 0))
        events = int(lifecycle.get("forward_event_count", 0))
        if expected <= 0 or resolved != expected:
            violations.append(
                _violation(
                    "resolved_mappings",
                    resolved,
                    f"== expected_count ({expected})",
                    "AWQ mapping resolution did not match the layer contract",
                )
            )
        if (
            completed <= 0
            or completed + int(lifecycle.get("skipped_count", 0)) != resolved
        ):
            violations.append(
                _violation(
                    "completed_mappings",
                    completed,
                    "nonzero and all resolved accounted",
                    f"unprocessed mappings={unprocessed}",
                )
            )
        if events <= 0:
            violations.append(
                _violation(
                    "mapping_forward_events",
                    events,
                    "> 0",
                    "AWQ activation targets never executed; no statistics can be valid",
                )
            )

    boundaries = record.get("boundaries", {})
    for boundary in BOUNDARIES:
        if boundary not in boundaries:
            violations.append(
                _violation(
                    f"{boundary}.capture",
                    None,
                    "reference and candidate present",
                    "the sequential calibration/propagation boundary hook did not fire",
                )
            )
    for boundary, metrics in boundaries.items():
        gates = (
            ("finite_fraction", 1.0, lambda value: value == 1.0, "== 1.0"),
            ("norm_ratio", None, lambda value: 0.1 <= value <= 10.0, "in [0.1, 10]"),
            ("cosine_similarity", None, lambda value: value >= 0.90, ">= 0.90"),
            ("relative_rmse", None, lambda value: value <= 0.50, "<= 0.50"),
        )
        for key, _expected, predicate, threshold in gates:
            value = metrics.get(key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not predicate(value)
            ):
                violations.append(
                    _violation(
                        f"{boundary}.{key}",
                        value,
                        threshold,
                        "candidate/reference activation fidelity failed",
                    )
                )

    if layer >= 3 and record.get("diagnostic_mode", "heavy") == "heavy":
        quant = record.get("quantization", {})
        sampled = int(quant.get("sampled_module_count", 0))
        if sampled != 9:
            violations.append(
                _violation(
                    "quantization.sampled_module_count",
                    sampled,
                    "== 9",
                    "representative expert qparams/reconstruction "
                    "evidence is incomplete",
                )
            )
        for module in quant.get("sampled_modules", []):
            if "reconstruction_error" in module:
                violations.append(
                    _violation(
                        f"{module.get('module')}.reconstruction",
                        module["reconstruction_error"],
                        "successful",
                        "fake-quant reconstruction probe raised",
                    )
                )
            scale = module.get("weight_scale", {})
            if scale.get("finite_fraction") != 1.0 or scale.get("zero_fraction") != 0.0:
                violations.append(
                    _violation(
                        f"{module.get('module')}.weight_scale",
                        scale,
                        "finite_fraction=1 and zero_fraction=0",
                        "invalid quantization scale",
                    )
                )
        if variant != "quant_only":
            for metric in lifecycle.get("completed_metrics", []):
                for key in ("initial_error", "best_error", "reduction", "best_ratio"):
                    value = metric.get(key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        violations.append(
                            _violation(
                                f"{metric.get('layer_name')}.{key}",
                                value,
                                "finite",
                                "AWQ grid-search metric is unavailable or non-finite",
                            )
                        )
            scales = record.get("scale_diagnostics", [])
            if len(scales) != int(lifecycle.get("completed_count", 0)):
                violations.append(
                    _violation(
                        "scale_diagnostics.count",
                        len(scales),
                        f"== completed_count ({lifecycle.get('completed_count', 0)})",
                        "smoothing scale evidence is incomplete",
                    )
                )
            for scale_record in scales:
                summary = scale_record.get("scale", {})
                if (
                    summary.get("finite_fraction") != 1.0
                    or summary.get("zero_fraction") != 0.0
                ):
                    violations.append(
                        _violation(
                            f"{scale_record.get('layer_name')}.scale",
                            summary,
                            "finite_fraction=1 and zero_fraction=0",
                            "invalid AWQ smoothing scale",
                        )
                    )
                residual = scale_record.get("inverse_compensation_max_relative_error")
                if (
                    not isinstance(residual, (int, float))
                    or not math.isfinite(residual)
                    or residual > 1e-5
                ):
                    violations.append(
                        _violation(
                            f"{scale_record.get('layer_name')}.inverse_compensation",
                            residual,
                            "<= 1e-5",
                            "offset-norm inverse-scale compensation is unstable",
                        )
                    )

    return {"status": "abort" if violations else "pass", "violations": violations}


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class LayerEvidenceWriter:
    """Durably record each layer and heartbeat before enforcing its verdict."""

    def __init__(self, output_dir: Path, *, variant: str):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.output_dir = Path(output_dir)
        self.variant = variant
        self.started = time.monotonic()
        self.completed_layers: list[int] = []

    def persist_and_enforce(self, record: dict[str, Any]) -> dict[str, Any]:
        layer = int(record["layer"])
        record = dict(record)
        record.setdefault("schema_version", 1)
        record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        record.setdefault("node", platform.node())
        record.setdefault("elapsed_seconds", time.monotonic() - self.started)
        verdict = evaluate_layer_record(record)
        record["verdict"] = verdict
        _write_json_atomic(
            self.output_dir / "layers" / f"layer-{layer:02d}.json", record
        )
        if verdict["status"] == "pass":
            self.completed_layers.append(layer)
            _write_json_atomic(
                self.output_dir / "heartbeat.json",
                {
                    "status": "running",
                    "variant": self.variant,
                    "last_completed_layer": layer,
                    "completed_layers": self.completed_layers,
                },
            )
            return verdict

        abort = {
            "schema_version": 1,
            "status": "aborted",
            "variant": self.variant,
            "layer": layer,
            "exception_type": "GuardedRunAbort",
            "message": (
                f"layer {layer} failed {len(verdict['violations'])} guard checks"
            ),
            "violations": verdict["violations"],
            "completed_layers": self.completed_layers,
            "layer_record": f"layers/layer-{layer:02d}.json",
        }
        _write_json_atomic(self.output_dir / "abort.json", abort)
        _write_json_atomic(self.output_dir / "heartbeat.json", abort)
        raise GuardedRunAbort(abort["message"])


def prepare_variant_config(config: Any, variant: str) -> Any:
    """Clone production configuration and change only the tested transform."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if config.quantization.scheme != "W4AFP8":
        raise ValueError("guarded MiniMax matrix requires production W4AFP8")
    prepared = copy.deepcopy(config)
    prepared.quantization.method = "quant_only" if variant == "quant_only" else "awq"
    return prepared


def _layer_from_name(name: str | None) -> int | None:
    match = _LAYER_RE.search(name or "")
    return int(match.group(1)) if match else None


class FullGuardController:
    """Coordinate bounded hooks and durable evidence across recipe modifiers."""

    def __init__(
        self, output_dir: Path, *, variant: str, diagnostic_mode: str = "heavy"
    ):
        if diagnostic_mode not in DIAGNOSTIC_MODES:
            raise ValueError(f"unknown diagnostic mode {diagnostic_mode!r}")
        self.output_dir = Path(output_dir)
        self.variant = variant
        self.diagnostic_mode = diagnostic_mode
        self.writer = LayerEvidenceWriter(self.output_dir, variant=variant)
        self.diagnostic_stages: list[dict[str, Any]] = []
        self.model = None
        self.module_names: dict[Any, str] = {}
        self.layers: dict[int, Any] = {}
        self.phase = {layer: "reference" for layer in range(60)}
        self.captures = {
            layer: {"reference": {}, "candidate": {}} for layer in range(60)
        }
        self.handles: list[Any] = []
        self.mapping_handles: list[Any] = []
        self.resolved_by_layer: dict[int, list[str]] = {}
        self.fire_counts: dict[str, int] = {}
        self.grid_by_layer: dict[int, list[dict[str, Any]]] = {}
        self.skipped_by_layer: dict[int, list[dict[str, Any]]] = {}
        self.scale_by_layer: dict[int, list[dict[str, Any]]] = {}
        self.quant_by_layer: dict[int, dict[str, Any]] = {}
        self.installed = False

    def install(self, model: Any, state: Any) -> None:
        if self.installed:
            return
        self.installed = True
        self.model = model
        self.module_names = {module: name for name, module in model.named_modules()}
        for name, module in model.named_modules():
            layer = _layer_from_name(name)
            if layer is not None and name.endswith(f"language_model.layers.{layer}"):
                self.layers[layer] = module
        if sorted(self.layers) != list(range(60)):
            raise RuntimeError(
                f"expected MiniMax decoder layers 0..59, got {sorted(self.layers)}"
            )
        state.sequential_trace_callback = self.validate_production_trace
        state.post_sequential_propagation_callback = self.after_propagation

    def start_calibration_hooks(self) -> None:
        """Install native hooks between FX tracing and calibration forward."""
        if self.handles:
            return
        for layer, decoder in self.layers.items():

            def layer_pre(_module, args, kwargs, _layer=layer):
                value = args[0] if args else kwargs.get("hidden_states")
                self._capture(_layer, "layer_input", value)

            self.handles.extend(
                [
                    decoder.register_forward_pre_hook(layer_pre, with_kwargs=True),
                    decoder.post_attention_layernorm.register_forward_hook(
                        lambda _module, _args, output, _layer=layer: self._capture(
                            _layer, "moe_input", output
                        )
                    ),
                    decoder.mlp.register_forward_hook(
                        lambda _module, _args, output, _layer=layer: self._capture(
                            _layer, "moe_output", output
                        )
                    ),
                    decoder.register_forward_hook(
                        lambda _module, _args, output, _layer=layer: self._capture(
                            _layer, "layer_output", output
                        )
                    ),
                ]
            )

    def _capture(self, layer: int, boundary: str, value: Any) -> None:
        phase = self.phase[layer]
        target = self.captures[layer][phase]
        if boundary not in target:
            target[boundary] = deterministic_sketch(value)

    def validate_production_trace(self, diagnostics: dict[str, Any]) -> None:
        from pipeline.m3_trace_diagnostic import persist_root_artifacts

        persist_root_artifacts(
            self.output_dir / "production_trace", {"status": "ok", **diagnostics}
        )
        matched = int(diagnostics.get("matched_target_count", 0))
        targets = int(diagnostics.get("target_node_count", 0))
        partitions = int(diagnostics.get("partition_count", 0))
        if matched != 60 or targets != matched or partitions <= 1:
            abort = {
                "status": "aborted",
                "variant": self.variant,
                "stage": "production_trace",
                "matched_target_count": matched,
                "target_node_count": targets,
                "partition_count": partitions,
                "message": (
                    "production oneshot trace lost decoder targets or partitions"
                ),
            }
            _write_json_atomic(self.output_dir / "abort.json", abort)
            _write_json_atomic(self.output_dir / "heartbeat.json", abort)
            raise GuardedRunAbort(abort["message"])

    def note_awq_start(self, mappings: list[Any]) -> None:
        for mapping in mappings:
            layer = _layer_from_name(mapping.smooth_name)
            if layer is None:
                raise RuntimeError(
                    f"resolved AWQ mapping has no decoder layer: {mapping.smooth_name}"
                )
            self.resolved_by_layer.setdefault(layer, []).append(mapping.smooth_name)
            target = mapping.activation_hook_target or mapping.balance_layers[0]
            self.fire_counts.setdefault(mapping.smooth_name, 0)

            def counter(_module, _args, _output, name=mapping.smooth_name):
                self.fire_counts[name] += 1

            self.mapping_handles.append(target.register_forward_hook(counter))
        expected = 128 if self.variant == "nosmooth" else 129
        invalid = {
            layer: len(self.resolved_by_layer.get(layer, []))
            for layer in range(3, 60)
            if len(self.resolved_by_layer.get(layer, [])) != expected
        }
        if invalid:
            raise RuntimeError(f"AWQ resolved mapping contract mismatch: {invalid}")

    def note_scale(self, mapping: Any, scales: torch.Tensor) -> None:
        layer = _layer_from_name(mapping.smooth_name)
        if layer is None:
            return
        weight = mapping.smooth_layer.weight.detach().float()
        scale = scales.detach().float().to(weight.device)
        flat = weight.reshape(-1)
        count = min(DEFAULT_SKETCH_VALUES, flat.numel())
        indices = _evenly_spaced_indices(flat.numel(), count, flat.device)
        before = flat.index_select(0, indices)
        if weight.ndim == 1:
            applied_scale = scale.index_select(0, indices)
        else:
            rows = torch.div(indices, weight.shape[1], rounding_mode="floor")
            first_scaled_row = weight.shape[0] - scale.numel()
            applied_scale = torch.ones_like(before)
            selected = rows >= first_scaled_row
            applied_scale[selected] = scale.index_select(
                0, rows[selected] - first_scaled_row
            )
        predicted = before / applied_scale
        restored = predicted * applied_scale
        denominator = before.abs().clamp_min(torch.finfo(torch.float32).tiny)
        compensation_error = _safe_float(
            ((restored - before).abs() / denominator).max()
        )
        self.scale_by_layer.setdefault(layer, []).append(
            {
                "layer_name": mapping.smooth_name,
                "scale": tensor_summary(scale),
                "effective_norm_before_sketch": tensor_summary(before),
                "predicted_effective_norm_after_sketch": tensor_summary(predicted),
                "sketch_value_count": count,
                "inverse_compensation_max_relative_error": compensation_error,
            }
        )

    def note_awq_epoch(
        self,
        modules: list[Any],
        completed: list[dict[str, Any]],
        skipped: list[dict[str, Any]],
    ) -> None:
        for metric in completed:
            layer = _layer_from_name(metric.get("layer_name"))
            if layer is not None:
                self.grid_by_layer.setdefault(layer, []).append(dict(metric))
        for metric in skipped:
            layer = _layer_from_name(metric.get("layer_name"))
            if layer is not None:
                self.skipped_by_layer.setdefault(layer, []).append(dict(metric))
        self.begin_candidate(modules)

    def _layers_for_modules(self, modules: list[Any]) -> list[int]:
        return sorted(
            {
                layer
                for module in modules
                if (layer := _layer_from_name(self.module_names.get(module)))
                is not None
            }
        )

    def begin_candidate(self, modules: list[Any]) -> None:
        for layer in self._layers_for_modules(modules):
            self.phase[layer] = "candidate"

    def _synchronize_stage(self, stage: str, modules: list[Any]) -> None:
        record = {
            "schema_version": 1,
            "stage": stage,
            "status": "synchronizing"
            if torch.cuda.is_available()
            else "skipped_no_cuda",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "node": platform.node(),
            "decoder_layers": self._layers_for_modules(modules),
        }
        self.diagnostic_stages.append(record)
        _write_json_atomic(
            self.output_dir / "diagnostic_stages.json", self.diagnostic_stages
        )
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            record["status"] = "error"
            record["exception_type"] = type(exc).__name__
            record["message"] = str(exc)
            _write_json_atomic(
                self.output_dir / "diagnostic_stages.json", self.diagnostic_stages
            )
            raise
        record["status"] = "complete"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(
            self.output_dir / "diagnostic_stages.json", self.diagnostic_stages
        )

    def _inspect_quant_params(self, modules: list[Any]) -> list[dict[str, Any]]:
        from compressed_tensors.quantization.utils import is_module_quantized

        selected_suffixes = tuple(
            f"mlp.experts.{expert}.{projection}"
            for expert in (0, 64, 127)
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        selected: list[dict[str, Any]] = []
        for module in modules:
            name = self.module_names.get(module, "")
            layer = _layer_from_name(name)
            if (
                layer is None
                or not name.endswith(selected_suffixes)
                or not is_module_quantized(module)
            ):
                continue
            scheme = getattr(
                getattr(module, "quantization_scheme", None), "weights", None
            )
            if scheme is None or not hasattr(module, "weight_scale"):
                continue
            item: dict[str, Any] = {
                "module": name,
                "weight_scale": tensor_summary(module.weight_scale),
            }
            zero_point = getattr(module, "weight_zero_point", None)
            if zero_point is not None:
                item["weight_zero_point"] = tensor_summary(zero_point)
            selected.append(
                {
                    "module_object": module,
                    "layer": layer,
                    "scheme": scheme,
                    "evidence": item,
                }
            )
        return selected

    def _collect_fake_quant_evidence(self, selected: list[dict[str, Any]]) -> None:
        from compressed_tensors.quantization import forward_quantize

        per_layer: dict[int, list[dict[str, Any]]] = {}
        for sample in selected:
            module = sample["module_object"]
            item = sample["evidence"]
            # Record the exact fake-quant inputs BEFORE launching the kernel and
            # persist immediately. A device-side assert inside forward_quantize
            # surfaces asynchronously (at the next synchronize) and is sticky --
            # it poisons the CUDA context, so nothing after it can be trusted or
            # written. Capturing the operand/qparam shapes up front, plus a
            # per-module synchronize below, attributes the assert to one expert
            # projection and leaves a durable breadcrumb (weight vs scale group
            # geometry is the usual out-of-bounds-index culprit for W4 grouped
            # fake-quant).
            item["fake_quant_inputs"] = _fake_quant_input_descriptor(
                module, sample["scheme"]
            )
            per_layer.setdefault(sample["layer"], []).append(item)
            self._persist_fake_quant_probe(per_layer)
            try:
                dequantized = forward_quantize(
                    module, module.weight, "weight", sample["scheme"]
                )
                if torch.cuda.is_available():
                    # Force any async device-side assert to raise HERE, against
                    # this module, instead of at the later stage barrier.
                    torch.cuda.synchronize()
                reference = deterministic_sketch(module.weight)
                candidate = deterministic_sketch(dequantized)
                item["reconstruction"] = compare_sketches(reference, candidate)
                finite = candidate[torch.isfinite(candidate)]
                item["dequantized_unique_count"] = int(torch.unique(finite).numel())
                if finite.numel():
                    item["endpoint_fraction"] = _safe_float(
                        ((finite == finite.min()) | (finite == finite.max()))
                        .float()
                        .mean()
                    )
            except Exception as exc:
                item["reconstruction_error"] = f"{type(exc).__name__}: {exc}"
                self._persist_fake_quant_probe(per_layer)
                raise
            self._persist_fake_quant_probe(per_layer)
        for layer, values in per_layer.items():
            self.quant_by_layer[layer] = {
                "sampled_module_count": len(values),
                "sampled_modules": values,
            }

    def _persist_fake_quant_probe(
        self, per_layer: dict[int, list[dict[str, Any]]]
    ) -> None:
        """Durably snapshot fake-quant probe progress so a sticky CUDA assert.

        (which kills the process before ``after_propagation`` persists the layer
        record) still leaves the failing module and its qparam geometry on disk.
        """
        snapshot = {
            str(layer): {
                "sampled_module_count": len(values),
                "sampled_modules": values,
            }
            for layer, values in per_layer.items()
        }
        try:
            _write_json_atomic(
                self.output_dir / "fake_quant_probe.json", snapshot
            )
        except Exception:
            pass

    @torch.no_grad()
    def note_quant_epoch(self, modules: list[Any]) -> None:
        from compressed_tensors.quantization import enable_quantization
        from compressed_tensors.quantization.utils import is_module_quantized

        self._synchronize_stage("post_native_quantization", modules)
        for module in modules:
            if is_module_quantized(module):
                module.apply(enable_quantization)
        self._synchronize_stage("post_enable_quantization", modules)

        if self.diagnostic_mode == "heavy":
            self._synchronize_stage("before_qparam_inspection", modules)
            selected = self._inspect_quant_params(modules)
            self._synchronize_stage("after_qparam_inspection", modules)
            self._synchronize_stage("before_fake_quantization", modules)
            self._collect_fake_quant_evidence(selected)
            self._synchronize_stage("after_fake_quantization", modules)

        self.begin_candidate(modules)

    def after_propagation(
        self,
        *,
        subgraph_index: int,
        num_subgraphs: int,
        modules: list[Any],
        propagated: bool,
    ) -> None:
        layers = self._layers_for_modules(modules)
        if not layers:
            return
        if len(layers) != 1:
            abort = {
                "status": "aborted",
                "variant": self.variant,
                "stage": "post_propagation",
                "subgraph_index": subgraph_index,
                "decoder_layers": layers,
                "message": "sequential subgraph contains multiple decoder layers",
            }
            _write_json_atomic(self.output_dir / "abort.json", abort)
            _write_json_atomic(self.output_dir / "heartbeat.json", abort)
            raise GuardedRunAbort(abort["message"])
        layer = layers[0]
        captures = self.captures[layer]
        boundaries = {
            boundary: compare_sketches(
                captures["reference"][boundary], captures["candidate"][boundary]
            )
            for boundary in BOUNDARIES
            if boundary in captures["reference"] and boundary in captures["candidate"]
        }
        resolved = self.resolved_by_layer.get(layer, [])
        completed = self.grid_by_layer.get(layer, [])
        skipped = self.skipped_by_layer.get(layer, [])
        accounted = {item.get("layer_name") for item in [*completed, *skipped]}
        expected = 0
        if layer >= 3 and self.variant != "quant_only":
            expected = 128 if self.variant == "nosmooth" else 129
        record = {
            "layer": layer,
            "variant": self.variant,
            "diagnostic_mode": self.diagnostic_mode,
            "subgraph_index": subgraph_index,
            "num_subgraphs": num_subgraphs,
            "propagated": propagated,
            "mapping_lifecycle": {
                "expected_count": expected,
                "resolved_count": len(resolved),
                "completed_count": len(completed),
                "skipped_count": len(skipped),
                "unprocessed_count": len(
                    [name for name in resolved if name not in accounted]
                ),
                "forward_event_count": sum(
                    self.fire_counts.get(name, 0) for name in resolved
                ),
                "completed_metrics": completed,
                "skipped_metrics": skipped,
            },
            "scale_diagnostics": self.scale_by_layer.get(layer, []),
            "quantization": self.quant_by_layer.get(layer, {}),
            "boundaries": boundaries,
        }
        self.writer.persist_and_enforce(record)

    def close(self) -> None:
        for handle in [*self.handles, *self.mapping_handles]:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles.clear()
        self.mapping_handles.clear()


def build_guarded_recipe(
    config: Any, variant: str, output_dir: Path, diagnostic_mode: str = "heavy"
):
    """Build the production quantization recipe with full-run guards."""
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from pipeline.minimax_m3_config import get_minimax_m3_awq_mappings

    controller = FullGuardController(
        output_dir, variant=variant, diagnostic_mode=diagnostic_mode
    )

    class GuardedQuantizationModifier(QuantizationModifier):
        guard: Any = None

        def on_initialize(self, state, **kwargs):
            result = super().on_initialize(state, **kwargs)
            self.guard.install(state.model, state)
            return result

        def on_calibration_start(self, state, event, **kwargs):
            result = super().on_calibration_start(state, event, **kwargs)
            self.guard.start_calibration_hooks()
            return result

        def on_sequential_epoch_end(self, state, event, modules, **kwargs):
            result = super().on_sequential_epoch_end(
                state, event, modules=modules, **kwargs
            )
            self.guard.note_quant_epoch(modules)
            return result

    quant = GuardedQuantizationModifier(
        targets=["Linear"],
        scheme=config.quantization.scheme,
        ignore=list(config.quantization.ignore),
        guard=controller,
    )
    if variant == "quant_only":
        return [quant], controller

    class GuardedAWQModifier(AWQModifier):
        guard: Any = None

        def on_initialize(self, state, **kwargs):
            result = super().on_initialize(state, **kwargs)
            self.guard.install(state.model, state)
            return result

        def on_calibration_start(self, state, event, **kwargs):
            result = super().on_calibration_start(state, event, **kwargs)
            self.guard.start_calibration_hooks()
            self.guard.note_awq_start(self._resolved_mappings)
            return result

        def _compute_best_scale(self, mapping, fp16_outputs, orig_layer_weights):
            scales = super()._compute_best_scale(
                mapping, fp16_outputs, orig_layer_weights
            )
            self.guard.note_scale(mapping, scales)
            return scales

        def on_sequential_epoch_end(self, state, event, modules, **kwargs):
            completed_before = len(self._error_metrics)
            skipped_before = len(self._skipped_error_metrics)
            result = super().on_sequential_epoch_end(
                state, event, modules=modules, **kwargs
            )
            self.guard.note_awq_epoch(
                modules,
                self._error_metrics[completed_before:],
                self._skipped_error_metrics[skipped_before:],
            )
            return result

    awq = GuardedAWQModifier(
        mappings=get_minimax_m3_awq_mappings(
            disable_mlp_input_smoothing=variant == "nosmooth"
        ),
        duo_scaling=config.quantization.awq_duo_scaling,
        guard=controller,
    )
    return [awq, quant], controller


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def run_guarded_full(
    *,
    variant: str,
    config_path: Path,
    output_dir: Path,
    model_id: str | None = None,
    diagnostic_mode: str = "heavy",
) -> dict[str, Any]:
    """Run one guarded full arm and save only after all layer guards pass."""
    import hashlib
    import os

    from llmcompressor import oneshot
    from pipeline import metrics
    from pipeline.calibration import build_calibration_dataset
    from pipeline.config import load_config
    from pipeline.minimax_m3_config import (
        ensure_minimax_m3_vllm_serve_config,
        patch_minimax_m3_for_text_calibration,
    )
    from pipeline.provenance import log_model_provenance
    from pipeline.quantize import _load_model_and_tokenizer, _persist_ignore_to_config
    from pipeline.vl_artifacts import ensure_vl_processor_artifacts

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = prepare_variant_config(load_config(config_path), variant)
    if model_id is not None:
        config.model.id = model_id
    random.seed(config.calibration.seed)
    torch.manual_seed(config.calibration.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.calibration.seed)
    _write_json_atomic(
        output_dir / "start.json",
        {
            "schema_version": 1,
            "status": "started",
            "variant": variant,
            "diagnostic_mode": diagnostic_mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            "command": [sys.executable, *sys.argv],
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "model_id": config.model.id,
            "node": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        },
    )

    controller = None
    try:
        model, tokenizer = _load_model_and_tokenizer(config)
        if not patch_minimax_m3_for_text_calibration(model):
            raise RuntimeError("loaded model is not recognized as MiniMax-M3")
        log_model_provenance(
            model,
            config.calibration.sequential_targets,
            out_path=output_dir / "model_provenance.json",
        )
        dataset = build_calibration_dataset(config.calibration, tokenizer)
        recipe, controller = build_guarded_recipe(
            config, variant, output_dir, diagnostic_mode=diagnostic_mode
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "processor": tokenizer,
            "trust_remote_code_model": config.model.trust_remote_code,
            "dataset": dataset,
            "recipe": recipe,
            "max_seq_length": config.calibration.max_seq_length,
            "num_calibration_samples": config.calibration.num_samples,
            "moe_calibrate_all_experts": config.calibration.moe_calibrate_all_experts,
            "output_dir": None,
            "save_compressed": False,
            "shuffle_calibration_samples": False,
        }
        if config.calibration.sequential_targets:
            kwargs["sequential_targets"] = config.calibration.sequential_targets
        # Force the sequential pipeline. oneshot's CalibrationPipeline._infer_pipeline
        # decides "sequential" vs "datafree" by matching modifier CLASS NAMES against
        # a hardcoded list ("AWQModifier", ...) plus isinstance(QuantizationModifier).
        # This harness wraps the modifiers in renamed subclasses (GuardedAWQModifier /
        # GuardedQuantizationModifier); AWQModifier is not a QuantizationModifier and
        # the guarded names match nothing, and W4AFP8 QuantizationModifier does not
        # require calibration data -- so inference silently returns "datafree". The
        # datafree pipeline calibrates the whole model in one epoch
        # (sequential_epoch_end(list(model.modules()))), never partitions per layer,
        # and never invokes the propagation callback -> after_propagation never fires
        # -> completed_layers == [] and the completeness gate aborts. Pinning the
        # pipeline makes the guards instrument the sequential per-layer AWQ path they
        # were built for, regardless of the guard subclass names.
        kwargs["pipeline"] = config.calibration.pipeline or "sequential"
        with metrics.capture_quant_metrics(output_dir / "quant_metrics.jsonl"):
            oneshot(**kwargs)

        completed = sorted(controller.writer.completed_layers)
        if completed != list(range(60)):
            raise GuardedRunAbort(
                f"oneshot returned without all decoder layer guards: {completed}"
            )
        checkpoint = output_dir / "checkpoint"
        if checkpoint.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint {checkpoint}")
        model.save_pretrained(
            str(checkpoint), save_compressed=True, quantization_format="pack-quantized"
        )
        tokenizer.save_pretrained(str(checkpoint))
        _persist_ignore_to_config(checkpoint, config.quantization.ignore)
        ensure_vl_processor_artifacts(
            checkpoint,
            config.model.id,
            trust_remote_code=config.model.trust_remote_code,
        )
        ensure_minimax_m3_vllm_serve_config(checkpoint, config.model.id)
        from contextlib import redirect_stderr, redirect_stdout

        from pipeline.verify_quant_checkpoint import main as verify_checkpoint_main

        static_log = output_dir / "static_checkpoint_verification.log"
        with static_log.open("w") as stream, redirect_stdout(stream), redirect_stderr(
            stream
        ):
            static_rc = verify_checkpoint_main(
                ["--ckpt", str(checkpoint), "--check-tensors"]
            )
        _write_json_atomic(
            output_dir / "static_checkpoint_verification.json",
            {
                "return_code": static_rc,
                "checkpoint": str(checkpoint),
                "log": str(static_log),
            },
        )
        if static_rc != 0:
            raise RuntimeError(
                "post-quantization static checkpoint verification failed "
                f"rc={static_rc}"
            )
        result = {
            "schema_version": 1,
            "status": "complete",
            "variant": variant,
            "diagnostic_mode": diagnostic_mode,
            "completed_layers": completed,
            "checkpoint": str(checkpoint),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(output_dir / "result.json", result)
        _write_json_atomic(output_dir / "heartbeat.json", result)
        return result
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "aborted" if isinstance(exc, GuardedRunAbort) else "error",
            "variant": variant,
            "diagnostic_mode": diagnostic_mode,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "completed_layers": list(controller.writer.completed_layers)
            if controller
            else [],
        }
        _write_json_atomic(output_dir / "failure.json", failure)
        if not (output_dir / "abort.json").exists():
            _write_json_atomic(output_dir / "abort.json", failure)
        _write_json_atomic(output_dir / "heartbeat.json", failure)
        raise
    finally:
        if controller is not None:
            controller.close()


def aggregate_runs(result_root: Path) -> dict[str, Any]:
    """Aggregate arms while retaining quality aborts as diagnostic outcomes."""
    arms = {}
    counts = {"complete": 0, "aborted": 0, "error": 0, "missing": 0}
    for variant in VARIANTS:
        arm_dir = Path(result_root) / variant
        result_path = arm_dir / "result.json"
        failure_path = arm_dir / "failure.json"
        rc_path = arm_dir / "rc"
        return_code = None
        if rc_path.exists():
            try:
                return_code = int(rc_path.read_text().strip())
            except ValueError:
                pass
        if result_path.exists():
            evidence = json.loads(result_path.read_text())
            status = "complete" if evidence.get("status") == "complete" else "error"
        elif failure_path.exists():
            evidence = json.loads(failure_path.read_text())
            status = evidence.get("status", "error")
            if status not in ("aborted", "error"):
                status = "error"
        else:
            evidence = None
            status = "missing"
        counts[status] += 1
        arms[variant] = {
            "status": status,
            "return_code": return_code,
            "evidence": evidence,
        }
    report = {
        "schema_version": 1,
        "status": "complete"
        if counts["missing"] == counts["error"] == 0
        else "incomplete",
        "counts": counts,
        "arms": arms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(Path(result_root) / "matrix.json", report)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    arm = subparsers.add_parser("arm", help="run one full guarded variant")
    arm.add_argument("--variant", choices=VARIANTS, required=True)
    arm.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arm.add_argument("--output-dir", type=Path, required=True)
    arm.add_argument("--model-id")
    arm.add_argument("--diagnostic-mode", choices=DIAGNOSTIC_MODES, default="heavy")
    aggregate = subparsers.add_parser("aggregate", help="aggregate three arm roots")
    aggregate.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "aggregate":
        aggregate_runs(args.result_root)
        return 0
    run_guarded_full(
        variant=args.variant,
        config_path=args.config,
        output_dir=args.output_dir,
        model_id=args.model_id,
        diagnostic_mode=args.diagnostic_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
