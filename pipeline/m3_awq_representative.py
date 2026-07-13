"""CPU-testable core for the MiniMax-M3 representative-layer AWQ diagnostic."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LAYERS = (8, 31, 59)
VARIANTS = ("offsetfix", "nosmooth")
BOUNDARIES = ("layer_input", "moe_input", "moe_output", "layer_output")
DEFAULT_CONFIG = Path("pipeline/configs/minimax_m3_full_calib.yaml")
PROBE_COUNT = 2
PROBE_MAX_TOKENS = 256


def _hook_trace_enabled() -> bool:
    """Whether to capture balance-layer forward-fire diagnostics (default on).

    Set ``M3_AWQ_HOOK_TRACE=0`` to disable. The diagnostics are additive and never
    change an arm's verdict; they exist to localize the ``completed=0`` failure.
    """
    value = os.environ.get("M3_AWQ_HOOK_TRACE", "1").lower()
    return value not in {"0", "false", "no"}


def _validate_layer(layer: int) -> None:
    if layer not in LAYERS:
        raise ValueError(f"layer {layer} is not a planned representative layer")


def layer_exclusion_pattern(layer: int) -> str:
    """Match language-model decoder layers other than ``layer``."""
    _validate_layer(layer)
    return (
        rf"re:.*language_model[.]layers[.]"
        rf"(?!{layer}(?:[.]|$))[0-9]+(?:[.]|$).*"
    )


def sequential_target_pattern(layer: int) -> str:
    """Match exactly one language-model decoder-layer module."""
    _validate_layer(layer)
    return rf"re:.*language_model[.]layers[.]{layer}$"


def prepare_arm_config(config: Any, layer: int) -> Any:
    """Clone production config and isolate quantization to one decoder layer."""
    _validate_layer(layer)
    if config.quantization.method != "awq" or config.quantization.scheme != "W4AFP8":
        raise ValueError("representative diagnostic requires production AWQ W4AFP8")
    prepared = copy.deepcopy(config)
    prepared.quantization.ignore.append(layer_exclusion_pattern(layer))
    prepared.calibration.sequential_targets = [sequential_target_pattern(layer)]
    return prepared


def unwrap_tensor(value: Any) -> Any:
    """Return the first tensor-like value from nested model outputs."""
    if hasattr(value, "detach") or isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        for item in value.values():
            try:
                return unwrap_tensor(item)
            except TypeError:
                pass
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return unwrap_tensor(item)
            except TypeError:
                pass
    raise TypeError(f"no tensor-like value in {type(value).__name__}")


def resolved_mapping_snapshot(
    mappings: list[Any], *, layer: int, variant: str
) -> list[dict[str, Any]]:
    """Validate and serialize resolved AWQ mappings before calibration runs."""
    _validate_layer(layer)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    marker = f".language_model.layers.{layer}."
    snapshot = []
    for mapping in mappings:
        names = [mapping.smooth_name, *mapping.balance_names]
        outside = [name for name in names if marker not in f".{name}"]
        if outside:
            raise RuntimeError(f"AWQ mapping outside selected layer {layer}: {outside}")
        snapshot.append(
            {
                "smooth_name": mapping.smooth_name,
                "balance_names": list(mapping.balance_names),
            }
        )
    has_mlp_input = any(
        item["smooth_name"].endswith("post_attention_layernorm") for item in snapshot
    )
    if variant == "nosmooth" and has_mlp_input:
        raise RuntimeError("nosmooth resolved an MLP-input smoothing mapping")
    if variant == "offsetfix" and not has_mlp_input:
        raise RuntimeError("offsetfix did not resolve its MLP-input smoothing mapping")
    if not snapshot:
        raise RuntimeError("no AWQ mappings resolved for selected layer")
    return snapshot


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def tensor_fidelity(reference: Any, candidate: Any) -> dict[str, float | None]:
    """Calculate JSON-safe fidelity statistics for two equal-shaped tensors."""
    ref = _as_numpy(reference)
    cand = _as_numpy(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"tensor shape mismatch: {ref.shape} != {cand.shape}")
    if ref.size == 0:
        raise ValueError("cannot compare empty tensors")

    ref = ref.reshape(-1)
    cand = cand.reshape(-1)
    finite_fraction = float(np.isfinite(cand).mean())
    reference_finite = bool(np.isfinite(ref).all())
    ref_l2 = float(np.linalg.norm(ref)) if reference_finite else None
    metrics: dict[str, float | None] = {
        "finite_fraction": finite_fraction,
        "reference_l2": ref_l2,
        "candidate_l2": None,
        "reference_max_abs": (
            float(np.max(np.abs(ref))) if reference_finite else None
        ),
        "candidate_max_abs": None,
        "norm_ratio": None,
        "cosine_similarity": None,
        "relative_rmse": None,
        "max_abs_error": None,
    }
    if not reference_finite or finite_fraction != 1.0:
        return metrics

    assert ref_l2 is not None
    cand_l2 = float(np.linalg.norm(cand))
    diff = cand - ref
    ref_rms = float(np.sqrt(np.mean(np.square(ref))))
    diff_rms = float(np.sqrt(np.mean(np.square(diff))))
    metrics.update(
        candidate_l2=cand_l2,
        candidate_max_abs=float(np.max(np.abs(cand))),
        norm_ratio=_safe_ratio(cand_l2, ref_l2),
        cosine_similarity=_cosine(ref, cand, ref_l2, cand_l2),
        relative_rmse=(
            0.0 if ref_rms == 0.0 and diff_rms == 0.0 else _safe_ratio(diff_rms, ref_rms)
        ),
        max_abs_error=float(np.max(np.abs(diff))),
    )
    return metrics


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else None
    result = numerator / denominator
    return float(result) if math.isfinite(result) else None


def _cosine(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_l2: float,
    candidate_l2: float,
) -> float | None:
    denominator = reference_l2 * candidate_l2
    if denominator == 0.0:
        return 1.0 if reference_l2 == candidate_l2 else None
    result = float(np.dot(reference, candidate) / denominator)
    return max(-1.0, min(1.0, result)) if math.isfinite(result) else None


def classify_boundaries(boundaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply broad catastrophic-corruption gates to aggregate boundaries."""
    unknown = set(boundaries).difference(BOUNDARIES)
    missing = set(BOUNDARIES).difference(boundaries)
    if unknown or missing:
        raise ValueError(
            f"boundary mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    classified: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for name in BOUNDARIES:
        reasons = _quality_failures(boundaries[name])
        classified[name] = {"passed": not reasons, "failures": reasons}
        failures.extend(f"{name}: {reason}" for reason in reasons)
    return {
        "verdict": "pass" if not failures else "quality_failure",
        "boundaries": classified,
        "failures": failures,
    }


def _quality_failures(metrics: dict[str, Any]) -> list[str]:
    gates = (
        ("finite_fraction", lambda value: value == 1.0, "must equal 1.0"),
        ("norm_ratio", lambda value: 0.1 <= value <= 10.0, "outside [0.1, 10.0]"),
        ("cosine_similarity", lambda value: value >= 0.90, "below 0.90"),
        ("relative_rmse", lambda value: value <= 0.50, "above 0.50"),
    )
    failures = []
    for key, predicate, description in gates:
        value = metrics.get(key)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{key} is unavailable or non-finite")
        elif not predicate(value):
            failures.append(f"{key} {description} (got {value})")
    return failures


def aggregate_matrix(root: Path) -> dict[str, Any]:
    """Aggregate all expected arm artifacts, preserving absent/crashed arms."""
    arms: dict[str, dict[str, Any]] = {}
    summary = {
        "pass": 0,
        "quality_failure": 0,
        "infrastructure_failure": 0,
        "missing": 0,
    }
    for variant in VARIANTS:
        for layer in LAYERS:
            name = f"{variant}-layer{layer}"
            arm_dir = root / name
            evidence_path = arm_dir / "arm.json"
            rc_path = arm_dir / "rc"
            return_code = None
            rc_valid = False
            if rc_path.is_file():
                try:
                    return_code = int(rc_path.read_text().strip())
                    rc_valid = True
                except (OSError, ValueError):
                    pass
            if rc_path.is_file() and (not rc_valid or return_code != 0):
                entry = {
                    "status": "infrastructure_failure",
                    "return_code": return_code,
                }
            elif evidence_path.is_file():
                try:
                    evidence = json.loads(evidence_path.read_text())
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    entry = {
                        "status": "infrastructure_failure",
                        "error": f"invalid arm.json: {error}",
                    }
                else:
                    identity_ok = (
                        isinstance(evidence, dict)
                        and evidence.get("layer") == layer
                        and evidence.get("variant") == variant
                    )
                    status = evidence.get("verdict") if isinstance(evidence, dict) else None
                    if not identity_ok:
                        entry = {
                            "status": "infrastructure_failure",
                            "error": "arm evidence identity/schema does not match directory",
                        }
                    elif status not in ("pass", "quality_failure"):
                        entry = {
                            "status": "infrastructure_failure",
                            "error": f"invalid verdict: {status!r}",
                        }
                    else:
                        entry = {"status": status, "evidence": evidence}
            elif rc_path.is_file():
                entry = {
                    "status": "infrastructure_failure",
                    "return_code": return_code,
                    "error": "arm exited successfully without arm.json",
                }
            else:
                entry = {"status": "missing"}
            arms[name] = entry
            summary[entry["status"]] += 1

    complete = summary["infrastructure_failure"] == summary["missing"] == 0
    verdict = "incomplete"
    if complete:
        verdict = "pass" if summary["quality_failure"] == 0 else "quality_failure"
    return {"arms": arms, "summary": summary, "verdict": verdict}


def _selected_decoder(model: Any, layer: int) -> tuple[str, Any]:
    suffix = f"language_model.layers.{layer}"
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name == suffix or name.endswith(f".{suffix}")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one decoder module ending {suffix!r}, got {matches}")
    return matches[0]


def expected_target_names(model: Any, ignore: list[str], layer: int) -> list[str]:
    """Resolve the exact Linear modules the arm may quantize before calibration."""
    from compressed_tensors.utils.match import match_name

    selected = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Linear":
            continue
        if any(match_name(name, pattern) for pattern in ignore):
            continue
        selected.append(name)
    marker = f".language_model.layers.{layer}.mlp.experts."
    invalid = [name for name in selected if marker not in f".{name}"]
    if invalid or not selected:
        raise RuntimeError(
            f"target isolation failed for layer {layer}: count={len(selected)} invalid={invalid[:20]}"
        )
    allowed_suffixes = (".gate_proj", ".up_proj", ".down_proj")
    invalid = [name for name in selected if not name.endswith(allowed_suffixes)]
    if invalid:
        raise RuntimeError(f"unexpected selected expert projections: {invalid[:20]}")
    return sorted(selected)


def validate_quantized_modules(
    model: Any, expected: list[str], *, layer: int
) -> list[str]:
    """Validate schemes after lifecycle initialization but before calibration."""
    actual = sorted(
        name
        for name, module in model.named_modules()
        if hasattr(module, "quantization_scheme")
    )
    if actual != sorted(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"initialized quantization targets differ for layer {layer}: "
            f"missing={missing[:20]} extra={extra[:20]}"
        )
    return actual


def build_arm_recipe(config: Any, *, layer: int, variant: str, expected: list[str]):
    """Build explicit per-layer AWQ + W4AFP8 modifiers with lifecycle auditing."""
    from pydantic import PrivateAttr
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from pipeline.minimax_m3_config import get_minimax_m3_awq_mappings

    disable = variant == "nosmooth"

    class AuditedAWQModifier(AWQModifier):
        audit_layer: int
        audit_variant: str
        expected_targets: list[str]
        _audit_snapshot: list[dict[str, Any]] = PrivateAttr(default_factory=list)
        _lifecycle_audit: dict[str, Any] = PrivateAttr(default_factory=dict)
        _skipped_error_metrics: list[dict[str, Any]] = PrivateAttr(
            default_factory=list
        )
        _capture_phase: str = PrivateAttr(default="reference")
        _captures: dict[str, dict[str, list[Any]]] = PrivateAttr(
            default_factory=lambda: {
                phase: {name: [] for name in BOUNDARIES}
                for phase in ("reference", "candidate")
            }
        )
        # Diagnostics for the completed=0 failure (native, HooksMixin-independent).
        _diag_fire_counts: dict[str, int] = PrivateAttr(default_factory=dict)
        _diag_handles: list[Any] = PrivateAttr(default_factory=list)
        _diag_timeline: list[dict[str, Any]] = PrivateAttr(default_factory=list)
        # Structural probe: which module level along mlp -> experts -> expert ->
        # gate_proj actually executes, and whether the executing objects are the
        # same objects AWQ resolved+hooked (id equality). Distinguishes routing
        # bypass, non-looping container, and stale/duplicate module objects.
        _struct_fire_counts: dict[str, int] = PrivateAttr(default_factory=dict)
        _struct_info: dict[str, Any] = PrivateAttr(default_factory=dict)
        _struct_handles: list[Any] = PrivateAttr(default_factory=list)

        def _record(self, boundary: str, value: Any) -> None:
            values = self._captures[self._capture_phase][boundary]
            if len(values) >= PROBE_COUNT:
                return
            tensor = unwrap_tensor(value)
            if tensor.ndim >= 2:
                tensor = tensor.narrow(-2, 0, min(tensor.shape[-2], PROBE_MAX_TOKENS))
            values.append(tensor.detach().float().cpu())

        def _install_boundary_hooks(self, model: Any) -> None:
            _, decoder = _selected_decoder(model, self.audit_layer)

            def layer_pre(_module, args, kwargs):
                self._record(
                    "layer_input", args[0] if args else kwargs.get("hidden_states")
                )

            self.register_hook(decoder, layer_pre, "forward_pre", with_kwargs=True)
            self.register_hook(
                decoder.post_attention_layernorm,
                lambda _module, _args, output: self._record("moe_input", output),
                "forward",
            )
            self.register_hook(
                decoder.mlp,
                lambda _module, _args, output: self._record("moe_output", output),
                "forward",
            )
            self.register_hook(
                decoder,
                lambda _module, _args, output: self._record("layer_output", output),
                "forward",
            )

        def on_calibration_start(self, state, event, **kwargs):
            try:
                result = super().on_calibration_start(state, event, **kwargs)
                validate_quantized_modules(
                    state.model, self.expected_targets, layer=self.audit_layer
                )
                self._audit_snapshot = resolved_mapping_snapshot(
                    self._resolved_mappings,
                    layer=self.audit_layer,
                    variant=self.audit_variant,
                )
                self._install_boundary_hooks(state.model)
                self._install_diagnostic_hooks()
                self._install_structure_probe(state.model)
            except Exception:
                self._remove_diagnostic_hooks()
                self._remove_structure_probe()
                self.remove_hooks()
                raise
            return result

        def _install_structure_probe(self, model: Any) -> None:
            """Probe the target layer's MoE dispatch chain at runtime.

            All balance targets showing zero forward events is consistent with
            several distinct root causes. This walks the *live* target decoder's
            ``mlp -> experts -> experts[0] -> gate_proj`` chain, records each
            object's runtime type, registers an independent native forward
            counter at every level, and checks whether ``experts[0].gate_proj``
            is the *same object* AWQ resolved into ``self._resolved_mappings``.

            Reading the per-level counts pinpoints where execution stops:

            * ``mlp`` never fires -> the sparse block is not entered for this
              layer at all (dense path, dead branch, or the layer's forward is
              not executed by the traced subgraph);
            * ``mlp`` fires but ``experts`` (container) does not -> the block
              runs but dispatches around the linearized experts object;
            * ``experts`` fires but ``experts[0]`` does not -> the container
              forward does not loop the per-expert child modules;
            * ``experts[0]`` fires but ``gate_proj`` does not -> the expert
              forward bypasses its own ``Linear`` submodules;
            * ``gate_proj`` fires but the AWQ balance counter stays 0, or
              ``resolved_object_is_live`` is False -> AWQ hooked a stale/duplicate
              module object (e.g. one captured by FX graph construction) rather
              than the object that executes.

            Fail-safe: any error disables the probe without affecting the arm.
            """
            self._struct_fire_counts = {}
            self._struct_info = {}
            self._struct_handles = []
            if not _hook_trace_enabled():
                return
            try:
                _, decoder = _selected_decoder(model, self.audit_layer)
                mlp = getattr(decoder, "mlp", None)
                targets: dict[str, Any] = {"mlp": mlp}
                experts = getattr(mlp, "experts", None) if mlp is not None else None
                targets["mlp.shared_experts"] = getattr(mlp, "shared_experts", None)
                targets["mlp.gate"] = getattr(mlp, "gate", None)
                targets["mlp.experts"] = experts
                expert0 = None
                if experts is not None:
                    try:
                        expert0 = experts[0]
                    except Exception:
                        expert0 = None
                targets["mlp.experts.0"] = expert0
                gate_proj = getattr(expert0, "gate_proj", None)
                targets["mlp.experts.0.gate_proj"] = gate_proj

                # Record runtime types and expert count for the report.
                self._struct_info["types"] = {
                    key: type(module).__name__
                    for key, module in targets.items()
                    if module is not None
                }
                self._struct_info["num_experts"] = (
                    len(experts) if experts is not None else None
                )

                # id equality: is the executing gate_proj the object AWQ hooked?
                resolved_by_id = {
                    id(layer)
                    for mapping in self._resolved_mappings
                    for layer in mapping.balance_layers
                }
                self._struct_info["resolved_object_is_live"] = (
                    gate_proj is not None and id(gate_proj) in resolved_by_id
                )

                for key, module in targets.items():
                    if module is None:
                        continue
                    self._struct_fire_counts.setdefault(key, 0)

                    def _counter(_module, _args, _output, _key=key):
                        self._struct_fire_counts[_key] += 1

                    self._struct_handles.append(
                        module.register_forward_hook(_counter)
                    )
            except Exception:
                # Diagnostics must never affect the arm's outcome.
                self._remove_structure_probe()

        def _remove_structure_probe(self) -> None:
            for handle in self._struct_handles:
                try:
                    handle.remove()
                except Exception:
                    pass
            self._struct_handles = []

        def _install_diagnostic_hooks(self) -> None:
            """Native per-balance-layer forward counters, keyed by smooth_name.

            AWQ accumulates smoothing stats via a HooksMixin forward hook on each
            mapping's activation target. This registers an *independent* native
            forward hook on the same module so we can tell two failure modes apart:

            * count == 0  -> the module's ``forward`` never ran during calibration
              (sequential FX tracing inlined it out of the subgraph, or MoE routing
              bypassed it), so AWQ's hook had nothing to fire on;
            * count  > 0 but ``_smooth_activation_stats`` stays empty -> the module
              ran but AWQ's activation-cache closure did not accumulate (implicates
              the closure / loss-mask / all-experts path, not tracing).
            """
            self._diag_fire_counts = {}
            self._diag_handles = []
            if not _hook_trace_enabled():
                return
            try:
                for mapping in self._resolved_mappings:
                    target = (
                        mapping.activation_hook_target or mapping.balance_layers[0]
                    )
                    name = mapping.smooth_name
                    self._diag_fire_counts.setdefault(name, 0)

                    def _counter(_module, _args, _output, _name=name):
                        self._diag_fire_counts[_name] += 1

                    self._diag_handles.append(target.register_forward_hook(_counter))
            except Exception:
                # Diagnostics must never affect the arm's outcome.
                self._remove_diagnostic_hooks()

        def _remove_diagnostic_hooks(self) -> None:
            for handle in self._diag_handles:
                try:
                    handle.remove()
                except Exception:
                    pass
            self._diag_handles = []

        def on_finalize(self, state, **kwargs) -> bool:
            self._remove_diagnostic_hooks()
            self._remove_structure_probe()
            return super().on_finalize(state, **kwargs)

        def on_sequential_epoch_end(self, state, event, **kwargs):
            self._diag_timeline.append(
                {
                    "epoch": len(self._diag_timeline),
                    "smooth_activation_stats_len": len(self._smooth_activation_stats),
                    "total_balance_forward_events": sum(
                        self._diag_fire_counts.values()
                    ),
                }
            )
            result = super().on_sequential_epoch_end(state, event, **kwargs)
            self._capture_phase = "candidate"
            return result

        def _log_error_metrics(self):
            resolved = [mapping.smooth_name for mapping in self._resolved_mappings]
            completed = [dict(metric) for metric in self._error_metrics]
            skipped = [dict(metric) for metric in self._skipped_error_metrics]
            accounted = {
                metric["layer_name"] for metric in [*completed, *skipped]
            }
            zero_fire = sorted(
                name for name, count in self._diag_fire_counts.items() if count == 0
            )
            self._lifecycle_audit = {
                "resolved_mapping_count": len(resolved),
                "resolved_mappings": resolved,
                "completed_mapping_count": len(completed),
                "completed_metrics": completed,
                "skipped_mapping_count": len(skipped),
                "skipped_metrics": skipped,
                "unprocessed_mappings": [
                    name for name in resolved if name not in accounted
                ],
                "diagnostics": {
                    "hook_trace_enabled": _hook_trace_enabled(),
                    "total_balance_forward_events": sum(
                        self._diag_fire_counts.values()
                    ),
                    "balance_layers_never_fired_count": len(zero_fire),
                    "balance_layers_never_fired": zero_fire,
                    "balance_forward_fire_counts": dict(self._diag_fire_counts),
                    "smooth_activation_stats_timeline": list(self._diag_timeline),
                    "structure_probe": {
                        "fire_counts": dict(self._struct_fire_counts),
                        **self._struct_info,
                    },
                },
            }
            return super()._log_error_metrics()

        @property
        def audit_snapshot(self) -> list[dict[str, Any]]:
            return list(self._audit_snapshot)

        @property
        def lifecycle_audit(self) -> dict[str, Any]:
            return copy.deepcopy(self._lifecycle_audit)

        @property
        def boundary_captures(self) -> dict[str, dict[str, list[Any]]]:
            return self._captures

    awq = AuditedAWQModifier(
        mappings=get_minimax_m3_awq_mappings(
            disable_mlp_input_smoothing=disable, layer=layer
        ),
        duo_scaling=config.quantization.awq_duo_scaling,
        audit_layer=layer,
        audit_variant=variant,
        expected_targets=expected,
    )
    class CapturingQuantizationModifier(QuantizationModifier):
        audit_layer: int
        expected_targets: list[str]
        capture_owner: Any
        _native_handles: list[Any] = PrivateAttr(default_factory=list)

        def _install_candidate_hooks(self, model: Any) -> None:
            from compressed_tensors.quantization import enable_quantization

            modules = dict(model.named_modules())
            for name in self.expected_targets:
                modules[name].apply(enable_quantization)
            _, decoder = _selected_decoder(model, self.audit_layer)

            def layer_pre(_module, args, kwargs):
                self.capture_owner._record(
                    "layer_input", args[0] if args else kwargs.get("hidden_states")
                )

            self._native_handles = [
                decoder.register_forward_pre_hook(layer_pre, with_kwargs=True),
                decoder.post_attention_layernorm.register_forward_hook(
                    lambda _module, _args, output: self.capture_owner._record(
                        "moe_input", output
                    )
                ),
                decoder.mlp.register_forward_hook(
                    lambda _module, _args, output: self.capture_owner._record(
                        "moe_output", output
                    )
                ),
                decoder.register_forward_hook(
                    lambda _module, _args, output: self.capture_owner._record(
                        "layer_output", output
                    )
                ),
            ]

        def on_sequential_epoch_end(self, state, event, modules, **kwargs):
            result = super().on_sequential_epoch_end(
                state, event, modules=modules, **kwargs
            )
            self._install_candidate_hooks(state.model)
            return result

        def on_calibration_end(self, state, event, **kwargs):
            try:
                return super().on_calibration_end(state, event, **kwargs)
            finally:
                for handle in self._native_handles:
                    handle.remove()
                self._native_handles.clear()

    quant = CapturingQuantizationModifier(
        targets=["Linear"],
        scheme=config.quantization.scheme,
        ignore=list(config.quantization.ignore),
        audit_layer=layer,
        expected_targets=expected,
        capture_owner=awq,
    )
    return [awq, quant], awq


def fidelity_from_captures(captures: dict[str, dict[str, list[Any]]]):
    """Compute per-probe and aggregate metrics from sequential-pipeline captures."""
    for phase in ("reference", "candidate"):
        for boundary in BOUNDARIES:
            count = len(captures.get(phase, {}).get(boundary, []))
            if count != PROBE_COUNT:
                raise RuntimeError(
                    f"{phase} {boundary} captured {count}, expected {PROBE_COUNT}"
                )
    per_probe = []
    for index in range(PROBE_COUNT):
        per_probe.append(
            {
                boundary: tensor_fidelity(
                    captures["reference"][boundary][index],
                    captures["candidate"][boundary][index],
                )
                for boundary in BOUNDARIES
            }
        )
    aggregate = {}
    for boundary in BOUNDARIES:
        reference = np.concatenate(
            [_as_numpy(value).reshape(-1) for value in captures["reference"][boundary]]
        )
        candidate = np.concatenate(
            [_as_numpy(value).reshape(-1) for value in captures["candidate"][boundary]]
        )
        aggregate[boundary] = tensor_fidelity(reference, candidate)
    return per_probe, aggregate


def _version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_arm(
    *, layer: int, variant: str, output_dir: Path, config_path: Path = DEFAULT_CONFIG,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Quantize one selected layer in memory and persist compact fidelity evidence."""
    import random
    import torch
    from llmcompressor import oneshot
    from pipeline.calibration import build_calibration_dataset
    from pipeline.config import load_config
    from pipeline.minimax_m3_config import patch_minimax_m3_for_text_calibration
    from pipeline.quantize import _load_model_and_tokenizer

    _validate_layer(layer)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    started = time.monotonic()
    config = prepare_arm_config(load_config(config_path), layer)
    if model_id is not None:
        config.model.id = model_id
    os.environ["M3_AWQ_DISABLE_MLP_INPUT_SMOOTH"] = "1" if variant == "nosmooth" else "0"
    random.seed(config.calibration.seed)
    np.random.seed(config.calibration.seed)
    torch.manual_seed(config.calibration.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.calibration.seed)

    start_manifest = {
        "schema_version": 1,
        "status": "started",
        "layer": layer,
        "variant": variant,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "command": [sys.executable, *sys.argv],
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "node": platform.node(),
    }
    _write_json_atomic(output_dir / "start.json", start_manifest)

    model, tokenizer = _load_model_and_tokenizer(config)
    if not patch_minimax_m3_for_text_calibration(model):
        raise RuntimeError("loaded model is not recognized as MiniMax-M3")
    decoder_name, _ = _selected_decoder(model, layer)
    config.calibration.sequential_targets = [decoder_name]
    dataset = build_calibration_dataset(config.calibration, tokenizer)
    token_hash = hashlib.sha256()
    for index in range(PROBE_COUNT):
        token_hash.update(
            np.asarray(
                dataset[index]["input_ids"][:PROBE_MAX_TOKENS], dtype=np.int64
            ).tobytes()
        )
    probe_manifest = {
        "indices": list(range(PROBE_COUNT)),
        "token_ids_sha256": token_hash.hexdigest(),
        "max_tokens": PROBE_MAX_TOKENS,
        "capture_source": "sequential calibration reference/propagation passes",
    }
    expected = expected_target_names(model, config.quantization.ignore, layer)
    recipe, audited_awq = build_arm_recipe(
        config, layer=layer, variant=variant, expected=expected
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
    if config.calibration.pipeline:
        kwargs["pipeline"] = config.calibration.pipeline
    oneshot(**kwargs)
    mapping_snapshot = audited_awq.audit_snapshot
    if not mapping_snapshot:
        raise RuntimeError("AWQ lifecycle audit snapshot was not retained")
    lifecycle_audit = audited_awq.lifecycle_audit
    _write_json_atomic(output_dir / "lifecycle.json", lifecycle_audit)
    if lifecycle_audit.get("completed_mapping_count", 0) == 0:
        raise RuntimeError(
            "AWQ completed zero mapping grid searches: "
            f"resolved={lifecycle_audit.get('resolved_mapping_count', 0)} "
            f"skipped={lifecycle_audit.get('skipped_mapping_count', 0)} "
            f"unprocessed={len(lifecycle_audit.get('unprocessed_mappings', []))}; "
            f"see {output_dir / 'lifecycle.json'}"
        )

    per_probe, metrics = fidelity_from_captures(audited_awq.boundary_captures)
    classification = classify_boundaries(metrics)
    evidence = {
        "schema_version": 1,
        "layer": layer,
        "variant": variant,
        "verdict": classification["verdict"],
        "classification": classification,
        "boundaries": metrics,
        "per_probe": per_probe,
        "targeted_modules": expected,
        "resolved_awq_mappings": mapping_snapshot,
        "awq_lifecycle": lifecycle_audit,
        "probe_manifest": probe_manifest,
        "configuration": {
            "config_path": str(config_path),
            "model_id": config.model.id,
            "method": config.quantization.method,
            "scheme": config.quantization.scheme,
            "num_samples": config.calibration.num_samples,
            "max_seq_length": config.calibration.max_seq_length,
            "sequential_targets": config.calibration.sequential_targets,
            "awq_duo_scaling": config.quantization.awq_duo_scaling,
            "awq_n_grid": getattr(audited_awq, "n_grid", None),
        },
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "llmcompressor": _version("llmcompressor"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "node": platform.node(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    _write_json_atomic(output_dir / "arm.json", evidence)
    return evidence


def render_report(matrix: dict[str, Any]) -> str:
    lines = ["# MiniMax-M3 AWQ Representative-Layer Matrix", "", f"Verdict: `{matrix['verdict']}`", "", "| Arm | Status |", "| --- | --- |"]
    for name, arm in matrix["arms"].items():
        lines.append(f"| `{name}` | `{arm['status']}` |")
    lines.extend(["", "```json", json.dumps(matrix["summary"], indent=2), "```", ""])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    arm = commands.add_parser("arm")
    arm.add_argument("--layer", type=int, choices=LAYERS, required=True)
    arm.add_argument("--variant", choices=VARIANTS, required=True)
    arm.add_argument("--output-dir", type=Path, required=True)
    arm.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arm.add_argument("--model-id")
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--result-root", type=Path, required=True)
    aggregate.add_argument("--matrix-json", type=Path, required=True)
    aggregate.add_argument("--report-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "arm":
        evidence = run_arm(
            layer=args.layer,
            variant=args.variant,
            output_dir=args.output_dir,
            config_path=args.config,
            model_id=args.model_id,
        )
        print(json.dumps({"verdict": evidence["verdict"], "output": str(args.output_dir)}))
        return 0
    matrix = aggregate_matrix(args.result_root)
    _write_json_atomic(args.matrix_json, matrix)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_report(matrix))
    return 0 if matrix["verdict"] != "incomplete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
