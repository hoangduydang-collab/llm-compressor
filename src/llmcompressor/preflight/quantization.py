"""Structural model/recipe compatibility checks before quantization.

This module deliberately invokes modifier initialization and AWQ's real mapping
resolver, but never enters calibration or runs a model forward. Callers should pass
a disposable model, preferably instantiated on the meta device, because quantization
scheme metadata is attached to targeted modules.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from compressed_tensors.registry import standardize_lookup_name
from compressed_tensors.utils import match_named_modules

from llmcompressor.core import State
from llmcompressor.modeling.offset_norm import NormCalibrationModule
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization.quantization.mixin import QuantizationMixin
from llmcompressor.modifiers.transform.awq import AWQModifier

SCHEMA_VERSION = 1

# These classes use ``output * (1 + weight)`` and therefore require conversion to
# ordinary norm semantics while AWQ/SmoothQuant changes weights. Keeping this policy
# explicit makes an absent registry adapter a hard, actionable preflight failure.
KNOWN_OFFSET_NORM_CLASSES = frozenset(
    {
        "GemmaRMSNorm",
        "Gemma2RMSNorm",
        "Gemma3RMSNorm",
        "Qwen3NextRMSNorm",
        "Qwen3_5RMSNorm",
        "Qwen3_5MoeRMSNorm",
        "MiniMaxM3VLRMSNorm",
    }
)


@dataclass(frozen=True)
class Finding:
    code: str
    message: str


@dataclass(frozen=True)
class AWQMappingReport:
    smooth_name: str
    balance_names: tuple[str, ...]
    parent_name: str
    activation_hook_target: str | None


@dataclass(frozen=True)
class NormAdapterReport:
    module_name: str
    module_class: str
    status: str
    adapter_class: str | None


@dataclass(frozen=True)
class PlannerReport:
    modifier: str
    targets: tuple[str, ...]
    ignore: tuple[str, ...]


@dataclass(frozen=True)
class QuantizationCompatibilityReport:
    methods: tuple[str, ...]
    planners: tuple[PlannerReport, ...]
    quantized_modules: tuple[str, ...]
    awq_mappings: tuple[AWQMappingReport, ...]
    norm_adapters: tuple[NormAdapterReport, ...]
    failures: tuple[Finding, ...]
    warnings: tuple[Finding, ...]
    unverified: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    @property
    def compatible(self) -> bool:
        return not self.failures

    @property
    def quantized_module_count(self) -> int:
        return len(self.quantized_modules)

    @property
    def awq_mapping_count(self) -> int:
        return len(self.awq_mappings)

    def to_dict(self) -> dict[str, Any]:
        # Round-tripping converts immutable tuples into canonical JSON arrays while
        # also proving every report field is serializable.
        result = json.loads(json.dumps(asdict(self)))
        result["compatible"] = self.compatible
        result["quantized_module_count"] = self.quantized_module_count
        result["awq_mapping_count"] = self.awq_mapping_count
        return result


def _resolve_norm_adapter(class_name: str) -> type | None:
    lookup = standardize_lookup_name(class_name)
    if lookup not in (
        set(NormCalibrationModule.registered_names())
        | set(NormCalibrationModule.registered_aliases())
    ):
        return None
    return NormCalibrationModule.get_value_from_registry(class_name)


def _quantized_module_names(
    model: torch.nn.Module, modifiers: list[Any]
) -> tuple[str, ...]:
    names: set[str] = set()
    for modifier in modifiers:
        if not isinstance(modifier, QuantizationMixin):
            continue
        names.update(
            name
            for name, module in match_named_modules(
                model, modifier.resolved_targets, modifier.ignore
            )
            if getattr(module, "quantization_scheme", None) is not None
            and not isinstance(module, torch.nn.Embedding)
        )
    return tuple(sorted(names))


def analyze_quantization_compatibility(
    model: torch.nn.Module, modifiers: list[Any]
) -> QuantizationCompatibilityReport:
    """Run planner-only compatibility checks on a disposable model instance."""
    failures: list[Finding] = []
    warnings: list[Finding] = []
    methods = tuple(type(modifier).__name__ for modifier in modifiers)

    supported = (AWQModifier, QuantizationMixin)
    unsupported = [
        name for name, mod in zip(methods, modifiers) if not isinstance(mod, supported)
    ]
    if unsupported:
        failures.append(
            Finding(
                "unsupported_modifier",
                f"pre-quantization gate does not support: {', '.join(unsupported)}",
            )
        )

    state = State(model=model)
    for modifier in modifiers:
        if not isinstance(modifier, supported):
            continue
        try:
            modifier.on_initialize(state)
        except Exception as error:  # planner errors are report data
            failures.append(
                Finding(
                    "planner_initialization_failed",
                    f"{type(modifier).__name__}: {type(error).__name__}: {error}",
                )
            )

    planners: list[PlannerReport] = []
    for modifier in modifiers:
        if not isinstance(modifier, QuantizationMixin):
            continue
        try:
            targets = tuple(sorted(modifier.resolved_targets))
        except Exception:
            targets = ()
        planners.append(
            PlannerReport(
                modifier=type(modifier).__name__,
                targets=targets,
                ignore=tuple(modifier.ignore),
            )
        )

    try:
        quantized_modules = _quantized_module_names(model, modifiers)
    except Exception as error:
        quantized_modules = ()
        failures.append(
            Finding(
                "target_inventory_failed",
                f"{type(error).__name__}: {error}",
            )
        )
    if not quantized_modules:
        failures.append(
            Finding(
                "no_quantized_modules",
                "the recipe resolved no non-Embedding modules with a weight scheme",
            )
        )

    resolved_reports: list[AWQMappingReport] = []
    norm_reports: list[NormAdapterReport] = []
    for modifier in modifiers:
        if not isinstance(modifier, AWQModifier):
            continue
        try:
            modifier._set_resolved_mappings(model)
        except Exception as error:
            failures.append(
                Finding(
                    "awq_mapping_resolution_failed",
                    f"{type(error).__name__}: {error}",
                )
            )
            failures.append(
                Finding(
                    "no_awq_mappings",
                    "AWQ resolved no compatible, quantized smooth/balance mappings",
                )
            )
            continue
        if not modifier._resolved_mappings:
            failures.append(
                Finding(
                    "no_awq_mappings",
                    "AWQ resolved no compatible, quantized smooth/balance mappings",
                )
            )
        for mapping in modifier._resolved_mappings:
            hook_name = None
            if mapping.activation_hook_target is not None:
                hook_name = type(mapping.activation_hook_target).__name__
            resolved_reports.append(
                AWQMappingReport(
                    smooth_name=mapping.smooth_name,
                    balance_names=tuple(mapping.balance_names),
                    parent_name=mapping.parent_name,
                    activation_hook_target=hook_name,
                )
            )
            class_name = type(mapping.smooth_layer).__name__
            adapter = _resolve_norm_adapter(class_name)
            if adapter is not None:
                status = "supported_offset"
                adapter_name = adapter.__name__
            elif class_name in KNOWN_OFFSET_NORM_CLASSES:
                status = "missing_offset_adapter"
                adapter_name = None
                failures.append(
                    Finding(
                        "missing_offset_norm_adapter",
                        f"AWQ smooth layer {mapping.smooth_name!r} uses known offset "
                        f"norm {class_name} but no NormCalibrationModule adapter is "
                        "registered",
                    )
                )
            else:
                status = "ordinary_or_unclassified"
                adapter_name = None
                if "Norm" in class_name and not isinstance(
                    mapping.smooth_layer, torch.nn.LayerNorm
                ):
                    warnings.append(
                        Finding(
                            "unclassified_norm_semantics",
                            f"AWQ smooth layer {mapping.smooth_name!r} uses "
                            f"unclassified norm class {class_name}",
                        )
                    )
            norm_reports.append(
                NormAdapterReport(
                    module_name=mapping.smooth_name,
                    module_class=class_name,
                    status=status,
                    adapter_class=adapter_name,
                )
            )

    if not any(
        isinstance(modifier, (AWQModifier, GPTQModifier)) for modifier in modifiers
    ):
        failures.append(
            Finding(
                "unsupported_method", "version 1 requires AWQModifier or GPTQModifier"
            )
        )

    return QuantizationCompatibilityReport(
        methods=methods,
        planners=tuple(planners),
        quantized_modules=quantized_modules,
        awq_mappings=tuple(resolved_reports),
        norm_adapters=tuple(norm_reports),
        failures=tuple(_deduplicate_findings(failures)),
        warnings=tuple(_deduplicate_findings(warnings)),
        unverified=(
            "calibration_dataset_quality",
            "activation_statistics",
            "quantization_error",
            "checkpoint_serving_abi",
            "runtime_quality",
        ),
    )


def _deduplicate_findings(findings: list[Finding]) -> list[Finding]:
    return list(dict.fromkeys(findings))
