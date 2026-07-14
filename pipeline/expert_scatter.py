"""Single-process expert-scatter orchestration for GPTQ calibration.

Phase 2 of `M3_QUANT_SPEEDUP_PLAN`. GPTQ quantizes each (linearized) expert
independently -- `quantize_weight` reads only that module's weight and its own
Hessian and writes only that module's params (see
`GPTQModifier.compress_module_list`). So the serial per-expert loop can be
replaced by a concurrent dispatch across `cuda:0..N-1` with **no inter-expert
communication and no change to the quantization math**.

This module owns exactly the device-agnostic part: assign experts to devices,
run a provided `quantize_fn` concurrently, gather results by name. That makes it
bit-parity testable on CPU with a mock `quantize_fn` -- the property we must
guarantee is that scatter produces, per expert, the identical result the serial
path would, regardless of worker count or device assignment.

NOT in this module (deliberately): the accelerate onload/offload relocation of
each expert onto its assigned device inside the GPTQ modifier. That touches
offload accounting (cf. the FSDP2 reshard bug class) and can only be validated on
GPU, so it is a separate, bench-gated wiring step. Keep this core pure.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class ScatterItem:
    """One expert's quantization work unit. `module` is optional so the core is
    testable without the compressed-tensors module machinery."""

    name: str
    hessian: torch.Tensor
    weight: torch.Tensor | None = None
    module: Any = None
    quant_args: Any = None
    extra: dict = field(default_factory=dict)


# quantize_fn(item, device) -> result. The result is whatever the caller wants
# to gather (e.g. quantize_weight's (loss, q_param_dict)); this core is agnostic.
QuantizeFn = Callable[[ScatterItem, torch.device], Any]


def assign_devices(items: list[ScatterItem], devices: list) -> list[torch.device]:
    """Balance experts across devices by Hessian size (a cost proxy), largest
    first -- longest-processing-time greedy, which bounds imbalance well. Pure
    and deterministic (stable tie-break on original index)."""
    if not devices:
        raise ValueError("assign_devices requires at least one device")
    devs = [torch.device(d) for d in devices]
    load = [0] * len(devs)
    assignment: list[torch.device | None] = [None] * len(items)
    order = sorted(
        range(len(items)),
        key=lambda i: (int(items[i].hessian.shape[0]), -i),
        reverse=True,
    )
    for i in order:
        k = min(range(len(devs)), key=lambda j: (load[j], j))
        assignment[i] = devs[k]
        load[k] += int(items[i].hessian.shape[0])
    return [d for d in assignment if d is not None]


def serial_quantize(items: list[ScatterItem], quantize_fn: QuantizeFn, device="cpu") -> dict:
    """Reference path: quantize every expert one-at-a-time on a single device.
    This is what the current GPTQ loop does; scatter must match it exactly."""
    dev = torch.device(device)
    return {item.name: quantize_fn(item, dev) for item in items}


def scatter_quantize(
    items: list[ScatterItem],
    devices: list,
    quantize_fn: QuantizeFn,
    max_workers: int | None = None,
) -> dict:
    """Quantize experts concurrently, one worker per device. Results are keyed by
    expert name and are independent of scheduling order. `quantize_fn` is
    responsible for moving the item's tensors onto the device it is handed."""
    if not items:
        return {}
    assignment = assign_devices(items, devices)
    workers = max_workers if max_workers is not None else len({str(d) for d in assignment})

    def work(idx: int):
        return items[idx].name, quantize_fn(items[idx], assignment[idx])

    results: dict = {}
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        # pool.map yields in submission order; consumed here in the main thread,
        # so the results dict is written single-threaded (no lock needed).
        for name, res in pool.map(work, range(len(items))):
            results[name] = res
    return results


def default_gptq_quantize_fn(item: ScatterItem, device: torch.device):
    """Adapter to llm-compressor's real GPTQ core. GPU path, imported lazily so
    this module stays importable (and CPU-testable) without compressed_tensors.

    NOTE: correctness of this adapter against the serial modifier path is a
    GPU-gated Phase-2 check, not covered by the CPU orchestration tests.
    """
    from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight

    module = item.module
    if module is None:
        raise ValueError(f"default_gptq_quantize_fn needs item.module for {item.name}")
    module.to(device)
    hessian = item.hessian.to(device)
    loss, q_param_dict = quantize_weight(
        module=module,
        quant_args=item.quant_args,
        hessian=hessian,
        **item.extra,
    )
    return loss, q_param_dict
