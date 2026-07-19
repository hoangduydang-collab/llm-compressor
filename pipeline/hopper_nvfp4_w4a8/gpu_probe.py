"""Bounded SM90 correctness probe for the patched Humming NVFP4 W4A8 path."""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any, Sequence

SHAPE_N = 128
SHAPE_K = 128
M_VALUES = (1, 8, 32)
FRAGMENT_OUTPUT_ROWS = (0, 1, 15, 16, 31, 32, 63, 64, 95, 96, 127)
SEED = 20260719
EXACT_RTOL = 0.01
EXACT_ATOL = 0.25


def _nbytes(tensor: Any) -> int:
    return tensor.nelement() * tensor.element_size()


def _metrics(actual: Any, reference: Any) -> dict[str, float]:
    import torch

    actual_f = actual.float()
    reference_f = reference.float()
    difference = (actual_f - reference_f).abs()
    relative = difference / reference_f.abs().clamp_min(1.0e-6)
    finite = torch.isfinite(actual_f)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.reshape(1, -1), reference_f.reshape(1, -1), dim=1
    ).item()
    return {
        "max_absolute": difference.max().item(),
        "max_relative": relative.max().item(),
        "cosine": cosine,
        "finite_fraction": finite.float().mean().item(),
    }


def _pack_e2m1(codes: Any) -> Any:
    # Humming's packed checkpoint ABI stores eight 4-bit values per int32.
    groups = codes.view(SHAPE_N, SHAPE_K // 8, 8)
    shifts = codes.new_tensor(tuple(range(0, 32, 4))).view(1, 1, 8)
    return ((groups & 0x0F) << shifts).sum(dim=-1, dtype=codes.dtype).contiguous()


def _make_weight_and_scales(torch: Any) -> tuple[Any, Any, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED)
    codes = torch.randint(
        0,
        16,
        (SHAPE_N, SHAPE_K),
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    )
    # Keep every scale-isolation dot product non-zero while retaining explicit
    # coverage of all sixteen E2M1 checkpoint codepoints on another row.
    codes[list(FRAGMENT_OUTPUT_ROWS)] = 0x02
    all_codes = torch.tensor(list(range(16)), device="cuda", dtype=torch.int32)
    codes[2, :16] = all_codes

    scale_codes = torch.empty(
        (SHAPE_N, SHAPE_K // 16), device="cuda", dtype=torch.uint8
    )
    scale_codes[:, 0::2] = 0x38  # E4M3 1.0, K0:16 of every K32 pair.
    scale_codes[:, 1::2] = 0x40  # E4M3 2.0, K16:32 of every K32 pair.
    scales = scale_codes.view(torch.float8_e4m3fn)
    return codes, scales, _pack_e2m1(codes)


def _weight_references(
    torch: Any, ops: Any, codes: Any, scales: Any
) -> tuple[Any, Any]:
    e2m1 = ops.dequant_weight(
        codes,
        exponent_bits=2,
        mantissa_bits=1,
        is_signed=True,
    ).float()
    expanded_scales = scales.float().repeat_interleave(16, dim=-1)
    exact_emulation = (e2m1 * expanded_scales * 0.125).to(torch.float8_e4m3fn).float()
    bf16_nvfp4 = (e2m1 * expanded_scales).to(torch.bfloat16)
    return exact_emulation, bf16_nvfp4


def _make_kernel(dtypes: Any, HummingKernel: Any) -> Any:
    return HummingKernel(
        shape_n=SHAPE_N,
        shape_k=SHAPE_K,
        block_shape=(16, 128, 64),
        warp_shape=(16, 32, 64),
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e4m3,
        num_stages=3,
        use_warp_spec=False,
        input_scale_group_size=0,
        weight_scale_group_size=16,
        weight_scale_type="group_tensor",
        has_zero_point=False,
        use_f16_accum=False,
        use_tma=False,
        use_cp_async=False,
        mma_type="wgmma",
        use_stream_k=False,
    )


def _launch(
    torch: Any,
    ops: Any,
    kernel: Any,
    inputs: Any,
    input_scale: Any,
    weight: Any,
    weight_scale: Any,
    global_scale: Any,
) -> Any:
    outputs = torch.zeros(
        (inputs.size(0), SHAPE_N), dtype=torch.bfloat16, device=inputs.device
    )
    return ops.launch_kernel(
        configs=[kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        global_scale=global_scale,
    )


def _exact_output(torch: Any, inputs_ref: Any, b8: Any) -> Any:
    # The overlay folds /8 into B registers and compensates exactly once here.
    return (inputs_ref.float().matmul(b8.T) * 8.0).to(torch.bfloat16)


def _bf16_output(torch: Any, inputs: Any, weight: Any) -> Any:
    return inputs.to(torch.bfloat16).matmul(weight.T).to(torch.bfloat16)


def _run_k16_isolation(
    torch: Any,
    ops: Any,
    kernel: Any,
    weight: Any,
    codes: Any,
    scales: Any,
    prepare_humming_weight_scale: Any,
    global_scale: Any,
) -> dict[str, Any]:
    changed_scales = scales.clone()
    changed_scales[:, 0] = torch.tensor(2.0, dtype=torch.float8_e4m3fn, device="cuda")
    scales_prepared = prepare_humming_weight_scale(scales, to_apply_on_c=False)
    changed_prepared = prepare_humming_weight_scale(changed_scales, to_apply_on_c=False)
    base_b8, _ = _weight_references(torch, ops, codes, scales)
    changed_b8, _ = _weight_references(torch, ops, codes, changed_scales)

    def isolated(group: int) -> tuple[Any, Any, Any, Any]:
        inputs = torch.zeros((1, SHAPE_K), dtype=torch.float8_e4m3fn, device="cuda")
        inputs[:, group * 16 : (group + 1) * 16] = 1.0
        input_scale = torch.ones((1, 1), dtype=torch.float32, device="cuda")
        baseline = _launch(
            torch,
            ops,
            kernel,
            inputs,
            input_scale,
            weight,
            scales_prepared,
            global_scale,
        )
        changed = _launch(
            torch,
            ops,
            kernel,
            inputs,
            input_scale,
            weight,
            changed_prepared,
            global_scale,
        )
        return (
            baseline,
            changed,
            _exact_output(torch, inputs.float(), base_b8),
            _exact_output(torch, inputs.float(), changed_b8),
        )

    group0 = isolated(0)
    group1 = isolated(1)
    group0_exact = torch.equal(group0[0], group0[2]) and torch.equal(
        group0[1], group0[3]
    )
    group1_exact = torch.equal(group1[0], group1[2]) and torch.equal(
        group1[1], group1[3]
    )
    result = {
        "changed_half_changes_output": not torch.equal(group0[0], group0[1]),
        "unchanged_half_is_unchanged": torch.equal(group1[0], group1[1]),
        "group0_matches_exact": group0_exact,
        "group1_matches_exact": group1_exact,
    }
    result["passed"] = all(result.values())
    return result


def _run_fragment_scale_isolation(
    torch: Any,
    ops: Any,
    kernel: Any,
    weight: Any,
    codes: Any,
    scales: Any,
    prepare_humming_weight_scale: Any,
    global_scale: Any,
) -> dict[str, Any]:
    """Exercise both K16 register pairs across every N16 output fragment."""

    cases = []
    baseline_prepared = prepare_humming_weight_scale(scales, to_apply_on_c=False)
    baseline_b8, _ = _weight_references(torch, ops, codes, scales)
    for output_row in FRAGMENT_OUTPUT_ROWS:
        for group in (0, 1):
            changed_scales = scales.clone()
            replacement = 2.0 if group == 0 else 1.0
            changed_scales[output_row, group] = torch.tensor(
                replacement, dtype=torch.float8_e4m3fn, device="cuda"
            )
            changed_prepared = prepare_humming_weight_scale(
                changed_scales, to_apply_on_c=False
            )
            changed_b8, _ = _weight_references(torch, ops, codes, changed_scales)
            inputs = torch.zeros((1, SHAPE_K), dtype=torch.float8_e4m3fn, device="cuda")
            inputs[:, group * 16 : (group + 1) * 16] = 1.0
            baseline = _launch(
                torch,
                ops,
                kernel,
                inputs,
                torch.ones((1, 1), dtype=torch.float32, device="cuda"),
                weight,
                baseline_prepared,
                global_scale,
            )
            changed = _launch(
                torch,
                ops,
                kernel,
                inputs,
                torch.ones((1, 1), dtype=torch.float32, device="cuda"),
                weight,
                changed_prepared,
                global_scale,
            )
            baseline_exact = _exact_output(torch, inputs.float(), baseline_b8)
            changed_exact = _exact_output(torch, inputs.float(), changed_b8)
            changed_columns = (baseline != changed).nonzero(as_tuple=False)[:, 1]
            passed = (
                torch.equal(baseline, baseline_exact)
                and torch.equal(changed, changed_exact)
                and torch.equal(
                    changed_columns,
                    torch.tensor([output_row], device="cuda"),
                )
            )
            cases.append(
                {"output_row": output_row, "k16_group": group, "passed": passed}
            )
    return {"passed": all(case["passed"] for case in cases), "cases": cases}


def _sass_evidence(kernel_filename: str, result_root: Path) -> dict[str, Any]:
    command = ["cuobjdump", "--dump-sass", kernel_filename]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    sass_path = result_root / "kernel.sass"
    sass_path.write_text(completed.stdout, encoding="utf-8")
    (result_root / "cuobjdump.stderr").write_text(completed.stderr, encoding="utf-8")
    upper = completed.stdout.upper()
    fp8_wgmma_found = (
        "WGMMA.MMA_ASYNC" in upper and "E4M3" in upper and completed.returncode == 0
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "sass_path": str(sass_path),
        "fp8_wgmma_found": fp8_wgmma_found,
    }


def run_probe(result_root: Path) -> dict[str, Any]:
    import torch
    from humming import dtypes, ops
    from humming.kernel.humming import HummingKernel
    from humming.layer import HummingLayerMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema
    from humming.utils.test import generate_random_inputs
    from humming.utils.weight import prepare_humming_weight_scale

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (9, 0):
        raise RuntimeError(f"SM90 required, found {capability}")

    codes, scales, checkpoint_weight = _make_weight_and_scales(torch)
    b8, bf16_weight = _weight_references(torch, ops, codes, scales)
    checkpoint_global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    checkpoint_global_scale_before = checkpoint_global_scale.clone()
    layer = torch.nn.Module()
    HummingLayerMethod.prepare_layer_meta(
        layer,
        shape_n=SHAPE_N,
        shape_k=SHAPE_K,
        weight_schema=HummingWeightSchema(
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e4m3,
            weight_scale_group_size=16,
            weight_scale_type="group_tensor",
        ),
        input_schema=HummingInputSchema(a_dtype=dtypes.float8e4m3),
        torch_dtype=torch.bfloat16,
    )
    layer.weight = torch.nn.Parameter(checkpoint_weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.global_scale = torch.nn.Parameter(
        checkpoint_global_scale, requires_grad=False
    )
    HummingLayerMethod.transform_humming_layer(layer)
    weight = layer.weight
    weight_scale = layer.weight_scale
    global_scale = layer.global_scale
    layer_transform = {
        "checkpoint_global_unchanged": torch.equal(
            checkpoint_global_scale, checkpoint_global_scale_before
        ),
        "effective_global_is_times_eight": torch.equal(
            global_scale, checkpoint_global_scale_before * 8.0
        ),
    }
    layer_transform["passed"] = all(layer_transform.values())
    kernel = _make_kernel(dtypes, HummingKernel)

    checkpoint_bytes = (
        _nbytes(checkpoint_weight) + _nbytes(scales) + _nbytes(checkpoint_global_scale)
    )
    transformed_bytes = _nbytes(weight) + _nbytes(weight_scale) + _nbytes(global_scale)
    persistent_ratio = transformed_bytes / checkpoint_bytes
    memory = {
        "checkpoint_bytes": checkpoint_bytes,
        "transformed_bytes": transformed_bytes,
        "persistent_ratio": persistent_ratio,
        "persistent_tensors": {
            "weight": {"dtype": str(weight.dtype), "shape": list(weight.shape)},
            "weight_scale": {
                "dtype": str(weight_scale.dtype),
                "shape": list(weight_scale.shape),
            },
            "global_scale": {
                "dtype": str(global_scale.dtype),
                "shape": list(global_scale.shape),
            },
        },
        "no_persistent_fp8_expanded_weight": (
            _nbytes(weight) == _nbytes(checkpoint_weight)
        ),
    }

    shape_results = []
    for shape_m in M_VALUES:
        inputs_orig, inputs_ref, inputs, input_scale = generate_random_inputs(
            m=shape_m,
            k=SHAPE_K,
            dtype=dtypes.float8e4m3,
        )
        first = _launch(
            torch,
            ops,
            kernel,
            inputs,
            input_scale,
            weight,
            weight_scale,
            global_scale,
        )
        second = _launch(
            torch,
            ops,
            kernel,
            inputs,
            input_scale,
            weight,
            weight_scale,
            global_scale,
        )
        exact = _exact_output(torch, inputs_ref, b8)
        bf16_nvfp4 = _bf16_output(torch, inputs_orig, bf16_weight)
        exact_metrics = _metrics(first, exact)
        bf16_metrics = _metrics(first, bf16_nvfp4)
        deterministic = torch.equal(first, second)
        exact_passed = exact_metrics["finite_fraction"] == 1.0 and torch.allclose(
            first, exact, rtol=EXACT_RTOL, atol=EXACT_ATOL
        )
        shape_results.append(
            {
                "m": shape_m,
                "exact_emulation": {**exact_metrics, "passed": exact_passed},
                "bf16_nvfp4": bf16_metrics,
                "deterministic": deterministic,
                "passed": exact_passed and deterministic,
            }
        )

    k16_isolation = _run_k16_isolation(
        torch,
        ops,
        kernel,
        weight,
        codes,
        scales,
        prepare_humming_weight_scale,
        global_scale,
    )
    fragment_scale_isolation = _run_fragment_scale_isolation(
        torch,
        ops,
        kernel,
        weight,
        codes,
        scales,
        prepare_humming_weight_scale,
        global_scale,
    )
    sass = _sass_evidence(kernel.kernel_filename, result_root)
    passed = (
        all(item["passed"] for item in shape_results)
        and k16_isolation["passed"]
        and fragment_scale_isolation["passed"]
        and persistent_ratio <= 1.10
        and memory["no_persistent_fp8_expanded_weight"]
        and layer_transform["passed"]
        and sass["fp8_wgmma_found"]
    )
    return {
        "passed": passed,
        "seed": SEED,
        "device_capability": list(capability),
        "shapes": shape_results,
        "k16_isolation": k16_isolation,
        "fragment_scale_isolation": fragment_scale_isolation,
        "memory": memory,
        "layer_transform": layer_transform,
        "kernel": {
            "kernel_id": kernel.kernel_id,
            "kernel_name": kernel.kernel_name,
            "kernel_filename": kernel.kernel_filename,
            "policy": "float8e4m3/float4e2m1/float8e4m3/g16/group_tensor/fp32",
        },
        "sass": sass,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_probe(args.output.parent)
    except Exception as exc:
        payload = {
            "passed": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
