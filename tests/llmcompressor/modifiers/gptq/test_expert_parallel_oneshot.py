"""Small real GLM sequential GPTQ, collective export and offline reload.

CPU/Gloo coverage is separate from the required executor NCCL qualification.
The weight-only and early block-FP8 cases compare the complete DDP and EP walk.
The historical mixed FP8-rest case checks EP isolation and save/reload. The new
case also freezes FP8 scale/payloads and uses a block-aligned real GLM fixture.
"""

import gc
import json
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.offload import disable_offloading
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from llmcompressor import oneshot
from llmcompressor.core import active_session
from llmcompressor.modeling.moe.expert_parallel import get_expert_parallel_context
from llmcompressor.modeling.moe.linearize import load_quantizable_moe
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines.cache import IntermediatesCache
from tests.llmcompressor.modeling.moe.test_expert_parallel_equivalence import (
    build_model,
)
from tests.llmcompressor.modifiers.gptq.test_expert_parallel_distributed import (
    _launch,
    _phase,
)

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires real torch.distributed Gloo support",
)


def _calibration(rank):
    generator = torch.Generator().manual_seed(101 + rank)
    data = [
        {
            "input_ids": torch.randint(3, 128, (1, length), generator=generator),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }
        for length in ((5, 7) if rank == 0 else (8, 6))
    ]
    return torch.utils.data.DataLoader(data, batch_size=None)


def _processor():
    # Fully local tokenizer: the pre-tokenized loader performs no tokenization.
    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}, unk_token="[UNK]")
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )


def _recipe(ep, dynamic, early_fp8=False, group_size=8):
    targets = ["re:.*mlp\\.(experts\\.\\d+\\.)?(gate_proj|up_proj|down_proj)$"]
    if early_fp8:
        targets = [r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"]
    scheme = {
        "targets": targets,
        "weights": {
            "num_bits": 4,
            "type": "int",
            "symmetric": True,
            "strategy": "group",
            "group_size": group_size,
        },
    }
    if dynamic:
        scheme["input_activations"] = {
            "num_bits": 8,
            "type": "float",
            "strategy": "token",
            "dynamic": True,
        }
    main = GPTQModifier(
        config_groups={"experts": scheme},
        expert_parallel=ep,
        block_size=group_size,
        actorder="static" if early_fp8 else "group",
    )
    if not dynamic:
        return [main]
    rest_scheme = "FP8_DYNAMIC"
    if early_fp8:
        from compressed_tensors.quantization import preset_name_to_scheme

        # The independent reference also uses tiny widths below activation
        # group size. These tests isolate block-FP8 WEIGHT preparation/replay.
        rest_scheme = preset_name_to_scheme("FP8_BLOCK", ["Linear"])
        rest_scheme.input_activations = None
    return [
        main,
        QuantizationModifier(
            **(
                {"config_groups": {"rest": rest_scheme}}
                if early_fp8
                else {"scheme": rest_scheme}
            ),
            quantize_weights_before_calibration=early_fp8,
            targets=["Linear"],
            ignore=targets + ["lm_head", "re:.*gate$"],
        ),
    ]


def _relative_errors(actual, expected):
    difference = (actual.double() - expected.double()).abs()
    return {
        "absolute_max": float(difference.max()),
        "relative_frobenius": float(
            difference.norm() / expected.double().norm().clamp_min(1e-12)
        ),
        "normalized_max": float(
            difference.max() / expected.double().abs().max().clamp_min(1e-12)
        ),
    }


def _lifecycle_worker(
    rank,
    workdir,
    dynamic,
    *,
    compare_baseline=True,
    disk_offload=False,
    early_fp8=False,
):
    device = (
        torch.device("cuda", rank)
        if dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    inputs = {
        name: value.to(device) for name, value in next(iter(_calibration(rank))).items()
    }
    reference_output, reference_hessians, reference_propagation = None, None, None
    reference_weight_output = None
    expected_fp8 = {}
    compare_baseline = compare_baseline and (not dynamic or early_fp8)
    for ep in (False, True) if compare_baseline else (True,):
        arm = "ep" if ep else "ddp"
        model = build_model(
            num_hidden_layers=3,
            first_k_dense_replace=1,
            n_experts=3,
            config_overrides={
                "hidden_size": 256,
                "intermediate_size": 256,
                "moe_intermediate_size": 256,
                "kv_lora_rank": 256,
                "q_lora_rank": 256,
                "qk_rope_head_dim": 128,
                "qk_nope_head_dim": 128,
                "v_head_dim": 128,
            }
            if early_fp8
            else None,
        )
        model.to(device)
        if disk_offload:
            from compressed_tensors.offload import offload_module
            from compressed_tensors.offload.module import remove_module_offload

            from pipeline.quantize import install_distributed_disk_update_offload_patch

            install_distributed_disk_update_offload_patch()
            # The low-level cache API writes into an existing directory; unlike
            # the model loader it does not prepare the offload folder itself.
            offload_dir = workdir / "offload"
            offload_dir.mkdir(parents=True, exist_ok=True)
            _phase(workdir, rank, "disk:offload-start")
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    # Linearization already gives experts CPU offload caches.
                    # Reuse CT's transition helper before installing disk caches.
                    remove_module_offload(module, onload_tensors=True)
                    offload_module(module, device, "disk", offload_dir=str(offload_dir))
            _phase(workdir, rank, "disk:offload-complete")
        model.config.use_cache = False
        model.config._attn_implementation = "eager"
        records, propagation = {}, []
        solve = GPTQModifier._solve_module

        def record_solve(modifier, module, *, _records=records):
            name = modifier._module_names[module]
            _records[name] = (
                modifier._hessians[module].detach().cpu().clone(),
                modifier._num_samples[module].item(),
            )
            return solve(modifier, module)

        update_cache = IntermediatesCache.update

        def record_propagation(cache, batch_index, values, *, _records=propagation):
            _records.append(
                (
                    batch_index,
                    {
                        name: value.detach().cpu().clone()
                        for name, value in values.items()
                        if isinstance(value, torch.Tensor)
                    },
                )
            )
            return update_cache(cache, batch_index, values)

        rest_epoch = QuantizationModifier.on_sequential_epoch_end

        def checked_rest_epoch(
            modifier, state, event, modules, *, _model=model, **kwargs
        ):
            expert_qparams = {
                (module, attribute): getattr(module, attribute).detach().clone()
                for name, module in _model.named_modules()
                if ".experts." in name
                and isinstance(module, torch.nn.Linear)
                and module in modules
                and getattr(module, "weight_scale", None) is not None
                for attribute in ("weight_scale", "weight_zero_point", "weight_g_idx")
                if getattr(module, attribute, None) is not None
            }
            result = rest_epoch(modifier, state, event, modules, **kwargs)
            if ep:
                for (module, attribute), expected in expert_qparams.items():
                    torch.testing.assert_close(
                        getattr(module, attribute), expected, rtol=0, atol=0
                    )
            return result

        _phase(workdir, rank, f"{arm}:initialize")
        with (
            patch.object(GPTQModifier, "_solve_module", record_solve),
            patch.object(IntermediatesCache, "update", record_propagation),
            patch.object(
                QuantizationModifier, "on_sequential_epoch_end", checked_rest_epoch
            ),
        ):
            oneshot(
                model=model,
                processor=_processor(),
                recipe=_recipe(
                    ep, dynamic, early_fp8, group_size=128 if early_fp8 else 8
                ),
                dataset=_calibration(rank),
                num_calibration_samples=4,
                sequential_targets=["GlmMoeDsaDecoderLayer"],
                pipeline="sequential",
                moe_calibrate_all_experts=True,
            )
        assert get_expert_parallel_context() is None
        _phase(workdir, rank, f"{arm}:quantized")
        gathered = [None, None]
        dist.all_gather_object(gathered, records)
        combined = {}
        for owner_records in gathered:
            assert not set(combined).intersection(owner_records)
            combined.update(owner_records)
        assert combined
        assert all(count == 4 for _, count in combined.values())
        assert propagation
        # Match the calibration cache lifetime: CT's QDQ temporarily patches
        # weight.data, so disk-backed weight reads must reuse that same onload.
        with torch.no_grad(), disable_offloading():
            output = model(**inputs).logits
        assert torch.isfinite(output).all()
        weight_output = output
        if early_fp8:
            from llmcompressor.utils.helpers import DisableQuantization

            # Compare the requested weight-calibration semantics separately from
            # dynamic activation rounding used in the save/reload forward below.
            with torch.no_grad(), disable_offloading(), DisableQuantization(model):
                weight_output = model(**inputs).logits
        if early_fp8:
            from compressed_tensors.quantization import quantize

            for name, module in model.named_modules():
                scheme = getattr(module, "quantization_scheme", None)
                if scheme is not None and scheme.weights.type == "float":
                    scale = module.weight_scale.detach().cpu().clone()
                    payload = quantize(
                        module.weight,
                        module.weight_scale,
                        module.weight_zero_point,
                        scheme.weights,
                        dtype=torch.float8_e4m3fn,
                    ).cpu()
                    expected_fp8[name] = (scale, payload)
        # Native/generic save mutates compressed offload state. Do not let a
        # faster rank begin that transition while a peer is still collecting
        # the frozen FP8 payload/scale evidence from shared disk-cache files.
        dist.barrier()
        if dynamic:
            from pipeline.quantize import _stamp_mixed_precision_formats

            _stamp_mixed_precision_formats(model)
        if not ep:
            reference_output = output.detach().clone()
            reference_weight_output = weight_output.detach().clone()
            reference_hessians = combined
            reference_propagation = propagation
            model.save_pretrained(str(workdir / "ddp-checkpoint"), save_compressed=True)
        else:
            if compare_baseline:
                assert len(propagation) == len(reference_propagation)
                parity = {}
                if early_fp8:
                    for index, ((_, actual), (_, expected)) in enumerate(
                        zip(propagation, reference_propagation)
                    ):
                        for name, value in actual.items():
                            parity[f"replay-{index}:{name}"] = _relative_errors(
                                value, expected[name]
                            )
                    for name, (hessian, _) in combined.items():
                        parity[f"hessian:{name}"] = _relative_errors(
                            hessian, reference_hessians[name][0]
                        )
                    parity["weight_only_logits"] = _relative_errors(
                        weight_output, reference_weight_output
                    )
                    (workdir / f"rank-{rank}-fp8-parity.json").write_text(
                        json.dumps(parity, indent=2, sort_keys=True)
                    )
                    (workdir / f"rank-{rank}-activation-logits.json").write_text(
                        json.dumps(_relative_errors(output, reference_output), indent=2)
                    )
                    # Distributed accumulation order can cross an INT4 rounding
                    # boundary. Gate aggregate error without unstable relative
                    # comparisons at individual values close to zero.
                    for name, errors in parity.items():
                        assert errors["relative_frobenius"] < 1e-3, (name, errors)
                        assert errors["normalized_max"] < 1e-2, (name, errors)
                for (batch, actual), (ref_batch, expected) in zip(
                    propagation, reference_propagation
                ):
                    assert batch == ref_batch and set(actual) == set(expected)
                    if not early_fp8:
                        for name, value in actual.items():
                            torch.testing.assert_close(
                                value, expected[name], rtol=3e-4, atol=2e-5
                            )
                assert set(combined) == set(reference_hessians)
                for name, (hessian, count) in combined.items():
                    expected_hessian, expected_count = reference_hessians[name]
                    assert count == expected_count == 4
                    # The first routed block sees identical prepared FP8 inputs;
                    # later blocks also carry preceding INT4 solve differences.
                    if not early_fp8 or name.startswith("model.layers.1."):
                        torch.testing.assert_close(
                            hessian, expected_hessian, rtol=3e-4, atol=2e-5
                        )
                        torch.testing.assert_close(
                            hessian / count,
                            expected_hessian / expected_count,
                            rtol=3e-4,
                            atol=2e-5,
                        )
                if not early_fp8:
                    torch.testing.assert_close(
                        output, reference_output, rtol=2e-3, atol=2e-4
                    )
            expected_experts = {
                name
                for name, module in model.named_modules()
                if ".experts." in name and isinstance(module, torch.nn.Linear)
            }
            assert len(expected_experts) == 18
            expected_keys = {
                f"{name}.{attribute}"
                for name in expected_experts
                for attribute in (
                    "weight_packed",
                    "weight_scale",
                    "weight_g_idx",
                    "weight_shape",
                )
                if attribute != "weight_g_idx" or not early_fp8
            }
            expected_quantized_output = output.detach().clone()
            if dynamic:
                from pipeline.quantize import _stamp_mixed_precision_formats

                formats = _stamp_mixed_precision_formats(model)
                assert {"pack-quantized", "float-quantized"} <= set(formats)
            _phase(workdir, rank, "ep:save-start")
            model.save_pretrained(str(workdir / "checkpoint"), save_compressed=True)
            _phase(workdir, rank, "ep:save-complete")
        del record_solve, checked_rest_epoch, record_propagation
        # Distributed disk caches share source-rank files. Keep every rank's
        # model/cache alive until all peers finish their pre-save inspection
        # and collective save; otherwise a faster rank's GC deletes files that
        # a slower rank still needs to onload.
        dist.barrier()
        del model, output, records, combined, gathered, propagation
        active_session().reset()
        gc.collect()
        dist.barrier()

    # All original models and the active session are released before readback.
    checkpoint = workdir / "checkpoint"
    with (checkpoint / "config.json").open() as stream:
        config = json.load(stream)
    quantization_config = config["quantization_config"]
    if early_fp8:
        from pipeline.serve_ignore import weight_map_of

        weight_map = weight_map_of(checkpoint)
        assert expected_fp8
        for name, (scale, payload) in expected_fp8.items():
            for suffix, expected in (("weight_scale", scale), ("weight", payload)):
                key = f"{name}.{suffix}"
                with safe_open(
                    str(checkpoint / weight_map[key]), framework="pt"
                ) as src:
                    actual = src.get_tensor(key)
                assert actual.dtype == expected.dtype
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    groups = quantization_config["config_groups"].values()
    int4_groups = [group for group in groups if group["weights"]["num_bits"] == 4]
    assert int4_groups and all(
        group["weights"]["group_size"] == (128 if early_fp8 else 8)
        for group in int4_groups
    )
    for group in int4_groups:
        activations = group.get("input_activations")
        assert bool(activations and activations["dynamic"]) == dynamic
    assert "lm_head" in quantization_config["ignore"]
    if compare_baseline:
        baseline_packed = {}
        for shard in (workdir / "ddp-checkpoint").glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                baseline_packed.update(
                    {
                        key: tensors.get_tensor(key)
                        for key in tensors.keys()
                        if key.endswith("weight_packed")
                    }
                )
        packed_differences = {}
        for shard in checkpoint.glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                for key in tensors.keys():
                    if key.endswith("weight_packed"):
                        value = tensors.get_tensor(key)
                        expected = baseline_packed[key]
                        assert value.shape == expected.shape
                        packed_differences[key] = int((value != expected).sum())
        assert set(packed_differences) == set(baseline_packed)
        # Record actual packed-word differences; numerical parity is the gate,
        # since quantization is not required to be bitwise reproducible.
        (workdir / f"rank-{rank}-packed-differences.json").write_text(
            json.dumps(packed_differences, sort_keys=True)
        )
    saved_keys = set()
    for shard in checkpoint.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            saved_keys.update(tensors.keys())
            for name in tensors.keys():
                if ".experts." in name:
                    tensor = tensors.get_tensor(name)
                    assert torch.isfinite(tensor).all()
                    if name.endswith("scale"):
                        assert (tensor > 0).all()
    assert expected_keys <= saved_keys
    assert any(".experts.2." in key for key in saved_keys)
    with load_quantizable_moe():
        reloaded = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map=str(device),
            local_files_only=True,
            attn_implementation="eager",
        )
    assert all(parameter.device.type != "meta" for parameter in reloaded.parameters())
    if early_fp8:
        for name, (scale, _) in expected_fp8.items():
            torch.testing.assert_close(
                reloaded.get_submodule(name).weight_scale.cpu(), scale, rtol=0, atol=0
            )
    reloaded.eval()
    with torch.no_grad():
        output = reloaded(**inputs).logits
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected_quantized_output, rtol=2e-3, atol=2e-4)
    _phase(workdir, rank, "ep:reload-forward-complete")


@pytest.mark.parametrize(
    "dynamic", [False, True], ids=["weight-only", "dynamic-fp8-rest"]
)
def test_real_glm_gloo_oneshot_collective_save_reload(tmp_path, dynamic):
    _launch(_lifecycle_worker, tmp_path, dynamic)


def _disk_lifecycle_worker(rank, workdir, dynamic):
    _lifecycle_worker(rank, workdir, dynamic, compare_baseline=False, disk_offload=True)


@pytest.mark.parametrize("dynamic", [False, True], ids=["weight-only", "fp8-rest"])
def test_real_glm_gloo_disk_oneshot_collective_save_reload(tmp_path, dynamic):
    # Exercise the same disk setup as the GPU gate even on CPU-only hosts.
    _launch(_disk_lifecycle_worker, tmp_path, dynamic)


def _preflight_worker(rank, workdir, defect):
    from compressed_tensors.quantization import QuantizationArgs

    model = build_model(num_hidden_layers=3, first_k_dense_replace=1, n_experts=3)
    recipe = _recipe(True, False)
    if defect == "fp8-dtype":
        recipe = _recipe(True, True, early_fp8=True)
        if rank == 1:
            projection = model.model.layers[0].self_attn.q_a_proj
            projection.weight.data = projection.weight.data.double()
    data = list(_calibration(rank))
    if defect == "zero-steps":
        data = []
    elif defect == "unequal-steps" and rank == 1:
        data = data[:1]
    elif defect == "static-activations":
        recipe[0].config_groups["experts"].input_activations = QuantizationArgs(
            num_bits=8,
            type="int",
            strategy="tensor",
            dynamic=False,
            observer="minmax",
        )
    calls = []
    handle = model.register_forward_pre_hook(lambda *args: calls.append(True))
    try:
        with pytest.raises(ValueError) as error:
            oneshot(
                model=model,
                processor=_processor(),
                recipe=recipe,
                dataset=torch.utils.data.DataLoader(data, batch_size=None),
                sequential_targets=["GlmMoeDsaDecoderLayer"],
                pipeline="sequential",
                moe_calibrate_all_experts=True,
            )
        assert not calls, "preflight failure must precede the first model forward"
        messages = [None, None]
        dist.all_gather_object(messages, str(error.value))
        assert messages[0] == messages[1]
        assert get_expert_parallel_context() is None
        _phase(workdir, rank, f"{defect}:rejected-before-forward")
    finally:
        handle.remove()
        active_session().reset()


@pytest.mark.parametrize(
    "defect", ["zero-steps", "unequal-steps", "static-activations", "fp8-dtype"]
)
def test_real_oneshot_preflight_rejects_before_forward(tmp_path, defect):
    _launch(_preflight_worker, tmp_path, defect)


def _early_fp8_worker(rank, workdir, disk):
    _lifecycle_worker(
        rank,
        workdir,
        True,
        compare_baseline=not disk,
        disk_offload=disk,
        early_fp8=True,
    )


@pytest.mark.parametrize("disk", [False, True], ids=["ddp-ep-parity", "disk"])
def test_real_glm_fp8_before_gptq_save_reload(tmp_path, disk):
    _launch(_early_fp8_worker, tmp_path, disk)


def test_fp8_preparation_matches_independent_two_block_reference():
    from compressed_tensors.utils import match_named_modules

    from pipeline.sglang_w4afp8_kernels import dequantize_block_fp8, quantize_block_fp8

    arms = []
    for explicit_reference in (True, False):
        model = build_model(num_hidden_layers=2, first_k_dense_replace=1, n_experts=2)
        recipe = _recipe(False, True, early_fp8=True)
        if explicit_reference:
            for _, module in match_named_modules(
                model, recipe[1].resolved_targets, recipe[1].ignore
            ):
                payload, scale = quantize_block_fp8(module.weight.detach())
                module.weight.data.copy_(dequantize_block_fp8(payload, scale))
            recipe[1].quantize_weights_before_calibration = False
        records, propagation = {}, []
        solve = GPTQModifier._solve_module
        update = IntermediatesCache.update

        def record_solve(modifier, module):
            records[modifier._module_names[module]] = modifier._hessians[module].clone()
            return solve(modifier, module)

        def record_update(cache, batch, values):
            propagation.append(
                {
                    name: value.detach().clone()
                    for name, value in values.items()
                    if isinstance(value, torch.Tensor)
                }
            )
            return update(cache, batch, values)

        with patch.object(GPTQModifier, "_solve_module", record_solve), patch.object(
            IntermediatesCache, "update", record_update
        ):
            oneshot(
                model=model,
                processor=_processor(),
                recipe=recipe,
                dataset=_calibration(0),
                num_calibration_samples=2,
                sequential_targets=["GlmMoeDsaDecoderLayer"],
                pipeline="sequential",
                moe_calibrate_all_experts=True,
            )
        arms.append((records, propagation))
        active_session().reset()
    reference, actual = arms
    assert reference[0] and reference[1]
    assert actual[0].keys() == reference[0].keys()
    for name, hessian in actual[0].items():
        torch.testing.assert_close(hessian, reference[0][name], rtol=1e-6, atol=1e-6)
    assert len(actual[1]) == len(reference[1])
    for observed, expected in zip(actual[1], reference[1]):
        assert observed.keys() == expected.keys()
        for name, value in observed.items():
            torch.testing.assert_close(value, expected[name], rtol=1e-6, atol=1e-6)
