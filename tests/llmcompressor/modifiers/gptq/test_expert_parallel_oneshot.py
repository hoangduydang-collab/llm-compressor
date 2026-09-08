"""Small real GLM sequential GPTQ, collective export and offline reload.

CPU/Gloo coverage is separate from the required executor NCCL qualification.
The weight-only case compares the complete DDP and EP walk. The mixed FP8-rest
case checks EP modifier isolation and save/reload: legacy mixed DDP overwrites
GPTQ scales, so it is not a valid numerical reference for the corrected EP path.
The separate expert test compares dynamic-activation GPTQ against its reference.
"""

import gc
import json
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
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


def _recipe(ep, dynamic):
    targets = ["re:.*mlp\\.(experts\\.\\d+\\.)?(gate_proj|up_proj|down_proj)$"]
    scheme = {
        "targets": targets,
        "weights": {
            "num_bits": 4,
            "type": "int",
            "symmetric": True,
            "strategy": "group",
            "group_size": 8,
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
        block_size=8,
        actorder="group",
    )
    if not dynamic:
        return [main]
    return [
        main,
        QuantizationModifier(
            scheme="FP8_DYNAMIC",
            targets=["Linear"],
            ignore=targets + ["lm_head", "re:.*gate$"],
        ),
    ]


def _lifecycle_worker(
    rank, workdir, dynamic, *, compare_baseline=True, disk_offload=False
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
    compare_baseline = compare_baseline and not dynamic
    for ep in (False, True) if compare_baseline else (True,):
        arm = "ep" if ep else "ddp"
        model = build_model(num_hidden_layers=3, first_k_dense_replace=1, n_experts=3)
        model.to(device)
        if disk_offload:
            from compressed_tensors.offload import offload_module

            from pipeline.quantize import install_distributed_disk_update_offload_patch

            install_distributed_disk_update_offload_patch()
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    offload_module(
                        module, device, "disk", offload_dir=str(workdir / "offload")
                    )
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
                recipe=_recipe(ep, dynamic),
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
        with torch.no_grad():
            output = model(**inputs).logits
        assert torch.isfinite(output).all()
        if not ep:
            reference_output = output.detach().clone()
            reference_hessians = combined
            reference_propagation = propagation
            model.save_pretrained(str(workdir / "ddp-checkpoint"), save_compressed=True)
        else:
            if compare_baseline:
                assert len(propagation) == len(reference_propagation)
                for (batch, actual), (ref_batch, expected) in zip(
                    propagation, reference_propagation
                ):
                    assert batch == ref_batch and set(actual) == set(expected)
                    for name, value in actual.items():
                        torch.testing.assert_close(
                            value, expected[name], rtol=3e-4, atol=2e-5
                        )
                assert set(combined) == set(reference_hessians)
                for name, (hessian, count) in combined.items():
                    expected_hessian, expected_count = reference_hessians[name]
                    assert count == expected_count == 4
                    torch.testing.assert_close(
                        hessian, expected_hessian, rtol=3e-4, atol=2e-5
                    )
                    torch.testing.assert_close(
                        hessian / count,
                        expected_hessian / expected_count,
                        rtol=3e-4,
                        atol=2e-5,
                    )
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
        del model, output, records, combined, gathered, propagation
        active_session().reset()
        gc.collect()
        dist.barrier()

    # All original models and the active session are released before readback.
    checkpoint = workdir / "checkpoint"
    with (checkpoint / "config.json").open() as stream:
        config = json.load(stream)
    quantization_config = config["quantization_config"]
    groups = quantization_config["config_groups"].values()
    int4_groups = [group for group in groups if group["weights"]["num_bits"] == 4]
    assert int4_groups and all(
        group["weights"]["group_size"] == 8 for group in int4_groups
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


def _preflight_worker(rank, workdir, defect):
    from compressed_tensors.quantization import QuantizationArgs

    model = build_model(num_hidden_layers=3, first_k_dense_replace=1, n_experts=3)
    recipe = _recipe(True, False)
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
    "defect", ["zero-steps", "unequal-steps", "static-activations"]
)
def test_real_oneshot_preflight_rejects_before_forward(tmp_path, defect):
    _launch(_preflight_worker, tmp_path, defect)
