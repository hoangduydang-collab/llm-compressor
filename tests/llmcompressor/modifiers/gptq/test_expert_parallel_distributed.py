"""Real two-process EP GPTQ numerical and persistent-publication regressions.

These CPU/Gloo tests use the real GLM experts, GPTQ solver and distributed
compressed-tensors caches. They do not qualify NCCL or checkpoint export.
"""

import copy
import gc
import os
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from compressed_tensors.offload import disable_offloading, offload_module
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.cache.dist_cpu import DistributedCPUCache
from compressed_tensors.offload.cache.dist_disk import DistributedDiskCache

from llmcompressor.core import State
from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.expert_parallel import expert_parallel_context
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight
from llmcompressor.modifiers.quantization.calibration import observe
from llmcompressor.modifiers.utils.hooks import HooksMixin
from tests.llmcompressor.modeling.moe import test_expert_parallel_equivalence as fixture

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires real torch.distributed Gloo support",
)


def _launch(worker, tmp_path, *args, backend="gloo"):
    """Bound process lifetime as well as individual collective operations."""
    context = mp.start_processes(
        _worker_entry,
        args=(worker, str(tmp_path / "rendezvous"), str(tmp_path), args, backend),
        nprocs=2,
        join=False,
        start_method="spawn" if backend == "nccl" else "fork",
    )
    process_timeout = 240 if backend == "nccl" else 120
    deadline = time.monotonic() + process_timeout
    try:
        while time.monotonic() < deadline:
            if context.join(timeout=1):
                return
        pytest.fail(
            f"EP worker exceeded {process_timeout} seconds; inspect rank phase files"
        )
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def _worker_entry(rank, worker, rendezvous, workdir, args, backend):
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2"
    )
    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        backend,
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60 if backend == "nccl" else 30),
    )
    try:
        worker(rank, Path(workdir), *args)
    finally:
        dist.destroy_process_group()


def _phase(workdir, rank, phase):
    with (workdir / f"rank-{rank}-phases.txt").open("a") as stream:
        stream.write(f"{phase}\n")


def _model():
    # Parameterize the existing fixture, including its real GLM construction;
    # three experts gives uneven ownership (two experts versus one).
    with patch.object(fixture, "N_EXPERTS", 3):
        experts = fixture.build_experts()
    model = torch.nn.Module()
    model.add_module("experts", experts)
    model.add_module("replicated", torch.nn.Linear(fixture.HIDDEN, 8, bias=False))
    return model.eval()


def _data(step, rank):
    with patch.object(fixture, "N_EXPERTS", 3):
        return fixture.routing(
            tokens=((3, 7), (6, 4))[step][rank], seed=100 + 2 * step + rank
        )


def _modifier(model, expert_parallel=False, dynamic=False):
    scheme = {
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
    modifier = GPTQModifier(
        config_groups={"group": {"targets": ["Linear"], **scheme}},
        actorder="group",
        block_size=8,
        expert_parallel=expert_parallel,
    )
    state = State(model=model)
    modifier.on_initialize(state)
    if expert_parallel:
        modifier.prepare_expert_parallel(model)
    modifier.on_calibration_start(state, None)
    return modifier, state


def _numerical_worker(rank, workdir, dynamic):
    from llmcompressor.modifiers.gptq import base as gptq_base

    torch.manual_seed(19)
    model = _model()
    reference = copy.deepcopy(model)
    ep, state = _modifier(model, expert_parallel=True, dynamic=dynamic)
    ref, ref_state = _modifier(reference, dynamic=dynamic)
    names = {module: name for name, module in model.named_modules()}
    reference_modules = dict(reference.named_modules())
    targets = sorted(ep._module_names, key=names.__getitem__)
    _phase(workdir, rank, "initialized")

    # The reference visits the exact same global calibration manifest one
    # rank's batch at a time: no artificial sample-count override is used.
    with torch.no_grad(), moe_calibration_context():
        for step in range(2):
            for source_rank in range(2):
                hidden, index, weights = _data(step, source_rank)
                reference.experts(hidden, index, weights)
                reference.replicated(hidden)
            hidden, index, weights = _data(step, rank)
            with expert_parallel_context():
                got = model.experts(hidden, index, weights)
                model.replicated(hidden)
            # Reference forward hooks have already recorded the intended four
            # contributions. Disable only GPTQ hooks for this output comparison.
            with HooksMixin.disable_hooks():
                expected = reference.experts(hidden, index, weights)
            torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    _phase(workdir, rank, "calibrated")
    expected_local = {m for m in targets if ep._ep_owners.get(m, rank) == rank}
    assert set(ep._hessians) == expected_local
    assert set(ep._num_samples) == expected_local
    for module in expected_local:
        ref_module = reference_modules[names[module]]
        hessian = ep._hessians[module].clone()
        count = ep._num_samples[module].clone()
        if module not in ep._ep_owners:
            dist.all_reduce(hessian)
            dist.all_reduce(count)
        assert count.item() == ref._num_samples[ref_module].item() == 4
        torch.testing.assert_close(
            hessian, ref._hessians[ref_module], rtol=2e-5, atol=1e-6
        )
        torch.testing.assert_close(
            hessian / count, ref._hessians[ref_module] / count, rtol=2e-5, atol=1e-6
        )

    # Real reference solver results, including nontrivial weight_g_idx.
    expected_results = {}
    observe(list(reference_modules[names[m]] for m in targets), "weight")
    for module in targets:
        ref_module = reference_modules[names[module]]
        with torch.no_grad():
            _, result = quantize_weight(
                ref_module,
                ref_module.quantization_scheme.weights,
                ref._hessians[ref_module] / ref._num_samples[ref_module],
                blocksize=8,
                percdamp=ref.dampening_frac,
            )
        expected_results[names[module]] = result

    publications = []
    publish = gptq_base.publish_gptq_result

    def record_publication(module, owner, local_result, *args, **kwargs):
        publications.append((names[module], owner))
        assert (local_result is not None) == (owner == rank)
        return publish(module, owner, local_result, *args, **kwargs)

    with (
        patch.object(gptq_base, "publish_gptq_result", record_publication),
        expert_parallel_context(),
    ):
        ep.on_sequential_epoch_end(state, None, targets)
    _phase(workdir, rank, "published")
    assert not ep._hessians and not ep._num_samples
    gathered = [None, None]
    dist.all_gather_object(gathered, publications)
    assert gathered[0] == gathered[1]
    assert len(publications) == len(targets)
    assert {name for name, _ in publications} == set(expected_results)
    assert any(owner == 1 for _, owner in publications)
    assert len([1 for _, owner in publications if owner == 0]) != len(
        [1 for _, owner in publications if owner == 1]
    )
    for module in targets:
        for attribute, expected in expected_results[names[module]].items():
            actual = getattr(module, attribute)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
            assert torch.isfinite(actual).all()
            if attribute.endswith("scale"):
                assert (actual > 0).all()
    ep.on_calibration_end(state, None)
    ref.on_calibration_end(ref_state, None)
    _phase(workdir, rank, "complete")


@pytest.mark.parametrize("dynamic", [False, True], ids=["weight-only", "dynamic-fp8"])
def test_real_gloo_expert_hessians_and_gptq_match_reference(tmp_path, dynamic):
    _launch(_numerical_worker, tmp_path, dynamic)


def _publication_worker(rank, workdir, cache_kind, slot_case):
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    from llmcompressor.modifiers.gptq.distributed import publish_gptq_result
    from llmcompressor.modifiers.quantization.calibration import initialize_observer
    from pipeline.quantize import install_distributed_disk_update_offload_patch

    torch.manual_seed(23)
    module = torch.nn.Linear(8, 6, bias=False)
    quant_args = QuantizationArgs(
        num_bits=4, symmetric=True, strategy="group", group_size=2, actorder="group"
    )
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"], weights=quant_args
    )
    initialize_observer(module, "weight")
    observe(module, "weight")
    with torch.no_grad():
        _, expected = quantize_weight(
            module, quant_args, torch.diag(torch.arange(1, 9).float())
        )
    assert "weight_g_idx" in expected
    # Global-scale publication is transport coverage; this supplemental tensor
    # does not claim that the CPU solver supports the NVFP4 recipe.
    expected["weight_global_scale"] = torch.tensor([1.75])
    for attribute, value in expected.items():
        if attribute == "weight":
            continue
        if slot_case == "missing" and attribute in (
            "weight_g_idx",
            "weight_global_scale",
        ):
            continue
        shape = (1,) if slot_case == "reshape" else value.shape
        module.register_parameter(
            attribute,
            torch.nn.Parameter(
                torch.full(shape, -3, dtype=value.dtype), requires_grad=False
            ),
        )
    if cache_kind == "disk":
        install_distributed_disk_update_offload_patch()
    kwargs = {"offload_dir": str(workdir)} if cache_kind == "disk" else {}
    offload_module(module, "cpu", cache_kind, **kwargs)
    cache = module._parameters
    assert isinstance(
        cache, DistributedDiskCache if cache_kind == "disk" else DistributedCPUCache
    )
    if cache_kind == "cpu" and slot_case == "private":
        # Existing CPU qparameters are allowed to be rank-private even when
        # weights use DistributedCPUCache; exercise that real cache state too.
        for attribute in expected:
            if attribute != "weight":
                cache.offloaded_values[attribute] = cache.offloaded_values[
                    attribute
                ].clone()
                assert not cache.offloaded_values[attribute].is_shared()
    _phase(workdir, rank, "offloaded")
    with disable_offloading():
        # Populate local onload entries with old contents; publication must
        # refresh those as well as the persistent cache.
        stale = {
            name: getattr(module, name)
            for name in expected
            if name in cache.offloaded_values
        }
        publish_gptq_result(module, 1, expected if rank == 1 else None)
        for attribute, value in expected.items():
            torch.testing.assert_close(
                getattr(module, attribute), value, rtol=0, atol=0
            )
        del stale
    assert not OffloadCache.keep_onloaded_values
    gc.collect()
    _phase(workdir, rank, "onloads-released")
    assert set(expected) <= set(cache.offloaded_values)
    for attribute, value in expected.items():
        # Bypass all cached onloads and reread actual backing storage.
        fresh = cache.onload(cache.offloaded_values[attribute]).detach().clone()
        torch.testing.assert_close(fresh, value, rtol=0, atol=0)
        assert fresh.device.type != "meta"
        assert torch.isfinite(fresh).all()
        if attribute.endswith("scale"):
            assert (fresh > 0).all()
    dist.barrier()
    _phase(workdir, rank, "complete")


@pytest.mark.parametrize(
    "cache_kind,slot_case",
    [
        ("cpu", "existing"),
        ("cpu", "private"),
        ("cpu", "missing"),
        ("cpu", "reshape"),
        ("disk", "existing"),
        ("disk", "missing"),
        ("disk", "reshape"),
    ],
)
def test_real_distributed_publication_survives_offload(tmp_path, cache_kind, slot_case):
    _launch(_publication_worker, tmp_path, cache_kind, slot_case)


def _manifest_worker(rank, workdir, mismatch):
    from llmcompressor.modifiers.gptq.distributed import agree_ep_manifest

    manifest = {
        "modules": ["experts.0.gate_proj"],
        "steps": 2,
        "scheme": {"weights": {"num_bits": 4}},
    }
    if rank == 1:
        if mismatch == "steps":
            manifest["steps"] = 3
        elif mismatch == "targets":
            manifest["modules"].append("experts.1.gate_proj")
        else:
            manifest["scheme"]["weights"]["num_bits"] = 8
    try:
        agree_ep_manifest(manifest)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("rank accepted inconsistent EP manifests")
    messages = [None, None]
    dist.all_gather_object(messages, message)
    assert messages[0] == messages[1]
    assert message
    _phase(workdir, rank, "rejected")


@pytest.mark.parametrize("mismatch", ["steps", "targets", "scheme"])
def test_manifest_disagreement_fails_on_every_real_rank(tmp_path, mismatch):
    _launch(_manifest_worker, tmp_path, mismatch)


def _coverage_worker(rank, workdir, defect):
    from llmcompressor.modifiers.gptq.distributed import validate_hessian_coverage

    model = _model()
    modifier, state = _modifier(model, expert_parallel=True)
    with torch.no_grad(), moe_calibration_context(), expert_parallel_context():
        hidden, index, weights = _data(0, rank)
        model.experts(hidden, index, weights)
        model.replicated(hidden)
    modules = sorted(modifier._module_names, key=modifier._module_names.__getitem__)
    owner_one = next(m for m in modules if modifier._ep_owners.get(m) == 1)
    if rank == 1 and defect == "missing":
        del modifier._hessians[owner_one]
        del modifier._num_samples[owner_one]
    elif rank == 0 and defect == "extra":
        modifier._hessians[owner_one] = torch.eye(owner_one.in_features)
        modifier._num_samples[owner_one] = torch.tensor(2.0)
    elif rank == 1 and defect == "zero-count":
        modifier._num_samples[owner_one].zero_()
    with pytest.raises(ValueError, match="Hessian coverage") as error:
        validate_hessian_coverage(
            modules,
            modifier._module_names,
            modifier._ep_owners,
            modifier._hessians,
            modifier._num_samples,
        )
    errors = [None, None]
    dist.all_gather_object(errors, str(error.value))
    assert errors[0] == errors[1]
    modifier.on_calibration_end(state, None)
    _phase(workdir, rank, "coverage-rejected")


@pytest.mark.parametrize("defect", ["missing", "extra", "zero-count"])
def test_actual_hessian_coverage_defects_fail_on_every_rank(tmp_path, defect):
    _launch(_coverage_worker, tmp_path, defect)
