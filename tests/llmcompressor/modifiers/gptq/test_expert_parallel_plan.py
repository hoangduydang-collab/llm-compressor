"""Preflight contracts for the existing EP adapter, before any traced forward."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import distributed as dist

from llmcompressor.core import State
from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.gptq.distributed import routed_expert_owners, solve_rounds
from llmcompressor.pipelines.sequential.pipeline import _prepare_expert_parallel
from pipeline.config import QuantizationConfig
from pipeline.recipe import build_recipe
from tests.llmcompressor.modeling.moe import test_expert_parallel_equivalence as fixture


@pytest.fixture
def process_group(tmp_path):
    dist.init_process_group(
        "gloo",
        init_method=f'file://{tmp_path / "rendezvous"}',
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=10),
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_actual_expert_ownership_excludes_shared_modules():
    with patch.object(fixture, "N_EXPERTS", 7):
        experts = fixture.build_experts()
    model = torch.nn.ModuleDict({"experts": experts, "shared": torch.nn.Linear(32, 32)})
    owners = routed_expert_owners(model, 3)
    counts = [0, 0, 0]
    for index in range(7):
        projections = [
            m for m in experts[index].modules() if isinstance(m, torch.nn.Linear)
        ]
        assert len({owners[m] for m in projections}) == 1
        counts[owners[projections[0]]] += 1
    assert counts == [3, 2, 2]
    assert model["shared"] not in owners
    assert len(owners) == 7 * 3


def test_rounds_cover_unequal_queues_once_with_one_result_per_owner():
    queues = [["a", "b", "c"], [], ["d"]]
    rounds = list(solve_rounds(queues))
    assert rounds == [[(0, "a"), (2, "d")], [(0, "b")], [(0, "c")]]
    assert sorted(module for row in rounds for _, module in row) == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    "case,expected",
    [
        ("empty", "nonzero"),
        ("routed", "moe_calibrate_all_experts"),
        ("target", "decoder-layer"),
        ("two_layers", "one decoder target"),
    ],
)
def test_preflight_rejects_unsupported_walk_before_trace(process_group, case, expected):
    modifier = GPTQModifier(expert_parallel=True, scheme="W4A16")
    args = SimpleNamespace(
        sequential_targets_per_subgraph=2 if case == "two_layers" else 1,
        propagate_error=True,
    )
    targets = ["Linear"] if case == "target" else ["GlmMoeDsaDecoderLayer"]
    batches = [] if case == "empty" else [torch.ones(1)]
    import contextlib

    context = (
        contextlib.nullcontext() if case == "routed" else moe_calibration_context()
    )
    with context, pytest.raises(ValueError, match=expected):
        _prepare_expert_parallel(torch.nn.Module(), batches, args, [modifier], targets)


def test_ep_recipe_is_opt_in_and_rejects_awq():
    assert not build_recipe(QuantizationConfig(method="gptq", scheme="W4A16"))[
        0
    ].expert_parallel
    assert build_recipe(
        QuantizationConfig(method="gptq", scheme="W4A16", gptq_expert_parallel=True)
    )[0].expert_parallel
    with pytest.raises(ValueError, match="method=gptq"):
        build_recipe(QuantizationConfig(method="awq", gptq_expert_parallel=True))


def test_prepare_rejects_static_expert_activation_observers(process_group):
    model = fixture.build_experts()
    modifier = GPTQModifier(
        expert_parallel=True,
        config_groups={
            "static": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 8,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "tensor",
                    "dynamic": False,
                },
            },
        },
    )
    state = State()
    state.update(model=model)
    modifier.on_initialize(state)
    with pytest.raises(ValueError, match="observer-free dynamic"):
        modifier.prepare_expert_parallel(model)


def _inconsistent_preflight_worker(rank, workdir, case):
    args = SimpleNamespace(sequential_targets_per_subgraph=1, propagate_error=True)
    if case == "flag":
        modifier = GPTQModifier(expert_parallel=(rank == 0), scheme="W4A16")
        with moe_calibration_context(), pytest.raises(
            ValueError, match="manifests differ"
        ):
            _prepare_expert_parallel(
                torch.nn.Module(),
                [torch.ones(1)],
                args,
                [modifier],
                ["GlmMoeDsaDecoderLayer"],
            )
    else:
        from llmcompressor.modifiers.gptq import base as gptq_base

        model = fixture.build_experts()
        modifier = GPTQModifier(
            expert_parallel=True,
            config_groups={
                "weights": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "group",
                        "group_size": 8,
                    },
                },
            },
        )
        state = State()
        state.update(model=model)
        modifier.on_initialize(state)
        original = gptq_base.routed_expert_owners

        def ownership(model, world_size):
            if rank == 0:
                raise ValueError("a routed module has conflicting owners")
            return original(model, world_size)

        with patch.object(gptq_base, "routed_expert_owners", ownership):
            with pytest.raises(ValueError, match="manifests differ"):
                modifier.prepare_expert_parallel(model)


@pytest.mark.parametrize("case", ["flag", "ownership"])
def test_rank_local_preflight_errors_fail_all_ranks(tmp_path, case):
    from tests.llmcompressor.modifiers.gptq.test_expert_parallel_distributed import (
        _launch,
    )

    _launch(_inconsistent_preflight_worker, tmp_path, case)
