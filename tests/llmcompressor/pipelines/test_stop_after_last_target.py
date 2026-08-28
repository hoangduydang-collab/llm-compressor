"""Early exit from the sequential walk once nothing is left to compress.

Motivation, measured on GLM-5.2 (run 20260828t052128z): the sequential pipeline
propagates activations through all 78 decoder layers even when the config only
targets two of them, and each untargeted layer costs ~2.5 min -- 19 GB of
weights read at ~127 MB/s of network storage. That is hours of a smoke run spent
walking layers it will never quantize.

The risk this trades against is severe and silent: stopping one subgraph too
early leaves a layer that SHOULD have been quantized sitting in BF16, and the
checkpoint still saves and still loads. So the helper is fail-closed -- it
returns None (walk everything) unless every targeted module can be attributed to
a subgraph at or before the cut. These tests pin that direction specifically,
including the GLM-shaped case where an autowrapped call leaves expert modules
with no graph node of their own.
"""

import torch

from llmcompressor.pipelines.sequential.pipeline import last_subgraph_with_targets


class FakeNode:
    def __init__(self, target):
        self.op = "call_module"
        self.target = target


class FakeGraph:
    def __init__(self, targets):
        self._nodes = [FakeNode(t) for t in targets]

    def find_nodes(self, op):
        return [n for n in self._nodes if n.op == op]


class FakeSubgraph:
    """Stands in for llmcompressor.pipelines.sequential.helpers.Subgraph.

    Only ``graph.find_nodes(op="call_module")`` is consulted by the helper, which
    is the point: it deliberately does not use ``submodules()``.
    """

    def __init__(self, targets):
        self.graph = FakeGraph(targets)


def build_model(layer_count, quantized_layers, *, expert_count=0):
    """A model shaped like a decoder stack, with schemes on chosen layers."""
    model = torch.nn.Module()
    layers = torch.nn.ModuleList()
    for index in range(layer_count):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Linear(4, 4)
        if expert_count:
            layer.mlp = torch.nn.Module()
            layer.mlp.experts = torch.nn.ModuleList(
                torch.nn.Linear(4, 4) for _ in range(expert_count)
            )
        else:
            layer.mlp = torch.nn.Linear(4, 4)
        layers.append(layer)
    model.layers = layers

    for index in quantized_layers:
        layer = model.layers[index]
        if expert_count:
            for expert in layer.mlp.experts:
                expert.quantization_scheme = object()
        else:
            layer.mlp.quantization_scheme = object()
    return model


def layer_subgraphs(layer_count):
    """One subgraph per layer, each naming the layer's direct submodules."""
    return [
        FakeSubgraph([f"layers.{i}.self_attn", f"layers.{i}.mlp"])
        for i in range(layer_count)
    ]


# --------------------------------------------------------------------------
# the win this exists for
# --------------------------------------------------------------------------


def test_stops_after_the_last_targeted_layer():
    model = build_model(78, quantized_layers=[3, 42])
    assert last_subgraph_with_targets(model, layer_subgraphs(78)) == 42


def test_skips_nothing_when_the_last_layer_is_targeted():
    model = build_model(78, quantized_layers=[3, 77])
    assert last_subgraph_with_targets(model, layer_subgraphs(78)) == 77


def test_single_targeted_layer_cuts_the_whole_tail():
    """The layer-3-only smoke: 74 of 78 layers become skippable."""
    model = build_model(78, quantized_layers=[3])
    assert last_subgraph_with_targets(model, layer_subgraphs(78)) == 3


# --------------------------------------------------------------------------
# fail-closed: the dangerous direction
# --------------------------------------------------------------------------


def test_returns_none_when_nothing_is_targeted():
    """No schemes at all means we cannot distinguish -- walk everything."""
    model = build_model(8, quantized_layers=[])
    assert last_subgraph_with_targets(model, layer_subgraphs(8)) is None


def test_returns_none_when_a_target_cannot_be_attributed():
    """The coverage invariant: one unplaceable target disables skipping.

    Here layer 60 is quantized but no subgraph mentions it. Returning 42 would
    silently ship layer 60 in BF16.
    """
    model = build_model(78, quantized_layers=[3, 42, 60])
    subgraphs = [s for i, s in enumerate(layer_subgraphs(78)) if i != 60]
    assert last_subgraph_with_targets(model, subgraphs) is None


def test_autowrapped_experts_are_attributed_via_an_ancestor_node():
    """GLM shape: experts carry the schemes but have no call_module node.

    `self.experts(...).view(*orig_shape)` cannot be traced (fx does not handle
    `*args` unpacking), so the autowrapper wraps the whole expression and the
    expert modules get zero nodes. Prefix matching against the `mlp` ancestor
    must still attribute them, otherwise coverage fails and we lose the feature
    on exactly the model that motivated it.
    """
    model = build_model(78, quantized_layers=[3, 42], expert_count=4)
    assert last_subgraph_with_targets(model, layer_subgraphs(78)) == 42


def test_expert_targets_with_no_ancestor_node_fail_closed():
    """Same shape, but the subgraph does not name `mlp` either -> refuse."""
    model = build_model(78, quantized_layers=[3, 42], expert_count=4)
    subgraphs = [FakeSubgraph([f"layers.{i}.self_attn"]) for i in range(78)]
    assert last_subgraph_with_targets(model, subgraphs) is None


# --------------------------------------------------------------------------
# prefix matching must not over-match
# --------------------------------------------------------------------------


def test_prefix_match_requires_a_dot_boundary():
    """`layers.1` must not claim `layers.11`'s modules."""
    model = build_model(20, quantized_layers=[11])
    # Only layer 1's subgraph exists, named with the bare prefix.
    subgraphs = [FakeSubgraph(["layers.1"])]
    assert last_subgraph_with_targets(model, subgraphs) is None


def test_exact_name_match_counts():
    model = build_model(4, quantized_layers=[2])
    subgraphs = [FakeSubgraph([f"layers.{i}.mlp"]) for i in range(4)]
    assert last_subgraph_with_targets(model, subgraphs) == 2


def test_empty_subgraph_list_is_none():
    model = build_model(4, quantized_layers=[1])
    assert last_subgraph_with_targets(model, []) is None
