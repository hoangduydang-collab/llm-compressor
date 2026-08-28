"""Tests for sequential weight prefetch / page-cache release.

The dangerous failure is not "prefetch did nothing" -- it is releasing a file a
later subgraph still needs, which converts an optimization into extra reads on the
slowest resource in the system. That case is tested directly.

A second class of failure is subtle: deciding WHAT to prefetch must not itself
onload the weights. `OffloadCache.__getitem__` onloads, so a naive
`cache.values()` would stream the whole layer off disk to plan a prefetch of that
same layer. A fake cache here raises if the Mapping interface is touched.
"""

from __future__ import annotations

import pytest
import torch

from llmcompressor.pipelines.sequential import weight_prefetch
from llmcompressor.pipelines.sequential.weight_prefetch import (
    _DONTNEED,
    _WILLNEED,
    WeightPrefetcher,
    plan_last_use,
    subgraph_source_files,
)


@pytest.fixture(autouse=True)
def _pretend_fadvise_exists(monkeypatch):
    """The syscall is Linux-only, but the file-selection and release logic is
    not, and this repo is developed on Windows. Force the capability flag on so
    the logic is covered everywhere; `_advise` itself is stubbed per-test."""
    monkeypatch.setattr(weight_prefetch, "_HAS_FADVISE", True)


class ExplodingOnGetCache(dict):
    """Stands in for OffloadCache: exposes `offloaded_values` and `index`, but
    raises if anything reads it as a Mapping (which would onload)."""

    def __init__(self, mapping: dict[str, torch.Tensor], index: dict):
        super().__init__()
        self.offloaded_values = mapping
        self.index = index

    def __getitem__(self, key):  # pragma: no cover - must never be called
        raise AssertionError(
            "planning a prefetch must not onload weights via __getitem__"
        )

    def values(self):  # pragma: no cover - must never be called
        raise AssertionError("planning a prefetch must not iterate cache values")


class FakeNode:
    def __init__(self, target):
        self.op = "call_module"
        self.target = target


class FakeGraph:
    def __init__(self, targets):
        self._targets = targets

    def find_nodes(self, op):
        assert op == "call_module"
        return [FakeNode(t) for t in self._targets]


class FakeSubgraph:
    def __init__(self, targets):
        self.graph = FakeGraph(targets)


def _model_with_offload(tmp_path, layout: dict[str, str]):
    """Build a module tree whose leaves are 'offloaded' to the given files.

    layout maps module name -> backing file name.
    """
    root = torch.nn.Module()
    index: dict[torch.Tensor, dict] = {}
    for name, filename in layout.items():
        path = tmp_path / filename
        if not path.exists():
            path.write_bytes(b"\0" * 4096)
        parent = root
        parts = name.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, torch.nn.Module())
            parent = getattr(parent, part)
        leaf = torch.nn.Module()
        meta = torch.empty(1, device="meta")
        index[meta] = {"safetensors_file": str(path), "weight_name": "w",
                       "dtype": "bfloat16"}
        leaf.__dict__["_parameters"] = ExplodingOnGetCache({"weight": meta}, index)
        setattr(parent, parts[-1], leaf)
    return root


# --- file discovery ---------------------------------------------------------

def test_subgraph_source_files_finds_backing_file(tmp_path):
    model = _model_with_offload(tmp_path, {"layers.0.proj": "shard-a.safetensors"})
    files = subgraph_source_files(model, FakeSubgraph(["layers.0.proj"]))
    assert files == {str(tmp_path / "shard-a.safetensors")}


def test_discovery_does_not_onload(tmp_path):
    """ExplodingOnGetCache raises on Mapping access; reaching here proves the
    planner read `offloaded_values` directly."""
    model = _model_with_offload(tmp_path, {"layers.0.proj": "shard-a.safetensors"})
    subgraph_source_files(model, FakeSubgraph(["layers.0"]))  # must not raise


def test_prefix_matching_covers_descendants(tmp_path):
    model = _model_with_offload(
        tmp_path,
        {"layers.0.attn.q": "a.safetensors", "layers.0.mlp.up": "b.safetensors"},
    )
    files = subgraph_source_files(model, FakeSubgraph(["layers.0"]))
    assert files == {str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")}


def test_unoffloaded_module_contributes_nothing(tmp_path):
    model = torch.nn.Module()
    model.layers = torch.nn.Module()
    model.layers.dense = torch.nn.Linear(4, 4)  # real params, not offloaded
    assert subgraph_source_files(model, FakeSubgraph(["layers.dense"])) == set()


# --- last-use planning (the correctness-critical part) ----------------------

def test_plan_last_use_records_the_latest_subgraph(tmp_path):
    """A shard shared by subgraphs 0 and 2 must be attributed to 2, or releasing
    after 0 evicts pages subgraph 2 still needs."""
    model = _model_with_offload(
        tmp_path,
        {"layers.0.proj": "shared.safetensors",
         "layers.1.proj": "only1.safetensors",
         "layers.2.proj": "shared.safetensors"},
    )
    subgraphs = [FakeSubgraph(["layers.0"]), FakeSubgraph(["layers.1"]),
                 FakeSubgraph(["layers.2"])]
    last = plan_last_use(model, subgraphs)
    assert last[str(tmp_path / "shared.safetensors")] == 2
    assert last[str(tmp_path / "only1.safetensors")] == 1


def test_release_does_not_drop_a_file_a_later_subgraph_needs(tmp_path, monkeypatch):
    model = _model_with_offload(
        tmp_path,
        {"layers.0.proj": "shared.safetensors", "layers.2.proj": "shared.safetensors",
         "layers.1.proj": "only1.safetensors"},
    )
    subgraphs = [FakeSubgraph(["layers.0"]), FakeSubgraph(["layers.1"]),
                 FakeSubgraph(["layers.2"])]
    advised: list[tuple[str, int]] = []
    monkeypatch.setattr(
        WeightPrefetcher, "_advise",
        staticmethod(lambda path, advice: advised.append((path, advice)) or True),
    )
    p = WeightPrefetcher(model, subgraphs, enabled=True)
    assert p.enabled

    p.release_through(0)
    dropped = {path for path, adv in advised if adv == _DONTNEED}
    assert str(tmp_path / "shared.safetensors") not in dropped, \
        "shared shard released too early; subgraph 2 still needs it"

    advised.clear()
    p.release_through(2)
    dropped = {path for path, adv in advised if adv == _DONTNEED}
    assert str(tmp_path / "shared.safetensors") in dropped


def test_release_is_idempotent(tmp_path, monkeypatch):
    model = _model_with_offload(tmp_path, {"layers.0.proj": "a.safetensors"})
    subgraphs = [FakeSubgraph(["layers.0"])]
    calls: list = []
    monkeypatch.setattr(WeightPrefetcher, "_advise",
                        staticmethod(lambda p, a: calls.append((p, a)) or True))
    p = WeightPrefetcher(model, subgraphs, enabled=True)
    assert p.release_through(0) == 1
    assert p.release_through(0) == 0, "already-released file re-advised"


# --- prefetch behaviour -----------------------------------------------------

def test_prefetch_advises_willneed(tmp_path, monkeypatch):
    model = _model_with_offload(
        tmp_path, {"layers.0.proj": "a.safetensors", "layers.1.proj": "b.safetensors"})
    subgraphs = [FakeSubgraph(["layers.0"]), FakeSubgraph(["layers.1"])]
    calls: list = []
    monkeypatch.setattr(WeightPrefetcher, "_advise",
                        staticmethod(lambda p, a: calls.append((p, a)) or True))
    p = WeightPrefetcher(model, subgraphs, enabled=True)
    assert p.prefetch(1) == 1
    assert calls == [(str(tmp_path / "b.safetensors"), _WILLNEED)]


def test_prefetch_past_the_end_is_harmless(tmp_path, monkeypatch):
    model = _model_with_offload(tmp_path, {"layers.0.proj": "a.safetensors"})
    subgraphs = [FakeSubgraph(["layers.0"])]
    monkeypatch.setattr(WeightPrefetcher, "_advise", staticmethod(lambda p, a: True))
    p = WeightPrefetcher(model, subgraphs, enabled=True)
    assert p.prefetch(5) == 0


def test_prefetch_depth_covers_multiple_subgraphs(tmp_path, monkeypatch):
    model = _model_with_offload(
        tmp_path,
        {"layers.0.proj": "a.safetensors", "layers.1.proj": "b.safetensors",
         "layers.2.proj": "c.safetensors"},
    )
    subgraphs = [FakeSubgraph([f"layers.{i}"]) for i in range(3)]
    monkeypatch.setattr(WeightPrefetcher, "_advise", staticmethod(lambda p, a: True))
    p = WeightPrefetcher(model, subgraphs, enabled=True, depth=2)
    assert p.prefetch(1) == 2


def test_prefetch_does_not_repeat_resident_files(tmp_path, monkeypatch):
    model = _model_with_offload(tmp_path, {"layers.0.proj": "a.safetensors"})
    subgraphs = [FakeSubgraph(["layers.0"])]
    monkeypatch.setattr(WeightPrefetcher, "_advise", staticmethod(lambda p, a: True))
    p = WeightPrefetcher(model, subgraphs, enabled=True)
    assert p.prefetch(0) == 1
    assert p.prefetch(0) == 0


def test_reprefetch_after_release(tmp_path, monkeypatch):
    """A released file must be re-advisable, or a shard shared by distant
    subgraphs would never be warmed again."""
    model = _model_with_offload(tmp_path, {"layers.0.proj": "a.safetensors"})
    subgraphs = [FakeSubgraph(["layers.0"])]
    monkeypatch.setattr(WeightPrefetcher, "_advise", staticmethod(lambda p, a: True))
    p = WeightPrefetcher(model, subgraphs, enabled=True)
    p.prefetch(0)
    p.release_through(0)
    assert p.prefetch(0) == 1


# --- disabled / inert paths -------------------------------------------------

def test_disabled_is_inert(tmp_path, monkeypatch):
    model = _model_with_offload(tmp_path, {"layers.0.proj": "a.safetensors"})
    monkeypatch.setattr(
        WeightPrefetcher, "_advise",
        staticmethod(lambda p, a: pytest.fail("advised while disabled")),
    )
    p = WeightPrefetcher(model, [FakeSubgraph(["layers.0"])], enabled=False)
    assert not p.enabled
    assert p.prefetch(0) == 0
    assert p.release_through(0) == 0


def test_enabled_but_no_disk_offload_becomes_inert(tmp_path):
    """A cpu/gpu-offloaded or plain model has no files; the prefetcher must
    disable itself rather than pretend."""
    model = torch.nn.Module()
    model.layers = torch.nn.Module()
    model.layers.dense = torch.nn.Linear(4, 4)
    p = WeightPrefetcher(model, [FakeSubgraph(["layers.dense"])], enabled=True)
    assert not p.enabled


def test_no_subgraphs_is_inert(tmp_path):
    model = _model_with_offload(tmp_path, {"layers.0.proj": "a.safetensors"})
    assert not WeightPrefetcher(model, [], enabled=True).enabled


def test_advise_failure_is_not_fatal(tmp_path):
    """fadvise on a vanished/unreadable file must not raise: it is a hint."""
    model = _model_with_offload(tmp_path, {"layers.0.proj": "gone.safetensors"})
    subgraphs = [FakeSubgraph(["layers.0"])]
    p = WeightPrefetcher(model, subgraphs, enabled=True)
    (tmp_path / "gone.safetensors").unlink()
    assert p.prefetch(0) == 0        # counted as not-advised, no exception
    assert p.release_through(0) == 0


def test_depth_is_at_least_one(tmp_path):
    model = _model_with_offload(tmp_path, {"layers.0.proj": "a.safetensors"})
    p = WeightPrefetcher(model, [FakeSubgraph(["layers.0"])], enabled=True, depth=0)
    assert p.depth == 1
