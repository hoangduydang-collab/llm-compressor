"""Memory behaviour of the distributed Hessian reduce under ``offload_hessians``.

The bug these tests pin down: ``_reduce_hessian_to_target_rank`` issued one
async ``dist.reduce`` per module and waited only after the loop. An onloaded
Hessian stays resident until its reduce is waited on -- NCCL holds a reference
that neither popping the dict entry nor the move-back to CPU can release -- so
the loop re-materialized the *entire* Hessian set on the accelerator and
``offload_hessians=True`` bought nothing. On GLM-5.2 (256 routed experts, hidden
6144) that is 76 GiB, and it OOM'd on a 144 MiB onload with 4.12 MiB free.

Tensors here are deliberately stand-ins (``FakeHessian``) rather than real
``torch`` tensors. The logic under test is device bookkeeping -- what is
resident, when it is released, in what order collectives are issued -- and a
CPU-only box cannot express "onloaded to an accelerator" with real tensors:
``meta`` cannot be copied back out, and ``cpu``-to-``cpu`` is unobservable. The
stand-in makes residency directly observable, which real tensors would not.
"""

import pytest

from llmcompressor.modifiers.gptq.base import (
    _HESSIAN_REDUCE_ONLOAD_BYTES,
    GPTQModifier,
    _chunk_by_bytes,
)

MiB = 1024**2
EXEC_DEVICE = "cuda:0"


class FakeHessian:
    """Records which device it currently lives on, without moving anything."""

    def __init__(self, nbytes, device=EXEC_DEVICE):
        self._nbytes = nbytes
        self.device = device

    def to(self, device):
        return FakeHessian(self._nbytes, device)

    def numel(self):
        return self._nbytes  # element_size() == 1, so numel == nbytes

    def element_size(self):
        return 1


class FakeWork:
    """Stand-in for a ``dist`` work handle that keeps its tensor alive."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.waited = False

    def wait(self):
        self.waited = True


class FakeDist:
    """Minimal ``torch.distributed`` stand-in that records the reduce schedule.

    ``windows`` is the list of in-flight batches: each entry is the list of
    reduce calls issued since the previous ``wait_for_comms``. That is exactly
    the quantity the bug is about -- everything in one window is simultaneously
    resident.
    """

    class ReduceOp:
        SUM = "sum"

    def __init__(self, rank=0, world_size=4):
        self._rank = rank
        self._world_size = world_size
        self.windows = [[]]

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size

    def reduce(self, tensor, op, dst, async_op):
        assert async_op is True
        assert op is FakeDist.ReduceOp.SUM
        self.windows[-1].append({"tensor": tensor, "dst": dst})
        return FakeWork(tensor)

    # --- observations -------------------------------------------------
    def close_window(self):
        self.windows.append([])

    def hessian_windows(self):
        """Reduce windows, keeping only the Hessian (not num_samples) calls."""
        return [
            [c for c in w if isinstance(c["tensor"], FakeHessian)]
            for w in self.windows
            if w
        ]

    def peak_window_bytes(self):
        return max(
            (sum(c["tensor"]._nbytes for c in w) for w in self.hessian_windows()),
            default=0,
        )


class Harness:
    """Carries only the attributes the reduce path touches.

    ``GPTQModifier`` is a pydantic model whose construction pulls in a full
    quantization config; the methods under test read three attributes and
    nothing else, so they are invoked unbound against this instead. That keeps
    the test aimed at the memory logic rather than at recipe validation.
    """

    def __init__(self, hessians, num_samples, offload_hessians):
        self._hessians = hessians
        self._num_samples = num_samples
        self.offload_hessians = offload_hessians

    reduce_to_target = GPTQModifier._reduce_hessian_to_target_rank
    # The outer method dispatches to this one, so it must be bound too.
    _reduce_hessian_chunk = GPTQModifier._reduce_hessian_chunk


@pytest.fixture
def patched(monkeypatch):
    """Install the fakes and return the FakeDist so tests can inspect it."""
    fake = FakeDist()
    monkeypatch.setattr("llmcompressor.modifiers.gptq.base.dist", fake)
    monkeypatch.setattr(
        "llmcompressor.modifiers.gptq.base.get_execution_device",
        lambda module: EXEC_DEVICE,
    )

    def wait_for_comms(pending):
        for work in pending:
            work.wait()
        pending.clear()
        fake.close_window()

    monkeypatch.setattr(
        "llmcompressor.modifiers.gptq.base.wait_for_comms", wait_for_comms
    )
    return fake


def build(n_modules, nbytes, world_size=4, offload=True, device=None):
    """A layer of ``n_modules`` equal-sized Hessians, round-robin owned."""
    modules = [f"expert.{i}" for i in range(n_modules)]
    start = device or ("cpu" if offload else EXEC_DEVICE)
    hessians = {m: FakeHessian(nbytes, start) for m in modules}
    num_samples = {m: object() for m in modules}
    module_to_rank = {m: i % world_size for i, m in enumerate(modules)}
    return modules, module_to_rank, Harness(hessians, num_samples, offload)


# --------------------------------------------------------------------------
# _chunk_by_bytes: the deterministic core
# --------------------------------------------------------------------------


def test_chunk_respects_budget():
    mods = [f"m{i}" for i in range(10)]
    hessians = {m: FakeHessian(100) for m in mods}
    chunks = list(_chunk_by_bytes(mods, hessians, 250))
    assert [len(c) for c in chunks] == [2, 2, 2, 2, 2]
    assert [m for c in chunks for m in c] == mods


def test_chunk_preserves_order_across_uneven_sizes():
    """Order is the deadlock-safety property: all ranks must agree on it."""
    mods = [f"m{i}" for i in range(6)]
    sizes = [10, 500, 10, 10, 500, 10]
    hessians = {m: FakeHessian(s) for m, s in zip(mods, sizes)}
    chunks = list(_chunk_by_bytes(mods, hessians, 520))
    assert [m for c in chunks for m in c] == mods
    for chunk in chunks:
        assert sum(hessians[m]._nbytes for m in chunk) <= 520


def test_chunk_emits_oversized_hessian_alone_rather_than_dropping_it():
    """The budget yields to correctness: nothing may be silently skipped."""
    mods = ["small", "huge", "small2"]
    hessians = {"small": FakeHessian(10), "huge": FakeHessian(10_000),
                "small2": FakeHessian(10)}
    chunks = list(_chunk_by_bytes(mods, hessians, 100))
    assert chunks == [["small"], ["huge"], ["small2"]]


def test_chunk_of_empty_module_list_yields_nothing():
    assert list(_chunk_by_bytes([], {}, 100)) == []


def test_chunk_is_identical_for_every_rank():
    """Same inputs -> same boundaries, which is what keeps ranks in lockstep."""
    mods = [f"m{i}" for i in range(50)]
    sizes = [(i * 37) % 500 + 1 for i in range(50)]
    hessians = {m: FakeHessian(s) for m, s in zip(mods, sizes)}
    runs = [
        [list(c) for c in _chunk_by_bytes(mods, hessians, 1000)] for _ in range(4)
    ]
    assert runs[0] == runs[1] == runs[2] == runs[3]


# --------------------------------------------------------------------------
# The regression itself
# --------------------------------------------------------------------------


def test_offloaded_reduce_keeps_peak_under_the_budget(patched):
    """GLM-5.2's module count: this is the case that OOM'd at 77.36 GiB."""
    # 256 routed experts x 3 projections = 768 modules. Sized uniformly at
    # gate_proj's 144 MiB rather than the real 144/144/16 mix, so the total is
    # 108 GiB instead of 76 -- a deliberate overshoot of the observed failure.
    modules, module_to_rank, harness = build(768, 144 * MiB)
    harness.reduce_to_target(modules, module_to_rank)

    peak = patched.peak_window_bytes()
    assert peak <= _HESSIAN_REDUCE_ONLOAD_BYTES
    # Guard against the fix degenerating into "one window per module", which
    # would bound memory but serialize every collective.
    assert len(patched.hessian_windows()) > 1
    assert max(len(w) for w in patched.hessian_windows()) > 1


def test_offloaded_reduce_covers_every_module_exactly_once_in_order(patched):
    modules, module_to_rank, harness = build(40, 512 * MiB)
    harness.reduce_to_target(modules, module_to_rank)

    reduced = [c["tensor"] for w in patched.hessian_windows() for c in w]
    assert len(reduced) == len(modules)
    # Each was onloaded before its reduce: an offloaded tensor would be
    # reduced on the wrong device and silently produce garbage.
    assert all(t.device == EXEC_DEVICE for t in reduced)


def test_offloaded_reduce_targets_the_owning_rank(patched):
    modules, module_to_rank, harness = build(40, 512 * MiB)
    harness.reduce_to_target(modules, module_to_rank)

    seen = [c for w in patched.hessian_windows() for c in w]
    for module, call in zip(modules, seen):
        assert call["dst"] == module_to_rank[module]


def test_non_owned_hessians_are_released_and_owned_ones_return_to_cpu(patched):
    modules, module_to_rank, harness = build(40, 512 * MiB)
    harness.reduce_to_target(modules, module_to_rank)

    owned = [m for m in modules if module_to_rank[m] == 0]
    assert set(harness._hessians) == set(owned)
    assert set(harness._num_samples) == set(owned)
    # Released back to CPU, or the offload is undone for the rest of the layer.
    assert all(harness._hessians[m].device == "cpu" for m in owned)


def test_release_happens_only_after_the_collective_is_waited_on(monkeypatch):
    """Freeing mid-flight reclaims nothing and would be a use-after-free."""
    fake = FakeDist()
    monkeypatch.setattr("llmcompressor.modifiers.gptq.base.dist", fake)
    monkeypatch.setattr(
        "llmcompressor.modifiers.gptq.base.get_execution_device",
        lambda module: EXEC_DEVICE,
    )
    pops_before_wait = []
    handles = []

    def wait_for_comms(pending):
        for work in pending:
            work.wait()
        handles.extend(pending)
        pending.clear()
        fake.close_window()

    monkeypatch.setattr(
        "llmcompressor.modifiers.gptq.base.wait_for_comms", wait_for_comms
    )

    modules, module_to_rank, harness = build(40, 512 * MiB)

    class Watched(dict):
        def pop(self, key, *default):
            if any(not h.waited for h in handles):
                pops_before_wait.append(key)
            return super().pop(key, *default)

    harness._hessians = Watched(harness._hessians)
    harness._num_samples = Watched(harness._num_samples)
    harness.reduce_to_target(modules, module_to_rank)

    assert pops_before_wait == []
    assert handles and all(h.waited for h in handles)


# --------------------------------------------------------------------------
# The non-offloaded path must be untouched (it is what validated MiniMax-M3)
# --------------------------------------------------------------------------


def test_non_offloaded_path_still_reduces_in_a_single_window(patched):
    modules, module_to_rank, harness = build(768, 144 * MiB, offload=False)
    harness.reduce_to_target(modules, module_to_rank)

    windows = patched.hessian_windows()
    assert len(windows) == 1
    assert len(windows[0]) == len(modules)


def test_non_offloaded_path_never_moves_a_hessian(patched):
    modules, module_to_rank, harness = build(20, 144 * MiB, offload=False)
    harness.reduce_to_target(modules, module_to_rank)

    owned = [m for m in modules if module_to_rank[m] == 0]
    assert set(harness._hessians) == set(owned)
    # No stray trip to CPU: without offloading these must stay put.
    assert all(harness._hessians[m].device == EXEC_DEVICE for m in owned)
