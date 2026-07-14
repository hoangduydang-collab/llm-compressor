# MiniMax-M3 Native Distributed Quantization Speed-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing MiniMax-M3 pipeline to use llm-compressor's native distributed GPTQ/AWQ and compressed-tensors shared offload, then provide a bounded 8xH100 smoke that produces decision-grade evidence.

**Architecture:** `pipeline.run` owns the distributed process lifecycle and one shared run directory. `pipeline.calibration` partitions the existing global calibration corpus before `oneshot`; `pipeline.quantize` keeps native recipes/model loading while making evidence and post-save writes rank-safe. A single-node `srun` wrapper launches `torchrun` for GPTQ and AWQ smokes sequentially.

**Tech Stack:** Python 3.11, PyTorch distributed, compressed-tensors offload, llm-compressor GPTQ/AWQ modifiers, Hugging Face Datasets, Bash, Slurm `srun`, pytest.

## Global Constraints

- Stay on `duy-branch`; do not create or switch branches.
- Reuse native llm-compressor/compressed-tensors distributed mechanisms; do not revive `pipeline/expert_scatter.py`, `pipeline/ep_moe.py`, or bespoke MoE quantization.
- The target cluster supports `srun`, not `sbatch`.
- Phase 1 runs GPTQ and AWQ partial-layer smokes only; full calibration is gated on the paired quality evaluation.
- Partial-layer output is evidence only and must not be presented as a usable checkpoint.
- Preserve the existing single-process `pipeline.run` behavior.
- Local planner verification may not require `torch`; cluster verification must use the executor's quantization environment and 8xH100 node.

---

### Task 1: Pipeline Distributed Lifecycle

**Files:**
- Create: `pipeline/distributed.py`
- Modify: `pipeline/run.py`
- Test: `pipeline/tests/test_distributed.py`

**Interfaces:**
- Produces: `DistributedContext.from_environment()`, `DistributedContext.broadcast_path(path)`, `DistributedContext.barrier()`, `DistributedContext.close()`, and `DistributedContext.rank_path(path)`.
- Consumes: `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` set by `torchrun`; `compressed_tensors.offload.init_dist`; `torch.distributed` after lazy import.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_single_process_context_is_noop(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    ctx = DistributedContext.from_environment()
    assert (ctx.enabled, ctx.rank, ctx.world_size, ctx.is_source) == (False, 0, 1, True)


def test_rank_path_suffixes_distributed_evidence():
    ctx = DistributedContext(enabled=True, rank=3, world_size=8, local_rank=3)
    assert ctx.rank_path(Path("run/quant_metrics.jsonl")) == Path(
        "run/quant_metrics.rank-3.jsonl"
    )
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest -q pipeline/tests/test_distributed.py`

Expected: collection fails because `pipeline.distributed` does not exist.

- [ ] **Step 3: Implement the minimal lazy-import context**

```python
@dataclass
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    _owns_process_group: bool = False

    @classmethod
    def from_environment(cls) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1:
            return cls()
        from compressed_tensors.offload import init_dist
        import torch.distributed as dist
        owned = not dist.is_initialized()
        if owned:
            init_dist()
        return cls(True, dist.get_rank(), dist.get_world_size(),
                   int(os.environ.get("LOCAL_RANK", dist.get_rank())), owned)
```

`broadcast_path` broadcasts a one-element object list from rank 0. `close` destroys
only a process group initialized by this context. All torch/compressed-tensors
imports remain inside distributed-only methods so CPU config/launcher tests collect
without those packages.

- [ ] **Step 4: Integrate the context into `pipeline.run`**

Move the existing post-argument-parsing body into `_run(args, dist_ctx)`. Rank 0
creates the timestamped run directory and broadcasts it; only rank 0 writes shared
config and metadata. Wrap `_run` in `try/finally` so `dist_ctx.close()` always runs.

- [ ] **Step 5: Run tests and confirm GREEN**

Run: `python -m pytest -q pipeline/tests/test_distributed.py`

Expected: all lifecycle tests pass without importing torch in single-process mode.

### Task 2: Deterministic Rank-Local Calibration Data

**Files:**
- Modify: `pipeline/calibration.py`
- Modify: `pipeline/quantize.py`
- Test: `pipeline/tests/test_calibration_partition.py`

**Interfaces:**
- Produces: `partition_bounds(num_samples: int, rank: int, world_size: int) -> tuple[int, int]` and `calibration_partition() -> CalibrationPartition`.
- Consumes: initialized `torch.distributed` only through lazy rank/world-size lookup.

- [ ] **Step 1: Write failing pure partition tests**

```python
def test_partition_bounds_cover_nondivisible_global_set():
    bounds = [partition_bounds(10, rank, 3) for rank in range(3)]
    assert bounds == [(0, 3), (3, 6), (6, 10)]


def test_partition_bounds_cover_fewer_samples_than_ranks():
    bounds = [partition_bounds(3, rank, 4) for rank in range(4)]
    assert bounds == [(0, 0), (0, 1), (1, 2), (2, 3)]
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest -q pipeline/tests/test_calibration_partition.py`

Expected: import fails because the partition helpers do not exist.

- [ ] **Step 3: Implement global-shuffle-then-select semantics**

`build_calibration_dataset` continues to load exactly `dataset_split[:num_samples]`
and shuffle with the configured seed. When distributed is initialized, select
`range(start, end)` from that shuffled dataset before chat templating/tokenization.
Return the Dataset as today and expose a compact manifest containing global count,
local count, bounds, rank/world size, and a SHA-256 over local token IDs.

- [ ] **Step 4: Pass the local count to native `oneshot`**

In `run_quantize`, set `num_calibration_samples=len(ds)` so the native sampler does
not reinterpret the global count on a rank-local dataset. Persist each manifest as
`calibration_partition.rank-<rank>.json`; use the unsuffixed name in single-process
mode.

- [ ] **Step 5: Run tests and confirm GREEN**

Run: `python -m pytest -q pipeline/tests/test_calibration_partition.py`

Expected: all partition and manifest tests pass.

### Task 3: Rank-Safe Quantization Evidence and Saving

**Files:**
- Modify: `pipeline/quantize.py`
- Modify: `pipeline/versioning.py`
- Test: `pipeline/tests/test_distributed_quantize_contract.py`

**Interfaces:**
- Consumes: `DistributedContext` passed from `pipeline.run` to `run_quantize`.
- Produces: per-rank metrics/provenance, collective checkpoint save, source-only post-processing, and aggregate distributed summary.

- [ ] **Step 1: Write failing contract tests**

Tests inspect a fake `DistributedContext` and assert that rank 3 resolves
`quant_metrics.rank-3.jsonl`, `model_provenance.rank-3.json`, and
`calibration_partition.rank-3.json`, while a single-process context preserves the
legacy filenames. A source/non-source test verifies only rank 0 performs tokenizer,
VL-artifact, config-patch, recipe, and shared-metadata writes.

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest -q pipeline/tests/test_distributed_quantize_contract.py`

Expected: failure because `run_quantize` has no distributed context contract.

- [ ] **Step 3: Implement rank-aware evidence paths**

Change the signature to:

```python
def run_quantize(
    cfg: PipelineConfig,
    run_dir: Path,
    dist_ctx: DistributedContext | None = None,
    *,
    save_checkpoint: bool = True,
) -> Path:
```

Use `dist_ctx or DistributedContext()` for backward compatibility. Every rank calls
`oneshot` and captures its own metrics. Every rank participates in
`model.save_pretrained`; then barrier. Only source rank saves tokenizer/processor,
patches the serialized config, writes recipe/shared metadata, and prints the shared
checkpoint completion line. Barrier again before return.

- [ ] **Step 4: Support evidence-only partial-layer smoke**

When `save_checkpoint=False`, skip all model/tokenizer/checkpoint post-processing,
write a source `smoke_complete.json` only after the final barrier, and return the
would-be checkpoint path without creating a usable checkpoint.

Add `pipeline.run --evidence-only` as an explicit `store_true` CLI flag. Reject it
unless the selected stage includes quantization, and pass
`save_checkpoint=not args.evidence_only` to `run_quantize`. The smoke launcher must
always include this flag; production launchers remain unchanged.

- [ ] **Step 5: Run tests and confirm GREEN**

Run: `python -m pytest -q pipeline/tests/test_distributed_quantize_contract.py`

Expected: all rank-path, write-ownership, barrier, and no-checkpoint tests pass.

### Task 4: Representative-Layer Targets and `srun` Smoke

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pipeline/recipe.py`
- Create: `pipeline/configs/minimax_m3_distributed_smoke.yaml`
- Create: `pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh`
- Test: `pipeline/tests/test_m3_distributed_quant_smoke.py`
- Modify: `M3_QUANT_SPEEDUP_PLAN.md`

**Interfaces:**
- Produces: `QuantizationConfig.targets`, native recipe target forwarding, and one executor command that runs GPTQ then AWQ on the same 8-GPU node.
- Consumes: exact MiniMax decoder module regexes for layers 3, 31, and 59.

- [ ] **Step 1: Write failing recipe/config/launcher tests**

Tests assert:

```python
assert config.quantization.targets == [
    "re:.*layers\\.(3|31|59)\\..*",
]
assert "srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8" in dry_run
assert "torchrun --nproc_per_node=8 -m pipeline.run" in dry_run
assert "--evidence-only" in dry_run
assert "sbatch" not in dry_run
assert dry_run.index("quantization.method=gptq") < dry_run.index(
    "quantization.method=awq"
)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest -q pipeline/tests/test_m3_distributed_quant_smoke.py`

Expected: tests fail because targets/config/launcher do not exist.

- [ ] **Step 3: Forward configurable targets into native recipes**

Add `targets: str | list[str] = "Linear"` to `QuantizationConfig`. Replace hardcoded
GPTQ/QuantizationModifier/AutoRound targets in `pipeline.recipe` with the configured
value and include it in `describe_recipe`. Do not alter modifier implementations.

- [ ] **Step 4: Add bounded smoke configuration and launcher**

The smoke config inherits production model/calibration settings explicitly but uses
8 small calibration samples at sequence length 512, disables sample generation and
serving, targets only layers 3/31/59, sets `device_map: auto_offload`, and uses a
caller-provided offload/output root. The launcher:

```bash
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --time="$TIME_LIMIT" \
  bash -lc "torchrun --nproc_per_node=8 -m pipeline.run ..."
```

It runs GPTQ and AWQ sequentially so one node never holds two MiniMax copies. It
captures stdout/stderr, `/usr/bin/time -v`, `nvidia-smi`, `/proc/meminfo`, exit code,
resolved command, git SHA, and package versions beneath one run root. It rejects a
nested Slurm allocation and contains no `sbatch`.

- [ ] **Step 5: Run tests and confirm GREEN**

Run: `python -m pytest -q pipeline/tests/test_m3_distributed_quant_smoke.py`

Expected: config and dry-run launcher contract tests pass.

### Task 5: Verification, Review, and Executor Handoff

**Files:**
- Modify: `M3_QUANT_SPEEDUP_PLAN.md`
- Create: `M3_DISTRIBUTED_QUANT_SPEEDUP_HANDOFF.md`

**Interfaces:**
- Produces: exact executor command, acceptance checklist, expected artifacts, and escalation rules.

- [ ] **Step 1: Run all locally available tests**

Run:

```bash
python -m pytest -q \
  pipeline/tests/test_distributed.py \
  pipeline/tests/test_calibration_partition.py \
  pipeline/tests/test_distributed_quantize_contract.py \
  pipeline/tests/test_m3_distributed_quant_smoke.py \
  pipeline/tests/test_m3_guarded_full_launcher.py \
  pipeline/tests/test_m3_awq_representative_launcher.py \
  pipeline/tests/test_m3_safe_diagnostic_full_launcher.py
bash -n pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh
```

Expected: all CPU tests pass and Bash syntax exits 0. Record torch-dependent tests as
executor-required when the planner environment lacks torch.

- [ ] **Step 2: Self-review the diff against the approved design**

Confirm single-process compatibility, no bespoke quantization code, no `sbatch`,
rank-local data/evidence, source-owned shared writes, collective save behavior,
failure-safe cleanup, and no usable partial checkpoint.

- [ ] **Step 3: Request focused code review and resolve findings**

Review the complete diff against this plan. Fix every Critical/Important issue and
rerun the focused tests.

- [ ] **Step 4: Commit and push `duy-branch`**

```bash
git add M3_QUANT_SPEEDUP_PLAN.md M3_DISTRIBUTED_QUANT_SPEEDUP_HANDOFF.md \
  docs/superpowers/plans/2026-07-15-m3-native-distributed-quantization-speedup.md \
  pipeline
git commit -m "feat: enable native distributed M3 quantization smoke"
git push origin duy-branch
```

- [ ] **Step 5: Executor cluster verification**

Executor runs the handoff command in the quantization environment, first with
`DRY_RUN=1`, then on one 8xH100 node. The executor returns raw rank logs, resolved
config, partition manifests, timing/memory samples, metrics, process exit codes, and
an aggregate comparison against historical single-process layer timings. No full
calibration begins until the planner accepts the evidence and the paired quality gate
selects a recipe.
