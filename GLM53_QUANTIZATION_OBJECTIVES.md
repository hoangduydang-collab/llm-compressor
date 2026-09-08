# GLM-5.3 quantization: three objectives and executor brief

Updated: 2026-09-09. Owner: planner; full-scale execution: remote executor.
This is the canonical shared brief for the three problems raised by the owner.
Read it before planning or executing GLM-5.3 quantization work. It records scope
and evidence; it is not a ready full-scale launch packet.

## Working principle and responsibilities

**Never build what others have already built.** Investigate this fork, upstream
llm-compressor/compressed-tensors, published implementations and serving-runtime
code before implementing an adapter. Cite the source/revision and explain the
remaining integration gap. Reuse existing solvers, loaders, offload caches and
checkpoint writers wherever possible.

The planner owns research, design, implementation and evidence interpretation,
and can run bounded local diagnostics with a few GPUs. The remote executor owns
representative and full-scale runs on the stronger Rancher cluster. The planner
has no access to that cluster. Do not assume its launcher, mounts, storage or
package versions match local Slurm. Use [PLANNER_EXECUTOR_PROTOCOL.md](PLANNER_EXECUTOR_PROTOCOL.md)
for cross-agent packets; local single-agent diagnostics follow
[FULL_STACK_AGENT_PROTOCOL.md](FULL_STACK_AGENT_PROTOCOL.md).

## 1. Finish expert-parallel distributed GPTQ — current priority

**Owner report:** GLM-5.3 distributed GPTQ ran out of memory on one H100 node.
Inspection implicated the number of experts within one decoder layer. The earlier
EP implementation was unfinished and the full practice run was abandoned because
of risk and time constraints.

**Objective:** complete expert-local Hessian accumulation and solving, retaining
replicated-module handling, calibration coverage, numerical behavior and reliable
collective checkpoint export. Keep the existing GPTQ math. Record phase timing
and memory/I/O evidence while doing this work, to inform objective 3.

**Current evidence:** the implementation reuses the fork's gathered-token EP
adapter and MoEQuant's expert-local/replicated ownership distinction. The initial
CPU regression passed 212 tests (4 CUDA skips); after disk setup/export repairs,
all 10 affected CPU tests pass. All three tiny two-T4 NCCL cases passed in job
830817: DDP/EP parity, weight-only disk save/reload, and mixed FP8-rest disk
save/reload. These are tiny real-GLM fixtures, not proof that the original OOM is
resolved. EP remains opt-in: `quantization.gptq_expert_parallel: true`.

Local H100 job **830962** is queued for this same bounded suite on two full H100s
(last recorded: pending resources; no H100 result yet); see the
[H100 queue contract](docs/glm53-ep-gptq-h100-queue.md). This does not replace the
remote depth-truncated, real-width EP4/EP8 experiment retaining all 256 experts.
Only representative memory/persistence/numerical evidence should unlock a full
run. Preserve calibration manifests and storage/prefetch settings for comparisons.

Read in order:
1. [Implementation and supported scope](docs/glm53-ep-gptq-implementation.md).
2. [GPU failures, repairs and passing evidence](docs/glm53-ep-gptq-gpu-validation.md).
3. [Approved design and prior work](docs/superpowers/specs/2026-09-08-glm53-expert-parallel-gptq-design.md).
4. [Implementation plan and representative gates](docs/superpowers/plans/2026-09-08-glm53-ep-gptq.md).

## 2. Produce the serving-required W4AFP8 formats during quantization

**Owner report:** after the GPTQ OOM, AWQ produced a GLM-5.3 W4AFP8 checkpoint.
Serving with SGLang required several post-quantization conversions because the
pipeline's output did not match the runtime's detailed format requirements.
The owner specifically found that the DSA indexer could not remain BF16 and had
to be converted to FP8. Other modules also needed specific formats/metadata.
The working checkpoint additionally needed an MTP layer-78 graft. MiniMax-M3
similarly needed patches for vLLM serving. These are reported run observations;
the exact contract must be checked against the executor's pinned runtime version.

**Objective:** determine the target serving contract before expensive calibration,
and make the recipe/export produce it as early as supported. Minimize later
re-quantization, conversions, lost accuracy and extra I/O. Treat W4AFP8 as a
runtime-specific checkpoint contract rather than just a recipe label.

**Executor inputs needed:** exact working SGLang revision, serving launch command,
source checkpoint/config, AWQ recipe, conversion/repatch/graft scripts and their
revisions, and the final working checkpoint's key/dtype/shape/quantization metadata.
The successful converted checkpoint is the reference for investigating which
steps can move earlier. Keep source/calibration inputs immutable.

Before implementing, inspect SGLang's actual loader/quantization expectations and
existing exporter/conversion support. Document the required format of routed and
shared experts, attention, indexer, output head, MTP and intentionally ignored
modules, including scales/layout/metadata. Validate the produced checkpoint with
the pinned serving runtime and paired quality checks. Do not claim that the EP
work or the current dynamic W4AFP8 preset already satisfies this objective.

Status: open. The owner authorized a quick implementation alongside objective 1
on 2026-09-09. [Early SGLang alignment](docs/glm53-sglang-w4afp8-early-alignment.md)
adds indexer wk/wq_b to the GLM-5.3 AWQ FP8_BLOCK recipe and preserves native block
FP8 in the existing converter, with exact verification. Direct native export,
expert activation-scale alignment, MTP assembly and runtime/quality qualification
remain open. No full AWQ rerun is authorized by this brief alone.

## 3. Diagnose shared-storage overhead across the entire run

**Owner report:** AWQ was very slow on shared storage on the inaccessible Rancher
cluster, making an expected roughly 10-hour run take roughly 20 hours. The later
breakdown below is more useful than attributing the entire delay to loading.

| Phase | Owner-reported duration | Evidence quality / note |
| --- | --- | --- |
| Load + dispatch | 1 h | Estimate |
| Calibration walk, 78 layers | 13–15 h | Estimate |
| Save, roughly 394.6 GB in 8 shards | 4 h | Estimate |
| Offline gate replay | 30 min | Estimate |
| SGLang-compatible W4AFP8 conversion | 2 h 26 min | Measured |
| DSA-indexer FP8 repatch | 32 min | Measured |
| MTP layer-78 graft | 15 min | Estimate |

These are not a synchronized wall-time trace. Phase overlap and the boundary of
“the 20-hour run” are unknown; do not sum the rows and present that as a measured
end-to-end runtime. Calibration and saving dominate the reported pre-conversion
work; the 1-hour load estimate does not establish loading as the sole bottleneck.
Repeated weight reads during calibration are a hypothesis to test, not a finding.

**Objective:** separate load/dispatch, per-layer calibration, Hessian reduction,
solve/publication, propagation, save, offline replay and conversion costs. Reuse
the phase records added during objective 1. Inspect existing loader/offload,
prefetch/prewarm, storage-staging and save mechanisms before building anything.
Do not assume an apparent storage delay is pure disk latency: include CPU/GPU
utilization, host RAM/page cache, rank skew, communication and cache state.

**Executor evidence needed:** raw per-rank phase records/logs and launch config;
actual mount/storage type and paths; model/shard layout; RAM and node-local storage
capacity; dependency revisions; cache/prefetch settings; available process/cgroup
I/O and memory counters; phase boundaries and whether post-conversion time was
included. `pipeline.metrics.summarize_phases(paths)` aggregates explicit rank
files. Keep missing counters unavailable; per-process read counts are not unique
shared-storage traffic, and nested timers must not be added together.

Status: timing instrumentation implemented; representative GPTQ evidence and
Rancher AWQ evidence still needed. Storage tuning must preserve calibration and
quality controls. Coordinate format changes with objective 2 to eliminate work
rather than merely accelerating avoidable conversions.

## Executor starting checklist

- Read this brief and the current linked implementation/evidence before proposing
  a run. A local Slurm diagnostic is not a Rancher launch packet.
- Return the requested runtime/storage/script facts through the repository's
  handoff process. The planner then supplies a concrete bounded execution packet.
- Keep tiny correctness, representative memory/performance and full-run quality
  conclusions separate. Preserve failures as well as passes.
- Current sequencing: **objective 1 first; collect evidence for 3 alongside it;
  objective 2 remains a separate serving-format workstream.**
