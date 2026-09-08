# GLM-5.3 expert-parallel GPTQ with phase timing

Date: 2026-09-08
Status: Draft; prior-work review expanded; implementation source selection pending; PLANNER_ANALYSIS
Repository baseline: 8945c64c3c9e7e01cdccc5c4c2e4c639e56dabac
Owner: planner; full-scale execution belongs to the remote execution agent.

## Decision and scope

Working direction: integrate opt-in expert-parallel (EP) GPTQ into the existing
sequential pipeline. The runtime contract below is provisional until the reuse
comparison selects the source implementation; it is not a decision to invent
a new EP dispatcher.
Measure its execution phases in the same runs, so later storage work has evidence.
The initial supported workload is single-node GLM-5.3, decoder-layer sequencing,
data-parallel calibration with all experts calibrated, and one EP group spanning
the ranks. Standard distributed GPTQ and AWQ keep their existing defaults.

This is an implementation design, not a cluster execution packet. It does not
authorize a GPU allocation. The eventual packet will name exact revisions,
inputs, environment, resource limits, commands, numerical gates, and output paths.

## Prior work verified on 2026-09-08

Inspected this fork at the baseline above, its history, and a separate upstream
checkout at `a1fbffce0af83fc4238ed5e3c7482d50959e408d`. Upstream issues/PRs below
were checked for actual merge status, not just whether they were closed.
No turnkey EP GPTQ integration was found in the inspected upstream source.
This is a bounded search finding, not a claim that no other implementation exists.

| Existing work | Verified status and relevance | Reuse decision |
| --- | --- | --- |
| [Distributed GPTQ #2333](https://github.com/vllm-project/llm-compressor/pull/2333), [distributed compression RFC #2180](https://github.com/vllm-project/llm-compressor/issues/2180) | Merged infrastructure already present here: rank-local calibration, Hessian reduction to assigned solver ranks, parallel solves, result broadcast. Hessians are still collected for all experts on each rank before reduction. | Keep the quantizer, replicated-module path, offload and export; this alone does not remove GLM's collection-time OOM. |
| [DeepSeek integration #1535](https://github.com/vllm-project/llm-compressor/pull/1535#issuecomment-2960390534) | Integration merged. Maintainer separately proposed dispatching one sequential target across GPUs plus asynchronous GPTQ. That comment is a proposal, not evidence that the merged PR delivered EP. | Prior architectural rationale for splitting a decoder layer when it cannot fit with Hessians. |
| [Sharded-expert GPTQ hang #2734](https://github.com/vllm-project/llm-compressor/issues/2734) | Open public attempt on DeepSeek-V4-Flash. Disjoint expert/module sets break Hessian collectives; observer synchronization is also implicated. Author reported progress after selective bypasses, not a completed validated checkpoint. The reproduction references an older revision. | Use the reported failure cases to design ownership/collective regression tests; do not copy blanket observer-sync suppression. |
| Fork GLM EP commits `416aaa0f`, `32e310e9`, `8f0b6684` | Ownership, gathered-token expert forward, and LinearExperts2D hook exist. Tiny forward tests do not establish complete GPTQ/offload/export integration. | Preserve useful tested adapters; compare them to published dispatch before extending them. |
| Fork M3 EP commit `dedcb67f`, `pipeline/ep_moe.py`, `M3_QUANT_SPEEDUP_PLAN.md` | Earlier MoEQuant-inspired dispatch/combine prototype was shelved when existing DDP proved sufficient for M3. The plan explicitly supersedes its old EP sections. | Read for reusable code/tests; do not revive the old M3 implementation wholesale. GLM's subsequent per-layer capacity failure requires a new decision. |
| [Weight-calibration parallelism #2785](https://github.com/vllm-project/llm-compressor/pull/2785) | Merged; module-parallel weight calibration, not EP activation forwarding. | Existing infrastructure, not a solution to replicated expert Hessians. |

### Published MoEQuant implementation versus the fork

[MoE-Quant](https://github.com/IST-DASLab/MoE-Quant) is public. Its
[quant.py](https://github.com/IST-DASLab/MoE-Quant/blob/master/quant.py) configures
`ep_size`, treats expert GPTQ handles as local, and reduces replicated handles.
Its expert dispatch is supplied by the public
[DeepSeek-V3 model implementation](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py),
which partitions experts and uses all-to-all token dispatch/combine. It also
supports sharing gate/up Hessians. We should assess this complete execution path,
not merely cite its ideas and then build an equivalent from scratch.

It is not a drop-in GLM runner: the inspected entrypoint asserts DeepSeek-V3,
assumes a particular 163-shard checkpoint naming layout, and slices calibration
samples using floor division. Its Hessian normalization, fallback choices, and
optional optimized solver differ from this fork. DeepSeek's routed-token forward
also differs from our current all-expert calibration policy. Those are concrete
adapter and numerical-comparison requirements, not reasons to disregard the code.

### Offload and loading work relevant to both problems 1 and 3

- [Layerwise pipeline #2748](https://github.com/vllm-project/llm-compressor/pull/2748)
  was closed **without merging** after its author acknowledged that existing
  sequential processing plus disk offloading covered the proposed use case.
  Reuse these facilities; do not build a second streaming loader merely because
  the checkpoint exceeds host memory. Incremental saving and measured CephFS
  performance still require separate evidence.
- [No-copy 3-D loading #2941](https://github.com/vllm-project/llm-compressor/pull/2941)
  is closed **unmerged**; its prerequisite
  [compressed-tensors #786](https://github.com/vllm-project/compressed-tensors/pull/786)
  is open. These target temporary copies when linearizing fused expert tensors.
  They are useful code references, not features we can assume are installed.
- [Memory discussion #2949](https://github.com/vllm-project/llm-compressor/issues/2949)
  corrected the claim that standard distributed offloading keeps a complete CPU
  copy per rank: shared storage already exists. Do not introduce another IPC layer.
- [Qparameter writeback #3066](https://github.com/vllm-project/llm-compressor/pull/3066)
  is open and reports broadcasts updating temporary onloaded tensors without
  persisting them. **This fork already addresses that class of failure** in
  `_broadcast_quantized_params`, commit `25a0324d`, using
  `update_offload_parameter` after communication completes. Retain that fix and
  its tests; compare upstream's expanded coverage rather than duplicating it.
  This inspection does not establish that every EP/offload variant is covered.

## Reuse and alternatives

1. Working preference: retain llm-compressor's quantizer, sequential pipeline,
   offload and export; adapt published EP ownership/dispatch where compatible.
   Compare DeepSeek/MoEQuant's runtime against the existing GLM prototype before
   choosing which dispatcher to integrate. Implement only the missing GLM,
   calibration-policy and lifecycle adapters.
2. Adapt the complete MoEQuant runner to GLM. This provides an established EP
   path but requires replacing DeepSeek/checkpoint assumptions and validating
   calibration semantics, quantization math and our export contract. Compare
   the actual adapter surface before rejecting or selecting it.
3. Use existing memory-bounded mechanisms without EP: expert-level sequential
   partitioning or CPU Hessian offload. Recorded partitioning interleaved decoder
   layers; recorded CPU transfers were expensive. Keep small controls, but neither
   is currently a demonstrated full-run answer.

Before implementation, make a source-to-change map for dispatch, ownership,
Hessian collection/normalization, observer synchronization, offload persistence,
and checkpoint save. For each new change name the existing source reused and
why an existing implementation cannot satisfy the remaining requirement.
Do not add custom communication primitives or another loader without this check.

Existing integration points:
- src/llmcompressor/modeling/moe/expert_parallel.py
- src/llmcompressor/modeling/moe/linear_experts.py
- src/llmcompressor/modifiers/gptq/base.py
- src/llmcompressor/pipelines/sequential/pipeline.py
- pipeline/calibration.py, pipeline/distributed.py, pipeline/metrics.py
- existing EP, offloaded reduction, qparameter persistence, and save tests.

## Execution and ownership contract

Keep the full module tree and shared CPU/disk offload indexes. Assign each routed
expert exactly one owner using the existing deterministic contiguous assignment.
Do not prune nonowned modules out of the model: tracing, metadata, observers, and
collective save currently expect a common structure.

Trace the usual decoder-layer graph before activating runtime EP. Check that ranks
agree on graph/module manifests. During calibration and post-quant propagation:
- Gather the rank-local token inputs and routing metadata for the current step.
- Only the owner executes each routed expert and accumulates its Hessians.
- Combine routed outputs and return each rank's original token slice.
- Keep attention and shared experts data-parallel.
- Keep EP active across both forwards; restore context on every exit.

Validate equal, nonzero dataloader step counts collectively before the walk.
Variable token lengths within a step are supported by the existing padded gather.
Initially reject unequal step counts rather than silently dropping or duplicating
samples; production 256-sample runs on 2/4/8 ranks can satisfy this contract.
Uneven expert counts remain supported by the assignment function.

Preserve GPTQ normalization as well as token coverage. The existing accumulator
counts a 2-D expert input as one batch contribution. Concatenating W ranks into
one 2-D input must not silently count one contribution where the DDP reference
counted W. Carry the reference contribution count explicitly, and compare raw
Hessians, counts, and normalized Hessians in tests.

Partition compression into:
- replicated targets: existing deterministic reduce and solve behavior;
- routed targets: solve on the expert owner, without reducing an already-complete
  Hessian to a different rank.

Publish updated weights and qparameters in a globally agreed order through the
existing offload-aware update mechanisms. Use bounded transfer windows; do not
onload the entire expert set merely to synchronize it. All ranks must enter any
DistributedDiskCache update barriers in the same order. Validate observer lifecycle
collectives too; do not assume all quantization hooks are rank-local.

Retain the current collective save path and verify that rank zero exports experts
it did not own. No new loader, distributed file writer, or inference kernel is
part of this first change. Save/propagation may still synchronize weights across
ranks; measure that cost instead of promising complete weight sharding everywhere.

For the documented GLM shapes, routed Hessian storage is approximately 76 GiB
without EP, 19 GiB at EP4, and 9.5 GiB at EP8. These exclude weights, gather
buffers, masks, solver scratch, and allocator overhead. They are capacity estimates,
not measured total VRAM. Gate/up Hessian sharing is an established follow-up
optimization if the measured budget calls for it, not a prerequisite for this design.

## Phase evidence

Extend existing per-rank JSONL and provenance artifacts; do not build a profiling
framework. Record paired start/end events, monotonic wall duration, rank, subgraph
and decoder-layer identity, status, and memory snapshots. Preserve unfinished phase
records when a process fails. Missing evidence must never appear as zero duration.

Record:
- model load plus dispatch, with subspans only where the loader exposes a reliable boundary;
- dataset preparation and tracing;
- each layer's calibration forward/Hessian accumulation;
- replicated Hessian reduction, expert solve, and weight/qparameter synchronization;
- each layer's quantized propagation and offload transition;
- save prewarm, collective checkpoint save, and offline verification.

Extend the existing layer-name parser, which currently recognizes M3's
language_model.layers names but misses GLM's model.layers names.

Capture per-rank allocated/reserved VRAM peaks, host memory, and timed process I/O
counter snapshots. Capture pod/cgroup memory, page-cache pressure, and I/O counters
once per node when available. Process counters do not directly measure unique
CephFS traffic; shared page-cache and cgroup counters must not be summed per rank.

Ordinary wall timings include asynchronous work and overlap. Use CUDA events or
bounded profiler traces for a representative diagnostic when compute/communication
attribution is needed; do not synchronize every operation in production. Report
critical-path elapsed time separately from summed per-rank work and nested spans.
Never derive pure disk latency by subtracting overlapping GPU/host timers.

Keep prefetch and storage placement constant within each EP comparison. A later
small storage comparison can use the same code and inputs on local versus shared
storage, with cache state recorded. GPTQ results inform shared load/save behavior;
they cannot establish AWQ grid-search costs.

## Validation sequence and acceptance

A. Local small-model tests:
- retain the existing forward tests and add real multiprocess CPU collective tests;
- compare exact sample coverage and counts, and tolerance-based Hessians/outputs;
- exercise variable token lengths, uneven expert ownership, unsupported unequal
  step counts, context cleanup, replicated modules, and observer synchronization;
- run actual GPTQ, save, reload, and verify nonzero-rank expert results are retained;
- ensure disabled EP retains existing DDP behavior.

CPU FP32 forward comparisons start with the existing rtol=1e-5, atol=1e-6 contract.
Raw input coverage and ownership require exact equality. Checkpoints must have all
expected keys, finite positive scales, and finite outputs. BF16 distributed
summation is not bitwise associative: report output and packed-weight differences,
and set GPU numerical acceptance thresholds in the executor packet before launch.
Do not treat simulated forward equivalence as end-to-end quantization equivalence.

B. Small 2-GPU integration:
Use a tiny real GLM model and real NCCL/offload/save. Compare against standard GPTQ
on identical calibration inputs. Include save/reload; an evidence-only run does
not establish completion.

C. Representative GLM memory and timing:
Use a depth-truncated BF16 checkpoint retaining original hidden dimensions and all
256 experts per MoE layer. Include dense-to-MoE and consecutive MoE boundaries.
A tiny-width fixture alone cannot validate the original OOM. First use 4 GPUs;
then qualify 8 GPUs with fixed global sample/token settings. Use small local
subsets for iteration; record storage and cache state for every timing claim.
For a real-shape DDP control that cannot fit, use the already-fixed Hessian offload
on a small calibration set and label the performance comparison accordingly.

D. Remote full run:
Only after numerical, ownership, memory, and save gates pass, issue the executor
a complete single-node 8-H100 packet. Preserve phase evidence and quantization
quality checks. EP completion means the checkpoint is complete and numerically
validated; SGLang-native export is the separately tracked problem 2. Partial
checkpoints must not be represented as servable production artifacts.

Planner handles research, implementation, CPU tests, and bounded GPU diagnostics
where available. Executor handles representative/full runs on its cluster.
Do not assume the old Slurm-only instructions apply to Rancher: the packet must
identify and verify its target cluster's actual launcher.

## User-supplied AWQ baseline (2026-09-08)

Load/dispatch 1h; 78-layer calibration 13-15h; save 394.6 GB in eight shards 4h;
offline replay 30m; SGLang conversion 2h26m; indexer repatch 32m; MTP graft 15m.
Only conversion and indexer repatch were stated as measured. Other figures are
estimates. Totals: 18-20h through save, 21h43m-23h43m including downstream work.
Treat GB as the checkpoint unit, consistent with the prior checkpoint record.

## Review checklist

- Context and prior implementation inspected; existing metrics and tests identified.
- Upstream/fork prior work and three approaches recorded; final implementation
  source selection still requires the explicit adapter comparison above.
- User's intended scope and planner/executor roles preserved.
- Visual companion unnecessary for this technical decision.
- Draft reviewed for contradictory ownership, unsupported equivalence claims, and
  accidental storage/serving scope expansion.
- Owner design approval, detailed implementation plan, and execution packet remain
  subsequent steps. No implementation or GPU validation is claimed here.
