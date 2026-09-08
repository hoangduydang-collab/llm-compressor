# GLM-5.3 expert-parallel GPTQ implementation

Status: local implementation and CPU validation; GPU qualification pending.
Owner: planner. The remote execution agent owns representative/full GPU runs.

## Enablement and supported scope

Use the existing `pipeline.run` entrypoint and recipe builder. Opt in with:

```yaml
quantization:
  method: gptq
  scheme: W4AFP8
  gptq_expert_parallel: true
calibration:
  moe_calibrate_all_experts: true
  sequential_targets:
    - GlmMoeDsaDecoderLayer
```

Merge these fields into a run config with its own verified input/output paths;
this snippet is not a standalone full-model launch configuration. The direct
library option is `GPTQModifier(expert_parallel=True)` with
`oneshot(..., pipeline="sequential", moe_calibrate_all_experts=True)` under an
initialized process group. Distributed disk offload additionally requires the
existing safe-update patch installed by `pipeline.run`.

The first supported mode is one node, one EP group, equal nonzero calibration
steps, one decoder target per subgraph, and all-expert calibration. Token lengths
and expert ownership counts may be uneven. Routed expert activations must be
absent or observer-free and dynamic. Static expert observers, fused global weight
statistics, conditioning transforms, conflicting ownership, mismatched EP flags,
and additional modifiers targeting routed experts are rejected. Disjoint FP8-rest
quantization is supported. Defaults remain EP-off.

This does not change the W4AFP8 serving contract. In particular, SGLang's static
MoE activation format, indexer conversion and MTP coverage remain problem 2.
Do not label a partial or unqualified checkpoint production-servable.

## Existing work reused

- The fork's `LinearExperts2D` gathered-token forward and `shard_experts` provide
  transport and ownership. No new all-to-all dispatcher was written.
- MoEQuant's expert-local versus replicated Hessian split informs the integration;
  llm-compressor's existing accumulator, solver, damping and activation ordering
  are retained.
- Replicated Hessians still use existing greedy assignment and bounded reduction.
  Expert owners solve complete Hessians without reducing them to another rank.
- Each rank retains at most one unpublished solved module per round. Every rank
  publishes the same module/attribute sequence through PyTorch broadcasts and
  compressed-tensors storage helpers. CPU updates are serialized where necessary;
  distributed disk writes use the existing source-rank patch and barriers.
- Existing loguru sinks, cache transitions, packing and collective saving remain
  in use. Packed cache parameter classes are normalized with CT's existing
  `to_tensor` helper before conversion to Accelerate; tensor identity/storage and
  disk-index keys are retained.

Detailed prior-work comparison and source revisions:
[approved design](superpowers/specs/2026-09-08-glm53-expert-parallel-gptq-design.md)
and [implementation plan](superpowers/plans/2026-09-08-glm53-ep-gptq.md).

## Numerical and lifecycle findings

Gathering W rank inputs into one expert call must retain W reference contribution
counts. New tests check raw Hessians, counts, normalized Hessians and actual
expert input coverage. These are tolerance-based numerical comparisons, not
claims of bitwise equivalence across distributed reductions.

Review found an existing mixed-modifier interaction: QuantizationModifier's epoch
callback visited every quantized subgraph module, so a disjoint FP8-rest pass
could overwrite GPTQ expert scales and onload non-owned experts. The EP path now
filters that callback to the modifier's resolved targets. Legacy EP-disabled
mixed DDP behavior is unchanged; it is explicitly excluded as a numerical
reference for the mixed test. Pure GPTQ has DDP-versus-EP comparisons through
intermediate propagation, output and packed checkpoint differences.

Full tiny-GLM tests exercise a dense layer followed by two MoE layers, actual
Gloo collectives, the real GPTQ solver, collective packed export, original-model
release and local-file reload. Separate tests cover private/shared CPU caches,
disk caches, optional qparameters, changed slot shapes and collective failures.

## Phase evidence for problem 3

Existing rank-local `quant_metrics.rank-N.jsonl` now contains paired structured
phase records from load/dispatch through dataset preparation, tracing, calibration,
Hessian reduction, solve, publication, propagation, offload cleanup, save prewarm,
checkpoint saving and existing offline verification. The records include monotonic
times, nesting, rank/world size, status and available process/GPU/cgroup counters.
GPU peak counters are explicitly scoped to the allocator's existing peak window;
logging neither initializes CUDA nor synchronizes or resets every operation.

Aggregate explicit rank paths using `pipeline.metrics.summarize_phases(paths)`.
Completeness requires the expected ranks and enclosing run spans. Paired failure
records count as complete evidence, not success. Node wall envelopes and summed
rank work are separate; nested/overlapping durations must not be added. Process
I/O counters do not measure unique CephFS traffic. No GLM-5.3 full-run performance
claim follows from these tiny CPU tests.

## Verification and remaining gates

Local environment: Python 3.12.3, torch 2.11.0+cu128, Transformers 5.12.1,
compressed-tensors 0.17.2a20260707, pytest 9.1.1. The development shell has no
CUDA device. A separate bounded two-A100 Slurm diagnostic is prepared in the
[local NCCL packet](glm53-ep-gptq-local-nccl-packet.md); its result is pending.
This CPU environment does not certify the remote executor environment.

The final local regression result is recorded in the validation update below.
The pre-change EP/offload baseline passed 98 tests. Development additionally
confirmed real two-process GPTQ/propagation/save/reload behavior and phase logging.

Three bounded two-GPU NCCL tests are available in:
`tests/llmcompressor/modifiers/gptq/test_expert_parallel_nccl.py`.
From a verified checkout and ML environment, inside an already allocated two-GPU
node, the test command is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q -rs tests/llmcompressor/modifiers/gptq/test_expert_parallel_nccl.py
```

The tests spawn their own two ranks; do not wrap this command in `torchrun`.
They cover DDP/EP parity, EP with disk offload, and mixed FP8-rest with disk offload,
including packed saving and reload. On a CPU host they skip, which is not a pass.
The process-group timeout is 60 seconds and the process deadline is 240 seconds
per case. These are small fixtures, not the real-width memory experiment.

A ready executor packet still needs the target cluster's actual launcher,
checkpoint/data paths, tested dependency revisions and resource limits. After the
NCCL gate, qualify a depth-truncated real-width GLM retaining all 256 experts at
EP4/EP8 before launching the full checkpoint. Preserve the same calibration
manifest and storage/prefetch settings across comparisons. The original OOM and
full-run quantization quality remain unqualified until that evidence returns.

## Validation update

The combined CPU regression passed **212 tests, with 4 CUDA-dependent skips** in
88.04 seconds. See the [retained test log](../results/glm53-ep-gptq/20260908-cpu/qualified.log).
The tiny pure-GPTQ comparison recorded 21 packed tensors per rank and zero differing
packed words on this fixture ([rank 0](../results/glm53-ep-gptq/20260908-cpu/packed-differences.rank-0.json),
[rank 1](../results/glm53-ep-gptq/20260908-cpu/packed-differences.rank-1.json)).
This is not a bitwise-equivalence guarantee for GPU or full-size runs.

The validation command unset the shell's ambient `SLURM_STEP_ID` for the existing
environment-snapshot test and retained pytest artifacts under `/tmp/pytest-of-e1129930/`.
Earlier attempts exposed those two test-harness constraints: one ambient-environment
assertion and one teardown file-leak assertion caused by retaining files outside
pytest's recognized temporary directory. Both attempt logs are retained alongside
the passing log; production code and the existing checks were not weakened.
No GPU or full-model result is claimed yet.
