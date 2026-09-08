# Queued H100 EP GPTQ correctness test

Owner authorization: 2026-09-09, “push, then you can queue a H100 gpu testing”.
Shared scope: [the three GLM-5.3 objectives](../GLM53_QUANTIZATION_OBJECTIVES.md).
This is the active local H100 diagnostic contract, following the passing
[two-T4 validation](glm53-ep-gptq-gpu-validation.md).

Run the same three tiny real-GLM NCCL cases: DDP/EP parity, weight-only disk
save/reload, and mixed FP8-rest disk save/reload. Numerical tolerances, 60-second
process-group timeout and 240-second per-case deadline stay unchanged. This is
not the representative 256-expert, real-width OOM experiment or a full quantization.

Resources: one node, two full H100s (`gpu:h100-96:2`), four CPU cores, 16 GiB host
memory, 15-minute runtime limit. No specific node is pinned; Slurm selects a
suitable node. Queue/controller lifetime is capped at eight hours, within the
current CPU controller allocation's lifetime. One submission; no automatic retry.

Launcher: top-level `srun` from detached tmux `glm53_h100_20260909`, using a clean
environment and `--cpu-bind=cores`. The controller remains running while pending.
The prior queued-allocation callback failure remains unresolved; any recurrence
must be recorded as a launch failure, not a test result. A pending job is not a
GPU pass. No scheduler, firewall or node configuration is modified.

The worker exports source/config/tests from fixed revision
`79e0d8a201e8c66be7bfc9b74b6d609399b8ca28` through `git archive` to a node-local
pytest directory. `PYTHONPATH` points at that snapshot, and preflight asserts that
llmcompressor, pipeline and tests are imported from it. Later branch edits cannot
change the tested code. The existing shared interpreter is used without upgrades;
preflight requires torch 2.11.0+cu128, Transformers 5.12.1 and compressed-tensors
0.17.2.a20260707, and records the actual environment. It also requires two visible
non-MIG H100 CUDA devices with compute capability 9.0 before pytest begins.

[Controller and worker scripts](../results/glm53-ep-gptq/20260909-h100-nccl/)
retain job ID, source/controller revisions, startup/GPU/environment records,
passed and failed pytest output, exit codes and per-rank lifecycle/packed-difference
artifacts. Snapshot and full tiny checkpoints remain node-local; small evidence
is copied to the shared result directory. The planner reads returned evidence
before changing objective 1's qualification status.

Queue status is recorded after submission in the shared result directory. The
remote Rancher executor should read the shared brief first; this Slurm command is
not a Rancher execution packet.

## Submission

Slurm job **830962** is queued, with the latest recorded state **PENDING
(Resources)** and no estimated start time. The detached controller is alive.
See the [timestamped queue snapshot](../results/glm53-ep-gptq/20260909-h100-nccl/queue-status.txt).
This records submission only; the worker has not started and no H100 result is claimed.
