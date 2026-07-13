# MiniMax-M3 Safe and Diagnostic Full Matrix

## Goal

Obtain usable full W4AFP8 checkpoints as quickly as possible while continuing to
localize the CUDA failure introduced by the guarded diagnostic code. Instrumentation
must never be allowed to block the production-safe quantization lanes.

## Matrix

After the existing pre-quantization compatibility gate and two-root trace smoke pass,
launch five independent top-level `srun --exclusive --nodes=1` jobs concurrently:

| Lane | Recipe | In-process diagnostics | Purpose |
| --- | --- | --- | --- |
| `safe-offsetfix` | full MiniMax AWQ mappings | none | primary checkpoint candidate |
| `safe-nosmooth` | AWQ minus only MoE-input smoothing | none | targeted smoothing ablation |
| `safe-quant_only` | W4AFP8 QuantizationModifier only | none | no-AWQ control |
| `diag-heavy-offsetfix` | full MiniMax AWQ mappings | existing full guards plus staged CUDA synchronization | locate the asynchronous CUDA fault |
| `diag-light-offsetfix` | full MiniMax AWQ mappings | lifecycle/activation evidence only; no qparam, weight, or fake-quant reads | determine whether tensor inspection causes the fault |

Every lane owns one exclusive node. One lane failing must not cancel or terminate any
other lane. The controller must be owned by detached tmux and reject nested Slurm.

## Production-safe contract

Safe lanes invoke the ordinary `python -m pipeline.run --stage quantize` entrypoint.
They must not instantiate guarded modifiers, install callbacks/hooks, manually enable
quantization, inspect quantization parameters or weights, perform fake quantization,
or enforce per-layer thresholds. Each lane receives a fresh unique output root and
refuses pre-existing output.

The quantizer's native log and `quant_metrics.jsonl` are the only progress evidence.
An external controller may retain logs and return codes but must not touch the worker's
CUDA state. After a successful quantization, the controller resolves exactly one
checkpoint and runs `pipeline.verify_quant_checkpoint --check-tensors`. A static-check
failure marks that lane failed but preserves its checkpoint and logs.

Meaningful accuracy checks are deliberately deferred until all successful checkpoints
pass the static checker. Per-layer generation or evaluation would introduce extra model
forwards and violate the production-safe contract.

## Diagnostic contract

Both diagnostic lanes use the guarded full runner. The heavy lane persists a stage
marker before each CUDA synchronization and synchronizes at these boundaries:

1. after the native quantization modifier returns;
2. after manually enabling quantization;
3. before and after quantization-parameter inspection;
4. before and after representative fake quantization.

If synchronization fails, the last durable stage identifies which preceding operation
launched the asynchronous failure. The light lane retains AWQ lifecycle, grid-search,
smoothing-scale, and bounded activation evidence, but performs no qparam attribute
checks, tensor summaries, weight reads, or fake quantization. It may enable quantization
only to capture candidate activations, with synchronization immediately after the
native modifier and immediately after enablement.

## Outcome interpretation

- Any safe lane producing a statically valid checkpoint advances to the existing short
  canonical-chat smoke before full quality evaluation.
- Heavy fails while light passes: qparam/weight/fake-quant inspection is the likely
  instrumentation fault.
- Heavy and light fail at the post-native synchronization: the native quantization
  path launched the CUDA fault.
- Light fails only after enablement: candidate activation setup is implicated.
- Safe quantization succeeds while diagnostics fail: treat diagnostic failures as
  instrumentation defects, not model-quality evidence.
- Safe checkpoints complete but all fail quality: investigate genuine W4AFP8/AWQ
  numerics using post-quantization evidence.

## Executor handoff

The executor first dry-runs the controller, then runs a one-lane `safe-quant_only`
smoke with the production calibration sizes overridden to a tiny dataset. Only after
that smoke creates a statically valid checkpoint may the full five-lane controller be
started. Return the Git revision, run ID, tmux session, every Slurm job/step/node,
commands, per-lane return codes, checkpoint paths, static reports, diagnostic stage
heartbeats, full log paths and hashes, and every runtime deviation.
