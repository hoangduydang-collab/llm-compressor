# MiniMax-M3 Guarded Full Quantization Matrix

## Goal

Replace the representative-layer AWQ canary as the primary experiment with full
production quantization runs that validate and persist evidence after every sequential
layer. Failures must stop before the next layer and identify the violated invariant.

The trace-only two-root discriminator remains a cheap structural preflight. The
representative runner remains available only for later targeted ablations.

## Parallel hypothesis matrix

Run three independent arms concurrently, each in detached tmux and a top-level
exclusive-node `srun` allocation:

| Arm | Transform | Question |
| --- | --- | --- |
| `offsetfix` | corrected offset RMSNorm plus all MiniMax AWQ mappings | Does the primary repair produce a healthy production checkpoint? |
| `nosmooth` | `offsetfix` minus only post-attention-norm to MoE-input smoothing | Is that specific smoothing mapping the corruption source? |
| `quant_only` | identical W4AFP8 target and ignore contract, no AWQ transform | Is corruption caused by AWQ smoothing rather than quantization/packing? |

The known-broken unregistered-offset run is not repeated. The previous 8-sample and
512-sample checkpoints already falsified calibration sample count as a sufficient fix.
Existing repaired GPTQ and cyankiwi AWQ checkpoints remain external quality controls.

## Per-layer evidence

An opt-in callback runs after quantized propagation of every sequential subgraph. It
does not run for normal compression. Sparse-layer records are written atomically before
the next layer starts and contain:

- sequential index, decoder path, timestamps, elapsed time, node/rank, and variant;
- expected, resolved, completed, skipped, and unprocessed AWQ mappings;
- independent forward-hook fire counts and activation-stat consumption;
- grid-search initial/best error, selected ratio, improvement, and skip reason;
- finite/zero/min/max/mean/percentile summaries for smoothing scales and quantization
  parameters;
- effective offset-norm invariants before and after smoothing;
- deterministic representative weight fake-quant error, saturation/code occupancy,
  cosine similarity, relative RMSE, and sign-flip rate;
- deterministic reference/candidate activation sketches at layer input, MoE input,
  MoE output, and decoder output, with finite rate, norms, norm ratio, cosine,
  relative RMSE, maximum error, and sign-flip rate.

Activation sketches reuse the existing calibration and propagation forwards and retain
only bounded deterministic token/channel samples on CPU. Weight diagnostics sample a
fixed small set of expert projections. No per-layer checkpoint, generation, serving,
or dataset replay is allowed.

## Abort contract

The guard raises only after its layer artifact and aggregate heartbeat are durable.
Every abort includes layer, mapping/module, check name, observed value, threshold,
exception type/message/traceback, and all completed earlier records.

Broad catastrophic thresholds match the prior representative design: all values must
be finite, output norm ratio must remain within `[0.1, 10]`, cosine must be at least
`0.90`, and relative RMSE at most `0.50`. AWQ arms additionally require nonzero hook
events and completed grid searches for every expected mapping. `quant_only` has no AWQ
mapping requirement but must have quantization parameters and healthy reconstruction.
Dense/ignored layers 0--2 are structural records and are not required to have mappings.

The trace preflight must report matched target nodes and multiple partitions. The full
run stops before calibration if it reports zero targets, partition collapse, incomplete
quantization target coverage, or missing offset-norm adapter coverage.

## Outcomes

- `offsetfix` passes: advance it to checkpoint static ABI validation and quality eval.
- `nosmooth` passes while `offsetfix` fails: the MLP-input smoothing mapping is causal.
- `quant_only` passes while both AWQ arms fail: AWQ transformations are causal; inspect
  which mapping family first violates the per-layer invariants.
- all arms fail at the same layer/metric: investigate shared quantization, model
  linearization, or runtime state rather than AWQ mapping choice.
- all arms pass layer guards: finish all checkpoints, then decide using canonical chat
  and paired accuracy/flip metrics; layer-local health is not an end-to-end quality pass.

## Executor evidence

Return the trace reports, per-arm start/config/provenance manifests, every atomic layer
record, heartbeat/aggregate report, complete log, controller and `srun` return codes,
Slurm job/step/node evidence, checkpoint paths for successful arms, abort reports for
failed arms, and hashes of retained logs. One arm failing must not cancel another.
