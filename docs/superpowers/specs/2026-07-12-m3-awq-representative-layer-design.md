# MiniMax-M3 AWQ Representative-Layer Diagnostic

## Goal

Test the two existing AWQ repair hypotheses quickly without completing or
exporting a full MiniMax-M3 quantization. The diagnostic must remain scoped to
the model-quality issue and must not begin CUDA-graph or throughput work.

## Experiment

Run six independent arms concurrently:

| Layer | Variant |
| --- | --- |
| 8 | corrected offset-norm smoothing (`offsetfix`) |
| 8 | MLP-input smoothing disabled (`nosmooth`) |
| 31 | corrected offset-norm smoothing (`offsetfix`) |
| 31 | MLP-input smoothing disabled (`nosmooth`) |
| 59 | corrected offset-norm smoothing (`offsetfix`) |
| 59 | MLP-input smoothing disabled (`nosmooth`) |

Layer 8 is the first previously observed corruption boundary. Layers 31 and 59
sample the middle and tail of the repeated sparse-layer stack. Each arm loads
the unquantized model, applies the existing production AWQ recipe only to the
selected layer, measures fidelity in memory, writes compact evidence, and
exits. It does not save or re-export a model checkpoint.

## Quantization isolation

Keep the production calibration dataset, sample count, sequence length,
W4AFP8 scheme, AWQ grid, mappings, and all existing router/shared-expert/
attention exclusions unchanged. Add an arm-specific exclusion for every
language-model decoder layer except the selected layer. Restrict sequential
processing to that selected decoder-layer module as well.

Before running on GPUs, a CPU/meta-model preflight must prove that:

1. exactly the selected layer's expert projections receive a quantization
   scheme;
2. no expert projection in any other layer is targeted;
3. AWQ resolves smoothing mappings for the selected layer only; and
4. `nosmooth` removes only the post-attention-norm to MLP-input mapping.

If any condition fails, the arm must stop before calibration.

## Fidelity measurement

Use a small, fixed probe subset drawn deterministically from the same
calibration corpus. Capture the selected layer's input, post-attention-norm
(MoE input), MoE output, and decoder-layer output before and after in-memory
quantization using identical tokenized inputs. Since no earlier layer is
modified, the selected layer receives the same input in both passes. The MoE
boundaries prevent the residual connection from hiding a broken expert path.

For every probe and in aggregate, record:

- finite fraction;
- L2 and maximum-absolute norms;
- candidate/reference norm ratio;
- cosine similarity;
- relative RMSE;
- maximum absolute error; and
- exact model, recipe, layer, variant, Git revision, environment, timing, and
  scheduler metadata.

Also record the resolved AWQ mapping names and quantized module names. Raw
hidden states remain outside Git; compact JSON evidence and complete logs are
returned through the repository.

## Decision rules

An arm is an infrastructure failure if loading, calibration, scheduling, or
metric capture does not complete. It is a quality failure if any aggregate
boundary has a finite fraction below 1.0, a candidate/reference norm ratio
outside `[0.1, 10.0]`, cosine similarity below 0.90, or relative RMSE above
0.50. These deliberately broad gates detect the previously observed
catastrophic corruption; continuous metrics and comparison with the existing
BF16/GPTQ/cyankiwi evidence determine which passing variant is better. The raw
metrics must remain in the report so a borderline classifier cannot hide the
evidence.

Interpretation across arms:

- `offsetfix` passes all three layers: corrected offset-norm handling is safe
  enough to justify the full AWQ rebuild.
- `nosmooth` passes while `offsetfix` fails: MLP-input smoothing remains the
  likely fault and the no-smoothing recipe should advance.
- both pass: both avoid the catastrophic local corruption; compare continuous
  error and run one mixed-checkpoint smoke before choosing the full recipe.
- both fail: the current AWQ root-cause hypothesis is insufficient; do not
  launch another full rebuild.

A pass is evidence about the AWQ direction, not proof that a fully quantized
checkpoint has production quality.

## Cluster launcher and outputs

Provide an `srun`-only launcher that starts all six one-GPU arms concurrently,
uses unique output/log directories, waits for every arm, preserves failures,
and aggregates a matrix JSON/report. The expected runtime is governed by one
layer of smoothing per arm rather than all 57 sparse layers. No arm may cancel
another arm.

If the controlling session disappears again, the returned evidence must
distinguish scheduler/session cancellation from quantization failure. The
handoff will require exact commands, job/step IDs, nodes, return codes, full
logs, and immediate scheduler accounting for abnormal exits.

## Testing

CPU tests cover layer-selection patterns, mapping isolation, metric arithmetic,
classification, aggregation with partial failures, dry-run command generation,
and the guarantee that the diagnostic never calls checkpoint save/export.
Existing MiniMax configuration, offset-norm, and AWQ mapping tests remain in
the verification set.
