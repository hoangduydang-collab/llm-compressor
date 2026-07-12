# MiniMax-M3 Quality Smoke Report

## Result

The smoke profile completed its infrastructure launch, but the smoke gate
failed. No model produced valid scored evidence, so this run must not be
treated as a quality comparison or as approval for the production matrix.

Run directory:

`results/m3-quality/20260712-073626-m3-quality`

The run launched the expected four arms across five nodes:

- BF16 reference
- In-house GPTQ
- cyankiwi AWQ
- AutoRound

## Per-arm failures

| Arm | Result | Failure |
| --- | --- | --- |
| `inhouse_gptq` | Failed before scoring | `enable_thinking=True` but `think_end_token=None` |
| `cyankiwi_awq` | Failed before scoring | `enable_thinking=True` but `think_end_token=None` |
| `aquaman_autoround` | Failed during vLLM initialization | Unsupported `weight_bits: 16`; supported values are 2, 3, 4, and 8 |
| `bf16` | Failed at Slurm task level | Task exited with code 2; no valid evidence artifact |

Relevant logs:

- `results/m3-quality/20260712-073626-m3-quality/logs/smoke-inhouse_gptq-smoke.err`
- `results/m3-quality/20260712-073626-m3-quality/logs/smoke-cyankiwi_awq-smoke.err`
- `results/m3-quality/20260712-073626-m3-quality/logs/smoke-aquaman_autoround-smoke.err`
- `results/m3-quality/20260712-073626-m3-quality/logs/smoke-bf16-smoke.err`

## Harness issue

The final smoke-gate validation also crashed while projecting probe
overhead:

```text
ValueError: probe timing inputs must be positive
```

This occurred because the failed arms produced zero probe tokens and zero
elapsed probe time. Consequently, `smoke_gate.json` was not generated.

## Interpretation

This is an evaluation configuration/infrastructure failure, not evidence that
any model passed or failed model quality. The next run should first:

1. Disable or correctly configure thinking mode by supplying
   `think_end_token` for MiniMax models.
2. Exclude AutoRound until its `weight_bits: 16` configuration is corrected or
   the evaluator supports that value.
3. Fix smoke-gate validation to report incomplete probe evidence instead of
   raising on zero timing inputs.
4. Rerun smoke and require `ready_for_production: true` before launching the
   production matrix.
