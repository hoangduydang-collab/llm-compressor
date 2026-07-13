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
any model passed or failed model quality. Raw-log analysis established that
in-house GPTQ and cyankiwi AWQ successfully initialized their full 8-GPU vLLM
engines before lm-eval rejected the reasoning arguments. MiniMax uses adaptive
thinking by default, and lm-eval 0.4.12 disallows `enable_thinking=True` for the
suite's multiple-choice/loglikelihood tasks. The retry therefore leaves
`enable_thinking` unset and uses `think_end_token=</mm:think>` only for output
stripping.

AutoRound's top-level `bits: 16` is an unquantized-default sentinel over mixed
2/3/4/8-bit module overrides. Faithful loading requires its pinned
OneCompression plugin and repository-specific loader, so it is deferred rather
than modified or counted as a quality failure. BF16 failed inside the
two-node Ray bootstrap before writing an arm manifest; the retry adds a
standalone topology gate and rank-local diagnostics. Smoke-gate validation is
also fixed to write a structured failure for zero probe evidence.
