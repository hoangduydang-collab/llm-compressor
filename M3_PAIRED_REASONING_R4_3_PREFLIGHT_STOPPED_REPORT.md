# Evidence packet: generated-reasoning paired evaluation r4.3

- Protocol version: 1
- Packet revision: `2026-07-15-r4.3`
- Required implementation ancestor: `15831064`
- Actual Git commit: `1eff025ad6999c8d4e3d9628d027b1fbe41bafaf`
- State: `RETURNED_FOR_ANALYSIS`
- Execution classification: `stopped_before_gpu_allocation`
- Run root:
  `results/m3-quality/20260715T064700Z-m3-paired-reasoning-r4`

## Factual outcome

Workspace, ancestor, environment, and `lm-eval==0.4.12` checks passed. The
r4.3 CPU preflight then failed during the generated-task harness contract
audit:

```text
ValueError: gpqa_diamond installed task does not expose metric/filter
'exact_match,flexible-extract'
```

The preflight log is preserved at:

`results/m3-quality/20260715T064700Z-m3-paired-reasoning-r4/preflight.log`

No smoke gate, production dry-run, GPU allocation, arm manifest, sample, or
quality result was created. No retry, package change, configuration edit, or
workaround was attempted.

## Limited executor interpretation

The r4.3 attempt is blocked by a mismatch between the pinned GPQA metric/filter
contract and the installed lm-eval task definition. This is not evidence about
GPTQ/AWQ quality. Planner analysis is required before any further execution.
