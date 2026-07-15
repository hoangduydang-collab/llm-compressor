# Evidence packet: generated-reasoning paired evaluation r4.2

- Protocol version: 1
- Packet revision: `2026-07-15-r4.2`
- Required implementation ancestor: `065ff302`
- Actual Git commit: `5e9181b1b5b4a87f9c43ddfe450c74086ff0d449`
- State: `RETURNED_FOR_ANALYSIS`
- Execution classification: `stopped_before_gpu_allocation`
- Run root:
  `results/m3-quality/20260715T063400Z-m3-paired-reasoning-r4`

## Factual outcome

Workspace, ancestor, environment, and `lm-eval==0.4.12` checks passed. The
r4.2 CPU preflight then failed while selecting the installed `mmlu_pro` task:

```text
ValueError: cannot select a single task object for 'mmlu_pro':
['mmlu_pro_biology', 'mmlu_pro_business', 'mmlu_pro_chemistry',
 'mmlu_pro_computer_science', 'mmlu_pro_economics', 'mmlu_pro_engineering',
 'mmlu_pro_health', 'mmlu_pro_history', 'mmlu_pro_law', 'mmlu_pro_math',
 'mmlu_pro_other', 'mmlu_pro_philosophy', 'mmlu_pro_physics',
 'mmlu_pro_psychology']
```

The preflight log is preserved at:

`results/m3-quality/20260715T063400Z-m3-paired-reasoning-r4/preflight.log`

No smoke gate, production dry-run, GPU allocation, arm manifest, sample, or
quality result was created. No retry, package change, configuration edit, or
workaround was attempted.

## Limited executor interpretation

The r4.2 attempt is blocked by the preflight’s assumption that `mmlu_pro`
resolves to one task object, while lm-eval exposes its subject subtasks. This is
not evidence about GPTQ/AWQ quality. Planner analysis is required before any
further execution.
