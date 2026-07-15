# Evidence packet: generated-reasoning paired evaluation r4.1

- Protocol version: 1
- Packet revision: `2026-07-15-r4.1`
- Required implementation ancestor: `3171dd8e`
- Actual Git commit: `b62ccfbdc46fde6a89b050209a425ff5eb6365dc`
- State: `RETURNED_FOR_ANALYSIS`
- Execution classification: `stopped_before_gpu_allocation`
- Run root:
  `results/m3-quality/20260715T062000Z-m3-paired-reasoning-r4`

## Factual outcome

Workspace, ancestor, environment, and `lm-eval==0.4.12` checks passed. The
fixed r4.1 CPU preflight then failed while inspecting the task's choice
formatter:

```text
doc_to_choice was called but not set in config
...
File "pipeline/m3_quality_preflight.py", line 198, in _representative_task_view
    choices = choice_formatter(doc) if callable(choice_formatter) else None
...
File ".../lm_eval/api/task.py", line 1306, in doc_to_choice
    raise TypeError
TypeError
```

The preflight log is preserved at:

`results/m3-quality/20260715T062000Z-m3-paired-reasoning-r4/preflight.log`

No smoke gate, production dry-run, GPU allocation, arm manifest, sample, or
quality result was created. No retry, package change, configuration edit, or
workaround was attempted.

## Limited executor interpretation

The r4.1 attempt remains blocked in the lm-eval task-definition choice-format
validation path. This is not evidence about GPTQ/AWQ quality. Planner analysis
is required before any further execution.
