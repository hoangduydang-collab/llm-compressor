# Evidence packet: generated-reasoning paired evaluation r4

- Protocol version: 1
- Packet revision: `2026-07-15-r4`
- Required implementation ancestor: `ca044dff`
- Actual Git commit: `3411ebb20161d52ad87fadcab7951e3bb319b673`
- State: `RETURNED_FOR_ANALYSIS`
- Execution classification: `stopped_before_gpu_allocation`
- Run root:
  `results/m3-quality/20260715T061300Z-m3-paired-reasoning-r4`

## Factual outcome

Workspace, ancestor, environment, and `lm-eval==0.4.12` checks passed. The
mandatory CPU preflight then failed while inspecting representative generated
task records, before the smoke gate or any GPU allocation:

```text
Traceback (most recent call last):
  ...
  File "pipeline/m3_quality_preflight.py", line 193, in _representative_task_view
    prompt = formatter(doc, num_fewshot, **kwargs)
  File ".../lm_eval/api/task.py", line 970, in fewshot_context
    partial(chat_template, add_generation_prompt=not gen_prefix)
TypeError: the first argument must be callable
```

The preflight log is preserved at:

`results/m3-quality/20260715T061300Z-m3-paired-reasoning-r4/preflight.log`

No smoke gate, dry-run, production launch plan, Slurm allocation, arm
manifest, sample, or quality result was created. No retry, package change,
configuration edit, or workaround was attempted.

## Limited executor interpretation

The failure is in the r4 preflight’s lm-eval chat-template formatting path.
It is not evidence about GPTQ/AWQ quality. Planner analysis is required before
any future execution packet.
