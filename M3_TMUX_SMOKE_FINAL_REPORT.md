# MiniMax-M3 tmux Smoke Final Report

Run root:

`results/m3-quality/20260712T162609Z-m3-quality-smoke-tmux`

Controller completed at `2026-07-12T17:00:31Z` with return code `1`.
The failure was limited to the Ray/BF16 diagnostic arms:

```text
ray=143
bf16=143
gptq=0
awq=0
```

## GPTQ and AWQ

Both quantized arms completed successfully with valid artifacts:

- `infrastructure_ok: true`
- `artifacts_valid: true`
- `tasks_scored: 5`
- `empty_count: 0`
- `periodic_loop_count: 0`
- distributed world size: 8
- identical sample-manifest hash
- identical 2,047-token probe corpus hash

Probe elapsed times were 7.37 seconds for repaired GPTQ and 8.78 seconds for
cyankiwi AWQ.

Smoke metrics are exploratory due to the small sample counts:

| Task | Repaired GPTQ | AWQ control |
|---|---:|---:|
| GPQA Diamond | 0/2 | 0/2 |
| IFEval strict / loose | 0.5 / 0.5 prompt-level | 0.333 / 0 prompt-level |
| AIME25 | 0/2 | 0/2 |
| MMLU-Pro | 12/14 | 14/14 |
| GSM8K | 2/2 | 2/2 |

These results do not establish general quality equivalence; the purpose of
this run was the discriminator smoke gate.

## Ray and BF16

Ray placement timed out waiting for its placement-group driver. The BF16 arm
then exited because it expected:

```text
ray_preflight/gate.json
```

while the placement diagnostic emitted the gate under:

```text
ray_placement/topology/gate.json
```

The complete logs and topology artifacts are included in the run root. No
attempt was made to patch or restart the completed run.

## Scope and executor notes

The run was launched through the detached tmux controller. The executor
created a fresh run root and repeated the CPU preflight rather than reusing
the earlier interrupted run root. The executor also diagnosed and documented
the first shell-precedence launch failure. No model, prompt, probe, resource,
or unrelated AWQ representative-layer job was modified.
