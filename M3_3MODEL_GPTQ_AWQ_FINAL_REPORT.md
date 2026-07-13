# MiniMax-M3 GPTQ/AWQ Smoke Results

Run root:

`results/m3-quality/20260712T175323Z-m3-quality-3model-tmux`

Both quantized arms completed successfully:

```text
gptq=0
awq=0
```

Each arm reported:

- `infrastructure_ok: true`
- `artifacts_valid: true`
- five tasks scored;
- zero empty outputs;
- zero periodic loops;
- tensor-parallel world size 8;
- 2,047-token teacher-forced probe completed;
- identical sample-manifest and probe-corpus hashes.

Probe times:

- repaired GPTQ: 7.42 seconds;
- cyankiwi AWQ: 8.68 seconds.

Smoke metrics are intentionally small-sample diagnostics:

| Task | Repaired GPTQ | AWQ |
|---|---:|---:|
| GPQA Diamond | 0/2 | 0/2 |
| IFEval prompt strict | 0.0 | 0.5 |
| IFEval instance strict | 0.5 | 0.667 |
| AIME25 | 0/2 | 0/2 |
| MMLU-Pro | 12/14 | 13/14 |
| GSM8K | 2/2 | 2/2 |

The complete arm-specific logs and artifacts are included under:

- `models/inhouse_gptq/shards/smoke/`
- `models/cyankiwi_awq/shards/smoke/`
- `logs/gptq-smoke.*`
- `logs/awq-smoke.*`

The BF16/Ray result is documented separately in
`M3_3MODEL_RAY_BF16_REPORT.md`.
