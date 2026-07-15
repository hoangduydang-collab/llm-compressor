# Partial evidence packet: MiniMax-M3 r4.7 BF16 smoke

- Protocol version: 1
- State: `EXECUTING` (partial Wave A evidence)
- Packet revision: `2026-07-15-r4.7`
- Expected Git ancestor: `e7834694`
- Actual Git commit executed: `5c6581dbd83ea0c0176896fefee71b8d37d49c9a`
- Execution classification: `stopped`
- Active packet: `M3_PRODUCTION_EVAL_HANDOFF.md`

## Factual outcome

The BF16 smoke controller launched its authorized two-node TP8xPP2/Ray arm.
The Ray topology preflight command completed far enough for the BF16 evaluation
allocation to start. vLLM then rejected pipeline parallelism during model
initialization because the MiniMax model does not implement vLLM's
`SupportsPP` interface. No BF16 evaluation tasks were scored.

The BF16 smoke controller returned nonzero. Per the active packet, BF16
production Wave B was not launched. The independent paired production and exact
GPTQ replay arms were left running.

## Scheduler and arm result

| Arm | Job / nodes | State | Return code | Gate | Result |
|---|---|---|---:|---|---|
| BF16 Ray smoke | `12941` / `gpu-h104`, `gpu-h105` | FAILED | `0:15` | `ready_for_production=false` | No tasks scored |

- Allocation start: `2026-07-15T16:06:34`
- Allocation end: `2026-07-15T16:08:14`
- Elapsed: `00:01:40`
- Reported distributed world size: `16`
- Smoke report: `results/m3-quality/20260715T160500Z-m3-bf16-reasoning-r4/smoke_report.json`
- Smoke gate: `results/m3-quality/20260715T160500Z-m3-bf16-reasoning-r4/smoke_gate.json`
- Arm manifest: `results/m3-quality/20260715T160500Z-m3-bf16-reasoning-r4/models/bf16/shards/smoke/arm_manifest.json`

## Exact launch and first failure

The executed BF16 smoke arm was:

```bash
srun --exclusive --nodes=2 --ntasks=2 --gpus-per-node=8 \
  --kill-on-bad-exit=1 \
  pipeline/slurm/test_m3_quality_eval_arm.sh \
  --profile smoke \
  --run-root results/m3-quality/20260715T160500Z-m3-bf16-reasoning-r4 \
  --matrix pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml \
  --model-label bf16 \
  --model /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3 \
  --shard smoke \
  --tasks gpqa_diamond_cot_zeroshot,mmlu_pro,gsm8k_cot,aime25 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --run-probe 0 \
  --probe-tokens 0
```

The first Python exception was:

```text
NotImplementedError: Pipeline parallelism is not supported for this model.
Supported models implement the SupportsPP interface.
```

It occurred in vLLM configuration validation before model evaluation:

```text
vllm/config/model.py:1179, in verify_with_parallel_config
```

Secondary scheduler messages show rank 0 exited with code 1 and rank 1 was
terminated by `--kill-on-bad-exit`; they are consequences of the first failure.

## Gate values

| Measurement | Value | Required | Result |
|---|---:|---|---|
| Infrastructure | false | true | fail |
| Artifacts valid | false | true | fail |
| Tasks scored | 0 | all smoke tasks | fail |
| Empty outputs | 0 | <= 1 | not decisive |
| Distributed world size | 16 | 16 | pass |
| `ready_for_production` | false | true | fail |

## Raw evidence

The BF16 result root contains the dry run, CPU preflight, cross-run contract
gate, ray preflight directory, smoke report/gate, arm manifest, and arm logs:

`results/m3-quality/20260715T160500Z-m3-bf16-reasoning-r4/`

Relevant raw files:

- `logs/smoke-bf16-smoke.err`
- `logs/smoke-bf16-smoke.out`
- `logs/smoke-controller.out`
- `smoke-controller.rc`
- `smoke_report.json`
- `smoke_gate.json`
- `models/bf16/shards/smoke/arm_manifest.json`

Scheduler metadata was captured with:

```bash
sacct -j 12941 \
  --format=JobIDRaw,JobName%40,State,ExitCode,NodeList,Elapsed,Start,End -P
scontrol show job 12941
```

## Deviations, retries, and downstream work

- No model, checkpoint, task, sample manifest, or generation parameter changed.
- No retry was attempted.
- No PP support patch or topology substitution was made.
- BF16 production Wave B was not launched.
- Paired production and exact GPTQ replay remained untouched and continued.
- The active r4.7 packet remains in progress because other Wave A arms are
  independent and have not yet returned.

## Limited executor interpretation

This is a topology/model-runtime incompatibility at vLLM initialization, not a
BF16 quality result. The packet's requested TP8xPP2/Ray smoke configuration
cannot reach evaluation with the installed vLLM model interface. The planner
must decide whether to abandon the BF16 companion, issue a new supported
topology packet, or otherwise revise the experiment.

## Questions for planner

1. Should the BF16 companion be closed as unsupported for this model/runtime?
2. If a replacement is desired, what explicitly authorized topology should be
   tested instead of PP2?
