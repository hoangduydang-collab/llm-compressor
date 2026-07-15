# Evidence packet: MiniMax-M3 paired reasoning r4.5 smoke

- Protocol version: 1
- State: `RETURNED_FOR_ANALYSIS`
- Packet revision executed: `2026-07-15-r4.5`
- Expected Git commit: `c5a0b755`
- Actual Git commit: `b44c318c3a67da6956da74de5f88080b4cdda728`
- Execution classification: `stopped`
- Evidence commit: recorded in final repository state

## Factual outcome

The r4.5 focused tests and CPU preflight passed. Both smoke arms then loaded
their models, processed all four configured tasks with all three seeds, and
produced valid structured artifacts. The AWQ arm passed its smoke checks.
The GPTQ arm failed the fail-closed smoke gate because one MMLU-Pro response
was empty. The chained production controller therefore exited without
launching production.

This is a smoke-readiness result, not a quality-adoption decision.

## Per-arm results

| Arm | Slurm job / node | State | Return code | Gate | Output |
|---|---|---|---:|---|---|
| cyankiwi AWQ | `12932` / `gpu-h117` | completed | 0 | passed | `results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/models/cyankiwi_awq/shards/smoke/` |
| in-house GPTQ | `12931` / `gpu-h101` | completed | 0 | failed: one empty output | `results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/models/inhouse_gptq/shards/smoke/` |

The arm processes returned zero; the overall smoke controller returned `1`
because the GPTQ readiness gate was false.

## Exact commands executed

```bash
python -m pytest -q \
  pipeline/tests/test_static_checkpoint.py \
  pipeline/tests/test_lmeval_runner.py \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_m3_quality_eval_runner.py \
  pipeline/tests/test_m3_quality_smoke_tmux.py \
  pipeline/tests/test_m3_quality_evidence.py

python -m pipeline.m3_quality_preflight \
  --matrix pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml \
  --run-root results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4

bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke \
  --matrix pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml \
  --run-root results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4
```

The smoke controller ran under tmux session
`m3-quality-20260715T075800Z-r4-5`. A separate gate controller was configured
to launch production only when `ready_for_production` was true.

## Environment and scheduler record

- Repository: `/mnt/nfs/hoangduy/projects/llm-compressor`
- Branch: `duy-branch`
- Environment: `/mnt/nfs/hoangduy/venvs/quant`
- Focused tests: `124 passed`
- Run ID: `20260715T075800Z-m3-paired-reasoning-r4`
- Result root: `results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4`
- Topology: two independent one-node, eight-GPU `srun` arms
- Sample manifest SHA-256: `f3e3a18323e9a06e59881308ed4aa5214a1b000d878abe8fea1ec5b23ac7edc9`
- Harness contract SHA-256: `a51895c9fffd6d6d8bb684be5674aacb55f0178c2eda0851a60d5a6e6543db9e`
- Seeds: `42`, `1234`, `4158`
- Tasks: GPQA Diamond, MMLU-Pro, GSM8K, AIME 2025

## Gate values

| Measurement | AWQ | GPTQ | Required | Result |
|---|---:|---:|---|---|
| Infrastructure | true | true | true | AWQ/GPTQ pass |
| Artifacts valid | true | true | true | AWQ/GPTQ pass |
| All tasks scored | true | true | true | AWQ/GPTQ pass |
| Empty outputs | 0 | 1 | 0 | GPTQ fails |
| Periodic loops | 0 | 0 | 0 | AWQ/GPTQ pass |
| Distributed world size | 8 | 8 | 8 | AWQ/GPTQ pass |
| `ready_for_production` | n/a | n/a | true | false |

The GPTQ empty output is:

- Task/subtask: `mmlu_pro` / `mmlu_pro_economics`
- Document ID: `45`
- Generation seed: `1234`
- Source: `models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl`
- Health artifact: `models/inhouse_gptq/shards/smoke/generation_health/mmlu_pro.json`
- Empty rate: `1/42 = 0.023809523809523808`
- Periodic-loop count: `0`
- Answer-extraction failures: `0`

## Observed facts

- Both arm manifests record commit `b44c318c`, identical tokenizer/chat-template
  hashes, and the identical sample manifest.
- Both arms completed all four tasks and all three seeds.
- AWQ `smoke_evidence.json` has `empty_count: 0`.
- GPTQ `smoke_evidence.json` has `empty_count: 1`.
- `smoke_gate.json` reports `ready_for_production: false`.
- `controller.rc` is `1`; no production job appeared in `squeue`.
- The GPTQ distributed shutdown warnings in stderr occurred after normal arm
  completion and are not the gate cause.

## Limited executor interpretation

The only mechanical blocker was one empty GPTQ generation. This is returned as
evidence; no claim is made here whether it is a transient serving event,
sampling-specific behavior, or a model issue.

## Deviations and retries

- No model, task, seed, sample manifest, topology, or generation parameter was
  changed.
- No retry was attempted.
- Production was not launched because the smoke gate failed.
- The independent r6 quantization speedup job was not modified or cancelled.

## Small committed artifacts

- The complete r4.5 run tree under
  `results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/`
- This evidence packet
- Updated r4.5 handoff state

## Missing artifacts

- Production artifacts: none expected; production was correctly not launched.
- Scheduler accounting snapshots: not generated because the packet stopped at the
  smoke gate; per-arm job IDs and nodes are preserved in arm manifests.

## First failure and last successful stage

- Last successful stage: both models served and completed all configured smoke
  tasks/seeds with valid artifacts.
- First failing operation: GPTQ smoke gate evaluated `empty_count == 1` instead
  of the required zero.
- Immediate evidence: `smoke_gate.json`, GPTQ `smoke_evidence.json`, and
  `generation_health/mmlu_pro.json`.

## Questions for planner

1. Should the single empty GPTQ MMLU-Pro response be investigated with a fresh
   diagnostic packet, or is the smoke gate result sufficient to reject this
   checkpoint?
2. If a retry is authorized, should it preserve the exact manifest and seed
   while adding per-request runtime diagnostics?

## Final repository state

- Executor status: `RETURNED_FOR_ANALYSIS`
- Independent r6 speedup execution remains active and is not covered by this
  packet.
