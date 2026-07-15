# Evidence packet: MiniMax-M3 r4.4 smoke and distributed quantization r5

- Protocol version: 1
- State: `RETURNED_FOR_ANALYSIS`
- Packet revisions executed: distributed quantization r5 and paired reasoning r4.4
- Expected Git commits: `b907eaf4` (r5 quantization), `849b2071` (r4.4 eval fix)
- Actual Git commit used for r4.4: `d3d245c6`
- Execution classification: `stopped`
- Evidence commit: recorded in the final repository state below

## Factual outcome

The r4.4 CPU preflight passed. Both newly authorized smoke allocations launched,
loaded their models, generated the two smoke requests, and then failed while
checkpointing evaluation results. The smoke gate was therefore false and no
production evaluation was launched.

The older distributed quantization r5 run also terminated. GPTQ reached
calibration and failed during distributed Loguru metrics cleanup. Its sequential
AWQ arm was then rejected by the shared-memory capacity guard before quantization.

These are separate failures. None is evidence of model output quality.

## Per-job and per-arm results

| Work | Scheduler ID / node | State | Return code | First failure | Output |
|---|---|---:|---:|---|---|
| r4.4 AWQ smoke | `12928` / `gpu-h117` | FAILED | 1 | conflicting duplicate `attempt_uid` during result checkpoint | `results/m3-quality/20260715T070500Z-m3-paired-reasoning-r4/models/cyankiwi_awq/shards/smoke/` |
| r4.4 GPTQ smoke | `12929` / `gpu-h123` | FAILED | 1 | conflicting duplicate `attempt_uid` during result checkpoint | `results/m3-quality/20260715T070500Z-m3-paired-reasoning-r4/models/inhouse_gptq/shards/smoke/` |
| r5 GPTQ quantization | `12927` / `gpu-h101` | FAILED | 1 | `loguru.remove(sink_id)` found no handler id 2 | `/mnt/nfs/hoangduy/results/m3-distributed-quant-smoke/20260715T050900Z-m3-ddp-quant-smoke-r5/gptq/` |
| r5 AWQ quantization | `12930` / `gpu-h101` | FAILED | 3 | `/dev/shm` guard: 213,176,926,208 bytes available, 900 GB required | `/mnt/nfs/hoangduy/results/m3-distributed-quant-smoke/20260715T050900Z-m3-ddp-quant-smoke-r5/awq/` |

The r4.4 controller returned `1`. `smoke_gate.json` reports
`ready_for_production: false`, with both models having zero scored tasks.

## Exact commands executed

### r4.4 preflight and smoke

```bash
python -m pipeline.m3_quality_preflight \
  --matrix pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml \
  --run-root results/m3-quality/20260715T070500Z-m3-paired-reasoning-r4

bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke \
  --matrix pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml \
  --run-root results/m3-quality/20260715T070500Z-m3-paired-reasoning-r4
```

The smoke controller ran under tmux session
`m3-quality-20260715T070500Z-r4-smoke`; it launched two one-node,
eight-GPU `srun` arms concurrently.

### r5 GPTQ

```bash
torchrun --nproc_per_node=8 -m pipeline.run \
  --config /mnt/nfs/hoangduy/projects/llm-compressor/pipeline/configs/minimax_m3_distributed_smoke.yaml \
  --stage quantize --evidence-only \
  --set quantization.method=gptq \
  --set model.id=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3 \
  --set model.offload_folder=/mnt/nfs/hoangduy/offload/m3-distributed-quant-smoke/20260715T050900Z-m3-ddp-quant-smoke-r5/gptq \
  --set output_dir=/mnt/nfs/hoangduy/results/m3-distributed-quant-smoke/20260715T050900Z-m3-ddp-quant-smoke-r5/gptq
```

The r5 controller then launched AWQ sequentially; its preflight stopped it
before `torchrun` because the shared-memory threshold was not met.

## Environment and scheduler evidence

- Repository: `/mnt/nfs/hoangduy/projects/llm-compressor`, branch `duy-branch`
- Environment: `/mnt/nfs/hoangduy/venvs/quant`
- Python `3.12.13`; torch `2.11.0`; transformers `5.12.1`;
  compressed-tensors `0.17.2a20260707`; CUDA build `13.0`
- r5 GPTQ host: H100 × 8, `gpu-h101`; start `2026-07-15T05:10:46`,
  end `2026-07-15T07:21:53`, elapsed `02:11:07`
- r5 controller log: `/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/20260715T050900Z-m3-ddp-quant-smoke-r5/controller.log`
- r4.4 result root: `results/m3-quality/20260715T070500Z-m3-paired-reasoning-r4`

## First failure and last successful stage

- r4.4 AWQ/GPTQ last successful stage: model serving and generation of 2/2
  smoke prompts. First failure: `_deduplicate_sample_rows` rejected a
  conflicting duplicate `attempt_uid`.
- r5 GPTQ last successful stage: distributed calibration progress was recorded
  (the stderr shows calibration/propagation through early layers). First
  failure: `pipeline/metrics.py:capture_quant_metrics` attempted to remove
  Loguru handler id 2 after it had already been removed.
- r5 AWQ last successful stage: launcher allocation. First failure: explicit
  `/dev/shm` capacity check.

## Raw evidence

The complete r4.4 smoke tree is committed with this report, including
`preflight/`, `logs/`, `smoke_report.json`, `smoke_gate.json`, manifests, and
return codes.

The large r5 raw logs remain on durable shared storage:

| Artifact | Absolute path | Bytes | SHA-256 |
|---|---|---:|---|
| r5 controller | `/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/20260715T050900Z-m3-ddp-quant-smoke-r5/controller.log` | 736 | `60033a7c49a5e523ad02a4f4cb6cf061731337de8c21c01a757505543ac3841e` |
| r5 GPTQ stderr | `/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/20260715T050900Z-m3-ddp-quant-smoke-r5/gptq/torchrun.err` | 17,048,450 | `eb09767eff2c2745e0262b4b7df7c5fa8c5f979cb88a4b0e3f547e02b0a0bb23` |
| r5 GPTQ stdout | `/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/20260715T050900Z-m3-ddp-quant-smoke-r5/gptq/torchrun.out` | 22,216 | `1c2e9a8cf5fba8067d59b799f2fc3a6ffda4ec10613236e7aa9ea37e14620258` |
| r5 command | `/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/20260715T050900Z-m3-ddp-quant-smoke-r5/gptq/command.txt` | 513 | `434f3c258f7e3e31bd31af62e87580a8235bde5c1f3afef75123736a15276c84` |
| r5 environment | `/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/20260715T050900Z-m3-ddp-quant-smoke-r5/gptq/environment.txt` | 3,142 | `d13f78ce9362182fd8abffe2bf15b97b359fd5879d4810ef1e580225835450de` |
| r5 resources | `/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/20260715T050900Z-m3-ddp-quant-smoke-r5/gptq/resources.log` | 92,572 | `375f8bd9a667d0516ecd31a07bc15eff85131e430a601e84d0fb8ec4cb0d9c87` |

## Limited executor interpretation

The r4.4 smoke failure is a deterministic eval-harness result-ID collision
after successful generation. The r5 GPTQ failure is a distributed metrics
handler-lifecycle bug. The r5 AWQ failure is an environmental capacity stop.
These are bounded observations returned for planner analysis; no fixes or
retries were attempted.

## Deviations and retries

- No experiment parameters, model paths, task counts, or topology were changed.
- No retries were attempted.
- Existing independent jobs were not cancelled because another arm failed.
- Production evaluation was not launched because the smoke gate was false.

## Questions for planner

1. Should the eval harness make `attempt_uid` include model/arm identity or
   otherwise avoid cross-rank duplicate checkpoint collisions?
2. Should distributed metrics cleanup make Loguru handler removal rank-safe and
   idempotent?
3. What shared-memory threshold and node-selection policy should the next AWQ
   packet require?

## Final repository state

- Evidence commit pushed: see the commit containing this report
- Branch synchronization: to be verified immediately after push
- Large raw artifacts: preserved at the absolute paths and hashes above
- Executor status: `RETURNED_FOR_ANALYSIS`; no downstream work authorized
