# Partial evidence packet: MiniMax-M3 r4.7 GPTQ empty-output replay

- Protocol version: 1
- State: `EXECUTING` (partial Wave A evidence)
- Packet revision: `2026-07-15-r4.7`
- Expected Git ancestor: `e7834694`
- Actual Git commit executed: `5c6581dbd83ea0c0176896fefee71b8d37d49c9a`
- Execution classification: `stopped`
- Active packet: `M3_PRODUCTION_EVAL_HANDOFF.md`

## Factual outcome

The authorized one-node, eight-GPU replay allocation started and loaded the
GPTQ overlay far enough to initialize the vLLM engine and distributed workers.
It then failed during vLLM W4A8 MoE weight construction before either the
256-token or 16,384-token control was generated.

No replay diagnostic JSON was produced. This run therefore does not determine
whether the original empty MMLU-Pro response reproduces.

The independent paired production arms remained running and were not modified.

## Scheduler result

| Arm | Job / node | State | Return code | Result |
|---|---|---|---:|---|
| Exact GPTQ replay | `12935` / `gpu-h117` | FAILED | `1` | Model initialization failed; no controls generated |

The replay controller return-code file is:

`results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/diagnostics/empty-output-replay-controller.rc`

Its value is `1`.

## Exact command

```bash
srun --exclusive --nodes=1 --ntasks=1 --gpus-per-node=8 \
  --kill-on-bad-exit=1 --time=12:00:00 \
  python -m pipeline.m3_empty_output_replay \
  --config pipeline/configs/eval_minimax_m3_reasoning_r4.yaml \
  --model /mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay \
  --samples results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl \
  --attempt-uid 8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878 \
  --out results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/diagnostics/empty-output-replay.json
```

## First failure

The first relevant Python exception was:

```text
AssertionError: intermediate_size_per_partition=384 must be divisible by 256
```

It occurred in vLLM's compressed-tensors W4A8 MoE weight path:

```text
vllm/model_executor/layers/quantization/compressed_tensors/
compressed_tensors_moe/compressed_tensors_moe_w4a8_fp8.py:72
```

The failure happened while constructing `MiniMaxM3MoE` and before any replay
sample was submitted. The subsequent `WorkerProc initialization failed`,
`Engine core initialization failed`, and `srun` termination messages are
secondary consequences.

## Raw evidence

The complete replay controller evidence is retained under:

`results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/`

Relevant files:

- `logs/empty-output-replay-controller.out`
- `logs/empty-output-replay-controller.err`
- `diagnostics/empty-output-replay-controller.rc`

The expected output
`diagnostics/empty-output-replay.json` is missing because model
initialization failed before report creation.

## Gate and protocol handling

- 256-token control: not run
- 16,384-token control: not run
- Replay classification: unavailable
- Retry: none
- Topology change: none
- Model/checkpoint change: none
- Paired production arms: left running
- BF16 Wave B: not launched

## Limited executor interpretation

This is a replay-path vLLM W4A8 shape-compatibility failure, not a replay
quality result and not evidence that the original empty output reproduced. The
planner must decide whether the replay configuration should be redesigned or
whether this diagnostic sidecar should be abandoned. No fix or retry was
attempted.

## Questions for planner

1. Why does this replay initialization see `intermediate_size_per_partition=384`
   when the earlier GPTQ smoke successfully served the same overlay at TP8?
2. Should a new packet reproduce the exact smoke serving configuration before
   attempting the two-cap replay?
