# MiniMax-M3 shared-expert stream RCA report — 20260710-072629

## Environment

- Host: `h125-gpu-polaris.pod4.lab.bitdeer.ai` (8 × NVIDIA H100 80GB)
- Repository commit: `55434aa756eb005f47b4a8b466b9f10d1a202b4e`
- torch: `2.11.0+cu130`
- vLLM: `0.24.0`
- FlashInfer: `0.6.12`
- Checkpoint: `/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4`
- Patch check: `4/4` patched

## Conditions

| Condition | `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` |
|---|---:|---:|
| control-t256 | 0 | 256 |
| stream-disabled | 1 | 256 |
| control-t128 | 0 | 128 |

All trials used TP=8, expert parallelism, `max_model_len=8192`, GPU utilization
0.9, CUDA graphs and breakable CUDA graphs enabled, and asynchronous CUDA
(`CUDA_LAUNCH_BLOCKING` unset).

## Trial results

| Condition | Trial | Verdict | Chat | Last capture progress |
|---|---|---|---:|---:|
| control-t256 | async_baseline_1 | server_ready | true | 51/51 |
| control-t256 | async_baseline_2 | graph_ima_collective | false | 43/51 |
| control-t256 | async_baseline_3 | graph_ima_collective | false | 43/51 |
| stream-disabled | async_baseline_1 | server_ready | true | 51/51 |
| stream-disabled | async_baseline_2 | server_ready | true | 51/51 |
| stream-disabled | async_baseline_3 | server_ready | true | 51/51 |
| control-t128 | async_baseline_1 | graph_ima_collective | false | 43/51 |
| control-t128 | async_baseline_2 | graph_ima_collective | false | 43/51 |
| control-t128 | async_baseline_3 | graph_ima_collective | false | 45/51 |

Artifacts:

- [`control-t256 summary.json`](20260710-072629-control-t256/summary.json)
- [`stream-disabled summary.json`](20260710-072629-stream-disabled/summary.json)
- [`control-t128 summary.json`](20260710-072629-control-t128/summary.json)
- [`comparison.json`](20260710-072629-comparison.json)

## Classification: narrowed

The stream-disabled control passed capture and chat in all three trials, while
the threshold-256 control reproduced IMA in two of three trials. This makes
the shared-expert stream a strong operational-workaround signal.

The result does not meet the strong-confirmation rule. Threshold-256 failures
occurred at 43/51 rather than the expected 15–17/51, and threshold-128 failures
remained at 43/51, 43/51, and 45/51 rather than moving to 31–33/51. The
threshold fingerprint therefore did not reproduce. The classifier's
`graph_ima_collective` label is not causal evidence on its own.

No bounded rerun was allowed or needed: the threshold-256 control did not pass
all three trials.

## Next action

Do not patch stream synchronization yet. First independently verify the
effective shared-expert threshold in each vLLM worker during graph capture,
then rerun the threshold-256 and threshold-128 controls with that verification.
Pending a source-level fix, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` is the
narrow operational workaround supported by this matrix.
