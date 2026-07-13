# MiniMax-M3 Three-Model Smoke: Ray/BF16 Stage

Run root:

`results/m3-quality/20260712T175323Z-m3-quality-3model-tmux`

## Ray topology result

The corrected Ray preflight completed successfully:

```json
{
  "ready": true,
  "expected_nodes": 2,
  "alive_nodes": 2,
  "visible_gpus": 16.0
}
```

The preflight artifacts are under `ray_preflight/`, including rank logs,
rank JSON, node/status output, and the gate files. This confirms the
two-node/16-GPU topology required by BF16 was available.

## BF16 runtime result

The BF16 arm connected to the validated Ray cluster and initialized vLLM with:

- tensor parallel size: 16;
- distributed backend: Ray;
- MiniMax-M3 BF16 checkpoint;
- eager execution and FP8 KV cache.

It did not reach a quality probe or produce a completed arm artifact. Slurm
terminated the step at `2026-07-12T18:26:51Z` while the vLLM runtime was still
initializing. The BF16 return code was `143`.

The runtime gate and Ray artifacts are under:

`models/bf16/shards/smoke/ray_runtime/`

The complete BF16 stdout/stderr is included under `logs/`.

## Scope

This report covers only the completed Ray/BF16 stage. The GPTQ and AWQ
quantized arms were still running when this snapshot was taken and are not
classified here. No jobs were cancelled or modified.
