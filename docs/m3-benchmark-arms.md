# MiniMax-M3 benchmark arms — checkpoint / port / recipe table

**This table is the owner of M3 arm provenance.** The benchmarks repo holds a single
in-house profile (`configs/minimax/minimax-m3-inhouse.sh`) plus thin arm bindings; it
deliberately does **not** carry quantization internals. That keeps the benchmarks repo
to benchmarking concerns only, and keeps quant-run details in the repo that produces
them.

## How the benchmarks repo consumes this

The in-house profile takes three required inputs and fails closed without them:

```bash
M3_ARM=r7 \
MODEL_PATH=/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260723T123927Z-m3-ddp-awq-full-r7-gatealpha/awq/MiniMax-M3-awq-W4AFP8/20260723-123953/checkpoint-vllm-w123 \
QUANT_RECIPE=awq-w4afp8-r7 ENDPOINT_PORT=8007 \
PROFILE=minimax-m3-inhouse bash performance/scripts/run_all.sh
```

Results are namespaced `results/minimax-m3-inhouse-<M3_ARM>/`, so arms never overwrite
each other.

**Arm bindings** (`minimax-m3.sh`, `minimax-m3-awq-inhouse{,-r6,-r7}.sh`,
`minimax-m3-gptq-r8-{fp8rest,uniformqkv}.sh`) exist so the launchers in
`pipeline/slurm/` keep reproducing their original runs byte-identically — they set the
three inputs and source the one real profile. **Do not copy a binding to add a new
arm**; pass the inputs instead.

## Arms

| `M3_ARM` | Method | Checkpoint | Profile default port | What changed |
|---|---|---|---|---|
| `gptq-base` | GPTQ W4AFP8 | `llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay` | 8000 | Original in-house GPTQ arm (ABI overlay export). |
| `r5` | AWQ W4AFP8 | `…/m3-distributed-awq-full/20260720T060340Z-m3-ddp-awq-full-r5-deadchan/awq/MiniMax-M3-awq-W4AFP8/20260720-060402/checkpoint-vllm-w123` | 8004 | First full distributed-AWQ run. Carried the up→down fold later found to be the AWQ regression cause. |
| `r6` | AWQ W4AFP8 | `…/20260723T092202Z-m3-ddp-awq-full-r6-noupdown/awq/MiniMax-M3-awq-W4AFP8/20260723-092256/checkpoint-vllm-w123` | 8004 | **up→down fold removed.** See `M3_AWQ_R6_REQUANT_HANDOFF.md`. |
| `r7` | AWQ W4AFP8 | `…/20260723T123927Z-m3-ddp-awq-full-r7-gatealpha/awq/MiniMax-M3-awq-W4AFP8/20260723-123953/checkpoint-vllm-w123` | 8007 | **Gate-alpha fold** — function-preserving down-side smoothing via the gate path. Requires the gate-alpha overlay in the serving vLLM (`pipeline/slurm/patch_vllm_m3_serve.py`). Passed capture-safe ABI smoke v4 (2026-07-24). See `docs/superpowers/plans/2026-07-23-m3-awq-gate-alpha-fold.md`. |
| `r8-fp8rest` | GPTQ W4AFP8 | `…/m3-distributed-r8-full/20260723T160426Z-m3-ddp-gptq-full-r8-fp8rest/gptq/MiniMax-M3-gptq-W4AFP8/20260723-160454/checkpoint-vllm-w123-v2` | 8005 | r8 with **qkv dequantized to BF16** (dequant-qkv serve fix). |
| `r8-uniformqkv` | GPTQ W4AFP8 | *same r8 quant run*, export `checkpoint-vllm-w123-v3-uniformqkv` | 8006 | r8 with **fully-fp8 fused qkv + indexer** (MXFP8 precedent). Same quantization run as `r8-fp8rest`, different export. |

Every in-house arm serves on **1 node, 8×H100, TP8/EP8, W4AFP8**, vLLM 0.24.0 with the
M3 overlay — which is why they share one benchmark profile.

### W4A8 kernel-backend arms (same checkpoint, different MoE kernel)

These arms hold the checkpoint fixed (`gptq-base`'s ABI overlay export) and vary only
the MoE GEMM backend. Humming arms need the patched side-install on `PYTHONPATH`
(see `DECLARED_PATCH_SHA256` in `pipeline/m3_humming_w4a8.py`) and
`VLLM_HUMMING_USE_F16_ACCUM=0`.

| `M3_ARM` | MoE kernel | Extra env | Port used (window `20260725T122256Z`) |
|---|---|---|---|
| `cutlass-w4afp8` | vLLM CUTLASS W4A8 | — | 8000 |
| `humming-w4afp8-indexed` | humming 0.1.10 indexed | `VLLM_HUMMING_MOE_GEMM_TYPE=indexed` | 8005 |
| `humming-w4afp8-grouped` | humming 0.1.10 grouped_contiguous | `VLLM_HUMMING_MOE_GEMM_TYPE=grouped_contiguous` | 8010 |

Perf results + attempt history (the grouped arm took three attempts; the first two are
invalid/confounded by kernel defects): [`m3-w4a8-three-arm-perf.md`](m3-w4a8-three-arm-perf.md).

> **Ports here are the profile *defaults*, not what every run used.** Launchers override
> `ENDPOINT_PORT` per run — e.g. `pipeline/slurm/run_perf_eval_r8r7_srun.sh` serves
> `r8-fp8rest` on 8002 and `r8-uniformqkv` on 8003 so four arms can share a node set.
> Trust the launcher for a given run, not this column.

## Serving

Reproducible recipe: [`m3-serving-recipe.md`](m3-serving-recipe.md) — base
`vllm/vllm-openai:v0.24.0` + `pipeline/slurm/patch_vllm_m3_serve.py` overlay, TP8 on
8×H100, `--tool-call-parser`/`--reasoning-parser minimax_m3`.

## Calibration artifacts (agentic perf shape)

The M3 agentic workload shape in the benchmarks profile was measured from tau2-bench run
`m3_calib_base_20260722T091007Z` (114 tasks / 1540 turns; agent = in-house GPTQ W4AFP8
endpoint, user-sim = DeepSeek-V4-Pro via the Bitdeer gateway). Raw artifacts —
including the derived `ag_block.txt` — are quant-run evidence and live here:

```
/mnt/nfs/hoangduy/results/m3-calibration/20260722T091007Z/calib/
```

Per-turn output is **bimodal** (p50 84 / p95 252 / max 992), so the flat
`AG_OUTPUT_TOKENS=100` mean understates tail decode load.

> **Planned:** re-derive this shape through the benchmarks repo's official seam,
> `performance/shapes/extract.py`, and freeze it into `performance/shapes/library/`.
> That repo's policy is that shapes are measured then frozen, never hand-authored or
> inlined in a profile — our current inline block predates our adoption of that seam.

## Known measurement gap on these arms

aiperf 0.11 reads prompt-cache hits from `usage.prompt_tokens_details.cached_tokens`,
which **vLLM 0.24 does not emit**, so per-turn cache-hit columns stay blank for every M3
arm. Use server-counter diffing instead
(`performance/workloads/scrape_cache_hit.py` — currently SGLang-keyed; needs vLLM
counter names added).
