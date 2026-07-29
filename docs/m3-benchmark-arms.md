# MiniMax-M3 benchmark arms — checkpoint / port / recipe table

> New to this? Start with [`M3_COLLABORATOR_GUIDE.md`](../M3_COLLABORATOR_GUIDE.md) —
> environment setup, serving, pitfalls, and known results end to end. This page is the
> arm-provenance reference it points at.

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
| `gptq-base` | GPTQ W4AFP8 | `llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay` | 8000 | Original in-house GPTQ arm (ABI overlay export). The overlay dir is ~116 KB of symlinks onto `…/gptq-checkpoint-vllm-w123`, differing only in `config.json` `ignore` rules (`overlay_provenance.json` records `tensor_payload_unchanged: true`). |
| `r5` | AWQ W4AFP8 | `…/m3-distributed-awq-full/20260720T060340Z-m3-ddp-awq-full-r5-deadchan/awq/MiniMax-M3-awq-W4AFP8/20260720-060402/checkpoint-vllm-w123` | 8004 | First full distributed-AWQ run. Carried the up→down fold later found to be the AWQ regression cause. |
| `r6` | AWQ W4AFP8 | `…/20260723T092202Z-m3-ddp-awq-full-r6-noupdown/awq/MiniMax-M3-awq-W4AFP8/20260723-092256/checkpoint-vllm-w123` | 8004 | **up→down fold removed.** See `M3_AWQ_R6_REQUANT_HANDOFF.md`. |
| `r7` | AWQ W4AFP8 | `…/20260723T123927Z-m3-ddp-awq-full-r7-gatealpha/awq/MiniMax-M3-awq-W4AFP8/20260723-123953/checkpoint-vllm-w123` | 8007 | **Gate-alpha fold** — function-preserving down-side smoothing via the gate path. Requires the gate-alpha overlay in the serving vLLM (`pipeline/slurm/patch_vllm_m3_serve.py`). Passed capture-safe ABI smoke v4 (2026-07-24). See `docs/superpowers/plans/2026-07-23-m3-awq-gate-alpha-fold.md`. |
| `r8-fp8rest` | GPTQ W4AFP8 | `…/m3-distributed-r8-full/20260723T160426Z-m3-ddp-gptq-full-r8-fp8rest/gptq/MiniMax-M3-gptq-W4AFP8/20260723-160454/checkpoint-vllm-w123-v2` | 8005 | r8 with **qkv dequantized to BF16** (dequant-qkv serve fix). |
| `r8-uniformqkv` | GPTQ W4AFP8 | *same r8 quant run*, export `checkpoint-vllm-w123-v3-uniformqkv` | 8006 | r8 with **fully-fp8 fused qkv + indexer** (MXFP8 precedent). Same quantization run as `r8-fp8rest`, different export. |

### Quality status per arm — read this before picking an arm

This column is the one thing the perf tables cannot tell you. Sources:
`M3_OFFICIAL_QUALITY_RESULTS.html` and the run trees under
`results/m3-official-quality/`.

| `M3_ARM` | Quality evidence | Verdict |
|---|---|---|
| `gptq-base` | `full4` 7 tasks (07-20) + `tok64k` 2 tasks (07-21) | **Ship.** Recovery 97.4–101.1% on all seven tasks; spend within 2% of BF16; symmetric flips. The only arm with a breadth verdict. |
| `r5` | `full4` + `tok64k` (07-20/21), sampling probe (07-22) | **Do not serve.** GPQA recovery 71.7%, spend 2.19×, exhaustion 38.9% (BF16 floor 12.6%) from reasoning non-termination. Kept for the root-cause record. |
| `r6` | `tok64k-awqr6` 2 tasks (07-23) + sampling probe (07-23) | **Clean on the tasks measured, balanced.** GPQA 98.7%, IFEval 98.6%; spend 1.13×/1.17×; exhaustion 14.7%/1.5%; flips 11✗/9✓ and 29✗/22✓. No breadth run. |
| `r7` | `tok64k-awqr7` 2 tasks (07-24) | **Clean on GPQA, weaker on IFEval.** GPQA 104.4% (flips 9✗/16✓), IFEval 95.7% (flips 37✗/16✓); spend 0.93×/1.25×. No breadth run. Also needs the gate-alpha serving overlay. |
| `r8-fp8rest`, `r8-uniformqkv` | **none** | **Perf-only.** No quality eval of either export exists. Do not quote `gptq-base`'s recovery figures for these — `r8` is a different recipe (FP8 for non-expert layers). |

> **Two traps this table exists to prevent.** (1) `gptq-base` is the quality-verified
> GPTQ arm; `r8-*` is a *newer but unevaluated* GPTQ recipe — newer is not better-verified.
> (2) The r6/r7 runs cover GPQA-Diamond and IFEval only, so "recovers BF16" is a
> two-task claim for them and a seven-task claim only for `gptq-base`.

### Quantization cost per recipe (measured)

Wall-clock from the run trees; calibration ends when the `checkpoint/` directory first
appears. All runs: 1 node, 8×H100, distributed calibration.

| Run | Method | Calibration | Save + vLLM export | Total |
|---|---|---|---|---|
| `r5-deadchan` | AWQ | 7.20 h | 0.25 h | 7.45 h |
| `r6-noupdown` | AWQ | **2.23 h** | 1.25 h | 3.48 h |
| `r7-gatealpha` | AWQ | 7.53 h | 7.10 h † | 14.63 h |
| `r8a-fp8rest` | AWQ | 2.03 h | 0.35 h | 2.38 h |
| `r8-fp8rest` | GPTQ | **3.12 h** | 10.2 h † (3 exports) | 13.30 h |

Dropping the up→down fold makes r6 ~3.4× cheaper to calibrate than r7 — the fold *was*
a large share of AWQ's cost, so r6 is both the cleaner and the cheaper AWQ.

† **Overlapping runs inflate the save phase.** r7's save (07-23 20:11 → 07-24 03:17) ran
concurrently with the whole of r8 (16:04 → 05:22) and both took 7–10 h to write, against
15 min–1.25 h for every run that had the NFS to itself. Budget the save phase as
~15 min–1.25 h **serialized**, and do not run two full quantizations at once if you care
about wall-clock.

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
