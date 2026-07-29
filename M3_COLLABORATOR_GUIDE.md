# MiniMax-M3 quantization & serving — collaborator guide

**Who this is for.** You work on the evaluation side and need to (a) stand up a
MiniMax-M3 quantized endpoint you can trust, and (b) understand what our quantization
pipeline produced and how good it is. **Part A** is the serving path you will use daily.
**Part B** is the quantization pipeline as reference — read it to understand where a
checkpoint came from, or if you need to make one.

Written 2026-07-29. **Assumes you are on the Iceland cluster** with access to
`/mnt/nfs/hoangduy/`. Every path below was verified to exist there on that date. Nothing
here is portable as-written to another cluster — the venvs, checkpoints, and the
`srun`-only scheduler constraint are all Iceland-specific.

**What this document does not own.** It front-doors other docs rather than restating
them, because those are maintained and will drift ahead of this one:

| Topic | Owner |
|---|---|
| Per-arm checkpoints, ports, recipe provenance, quality status, quant cost | [`docs/m3-benchmark-arms.md`](docs/m3-benchmark-arms.md) |
| The vLLM patch overlay in full | [`docs/m3-serving-recipe.md`](docs/m3-serving-recipe.md) |
| Quality results & the AWQ story | [`M3_OFFICIAL_QUALITY_RESULTS.html`](M3_OFFICIAL_QUALITY_RESULTS.html) |
| Serving performance | [`M3_OFFICIAL_PERF_RESULTS.html`](M3_OFFICIAL_PERF_RESULTS.html), [`docs/m3-two-axis-perf.md`](docs/m3-two-axis-perf.md) |
| Speculative decoding | [`M3_OFFICIAL_SPECDEC_RESULTS.html`](M3_OFFICIAL_SPECDEC_RESULTS.html), [`docs/m3-specdec-eagle3.md`](docs/m3-specdec-eagle3.md) |
| Symptom→fix knowledge base (2,100 lines, newest first) | [`BUGS_AND_FIXES.md`](BUGS_AND_FIXES.md) |
| The pipeline's own stage docs | [`pipeline/README.md`](pipeline/README.md) |

---

## Sixty-second orientation

We quantize **MiniMax-M3** (a ~920 GB BF16 VL-MoE reasoning model) to **W4AFP8** — INT4
group-128 weights with dynamic per-token FP8 activations — so it runs on **one 8×H100
node** instead of two. Weights go 796 GB → 225 GB, and decode throughput improves
**~3.8× per GPU** against the BF16 baseline.

The result is a standard `compressed-tensors` checkpoint served by **released vLLM 0.24.0
plus a Python patch overlay** — not a custom fork build, and not a custom serving stack.
You point `vllm serve` at a directory and get an OpenAI-compatible endpoint.

Three things that trip up everyone new:

1. **The patch overlay is mandatory and is not about "our fork".** Stock vLLM, NVIDIA's
   build, *and* the community `toncao` fork all need it. See [A2](#a2-why-vllm-needs-patches).
2. **"Newer recipe" ≠ "better verified".** GPTQ `r8` is newer than `gptq-base` and has no
   quality evaluation at all. See [A1](#a1-pick-an-arm).
3. **`rc=0` is not evidence a run worked.** Gate on completed-request counts and the
   verification steps in [A4](#a4-verify-before-you-trust-it).

---

# Part A — Serving (the path you will use)

## A0. Environment preconditions

This is the most common day-one blocker. There is **no system `pip` or `venv`** on the
nodes; use the prebuilt venvs.

```bash
source /mnt/nfs/hoangduy/env.sh                        # 1. sets $UV, caches, WORK_ROOT
source /mnt/nfs/hoangduy/venvs/quant/bin/activate      # 2. AFTER env.sh so it wins on PATH
export HOME=/mnt/nfs/hoangduy                          # 3. Iceland's HOME is not writable
```

**Order matters** — `env.sh` first, then the venv, then `HOME`.

| venv | Engine | Use for |
|---|---|---|
| **`quant`** | vLLM **0.24.0**, torch 2.11.0, lm-eval 0.4.12 | **Default.** Quantization *and* the qualified M3 serving path |
| `serve` | vLLM 0.23.1rc1.dev643 | Older serving comparisons only |
| `serve-026` | vLLM 0.26.0 + merged humming 0.1.10 | 0.26.0 work — has open blockers, see [A8](#a8-known-broken--do-not-use) |
| `humming-0.1.10-site`, `humming-0.1.11-site` | Patched Humming side-installs | Put on `PYTHONPATH` for Humming-kernel arms. **Never** install into `quant` |
| `sglang-eval` | SGLang 0.5.13.post1 | SGLang-native checkpoints (e.g. GLM-5.2). Do **not** `pip install -e .` here — it upgrades torch and breaks FlashInfer |

**Scheduler.** Iceland accepts top-level **`srun` only** — `sbatch` is not
supported. Long runs must be launched from a persistent detached controller (normally
`tmux`), or they die with your SSH session. (The `.sbatch` files under `pipeline/slurm/`
predate this constraint and do not work here; use the `run_*_srun.sh` launchers.)

**Give your srun step CPUs.** Without an explicit `--cpus-per-task`, Iceland's Slurm 21.08 binds
the whole step to **one physical core** even on an exclusive node. Pass
`--cpus-per-task=192`. This silently serializes dataloading and NCCL progress threads and
looks like "the GPUs are slow."

**Don't edit a launcher while a run is using it.** The arm scripts are re-read per serve,
so an edit mid-run changes later cells and silently invalidates the comparison.

## A1. Pick an arm

`docs/m3-benchmark-arms.md` owns the full table. The short version — and the part that
matters for evaluation work is the **quality** column, which the perf tables cannot tell
you:

| Arm | Method | Quality evidence | Use it? |
|---|---|---|---|
| **`gptq-base`** | GPTQ W4AFP8 | 7 tasks + 2-task 64k depth | ✅ **Default.** Recovery 97.4–101.1% on all seven tasks; spend within 2% of BF16; symmetric flips. The only arm with a **breadth** verdict |
| **`r6`** | AWQ W4AFP8 | 2 tasks (GPQA, IFEval) + sampling probe | ✅ Clean and balanced on what was measured: GPQA 98.7%, IFEval 98.6%. No breadth run |
| `r7` | AWQ W4AFP8 | 2 tasks | ⚠️ GPQA 104.4% but IFEval 95.7%. Also needs the gate-alpha overlay |
| `r5` | AWQ W4AFP8 | 7 tasks + 2-task depth | ❌ **Do not serve.** GPQA recovery 71.7% from reasoning non-termination |
| `r8-fp8rest`, `r8-uniformqkv` | GPTQ W4AFP8 | **none** | ⚠️ Perf-only. Different recipe from `gptq-base` — do not reuse its recovery figures |
| MXFP8 (vendor) | W8A16 | 7 tasks | ✅ Recovery 97.5–100.6%. Useful external control |
| cyankiwi (community) | W4A16 | 7 tasks + depth | ❌ Quality-disqualified; runaway generations |
| BF16 | — | baseline | Reference only — needs **2 nodes, TP16** (Ray) |

Two canonical paths to copy:

```bash
# Default: in-house GPTQ, quality-verified across seven tasks
CKPT=/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay

# In-house AWQ r6
CKPT=/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260723T092202Z-m3-ddp-awq-full-r6-noupdown/awq/MiniMax-M3-awq-W4AFP8/20260723-092256/checkpoint-vllm-w123
```

> The GPTQ path is a **~116 KB directory of symlinks** onto
> `…/gptq-checkpoint-vllm-w123`, differing only in `config.json` `ignore` rules
> (`overlay_provenance.json` records `tensor_payload_unchanged: true`). It is the correct
> thing to serve; it is not a broken or partial copy.

**Never delete a checkpoint.** Each is 215–225 GB and several are irreproducible without
a multi-hour run. Ask first, always.

## A2. Why vLLM needs patches

**The gap.** vLLM does not support M3's W4A8 MoE path out of the box. M3's MoE uses the
SwiGLU-OAI activation in an *uninterleaved* layout; vLLM's CUTLASS W4A8 grouped-GEMM
expert kernel does not declare support for that activation, so kernel selection raises
`NotImplementedError` before a single token is generated. A second gap sits right behind
it: the W4A8 call site passes no clamp scalars, and the activation branch asserts without
them.

**This is not "our fork is missing something".** Both gaps are present in stock vLLM, in
NVIDIA's build, and in the `toncao/vllm@minimax-m3-compressed-tensors` fork. Serving M3
W4A8 was never "use the fork" — it is **"0.24.0 + this overlay"**. The overlay is pure
Python layered on the released wheel's precompiled binaries: **no CUDA recompilation.**

**What the overlay does**, in four groups (full table with file paths and rationale in
`docs/m3-serving-recipe.md`; the authoritative list is `_patch_targets()` in the script):

| Group | Edits | What it buys |
|---|---|---|
| **W4A8 kernel admission** | 2 | Declares `SWIGLUOAI_UNINTERLEAVE` supported and defaults the clamp scalars (`limit=7.0, alpha=1.702, beta=1.0`). **Required always** — without these the model cannot load |
| **CUDA-graph safety** | 5 | Fixes an illegal-memory-access during graph capture, from four separate causes: FlashInfer fused all-reduce, `record_stream` under multi-stream capture, a dropped `torch.cuda.synchronize()` before capture cleanup, and NaN router logits from padding tokens producing out-of-bounds expert IDs (vLLM #39288/#39391). Needed only with graphs **on** |
| **0.26.0 regression** | 1 | Restores head-major `topk_indices_buffer` allocation off SM100. A no-op on 0.24.0 |
| **Arm-specific** | 2 + 1 | Humming MoE-backend admission; gate-alpha fold support (**required for AWQ `r7`**) |

**Removal criteria:** delete the overlay once a vLLM release serves M3 W4A8
(SwiGLU-OAI uninterleaved) natively.

### Applying and checking it

```bash
python pipeline/slurm/patch_vllm_m3_serve.py            # idempotent, fail-loud
python pipeline/slurm/patch_vllm_m3_serve.py --check    # exit 1 if unpatched
```

Run `--check` **before every serve**. Expected tail on a healthy `quant` venv:

```
vLLM 0.24.0 at /mnt/nfs/hoangduy/venvs/quant/lib/python3.12/site-packages/vllm
already patched: .../fused_moe/experts/cutlass_moe.py
... (eight required edits) ...
r7 gate-alpha: utils.py: already injected
STATUS: patched
```

The overlay edits `site-packages` on purpose: vLLM worker subprocesses are spawned fresh,
so in-process monkeypatches never reach `Worker_TP*`.

## A3. Serve it

**Free the GPUs first.** A crashed run leaves workers holding ~70 GiB/GPU and the next
serve dies with a confusing `Free memory on device cuda:X ... less than
gpu_memory_utilization`:

```bash
bash pipeline/slurm/free_gpus.sh          # kills only YOUR leftovers, then verifies
FORCE=0 bash pipeline/slurm/free_gpus.sh  # verify only, kill nothing
```

It never touches another user's processes, and it exits 1 if the GPUs are still occupied
by anyone.

**The serve command** (1 node, 8×H100, TP8 + expert parallelism):

```bash
vllm serve "$CKPT" \
  --served-model-name MiniMaxAI/MiniMax-M3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --block-size 128 \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --tool-call-parser minimax_m3 --reasoning-parser minimax_m3 --enable-auto-tool-choice \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

`--enforce-eager` (CUDA graphs off) is the conservative default and matches the profile
the paired quality evals ran under. Dropping it turns graphs on for more speed, but only
after confirming the graph-safety edits *and* their env knobs are set — notably
`LLMC_M3_CAPTURE_SYNC=sync`, without which the capture IMA survives every other mitigation.

**If you are driving the benchmarks repo**, do not hand-write the command — use the
profile seam, which fails closed without its three inputs:

```bash
M3_ARM=r6 \
MODEL_PATH="$CKPT" \
QUANT_RECIPE=awq-w4afp8-r6 ENDPOINT_PORT=8004 \
PROFILE=minimax-m3-inhouse bash performance/scripts/run_all.sh
```

Results are namespaced `results/minimax-m3-inhouse-<M3_ARM>/`, so arms never overwrite
each other. **Do not copy an existing arm binding to add a new arm** — pass the inputs.

## A4. Verify before you trust it

M3 has a specific, nasty failure mode: **the checkpoint loads, the server answers, and
the output is garbage** (empty strings or `\r\n` repetitions). It comes from a dropped
MoE-router `ignore` rule (see [A5](#a5-symptom--cause--fix)) and it will quietly poison an
eval run. Check these four things:

```bash
# 1. Overlay applied
python pipeline/slurm/patch_vllm_m3_serve.py --check && echo OVERLAY-OK

# 2. The checkpoint's ignore list retains the router, lm_head, vision, indexer
python -c "
import json,sys
ig=json.load(open('$CKPT/config.json'))['quantization_config']['ignore']
print('ignore:', ig)
assert any('gate' in p for p in ig), 'MoE ROUTER MISSING FROM ignore -- output will be garbage'
print('IGNORE-OK')"

# 3. The model actually answers coherently
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"MiniMaxAI/MiniMax-M3",
  "messages":[{"role":"user","content":"Name three prime numbers."}],
  "max_tokens":128}' | python -m json.tool

# 4. Record the fingerprint that identifies this exact serving stack
#    (printed in the server log; looks like vllm-0.24.0-tp8-ep-<hash>)
grep -o 'vllm-[0-9.]*-tp[0-9]*-ep-[a-f0-9]*' serve.log | head -1
```

**`rc=0` is not evidence.** We have had `aiperf` exit 0 with every single request
returning 500. Always gate on the **completed-request count**, not the exit code.

## A5. Symptom → cause → fix

Look up what you actually see. `BUGS_AND_FIXES.md` has the long-form post-mortems.

| Symptom | Cause | Fix |
|---|---|---|
| Empty output, or `\r\n` repeated | The MoE router (`mlp.gate`) got pruned from the saved `quantization_config.ignore`. llm-compressor drops ignore patterns that didn't match a *quantized* module, so vLLM treats the unquantized router as quantized → broken routing | Verify step 2 in [A4](#a4-verify-before-you-trust-it). `quantize.py` now re-adds them via `_persist_ignore_to_config()`; older checkpoints need the repair snippet in `pipeline/README.md` |
| `NotImplementedError` on activation at startup | Overlay not applied (edits 1–2) | `python pipeline/slurm/patch_vllm_m3_serve.py` |
| Illegal memory access under concurrency, graphs **on** | One of four capture bugs — most often the dropped pre-capture `synchronize()` | Set `LLMC_M3_CAPTURE_SYNC=sync`; or serve `--enforce-eager` |
| Illegal memory access on **0.26.0**, even at k=0 | `topk_indices_buffer` layout regression | Overlay edit 8. See `docs/m3-026-topk-buffer-layout.md` |
| `Free memory on device cuda:X ... less than gpu_memory_utilization` | Leftover workers from a crashed run holding ~70 GiB/GPU | `bash pipeline/slurm/free_gpus.sh` |
| A CUDA error names a kernel that makes no sense | **CUDA errors are sticky** — the reported kernel is not the faulting one | Re-run with `CUDA_LAUNCH_BLOCKING=1` to pin the real site. Never trust the first-reported kernel |
| Benchmark "passes" but numbers are absurd | Exit code ignored the fact that all requests failed | Gate on completed count |
| Cross-node NCCL/gloo hangs at init | Hostnames don't route between nodes | `export NCCL_SOCKET_IFNAME=intranet` (and the gloo equivalent) |
| Model won't load: expert-width error | The CUTLASS W4A8 MoE kernel needs each expert's `moe_intermediate_size` divisible by **256** | See [B4](#b4-pre-quantization-static-gates). Sharding cannot fix it; only padding or a scheme change |
| Prompt-cache hit columns are blank | aiperf 0.11 reads `usage.prompt_tokens_details.cached_tokens`, which **vLLM 0.24 never emits** — prefix caching *is* on | Use warm-vs-cold deltas or server-counter diffing, not the usage field |
| SGLang won't load our W4AFP8 checkpoint | W4AFP8 MoE support in SGLang is an open, unmerged PR | Use vLLM. See [A8](#a8-known-broken--do-not-use) |

## A6. What performance to expect

**Quote per-user output speed (1/ITL) first.** It is what a user feels. Server-aggregate
throughput is secondary and can look good while every individual request is slow.

Measured on 8×H100, in-house GPTQ W4AFP8 on the Humming indexed kernel, window
`20260726T132617Z` (source: `docs/m3-two-axis-perf.md`):

| Concurrency | TPOT p50 (ms) | Output speed (tok/s/user) | Total output tok/s | vs BF16 per GPU |
|---|---|---|---|---|
| 1 | 7.30 | **137** | 137 | 3.4× |
| 4 | 8.82 | 113 | 451 | 3.7× |
| 16 | 12.23 | 82 | 1300 | 3.7× |
| 64 | 19.40 | 52 | 3267 | **3.8×** |

TPOT p50 and total output tok/s are the measured aiperf fields (`inter_token_latency`,
`output_token_throughput`); output speed is `1000 ÷ TPOT`. On a pinned-output shape the two
agree, but under natural output lengths prefer aiperf's own
`output_token_throughput_per_user` — the mean of per-request 1/ITL is not the reciprocal of
the mean ITL, and the two diverge by ~1% once speculative decoding adds per-request variance.

Against the BF16 baseline (16 GPUs, 2 nodes): weights 225 GB vs 796 GB, and **0.68 vs
2.61 GPU-hours per 1M tokens** — 3.8× cheaper.

Two independent multipliers stack on top of the format choice:

- **MoE kernel.** Humming indexed 0.1.10 beats vLLM's CUTLASS W4A8 by **+34% at conc 1**
  (137 vs 102 tok/s/user), compressing to +24–27% at conc 10. Requires the patched
  Humming side-install on `PYTHONPATH`.
- **Speculative decoding (EAGLE3).** A further 1.18–2.53× on top, workload-dependent:
  best case 345.9 tok/s/user for a single user on code-shaped work, floor 1.50× on loaded
  creative-writing traffic. **There is no single best draft depth** — the optimum moves
  with load and prompt entropy. Read `M3_OFFICIAL_SPECDEC_RESULTS.html` before enabling it.

> **Do not compare numbers across serving configs.** Kernel, concurrency, workload tier,
> graphs on/off, and draft depth each move throughput by more than the differences you are
> likely trying to measure. Record the config next to every number.

## A7. Reproducibility obligations

If you publish a score, these must be recorded — this is a hard requirement of the repo's
evaluation-harness contract, not a nicety.

**Serving stack:** base vLLM version, overlay `--check` status, the `system_fingerprint`
(`vllm-0.24.0-tp8-ep-<hash>`), checkpoint path **and** hash, TP/EP topology, graphs
on/off, `--enforce-eager`.

**Harness:** tokenizer and chat-template hashes, task aliases and harness version,
few-shot counts, metrics, generation/sampling parameters, reasoning mode
(thinking on/off), sample-manifest hash.

**And state comparability explicitly.** Our quality numbers are **valid for
quant-vs-quant and quant-vs-BF16 decisions** — all arms run an identical protocol. They
are **not directly comparable to public leaderboards**: we use greedy decoding (MiniMax's
own recipe is temperature 1.0 / top_p 0.95) and a task subset. A paired subset can be
perfectly valid for a model-to-model decision without being a leaderboard score. Never
conflate the two.

Known gap to close, not hide: the quality-eval `run_manifest.json` currently records only
`lm_eval_version`; vLLM version and patch status live in the serving-diagnostic run dirs.

## A8. Known-broken / do-not-use

- **vLLM 0.26.0** (`serve-026`) has open blockers beyond the topk-buffer fix — see
  `docs/m3-dspark-blockers-026.md`. 0.24.0 + overlay is the qualified path.
- **SGLang cannot load our W4AFP8 MoE checkpoints.** Support is an unmerged PR
  ([sgl-project/sglang#21741](https://github.com/sgl-project/sglang/pull/21741)). The
  checkpoint format itself is engine-agnostic — there is no "SGLang format" to convert to
  — but stock SGLang will mis-route to `CompressedTensorsWNA16MoE`. Standardize on
  **W4AFP8** for anything that must eventually run on both engines; NVIDIA W4A8 with INT8
  activations is effectively vLLM-only today.
- **AWQ `r5`** — serving it will produce reasoning loops that exhaust the generation
  budget and emit nothing. Use `r6`.
- **GPTQ `r8-*`** — no quality evaluation exists. Perf comparisons only.
- **Never install Humming into the `quant` venv.** Use the side-install on `PYTHONPATH`.

## A9. Reporting a problem so it's actionable

Attach: `serve.log`, the output of `patch_vllm_m3_serve.py --check`, `cell-config.txt` (if
from a benchmark arm), the checkpoint path, and the `system_fingerprint`. Raw evidence
lives under `/mnt/nfs/hoangduy/results/<study>/<UTC-timestamp>/`; point at the run
directory rather than pasting fragments.

---

# Part B — The quantization pipeline (reference)

## B1. What it does

Two separate lifecycles meet at the checkpoint. You almost certainly only touch the
second.

```
PRODUCE (once, hours, 8×H100)
  config.yaml → calibration → oneshot (GPTQ/AWQ) → save pack-quantized
             → persist `ignore` into config.json → vLLM re-export → verify
                                    │
                                    ▼
                          [ compressed-tensors checkpoint ]   ← the interface
                                    │
CONSUME (repeatedly, minutes)       ▼
  patch overlay → vllm serve → OpenAI-compatible endpoint → your eval
```

The pipeline is config-driven and one-command:

```bash
python -m pipeline.run --config pipeline/configs/minimax_m3.yaml
```

Stages are `quantize → serve-verify → eval-gate`; run one with
`--stage quantize|serve|eval`, reuse a checkpoint with `--checkpoint <dir>`, override any
field with `--set a.b.c=value`. Output lands in `artifacts/<run_slug>/<timestamp>/`
with `checkpoint/`, the resolved `config.yaml`, `recipe.json`, `metadata.json` (git SHA,
package versions, GPU SM), `serve_report.json`, `eval_report.json`.

`quantization.method` ∈ {`gptq`, `awq`, `smoothquant+{gptq,awq}`, `autoround`,
`spinquant+{gptq,awq}`}; `quantization.scheme` ∈ {`W4AFP8`, `W4A8`}. Both Hopper-native,
saved as `pack-quantized`.

## B2. Launching a full M3 quantization

Multi-GPU (DDP) calibration for **both** AWQ and GPTQ already exists in this fork — it
just needs a `torchrun`-style launch, which the wrapper handles. Do not write bespoke
expert-parallel code; that detour is documented in `CLAUDE.md` as a worked example of what
not to do.

Always dry-run first (prints the exact `srun` command and runs the preflights):

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-ddp-awq-full-r6-noupdown"
DRY_RUN=1 METHODS=awq EVIDENCE_ONLY=0 \
  CONFIG=pipeline/configs/minimax_m3_distributed_awq_full.yaml \
  RESULT_ROOT=/mnt/nfs/hoangduy/results/m3-distributed-awq-full/$RUN_ID \
  LOG_ROOT=/mnt/nfs/hoangduy/logs/m3-distributed-awq-full/$RUN_ID \
  OFFLOAD_ROOT=/mnt/nfs/hoangduy/offload/m3-distributed-awq-full/$RUN_ID \
  bash pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh
```

Drop `DRY_RUN=1` to launch, **from a `tmux` controller**. `METHODS="gptq awq"` runs both
concurrently on separate exclusive nodes. `EVIDENCE_ONLY=0` is what saves a checkpoint
(the default `1` writes metrics only).

Preflights that exist because they each cost us a failed multi-hour run:

- **`/dev/shm` must hold the whole checkpoint (~869 GB), not an IPC floor.** The
  distributed CPU offload keeps one full shared model copy in `/dev/shm`. A 128 GB floor
  once let AWQ launch on a node with 213 GB free and die mid-load. `MIN_SHM_AVAILABLE_BYTES=auto`
  sizes the gate from the safetensors index.
- **`CPUS_PER_TASK=192`** — see the one-core binding trap in [A0](#a0-environment-preconditions).
- **`MIN_MEM_AVAILABLE_BYTES`** guards host RAM.

## B3. Known results

### Quantization cost (measured, 1 node 8×H100)

Calibration ends when the `checkpoint/` directory first appears.

| Run | Method | Calibration | Save + vLLM export | Total |
|---|---|---|---|---|
| `r5-deadchan` | AWQ | 7.20 h | 0.25 h | 7.45 h |
| **`r6-noupdown`** | AWQ | **2.23 h** | 1.25 h | 3.48 h |
| `r7-gatealpha` | AWQ | 7.53 h | 7.10 h † | 14.63 h |
| `r8a-fp8rest` | AWQ | 2.03 h | 0.35 h | 2.38 h |
| **`r8-fp8rest`** | GPTQ | **3.12 h** | 10.2 h † (3 exports) | 13.30 h |

Rules of thumb: **GPTQ ≈ 3 h, AWQ ≈ 2–7.5 h depending on recipe.** The AWQ spread is real
and mechanistic — dropping the up→down fold (r6) removes most of AWQ's smoothing search,
making it ~3.4× cheaper to calibrate than r7.

† **Two concurrent full runs inflate the save phase to 7–10 h**, versus 15 min–1.25 h for
any run that has the NFS to itself. r7's save overlapped the whole of r8. Budget the save
phase serialized, and don't run two full quantizations at once if wall-clock matters.

The project target is a full quantization in **4–8 h** (goal 1, `PROJECT_GOALS.md`);
calibration is comfortably inside that, and the save/export phase is what needs attention.

### Quality (paired vs BF16, 64k budget, greedy, thinking on)

BF16 baseline: GPQA-Diamond 0.803, IFEval 0.893.

| Arm | GPQA | rec % | exhausted | spend | IFEval | rec % | exhausted | spend |
|---|---|---|---|---|---|---|---|---|
| BF16 | 0.803 | — | 12.6% | 1.00× | 0.893 | — | 0.9% | 1.00× |
| GPTQ `gptq-base` | 0.803 | 100.0 | 11.6% | 1.01× | 0.874 | 97.9 | 1.1% | 1.02× |
| MXFP8 | 0.828 | 103.1 | 7.6% | 0.80× | 0.885 | 99.1 | 1.3% | 1.08× |
| AWQ `r5` | 0.576 | 71.7 | 38.9% | 2.19× | 0.811 | 90.9 | 9.2% | 3.69× |
| **AWQ `r6`** | 0.793 | **98.7** | 14.7% | 1.13× | 0.880 | **98.6** | 1.5% | 1.17× |
| AWQ `r7` | 0.838 | 104.4 | 10.6% | 0.93× | 0.854 | 95.7 | 1.9% | 1.25× |
| cyankiwi | 0.455 | 56.7 | 55.6% | 3.06× | 0.782 | 87.6 | 13.7% | 5.09× |

Across all **seven** tasks (32k budget), GPTQ recovers 97.4–101.1% and MXFP8 97.5–100.6%.
**The seven-task AWQ column is `r5` only** — r6/r7 were measured on the two damaged tasks.

**Read `exhausted` before `score`.** It is the fraction of responses that hit the
generation ceiling without emitting a final answer, and it is the metric that identifies
*why* a score moved. AWQ r5's 30-point GPQA loss was not lost capability: when it answered,
its accuracy matched BF16. It fell into reasoning loops, burned the budget, and emitted
nothing — which scores zero. The root cause was AWQ folding smoothing scales through M3's
**up→down** path, which is not function-preserving across the clamped GLU
`(clamp(up)+1)·glu`. Removing the fold collapsed token spend from 2.19×/3.69× back to
~1.1×, which is the proof it was the same defect. Full story with transcripts:
`M3_OFFICIAL_QUALITY_RESULTS.html`.

## B4. Pre-quantization static gates

Two checks fail fast, before any GPU time:

**1. Expert width must be a multiple of 256.** The vLLM CUTLASS W4A8 grouped-GEMM MoE
kernel requires each routed expert's `moe_intermediate_size` divisible by 256, on the
**per-partition** width. With expert parallelism each rank holds whole experts so the full
width applies; with plain tensor parallelism the width divides by TP and can break the
requirement (768 with TP=2 → 384). Sharding cannot rescue a non-256 width — only padding
or a scheme change.

```bash
python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('<model_id>', trust_remote_code=True); print(getattr(c,'moe_intermediate_size', getattr(getattr(c,'text_config',c),'moe_intermediate_size',None)))"
```

Qwen1.5-MoE-A2.7B is 1408 (5.5 × 256) → incompatible. Qwen3-30B-A3B is 768 (3 × 256) → fine.

**2. The `ignore` list must survive serialization** — the garbage-output bug in
[A5](#a5-symptom--cause--fix). `quantize.py` handles this now; verify it on any checkpoint
you didn't produce.

Also available: `pipeline/prequant_compatibility.py` (model-level gate),
`pipeline/verify_quant_checkpoint.py`, and `pipeline/m3_checkpoint_scale_audit.py`.

---

## Glossary — read this once, it prevents real confusion

| Term | Meaning |
|---|---|
| **W4AFP8** | INT4 group-128 weights + dynamic **per-token FP8 (E4M3)** activations. Our production scheme |
| **W4A8** | In this repo, used **almost interchangeably with W4AFP8** — most "W4A8" references mean the FP8-activation variant. A genuinely distinct INT4 + **INT8**-activation W4A8 also exists and is effectively vLLM-only. When it matters, check `config.json` |
| **W4A16** | INT4 weights, 16-bit activations (Marlin kernels). cyankiwi's format |
| **`pack-quantized`** | The compressed-tensors on-disk format for packed INT4. Engine-agnostic |
| **arm** | One (checkpoint × kernel × topology) configuration under test. Registry: `docs/m3-benchmark-arms.md` |
| **recovery %** | candidate score ÷ baseline score on the same task |
| **flips** | Individual answers that changed direction vs baseline. Symmetric = noise; one-sided = damage |
| **exhausted** | Response hit the generation ceiling without emitting a final answer → scores zero |
| **CUTLASS / Humming / Marlin / Machete** | Alternative GEMM kernel backends. Kernel choice alone moves throughput ~34% |
| **EAGLE3 / draft depth `k`** | Speculative decoding and how many tokens the drafter proposes per step |
| **overlay** | The Python patch set applied to installed vLLM ([A2](#a2-why-vllm-needs-patches)); also, separately, a symlink directory that changes only a checkpoint's `config.json` |
| **`r5`…`r8a`** | Sequential in-house quantization recipe revisions. Higher is *newer*, **not** necessarily better-verified |

## Working agreements

- **Never delete a checkpoint** without asking. 215–225 GB each; several are
  irreproducible without a multi-hour run.
- **Results live under `/mnt/nfs/hoangduy/`**, not in the repo. Commit small evidence
  files only.
- **`srun` from tmux; never `sbatch`.**
- **Don't edit a launcher mid-run.**
- **Fix a broken comparison arm; don't drop it.** A failing arm is the task, not a reason
  to narrow scope.
- **Report outcomes honestly** — if a gate failed or a step was skipped, say so with the
  raw output. Raw evidence is preserved precisely so conclusions can be re-derived.
