# MiniMax-M3 quantization & serving — collaborator guide

**Who this is for.** You work on evaluation and need to (a) stand up a MiniMax-M3
quantized endpoint you can trust, and (b) understand what our quantization pipeline
produced and how good it is. **Part A** is the serving path you will use daily. **Part B**
is the pipeline as reference — read it to understand where a checkpoint came from, or if
you need to make one.

Written 2026-07-29; every command in Part A was executed end-to-end on 2026-07-31.
**Iceland cluster only** — the venvs, checkpoints, and `srun`-only scheduler constraint
do not port anywhere else. You need access to `/mnt/nfs/hoangduy/`.

**Where the details live.** This guide front-doors other docs rather than restating them;
when in doubt, the owner doc wins:

| Topic | Owner |
|---|---|
| Per-arm checkpoints, ports, recipes, quality status, quant cost | [`docs/m3-benchmark-arms.md`](docs/m3-benchmark-arms.md) |
| The vLLM patch overlay in full | [`docs/m3-serving-recipe.md`](docs/m3-serving-recipe.md) |
| Quality results & the AWQ story | [`M3_OFFICIAL_QUALITY_RESULTS.html`](M3_OFFICIAL_QUALITY_RESULTS.html) |
| Serving performance | [`M3_OFFICIAL_PERF_RESULTS.html`](M3_OFFICIAL_PERF_RESULTS.html), [`docs/m3-two-axis-perf.md`](docs/m3-two-axis-perf.md) |
| Speculative decoding | [`M3_OFFICIAL_SPECDEC_RESULTS.html`](M3_OFFICIAL_SPECDEC_RESULTS.html), [`docs/m3-specdec-eagle3.md`](docs/m3-specdec-eagle3.md) |
| Symptom→fix knowledge base (newest first) | [`BUGS_AND_FIXES.md`](BUGS_AND_FIXES.md) |
| Pipeline stage docs | [`pipeline/README.md`](pipeline/README.md) |

---

## Sixty-second orientation

We quantize **MiniMax-M3** (a ~920 GB BF16 VL-MoE reasoning model) to **W4AFP8** — INT4
group-128 weights, dynamic per-token FP8 activations — so it runs on **one 8×H100 node**
instead of two. Weights shrink 796 GB → 225 GB; decode throughput improves **~3.8× per
GPU** over BF16.

The result is a standard `compressed-tensors` checkpoint served by **released vLLM 0.24.0
plus a Python patch overlay** — no fork build, no custom serving stack. You point
`vllm serve` at a directory and get an OpenAI-compatible endpoint.

Three things that trip up everyone new:

1. **The patch overlay is mandatory**, and it is not a quirk of our fork — stock vLLM,
   NVIDIA's build, and the community `toncao` fork all need it. See [A2](#a2-why-vllm-needs-patches).
2. **"Newer recipe" ≠ "better verified".** GPTQ `r8` is newer than `gptq-base` and has no
   quality evaluation at all. See [A1](#a1-pick-an-arm).
3. **`rc=0` is not evidence a run worked.** Gate on completed-request counts and the
   checks in [A4](#a4-verify-before-you-trust-it).

---

# Part A — Serving (the path you will use)

## A0. Environment preconditions

The most common day-one blocker. There is **no system `pip` or `venv`** on the nodes; use
the prebuilt venvs:

```bash
source /mnt/nfs/hoangduy/env.sh                        # 1. sets $UV, caches, WORK_ROOT
source /mnt/nfs/hoangduy/venvs/quant/bin/activate      # 2. AFTER env.sh so it wins on PATH
export HOME=/mnt/nfs/hoangduy                          # 3. Iceland's HOME is not writable
```

**Order matters** — `env.sh`, then the venv, then `HOME`. Run everything in this guide
from the **repo root** (`/mnt/nfs/hoangduy/projects/llm-compressor`); the
`python -m pipeline.…` commands fail from anywhere else.

| venv | Engine | Use for |
|---|---|---|
| **`quant`** | vLLM **0.24.0**, torch 2.11.0, lm-eval 0.4.12 | **Default.** Quantization *and* the qualified M3 serving path |
| `serve` | vLLM 0.23.1rc1.dev643 | Older serving comparisons only |
| `serve-026` | vLLM 0.26.0 + merged humming 0.1.10 | 0.26.0 work — has open blockers, see [A8](#a8-known-broken--do-not-use) |
| `humming-0.1.10-site`, `humming-0.1.11-site` | Patched Humming side-installs | Put on `PYTHONPATH` for Humming arms. **Never** install into `quant` |
| `sglang-eval` | SGLang 0.5.13.post1 | SGLang-native checkpoints (e.g. GLM-5.2). No `pip install -e .` — it upgrades torch and breaks FlashInfer |
| `benchmarks` | lm-eval **0.4.10** + openai (HTTP client only) | Drives lm-eval against a running endpoint. Its lm-eval is older than `quant`'s — don't assume versions match |

You'll also see `quant-sub4`, `serve-sub4`, and `humming-main-site`: those belong to a
separate sub-4-bit (W2A16) track on a different model line — not for M3 work. `perf`
holds aiperf; `main` and `quant-tf514-trial` are old experiments.

**Scheduler.** Iceland accepts top-level **`srun` only** — no `sbatch`. Launch long runs
from `tmux` or they die with your SSH session. Ignore the `.sbatch` files and `submit_*.sh`
wrappers under `pipeline/slurm/`; they predate this constraint. Use the `run_*_srun.sh`
launchers.

**Give your srun step CPUs.** Without `--cpus-per-task`, Slurm binds the whole step to
**one physical core** even on an exclusive node — dataloading and NCCL serialize and it
looks like "the GPUs are slow". Pass `--cpus-per-task=192`.

**Don't edit a launcher while a run is using it.** Arm scripts are re-read per serve; a
mid-run edit silently changes later cells and invalidates the comparison.

## A1. Pick an arm

`docs/m3-benchmark-arms.md` owns the full table. The short version — the **quality**
column is what the perf tables can't tell you:

| Arm | Method | Quality evidence | Use it? |
|---|---|---|---|
| **`gptq-base`** | GPTQ W4AFP8 | 7 tasks + 2-task 64k depth | ✅ **Default.** Recovery 97.4–101.1% on all seven tasks; spend within 2% of BF16; symmetric flips. The only arm with a breadth verdict |
| **`r6`** | AWQ W4AFP8 | 2 tasks + sampling probe | ✅ Clean on what was measured: GPQA 98.7%, IFEval 98.6%. No breadth run |
| `r7` | AWQ W4AFP8 | 2 tasks | ⚠️ GPQA 104.4% but IFEval 95.7% with one-sided flips (37✗/16✓). Needs the gate-alpha overlay |
| `r5` | AWQ W4AFP8 | 7 tasks + depth | ❌ **Do not serve.** GPQA recovery 71.7% from reasoning non-termination |
| `r8-fp8rest`, `r8-uniformqkv` | GPTQ W4AFP8 | **none** | ⚠️ Perf-only. One quant run, two exports; different recipe from `gptq-base` — don't reuse its recovery figures |
| MXFP8 (vendor) | W8A16 | 7 tasks | ✅ Recovery 97.5–100.6%. Useful external control |
| cyankiwi (community) | W4A16 | 7 tasks + depth | ❌ Quality-disqualified; runaway generations |
| BF16 | — | baseline | Reference only — needs **2 nodes, TP16** (Ray) |

An **arm is not a checkpoint**: the registry also lists three kernel-backend arms that all
serve the `gptq-base` checkpoint on different MoE kernels. You pick the kernel at serve
time — see [A3](#a3-serve-it).

Two canonical paths to copy:

```bash
# Default: in-house GPTQ, quality-verified across seven tasks
CKPT=/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay

# In-house AWQ r6
CKPT=/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260723T092202Z-m3-ddp-awq-full-r6-noupdown/awq/MiniMax-M3-awq-W4AFP8/20260723-092256/checkpoint-vllm-w123
```

> The GPTQ path is a tiny directory of **symlinks** onto `…/gptq-checkpoint-vllm-w123`,
> differing only in `config.json` `ignore` rules. It is the correct thing to serve, not a
> broken copy.

**Never delete a checkpoint.** Each is 215–225 GB and several are irreproducible without a
multi-hour run. Ask first, always.

## A2. Why vLLM needs patches

vLLM cannot serve M3's W4A8 MoE out of the box: M3 uses the SwiGLU-OAI activation in an
*uninterleaved* layout, which vLLM's CUTLASS W4A8 expert kernel doesn't declare support
for — startup dies with `NotImplementedError` before generating a token. A second gap sits
behind it (missing clamp scalars), and several CUDA-graph capture bugs behind that.

This is not a defect of "our fork" — stock vLLM, NVIDIA's build, and the
`toncao/vllm@minimax-m3-compressed-tensors` fork all have the same gaps. The fix is
**released vLLM 0.24.0 + this overlay**: pure-Python edits on the installed wheel, no CUDA
recompilation. Delete the overlay once a vLLM release serves M3 W4A8 natively.

What it does, in four groups (file-level detail in `docs/m3-serving-recipe.md`; the
authoritative list is `_patch_targets()` in the patch script):

| Group | Edits | What it buys |
|---|---|---|
| **W4A8 kernel admission** | 2 | Declares `SWIGLUOAI_UNINTERLEAVE` supported, defaults the clamp scalars. **Required always** — the model cannot load without them |
| **CUDA-graph safety** | 5 | Fixes an illegal-memory-access during graph capture (four causes, incl. NaN router logits → out-of-bounds expert IDs). Graphs are on for every published result, so treat as required |
| **0.26.0 regression** | 1 | Restores the `topk_indices_buffer` layout off SM100. No-op on 0.24.0 |
| **Arm-specific** | 2 + 1 | Humming backend admission; gate-alpha fold (**required for AWQ `r7`**) |

Plus one optional diagnostics edit whose anchor only exists on 0.26.0; it never gates the
patched/unpatched status.

### Applying and checking it

```bash
python pipeline/slurm/patch_vllm_m3_serve.py            # idempotent, fail-loud
python pipeline/slurm/patch_vllm_m3_serve.py --check    # non-zero if unhealthy
```

Run `--check` **before every serve**. Healthy output on the `quant` venv ends like this:

```
vLLM 0.24.0 at /mnt/nfs/hoangduy/venvs/quant/lib/python3.12/site-packages/vllm
flashinfer 0.6.12
already patched: .../fused_moe/experts/cutlass_moe.py
... (nine `already patched` lines: 8 required edits + the optional one) ...
MoE quality probe: model.py: already injected; model.py: already injected
r7 gate-alpha: utils.py: already injected
STATUS: patched
```

The overlay edits `site-packages` deliberately: vLLM workers are spawned fresh, so
in-process monkeypatches never reach them.

## A3. Serve it

**Get a node.** Allocate an 8×H100 node from `tmux` (the allocation dies with your SSH
session):

```bash
srun -w <node> --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
  --time=04:00:00 --pty bash
```

Beware: **`sinfo` "idle" does not mean the GPUs are free** — nodes carry non-Slurm
processes the scheduler can't see. Let `free_gpus.sh` below be the judge; if the occupants
are another user's, pick a different node — never kill them.

**Free the GPUs.** A crashed run leaves workers holding ~70 GiB/GPU, and the next serve
dies with a confusing `Free memory on device cuda:X ...` error:

```bash
bash pipeline/slurm/free_gpus.sh          # kills only YOUR leftovers, then verifies
FORCE=0 bash pipeline/slurm/free_gpus.sh  # verify only, kill nothing
```

It never touches another user's processes and exits 1 if the GPUs are still occupied by
anyone. It **will** kill your own *other* vLLM processes on the node — don't run it where
you have a healthy serve you want to keep.

**Serve with the repo launcher, not a hand-written command.**
`pipeline/slurm/run_vllm_http_serve_smoke.sh` produced every published M3 serving result:
it runs the `--check` preflight, sets the graph-safety env knobs, attaches the parsers,
and applies the gate-alpha overlay when needed.

```bash
CKPT="$CKPT" SERVED_NAME=MiniMaxAI/MiniMax-M3 PORT=8000 \
MAX_MODEL_LEN=65536 \
LOG=serve.log PID_FILE=serve.pid \
  bash pipeline/slurm/run_vllm_http_serve_smoke.sh |& tee launcher.log
```

Keep the `tee` — the launcher prints its effective config (including which MoE kernel it
chose) to stdout, while `serve.log` gets only vLLM's output.

The launcher **returns immediately** after backgrounding the server; it does not wait or
smoke-test. Loading 225 GB from NFS takes **~10–30 min** — wait before the A4 checks:

```bash
until curl -sf localhost:8000/v1/models >/dev/null; do
  kill -0 "$(cat serve.pid)" || { echo "serve died — check serve.log"; break; }
  sleep 10
done
```

Once it's up, `pipeline/slurm/smoke_chat_completions.sh` is the quick smoke.

If you must serve by hand, this is what the launcher runs:

```bash
vllm serve "$CKPT" \
  --served-model-name MiniMaxAI/MiniMax-M3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --block-size 128 \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --disable-custom-all-reduce \
  --language-model-only \
  --tool-call-parser minimax_m3 --reasoning-parser minimax_m3 --enable-auto-tool-choice \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

Keep every flag — all published results used them (`--language-model-only` skips the VL
multimodal budget).

**CUDA graphs are ON, by design.** There is no `--enforce-eager` above; every published M3
number — quality and performance — was measured with graphs on
(`CUDAGraphMode.FULL_AND_PIECEWISE`). Serve eager and you are measuring a different
configuration. Graphs-on needs these, which the launcher sets for you:

| Env | Launcher default | Why |
|---|---|---|
| `LLMC_M3_CAPTURE_SYNC` | **`sync`** | Restores a pre-capture `torch.cuda.synchronize()`; without it the capture crashes with an illegal memory access |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | `0` (stream on) | Stream-off was the old workaround; obsolete since the capture-sync fix |
| `ENFORCE_EAGER` | `0` | `1` is an escape hatch only — it skips capture and changes the perf profile |

Other launcher defaults: `TP=8`, `MAX_MODEL_LEN=8192`, `GPU_UTIL=0.9`,
`KV_CACHE_DTYPE=fp8`, `BLOCK_SIZE=128`, `SERVE_VENV=…/venvs/quant`. Set `MAX_MODEL_LEN`
explicitly — 8192 is a smoke default. Two more to know: `PATCH_CKPT_CONFIG=1` edits the
checkpoint's own `config.json` in place on serve, and a stale-but-live `PID_FILE` makes
the launcher exit 0 **without serving** — delete old pid files.

### Pick the MoE kernel — the default is *not* the fast one

⚠️ **`M3_W4A8_BACKEND` defaults to `cutlass`.** The qualified production kernel is
**Humming indexed 0.1.10** — ~34% faster at concurrency 1 and what every published perf
number used. On the default you get ~102 tok/s/user instead of 137 and will not reproduce
our numbers.

```bash
export PYTHONPATH=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site${PYTHONPATH:+:$PYTHONPATH}
M3_W4A8_BACKEND=humming \
CKPT="$CKPT" SERVED_NAME=MiniMaxAI/MiniMax-M3 PORT=8000 MAX_MODEL_LEN=65536 \
LOG=serve.log PID_FILE=serve.pid \
  bash pipeline/slurm/run_vllm_http_serve_smoke.sh |& tee launcher.log
```

The launcher then handles the Humming flags, env defaults, `LD_LIBRARY_PATH`, and a
fail-closed preflight recorded in `serve.log.humming-preflight.json`. Two things it does
**not** do:

1. **Set `PYTHONPATH`** — the export above is on you (on `serve-026` Humming is in-venv
   and no export is needed). Forgetting it fails loudly at preflight; it will not silently
   serve CUTLASS.
2. **Run the backend attestation** that proves which kernel actually served. Run it once
   the server is up:

   ```bash
   python -m pipeline.m3_humming_w4a8 attest \
     --preflight serve.log.humming-preflight.json \
     --log serve.log --out backend-attestation.json
   ```

   It fails non-zero on any CUTLASS/Marlin/unquantized fallback marker in the serve log.

`VLLM_HUMMING_MOE_GEMM_TYPE=grouped_contiguous` is the other measured option (slower at
low/mid load). Kernel choice is a serving knob only — it never affects output quality.

**Driving the benchmarks repo** (`/mnt/nfs/hoangduy/projects/benchmarks`)? Use the
profile seam — it fails closed without its three inputs:

```bash
M3_ARM=r6 \
MODEL_PATH="$CKPT" \
QUANT_RECIPE=awq-w4afp8-r6 ENDPOINT_PORT=8004 \
PROFILE=minimax-m3-inhouse bash performance/scripts/run_all.sh
```

Results are namespaced `results/minimax-m3-inhouse-<M3_ARM>/`, so arms never overwrite
each other. Don't copy an existing arm binding to add a new arm — pass the inputs.

## A4. Verify before you trust it

M3's nastiest failure mode: **the checkpoint loads, the server answers, and the output is
garbage** (empty strings or `\r\n` repetitions). It comes from a dropped MoE-router
`ignore` rule ([A5](#a5-symptom--cause--fix)) and will quietly poison an eval run. Check
four things:

```bash
# 1. Overlay applied
python pipeline/slurm/patch_vllm_m3_serve.py --check && echo OVERLAY-OK

# 2. The checkpoint's ignore list retains the router
python -c "
import json,sys
ig=json.load(open('$CKPT/config.json'))['quantization_config']['ignore']
print(len(ig), 'ignore rules; router rules:', [p for p in ig if 'gate' in p])
assert any('gate' in p for p in ig), 'MoE ROUTER MISSING FROM ignore -- output will be garbage'
print('IGNORE-OK')"

# 3. The model actually answers coherently
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"MiniMaxAI/MiniMax-M3",
  "messages":[{"role":"user","content":"Name three prime numbers."}],
  "max_tokens":512}' | python -m json.tool
# M3 thinks before answering; a small max_tokens truncates the visible answer
# and looks like a failure when it isn't.

# 4. Confirm which kernel you got, and record the stack fingerprint
grep -oE 'enforce_eager=[A-Za-z]+|CUDAGraphMode\.[A-Z_]+' serve.log | head -2
grep 'w4a8-backend' launcher.log                  # launcher stdout, not serve.log
cat serve.log.humming-preflight.json 2>/dev/null  # Humming arms only
# system_fingerprint (vllm-0.24.0-tp8-ep-<hash>) is in each chat response body —
# read it from step 3's output; it is not written to any log.
```

**`rc=0` is not evidence.** We have had a benchmark exit 0 with every request returning
500\. Gate on the **completed-request count**, not the exit code.

## A5. Symptom → cause → fix

Look up what you actually see. `BUGS_AND_FIXES.md` has the long-form post-mortems.

| Symptom | Cause | Fix |
|---|---|---|
| Empty output, or `\r\n` repeated | The MoE router (`mlp.gate`) got pruned from the saved `quantization_config.ignore`, so vLLM treats the unquantized router as quantized → broken routing | Verify step 2 in [A4](#a4-verify-before-you-trust-it). `quantize.py` re-adds the rules now; older checkpoints need the repair snippet in `pipeline/README.md` |
| `NotImplementedError` on activation at startup | Overlay not applied | `python pipeline/slurm/patch_vllm_m3_serve.py` |
| Illegal memory access under concurrency, graphs on | One of four capture bugs — usually the dropped pre-capture `synchronize()` | `LLMC_M3_CAPTURE_SYNC=sync` (the launcher default). `ENFORCE_EAGER=1` only as a last resort — don't publish numbers from it |
| Illegal memory access on **0.26.0**, even at k=0 | `topk_indices_buffer` layout regression | Overlay edit 8. See `docs/m3-026-topk-buffer-layout.md` |
| `Free memory on device cuda:X ...` at startup | Leftover workers from a crashed run | `bash pipeline/slurm/free_gpus.sh` |
| A CUDA error names a kernel that makes no sense | **CUDA errors are sticky** — the reported kernel is not the faulting one | Re-run with `CUDA_LAUNCH_BLOCKING=1` to find the real site |
| Benchmark "passes" but numbers are absurd | All requests failed; the exit code didn't care | Gate on completed count |
| Throughput ~25% below published numbers | You're on CUTLASS — the default backend | `M3_W4A8_BACKEND=humming`, then check the attestation ([A3](#a3-serve-it)) |
| Cross-node NCCL/gloo hangs at init | Hostnames don't route between nodes | `export NCCL_SOCKET_IFNAME=intranet` (and the gloo equivalent) |
| Model won't load: expert-width error | CUTLASS W4A8 MoE needs `moe_intermediate_size` divisible by **256** | See [B4](#b4-pre-quantization-static-gates). Sharding can't fix it |
| Prompt-cache hit columns are blank | vLLM 0.24 never emits `cached_tokens` in usage — prefix caching *is* on | Measure warm-vs-cold deltas instead |
| SGLang won't load our W4AFP8 checkpoint | SGLang support is an open, unmerged PR | Use vLLM. See [A8](#a8-known-broken--do-not-use) |

## A6. What performance to expect

**Quote per-user output speed (1/ITL) first** — it's what a user feels. Server-aggregate
throughput can look good while every individual request is slow.

Measured on 8×H100, in-house GPTQ W4AFP8, Humming indexed kernel (source:
`docs/m3-two-axis-perf.md`):

| Concurrency | TPOT p50 (ms) | Output speed (tok/s/user) | Total output tok/s | vs BF16 per GPU |
|---|---|---|---|---|
| 1 | 7.30 | **137** | 137 | 3.4× |
| 4 | 8.82 | 113 | 451 | 3.7× |
| 16 | 12.23 | 82 | 1300 | 3.7× |
| 64 | 19.40 | 52 | 3267 | **3.8×** |

Output speed here is `1000 ÷ TPOT`. Under natural (unpinned) output lengths, prefer
aiperf's own `output_token_throughput_per_user` — mean per-request 1/ITL is not the
reciprocal of mean ITL.

Against BF16 (16 GPUs, 2 nodes): weights 225 GB vs 796 GB, and **0.68 vs 2.61 GPU-hours
per 1M tokens**.

Two multipliers stack on the format choice:

- **MoE kernel** (already in the table): Humming indexed beats CUTLASS by **+34% at
  conc 1** (137 vs 102 tok/s/user) and +28–34% at conc 10. Opt-in via
  `M3_W4A8_BACKEND=humming` ([A3](#a3-serve-it)). If your conc-1 number is ~102, you are
  on CUTLASS.
- **Speculative decoding (EAGLE3)** — not in the table: a further 1.18–2.53×,
  workload-dependent (best 345.9 tok/s/user single-user on code; floor 1.50× on loaded
  creative writing). There is no single best draft depth, and the drafter's reference
  model is **MXFP8, not BF16**. Read `M3_OFFICIAL_SPECDEC_RESULTS.html` before enabling.

> **Never compare numbers across serving configs.** Kernel, concurrency, workload,
> graphs, and draft depth each move throughput more than whatever you're trying to
> measure. Record the config next to every number.

## A7. If you publish a score

Record all of this — it's a hard requirement of the repo's evaluation-harness contract:

- **Serving stack:** vLLM version + venv, overlay `--check` status, `system_fingerprint`,
  checkpoint path **and** hash, TP/EP topology, graphs on/off, `M3_W4A8_BACKEND` +
  `VLLM_HUMMING_MOE_GEMM_TYPE`, `max_model_len`, and (if specdec) drafter + draft depth.
- **Harness:** tokenizer/chat-template hashes, task aliases + harness version, few-shot
  counts, metrics, sampling parameters, reasoning mode, sample-manifest hash.

**State comparability explicitly.** Our quality numbers are valid for quant-vs-quant and
quant-vs-BF16 decisions (identical protocol across arms). They are **not** comparable to
public leaderboards: we use greedy decoding (MiniMax's recipe is temp 1.0 / top_p 0.95)
and a task subset. Don't conflate the two.

Known gap: the quality-eval `run_manifest.json` records only `lm_eval_version`; the vLLM
version and patch status live in the serving-diagnostic run dirs.

## A8. Known-broken / do-not-use

- **vLLM 0.26.0** (`serve-026`): open blockers — see `docs/m3-dspark-blockers-026.md`.
  0.24.0 + overlay is the qualified path.
- **SGLang** cannot load our W4AFP8 MoE checkpoints (support is an unmerged PR,
  [sgl-project/sglang#21741](https://github.com/sgl-project/sglang/pull/21741)). The
  checkpoint format is engine-agnostic; stock SGLang just mis-routes it. Use vLLM.
- **AWQ `r5`**: reasoning loops that exhaust the budget and emit nothing. Use `r6`.
- **GPTQ `r8-*`**: no quality evaluation. Perf comparisons only.
- **Never install Humming into `quant`.** Use the `PYTHONPATH` side-install.

## A9. Reporting a problem

Attach: `serve.log`, `launcher.log`, the `--check` output, the checkpoint path, and the
`system_fingerprint` (plus `cell-config.txt` if from a benchmark arm). Raw evidence lives
under `/mnt/nfs/hoangduy/results/<study>/<UTC-timestamp>/` — point at the run directory
rather than pasting fragments.

---

# Part B — The quantization pipeline (reference)

## B1. What it does

Two lifecycles meet at the checkpoint; you almost certainly only touch the second:

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

Stages: `quantize → serve-verify → eval-gate`. Run one with `--stage quantize|serve|eval`;
reuse a checkpoint with `--checkpoint <dir>` (serve/eval only); override any field with
`--set a.b.c=value`. Output lands in `artifacts/<run_slug>/<timestamp>/` with
`checkpoint/`, the resolved config, `recipe.json`, `metadata.json`, `serve_report.json`,
`eval_report.json`.

`quantization.method` ∈ {`gptq`, `awq`, `smoothquant+{gptq,awq}`, `autoround`,
`spinquant+{gptq,awq}`}. `quantization.scheme`: `W4AFP8` and `W4A8` are the production
choices; W4 schemes save as `pack-quantized`, except `fp8_dynamic_targets` recipes (the
r8 configs), which save as `mixed-precision`.

## B2. Launching a full M3 quantization

Multi-GPU (DDP) calibration for both AWQ and GPTQ exists in this fork; the wrapper handles
the `torchrun` launch. Don't write expert-parallel code — `CLAUDE.md` documents that
detour as the canonical example of what not to do.

Dry-run first — it prints the exact `srun` command. (Preflights run on the node at real
launch, **not** during a dry-run.)

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-ddp-awq-full-r6-noupdown"
DRY_RUN=1 METHODS=awq EVIDENCE_ONLY=0 \
  CONFIG=pipeline/configs/minimax_m3_distributed_awq_full.yaml \
  RESULT_ROOT=/mnt/nfs/hoangduy/results/m3-distributed-awq-full/$RUN_ID \
  LOG_ROOT=/mnt/nfs/hoangduy/logs/m3-distributed-awq-full/$RUN_ID \
  OFFLOAD_ROOT=/mnt/nfs/hoangduy/offload/m3-distributed-awq-full/$RUN_ID \
  bash pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh
```

Drop `DRY_RUN=1` to launch, from a `tmux` shell **outside any Slurm allocation** (the
controller refuses to start inside one). `METHODS="gptq awq"` runs both on separate
exclusive nodes. `EVIDENCE_ONLY=0` is what saves a checkpoint (the default `1` writes
metrics only). This path is **quantize-only** — serving and eval happen separately via
Part A.

Preflights that each exist because they cost us a failed multi-hour run:

- **`/dev/shm` must hold the whole model** (~913 GB gate for M3) — the distributed offload
  keeps one shared copy there. `MIN_SHM_AVAILABLE_BYTES=auto` sizes the gate from the
  safetensors index.
- **`CPUS_PER_TASK=192`** — the one-core binding trap ([A0](#a0-environment-preconditions)).
- **`MIN_MEM_AVAILABLE_BYTES`** guards host RAM.

## B3. Known results

### Quantization cost (measured, 1 node 8×H100)

| Run | Method | Calibration | Save + vLLM export | Total |
|---|---|---|---|---|
| `r5-deadchan` | AWQ | 7.20 h | 0.25 h | 7.45 h |
| **`r6-noupdown`** | AWQ | **2.23 h** | 1.25 h | 3.48 h |
| `r7-gatealpha` | AWQ | 7.53 h | 7.10 h † | 14.63 h |
| `r8a-fp8rest` | AWQ | 2.03 h | 0.35 h | 2.38 h |
| **`r8-fp8rest`** | GPTQ | **3.12 h** | 10.2 h † (3 exports) | 13.30 h |

Rules of thumb: **GPTQ ≈ 3 h; AWQ ≈ 2–7.5 h depending on recipe** (dropping the up→down
fold is what makes r6 ~3.4× cheaper to calibrate than r7).

† Save phases of concurrent runs contend on NFS and inflate to 7–10 h, vs ≤ 1.25 h alone.
Don't run two full quantizations at once if wall-clock matters.

### Quality (paired vs BF16, 64k budget, greedy, thinking on)

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
The seven-task AWQ data is `r5` only — r6/r7 were measured on the two damaged tasks.

**Read `exhausted` before `score`.** It is the fraction of responses that hit the
generation ceiling without a final answer, and it tells you *why* a score moved. r5's
30-point GPQA loss was not lost capability — when it answered, it matched BF16; it fell
into reasoning loops and emitted nothing, which scores zero. The cause was AWQ folding
smoothing scales through M3's up→down path, which is not function-preserving across the
clamped GLU; removing the fold (r6) collapsed token spend from 2.19×/3.69× to 1.13×/1.17×.
Full story with transcripts: `M3_OFFICIAL_QUALITY_RESULTS.html`.

## B4. Pre-quantization static gates

Two checks to run **before** burning GPU time. (The automated width gate fires later, at
serve-verify against the saved checkpoint — so check up front yourself, with the one-liner
below or `pipeline/prequant_compatibility.py`.)

**1. Expert width must be a multiple of 256.** The CUTLASS W4A8 MoE kernel requires each
expert's `moe_intermediate_size` divisible by 256 on the **per-partition** width: with
expert parallelism each rank holds whole experts, so the full width applies; with plain TP
the width divides (768 at TP=2 → 384 → broken). Sharding can't rescue a bad width — only
padding or a scheme change.

```bash
python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('<model_id>', trust_remote_code=True); print(getattr(c,'moe_intermediate_size', getattr(getattr(c,'text_config',c),'moe_intermediate_size',None)))"
```

Qwen1.5-MoE-A2.7B: 1408 → incompatible. Qwen3-30B-A3B: 768 → fine.

**2. The `ignore` list must survive serialization** — the garbage-output bug in
[A5](#a5-symptom--cause--fix). `quantize.py` handles this now; verify it on any checkpoint
you didn't produce.

Also available: `pipeline/verify_quant_checkpoint.py` and
`pipeline/m3_checkpoint_scale_audit.py`.

---

## Glossary

| Term | Meaning |
|---|---|
| **W4AFP8** | INT4 group-128 weights + dynamic per-token FP8 (E4M3) activations. Our production scheme |
| **W4A8** | Used almost interchangeably with W4AFP8 here — most "W4A8" references mean the FP8-activation variant. A distinct INT4+INT8 W4A8 also exists (vLLM-only). When it matters, check `config.json` |
| **W4A16** | INT4 weights, 16-bit activations (Marlin kernels). cyankiwi's format |
| **`pack-quantized`** | The compressed-tensors on-disk format for packed INT4. Engine-agnostic |
| **arm** | One (checkpoint × kernel × topology) configuration under test. Registry: `docs/m3-benchmark-arms.md` |
| **recovery %** | candidate score ÷ baseline score on the same task |
| **flips** | Answers that changed direction vs baseline. Symmetric = noise; one-sided = damage |
| **exhausted** | Response hit the generation ceiling without a final answer → scores zero |
| **CUTLASS / Humming / Marlin / Machete** | Alternative GEMM kernel backends. Kernel choice alone moves throughput ~34% |
| **EAGLE3 / draft depth `k`** | Speculative decoding; how many tokens the drafter proposes per step |
| **overlay** | The Python patch set applied to installed vLLM ([A2](#a2-why-vllm-needs-patches)); also, separately, a symlink directory that changes only a checkpoint's `config.json` |
| **`r5`…`r8a`** | Sequential in-house recipe revisions. Higher = newer, **not** better-verified |

## Working agreements

- **Never delete a checkpoint** without asking. 215–225 GB each; several are
  irreproducible.
- **Results live under `/mnt/nfs/hoangduy/`**, not in the repo. Commit small evidence
  files only.
- **`srun` from tmux; never `sbatch`.**
- **Don't edit a launcher mid-run.**
- **Fix a broken comparison arm; don't drop it.**
- **Report outcomes honestly** — if a gate failed or a step was skipped, say so, with the
  raw output.
