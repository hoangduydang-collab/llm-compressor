# MiniMax-M3 CUDA-graph RCA — cluster runbook

Run this on a free 8-GPU node (e.g. `polaris-h119`) after pulling
`duy-branch` (commit that adds the RCA harness, or later).

**Goal:** Classify whether the HTTP async CUDA-graph IMA is MoE routing,
breakable-cudagraph/collective capture, or graph memory lifetime — **without
editing vLLM site-packages**.

**Do not** treat `DEBUG_CUDAGRAPH=1` / `CUDA_LAUNCH_BLOCKING=1` as a root-cause
fix. That is a `masked_pass` only.

---

## Prerequisites (once per session)

```bash
cd /mnt/nfs/hoangduy/llm-compressor   # adjust if your NFS clone path differs
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

python pipeline/slurm/patch_vllm_m3_serve.py --check
# expect: 4/4 patched + flashinfer >= 0.6.10 (ideally 0.6.12)

bash pipeline/slurm/free_gpus.sh
# expect: all GPUs >= 70 GiB free; refuse if another user holds them
```

**Why:** Workers are spawned subprocesses; missing patches re-debug a “fixed”
bug. Dirty GPUs cause OOM that looks like serve failure.

---

## Task 0 — Local dry-run sanity (optional on cluster)

```bash
DRY_RUN=1 bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
```

**Why:** Confirms the matrix emits distinct configs / unique ports and does
**not** call `nohup` / `free_gpus` / `kill`. Safe if you want to verify the
pulled scripts before burning GPU time.

---

## Task 1 — Full HTTP RCA matrix (primary)

```bash
bash pipeline/slurm/free_gpus.sh
bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
```

Results land under:

```text
/mnt/nfs/hoangduy/logs/m3-cudagraph-rca/<run_id>/
  run_manifest.json
  summary.json
  <case>/serve.log
  <case>/chat.json
  <case>/meta.json
  <case>/result.json
  <case>/cuda_coredump_*     # only for async_coredump if dump was written
```

**Why each case exists:**

| Case | Knobs | Why |
|------|-------|-----|
| `async_baseline_1..3` | `ENFORCE_EAGER=0 DEBUG_CUDAGRAPH=0` | Is async IMA deterministic or flaky? |
| `graphs_off` | `ENFORCE_EAGER=1` | Must PASS; else failure is not graph-capture-specific |
| `blocking_mask` | `DEBUG_CUDAGRAPH=1` | Confirms sync masks the bug → `masked_pass`, not fixed |
| `breakable_off` | `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | If async PASS here, implicates breakable capture path |
| `async_coredump` | async + CUDA core dump env | Preserve failing schedule; name device kernel later |

**Success criteria for this task:**

- `summary.json` exists with a `verdict` per case (`server_ready`,
  `masked_pass`, `graph_ima_*`, `graphs_off_failed`, or `inconclusive`).
- No ambiguous “looks fine” without a verdict.
- `graphs_off` is `server_ready` (with chat) or the “graphs-only” claim is
  rejected.
- At least one async fail either has a core dump file **or** `result.json`
  notes why the dump is missing.

**Shorter first pass (if time-limited):**

```bash
MATRIX_CASES=async_baseline_1,graphs_off,blocking_mask,breakable_off,async_coredump \
  bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
```

---

## Task 2 — Inspect summary (no GPU)

```bash
RUN=$(ls -td /mnt/nfs/hoangduy/logs/m3-cudagraph-rca/*/ | head -1)
cat "$RUN/summary.json"
python -m pipeline.m3_cudagraph_evidence "$RUN/async_baseline_1/serve.log" \
  --chat "$RUN/async_baseline_1/chat.json" \
  --meta "$RUN/async_baseline_1/meta.json"
```

**Why:** Machine-readable verdict drives the next branch. Precedence in the
classifier: MoE symbol > collective symbol > empty_cache memory hint >
unclassified.

---

## Task 3 — Name the faulting kernel (only if async failed)

If `async_coredump` (or any async case) produced a dump:

```bash
ls -lt /mnt/nfs/hoangduy/logs/m3-cudagraph-rca/*/async_coredump/cuda_coredump_* 2>/dev/null | head
cuda-gdb
# (cuda-gdb) target cudacore /path/to/cuda_coredump_<host>.<pid>.<ts>
# (cuda-gdb) info cuda kernels
# (cuda-gdb) bt
```

Record the **first device kernel name** into the trial meta / a note file, e.g.:

```bash
echo '{"faulting_kernel":"PASTE_KERNEL_NAME"}' > /tmp/fk.json
# re-classify with:
python -m pipeline.m3_cudagraph_evidence "$RUN/async_coredump/serve.log" \
  --meta <(python -c "import json; m=json.load(open('$RUN/async_coredump/meta.json')); m['faulting_kernel']='PASTE'; print(json.dumps(m))" ) \
  -o "$RUN/async_coredump/result_with_kernel.json"
```

**Why:** Python stacks blame `empty_cache()` under async CUDA. Only a device
backtrace (or a named symbol in the log) is root-cause evidence.

If **no dump** was written, document that explicitly, then escalate:

```bash
# optional follow-up (heavy): compute-sanitizer on a single async fail
# Prefer after Task 1 summary is reviewed — do not start this blindly.
```

---

## Task 4 — Update docs with the classified result

Edit the “Matrix result” line in `BUGS_AND_FIXES.md` under
**HTTP async cudagraph IMA — RCA matrix protocol** with:

- `run_id` / path to `summary.json`
- per-case verdicts
- faulting kernel if known
- next branch:

| Verdict | Next (do **not** implement in this runbook unless asked) |
|---------|----------------------------------------------------------|
| `graph_ima_moe` | Compare route vs vLLM #39391 / FlashInfer finalize |
| `graph_ima_collective` | `capture_error_mode=thread_local` / rank barriers (#46253) |
| `graph_ima_memory_lifetime` | Address-lifetime follow-up (#45487) |
| `graph_ima_unclassified` | Keep tactical mask; `compute-sanitizer` before code changes |
| `server_ready` on `DEBUG_CUDAGRAPH=0` | Removal criteria for default `DEBUG_CUDAGRAPH=1` |

**Why:** Prevents the next session from re-guessing; removal criteria for the
tactical sync mask are explicit.

---

## What the cluster agent must **not** do

- Do not edit vLLM under `site-packages` during this matrix.
- Do not set `DEBUG_CUDAGRAPH=1` as the “fix” and stop.
- Do not kill other users’ GPU processes (`free_gpus.sh` already refuses).
- Do not enable `M3_MOE_PROBE=1` during capture (device sync poisons capture).
- Do not inherit a stale `MODEL_CKPT` from Nemotron; the launcher prefers `CKPT=`.

---

## Quick reference — harness files

| File | Role |
|------|------|
| `pipeline/slurm/test_m3_http_cudagraph_matrix.sh` | Sequential matrix runner |
| `pipeline/slurm/run_vllm_http_serve_smoke.sh` | HTTP `vllm serve` (supports `PRINT_EFFECTIVE_CONFIG=1`) |
| `pipeline/slurm/smoke_chat_completions.sh` | Chat smoke after ready |
| `pipeline/m3_cudagraph_evidence.py` | Log → JSON verdict |
| `pipeline/slurm/free_gpus.sh` | Preflight between trials |
| `pipeline/slurm/patch_vllm_m3_serve.py --check` | Patch / flashinfer gate |
| `BUGS_AND_FIXES.md` | Protocol + place to record results |

---

## Paste-ready one-liner for the agent

```bash
source /mnt/nfs/hoangduy/env.sh && \
source /mnt/nfs/hoangduy/venvs/quant/bin/activate && \
cd /mnt/nfs/hoangduy/llm-compressor && \
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD" && \
python pipeline/slurm/patch_vllm_m3_serve.py --check && \
bash pipeline/slurm/free_gpus.sh && \
bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh && \
ls -td /mnt/nfs/hoangduy/logs/m3-cudagraph-rca/*/ | head -1 | xargs -I{} cat {}/summary.json
```

Then perform Task 3 if any async case failed and a core dump exists; then Task 4.
