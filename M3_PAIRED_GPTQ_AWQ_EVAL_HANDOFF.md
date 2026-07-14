# Handoff: Paired quality eval — in-house GPTQ vs cyankiwi AWQ (2026-07-14)

## Goal

Run a **quick (~1–2h)** paired quality comparison of the repaired in-house
**GPTQ** checkpoint against the working **cyankiwi AWQ** reference, using the
standard vLLM quality harness. Smoke already passed (coherent output, GSM8K 2/2,
MMLU-Pro 12/14 — `M3_3MODEL_GPTQ_AWQ_FINAL_REPORT.md`); this run confirms it
holds at (sub-)eval scale before we commit to the full run. **This is the
go/no-go gate for adopting GPTQ as the production recipe** (and for starting
`M3_QUANT_SPEEDUP_PLAN.md`).

Scope: model quality only. Do **not** start serving-throughput/CUDA-graph work.
A gate FAIL is a valid result — return it, do not tune to pass.

## Quick smoke result (2026-07-14)

Run ID:
`20260714T064000Z-m3-paired-gptq-awq-quick`

The repaired-overlay GPTQ and cyankiwi AWQ smoke arms both completed
successfully:

- Controller return code: `0`
- GPTQ arm return code: `0`
- cyankiwi AWQ arm return code: `0`
- Both arms scored all five tasks with the identical sample manifest
- Both had `infrastructure_ok=true`, `artifacts_valid=true`, no empty outputs,
  no periodic loops, and distributed world size 8
- `smoke_gate.ready_for_production=true`

Smoke aggregate results (two samples per generative task; MMLU-Pro has 14
scored leaves) were:

| Task | cyankiwi AWQ | in-house GPTQ |
| --- | ---: | ---: |
| GPQA Diamond | 0/2 | 0/2 |
| IFEval prompt-level strict | 0.5 | 0.0 |
| AIME 2025 | 0/2 | 0/2 |
| MMLU-Pro | 12/14 | 12/14 |
| GSM8K | 2/2 | 2/2 |

The IFEval difference is directional only because this is the low-n smoke;
the smoke gate is an infrastructure/readiness gate, not the definitive
production-quality verdict. The GPTQ distributional probe also completed
without degeneration failures. Its vLLM runtime used the expected metadata
overlay and the recorded SWIGLU clamp patch; the raw GPTQ source was not
served.

Complete raw evidence, including preflight diagnostics, resolved manifests,
per-task sample JSONL, generation-health files, distributional probe JSONL and
summaries, arm logs, return codes, and smoke gate/report files is committed
under:
`results/m3-quality/20260714T064000Z-m3-paired-gptq-awq-quick/`

This quick smoke unblocks the four-arm quick production paired run. It does
not yet justify adopting GPTQ or starting the speed-up work; use the paired
production gate for that decision.

## Which config to run

- **Quick (run this now):** `pipeline/configs/minimax_m3_paired_gptq_awq_quick.yaml`
  — every task stratified-subsampled to `sampling.production_samples_per_task`
  (100) via the seeded manifest, so the paired McNemar stays exact (both models
  see identical indices) while bounding wall-clock. Production = 4 arms, **430
  samples/model** (gpqa 100, ifeval 100, aime 30, mmlu_pro 100, gsm8k 100).
  **Low-n caveat:** per-task n is small → deltas are directional and gates are
  advisory, not a definitive verdict.
- **Full (definitive, later):** `pipeline/configs/minimax_m3_paired_gptq_awq.yaml`
  — same setup, full task sets / MMLU-Pro 2000. Run this once the quick pass
  looks good.

Timing knob: if the first arm projects past ~2h, lower
`sampling.production_samples_per_task` (e.g. 50) and re-run; raise it for more
signal. Do **not** reduce `eval.gen_kwargs.max_gen_toks` — the reasoning tasks
need the full budget, and both models share it so pairing is unaffected either
way.

## What the planner set up

- New quick + full matrices (above). `baseline_label: cyankiwi_awq` → the
  harness emits a single paired comparison `comparisons.inhouse_gptq` =
  GPTQ-vs-cyankiwi McNemar. Two arms, both `nodes: 1`, `tensor_parallel_size: 8`,
  backend `mp`. **No BF16/Ray arm**, so the launcher's 2-node Ray topology
  preflight never fires (`MAX_ARM_NODES == 1`).
- Small harness change (tested, default path unchanged): when
  `sampling.production_samples_per_task` is set, `build_profile_sample_manifests`
  stratified-subsamples **every** task through the manifest (not just MMLU-Pro).
  This caps wall-clock while preserving exact paired sampling — required because
  the runner treats exact-samples and lm-eval `limit` as mutually exclusive
  (`lmeval_runner.py:220`), so per-task `limit` in the eval config would crash
  smoke. Verified: 43 harness tests pass; production manifest caps to 430
  samples/model.
- Validated locally with `load_matrix` / `build_launch_plan` /
  `build_profile_sample_manifests`:
  - baseline `cyankiwi_awq`; smoke = 2 arms, `total_nodes=2`, `max_arm_nodes=1`;
  - production expected arms = `(cyankiwi_awq, reasoning/broad)`,
    `(inhouse_gptq, reasoning/broad)` = 4 arms, `total_nodes=4`.
- Checkpoints:
  - GPTQ **raw source** (fails serving-ABI — do NOT serve directly):
    `/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123`
  - GPTQ **ABI overlay** (serve this; the matrix points here; built in Step 0):
    `/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay`
  - cyankiwi: `/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4`
  - tokenizer source (preflight only): `/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3`

> **Why an overlay:** the raw `gptq-checkpoint-vllm-w123` fails static serving-ABI
> validation — its `config.json` `quantization_config.ignore` doesn't cover the
> vLLM runtime router (`block_sparse_moe.gate`) or shared-expert projections, so
> preflight (correctly) rejects it. The overlay is **metadata-only**: it symlinks
> every tensor shard unchanged and only rewrites `config.json` to add
> `re:.*block_sparse_moe[.]shared_experts[.].*` and `re:.*block_sparse_moe[.]gate$`.
> This is exactly the checkpoint the passing smoke served. A prior planner config
> mistakenly pointed at the raw source — hence the earlier preflight failure.

## Run procedure

All `srun`, not `sbatch`; launch from a login/control shell **outside** any
existing allocation. Use a detached tmux server to own the controller if the
run is long (same pattern as prior quality handoffs); do not let a Cursor
foreground call own the srun.

### 1. Setup + preflight (no GPUs consumed for the eval itself)

```bash
git pull --ff-only origin duy-branch
cd /mnt/nfs/hoangduy/projects/llm-compressor
source /mnt/nfs/hoangduy/venvs/quant/bin/activate   # the working vLLM env
export PYTHONPATH="$PWD/src:$PWD"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-paired-gptq-awq-quick"
RUN_ROOT="results/m3-quality/$RUN_ID"
MATRIX=pipeline/configs/minimax_m3_paired_gptq_awq_quick.yaml   # full: minimax_m3_paired_gptq_awq.yaml
mkdir -p "$RUN_ROOT"

# --- Step 0: build the ABI overlay the matrix points at (CPU, ~1 min, metadata-only) ---
SOURCE_GPTQ=/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123
OVERLAY_GPTQ=/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay
# Sanity: the raw source MUST fail serving-ABI (this is expected).
if python -m pipeline.m3_serve_abi --checkpoint "$SOURCE_GPTQ" --out "$RUN_ROOT/static_direct_gptq.json"; then
  echo "ERROR: raw GPTQ source unexpectedly PASSED ABI; stop and report" >&2; exit 1
fi
# Build overlay only if absent (prepare-overlay refuses to overwrite; it symlinks
# tensors and only rewrites config.json to add the two vLLM ignore aliases).
if [[ ! -e "$OVERLAY_GPTQ" ]]; then
  python -m pipeline.m3_routed_diagnostics prepare-overlay \
    --source "$SOURCE_GPTQ" --destination "$OVERLAY_GPTQ" \
    --add-vllm-shared-expert-ignore --add-vllm-router-ignore
fi
# Verify provenance: identical safetensors index hash, distinct config hash,
# both aliases added, tensor_payload_unchanged == true.
python -m json.tool "$OVERLAY_GPTQ/overlay_provenance.json"
# Confirm the overlay now PASSES serving-ABI before spending any preflight time.
python -m pipeline.m3_serve_abi --checkpoint "$OVERLAY_GPTQ" --out "$RUN_ROOT/static_overlay_gptq.json"

python -m pipeline.m3_quality_preflight --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/preflight.log"
```

Preflight now inspects the **overlay** (matrix `inhouse_gptq.path`) and both
models' serving ABIs should report valid. If the overlay already exists from a
prior run, Step 0 reuses it (the tensor symlinks still resolve to the same
source payloads).

Preflight must produce `run_manifest.json`, `preflight/resolved_eval_config.yaml`,
`preflight/resolved_tasks.json`, both sample manifests, both probe corpora, and
`preflight/checkpoint_diagnostics/{cyankiwi_awq,inhouse_gptq}.json`, and must
report `baseline_label: cyankiwi_awq` with exactly the two models. If a task
alias/split changed, diagnose the installed lm-eval registry and make the
smallest documented fix; do not silently substitute a benchmark.

### 2. Smoke (2 nodes), then gate

```bash
# Inspect the plan — expect: arms=2 total_nodes=2, and NO srun ray-check line.
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$MATRIX" --run-root "$RUN_ROOT" --dry-run

bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$MATRIX" --run-root "$RUN_ROOT"
SMOKE_RC=$?
python -m json.tool "$RUN_ROOT/smoke_gate.json"
```

Stop and return evidence if `SMOKE_RC != 0` or `ready_for_production` is not
`true`. (This should pass — the checkpoints already smoked cleanly; a failure
here means an env/harness regression, not a model regression.)

### 3. Production (4 nodes) — only after smoke gate passes

```bash
# Inspect — expect: 4 arms, total_nodes=4, no ray-check line.
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  --smoke-gate "$RUN_ROOT/smoke_gate.json" --dry-run

# Quick run: ~1-2h target; 03:00:00 is a generous ceiling. (Use 08:00:00 for the
# full matrix.) 16k-token gens are kept; only sample counts are reduced.
TIME_LIMIT=03:00:00 bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  --smoke-gate "$RUN_ROOT/smoke_gate.json"
PRODUCTION_RC=$?
```

Do not cancel a healthy arm because another fails; preserve all completed
evidence. If 4 nodes aren't simultaneously available, start as many arms as
possible and document queueing — keep the same topology (1 node/arm, tp8, mp).

### 4. Aggregate (run even if one arm failed)

```bash
set +e
python -m pipeline.m3_quality_eval aggregate --matrix "$MATRIX" --root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/aggregate.log"
AGGREGATE_RC=${PIPESTATUS[0]}   # 0 => gates.json quality_ok true
set -e
```

## What to return (commit + push, then stop for planner analysis)

- `run_manifest.json`, all `preflight/` artifacts + checkpoint diagnostics;
- per-arm `arm_manifest.json`, `arm_complete.json`, `return_code.txt`,
  `aggregate.json`, normalized per-sample JSONL, `generation_health.json`, and
  (broad shard) `distributional_probe.jsonl` + summary, plus stdout/stderr;
- `matrix.json`, `gates.json`, `report.md`, `smoke_report.json`,
  `smoke_gate.json`, `aggregate.log`;
- exact commands, git commit, vLLM/lm-eval/torch versions, Slurm job/step IDs,
  nodes, wall times, exit codes, any retry/deviation;
- SHA-256 + durable NFS path for any large artifact kept out of git.

## Interpretation (the planner will decide, but flag these)

The decision object is `comparisons.inhouse_gptq` (GPTQ = candidate B,
cyankiwi = reference A) and `gates.json`:

- **`score_recovery_ratio`** per task = GPTQ / cyankiwi. Macro-recovery gate is
  `>= 0.98` (GPTQ within 2% of the reference).
- **`delta`** = GPTQ − cyankiwi accuracy per task; `max_task_drop <= 0.02`.
- **`regressions_a_correct_b_wrong`** = samples cyankiwi got right but GPTQ got
  wrong; `conditional_regression <= 0.05`.
- **`perplexity_ratio`** (broad-shard probe) `<= 1.10`; **degeneration failures**
  must be 0.
- `gates.quality_ok == true` ⇒ GPTQ is statistically on par with the reference
  AWQ ⇒ adopt GPTQ as the production recipe and green-light
  `M3_QUANT_SPEEDUP_PLAN.md`.

## Do not

- Do not serve the raw `gptq-checkpoint-vllm-w123` source — always the overlay.
- Do not start serving-perf/CUDA-graph work, AutoRound, or the speed-up
  implementation in this run.
- Do not delete any checkpoint.
- Do not add a BF16 arm here — if a recovery-vs-BF16 comparison is later wanted,
  the 3-model `minimax_m3_quality_matrix.yaml` already exists for that (costs the
  extra 2-node Ray arm).
