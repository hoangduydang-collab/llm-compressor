# MiniMax-M3 Quantized Checkpoint Handoff

## Goal and current status

Verify full-calibration MiniMax-M3 W4A8 AWQ/GPTQ checkpoints with the
MiniMax-M3-specific vLLM serve path and identify why generation is garbage.

The checkpoint currently **loads successfully but has not passed quality
verification**. Do not delete any original checkpoint.

## Checkpoints

| Checkpoint | Status |
| --- | --- |
| `artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint` | Original AWQ W4A8; loads, outputs repeated `arring...` garbage. |
| `artifacts/MiniMax-M3-gptq-W4AFP8/20260709-064842/checkpoint` | Original GPTQ W4A8; loads, also outputs garbage. |
| `artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123` | Portable AWQ re-export; statically validated and loads, but still outputs garbage. |
| `/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4` | Working reference model. |

The re-export is 225 GB. It rewrites only Safetensors tensor names; raw tensor
payloads are copied byte-for-byte.

## Confirmed results

1. The original AWQ and GPTQ checkpoints pass structural checkpoint audits and
   load under the M3 vLLM serve setup, but produce nonsensical generation.
2. Serving the original AWQ checkpoint with W4A16 (no runtime FP8 activation
   quantization) still produced `arring...`. W4A8 activation quantization is
   therefore not the sole cause.
3. The original routed-expert tensor layout uses descriptive names:
   `gate_proj`, `down_proj`, and `up_proj`. The installed NVIDIA vLLM M3 loader
   maps only `w1`, `w2`, and `w3` for routed experts:

   - `w1 = gate`
   - `w2 = down`
   - `w3 = up`

4. `pipeline.reexport_minimax_m3_vllm` created a portable layout by renaming
   routed keys:

   - `gate_proj -> w1`
   - `down_proj -> w2`
   - `up_proj -> w3`

   Shared-expert keys intentionally remain descriptive; cyankiwi uses the same
   shared-expert naming.
5. Static re-export validation passed: 5 shards, 67,192 keys, 65,664 routed
   keys renamed. The portable checkpoint completed an 8-H100 vLLM serve with
   `loaded=true`, `rc=0`, and a nonempty output, but the output remained
   garbage:

   ```text
   seringk seringk seringk mempunastast...
   ```

   This proves the routed-key layout mismatch was real but not sufficient to
   explain the quality failure.

## Relevant code changes

- `pipeline/reexport_minimax_m3_vllm.py`
  - Header-only Safetensors re-export utility.
  - Refuses to overwrite a destination, validates transformed index/header
    keys and raw payload byte counts.
- `pipeline/tests/test_reexport_minimax_m3_vllm.py`
  - Tests routed key aliases and payload-preserving shard rewrite.
- `pipeline/slurm/patch_vllm_m3_serve.py`
  - Adds environment-gated `M3_LOAD_AUDIT=1` instrumentation.
  - The audit is injected into vLLM site-packages when `serve_verify` runs.
- `pipeline/serve_verify.py`
  - Installs the optional loader audit before vLLM worker creation.
- `pipeline/tests/test_patch_vllm_m3_serve.py`
  - Covers audit injection/idempotence.

CPU tests run successfully:

```bash
PYTHONPATH="$PWD" pytest -q \
  pipeline/tests/test_reexport_minimax_m3_vllm.py \
  pipeline/tests/test_patch_vllm_m3_serve.py
# 5 passed
```

## Logs and reports

- Original AWQ loader audit:
  `/mnt/nfs/hoangduy/logs/m3-awq-load-audit-srun.log`
- Direct M3 loader audit:
  `/mnt/nfs/hoangduy/logs/m3-awq-loader-audit2-srun.log`
- Re-export:
  `/mnt/nfs/hoangduy/logs/m3-awq-w123-reexport.log`
- Portable checkpoint serve:
  `/mnt/nfs/hoangduy/logs/m3-awq-w123-verify-srun.log`
- Portable checkpoint report:
  `artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123/serve_report.json`

The serve report's `sane_output=true` only means the response is nonempty. It
does not measure semantic quality and is a false positive for these runs.

## Important caveats for further analysis

- Both our checkpoints and cyankiwi place `n_shared_experts=1` under
  `text_config`, not the top-level config. The current audit heuristic warns
  about this, but cyankiwi works, so this warning alone is not root-cause
  evidence.
- The loader audit established that the original routed keys did not reach a
  fused-MoE parameter loader. Its aggregate alias diagnostic should be treated
  as supporting evidence rather than a final proof: the vLLM mapping contains
  full per-expert prefixes, and the audit's presentation is verbose.
- The portable checkpoint's unchanged garbage means the next investigation
  should compare actual loaded parameter values and shared-expert contributions
  against cyankiwi, rather than retrying activation dtype or routed key aliases.
- The existing MoE forward probe is guarded against CUDA graph capture, but
  prior runs did not obtain a useful real-prompt measurement.

## Suggested planner work

1. Design a minimal, low-risk comparison that samples loaded routed and shared
   parameter statistics from cyankiwi versus the portable AWQ checkpoint after
   vLLM construction.
2. Determine whether the shared-expert module is instantiated and receives its
   tensors in both runs, including its contribution during a real prefill.
3. Check whether compressed-tensors W4A8 expert loading supports the exported
   `weight_packed`, `weight_scale`, and `weight_shape` layout equivalently to
   cyankiwi's W4A16 checkpoint.
4. Only after a quality-positive serve should the user be asked again to delete
   the obsolete 225 GB original checkpoint.

## Active next handoff (2026-07-11, shared-expert repair via srun)

Pull the handed-off commit and run `MINIMAX_M3_QUALITY_RUNBOOK.md`. The previous
matrix proved a serve-time naming failure: every candidate rank sees 171 shared
tensors and leaves all 171 unmatched; both candidate schemes create zero packed
shared parameters and all 48 probes have zero shared output. Reference shared
weights and outputs are healthy, candidate first-MoE inputs agree across W4A8
and W4A16, and LM-head hashes match.

The implementation creates an immutable metadata overlay that retains the
Transformers ignore regex and adds the vLLM-native alias
`re:.*block_sparse_moe[.]shared_experts[.].*`. No tensor shard is copied,
rewritten, packed, or re-quantized.

Use `pipeline/slurm/run_m3_shared_expert_repair_srun.sh`. It launches repaired
W4A8 offline diagnostics, repaired W4A16 offline diagnostics, and repaired W4A8
canonical HTTP concurrently on three exclusive eight-GPU nodes through `srun`.
Return and commit the complete compact bundle, then stop for primary-agent
analysis. Do not enable CUDA graphs or begin the second issue in this run.


## Current handoff: AWQ offset-norm repair plus GPTQ control

The layer-boundary matrix localized the first AWQ corruption to layer 8 between
attention output and MoE input. MiniMax-M3's Transformers class is
MiniMaxM3VLRMSNorm, a Gemma-style norm with effective weight 1 + weight.
The existing offset-norm calibration registry did not recognize this class, so
generic AWQ divided the zero-centered raw parameter instead of the effective
weight.

Prepare three checkpoints concurrently using only srun:

    DRY_RUN=1 bash pipeline/slurm/run_m3_awq_gptq_prepare_srun.sh
    bash pipeline/slurm/run_m3_awq_gptq_prepare_srun.sh

This re-exports the existing GPTQ checkpoint and separately quantizes AWQ with
the corrected offset norm and with MLP-input smoothing disabled. After all
three preparation jobs succeed, launch the twelve-node matrix:

    DRY_RUN=1 bash pipeline/slurm/run_m3_awq_gptq_repair_srun.sh
    bash pipeline/slurm/run_m3_awq_gptq_repair_srun.sh

The GPTQ portable checkpoint finishes much earlier than either AWQ rebuild. Do
not wait: launch the early six-node phase while both quantizations continue:

    bash pipeline/slurm/run_m3_gptq_early_srun.sh

Record the printed MATRIX_ID. Once both AWQ variants finish, add only their six
arms to the same evidence directory:

    MATRIX_ID=<printed-id> bash pipeline/slurm/run_m3_awq_repair_finish_srun.sh

The early phase writes comparison_early.json; the finish phase writes the final
comparison.json without rerunning reference, GPTQ, AWQ control, or tensor audit.

Every offline arm probes all sparse layers 3-59. Return the complete
results/m3-awq-gptq-repair/<matrix-id>/ tree, preparation and matrix logs,
checkpoint paths, job/node/return codes, deviations, retries, and retained-log
hashes. Do not start CUDA-graph work.


### Repair the missing staged tensor audit

The first staged run completed all serving arms but its tensor audit artifact was
missing. Pull the classifier/audit fix and rerun only that one allocation:

    MATRIX_ID=20260712-045912-awq-gptq-staged \
      bash pipeline/slurm/rerun_m3_checkpoint_scale_audit_srun.sh

This returns checkpoint_scale_audit.json, its full log and return code, then
regenerates comparison_early.json. Explosion detection is now reference-relative:
the normal approximately 10k residual at layer 5 is no longer a false boundary.


## Current handoff: vLLM-first paired quality matrix (2026-07-12)

### Scope and stop condition

This run addresses the **model-quality issue only**. Do not start serving
throughput/CUDA-graph diagnosis. Compare exactly BF16, in-house GPTQ, cyankiwi
AWQ, and aquaman AutoRound using paired prompts and direct vLLM execution.
A quality-gate failure is a valid experimental result and must be returned; an
infrastructure failure must be reported separately.

The launcher deliberately has two stages:

1. Four concurrent smoke arms use five nodes total (BF16: 2 nodes/16 H100;
   each quantized model: 1 node/8 H100). Each task gets two exact samples,
   generations are capped at 256 tokens, and each model gets a 2,048-token
   distributional probe.
2. Only a passed `smoke_gate.json` unlocks eight concurrent production arms
   using ten nodes total. BF16's two arms each use 2 nodes/16 H100 with Ray;
   the six quantized arms each use 1 node/8 H100.

Use `srun`, not `sbatch`.

### Pull, environment, and dynamic preflight

The AutoRound checkpoint is not preinstalled on the executor cluster. Download
the exact Hugging Face repository to the path already configured in
`pipeline/configs/minimax_m3_quality_matrix.yaml` before running preflight:

```bash
AUTOROUND_REPO=aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx
AUTOROUND_REV=40b00366dcb6bcf16b1710f242b12ce76dd34b9f
AUTOROUND_DIR=/mnt/nfs/hoangduy/hf_assets/aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx

mkdir -p "$(dirname "$AUTOROUND_DIR")"
hf download "$AUTOROUND_REPO" \
  --revision "$AUTOROUND_REV" \
  --local-dir "$AUTOROUND_DIR"
printf '%s\n' "$AUTOROUND_REV" > "$AUTOROUND_DIR/DOWNLOAD_REVISION"
```

Source: <https://huggingface.co/aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx>.
The pinned revision prevents a multi-hour evaluation from silently changing if
the model repository is updated. The model card reports an approximately
176-GiB checkpoint, so check shared-filesystem capacity first. `hf download` is
resumable; rerun the identical command after a network interruption. If the
`hf` command is unavailable, install/update `huggingface_hub` in the executor's
virtual environment (`python -m pip install -U huggingface_hub`) and retry.

Verify every file referenced by the safetensors index rather than relying only
on directory existence:

```bash
python - "$AUTOROUND_DIR" "$AUTOROUND_REV" <<'PYVERIFY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
revision = sys.argv[2]
required = [root / "config.json", root / "model.safetensors.index.json"]
missing = [str(path) for path in required if not path.is_file()]
if not missing:
    index = json.loads(required[1].read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing.extend(str(root / shard) for shard in shards if not (root / shard).is_file())
else:
    shards = []
if missing:
    raise SystemExit("incomplete AutoRound download; missing:\n" + "\n".join(missing[:20]))
recorded = (root / "DOWNLOAD_REVISION").read_text().strip()
if recorded != revision:
    raise SystemExit(f"revision mismatch: expected {revision}, recorded {recorded}")
print(f"AutoRound checkpoint ready: {root} ({len(shards)} indexed shards, revision {revision})")
PYVERIFY
```

The pinned index currently references **36 weight files**: 35 numbered files
(`model-00001-of-00035.safetensors` through
`model-00035-of-00035.safetensors`) plus `model_extra_tensors.safetensors`.
If the verification prints another count, preserve the output and stop for
primary-agent review rather than editing the matrix or selecting a different
checkpoint.

The model card's canonical serving recipe uses OneCompression at tag
`m3-serving-v1` plus repository-specific loader glue. Do not install that plugin
into the shared environment merely to make preflight pass. Run our mandatory
matrix smoke first. If the AutoRound arm fails with an unknown
`autoround_mixed` quantization method, missing OneCompression plugin, or loader
key mismatch, stop before production and return its complete logs and smoke
artifacts. That is an integration incompatibility for the primary agent to
resolve; do not silently switch only this arm to the model card's custom server,
because doing so would invalidate the paired direct-vLLM comparison.

```bash
git pull
cd /mnt/nfs/hoangduy/projects/llm-compressor
source <the-working-vllm-environment>/bin/activate
export PYTHONPATH="$PWD"
RUN_ID="$(date +%Y%m%d-%H%M%S)-m3-quality"
RUN_ROOT="results/m3-quality/$RUN_ID"
MATRIX=pipeline/configs/minimax_m3_quality_matrix.yaml
mkdir -p "$RUN_ROOT"

python -m pipeline.m3_quality_preflight \
  --matrix "$MATRIX" \
  --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/preflight.log"
```

Preflight intentionally performs the runtime-dependent work on the capable
cluster: verifies all checkpoint paths, resolves installed lm-eval task aliases
and leaf splits, creates immutable smoke/production sample manifests, builds
calibration-disjoint 2,048/49,152-token probe corpora, records tokenizer/chat
hashes and Git/software provenance, and writes checkpoint diagnostics.
Do not proceed unless `run_manifest.json`, `preflight/resolved_eval_config.yaml`,
both sample manifests, both probe corpora, `resolved_tasks.json`, and four
checkpoint diagnostic JSONs exist.

If preflight fails because a task alias or dataset split changed, diagnose the
installed lm-eval task registry locally and make the smallest documented fix.
Do not silently substitute a benchmark or reduce production sample counts.

### Mandatory smoke, then parallel production

First inspect the exact allocations without consuming GPUs:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$MATRIX" --run-root "$RUN_ROOT" --dry-run
```

It must print 4 arms and `total_nodes=5`. Then run:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$MATRIX" --run-root "$RUN_ROOT"
SMOKE_RC=$?
python -m json.tool "$RUN_ROOT/smoke_gate.json"
```

Stop and return evidence immediately if `SMOKE_RC != 0` or
`ready_for_production` is not true. Return every smoke stdout/stderr log,
`smoke_report.json`, `smoke_gate.json`, each `smoke_evidence.json`, each arm
manifest/return code, and the preflight artifacts. Include Slurm job IDs,
nodes, environment/version output, exception traces, and any deviation or
retry. Do not launch a five-hour run after a failed smoke.

After smoke passes:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  --smoke-gate "$RUN_ROOT/smoke_gate.json" --dry-run
```

It must print 8 arms and `total_nodes=10`. Launch all arms concurrently:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  --smoke-gate "$RUN_ROOT/smoke_gate.json"
PRODUCTION_RC=$?
```

Do not cancel healthy arms merely because another arm fails; preserve all
completed evidence. If cluster contention prevents ten simultaneous nodes,
keep the same arms and resource topology but start as many independent arms as
available, documenting start/end times and queueing.

### Aggregate and return contract

Run aggregation even when one production arm failed; its infrastructure report
is part of the diagnosis:

```bash
set +e
python -m pipeline.m3_quality_eval aggregate \
  --matrix "$MATRIX" --root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/aggregate.log"
AGGREGATE_RC=${PIPESTATUS[0]}
set -e
```

Commit and push the complete compact evidence tree and this handoff update.
Return all of the following so the primary agent can do the analysis without
asking for a rerun:

- `run_manifest.json`, all preflight manifests/configs/hashes, resolved task
  names and leaf sizes, probe corpora metadata, and checkpoint diagnostics;
- every arm's `arm_manifest.json`, `arm_complete.json`, `return_code.txt`,
  aggregate metrics, normalized per-sample JSONL, generation-health JSON,
  distributional probe JSONL+summary, and stdout/stderr log;
- root `matrix.json`, `gates.json`, `report.md`, `aggregate.log`, smoke report
  and gate;
- exact command lines, Git commit, Python/PyTorch/CUDA/vLLM/lm-eval versions,
  Slurm job IDs/nodes, wall times, exit codes, retries, OOMs, and deviations;
- log/artifact SHA-256 values if any large file must remain outside Git, plus
  its exact durable path. Never replace a missing artifact with a prose summary.

The decision metrics are downstream accuracy, paired accuracy delta and CI,
flip/regression/recovery rates, exact/asymptotic McNemar evidence, score
recovery, teacher-forced NLL/perplexity/top-k drift, generation degeneration,
and checkpoint quantization coverage/scales/saturation. Serving latency and
throughput remain explicitly out of scope until quality is resolved.

## AWQ re-quantization interruption

The latest AWQ preparation attempt did not complete. Both `awq-offsetfix` and
`awq-nosmooth` were terminated by Slurm step timeout during expert smoothing;
neither output directory is a valid checkpoint. See
`M3_AWQ_REQUANTIZATION_REPORT.md` for exact paths, nodes, timestamps, log
evidence, and the rerun requirements. Do not serve or publish either partial
output.

### Immediate rerun

The first retry was launched from a tool-managed background shell. Although no
`scancel`, `kill`, or other termination command was issued, both steps were
cancelled together after about six hours while their 96-hour allocations still
had days remaining. The exact source cannot be proven because the launcher
later disappeared and Slurm accounting records were unavailable. Treat
launcher/session teardown as a plausible contributor. The next retry must use a
persistent login-shell context or a detached/batch allocation that survives
the controlling session.

Do not change calibration, AWQ mappings, smoothing, or checkpoint export. From
a persistent login shell that is **not** inside an existing Slurm allocation,
pull the latest commit and run:

    test -z "${SLURM_JOB_ID:-}" || { echo "leave parent allocation first"; exit 1; }
    unset SRUN_ARGS
    TIME_LIMIT=96:00:00 bash pipeline/slurm/run_m3_awq_gptq_prepare_srun.sh

The already completed GPTQ preparation should be a cheap no-op; the two AWQ
variants must restart and run concurrently. Do not reuse or serve either
partial timestamped checkpoint from the interrupted attempt.

If both AWQ variants finish, verify each portable checkpoint has `config.json`,
`model.safetensors.index.json`, and every shard referenced by that index. Then
run only the pending AWQ finish phase against the existing staged matrix:

    MATRIX_ID=20260712-045912-awq-gptq-staged \
      bash pipeline/slurm/run_m3_awq_repair_finish_srun.sh

Commit the preparation logs, exact output paths, return codes, checkpoint
validation, the six AWQ arm artifacts, and the regenerated `comparison.json`.
If a preparation arm exits early again, do not retry dynamically: immediately
capture and commit its complete log plus `sacct`/`scontrol` output for the
allocation and step, including requested/effective time limit, state, exit
code, reason, and node, before the scheduler record expires.

## Superseding handoff: representative-layer AWQ diagnostic

Do **not** retry either full AWQ rebuild. The two cancellations did not test the
AWQ hypotheses and made full preparation the slow path. The active task is now
an in-memory six-arm diagnostic over layers 8, 31, and 59 for `offsetfix` and
`nosmooth`. Each arm quantizes one layer only, captures two reference and two
candidate propagation passes inside the sequential calibration pipeline, writes
compact local-fidelity evidence, and exits without saving a checkpoint.

Pull the latest commit, activate the cluster quantization environment, and run
the CPU/config preflight before allocating GPUs:

    python -m pytest -q \
      pipeline/tests/test_m3_awq_representative.py \
      pipeline/tests/test_m3_awq_representative_launcher.py \
      pipeline/tests/test_m3_node_exclusivity.py \
      pipeline/tests/test_minimax_m3_config.py \
      tests/llmcompressor/modeling/test_offset_norm_minimax_m3.py
    bash -n pipeline/slurm/run_m3_awq_representative_srun.sh

### Cursor-safe detached launch (required)

Launch from a login/control shell **outside every Slurm allocation**. Confirm
`[[ -z "${SLURM_JOB_ID:-}" ]]` before the real launch. Inside an existing
allocation, Slurm defines `srun --exclusive` as step-level `--exact`, so several
one-GPU steps may share one eight-GPU node. The wrapper now refuses that unsafe
context. From the outside context, each top-level `srun --exclusive` requests a
separate whole-node allocation; the six arms therefore require six nodes.

Cursor must **not** run `run_m3_awq_representative_srun.sh` directly as a
background tool task. `srun` is client-owned: destroying Cursor's controller
process cancels the Slurm steps. Do not use plain shell backgrounding or the
older detached-process wrappers for this matrix. A detached tmux server must
own the controller.

Each representative arm must also request an exclusive node. A one-GPU
request alone is insufficient: Slurm can place multiple full-model arms on the
same eight-GPU node, exhausting host RAM and producing `rc=137` (SIGKILL/OOM).
The launcher therefore includes `srun --exclusive --nodes=1`; preserve this
flag in all future runs and verify the six arms land on six distinct nodes.

First print and inspect the tmux launch plus the exact evidence roots. This is
read-only and does not create a session:

    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-awq-representative"
    SESSION_NAME="m3-awq-$RUN_ID"
    DRY_RUN=1 RUN_ID="$RUN_ID" SESSION_NAME="$SESSION_NAME" \
      bash pipeline/slurm/start_m3_awq_representative_tmux.sh

The dry run prints all six underlying `srun` commands. Record its literal
`RUN_ID` and `SESSION_NAME`. Then, in a new short **foreground** Cursor tool
call, reuse those exact printed values to create the detached session. Do not
recompute the timestamp, add `&`, or ask Cursor to keep this call alive:

    RUN_ID='<RUN_ID printed by the dry run>'
    SESSION_NAME='<SESSION_NAME printed by the dry run>'
    RUN_ID="$RUN_ID" SESSION_NAME="$SESSION_NAME" \
      bash pipeline/slurm/start_m3_awq_representative_tmux.sh

The wrapper returns only after `tmux has-session` succeeds. Record the printed
`RUN_ID`, `SESSION_NAME`, `LOG_ROOT`, `RESULT_ROOT`, and monitoring commands.
At that point the Cursor tool may exit or be interrupted; the tmux server, not
Cursor, owns the `srun` controller.

The wrapper rejects a live duplicate session and any completed/partial evidence
already present for that `RUN_ID`. Never delete those guards to reuse an ID;
choose a fresh ID instead.

Verify survival from a **new** Cursor tool call by copying the literal commands
printed by the wrapper. Shell variables do not carry between Cursor tool calls;
if entering them manually, first restore the printed values:

    SESSION_NAME='<printed SESSION_NAME>'
    RESULT_ROOT='<printed RESULT_ROOT>'
    tmux has-session -t "=$SESSION_NAME"
    tmux capture-pane -pt "=$SESSION_NAME" -S -80
    squeue -u "$USER" -o '%.18i %.28j %.8T %.10M %.10l %.6D %R'

Before accepting the run, confirm the six running arm jobs show six distinct
single-node `NODELIST` values. Pending jobs are acceptable; two running arms on
the same node are not. Stop and return scheduler evidence if colocation occurs.

Poll with `tmux capture-pane` and `tail` of the printed `controller.log`; these
commands return immediately. Attaching is optional:

    tmux attach-session -t "=$SESSION_NAME"

Detach manually with `Ctrl-b d`. Never run `tmux kill-session` or
`tmux kill-server` while an arm is active. Do not leave a Cursor tool blocked in
`tmux attach-session` for monitoring.

Interpret session state carefully:

- tmux exists and Slurm steps exist: healthy; continue polling;
- tmux is gone and `controller.rc` exists: the controller finished; read the rc,
  `matrix.json`, `report.md`, and logs;
- tmux is gone, `controller.rc` is absent, or Slurm steps vanish unexpectedly:
  treat as infrastructure failure, immediately capture `scontrol`/`sacct` and
  logs, and do not restart dynamically.

The tmux-owned controller creates fresh run-specific log and result roots,
starts all six one-GPU `srun` steps concurrently, waits for every arm, and
aggregates after all returns.

Do not change the production 512x2048 calibration, W4AFP8 scheme, AWQ grid,
layer set, mappings, thresholds, or variants at runtime. Do not save/export a
checkpoint, launch vLLM, run HTTP generation, retry a failed arm, or begin the
CUDA-graph issue. A quality failure is a valid result.

Return through Git:

- the complete run-specific result tree, including every `start.json`,
  successful `arm.json`, per-arm `rc`, `matrix.json`, and `report.md`;
- all six complete logs when reasonably sized, otherwise durable log paths,
  sizes, and SHA-256 hashes plus committed head/tail excerpts;
- exact command, Git revision, environment/package versions, run/result/log
  roots, Slurm job/step IDs, nodes, elapsed times, return codes, retries (expected
  zero), OOMs, and deviations;
- immediate `scontrol`/`sacct` evidence for any abnormal exit before scheduler
  records disappear.

Commit and push the evidence, then stop for primary-agent analysis. Do not
interpret a mostly passing matrix as permission to start full quantization.


## Historical handoff: three-model smoke recovery (deferred)

This section supersedes the four-model commands above. AutoRound is deferred:
its mixed-bit checkpoint requires a pinned OneCompression plugin plus a
repository-specific translating/dequantizing loader in both lm-eval and the
distributional probe. Do not download, mutate, substitute, or launch it in this
retry. The active comparison is BF16, in-house GPTQ, and cyankiwi AWQ.

Pull the latest branch and create a fresh run root because the previous
resolved config contains invalid reasoning arguments:

```bash
git pull
cd /mnt/nfs/hoangduy/projects/llm-compressor
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD"
RUN_ID="$(date +%Y%m%d-%H%M%S)-m3-quality-3model"
RUN_ROOT="results/m3-quality/$RUN_ID"
MATRIX=pipeline/configs/minimax_m3_quality_matrix.yaml
mkdir -p "$RUN_ROOT"
python -m pipeline.m3_quality_preflight \
  --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/preflight.log"
```

Preflight must report three active models, six production arms, adaptive
MiniMax reasoning (`enable_thinking: null`,
`think_end_token: </mm:think>`), and checkpoint diagnostics for exactly BF16,
in-house GPTQ, and cyankiwi AWQ.

Inspect the launch plan:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$MATRIX" --run-root "$RUN_ROOT" --dry-run
```

Expected: one synchronous two-node Ray topology check followed by three
concurrent model arms with `total_nodes=4`. Then run it:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$MATRIX" --run-root "$RUN_ROOT"
SMOKE_RC=$?
python -m json.tool "$RUN_ROOT/ray_preflight/gate.json"
python -m json.tool "$RUN_ROOT/smoke_gate.json"
```

The Ray gate must show two alive nodes and at least 16 visible GPUs. Return
`ray-preflight.out/err`, `ray_preflight/rank-*.log`, rank JSON, `ray_status.txt`,
`ray_nodes.json`, and every BF16 `ray_runtime/` artifact even if topology fails.
The launcher must still write `smoke_gate.json` for model/probe failures; stop
before production unless `ready_for_production` is exactly true.

After a passed smoke only:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  --smoke-gate "$RUN_ROOT/smoke_gate.json" --dry-run
# Expected: 6 arms, total_nodes=8.

bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  --smoke-gate "$RUN_ROOT/smoke_gate.json"
```

Commit and push the same complete artifact contract listed above: all preflight
files, rank/Ray evidence, arm manifests and return codes, raw stdout/stderr,
normalized samples, generation health, probe records/summaries, root matrix,
gates and report, exact commands, Slurm job/step IDs and nodes, software
versions, wall times, retries, and deviations. Do not begin AutoRound adapter or
serving-performance work.

## Representative diagnostic final status

The current six-arm run is documented in
`M3_AWQ_REPRESENTATIVE_DIAGNOSTIC_REPORT.md`. At the final 2026-07-12
~17:15 UTC verification, the exclusive-node launch had completed and all six
arms exited with `rc=1`. Every arm raised `ZeroDivisionError` while computing
`avg_reduction` in `llmcompressor/modifiers/transform/awq/base.py:846`.
The compact matrix therefore reports six `infrastructure_failure` arms and an
`incomplete` overall verdict.

The complete logs and compact result artifacts remain at the durable absolute
paths listed in the report, with SHA256 hashes recorded there. Do not
start a full AWQ rebuild until the empty-metric failure is analyzed and the
representative diagnostic is corrected.


## Active correction handoff: AWQ lifecycle smoke and repaired GPTQ validation

The prior exclusive-node representative run was infrastructure-clean but all
six arms finalized with zero AWQ mapping metrics. Pull the latest commit. The
AWQ core now records explicit skipped mappings and safely handles an empty
summary; the representative harness persists `lifecycle.json` with resolved,
completed, skipped, and unprocessed mappings.

First run exactly one arm from a login/control shell outside Slurm. Use a fresh
ID and tmux session:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-awq-one-arm"
SESSION_NAME="m3-awq-$RUN_ID"
DRY_RUN=1 ARM_FILTER=offsetfix-layer8 RUN_ID="$RUN_ID" \
  SESSION_NAME="$SESSION_NAME" \
  bash pipeline/slurm/start_m3_awq_representative_tmux.sh
ARM_FILTER=offsetfix-layer8 RUN_ID="$RUN_ID" SESSION_NAME="$SESSION_NAME" \
  bash pipeline/slurm/start_m3_awq_representative_tmux.sh
```

A useful smoke requires `controller.rc=0`, `offsetfix-layer8/rc=0`, a nonempty
`lifecycle.json`, `completed_mapping_count > 0`, and `arm.json`. If it fails,
return lifecycle evidence and the complete log immediately; do not run the
other five arms. The counts now distinguish uncached/unprocessed mappings from
explicit `no_parent_outputs` or `nonfinite_parent_outputs` skips.

If and only if the one-arm smoke passes, immediately start a fresh unfiltered
six-arm tmux run on six exclusive nodes using the standard commands above. Do
not reuse the one-arm result root.

In parallel with the AWQ one-arm smoke, run a fresh BF16-only quality smoke
from `M3_QUALITY_THREE_MODEL_SMOKE_RECOVERY_HANDOFF.md` with
`QUALITY_ARM_FILTER=bf16`. BF16 now uses two exclusive 8xH100 nodes, TP8×PP2, and the
Ray backend; the failed TP16×PP1 path is removed from the baseline workflow.
Return its paired probe and smoke metrics. Do not start production until the
primary agent validates the BF16 comparison.

## Active handoff pointer

The active cluster tasks are the two parallel tracks in **Active correction
handoff** above: one filtered AWQ lifecycle smoke (expanding to six arms only on
success) and the corrected three-model quality smoke including BF16. Both must
use their tmux wrappers from outside every Slurm allocation. Return evidence and
stop before production or full-model re-quantization.
