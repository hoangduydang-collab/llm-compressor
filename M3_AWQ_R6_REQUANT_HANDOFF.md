# Execution packet: M3 AWQ r6 requant (up->down fold removed) + stuck-item probe

- Protocol version: 1
- State: READY_FOR_EXECUTOR
- Packet revision: 2026-07-23-r1
- Planner owner: planner session 2026-07-23 (root-cause: up->down fold not function-preserving)
- Intended executor: any executor
- Base Git commit: 3a323808270f486047ed53110229c83db5866753
- Decision question: does removing the up->down AWQ smoothing fold (r6) collapse
  the in-house AWQ GPQA non-termination excess toward GPTQ's rate?

## Objective and hypothesis

BUGS_AND_FIXES.md ("AWQ up->down smoothing fold is not function-preserving on
MiniMax-M3", 2026-07-23) identifies the r5 recipe's per-expert up->down fold as
the leading root-cause candidate for the AWQ reasoning non-termination
(M3_OFFICIAL_QUALITY_RESULTS.html). r6 requantizes with that mapping removed
(commit above). This run (a) produces the r6 W4AFP8 checkpoint under the
unchanged 512x2048 calibration contract, (b) proves via a fail-closed
weight-level gate that the fold is absent, and (c) re-runs the existing
stuck-item sampling probe on the r6 arm. Prediction: r6's greedy
non-termination rate on the tok64k AWQ exhausted-item set drops from r5's
~76% toward the GPTQ control's ~52% (probe scale), and its excess budget
exhaustion largely disappears. The result decides between "root cause
confirmed -> paired eval + ship path" and "residual RTN damage -> r7 / GPTQ".

## Scope and non-goals

- In scope: one AWQ-only full-calibration DDP quantization; the two checkpoint
  gates below; vLLM re-export; one single-node sampling-probe arm.
- Not authorized: GPTQ requantization; any full quality-suite evaluation; any
  recipe/config/code edits; serving experiments beyond the probe arm;
  deleting or overwriting any existing checkpoint or result tree.

## Preconditions and exact environment

- Repository path: /mnt/nfs/hoangduy/projects/llm-compressor
- Branch: duy-branch
- Environment activation: `source /mnt/nfs/hoangduy/env.sh && source /mnt/nfs/hoangduy/venvs/quant/bin/activate`
  (the launchers source these themselves; activation is only needed for the
  preflight/gate commands you run directly)
- Required environment variables: none beyond those set inline in Commands
- Required package/version checks:
  `python -c "import llmcompressor, compressed_tensors, torch; print(torch.__version__)"`
  (must import cleanly inside the quant venv)

## Required inputs

| Input | Exact path or identifier | Required validation |
| --- | --- | --- |
| BF16 base | /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3 | `test -f .../model.safetensors.index.json` |
| Quant config | pipeline/configs/minimax_m3_distributed_awq_full.yaml | tracked at base commit (no edits) |
| Calibration data | HuggingFaceH4/ultrachat_200k (HF cache) | resolved by pipeline; cache at /mnt/nfs/hoangduy/cache/huggingface |
| tok64k probe doc set | /mnt/nfs/hoangduy/results/m3-official-quality/20260721T154830Z-tok64k/results | `ls` the glob used in Launch step 3 resolves at least one samples file |
| r5 probe baseline (comparison only, read-only) | /mnt/nfs/hoangduy/results/m3-sampling-probe/20260722T031036Z/sampling/awq.jsonl | exists |

## Workspace policy

- Protected paths: all tracked files; /mnt/nfs/hoangduy/hf_assets/**; all
  existing /mnt/nfs/hoangduy/results/** trees (append-only: new run roots only).
- Permitted untracked roots: /mnt/nfs/hoangduy/results/m3-distributed-awq-full/$RUN_ID,
  /mnt/nfs/hoangduy/logs/m3-distributed-awq-full/$RUN_ID,
  /mnt/nfs/hoangduy/offload/m3-distributed-awq-full/$RUN_ID,
  /mnt/nfs/hoangduy/results/m3-sampling-probe/$PROBE_ID.
- Record and proceed: pre-existing untracked artifacts/ and results/ entries in
  `git status` (planner's local analysis outputs; do not touch).
- Stop: any tracked-file diff at setup; RUN_ID collision with an existing
  directory; base commit not reachable.

## Resource contract

- Nodes: 1 x 8xH100 (quant), then 1 x 8xH100 (probe arm). Sequential; no
  concurrent second node required.
- Exclusivity: exclusive (`srun --exclusive`), as encoded in the launchers.
- Task/process layout: quant = torchrun 8 ranks, 1 node (launcher-owned);
  probe = vLLM serve TP8 + local probe driver (launcher-owned).
- Time limit: quant 24:00:00 (expect ~7-8 h; r5 took ~7 h); probe 12:00:00
  (expect ~2-4 h).
- Expected runtime total: 10-13 h.

## Commands

Run the controller inside a persistent tmux session on the login node
(`tmux new -s m3-r6`). Never launch from inside an existing allocation.

### Setup and revision verification

```bash
cd /mnt/nfs/hoangduy/projects/llm-compressor
git fetch origin && git checkout duy-branch && git pull --ff-only
git rev-parse HEAD   # MUST print 3a323808270f486047ed53110229c83db5866753 or a descendant that contains it
git merge-base --is-ancestor 3a323808270f486047ed53110229c83db5866753 HEAD && echo ancestor-ok
git status --porcelain | grep -v '^??' && { echo "STOP: tracked changes"; } || echo tracked-clean
```

### Preflight

```bash
source /mnt/nfs/hoangduy/env.sh && source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
python -m pytest tests/pipeline/test_minimax_m3_awq_mappings.py -q
```

Expected: `4 passed`, return code 0.
Stop if: any test fails (the r6 mapping set is wrong at this revision).

### Dry run

```bash
cd /mnt/nfs/hoangduy/projects/llm-compressor
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-ddp-awq-full-r6-noupdown"
DRY_RUN=1 METHODS=awq EVIDENCE_ONLY=0 \
  CONFIG=pipeline/configs/minimax_m3_distributed_awq_full.yaml \
  RESULT_ROOT=/mnt/nfs/hoangduy/results/m3-distributed-awq-full/$RUN_ID \
  LOG_ROOT=/mnt/nfs/hoangduy/logs/m3-distributed-awq-full/$RUN_ID \
  OFFLOAD_ROOT=/mnt/nfs/hoangduy/offload/m3-distributed-awq-full/$RUN_ID \
  bash pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh
```

Expected: exactly ONE printed srun command (awq only, no gptq), containing
`--stage quantize`, `quantization.method=awq`, and NO `--evidence-only` flag.

### Launch

Step 1 — quantization (reuse the SAME RUN_ID from the dry run):

```bash
METHODS=awq EVIDENCE_ONLY=0 \
  CONFIG=pipeline/configs/minimax_m3_distributed_awq_full.yaml \
  RESULT_ROOT=/mnt/nfs/hoangduy/results/m3-distributed-awq-full/$RUN_ID \
  LOG_ROOT=/mnt/nfs/hoangduy/logs/m3-distributed-awq-full/$RUN_ID \
  OFFLOAD_ROOT=/mnt/nfs/hoangduy/offload/m3-distributed-awq-full/$RUN_ID \
  RUN_ID=$RUN_ID \
  bash pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh \
  2>&1 | tee /mnt/nfs/hoangduy/logs/m3-distributed-awq-full/$RUN_ID-controller.log
```

Step 2 — checkpoint gates + re-export (after step 1 exits 0). `CKPT` is the
single timestamped checkpoint directory the run creates:

```bash
CKPT=$(ls -d /mnt/nfs/hoangduy/results/m3-distributed-awq-full/$RUN_ID/awq/MiniMax-M3-awq-W4AFP8/*/checkpoint | head -1)
echo "CKPT=$CKPT"

# Gate A (NEW, decisive for this packet): no residual up->down fold.
python -m pipeline.m3_verify_no_updown_fold \
  --base /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3 \
  --checkpoint "$CKPT" \
  --output "$(dirname "$CKPT")/no_updown_fold_gate.json"
# MUST exit 0 (gate PASSED). r5 fails this gate with relerr med 0.20-0.25;
# a clean r6 sits at int4 noise (~0.03-0.08).

# Gate B: MoE-input fold consistency + magnitude (unchanged from r5 practice).
python -m pipeline.m3_checkpoint_scale_audit \
  --base /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3 \
  --reference /mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260720T060340Z-m3-ddp-awq-full-r5-deadchan/awq/MiniMax-M3-awq-W4AFP8/20260720-060402/checkpoint \
  --awq "$CKPT" \
  --gptq /mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123 \
  --output "$(dirname "$CKPT")/scale_audit_r6_vs_base.json"

# Re-export for stock vLLM (same step r5 used; output sibling of checkpoint):
python -m pipeline.reexport_minimax_m3_vllm "$CKPT" "$CKPT-vllm-w123"
```

Step 3 — stuck-item probe, single arm, one node (after step 2 gates pass).
Uses the existing probe arm runner with the r6 checkpoint against the SAME
tok64k AWQ doc set as the 07-22 probe (directly comparable):

```bash
export PROBE_ID="$(date -u +%Y%m%dT%H%M%SZ)-r6"
export PROBE_ROOT=/mnt/nfs/hoangduy/results/m3-sampling-probe/$PROBE_ID
mkdir -p "$PROBE_ROOT"
TOK=/mnt/nfs/hoangduy/results/m3-official-quality/20260721T154830Z-tok64k/results

ROOT="$PROBE_ROOT" ARM=awq-r6 MODE=local CKPT="$CKPT-vllm-w123" PORT=8004 \
SAMPLES_GLOB="$TOK/minimax-m3-awq-inhouse/*/quality/_lm_eval/*gpqa*/samples_*.jsonl" \
N_EXHAUSTED=50 \
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
     --time=12:00:00 --kill-on-bad-exit=1 --job-name=m3-sprobe-awq-r6 --export=ALL \
     bash pipeline/slurm/sampling_probe_arm.sh \
     > "$PROBE_ROOT/awq-r6-srun.log" 2>&1
echo "probe rc=$?"
```

If the SAMPLES_GLOB above resolves nothing, STOP and check the actual layout
under $TOK (the glob must select the same GPQA samples files the 07-22 probe
used for the in-house AWQ arm; see `G()` in
pipeline/slurm/run_sampling_probe_srun.sh at the base commit for the canonical
glob construction) — do not substitute a different doc set silently.

### Monitoring

```bash
# quant progress (per-mapping heartbeat):
tail -f /mnt/nfs/hoangduy/logs/m3-distributed-awq-full/$RUN_ID/awq/torchrun.err | grep --line-buffered -E "Smoothing|error reduction|Propagating|Calibrating|ERROR|Traceback"
# node health snapshot files: $LOG_ROOT/awq/node_preflight.txt, environment.txt
# probe: tail -f $PROBE_ROOT/awq-r6-srun.log
```

### Aggregation and packaging

```bash
python - <<'PY'
import json, glob, os
root = os.environ["PROBE_ROOT"]
rows = [json.loads(l) for f in glob.glob(f"{root}/sampling/*.jsonl") for l in open(f)]
by = {}
for r in rows:
    k = (r.get("cohort"), r.get("regime"))
    by.setdefault(k, []).append(r)
for k, v in sorted(by.items()):
    n = len(v)
    nt = sum(1 for r in v if r.get("finish_reason") == "length")
    print(f"{k}: n={n} non-terminated={nt} ({nt/max(n,1)*100:.1f}%)")
PY
```

## Expected jobs and independence rules

| Job or arm | Resources | Expected output | Failure effect on other jobs |
| --- | --- | --- | --- |
| awq r6 quant | 1 node, 8xH100, torchrun 8 ranks | checkpoint + quant_metrics under $RESULT_ROOT/awq | probe cannot run; stop after evidence capture |
| gates + reexport | login/CPU | no_updown_fold_gate.json (passed:true), scale audit json, $CKPT-vllm-w123 | gate failure = STOP, return evidence |
| probe arm awq-r6 | 1 node, 8xH100 | $PROBE_ROOT/sampling/awq-r6.jsonl | independent of quant node |

## Success gates and expected artifacts

- Gate: `no_updown_fold_gate.json` field `passed == true` (exit code 0).
- Gate: scale audit r6 — MoE-input norm-implied scale mean per layer in
  [0.05, 20] (same bounds as r5 practice), no non-finite values.
- Gate: probe arm exits 0 and `$PROBE_ROOT/sampling/*.jsonl` has both greedy
  and sampled rows for ≥ 45 exhausted-cohort docs.
- Expected artifacts: checkpoint tree + `quant_metrics.rank-*.jsonl` +
  `recipe.json` + `metadata.json`; both gate JSONs; `$CKPT-vllm-w123`;
  probe jsonl + srun log.

## Allowed adaptations

- Retry the quant launch on a different node after an infrastructure-class
  failure (preflight shm/mem gate, node fault) — fresh RUN_ID required.
- Probe port may be changed if 8004 is occupied on the allocated node.

## Pre-authorized record-and-proceed conditions

- Stale `/dev/shm/torch_*` cleanup messages in node preflight (handled by the
  launcher; record the count).
- Pre-existing untracked planner artifacts in `git status` (record, ignore).

## Pre-authorized retries

- Trigger: node-level infrastructure failure before quantization reaches the
  Smoothing phase (preflight gate, OOM at load, NCCL init failure).
- Maximum retry count: 2
- Fresh run ID required: yes
- Inputs that must remain unchanged: config, base commit, MODEL_ID, calibration contract.

## Stop-and-return conditions

- Gate A (`m3_verify_no_updown_fold`) FAILS on the r6 checkpoint — this
  falsifies the code change and must come back to the planner untouched.
- Any tracked-file modification required to proceed.
- Quantization crashes during/after the Smoothing phase (post-mortem class
  bugs live here; preserve logs + partial evidence, do not retry blindly).
- SAMPLES_GLOB resolves to nothing (see Launch step 3 note).

## Prohibited actions

- Editing any tracked file (including launchers and configs) — the METHODS
  knob and gates are already in the base commit.
- Re-running GPTQ, BF16, or cyankiwi probe arms (07-22 baselines are reused).
- Deleting/overwriting any existing checkpoint or results tree.
- Any full quality-suite evaluation (that is a separate, later packet).

## Return contract

- Commit to the repo (small files only) under a new `results/m3-awq-r6/`
  folder: no_updown_fold_gate.json, scale_audit_r6_vs_base.json, the probe
  aggregation printout as summary.txt, controller/srun rc files, and a
  RETURN.md with: scheduler job ids, node names, exact commands as run,
  start/end timestamps, return codes, and the CKPT absolute path.
- Large artifacts (checkpoint, vllm-w123 export, probe jsonl): report absolute
  path, `du -sb` byte size, and SHA-256 of the safetensors index (not the
  shards) in RETURN.md.
- Required factual table in RETURN.md: for cohort=exhausted — greedy
  non-terminated %, sampled non-terminated %, n docs — side by side with the
  same numbers computed from the 07-22 r5 probe
  (/mnt/nfs/hoangduy/results/m3-sampling-probe/20260722T031036Z/sampling/awq.jsonl)
  and the GPTQ control (.../gptq.jsonl). No interpretation required; the
  planner owns the verdict.
