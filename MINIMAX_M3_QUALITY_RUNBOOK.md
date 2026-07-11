# MiniMax-M3 routed-expert diagnostic runbook

## Objective

Run three canonical offline diagnostic arms concurrently through `srun` to
identify whether the portable W4A8 candidate fails because of activation
handling or routed-expert INT4 weights/loading.

The completed canonical matrix `20260711-135100-canonical-chat` established:

- cyankiwi passes offline and HTTP with identical correct answers;
- the candidate produces identical garbage offline and HTTP;
- prompts, token counts, eager serving, and software envelopes match;
- the candidate keeps attention, MSA indexers, shared experts, dense layers
  0–2, vision, and `lm_head` unquantized, leaving routed experts as the primary
  quantized boundary.

This run does not fix the candidate, re-quantize, use HTTP, enable CUDA graphs,
or investigate the second serving issue.

## Arms and invariants

| Arm | Checkpoint/runtime scheme | Purpose |
| --- | --- | --- |
| `reference_w4a16` | cyankiwi W4A16 | Valid diagnostic control |
| `candidate_w4a8` | portable candidate W4A8 | Reproduce candidate failure with probes |
| `candidate_w4a16` | same candidate payload, config-only activations disabled | Separate W4A8 activation handling from INT4 weights/loading |

Each arm uses a different exclusive eight-GPU node and the same pushed commit,
environment, canonical chat prompts, eager TP8+EP, block size 128, FP8 KV cache,
2048 context, 0.85 utilization, disabled custom all-reduce, and disabled shared
expert auxiliary stream.

Diagnostics are identical:

```text
M3_LOAD_AUDIT=1
M3_PARAM_FINGERPRINT=1
M3_PARAM_FINGERPRINT_LAYERS=3,59
M3_MOE_PROBE=1
M3_MOE_PROBE_RECOMPUTE=1
M3_MOE_PROBE_MAX_TOKENS=256
```

The repaired fingerprint sampler uses bounded strided slices rather than CUDA
`linspace/index_select`. The MoE probe records rank-aligned first-prefill input,
routed output, shared output, and combined output norms/digests. The W4A16 arm
uses a metadata overlay and never modifies candidate files.

## Preflight

```bash
git pull --ff-only origin duy-branch
git status --short
git rev-parse HEAD

test -x pipeline/slurm/test_m3_routed_diagnostics_arm.sh
test -x pipeline/slurm/run_m3_routed_diagnostics_srun.sh
test -f pipeline/m3_routed_diagnostics.py

REFERENCE_CKPT=/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4
CANDIDATE_CKPT=/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123
MODEL_ID=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3

test -f "$REFERENCE_CKPT/config.json"
test -f "$REFERENCE_CKPT/model.safetensors.index.json"
test -f "$CANDIDATE_CKPT/config.json"
test -f "$CANDIDATE_CKPT/model.safetensors.index.json"
test -f "$MODEL_ID/config.json"

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD"

pytest -q \
  pipeline/tests/test_patch_vllm_m3_serve.py \
  pipeline/tests/test_m3_quality_evidence.py \
  pipeline/tests/test_m3_routed_diagnostics.py \
  pipeline/tests/test_m3_routed_diagnostics_runner.py \
  pipeline/tests/test_run_m3_routed_diagnostics_srun.py \
  pipeline/tests/test_serve_verify_quality.py

MATRIX_ID=preflight-routed-diag DRY_RUN=1 \
  bash pipeline/slurm/run_m3_routed_diagnostics_srun.sh
```

The worktree must be clean. Dry-run output must contain exactly three `srun`
commands, each with `--exclusive --nodes=1 --ntasks=1 --gres=gpu:8`, and no
`sbatch` command.

## Live command

Run from a Slurm context allowed to start three concurrent `srun` allocations:

```bash
MATRIX_ID="$(date +%Y%m%d-%H%M%S)-routed-diagnostics"
MATRIX_ID="$MATRIX_ID" \
TIME_LIMIT=02:00:00 \
bash pipeline/slurm/run_m3_routed_diagnostics_srun.sh \
  2>&1 | tee "/mnt/nfs/hoangduy/logs/m3-routed-diagnostics-$MATRIX_ID-launcher.log"
```

If required, pass a recorded scheduler override such as
`SRUN_ARGS="--partition=h100"`. The executor may adapt reservation mechanics,
node constraints, time, and NFS roots, but must preserve separate nodes and all
quality/diagnostic variables. Do not translate this run back to `sbatch`.

The launcher updates and checks the shared vLLM site-packages once before any
workers spawn, starts all three `srun` commands in parallel, waits for every arm,
rebundles after logs close, and writes the aggregate comparison even when an arm
fails.

## Required evidence

```text
results/m3-routed-diagnostics/<matrix_id>/
├── comparison.json
├── reference_w4a16/
├── candidate_w4a8/
└── candidate_w4a16/
```

Each arm must contain:

- `arm_manifest.json` with code/checkpoint hashes, node/job, scheme, diagnostics,
  deviations, retries, and return code;
- unmodified `serve_report.json` and canonical raw outputs;
- `parameter_fingerprints.jsonl` plus summaries;
- `moe_probe_records.jsonl` with records from all eight ranks;
- `loader_audit.txt`, `notable_log_excerpt.txt`, patch status, software/GPU
  provenance, and full-log artifact hashes/retention.

Interpret aggregate verdicts as follows:

- `w4a8_activation_boundary`: candidate W4A16 recovers while W4A8 fails;
- `routed_weight_or_loader_boundary`: both candidate schemes fail after the two
  candidate arms match at their first-MoE inputs;
- `unquantized_load_boundary`: `lm_head` or shared-expert fingerprints differ
  between reference and candidate;
- `overlay_pre_moe_divergence`: the candidate W4A8 and config-only W4A16 arms
  do not enter their first routed expert identically, invalidating the overlay
  as a one-variable control;
- `inconclusive_missing_diagnostics`: a required loader/fingerprint/probe signal
  is absent;
- `invalid_reference`, `infrastructure_failure`, or `inconclusive_missing_arms`:
  do not diagnose the candidate past that boundary.

Do not treat different routed-expert hashes between cyankiwi and the candidate as
a defect by themselves: their INT4 schemes differ. Only `lm_head` and shared-expert fingerprints are exact cross-checkpoint
controls. Reference attention is W4A16 while candidate attention is BF16, and
the MSA indexer may be fused into QKV, so neither is required to hash equally.
Likewise, reference/candidate first-MoE input digests may differ legitimately;
exact first-input equality is required only between candidate W4A8 and its W4A16
overlay. The decisive signals are that candidate-pair control, routed/combined
norms, and W4A16 recovery.

## Commit and return

```bash
EVIDENCE_ROOT="results/m3-routed-diagnostics/$MATRIX_ID"
python -m json.tool "$EVIDENCE_ROOT/comparison.json"
find "$EVIDENCE_ROOT" -type f -size +5M -print
rg -n -i 'api[_-]?key|token=|authorization:|bearer ' "$EVIDENCE_ROOT" || true
git add "$EVIDENCE_ROOT"
git commit -m "data: add MiniMax-M3 routed diagnostics $MATRIX_ID"
git push origin duy-branch
```

Return the result and code commits, matrix ID, all three `srun` job IDs/nodes,
aggregate and arm outcomes, deviations/retries, missing diagnostics, and full-log
paths/hashes/retention. Stop for primary-agent analysis; do not implement a fix
or resume CUDA-graph RCA.
