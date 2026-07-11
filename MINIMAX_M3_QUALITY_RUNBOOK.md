# MiniMax-M3 paired quality runbook

## Objective and ownership

Run one eager-mode comparison between the working
`cyankiwi/MiniMax-M3-AWQ-INT4` reference and the portable full-calibration
AWQ W4A8 checkpoint. Return enough compact evidence through Git for analysis on
the non-GPU cluster.

The primary agent owns experiment design, static analysis, and evidence
interpretation. The GPU agent owns preflight, execution, runtime-only
adaptation, preservation of full logs, and the compact result commit. Use
runtime judgment for scheduler, node, paths, and owned stale processes, but do
not bundle a speculative loader fix into this comparison.

This run does not investigate CUDA graphs, re-quantize, or delete checkpoints.

## Current follow-up: reference without parameter fingerprinting

This section supersedes the paired command below for the next allocation.
Run **only cyankiwi** with one variable changed from failed run
`20260711-114831`:

```text
M3_PARAM_FINGERPRINT: 1 -> 0
```

Keep `M3_LOAD_AUDIT=1` and `M3_MOE_PROBE=1`. The failed run produced a
device assertion after the fingerprint hook sampled loaded CUDA parameters;
compressed-tensors later reported the poisoned stream at `k_scale.item()`.
This A/B determines whether the reference baseline is healthy when the
suspected perturbation is absent. Do not run the candidate and do not repair
the sampler in the same trial.

After the normal preflight and on one clean eight-GPU node:

```bash
set -o pipefail
RUN_ID="$(date +%Y%m%d-%H%M%S)-reference-no-fingerprint"
RUN_DIR="/mnt/nfs/hoangduy/logs/m3-paired-quality/$RUN_ID"
CASE_DIR="$RUN_DIR/cyankiwi_reference"
EVIDENCE_DIR="$PWD/results/m3-paired-quality/$RUN_ID"
REFERENCE_CKPT=/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4
MODEL_ID=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
CONFIG=pipeline/configs/minimax_m3_full_calib.yaml

mkdir -p "$CASE_DIR" "$EVIDENCE_DIR/cyankiwi_reference"
ln -s "$(realpath "$REFERENCE_CKPT")" "$CASE_DIR/checkpoint"

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export FLASHINFER_USE_CUDA_NORM=1
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
export M3_LOAD_AUDIT=1
export M3_MOE_PROBE=1
export M3_PARAM_FINGERPRINT=0
export M3_QUALITY_CASE=cyankiwi_reference_no_fingerprint

python pipeline/slurm/patch_vllm_m3_serve.py >"$RUN_DIR/patch_status.txt" 2>&1
FORCE=0 MIN_FREE_GIB=70 bash pipeline/slurm/free_gpus.sh

date -Is >"$CASE_DIR/started_at.txt"
set +e
python -m pipeline.run --config "$CONFIG" --stage serve \
  --checkpoint "$CASE_DIR/checkpoint" \
  --set model.id="$MODEL_ID" \
  --set serve.tensor_parallel_size=8 \
  --set serve.enable_expert_parallel=true \
  --set serve.block_size=128 \
  --set serve.kv_cache_dtype=fp8 \
  --set serve.max_model_len=2048 \
  --set serve.gpu_memory_utilization=0.85 \
  --set serve.enforce_eager=true \
  --set serve.disable_custom_all_reduce=true \
  --set eval.enabled=false \
  2>&1 | tee "$CASE_DIR/serve.log"
RC="${PIPESTATUS[0]}"
set -e
date -Is >"$CASE_DIR/finished_at.txt"
echo "$RC" >"$CASE_DIR/return_code.txt"
```

Stop after this reference run regardless of outcome. A pass confirms only that
fingerprinting perturbed the previous reference run; it does not authorize
running or fixing the candidate yet.

### Required follow-up return

Commit `results/m3-paired-quality/<run_id>/` containing:

- `run_manifest.json`: code commit, exact command, start/end, host/job/GPU
  provenance, checkpoint config/index hashes, all diagnostic flags (including
  fingerprint `0`), return code, deviations, and retry history;
- unmodified `serve_report.json` with both raw quality outputs if generation
  was reached;
- `loader_audit.txt`, `moe_probe.txt`, and the last 300 notable
  warning/error/diagnostic lines;
- software versions, `nvidia-smi` inventory/topology, and patch status;
- `artifact_index.json` with absolute paths, sizes, SHA-256 hashes, and
  retention for the full serve and operator logs;
- `comparison.json` with one of:
  `reference_pass_without_fingerprint`,
  `reference_failed_without_fingerprint`, or
  `inconclusive_missing_evidence`.

`parameter_fingerprints.jsonl` must be empty or absent by design. Loader
audit and MoE evidence are still required when their respective stage is
reached. Preserve the earliest traceback around the first CUDA error rather
than only the later `empty_cache()` report.

Before pushing, run the existing size/secret checks and name both the result
commit and executed code commit in the return message.

## Comparison contract

These are invariants between the two cases:

- repository commit and clean worktree;
- Python/vLLM environment, node, and GPU topology;
- TP=8, expert parallelism, eager execution, block size 128, and FP8 KV cache;
- `max_model_len=2048` and GPU utilization 0.85;
- greedy two-prompt quality suite;
- loader audit, MoE probe, and parameter fingerprints enabled;
- persistent vLLM source state established before the reference.

The GPU agent may adapt Slurm mechanics, node choice, cluster paths,
`MIN_FREE_GIB`, and log retention. Record every adaptation in the manifest's
`deviations` with its reason and whether it changes a comparison variable.
Never silently rerun with different settings.

## Preflight

From the repository root:

```bash
git pull --ff-only origin duy-branch
git status --short
git rev-parse HEAD
test -x pipeline/slurm/test_m3_paired_quality.sh
test -f pipeline/m3_quality_evidence.py
```

`git status --short` must be empty. Record the commit as `code_commit`.

```bash
REFERENCE_CKPT=/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4
CANDIDATE_CKPT=artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123
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
  pipeline/tests/test_m3_quality_evidence.py \
  pipeline/tests/test_m3_paired_quality_runner.py \
  pipeline/tests/test_patch_vllm_m3_serve.py \
  pipeline/tests/test_serve_verify_m3_env.py \
  pipeline/tests/test_serve_verify_quality.py \
  pipeline/tests/test_reexport_minimax_m3_vllm.py

DRY_RUN=1 \
REFERENCE_CKPT="$REFERENCE_CKPT" \
CANDIDATE_CKPT="$CANDIDATE_CKPT" \
MODEL_ID="$MODEL_ID" \
bash pipeline/slurm/test_m3_paired_quality.sh
```

Do not reserve eight GPUs until the focused tests and dry run pass. If the live
vLLM fork requires a diagnostic-only compatibility change, commit it
separately, rerun the checks, and record the new code commit.

## Exact live command

On one clean eight-GPU node:

```bash
set -o pipefail
RUN_ID="$(date +%Y%m%d-%H%M%S)"
REFERENCE_CKPT="$REFERENCE_CKPT" \
CANDIDATE_CKPT="$CANDIDATE_CKPT" \
MODEL_ID="$MODEL_ID" \
RUN_ID="$RUN_ID" \
bash pipeline/slurm/test_m3_paired_quality.sh \
  2>&1 | tee "/mnt/nfs/hoangduy/logs/m3-paired-quality-$RUN_ID-operator.log"
```

The runner hashes checkpoint metadata, records exact commands and environment
provenance, establishes one persistent vLLM source state, runs cyankiwi first,
and stops before the candidate if the reference fails readiness or smoke
quality. Full logs remain under
`/mnt/nfs/hoangduy/logs/m3-paired-quality/<run_id>/`; compact evidence goes
to `results/m3-paired-quality/<run_id>/`.

## Runtime decisions

- Reference failure invalidates the baseline. Return its evidence and stop.
- Candidate infrastructure failure returns preserved evidence and a nonzero
  status.
- Missing loader, fingerprint, or MoE markers means
  `inconclusive_missing_evidence`; do not reinterpret absence as a broken
  model component.
- A retry that changes an invariant gets a new `RUN_ID`. Preserve and relate
  both attempts.
- Record operator mistakes, OOMs, timeouts, instrumentation errors, cleanup,
  retries, and all environment changes.
- Never overwrite a checkpoint or kill another user's process.

## Required compact evidence

```text
results/m3-paired-quality/<run_id>/
├── run_manifest.json
├── software_versions.txt
├── nvidia_smi.csv
├── nvidia_topology.txt
├── patch_status.txt
├── comparison.json
├── artifact_index.json
├── cyankiwi_reference/
│   ├── serve_report.json
│   ├── return_code.txt
│   ├── parameter_fingerprints.jsonl
│   ├── fingerprint_summaries.jsonl
│   ├── loader_audit.txt
│   ├── moe_probe.txt
│   └── notable_log_excerpt.txt
└── portable_awq_w4a8/
    └── the same files
```

If the reference stops the matrix, candidate files may be absent, but
`comparison.json` must say `invalid_reference`.

Raw prompt outputs remain in `serve_report.json`. Do not replace them with a
summary. `artifact_index.json` gives absolute path, byte size, and SHA-256 for
full logs outside Git. Add the operator log and its retention deadline if the
runner did not index it.

## Review, commit, and return

```bash
EVIDENCE_DIR="results/m3-paired-quality/$RUN_ID"
python -m json.tool "$EVIDENCE_DIR/run_manifest.json" >/dev/null
python -m json.tool "$EVIDENCE_DIR/comparison.json"
python -m json.tool "$EVIDENCE_DIR/artifact_index.json" >/dev/null
find "$EVIDENCE_DIR" -type f -size +5M -print
rg -n -i 'api[_-]?key|token=|authorization:|bearer ' "$EVIDENCE_DIR" || true
```

No compact file should exceed 5 MB. Inspect secret-scan matches. Before commit,
ensure the manifest includes start/end times, case order, exact code commit,
dirty state, scheduler IDs, deviations, retries, and full-log retention.

```bash
git add "results/m3-paired-quality/$RUN_ID"
git commit -m "data: add MiniMax-M3 paired quality evidence $RUN_ID"
git push origin duy-branch
```

Do not commit checkpoints, caches, site-packages, or full logs. Return the
result commit, code commit, `RUN_ID`, verdict, full-log location/retention,
runtime interpretation, and proposed next hypothesis.

## Completion checklist

- [ ] Commit, commands, order, timestamps, hashes, flags, and deviations are
      recorded.
- [ ] Host, scheduler, GPU/topology, driver/CUDA, Python, vLLM, Torch,
      compressed-tensors, FlashInfer, Safetensors, and Transformers are
      recorded.
- [ ] Reference raw outputs, quality decisions, loader audit, fingerprints, and
      MoE evidence are present.
- [ ] Candidate has the same evidence, or the reference correctly stopped it.
- [ ] Operational failures, retries, cleanup, and instrumentation errors are
      recorded.
- [ ] Every non-Git artifact has path, size, SHA-256, and retention.
- [ ] Compact files contain no credential and are each at most 5 MB.
- [ ] The result commit is pushed and identifies the executed code commit.
