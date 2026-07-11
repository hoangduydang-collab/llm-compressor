# MiniMax-M3 canonical chat quality matrix runbook

## Objective and ownership

Run four eager-mode canonical-chat arms concurrently: cyankiwi and the portable
W4A8 checkpoint through offline vLLM and HTTP chat-completions. Return compact,
auditable evidence through Git for analysis on the non-GPU cluster.

The primary agent owns experiment design, implementation, and interpretation.
The GPU agent owns preflight, four-node scheduling, runtime-only adaptations,
execution, evidence inspection, and the result commit. Do not repair a model,
change prompts, enable diagnostics, re-quantize, or investigate CUDA graphs in
this matrix.

## Why this supersedes the raw-completion runs

Run `20260711-125747-reference-sequential` proved batching was not the cause of
the earlier reference output. The arithmetic case emitted only token `200020`,
MiniMax-M3's EOS token, while the France case continued a raw-text loop. Those
bare prompts omitted the model's required chat roles and assistant generation
prefix, so they are not a valid quality baseline.

The handed-off offline verifier now applies the official tokenizer chat template
with `add_generation_prompt=True` and `thinking_mode="disabled"`. HTTP uses the
OpenAI-compatible chat endpoint with the same user messages and thinking mode.

## Matrix contract

| Arm | Checkpoint | Interface |
| --- | --- | --- |
| `reference_offline_chat` | cyankiwi W4A16 | Offline canonical chat template |
| `candidate_offline_chat` | Portable W4A8 | Offline canonical chat template |
| `reference_http_chat` | cyankiwi W4A16 | `/v1/chat/completions` |
| `candidate_http_chat` | Portable W4A8 | `/v1/chat/completions` |

All arms must retain:

- one clean eight-GPU node per arm, running concurrently;
- the same pushed code commit and Python/vLLM environment;
- TP=8, expert parallelism, eager mode, block size 128, FP8 KV cache;
- `max_model_len=2048`, GPU utilization 0.85, custom all-reduce disabled;
- shared-expert auxiliary stream disabled;
- two fixed user messages, 64 output tokens, temperature 0;
- `thinking_mode=disabled`;
- `M3_LOAD_AUDIT=0`, `M3_MOE_PROBE=0`, `M3_PARAM_FINGERPRINT=0`.

The GPU agent may adapt partition, node constraints, time limit, NFS roots,
ports, and other scheduler mechanics. Record changes and retry history in each
`arm_manifest.json`; never silently change a quality variable. A retry gets a
new matrix ID or an explicitly related attempt directory.

## Preflight

From the repository root:

```bash
git pull --ff-only origin duy-branch
git status --short
git rev-parse HEAD

test -x pipeline/slurm/test_m3_chat_quality_arm.sh
test -x pipeline/slurm/submit_m3_chat_quality_matrix.sh
test -f pipeline/m3_chat_quality.py

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
  pipeline/tests/test_serve_verify_quality.py \
  pipeline/tests/test_m3_quality_evidence.py \
  pipeline/tests/test_m3_chat_quality.py \
  pipeline/tests/test_m3_chat_quality_runner.py \
  pipeline/tests/test_submit_m3_chat_quality_matrix.py \
  pipeline/tests/test_serve_verify_m3_env.py \
  pipeline/tests/test_patch_vllm_m3_serve.py \
  pipeline/tests/test_reexport_minimax_m3_vllm.py

MATRIX_ID=preflight-canonical-chat DRY_RUN=1 \
  bash pipeline/slurm/submit_m3_chat_quality_matrix.sh
```

The worktree must be clean before submission. Dry-run output must contain four
`sbatch` commands with the four fixed arm names.

## Submit four nodes

Use one shared matrix ID. Scheduler settings are intentionally overrideable:

```bash
MATRIX_ID="$(date +%Y%m%d-%H%M%S)-canonical-chat"
MATRIX_ID="$MATRIX_ID" \
TIME_LIMIT=02:00:00 \
bash pipeline/slurm/submit_m3_chat_quality_matrix.sh \
  | tee "/mnt/nfs/hoangduy/logs/m3-chat-quality-submit-$MATRIX_ID.txt"
```

If the default partition is unsuitable, add a recorded override such as
`SBATCH_ARGS="--partition=h100"`. Do not put four arms on one node. Each job runs only its named arm and bundles evidence even after a
serve failure.

Monitor all returned job IDs. Do not cancel successful siblings when one arm
fails. Preserve scheduler failure, OOM, timeout, startup error, malformed HTTP
response, and cleanup evidence as distinct outcomes.

## Aggregate

After all four jobs finish, rebuild each compact arm once so scheduler logs
are closed before their final hashes are recorded, then aggregate:

```bash
EVIDENCE_ROOT="results/m3-chat-quality/$MATRIX_ID"
FULL_ROOT="/mnt/nfs/hoangduy/logs/m3-chat-quality/$MATRIX_ID"
for arm in reference_offline_chat candidate_offline_chat reference_http_chat candidate_http_chat; do
  python -m pipeline.m3_chat_quality bundle-arm \
    --run-dir "$FULL_ROOT/$arm" \
    --evidence-dir "$EVIDENCE_ROOT/$arm"
done
python -m pipeline.m3_chat_quality aggregate --evidence-root "$EVIDENCE_ROOT"
python -m json.tool "$EVIDENCE_ROOT/comparison.json"
```

Expected compact structure:

```text
results/m3-chat-quality/<matrix_id>/
├── comparison.json
├── reference_offline_chat/
├── candidate_offline_chat/
├── reference_http_chat/
└── candidate_http_chat/
```

Each arm directory must contain `arm_manifest.json`, `arm_report.json`, raw HTTP
requests/responses when applicable, software/GPU provenance, return code,
notable traceback context, and `artifact_index.json` with full-log hashes.
Offline reports must preserve `rendered_prompt`, token IDs, finish reason, stop
reason, and raw text. HTTP reports must preserve the unmodified response body.

Interpret only `comparison.json` verdicts:

- `candidate_quality_pass`: both references and both candidates passed;
- `candidate_quality_fail`: both references passed and both candidates failed;
- `candidate_interface_disagreement`: references passed but candidate interfaces differ;
- `invalid_reference`: at least one canonical reference failed quality;
- `infrastructure_failure`: at least one arm did not reach a valid response;
- `inconclusive_missing_arms`: one or more arm bundles are absent.

Do not infer a checkpoint defect from a failed reference or infrastructure arm.
Do not proceed to CUDA-graph RCA until the primary agent analyzes this matrix.

## Review, commit, and return

```bash
find "$EVIDENCE_ROOT" -type f -size +5M -print
rg -n -i 'api[_-]?key|token=|authorization:|bearer ' "$EVIDENCE_ROOT" || true
git status --short
git add "$EVIDENCE_ROOT"
git commit -m "data: add MiniMax-M3 canonical chat matrix $MATRIX_ID"
git push origin duy-branch
```

Do not commit checkpoints, caches, site-packages, full serve logs, PID files, or
credentials. Return:

- result commit and executed code commit;
- matrix ID and four scheduler job IDs/nodes;
- aggregate verdict and one-line arm outcomes;
- every deviation and retry;
- full-log absolute paths, sizes, SHA-256 hashes, and retention deadlines;
- any missing evidence or runtime judgment made by the executor.
