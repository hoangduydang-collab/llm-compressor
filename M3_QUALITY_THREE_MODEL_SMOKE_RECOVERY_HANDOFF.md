# MiniMax-M3 GPTQ Discriminator Executor Handoff

## Objective and role split

Determine whether the in-house GPTQ quality failure comes from quantization and
calibration or from export, loading, or runtime behavior. Quality is primary;
serving performance, AutoRound, production evaluation, and fresh
re-quantization remain deferred.

The primary agent owns experimental interpretation and code changes. The
capable-cluster executor owns runtime checks, safe operational diagnosis,
execution, artifact collection, and factual reporting. Fix transient environment
issues such as a missing package or stale Ray process when necessary, but do
not change checkpoints, prompts, sample identities, probe corpus, or
quantization settings without recording the deviation and stopping when it
breaks comparability.

Checkpoints:

- BF16: `/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3`
- GPTQ: `/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123`
- AWQ control: `/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4`

## Pull and CPU validation

```bash
cd /mnt/nfs/hoangduy/projects/llm-compressor
git pull
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD"
python -m pytest -q \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_eval_health.py \
  pipeline/tests/test_static_checkpoint.py \
  pipeline/tests/test_eval_distributional.py \
  pipeline/tests/test_m3_quality_eval_runner.py
```

Use this working environment throughout. Ray 2.56.0 may be installed if still
missing; record any installation and exact versions.

## Fresh preflight

This is a hard CPU-only serving ABI gate, not bookkeeping. Before task or probe
preparation, it compares each quantized checkpoint's packed/plain tensor
inventory with compressed-tensors decisions under both Transformers and vLLM
MiniMax-M3 names. Plain quantizable vLLM modules not ignored, ignored packed
modules, packed/plain collisions, missing scales, malformed regexes, or invalid
quantization targets abort preflight.

First preserve the known direct-checkpoint failure as evidence, then build an
immutable portable serving view. The direct check must fail with exactly the
router/shared-expert namespace misses already diagnosed; a different failure is
new evidence and must stop the run.

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)-m3-gptq-repaired"
RUN_ROOT="results/m3-quality/$RUN_ID"
SOURCE_GPTQ=/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123
REPAIRED_GPTQ="$RUN_ROOT/checkpoints/inhouse-gptq-portable"
SOURCE_MATRIX=pipeline/configs/minimax_m3_quality_matrix.yaml
MATRIX="$RUN_ROOT/repaired_matrix.yaml"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/static_direct"

if python -m pipeline.m3_serve_abi --checkpoint "$SOURCE_GPTQ" \
  --out "$RUN_ROOT/static_direct/inhouse_gptq.json"; then
  echo "ERROR: direct GPTQ unexpectedly passed; stop and report" >&2
  exit 1
fi
python -m pipeline.m3_routed_diagnostics prepare-overlay \
  --source "$SOURCE_GPTQ" --destination "$REPAIRED_GPTQ" \
  --add-vllm-shared-expert-ignore --add-vllm-router-ignore
python - "$SOURCE_MATRIX" "$MATRIX" "$REPAIRED_GPTQ" <<'PY'
import sys, yaml
from pathlib import Path
source, destination, repaired = map(Path, sys.argv[1:])
data = yaml.safe_load(source.read_text())
model = next(m for m in data["models"] if m["label"] == "inhouse_gptq")
model["path"] = str(repaired.resolve())
destination.write_text(yaml.safe_dump(data, sort_keys=False))
PY

python -m pipeline.m3_quality_preflight \
  --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/preflight.log"
```

The overlay must contain `overlay_provenance.json`. Require distinct source and
overlay config hashes, identical source and overlay index hashes, both vLLM
aliases in `added_ignore_rules`, and `tensor_payload_unchanged: true`. This is a
metadata/export repair, not a new quantization. The full preflight now inspects
BF16, repaired GPTQ, and cyankiwi AWQ before raising once, so return all three
`preflight/serving_abi/*.json` files even when any model fails. Do not proceed to
GPU unless all three are valid.

Confirm every MMLU-Pro leaf in `preflight/resolved_tasks.json` has its filtered
subject size rather than 12,032. Validate both manifests before allocating GPUs:

```bash
python - "$RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]) / "preflight"
resolved = json.load(open(root / "resolved_tasks.json"))
sizes, aliases = resolved["leaf_sizes"], resolved["aliases"]
reverse = {installed: canonical for canonical, installed in aliases.items()}
for profile in ("smoke", "production"):
    tasks = json.load(open(root / f"{profile}_sample_manifest.json"))["tasks"]
    for task, leaves in tasks.items():
        canonical = reverse[task]
        for leaf, indices in leaves.items():
            assert not indices or max(indices) < sizes[canonical][leaf], (
                profile, task, leaf, sizes[canonical][leaf], max(indices)
            )
print("sample bounds valid")
PY
```

Stop on failure and return the preflight tree and traceback. Never edit a
generated manifest manually.

Require `preflight/serving_abi/<model>.json` for every active model. Every
quantized report must say `"valid": true`. Never bypass a failure to reach GPU
execution. Source-only `mlp.shared_experts` or `mlp.gate` matches do not satisfy
vLLM's `block_sparse_moe` contract; return the report and config/index instead.
The same gate can be run independently without task packages or GPUs:

```bash
python -m pipeline.m3_serve_abi --checkpoint /path/to/checkpoint \
  --out /tmp/serving_abi.json
```


## Canary after static validation, before full re-quantization

Only after the static checker passes may a new quantization recipe advance to a
representative-layer canary. Quantize layer 3 (the first MoE layer) and one
mid/late layer such as layer 35 with the intended AWQ/GPTQ and activation
scheme. Do not start a 7-to-15-hour full model run first.

Return matched calibration/AWQ mappings, packed weights and scale metadata,
reference-dequant versus exported-dequant error, BF16-versus-dequant cosine,
normalized MSE and SQNR, fixed-input layer-output error, the incremental error
from dynamic FP8 activations, and the canary ABI report. A valid canary permits
full quantization but does not replace final end-to-end quality evaluation.

## Parallel smoke execution

Use at least four 8xH100 nodes. `sbatch` is unavailable; use only `srun --exclusive`. Repaired GPTQ, AWQ, and the Ray placement diagnostic run concurrently. Pass `--model "$REPAIRED_GPTQ"` to the GPTQ arm; never serve the direct source checkpoint in this run.
The quantized arms now run a 2,048-token teacher-forced probe before lm-eval,
so benchmark failure cannot erase distribution evidence.

The Cursor tool must not own any `srun` process. Run the real wrapper from a
login/control shell outside every Slurm allocation; `[[ -z
"${SLURM_JOB_ID:-}" ]]` must succeed. Nested `srun --exclusive` is only
step-exclusive and may colocate jobs, so both wrapper and controller now refuse
an inherited `SLURM_JOB_ID`. Top-level `srun --exclusive` gives each arm a
whole-node allocation.

Start the unchanged four-arm controller through the tested detached tmux wrapper:

```bash
export RUN_ID RUN_ROOT MATRIX REPAIRED_GPTQ
DRY_RUN=1 SESSION_NAME="m3-quality-$RUN_ID" \
  bash pipeline/slurm/start_m3_quality_smoke_tmux.sh
SESSION_NAME="m3-quality-$RUN_ID" \
  bash pipeline/slurm/start_m3_quality_smoke_tmux.sh
```

The launcher returns only after `tmux has-session` verifies the detached
session. At that point the Cursor tool may exit without terminating `srun`.
Record the printed session, controller log, and run root. Monitor independently:

```bash
tmux has-session -t "=m3-quality-$RUN_ID"
tmux capture-pane -pt "=m3-quality-$RUN_ID" -S -80
tail -f "$RUN_ROOT/logs/controller.log"
squeue -u "$USER" -o '%.18i %.28j %.8T %.10M %.10l %.6D %R'
cat "$RUN_ROOT/controller.rc"
```

Confirm concurrent running jobs have disjoint `NODELIST` values: GPTQ and AWQ
one node each, and Ray two other nodes. BF16 later receives two exclusive nodes
after Ray finishes. Do not accept colocated running arms.

Attaching is optional: `tmux attach-session -t "=m3-quality-$RUN_ID"`. Detach
with `Ctrl-b d`; do not kill the tmux server. The controller starts repaired
GPTQ, cyankiwi AWQ, and the Ray placement diagnostic concurrently, waits for
Ray to release its nodes, then runs the ten-minute BF16 diagnostic. It records
all four return codes in `executor_return_codes.txt` and its own final status
atomically in `controller.rc`.

If the tmux session disappears, first check `controller.rc`, the durable
controller/arm logs, and Slurm state. A completed session normally disappears
after writing `controller.rc`. If the file is absent while Slurm steps remain,
do not relaunch or cancel them; report the scheduler state. Never reuse a stale
`RUN_ROOT` or session name.

If resource policy delays BF16 while both quantized arms occupy nodes, the
controller waits; never cancel a quantized arm. Do not extend the BF16 timeout
without primary-agent approval.

## Distribution comparisons

Prefer BF16 when its probe completes. Otherwise compare GPTQ against AWQ so the
primary agent still receives a same-corpus discriminator; do not describe AWQ
as numerically equivalent to BF16.

```bash
GPTQ="$RUN_ROOT/models/inhouse_gptq/shards/smoke/distributional_probe.jsonl"
AWQ="$RUN_ROOT/models/cyankiwi_awq/shards/smoke/distributional_probe.jsonl"
BF16="$RUN_ROOT/models/bf16/shards/smoke/distributional_probe.jsonl"
if [[ -s "$GPTQ" && -s "$AWQ" ]]; then
  python -m pipeline.m3_distributional_probe compare \
    --reference "$AWQ" --candidate "$GPTQ" \
    --out "$RUN_ROOT/gptq_vs_awq_distributional.json"
fi
if [[ -s "$GPTQ" && -s "$BF16" ]]; then
  python -m pipeline.m3_distributional_probe compare \
    --reference "$BF16" --candidate "$GPTQ" \
    --out "$RUN_ROOT/gptq_vs_bf16_distributional.json"
fi
if [[ -s "$AWQ" && -s "$BF16" ]]; then
  python -m pipeline.m3_distributional_probe compare \
    --reference "$BF16" --candidate "$AWQ" \
    --out "$RUN_ROOT/awq_vs_bf16_distributional.json"
fi
```

Report globally, by length bucket, and by position quartile:
`argmax_flip_ratio`, top-5/top-20 Jaccard, observed-token NLL and perplexity
ratio, mean/median/p95/p99 absolute log-probability error, reference-argmax
candidate-rank distribution, and missing-top-k rate. Do not label a top-k-only
calculation as full-vocabulary KL divergence.

## Stop/go decisions

Do not launch production, layer-boundary reruns, or re-quantization in this
handoff.

- Large GPTQ teacher-forced drift with coherent AWQ: return evidence for the
  next offline-dequant-versus-loaded localization.
- Close teacher-forced GPTQ but garbage autoregressive output: return rendered
  prompts, token IDs, health, and vLLM logs; focus shifts to generation/KV-cache.
- Ready 16-bundle Ray group but BF16 vLLM timeout: classify as vLLM-Ray
  integration evidence and retain both nodes' Ray logs.
- Benchmark failure after a successful probe: preserve it; diagnose the task
  failure but do not immediately rerun or change the experiment.

## Required return package

Commit and push the complete run root, excluding checkpoints. Include:

- preflight log, resolved tasks/config, exact manifests and hashes, probe corpus,
  run manifest, and checkpoint diagnostics;
- every arm's stdout/stderr, manifest, return code, completion marker, smoke
  evidence, aggregates, samples, generation health, rendered prompts/token IDs,
  distribution JSONL, and summary;
- all distribution comparison JSON;
- placement-group JSON, before/after placement listings and `ray status`,
  topology gate/rank files, both rank Ray-log archives, and cleanup output;
- Python, CUDA, PyTorch, vLLM, Ray, lm-eval, transformers, and llm-compressor
  versions plus node/GPU identities;
- `executor_return_codes.txt` and `EXECUTOR_NOTES.md` with exact commands,
  timestamps, allocation, deviations, retries, observations, hypotheses, and
  every missing artifact plus reason.

Keep observations separate from hypotheses. Explicitly answer:

1. Did all completed probes use the same corpus hash and paired token count?
2. Does GPTQ drift exist on short inputs or grow by position and length?
3. Is the reference argmax absent from GPTQ top-20, rank-displaced, or retained?
4. Are GPTQ generations now health-applicable, and what are cap, loop,
   repetition, and answer-extraction rates versus AWQ?
5. Did the 16-bundle Ray group become ready? If so, where did BF16 vLLM stop?
6. Is the returned evidence sufficient for full primary-agent analysis without
   another cluster round trip?
