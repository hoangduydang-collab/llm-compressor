# Execution packet: MiniMax-M3 task-isolated paired quality rerun

- Protocol version: 1
- State: `READY_FOR_EXECUTOR`
- Packet revision: 2026-07-14-r1
- Planner owner: Codex planner
- Intended executor: cluster executor
- Base Git commit: `1e2da818a08df09bf1ff0268702bda34bf89ee6e`
- Decision question: Does repaired in-house GPTQ preserve enough quality versus
  cyankiwi AWQ to justify later performance evaluation?

## Objective and boundary

Run five 100-sample paired tasks plus one 8,192-token probe for each model as
twelve independent one-node arms. Slurm may run at most six arms concurrently;
each arm has an independent eight-hour limit.

This packet does not authorize retries, parameter changes, speed tests, model
adoption, publication, or any downstream run. Return evidence after this run.

## Planner correction to historical evidence

Do not edit the historical executor report. Its statement that GPTQ broad timed
out during startup is inaccurate: raw stdout records model loading in about 296
seconds, and stderr reaches 96/100 MMLU-Pro prompts before timeout. This is a
planner addendum only; it is not a quality verdict.

## Exact environment and revision

```bash
cd /mnt/nfs/hoangduy/projects/llm-compressor
git fetch origin
git checkout duy-branch
git pull --ff-only origin duy-branch
git merge-base --is-ancestor 1e2da818a08df09bf1ff0268702bda34bf89ee6e HEAD
git rev-parse HEAD
git status --short
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python --version
python -m pip show torch transformers vllm lm-eval
```

Stop if the ancestor check fails, the worktree is not clean, or the named
environment/packages are unavailable.

## Inputs and fresh run root

```bash
MATRIX=pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml
SMOKE_ROOT=results/m3-quality/20260714T064000Z-m3-paired-gptq-awq-quick
SMOKE_GATE="$SMOKE_ROOT/smoke_gate.json"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-paired-gptq-awq-task-isolated"
RUN_ROOT="results/m3-quality/$RUN_ID"
test -f "$SMOKE_GATE"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
printf 'run_id=%s\nrun_root=%s\n' "$RUN_ID" "$RUN_ROOT" | tee "$RUN_ROOT/controller.log"
```

## Preflight and smoke-gate reuse check

```bash
python -m pipeline.m3_quality_preflight \
  --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/preflight.log"
test "${PIPESTATUS[0]}" -eq 0

python - "$SMOKE_ROOT/run_manifest.json" "$RUN_ROOT/run_manifest.json" \
  pipeline/configs/minimax_m3_paired_gptq_awq_quick.yaml "$MATRIX" \
  "$SMOKE_GATE" <<'PY' | tee "$RUN_ROOT/smoke_reuse_check.json"
import json, sys, yaml
from pathlib import Path

old_manifest = json.load(open(sys.argv[1], encoding="utf-8"))
new_manifest = json.load(open(sys.argv[2], encoding="utf-8"))
old_matrix = yaml.safe_load(Path(sys.argv[3]).read_text(encoding="utf-8"))
new_matrix = yaml.safe_load(Path(sys.argv[4]).read_text(encoding="utf-8"))
gate = json.load(open(sys.argv[5], encoding="utf-8"))
fields = (
    "sample_manifest_sha256", "eval_config_sha256", "tokenizer_sha256",
    "chat_template_sha256",
)
checks = {field: old_manifest.get(field) == new_manifest.get(field) for field in fields}
def model_contract(raw):
    return [
        {
            key: model.get(key, 1 if key == "pipeline_parallel_size" else None)
            for key in (
                "label", "path", "kind", "nodes", "tensor_parallel_size",
                "pipeline_parallel_size", "distributed_executor_backend",
            )
        }
        for model in raw["models"]
    ]
checks["model_and_serving_contract"] = model_contract(old_matrix) == model_contract(new_matrix)
checks["prior_gate_ready"] = gate.get("ready_for_production") is True
result = {"reusable": all(checks.values()), "checks": checks}
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["reusable"] else 1)
PY
test "${PIPESTATUS[0]}" -eq 0
```

Stop and return if preflight or any reuse check fails. The matrix hash is
expected to differ because the shard topology changed.

## Dry run and launch

```bash
pipeline/slurm/submit_m3_quality_eval_array.sh \
  --matrix "$MATRIX" --run-root "$RUN_ROOT" --smoke-gate "$SMOKE_GATE" \
  --dry-run | tee "$RUN_ROOT/array_dry_run.log"
test "$(grep -c '^index=' "$RUN_ROOT/array_dry_run.log")" -eq 12
grep -F -- '--array=0-11%6' "$RUN_ROOT/array_dry_run.log"
grep -F -- '--time=08:00:00' "$RUN_ROOT/array_dry_run.log"
rm "$RUN_ROOT/production_launch_plan.json"

pipeline/slurm/submit_m3_quality_eval_array.sh \
  --matrix "$MATRIX" --run-root "$RUN_ROOT" --smoke-gate "$SMOKE_GATE" \
  2>&1 | tee "$RUN_ROOT/submission.log"
test "${PIPESTATUS[0]}" -eq 0
ARRAY_JOB_ID="$(cat "$RUN_ROOT/array_job_id.txt")"
printf 'array_job_id=%s\n' "$ARRAY_JOB_ID" | tee -a "$RUN_ROOT/controller.log"
```

Expected arms are `cyankiwi_awq` and `inhouse_gptq` crossed with
`gpqa_diamond`, `ifeval`, `aime_2025`, `mmlu_pro`, `gsm8k`, and
`distributional_probe`. Do not cancel healthy array tasks when another fails.

## Monitor and capture scheduler evidence

```bash
squeue -j "$ARRAY_JOB_ID" -o '%.18i %.9T %.20N %.10M %.10l'
sacct -j "$ARRAY_JOB_ID" --array \
  --format=JobIDRaw,JobName%40,State,ExitCode,NodeList,AllocTRES,Submit,Start,End,Elapsed,Timelimit \
  | tee "$RUN_ROOT/sacct-live.txt"
```

Wait until every array element is terminal. Queueing is allowed. Do not retry a
failed element or extend its time limit.

## Aggregate after all array elements are terminal

```bash
sacct -j "$ARRAY_JOB_ID" --array \
  --format=JobIDRaw,JobName%40,State,ExitCode,NodeList,AllocTRES,Submit,Start,End,Elapsed,Timelimit \
  | tee "$RUN_ROOT/sacct-final.txt"
scontrol show job "$ARRAY_JOB_ID" -dd | tee "$RUN_ROOT/scontrol-final.txt"

set +e
python -m pipeline.m3_quality_eval aggregate \
  --matrix "$MATRIX" --root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/aggregate.log"
AGGREGATE_RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$AGGREGATE_RC" >"$RUN_ROOT/aggregate.return_code.txt"
python -m json.tool "$RUN_ROOT/matrix.json"
python -m json.tool "$RUN_ROOT/gates.json"
```

A nonzero aggregate result is evidence to return, not permission to retry.

## Required return packet

Commit the following small evidence, preserving raw files:

- `run_manifest.json`, all `preflight/` manifests/configs, `preflight.log`, and
  `smoke_reuse_check.json`;
- `production_launch_plan.json`, `array_dry_run.log`, `submission_command.txt`,
  `submission.log`, and `array_job_id.txt`;
- every scheduler stdout/stderr log and each arm's `arm_manifest.json`,
  `aggregate.json`, `return_code.txt`, and `arm_complete.json`;
- sample/generation-health/probe artifacts, `sacct-final.txt`,
  `scontrol-final.txt`, `matrix.json`, `gates.json`, `report.md`,
  `aggregate.log`, and `aggregate.return_code.txt`.

The factual return report must list every array index with arm, scheduler state,
node, elapsed time, exit code, gate result, and artifact path; exact commands;
environment versions; deviations; missing artifacts; first failure and last
successful stage. For any large file not committed, record absolute path, byte
size, and SHA-256.

Commit and push the evidence on `duy-branch`, set the packet state to
`RETURNED_FOR_ANALYSIS`, verify a clean synchronized worktree, and stop. Only the
planner decides the quality verdict and next action.
