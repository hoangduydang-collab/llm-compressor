# Execution packet: MiniMax-M3 paired generated-reasoning r4

- Protocol version: 1
- State: `READY_FOR_EXECUTOR`
- Packet revision: 2026-07-15-r4.5
- Planner owner: Codex planner
- Intended executor: cluster executor
- Required fix commit: `c5a0b755`
- Branch: `duy-branch`
- Retry authorization: one fresh r4.5 run after filter and sampling fixes
- Executor evidence: `M3_R4_4_AND_DDP_R5_FAILURE_EVIDENCE.md`

Revisions r1-r3 are historical and superseded by this r4 packet. Do not reuse
their GPQA/MMLU-Pro/GSM8K scores: r4 changes the reasoning harness to stock
generated-answer lm-eval tasks and runs three paired sampling seeds.

The first r4 attempt at
`results/m3-quality/20260715T061300Z-m3-paired-reasoning-r4` stopped before GPU
allocation because preflight passed a Jinja template string where lm-eval
0.4.12 requires a callable chat renderer. Commit `3171dd8e` fixes that source
error and adds a regression test.

The r4.1 attempt at
`results/m3-quality/20260715T062000Z-m3-paired-reasoning-r4` also stopped before
GPU allocation. For generated-answer tasks, lm-eval exposes a `doc_to_choice`
method even though the task config intentionally leaves `doc_to_choice` unset;
calling that method raises by design. Commit `065ff302` makes preflight inspect
the processed GPQA document's `choices` field in that case and adds a regression
test. Preserve both stopped evidence roots, but do not reuse either incomplete
run root.

The r4.2 attempt at
`results/m3-quality/20260715T063400Z-m3-paired-reasoning-r4` stopped before GPU
allocation because preflight assumed every installed alias resolves to one task
object. In lm-eval 0.4.12, `mmlu_pro` is a group whose flat `tasks` map contains
14 subject leaves. Commit `15831064` handles the native group shape, audits the
shared output type, task version, and metric/filter contract across every leaf,
and uses a deterministic leaf only for representative-prompt stability. Preserve
all three stopped evidence roots, but do not reuse them; this packet authorizes
no additional r4.3 attempt.

The r4.3 attempt at
`results/m3-quality/20260715T064700Z-m3-paired-reasoning-r4` stopped before GPU
allocation when the contract audit incorrectly reported that GPQA did not expose
`exact_match,flexible-extract`. lm-eval's `TaskConfig` is both a dataclass and a
`dict` subclass, but stores its values in dataclass attributes; the preflight
adapter treated its empty underlying mapping as authoritative. Commit `849b2071`
reads the native attributes first, restoring the installed metric/filter,
dataset, and choice configuration fields. Preserve all four stopped evidence
roots, but do not reuse them; this packet authorizes exactly one fresh r4.4
attempt.

## Decision and fixed experiment

Compare the repaired in-house GPTQ checkpoint against cyankiwi AWQ under the
same paper-grade generated-reasoning harness. This is a paired directional
quantization comparison, not a BF16 recovery measurement or a reproduction of
MiniMax's private benchmark recipe.

| Canonical task | Installed lm-eval 0.4.12 task | Shots | Metric/filter | Questions |
| --- | --- | ---: | --- | ---: |
| GPQA Diamond | `gpqa_diamond_cot_zeroshot` v2.2 | 0 | `exact_match,flexible-extract` | 100 |
| MMLU-Pro | `mmlu_pro` | 5 | `exact_match,custom-extract` | 100 |
| GSM8K | `gsm8k_cot` | 8 | `exact_match,strict-match` | 100 |
| AIME 2025 | `aime25` | 0 | `exact_match,none` | all 30 |

Every question runs seeds `42`, `1234`, and `4158`. Generation uses explicit
thinking, `temperature=1.0`, `top_p=0.95`, and `max_gen_toks=16384`. The harness
passes `do_sample=true` through lm-eval so its generation normalizer cannot
silently replace positive-temperature sampling with greedy decoding; lm-eval
removes that compatibility key before constructing vLLM `SamplingParams`.

Production is exactly four independent top-level allocations:

1. AWQ: GPQA.
2. GPTQ: GPQA.
3. AWQ: MMLU-Pro, then GSM8K, then AIME in one loaded-model process.
4. GPTQ: MMLU-Pro, then GSM8K, then AIME in one loaded-model process.

Each allocation is one node, eight H100s, and has a `24:00:00` limit. The
cluster supports `srun`, not `sbatch`. No distributional probe or IFEval is in
scope.

## 1. Pull and validate the workspace

Run on the login/controller node, outside any Slurm allocation:

```bash
set -euo pipefail
cd /mnt/nfs/hoangduy/projects/llm-compressor
git fetch origin
git checkout duy-branch
git pull --ff-only origin duy-branch
git merge-base --is-ancestor 849b2071 HEAD
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
test -z "${SLURM_JOB_ID:-}"
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
git ls-files --others --exclude-standard \
  | awk '!/^(results|artifacts)\//' \
  | tee /tmp/m3-r4-workspace-blockers
test ! -s /tmp/m3-r4-workspace-blockers
python - <<'PY'
import importlib.metadata
assert importlib.metadata.version("lm-eval") == "0.4.12"
PY
```

Stop if a check fails. Existing untracked data under `results/` or `artifacts/`
is record-and-proceed evidence; do not delete it.

## 2. Create a fresh run and execute CPU preflight

```bash
set -euo pipefail
MATRIX=pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-paired-reasoning-r4"
RUN_ROOT="results/m3-quality/$RUN_ID"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
date -u +%FT%TZ >"$RUN_ROOT/controller_start_utc.txt"

python -m pipeline.m3_quality_preflight \
  --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/preflight.log"
test "${PIPESTATUS[0]}" -eq 0

python - "$RUN_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1])
contract_path = root / "preflight/harness_contract.json"
contract = json.loads(contract_path.read_text())
run = json.loads((root / "run_manifest.json").read_text())
assert contract["valid"] is True
assert contract["lm_eval_version"] == "0.4.12"
assert contract["generation"]["generation_seeds"] == [42, 1234, 4158]
assert contract["tasks"]["gpqa_diamond"]["installed_name"] == "gpqa_diamond_cot_zeroshot"
assert contract["tasks"]["gpqa_diamond"]["task_version"] == "2.2"
assert all(task["output_type"] == "generate_until" for task in contract["tasks"].values())
assert run["harness_contract_sha256"] == hashlib.sha256(contract_path.read_bytes()).hexdigest()
assert run["expected_question_counts"] == {
    "gpqa_diamond_cot_zeroshot": 100,
    "mmlu_pro": 100,
    "gsm8k_cot": 100,
    "aime25": 30,
}
print(json.dumps({"valid": True, "contract": contract_path.as_posix()}, indent=2))
PY
```

Preflight must finish before any GPU allocation. It validates checkpoint serving
ABI, exact task aliases/output types/metrics/shots, GPQA v2.2, prompt stability,
choice mapping, tokenizer/chat-template equality, sample manifests, and the
harness hash. Do not bypass or edit a failed preflight artifact.

## 3. Run the fresh r4 smoke gate

Do not reuse an r1-r3 smoke gate because the task and generation contracts
changed.

```bash
set -euo pipefail
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/smoke_controller.log"
test "${PIPESTATUS[0]}" -eq 0
python - "$RUN_ROOT/smoke_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
assert gate["ready_for_production"] is True, gate
assert all("probe_budget" not in item["checks"] for item in gate["models"].values())
PY
```

The smoke profile uses two one-node arms and no runtime/distributional probe.
Stop on failure and return evidence; no retry is authorized.

## 4. Verify the four production commands

```bash
set -euo pipefail
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  --smoke-gate "$RUN_ROOT/smoke_gate.json" --dry-run \
  | tee "$RUN_ROOT/srun_dry_run.log"
test "$(grep -c '^srun ' "$RUN_ROOT/srun_dry_run.log")" -eq 4
test "$(grep -c -- '--nodes=1' "$RUN_ROOT/srun_dry_run.log")" -eq 4
test "$(grep -c -- '--gpus-per-node=8' "$RUN_ROOT/srun_dry_run.log")" -eq 4
test "$(grep -c -- '--time 24:00:00' "$RUN_ROOT/srun_dry_run.log")" -eq 4
test "$(grep -c -- '--run-probe 0' "$RUN_ROOT/srun_dry_run.log")" -eq 4
grep -q 'total_nodes=4' "$RUN_ROOT/srun_dry_run.log"
! grep -q sbatch "$RUN_ROOT/srun_dry_run.log"
```

## 5. Launch persistently

The controller must remain outside another allocation. It starts all four
top-level `srun` commands concurrently.

```bash
set -euo pipefail
SESSION="m3-quality-$RUN_ID"
tmux new-session -d -s "$SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD/src:$PWD' && bash pipeline/slurm/run_m3_quality_eval_srun.sh --profile production --matrix '$MATRIX' --run-root '$RUN_ROOT' --smoke-gate '$RUN_ROOT/smoke_gate.json' >'$RUN_ROOT/controller.log' 2>&1; printf '%s\n' \$? >'$RUN_ROOT/controller.rc'"
printf '%s\n' "$SESSION" >"$RUN_ROOT/tmux_session.txt"
```

Model loading is amortized on the three-task arm. Seed and task checkpoints are
written atomically, so evidence survives interruption, but interruption does
not authorize a retry.

## 6. Monitor without taking ownership

```bash
set -euo pipefail
tmux capture-pane -pt "$SESSION" -S -200
squeue -u "$USER" -o '%.18i %.9T %.20N %.10M %.10l'
tail -n 100 "$RUN_ROOT/controller.log"
find "$RUN_ROOT/models" -name seed_progress.json -print -exec cat {} \;
```

Wait for `controller.rc`. Do not cancel healthy sibling arms if one arm fails.

## 7. Capture scheduler evidence and aggregate

```bash
set -euo pipefail
python - "$RUN_ROOT" <<'PY' >"$RUN_ROOT/slurm_job_ids.txt"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ids = sorted({
    json.loads(path.read_text()).get("slurm_job_id")
    for path in root.glob("models/*/shards/*/arm_manifest.json")
})
for job_id in ids:
    if job_id:
        print(job_id)
PY
while read -r job_id; do
  sacct -j "$job_id" \
    --format=JobIDRaw,JobName%40,State,ExitCode,NodeList,AllocTRES,Submit,Start,End,Elapsed,Timelimit
  scontrol show job "$job_id" -dd
done <"$RUN_ROOT/slurm_job_ids.txt" \
  | tee "$RUN_ROOT/slurm_accounting_final.txt"

set +e
python -m pipeline.m3_quality_eval aggregate \
  --matrix "$MATRIX" --root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/aggregate.log"
AGGREGATE_RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$AGGREGATE_RC" >"$RUN_ROOT/aggregate.return_code.txt"
```

Aggregation is partial-safe for infrastructure failures, but a successful r4
merge is fail-closed: every expected question must contain all three seeds,
attempt IDs must be unique, harness hashes must match, and both AWQ and GPTQ
must provide all generation-health counters. Statistical confidence intervals
resample questions while keeping their three seed outcomes together.

## 8. Return evidence

Return the full small evidence tree: preflight reports, harness contract,
sample manifests, smoke report/gate, launch plan, dry-run, controller log/rc,
four arm manifests and return codes, seed progress, aggregates, sample JSONL,
generation-health summaries, comparisons, scheduler accounting, matrix,
gates, report, commands, versions, timings, deviations, and missing artifacts.
For retained large files record absolute path, byte size, and SHA-256.

Commit and push evidence on `duy-branch`, change this packet state to
`RETURNED_FOR_ANALYSIS`, and stop. Do not retry, interpret quality, adopt a
model, begin performance work, or publish results.

## Planner analysis and r4.5 authorization

The r4.4 jobs ended but produced no quality scores. Both models loaded and
generated both smoke requests, then checkpointing failed because lm-eval logs
one sample record for every configured filter pipeline. GPQA's
`strict-match` and `flexible-extract` records described the same document and
generation seed, so they correctly shared one `attempt_uid` but contained
different extracted answers and metrics. Adding model or filter identity to
the UID would hide this conflict and break paired comparison semantics.

Commit `c5a0b755` instead selects the filter named by each task's configured
metric before normalization and deduplication. It fails closed if lm-eval logs
filter names but omits the configured filter. The same commit preserves
`do_sample=true` in the lm-eval generation override. r4.4 logs showed lm-eval
otherwise receiving `do_sample=false` from task defaults and forcing
temperature to zero, so those two smoke generations are not reusable.

This section supersedes r4.4 authorization. Run exactly one fresh r4.5 smoke
and, only if its gate passes, the four-arm production evaluation defined above.
All models, task aliases, exact sample manifests, question caps, shots, three
seeds, generation parameters, topology, and time limits remain unchanged.

```bash
set -euo pipefail
cd /mnt/nfs/hoangduy/projects/llm-compressor
git fetch origin
git checkout duy-branch
git pull --ff-only origin duy-branch
git merge-base --is-ancestor c5a0b755 HEAD

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
test -z "${SLURM_JOB_ID:-}"
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"

python -m pytest -q \
  pipeline/tests/test_static_checkpoint.py \
  pipeline/tests/test_lmeval_runner.py \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_m3_quality_eval_runner.py \
  pipeline/tests/test_m3_quality_smoke_tmux.py \
  pipeline/tests/test_m3_quality_evidence.py

MATRIX=pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-paired-reasoning-r4"
RUN_ROOT="results/m3-quality/$RUN_ID"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

python -m pipeline.m3_quality_preflight \
  --matrix "$MATRIX" --run-root "$RUN_ROOT"

bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$MATRIX" --run-root "$RUN_ROOT"

python - "$RUN_ROOT/smoke_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
assert gate["ready_for_production"] is True, gate
PY

SESSION="m3-quality-$RUN_ID"
tmux new-session -d -s "$SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD/src:$PWD' && bash pipeline/slurm/run_m3_quality_eval_srun.sh --profile production --matrix '$MATRIX' --run-root '$RUN_ROOT' --smoke-gate '$RUN_ROOT/smoke_gate.json' >'$RUN_ROOT/controller.log' 2>&1; printf '%s\n' \$? >'$RUN_ROOT/controller.rc'"
printf 'RUN_ID=%s\nRUN_ROOT=%s\nSESSION=%s\n' "$RUN_ID" "$RUN_ROOT" "$SESSION"
```

Wait for `controller.rc`, then follow Sections 6-8 above verbatim to collect,
aggregate, commit, and push the evidence. On any preflight, smoke, or production
failure, preserve the first exception and partial artifacts, return the packet
for analysis, and stop. No r4.5 retry or runtime patch is authorized.
