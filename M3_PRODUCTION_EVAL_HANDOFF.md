# MiniMax-M3 r4.7 Production + BF16 Companion + Raw Replay Handoff

- Protocol version: 1
- State: `READY_FOR_EXECUTOR`
- Packet revision: `2026-07-15-r4.7`
- Required branch: `duy-branch`
- Required ancestor: `e7834694`
- Scheduler: top-level `srun` only; `sbatch` is prohibited
- Maximum allocation authorized by this packet: 8 nodes concurrently

## Planner decision and fixed contract

The returned r4.5 smoke is promoted under its committed policy. Both AWQ and
GPTQ completed all tasks and seeds; the one processed empty GPTQ MMLU-Pro row
is a score-zero quality observation and health warning, not a setup failure.
Do not delete, replace, reinterpret, or retry it. Do not add `min_tokens`, retry
logic, or any harness change.

The paired and BF16 runs retain the committed r4 task, sample, seed, prompt,
chat-template, filter, metric, generation, checkpoint, and topology contracts.
Smoke uses the setup-only 256-token cap; production uses 16,384 tokens. The
replay is a diagnostic sidecar only: it loads the GPTQ model once and runs the
same saved attempt at exactly 256 and 16,384 tokens. The executor records raw
and postprocessed stages but does not score, repair, or interpret the replay.

The packet owns these allocations:

- Paired production: four independent one-node arms, maximum 4 nodes.
- Exact GPTQ replay: one one-node allocation, maximum 1 node.
- BF16 smoke: one two-node TP8xPP2/Ray arm, maximum 2 nodes.
- BF16 production: two concurrent two-node TP8xPP2/Ray arms, maximum 4 nodes.

Wave A is exactly `4 + 1 + 2 = 7` nodes. Wave B starts only after both the
replay and BF16 smoke have ended. It may overlap four-node paired production,
for exactly `4 + 4 = 8` nodes. The replay allocation must have ended before
Wave B, so this packet can never reach nine nodes.

## 1. Pull and verify outside Slurm

Run from a login shell outside every Slurm allocation:

```bash
cd /mnt/nfs/hoangduy/projects/llm-compressor
git switch duy-branch
git pull --ff-only
git merge-base --is-ancestor e7834694 HEAD
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD"
[[ -z "${SLURM_JOB_ID:-}" ]]

python -m pytest -q \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_m3_quality_eval_runner.py \
  pipeline/tests/test_m3_quality_smoke_tmux.py \
  pipeline/tests/test_m3_quality_evidence.py \
  pipeline/tests/test_m3_empty_output_replay.py
```

Stop and return evidence if the ancestor, environment, outside-allocation
assertion, or CPU suite fails. Every command below must be launched from this
login shell. Never put an `srun` command inside an existing allocation.

## 2. Promote the existing paired smoke

```bash
PAIR_MATRIX=pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml
PAIR_ROOT=results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4

python -m pipeline.m3_quality_eval smoke-gate \
  --matrix "$PAIR_MATRIX" \
  --report "$PAIR_ROOT/smoke_report.json" \
  --out "$PAIR_ROOT/smoke_gate.json"

python - "$PAIR_ROOT/smoke_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
assert gate["ready_for_production"] is True, gate
gptq = gate["models"]["inhouse_gptq"]
assert gptq["empty_output_count"] == 1, gptq
assert gptq["max_smoke_empty_outputs"] == 1, gptq
assert gptq["warnings"], gptq
PY

bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$PAIR_MATRIX" --run-root "$PAIR_ROOT" \
  --smoke-gate "$PAIR_ROOT/smoke_gate.json" --dry-run
```

The dry run must show exactly four one-node, eight-GPU production arms and no
BF16 arm. Do not proceed if it differs.

## 3. Prepare and gate the independent BF16 root

Use a fresh root and never write BF16 artifacts into `PAIR_ROOT`:

```bash
BF16_MATRIX=pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml
BF16_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-bf16-reasoning-r4"
BF16_ROOT="results/m3-quality/$BF16_RUN_ID"

python -m pipeline.m3_quality_preflight \
  --matrix "$BF16_MATRIX" --run-root "$BF16_ROOT"

python -m pipeline.m3_quality_eval contract-gate \
  --reference-root "$PAIR_ROOT" --candidate-root "$BF16_ROOT" \
  --out "$BF16_ROOT/cross_run_contract_gate.json"

python - "$BF16_ROOT/cross_run_contract_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
assert gate["valid"] is True, gate["mismatches"]
PY

bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$BF16_MATRIX" --run-root "$BF16_ROOT" --dry-run
```

The dry run must show the two-node Ray topology preflight followed by one
two-node TP8xPP2/Ray smoke arm, with no AWQ or GPTQ model.

## 4. Wave A: three detached controllers, seven nodes

Create all three detached controllers from the outside-allocation login shell.
Each controller owns its stdout, stderr, and return-code file:

```bash
PAIR_SESSION="m3-pair-r47-prod-$(date -u +%H%M%S)"
REPLAY_SESSION="m3-gptq-r47-replay-$(date -u +%H%M%S)"
BF16_SMOKE_SESSION="m3-bf16-r47-smoke-$(date -u +%H%M%S)"
mkdir -p "$PAIR_ROOT/logs" "$PAIR_ROOT/diagnostics" "$BF16_ROOT/logs"
[[ -z "${SLURM_JOB_ID:-}" ]]

# Four nodes: paired 100-sample production.
tmux new-session -d -s "$PAIR_SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD' && bash pipeline/slurm/run_m3_quality_eval_srun.sh --profile production --matrix '$PAIR_MATRIX' --run-root '$PAIR_ROOT' --smoke-gate '$PAIR_ROOT/smoke_gate.json' >'$PAIR_ROOT/logs/production-controller.out' 2>'$PAIR_ROOT/logs/production-controller.err'; printf '%s\n' \$? >'$PAIR_ROOT/production-controller.rc'"

# One node: exact GPTQ raw replay, one model load and two controls.
tmux new-session -d -s "$REPLAY_SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD' && srun --exclusive --nodes=1 --ntasks=1 --gpus-per-node=8 --kill-on-bad-exit=1 --time=12:00:00 python -m pipeline.m3_empty_output_replay --config pipeline/configs/eval_minimax_m3_reasoning_r4.yaml --model /mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay --samples '$PAIR_ROOT/models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl' --attempt-uid 8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878 --out '$PAIR_ROOT/diagnostics/empty-output-replay.json' >'$PAIR_ROOT/logs/empty-output-replay-controller.out' 2>'$PAIR_ROOT/logs/empty-output-replay-controller.err'; printf '%s\n' \$? >'$PAIR_ROOT/diagnostics/empty-output-replay-controller.rc'"

# Two nodes: BF16 TP8xPP2/Ray smoke.
tmux new-session -d -s "$BF16_SMOKE_SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD' && bash pipeline/slurm/run_m3_quality_eval_srun.sh --profile smoke --matrix '$BF16_MATRIX' --run-root '$BF16_ROOT' >'$BF16_ROOT/logs/smoke-controller.out' 2>'$BF16_ROOT/logs/smoke-controller.err'; printf '%s\n' \$? >'$BF16_ROOT/smoke-controller.rc'"

# Wave A allocation assertion: 4 + 1 + 2 = 7 nodes.
test $((4 + 1 + 2)) -eq 7
squeue -u "$USER"
tmux ls
```

Do not relaunch any arm while its controller is alive. A failed controller is
evidence, not authorization to retry.

## 5. Wave B gate and BF16 production

Do not run this gate until both `REPLAY_SESSION` and `BF16_SMOKE_SESSION` have
ended. Their synchronous controller commands write rc files only after their
top-level allocations end, so the checks below explicitly prove the replay's
one-node allocation and the BF16 smoke allocation have ended before Wave B:

```bash
! tmux has-session -t "$REPLAY_SESSION" 2>/dev/null
! tmux has-session -t "$BF16_SMOKE_SESSION" 2>/dev/null
test -f "$PAIR_ROOT/diagnostics/empty-output-replay-controller.rc"
test -f "$BF16_ROOT/smoke-controller.rc"
test "$(cat "$PAIR_ROOT/diagnostics/empty-output-replay-controller.rc")" = 0
test "$(cat "$BF16_ROOT/smoke-controller.rc")" = 0

python - "$PAIR_ROOT/diagnostics/empty-output-replay.json" <<'PY'
import json, sys
replay = json.load(open(sys.argv[1]))
assert replay["fixed_caps"] == [256, 16384], replay
assert len(replay["controls"]) == 2, replay
assert "classification" in replay, replay
PY

python - "$BF16_ROOT/smoke_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
assert gate["ready_for_production"] is True, gate
PY

python -m pipeline.m3_quality_eval contract-gate \
  --reference-root "$PAIR_ROOT" --candidate-root "$BF16_ROOT" \
  --out "$BF16_ROOT/cross_run_contract_gate.json"

python - "$BF16_ROOT/cross_run_contract_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
assert gate["valid"] is True, gate["mismatches"]
PY

bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$BF16_MATRIX" --run-root "$BF16_ROOT" \
  --smoke-gate "$BF16_ROOT/smoke_gate.json" --dry-run
```

The dry run must show exactly two concurrent two-node TP8xPP2/Ray arms,
`bf16/gpqa` and `bf16/reasoning_suite`, each with the committed 24-hour limit.
After verifying that the replay allocation has ended, launch Wave B:

```bash
BF16_PROD_SESSION="m3-bf16-r47-prod-$(date -u +%H%M%S)"
tmux new-session -d -s "$BF16_PROD_SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD' && bash pipeline/slurm/run_m3_quality_eval_srun.sh --profile production --matrix '$BF16_MATRIX' --run-root '$BF16_ROOT' --smoke-gate '$BF16_ROOT/smoke_gate.json' >'$BF16_ROOT/logs/production-controller.out' 2>'$BF16_ROOT/logs/production-controller.err'; printf '%s\n' \$? >'$BF16_ROOT/production-controller.rc'"

# If paired production is still active: 4 + 4 = 8, never 9.
test $((4 + 4)) -eq 8
squeue -u "$USER"
```

Paired production need not end before Wave B. If it is still active, its four
nodes and BF16 production's four nodes overlap at the packet maximum of eight.
No replay allocation remains, and no other packet-owned allocation may start.

## 6. Aggregate and return evidence

After each production controller ends, aggregate without hiding a quality-gate
rc. Record nonzero results as evidence; they do not authorize a retry:

```bash
set +e
python -m pipeline.m3_quality_eval aggregate \
  --root "$PAIR_ROOT" --matrix "$PAIR_MATRIX"
printf '%s\n' "$?" >"$PAIR_ROOT/aggregate.rc"

python -m pipeline.m3_quality_eval aggregate \
  --root "$BF16_ROOT" --matrix "$BF16_MATRIX"
printf '%s\n' "$?" >"$BF16_ROOT/aggregate.rc"
set -e
```

Return both exact eval roots in `RETURNED_FOR_ANALYSIS` state. Commit and push
small evidence; do not commit model checkpoints or caches. The return must
include or index the following exact contracts, with exact paths, byte sizes,
SHA-256 hashes, and bounded failure excerpts for artifacts left on cluster
storage:

- replay sidecar
  `$PAIR_ROOT/diagnostics/empty-output-replay.json`, including both controls'
  raw text/token IDs/finish and stop reasons, post-thinking and post-task-stop
  text, and `classification`;
- replay stdout/stderr
  `$PAIR_ROOT/logs/empty-output-replay-controller.out` and
  `$PAIR_ROOT/logs/empty-output-replay-controller.err`, plus scheduler job ID,
  node list, and GPU identity for the replay allocation;
- both run roots, `$PAIR_ROOT` and `$BF16_ROOT`, including preflight manifests,
  resolved tasks, sample manifests, launch plans, raw JSON/JSONL responses,
  logs, reports, package versions, scheduler job/node identities, and hashes;
- every controller rc:
  `$PAIR_ROOT/production-controller.rc`,
  `$PAIR_ROOT/diagnostics/empty-output-replay-controller.rc`,
  `$BF16_ROOT/smoke-controller.rc`, and
  `$BF16_ROOT/production-controller.rc`;
- every arm rc at
  `$PAIR_ROOT/models/*/shards/*/return_code.txt` and
  `$BF16_ROOT/models/*/shards/*/return_code.txt`, plus both `aggregate.rc`
  files;
- the paired and BF16 smoke reports and gates, especially
  `$PAIR_ROOT/smoke_gate.json`, `$BF16_ROOT/smoke_gate.json`, and
  `$BF16_ROOT/cross_run_contract_gate.json`;
- per-arm and merged `generation_health` artifacts, final `gates.json`
  `health_advisory`, comparison matrices, and Markdown/JSON reports.

For every nonzero rc, report the first failing operation and last successful
stage. Empty responses remain complete score-zero observations and health
advisories. The executor does not retry, discard, repair, or interpret them and
does not change any scientific or runtime contract.
