# MiniMax-M3 r4.5 Production + BF16 Companion Handoff

- Protocol version: 1
- State: `READY_FOR_EXECUTOR`
- Packet revision: `2026-07-15-r4.6`
- Required branch: `duy-branch`
- Required ancestor: `05eeeaeb`
- Scheduler: top-level `srun` only; `sbatch` is unavailable
- Maximum allocation authorized by this packet: 6 nodes concurrently

## Planner decision on returned r4.5 smoke

Both AWQ and GPTQ loaded, completed all four tasks under all three seeds, wrote
all 60 expected attempt rows, returned arm rc 0, and passed every
infrastructure/artifact check. GPTQ produced one empty MMLU-Pro response at
`mmlu_pro_economics`, doc 45, seed 1234; the other 59 GPTQ attempts and all 60
AWQ attempts were non-empty.

This is an isolated quality observation, not a setup failure. Smoke exists to
validate that the execution path works, so r4.6 permits at most one empty
generation per model in smoke and records it as a warning. Full production
remains fail-closed at `max_degeneration_failures: 0`. Do not delete, replace,
or reinterpret the empty row, and do not add `min_tokens`, retry logic, or any
other harness change. Re-evaluate the existing smoke report under the corrected
policy; do not spend GPU time rerunning the same smoke.

## Fixed scientific and resource contract

The paired and BF16 runs use the same committed r4 eval configuration, resolved
lm-eval tasks, 100-question GPQA/MMLU-Pro/GSM8K subsets, all 30 AIME 2025
questions, seeds `42/1234/4158`, prompt/chat-template contract, filters,
metrics, and generation settings. Smoke's 256-token cap is setup-only and is
not quality evidence; production uses the pinned 16,384-token reasoning cap.

- Paired production: four independent one-node arms, maximum 4 nodes.
- BF16 smoke: one two-node TP8xPP2/Ray arm, maximum 2 nodes.
- BF16 production: two concurrent two-node TP8xPP2/Ray arms, maximum 4 nodes.
- Wave A may run paired production and BF16 smoke together (4 + 2 = 6 nodes).
- Wait for paired production to finish before starting BF16 production. This
  keeps at least five of the approximately eleven idle nodes outside the packet.

## 1. Pull and verify

Run from a login shell, outside every Slurm allocation:

```bash
cd /mnt/nfs/hoangduy/projects/llm-compressor
git switch duy-branch
git pull --ff-only
git merge-base --is-ancestor 05eeeaeb HEAD
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD"
[[ -z "${SLURM_JOB_ID:-}" ]]

python -m pytest -q \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_m3_quality_eval_runner.py \
  pipeline/tests/test_m3_quality_smoke_tmux.py \
  pipeline/tests/test_m3_quality_evidence.py
```

Stop and return evidence if the ancestor, environment, or tests fail.

## 2. Promote the completed paired smoke

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

Use a fresh root; never write BF16 artifacts into `PAIR_ROOT`:

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

The BF16 dry run must show a two-node Ray topology preflight followed by one
two-node TP8xPP2/Ray smoke arm, with no AWQ or GPTQ model.

## 4. Wave A: paired production + BF16 smoke (six nodes maximum)

Choose unique tmux names and preserve controller return codes:

```bash
PAIR_SESSION="m3-pair-r46-prod-$(date -u +%H%M%S)"
BF16_SMOKE_SESSION="m3-bf16-r4-smoke-$(date -u +%H%M%S)"
mkdir -p "$PAIR_ROOT/logs" "$BF16_ROOT/logs"

tmux new-session -d -s "$PAIR_SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD' && bash pipeline/slurm/run_m3_quality_eval_srun.sh --profile production --matrix '$PAIR_MATRIX' --run-root '$PAIR_ROOT' --smoke-gate '$PAIR_ROOT/smoke_gate.json' >'$PAIR_ROOT/logs/production-controller.out' 2>'$PAIR_ROOT/logs/production-controller.err'; printf '%s\n' \$? >'$PAIR_ROOT/production-controller.rc'"

tmux new-session -d -s "$BF16_SMOKE_SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD' && bash pipeline/slurm/run_m3_quality_eval_srun.sh --profile smoke --matrix '$BF16_MATRIX' --run-root '$BF16_ROOT' >'$BF16_ROOT/logs/smoke-controller.out' 2>'$BF16_ROOT/logs/smoke-controller.err'; printf '%s\n' \$? >'$BF16_ROOT/smoke-controller.rc'"

squeue -u "$USER"
tmux ls
```

Do not relaunch an arm while its controller is alive. When BF16 smoke ends,
require controller rc 0 and `smoke_gate.json.ready_for_production == true`.
One isolated empty is a warning; two empties, any loop, missing task/seed,
invalid artifact, wrong world size, or Ray failure stops BF16 without retry.

## 5. Wave B: BF16 production after paired production ends

Do not start this wave until `PAIR_SESSION` has ended and
`PAIR_ROOT/production-controller.rc` exists. A nonzero paired rc does not
authorize an automatic retry; return its evidence and continue with BF16 only
if the failure did not change shared code, datasets, or the scientific contract.

```bash
python - "$BF16_ROOT/smoke_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
assert gate["ready_for_production"] is True, gate
PY

python -m pipeline.m3_quality_eval contract-gate \
  --reference-root "$PAIR_ROOT" --candidate-root "$BF16_ROOT" \
  --out "$BF16_ROOT/cross_run_contract_gate.json"

bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$BF16_MATRIX" --run-root "$BF16_ROOT" \
  --smoke-gate "$BF16_ROOT/smoke_gate.json" --dry-run

BF16_PROD_SESSION="m3-bf16-r4-prod-$(date -u +%H%M%S)"
tmux new-session -d -s "$BF16_PROD_SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD' && bash pipeline/slurm/run_m3_quality_eval_srun.sh --profile production --matrix '$BF16_MATRIX' --run-root '$BF16_ROOT' --smoke-gate '$BF16_ROOT/smoke_gate.json' >'$BF16_ROOT/logs/production-controller.out' 2>'$BF16_ROOT/logs/production-controller.err'; printf '%s\n' \$? >'$BF16_ROOT/production-controller.rc'"
```

The dry run must show exactly two concurrent two-node TP8xPP2/Ray arms:
`bf16/gpqa` and `bf16/reasoning_suite`. Each arm has the committed 24-hour
limit. Do not alter samples, seeds, task aliases, generation settings, topology,
or checkpoint paths at runtime.

## 6. Return evidence

After each production controller ends, aggregate without hiding a quality-gate
rc. Record the rc rather than treating it as an execution retry request:

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

Commit and push small evidence, raw text logs, JSON/JSONL outputs, launch plans,
manifests, controller/arm return codes, scheduler job/node identities, package
versions, contract gates, generation-health files, and reports. Do not commit
model checkpoints or caches. Return both run roots in `RETURNED_FOR_ANALYSIS`
state with the first failing operation and last successful stage for any
nonzero rc. Do not interpret model quality, retry, or change the harness.
