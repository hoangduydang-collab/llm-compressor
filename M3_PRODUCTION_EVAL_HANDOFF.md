# Execution packet: MiniMax-M3 r4.8 recovery qualification

- Protocol version: 1
- State: `READY_FOR_EXECUTOR`
- Packet revision: `2026-07-16-r4.8`
- Planner owner: Codex planner
- Intended executor: any authorized cluster executor
- Base Git commit: `a395312ea60f643b626eed1a5731187752eb5f50`
- Required branch: `duy-branch`
- Decision question: Can the repaired exact GPTQ replay initialize under the proven smoke serving envelope, and can BF16 TP16xPP1/Ray initialize and complete the fixed r4 smoke grid with enough evidence to classify the first runtime boundary?

## r4.7 instructions are SUPERSEDED

All executable launch instructions from packet `2026-07-15-r4.7` are
**SUPERSEDED by this r4.8 packet**. Do not execute its paired-production, replay,
BF16 TP8xPP2 smoke, Wave B, aggregation, or BF16 production instructions.

This replacement does not cancel or relaunch the four-node paired GPTQ/AWQ
production run, which already runs independently under
`results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4`. Its root and the
r4.7 replay/BF16 partial evidence are immutable. Preserve any healthy active
paired arms. This packet owns only the repaired exact GPTQ replay and one BF16
TP16xPP1 qualification smoke.

## Objective, hypothesis, and scope

The r4.7 replay omitted typed serving fields that the quality wrapper had
forwarded, so it failed during GPTQ model construction before either diagnostic
control. The shared argument builder is now repaired. The exact same saved
attempt should initialize with the proven expert-parallel serving envelope and
return raw and postprocessed evidence at the fixed caps `[256, 16384]`.

The r4.7 BF16 TP8xPP2/Ray arm failed because MiniMax-M3 does not implement
vLLM's pipeline-parallel interface. The committed matrix now requests the
existing vLLM multi-node alternative, TP16xPP1/Ray. This packet qualifies that
topology with the smoke profile only; it does not assume the topology works.

In scope:

- one repaired exact GPTQ replay using the unchanged attempt UID and fixed caps;
- one fresh BF16 CPU preflight, cross-run contract gate, TP16xPP1 dry run, and
  TP16xPP1/Ray smoke qualification;
- monitoring, immutable evidence preservation, factual boundary
  classification, and a protocol-compliant return.

Not authorized:

- paired-production launch, relaunch, cancellation, mutation, or aggregation;
- BF16 production or any `--profile production` launch;
- TP8xPP2 or any topology fallback/substitution;
- automatic retry or resume of either correction arm;
- any local vLLM/MiniMax patch, including a `SupportsPP` patch;
- model, checkpoint, task, sample, prompt, seed, generation, gate, timeout, or
  harness changes.

## Preconditions and exact environment

- Repository: `/mnt/nfs/hoangduy/projects/llm-compressor`
- Branch: `duy-branch`
- Environment: `/mnt/nfs/hoangduy/venvs/quant`
- Scheduler: top-level `srun` from detached `tmux` sessions outside Slurm.
- Allocation method: no nested allocation; `sbatch` is prohibited.
- Packet-owned resources: one exclusive 8xH100 node for replay plus two
  exclusive 8xH100 nodes for BF16 smoke.
- Time limits: replay 12 hours; BF16 topology preflight and smoke allocations
  12 hours. Only BF16 smoke receives the 900-second placement watchdog and
  10,800-second model-initialization watchdog.
- Concurrency: the two correction controllers may overlap and use three nodes.
  If the four paired-production nodes remain active, packet-owned concurrency is
  exactly `4 + 1 + 2 = 7` nodes.

Required inputs:

| Input | Exact path or identifier | Required validation |
| --- | --- | --- |
| Paired root | `results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4` | Directory, saved MMLU-Pro smoke shard, and original r4.7 replay evidence exist |
| GPTQ overlay | `/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay` | Directory exists |
| Replay attempt | `8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878` | Replay module validates task, doc, seed, prompt hash, and original arguments before GPU initialization |
| Replay config | `pipeline/configs/eval_minimax_m3_reasoning_r4.yaml` | Tracked at the verified revision |
| BF16 matrix | `pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml` | Dry-run plan must contain one two-node TP16xPP1/Ray smoke arm |
| BF16 checkpoint | `/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3` | Directory exists |
| r4.7 BF16 root | `results/m3-quality/20260715T160500Z-m3-bf16-reasoning-r4` | Preserve as immutable partial evidence |

## Workspace policy

Tracked changes are blocking. The existing paired root, r4.7 replay files, and
r4.7 BF16 root are protected and must not be written, deleted, renamed, or
rehashed into replacement files. The only permitted new artifacts are the four
fresh r4.8 replay paths under the paired root, one fresh r4.8 BF16 root, and the
small return packet under `evidence/m3-r48/$BF16_RUN_ID` created only after both
controllers end.
Unrelated untracked files outside those exact paths are record-and-proceed only
when they do not shadow a tracked path, any required input, or either fresh
output path; enumerate them in the return. Any collision is a stop condition.

## Setup, revision verification, and CPU suite

Run this block from one persistent login shell. Every later command uses the
variables from this shell.

```bash
set -euo pipefail
cd /mnt/nfs/hoangduy/projects/llm-compressor
git switch duy-branch
git pull --ff-only
CODE_BASE=a395312ea60f643b626eed1a5731187752eb5f50
ACTUAL_COMMIT="$(git rev-parse HEAD)"
ORIGIN_COMMIT="$(git rev-parse origin/duy-branch)"
test "$ACTUAL_COMMIT" = "$ORIGIN_COMMIT"
git merge-base --is-ancestor "$CODE_BASE" "$ACTUAL_COMMIT"
python - "$CODE_BASE" "$ACTUAL_COMMIT" <<'PY'
import subprocess, sys
code_base, actual = sys.argv[1:]
allowed = [
    "M3_PRODUCTION_EVAL_HANDOFF.md",
    "docs/superpowers/specs/2026-07-14-m3-task-isolated-paired-quality-rerun-design.md",
]
changed = subprocess.check_output(
    ["git", "diff", "--name-only", f"{code_base}..{actual}"], text=True
).splitlines()
assert changed == allowed, {"expected": allowed, "actual": changed}
PY
printf 'code_base=%s\nactual_commit=%s\norigin_commit=%s\n' \
  "$CODE_BASE" "$ACTUAL_COMMIT" "$ORIGIN_COMMIT"
git diff --quiet
git diff --cached --quiet
git status --short
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD"
test -z "${SLURM_JOB_ID:-}"
python --version
python -m pip show llmcompressor lm_eval vllm ray torch

python -m pytest -q \
  pipeline/tests/test_m3_empty_output_replay.py \
  pipeline/tests/test_lmeval_runner.py \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_m3_quality_eval_runner.py \
  pipeline/tests/test_m3_quality_evidence.py
```

`Base Git commit` deliberately means the last reviewed commit that changes
executable code or configuration. The active packet/spec documentation commit
is later, so the executor must run the exact `origin/duy-branch` tip after pull
and prove that the complete `CODE_BASE..ACTUAL_COMMIT` diff contains exactly the
two allowed documentation files above. This rejects arbitrary executable drift
while allowing the packet to name a deterministic executable base.

Stop before allocation if revision/origin equality, the allowed-diff gate, the
clean tracked-workspace checks, environment activation, outside-allocation
assertion, package checks, or any CPU test fails. Preserve `ACTUAL_COMMIT` for
the return manifest. Do not clean or delete unrelated files.

## Fresh paths and collision gate

```bash
set -euo pipefail
PAIR_ROOT=results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4
BF16_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-bf16-tp16-qualification-r48"
BF16_ROOT="results/m3-quality/$BF16_RUN_ID"
REPLAY_JSON="$PAIR_ROOT/diagnostics/empty-output-replay-r48.json"
REPLAY_OUT="$PAIR_ROOT/logs/empty-output-replay-r48.out"
REPLAY_ERR="$PAIR_ROOT/logs/empty-output-replay-r48.err"
REPLAY_RC="$PAIR_ROOT/diagnostics/empty-output-replay-r48.rc"
R47_REPLAY_OUT="$PAIR_ROOT/logs/empty-output-replay-controller.out"
R47_REPLAY_ERR="$PAIR_ROOT/logs/empty-output-replay-controller.err"
R47_REPLAY_RC="$PAIR_ROOT/diagnostics/empty-output-replay-controller.rc"
R47_BF16_ROOT=results/m3-quality/20260715T160500Z-m3-bf16-reasoning-r4

test -d "$PAIR_ROOT"
test -f "$PAIR_ROOT/models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl"
test -d artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay
test -d /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
test -f "$R47_REPLAY_OUT"
test -f "$R47_REPLAY_ERR"
test -f "$R47_REPLAY_RC"
test -d "$R47_BF16_ROOT"
test "$REPLAY_OUT" != "$R47_REPLAY_OUT"
test "$REPLAY_ERR" != "$R47_REPLAY_ERR"
test "$REPLAY_RC" != "$R47_REPLAY_RC"
test ! -e "$REPLAY_JSON"
test ! -e "$REPLAY_OUT"
test ! -e "$REPLAY_ERR"
test ! -e "$REPLAY_RC"
test ! -e "$BF16_ROOT"
mkdir -p "$PAIR_ROOT/logs" "$PAIR_ROOT/diagnostics"
```

If any fresh replay artifact or `BF16_ROOT` exists, stop and return the exact
collision. Do not pick a replacement name after a collision. The original r4.7
stdout, stderr, rc, partial reports, and full r4.7 BF16 root remain immutable.

## BF16 preflight, contract gate, and dry run

These commands create and validate only the fresh BF16 root; they do not
allocate GPUs.

```bash
set -euo pipefail
BF16_MATRIX=pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml
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

TIME_LIMIT=12:00:00 bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$BF16_MATRIX" --run-root "$BF16_ROOT" \
  --dry-run >"$BF16_ROOT/logs/smoke-dry-run-r48.out"

python - "$BF16_ROOT/smoke_launch_plan.json" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
arms = plan["arms"]
assert len(arms) == 1, arms
arm = arms[0]
assert arm["model_label"] == "bf16", arm
assert arm["nodes"] == 2, arm
assert arm["tensor_parallel_size"] == 16, arm
assert arm["pipeline_parallel_size"] == 1, arm
assert arm["distributed_executor_backend"] == "ray", arm
assert plan["total_nodes"] == 2, plan
PY

test "$(grep -c -- '--profile production' "$BF16_ROOT/logs/smoke-dry-run-r48.out" || true)" -eq 0
```

Expected: the contract gate is valid and the dry run shows exactly the
two-node Ray topology preflight followed by one two-node TP16xPP1/Ray smoke arm,
with no GPTQ/AWQ arm. Stop and return if any gate or assertion fails.

## Launch exactly two detached correction controllers

Run from the same outside-allocation login shell after every preceding gate
passes. Both controllers are independent. Failure of either does not cancel the
other, relaunch it, or affect paired production.

```bash
set -euo pipefail
REPLAY_SESSION="m3-gptq-r48-replay-$(date -u +%H%M%S)"
BF16_SMOKE_SESSION="m3-bf16-r48-smoke-$(date -u +%H%M%S)"
BF16_OUT="$BF16_ROOT/logs/smoke-controller-r48.out"
BF16_ERR="$BF16_ROOT/logs/smoke-controller-r48.err"
BF16_RC="$BF16_ROOT/smoke-controller-r48.rc"
PACKET_START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
REPLAY_PAYLOAD="cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD' && srun --exclusive --nodes=1 --ntasks=1 --gpus-per-node=8 --kill-on-bad-exit=1 --time=12:00:00 python -m pipeline.m3_empty_output_replay --config pipeline/configs/eval_minimax_m3_reasoning_r4.yaml --model /mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay --samples '$PAIR_ROOT/models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl' --attempt-uid 8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878 --out '$REPLAY_JSON' >'$REPLAY_OUT' 2>'$REPLAY_ERR'; printf '%s\n' \$? >'$REPLAY_RC'"
BF16_PAYLOAD="cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD' && M3_PLACEMENT_TIMEOUT_SECONDS=900 M3_MODEL_INIT_TIMEOUT_SECONDS=10800 TIME_LIMIT=12:00:00 bash pipeline/slurm/run_m3_quality_eval_srun.sh --profile smoke --matrix '$BF16_MATRIX' --run-root '$BF16_ROOT' >'$BF16_OUT' 2>'$BF16_ERR'; printf '%s\n' \$? >'$BF16_RC'"

test -z "${SLURM_JOB_ID:-}"
test ! -e "$BF16_OUT"
test ! -e "$BF16_ERR"
test ! -e "$BF16_RC"
! tmux has-session -t "$REPLAY_SESSION" 2>/dev/null
! tmux has-session -t "$BF16_SMOKE_SESSION" 2>/dev/null
printf '%s\n' "$REPLAY_PAYLOAD" >"$BF16_ROOT/replay-controller-command-r48.txt"
printf '%s\n' "$BF16_PAYLOAD" >"$BF16_ROOT/bf16-controller-command-r48.txt"
printf 'packet_start_utc=%s\nreplay_session=%s\nbf16_session=%s\n' \
  "$PACKET_START_UTC" "$REPLAY_SESSION" "$BF16_SMOKE_SESSION" \
  >"$BF16_ROOT/controller-metadata-r48.txt"
python --version >"$BF16_ROOT/package-versions-r48.txt" 2>&1
python -m pip show llmcompressor lm_eval vllm ray torch \
  >>"$BF16_ROOT/package-versions-r48.txt" 2>&1
nvidia-smi --query-gpu=driver_version --format=csv,noheader \
  >"$BF16_ROOT/driver-versions-r48.txt" 2>&1 || \
  printf 'not reached on login host\n' >"$BF16_ROOT/driver-versions-r48.txt"
squeue -u "$USER" >"$BF16_ROOT/squeue-at-launch-r48.txt" 2>&1

tmux new-session -d -s "$REPLAY_SESSION" "$REPLAY_PAYLOAD"
tmux new-session -d -s "$BF16_SMOKE_SESSION" "$BF16_PAYLOAD"

test $((1 + 2)) -eq 3
test $((4 + 1 + 2)) -eq 7
tmux ls
squeue -u "$USER"
```

The replay controller's only allocation is the exact command below, wrapped by
the first detached session:

```text
srun --exclusive --nodes=1 --ntasks=1 --gpus-per-node=8 \
  --kill-on-bad-exit=1 --time=12:00:00 \
  python -m pipeline.m3_empty_output_replay \
  --config pipeline/configs/eval_minimax_m3_reasoning_r4.yaml \
  --model /mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay \
  --samples "$PAIR_ROOT/models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl" \
  --attempt-uid 8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878 \
  --out "$REPLAY_JSON"
```

The BF16 controller's only evaluation profile is this smoke command, wrapped by
the second detached session:

```text
M3_PLACEMENT_TIMEOUT_SECONDS=900 \
M3_MODEL_INIT_TIMEOUT_SECONDS=10800 \
TIME_LIMIT=12:00:00 \
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke \
  --matrix pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml \
  --run-root "$BF16_ROOT"
```

The duplicate expanded commands above are documentation of the two controller
payloads, not additional launches. Execute only the two `tmux new-session`
commands. There is no third correction controller and no paired or BF16
production command.

## Monitoring and exact stop rules

Monitoring is observational and does not authorize cancellation, adaptation,
or retry:

```bash
set -euo pipefail
tmux ls
squeue -u "$USER"
tail -n 100 "$REPLAY_OUT" "$REPLAY_ERR" "$BF16_OUT" "$BF16_ERR"
find "$BF16_ROOT/models/bf16/shards/smoke" -maxdepth 3 -type f -print | sort
```

- No automatic retry is authorized. A missing output, nonzero rc, timeout,
  scheduler failure, or controller disappearance is evidence to preserve and
  return, not permission to relaunch.
- If no `CREATED` placement group exists at 15 minutes, classify a placement
  stall enforced by `M3_PLACEMENT_TIMEOUT_SECONDS=900`. Return the placement
  state captured at the deadline and the timeout marker.
- If no model-ready progress exists at three hours, classify a stalled-load
  finding enforced by `M3_MODEL_INIT_TIMEOUT_SECONDS=10800`. The top-level BF16
  allocations remain bounded at 12 hours.
- If placement succeeds but workers do not start, return worker-start evidence.
  If workers start but weights do not load, return weight-load evidence. If
  weights load but communication fails, return the first collective error. If
  the model becomes ready but requests or scoring fail, return request-generation
  or evaluation evidence respectively.
- Stop and return on any contract mismatch, topology other than TP16xPP1/Ray,
  output collision, protected-root mutation, unauthorized command requirement,
  or material ambiguity. Preserve ephemeral scheduler and Ray evidence first.

## Expected jobs, gates, and independence

| Controller | Resources | Required result or evidence | Effect of failure |
| --- | --- | --- | --- |
| Repaired exact GPTQ replay | one exclusive node, 8 GPUs, TP8, 12 hours | Replay JSON for caps `[256, 16384]`, or first model-load failure with effective vLLM arguments | Preserve evidence; BF16 and paired production continue |
| BF16 qualification | one two-node, 16-GPU TP16xPP1/Ray smoke arm after its two-node topology gate, 12 hours | Complete fixed smoke grid and true smoke gate, or evidence for the first failed boundary | Preserve evidence; replay and paired production continue; no BF16 production |

The smoke is setup qualification, not BF16 quality evidence and not directly
score-comparable to a public benchmark. Even a passing smoke does not authorize
production. Allowed adaptations: none. Pre-authorized retries: trigger `None`,
maximum retry count `0`, fresh run ID required `yes` for any separately approved
future attempt, and all inputs must remain unchanged unless a later planner
packet says otherwise. Pre-authorized record-and-proceed conditions: unrelated
non-shadowing untracked files only, with exact enumeration in the return.

## Return contract

After both correction controllers have ended or stopped, return the full fresh
BF16 root and index the immutable r4.7 partial reports
`M3_R4_7_GPTQ_REPLAY_STOPPED_REPORT.md` and
`M3_R4_7_BF16_SMOKE_STOPPED_REPORT.md`. Commit and push small evidence; retain
large artifacts on durable shared storage with absolute path, byte size, and
SHA-256. Record packet revision, expected and actual Git commits, executed
commands, package/driver versions, start/end times, controller/session names,
scheduler job/step IDs, nodes, GPU identities, topology, every rc, all
deviations, and final repository state.

The replay return must include either:

- `$REPLAY_JSON` with fixed caps `[256, 16384]`, raw and effective arguments,
  raw text, output token IDs, token counts, finish/stop reasons, thinking-marker
  presence, every postprocessing stage, prompt/checkpoint/environment identity,
  and classification; or
- the first model-load failure with the complete effective vLLM arguments and
  `$REPLAY_OUT`, `$REPLAY_ERR`, and `$REPLAY_RC`.

The BF16 return must include or index:

- the topology gate and full `ray_preflight` directory;
- `placement-monitor.log`, placement-group state, requested bundles and Ray
  node resources, and `placement-at-deadline.log` when the deadline is reached;
- both rank-local GPU monitors;
- both rank-local Ray-log archives or explicit `.missing` markers;
- `placement-timeout.json` and `model-init-timeout.json` when applicable;
- the arm manifest, arm/controller rc, smoke report and smoke gate;
- scheduler identity and the first and last worker/model-loading progress
  markers, plus the first NCCL/vLLM/request/evaluation error when applicable;
- the entire fresh `$BF16_ROOT`, including preflight manifests, resolved tasks,
  sample manifests, launch plan, raw outputs, logs, reports, package versions,
  sizes, and hashes.

## Completion detection and return packaging

Run this block after both launches. It waits only for the two named detached
controllers, then packages whatever evidence exists. A missing or nonzero rc is
recorded as scientific/runtime evidence and does not make packaging fail or
authorize a retry. Before running it, set `BOUNDARY_CLASSIFICATION` to one of
the six authorized factual boundaries and set `LAST_SUCCESSFUL_STAGE`,
`FIRST_FAILING_OPERATION`, and `BOUNDARY_EVIDENCE` to exact preserved facts. A
successful qualification must use `BOUNDARY_CLASSIFICATION=evaluation` and the
exact sentinel `FIRST_FAILING_OPERATION=none`; every non-success return must use
a factual non-`none` operation. Set `DEVIATIONS=none` when there were no
deviations, otherwise provide the exact deviation record.

```bash
set -euo pipefail
while tmux has-session -t "$REPLAY_SESSION" 2>/dev/null || tmux has-session -t "$BF16_SMOKE_SESSION" 2>/dev/null; do
  date -u +%Y-%m-%dT%H:%M:%SZ
  tmux ls 2>/dev/null || true
  squeue -u "$USER" || true
  sleep 60
done

read_rc_or_missing() {
  if test -f "$1"; then
    tr -d '[:space:]' <"$1"
  else
    printf 'missing'
  fi
}
REPLAY_RC_VALUE="$(read_rc_or_missing "$REPLAY_RC")"
BF16_RC_VALUE="$(read_rc_or_missing "$BF16_RC")"
case "${BOUNDARY_CLASSIFICATION:-}" in
  placement|"worker start"|"weight load"|"collective communication"|"request generation"|evaluation) ;;
  *) echo "set BOUNDARY_CLASSIFICATION to one authorized factual boundary" >&2; exit 1 ;;
esac
test -n "${LAST_SUCCESSFUL_STAGE:-}"
test -n "${FIRST_FAILING_OPERATION:-}"
test -n "${BOUNDARY_EVIDENCE:-}"
test -n "${DEVIATIONS:-}"
if test "$REPLAY_RC_VALUE" = 0 && test "$BF16_RC_VALUE" = 0; then
  test "$BOUNDARY_CLASSIFICATION" = evaluation
  test "$FIRST_FAILING_OPERATION" = none
else
  test "$FIRST_FAILING_OPERATION" != none
fi

EVIDENCE_DIR="evidence/m3-r48/$BF16_RUN_ID"
test ! -e "$EVIDENCE_DIR"
mkdir -p "$EVIDENCE_DIR"

PACKET_END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'replay_rc=%s\nbf16_rc=%s\n' "$REPLAY_RC_VALUE" "$BF16_RC_VALUE" \
  | tee "$EVIDENCE_DIR/controller-return-codes.txt"
git rev-parse HEAD >"$EVIDENCE_DIR/actual-git-commit.txt"
git rev-parse origin/duy-branch >"$EVIDENCE_DIR/origin-git-commit.txt"
printf '%s\n' "$CODE_BASE" >"$EVIDENCE_DIR/code-base.txt"
git status --short >"$EVIDENCE_DIR/git-status-before-packaging.txt"
squeue -u "$USER" >"$EVIDENCE_DIR/squeue-final.txt" 2>&1 || true
if ! sacct -u "$USER" --starttime "$PACKET_START_UTC" \
  --format=JobIDRaw,JobName%40,State,ExitCode,NodeList,Elapsed,Start,End -P \
  >"$EVIDENCE_DIR/sacct-final.txt" 2>&1; then
  printf 'missing: scheduler accounting not reached or unavailable\n' \
    >"$EVIDENCE_DIR/sacct-final.txt"
fi
printf 'packet_end_utc=%s\n' "$PACKET_END_UTC" \
  >"$EVIDENCE_DIR/packet-end-r48.txt"

python - "$BF16_ROOT" "$PAIR_ROOT" "$EVIDENCE_DIR" \
  "$REPLAY_JSON" "$REPLAY_OUT" "$REPLAY_ERR" "$REPLAY_RC" \
  "$REPLAY_RC_VALUE" "$BF16_RC_VALUE" "$CODE_BASE" "$ACTUAL_COMMIT" \
  "$PACKET_START_UTC" "$PACKET_END_UTC" "$REPLAY_SESSION" \
  "$BF16_SMOKE_SESSION" "$DEVIATIONS" <<'PY'
import hashlib, json, shutil, subprocess, sys
from pathlib import Path

(bf16_root, pair_root, evidence_dir, replay_json, replay_out, replay_err,
 replay_rc, replay_rc_value, bf16_rc_value, code_base, actual_commit,
 packet_start, packet_end, replay_session, bf16_session, deviations) = sys.argv[1:]
bf16_root, pair_root, evidence_dir = map(Path, (bf16_root, pair_root, evidence_dir))
small_limit = 2 * 1024 * 1024
aggregate_limit = 20 * 1024 * 1024
file_limit = 500
copy_aggregate_limit = 18 * 1024 * 1024
copy_file_limit = 450
records = []

def record(path, logical_root):
    if not path.is_file():
        return
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    records.append({
        "path": str(path.resolve()),
        "relative_path": str(path.relative_to(logical_root)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })

for path in sorted(bf16_root.rglob("*")):
    record(path, bf16_root)
for raw in map(Path, (replay_json, replay_out, replay_err, replay_rc)):
    if raw.is_file():
        record(raw, pair_root)

copied_bytes = copied_files = 0
for item in records:
    source = Path(item["path"])
    eligible = source.suffix in {
        ".json", ".jsonl", ".log", ".out", ".err", ".rc", ".txt"
    } and source.stat().st_size <= small_limit
    if not eligible:
        item["small_packet"] = "indexed_not_eligible"
        continue
    if copied_files + 1 > copy_file_limit or copied_bytes + source.stat().st_size > copy_aggregate_limit:
        item["small_packet"] = "indexed_aggregate_cap"
        continue
    relative = Path("small-artifacts") / (
        "bf16" if str(source).startswith(str(bf16_root.resolve())) else "replay"
    ) / item["relative_path"]
    target = evidence_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(source, target)
    item["small_packet"] = str(relative)
    copied_files += 1
    copied_bytes += source.stat().st_size

r47 = []
for relative in (
    "M3_R4_7_GPTQ_REPLAY_STOPPED_REPORT.md",
    "M3_R4_7_BF16_SMOKE_STOPPED_REPORT.md",
):
    path = Path(relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    r47.append({
        "repo_relative_path": relative,
        "absolute_path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "git_blob": subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{relative}"], text=True
        ).strip(),
    })

evidence_paths = {
    "planned_topology": str((bf16_root / "smoke_launch_plan.json").resolve()),
    "actual_topology": str((bf16_root / "ray_preflight").resolve()),
    "gpu_monitors": str((bf16_root / "models/bf16/shards/smoke").resolve()),
    "scheduler_snapshot": str((evidence_dir / "sacct-final.txt").resolve()),
}
def runtime_status(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "status": "available" if path.exists() else "missing/not reached",
    }

runtime_metadata = {
    "replay_command": runtime_status(bf16_root / "replay-controller-command-r48.txt"),
    "bf16_command": runtime_status(bf16_root / "bf16-controller-command-r48.txt"),
    "controller_metadata": runtime_status(bf16_root / "controller-metadata-r48.txt"),
    "package_versions": runtime_status(bf16_root / "package-versions-r48.txt"),
    "driver_versions": runtime_status(bf16_root / "driver-versions-r48.txt"),
    "scheduler_launch": runtime_status(bf16_root / "squeue-at-launch-r48.txt"),
    "scheduler_final": runtime_status(evidence_dir / "sacct-final.txt"),
    "planned_topology": runtime_status(bf16_root / "smoke_launch_plan.json"),
    "actual_topology": runtime_status(bf16_root / "ray_preflight"),
    "gpu_evidence": runtime_status(bf16_root / "models/bf16/shards/smoke"),
}
manifest = {
    "schema_version": 1,
    "packet_revision": "2026-07-16-r4.8",
    "code_base": code_base,
    "actual_git_commit": actual_commit,
    "packet_start_utc": packet_start,
    "packet_end_utc": packet_end,
    "sessions": {"replay": replay_session, "bf16": bf16_session},
    "replay_controller_rc": replay_rc_value,
    "bf16_controller_rc": bf16_rc_value,
    "deviations": deviations,
    "planned_topology": {"replay": "1 node, 8 GPUs, TP8", "bf16": "2 nodes, 16 GPUs, TP16xPP1/Ray"},
    "evidence_paths": evidence_paths,
    "runtime_metadata": runtime_metadata,
    "missing_runtime_metadata_sentinel": "missing/not reached",
    "bf16_root": str(bf16_root.resolve()),
    "paired_root": str(pair_root.resolve()),
    "immutable_r47_reports": r47,
    "small_packet_limits": {"per_file_bytes": small_limit, "aggregate_bytes": aggregate_limit, "file_count": file_limit, "copy_reservation_bytes": copy_aggregate_limit, "copy_reservation_files": copy_file_limit},
    "small_packet_actual": {"bytes": copied_bytes, "files": copied_files},
    "artifacts": records,
}
(evidence_dir / "evidence-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)

report = [
    "# MiniMax-M3 r4.8 correction evidence",
    "",
    "- State: `RETURNED_FOR_ANALYSIS`",
    f"- Code base: `{code_base}`",
    f"- Actual Git commit: `{actual_commit}`",
    f"- Start/end UTC: `{packet_start}` / `{packet_end}`",
    f"- Sessions: replay `{replay_session}`; BF16 `{bf16_session}`",
    f"- Replay controller rc: `{replay_rc_value}`",
    f"- BF16 controller rc: `{bf16_rc_value}`",
    f"- BF16 durable root: `{bf16_root.resolve()}`",
    f"- Artifact records: {len(records)}",
    f"- Small packet: {copied_files} files / {copied_bytes} bytes",
    f"- Deviations: {deviations}",
    "- Retry: none",
    "- Strategic interpretation: none; returned to planner",
    "",
    "Nonzero or missing controller return codes are preserved evidence, not a",
    "packaging failure and not authorization to retry, patch, or change topology.",
]
(evidence_dir / "evidence-report.md").write_text(
    "\n".join(report) + "\n", encoding="utf-8"
)
PY

printf '\n- Factual boundary: `%s`\n- Last successful stage: %s\n- First failing operation: %s\n- Boundary evidence: `%s`\n' \
  "$BOUNDARY_CLASSIFICATION" "$LAST_SUCCESSFUL_STAGE" \
  "$FIRST_FAILING_OPERATION" "$BOUNDARY_EVIDENCE" \
  >>"$EVIDENCE_DIR/evidence-report.md"

python - "$EVIDENCE_DIR" <<'PY'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1])
lines = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    if path.name == "SHA256SUMS":
        continue
    lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")
(root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
files = [p for p in root.rglob("*") if p.is_file()]
total = sum(p.stat().st_size for p in files)
if len(files) > 500 or total > 20 * 1024 * 1024:
    raise SystemExit(f"small evidence packet exceeds bound: files={len(files)} bytes={total}")
PY

git add -- "$EVIDENCE_DIR"
git diff --cached --quiet && {
  echo "no small evidence files were staged" >&2
  exit 1
}
git commit -m "evidence(m3): return r4.8 recovery qualification"
git push origin duy-branch
git status --short
```

The command rejects any classification outside the six authorized factual
boundaries. The manifest records an absolute path, byte size, and SHA-256 for every file in
the durable fresh BF16 root and every available fresh replay artifact. Files at
most 2 MiB with raw-log or structured-evidence suffixes are copied into the
small Git packet; large files remain in their immutable durable roots and are
indexed rather than committed. This is evidence classification only, not
diagnosis or authorization to continue.

Classify the factual first boundary as exactly one of: `placement`,
`worker start`, `weight load`, `collective communication`, `request generation`,
or `evaluation`. Explain missing artifacts proportionally to how far execution
progressed. Do not infer a strategic verdict.

## Final instruction

Return in `RETURNED_FOR_ANALYSIS`, commit and push the complete evidence packet,
and stop. Do not retry, patch, fall back, launch paired work, or launch BF16
production. Only the planner may interpret the evidence and authorize a later
packet.

## Planner analysis and BF16 r49 authorization (2026-07-17)

### Root cause of the BF16 TP16/Ray engine-init hangs (r48, r48b, 20260712)

All three BF16 multi-node attempts died the same way: the vLLM EngineCore's
last log line was `Connected to Ray cluster.`, no placement group was ever
created, and all 16 GPUs sat at 0% until the arm was killed. The r48b
diagnostics (`.../20260716T081818Z-.../diagnostics/`) show the Ray cluster
itself was healthy: both nodes registered, 16/16 GPUs available, zero pending
demands.

The cause is the launcher's srun steps, not Ray or the model: with no
`--cpus-per-task`, Slurm 21.08 binds each step task to ONE physical core
(`Cpus_allowed_list: 0,96`; `SLURM_CPUS_ON_NODE=2` in the r48b rank logs,
verified again by a probe job). The entire per-node stack — GCS, raylet,
dashboard agents, the vLLM driver, EngineCore — shared 2 hardware threads, and
when the first driver connected, the raylet prestarted 192 idle Python workers
(raylet.out, 08:21:00-08:21:13) into that same 2-thread cgroup. The driver's
CoreWorker construction starves and never completes.

Causal loop closed locally on an idle 8xH100 host (2026-07-17):
- pinned to HTs {0,96} (the exact cluster mask): vLLM ray-backend engine init
  froze at `Connected to Ray cluster.` for 300+ s — the cluster signature;
- unpinned on the same host seconds later: the same init passed that point in
  ~4 s, created the placement group, spawned both RayWorkerProc actors,
  initialized NCCL, and proceeded to model load.

### The r49 fix (env-only, no launcher edits while other runs execute them)

`run_m3_quality_eval_srun.sh` and `test_m3_quality_eval_arm.sh` are being
executed by in-flight production arms (MXFP8 resume, jobs 12969/12970) and a
bash script must not be rewritten under a running reader, so r49 applies the
fix through the environment, which the launcher's sruns inherit:

- `SLURM_CPUS_PER_TASK=192` — Slurm 21.08 srun reads this input variable as
  `--cpus-per-task` (probe-verified: step went from 2 HTs to 188 CPUs);
- `RAY_enable_worker_prestart=0` — belt-and-braces: stops the raylet's
  192-worker prestart storm on first driver connect (flag present in the
  installed Ray 2.56 core).

Controller: `/mnt/nfs/hoangduy/claude/config/jobs/9e08e54d/tmp/bf16_relaunch_controller_r49.sh`
(r48b controller + the two exports; placement watchdog stays disabled because
its `ray list` probe needs `ray[default]` and false-kills healthy arms; the 3h
model-init watchdog is the backstop; TIME_LIMIT 12h).

### Follow-up once the hot scripts go cold (required, tracked)

When the MXFP8 resume and r8 quant smoke controllers exit, bake
`--cpus-per-task="${CPUS_PER_TASK:-192}"` into both sruns of
`run_m3_quality_eval_srun.sh`, replace the broken `ray list` placement
monitor/watchdog probes in `test_m3_quality_eval_arm.sh` with
`ray.util.placement_group_table()` (works without `ray[default]`), and add
launcher tests mirroring `test_m3_distributed_quant_smoke.py`'s
`--cpus-per-task` assertions. The quant-smoke launcher already carries the
inline fix (commit 70e5836d).

## r49 outcome and r50 authorization (2026-07-17)

r49 (SLURM_CPUS_PER_TASK=192 + RAY_enable_worker_prestart=0) confirmed the
CPU-starvation fix: the engine passed the historical hang point in ~5 s
("Connected to Ray cluster" -> "Creating a new placement group"), all 16
RayWorkerProc actors spawned across both nodes, and the arm FAILED FAST
(rc=1 at +2.5 min instead of hanging for hours) at the next-deeper layer:
16-rank NCCL init raised "remote process exited or there was a network error"
(`pynccl ncclCommInitRank`, `.../20260717T054606Z-.../logs/smoke-bf16-smoke.*`).

Isolation probes (2 nodes x 1 GPU, plain torch.distributed, no vLLM/Ray):
1. `MASTER_ADDR=<hostname>` fails BEFORE NCCL: TCPStore connect to
   `gpu-h104:29511` times out — node hostnames are not reachable cross-node.
   Node networks: `intranet` 10.2.4.x/24 (routable; what Ray uses), `storage`
   10.3.4.x/24, and 8 IB rails (`ibp*`) carrying only IPv6 link-local
   addresses.
2. With `NCCL_SOCKET_IFNAME=intranet` (+ intranet IP as MASTER_ADDR): 2-node
   allreduce SUCCEEDS over **NET/IB with GPUDirect RDMA** (12 mlx5 devices
   visible, OOB over intranet). The fabric is healthy; only NCCL's default
   out-of-band interface selection (which prefers the unroutable ib*/storage
   interfaces) is wrong for this cluster.

r50 = r49 controller + `NCCL_SOCKET_IFNAME=intranet` and
`GLOO_SOCKET_IFNAME=intranet` (gloo defaults to hostname resolution, which
does not resolve here). Controller:
`/mnt/nfs/hoangduy/claude/config/jobs/9e08e54d/tmp/bf16_relaunch_controller_r50.sh`.
These two exports belong in the baked-in launcher fix alongside
`--cpus-per-task` once the hot scripts go cold.
