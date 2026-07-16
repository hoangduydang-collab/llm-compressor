# Execution packet: MiniMax-M3 r4.8 recovery qualification

- Protocol version: 1
- State: `READY_FOR_EXECUTOR`
- Packet revision: `2026-07-16-r4.8`
- Planner owner: Codex planner
- Intended executor: any authorized cluster executor
- Base Git commit: `340a43f9817d5232cb016142c3d0655d192a47ad`
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
fresh r4.8 replay paths under the paired root and one fresh r4.8 BF16 root.
Unrelated untracked files outside those exact paths are record-and-proceed only
when they do not shadow a tracked path, any required input, or either fresh
output path; enumerate them in the return. Any collision is a stop condition.

## Setup, revision verification, and CPU suite

Run this block from one persistent login shell. Every later command uses the
variables from this shell.

```bash
cd /mnt/nfs/hoangduy/projects/llm-compressor
git switch duy-branch
git pull --ff-only
git merge-base --is-ancestor 340a43f9817d5232cb016142c3d0655d192a47ad HEAD
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

Stop before allocation if revision verification, the clean tracked-workspace
checks, environment activation, outside-allocation assertion, package checks,
or any CPU test fails. Record `git rev-parse HEAD` as the actual executed
commit. Do not clean or delete unrelated files.

## Fresh paths and collision gate

```bash
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
REPLAY_SESSION="m3-gptq-r48-replay-$(date -u +%H%M%S)"
BF16_SMOKE_SESSION="m3-bf16-r48-smoke-$(date -u +%H%M%S)"
BF16_OUT="$BF16_ROOT/logs/smoke-controller-r48.out"
BF16_ERR="$BF16_ROOT/logs/smoke-controller-r48.err"
BF16_RC="$BF16_ROOT/smoke-controller-r48.rc"

test -z "${SLURM_JOB_ID:-}"
test ! -e "$BF16_OUT"
test ! -e "$BF16_ERR"
test ! -e "$BF16_RC"
! tmux has-session -t "$REPLAY_SESSION" 2>/dev/null
! tmux has-session -t "$BF16_SMOKE_SESSION" 2>/dev/null

tmux new-session -d -s "$REPLAY_SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD' && srun --exclusive --nodes=1 --ntasks=1 --gpus-per-node=8 --kill-on-bad-exit=1 --time=12:00:00 python -m pipeline.m3_empty_output_replay --config pipeline/configs/eval_minimax_m3_reasoning_r4.yaml --model /mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay --samples '$PAIR_ROOT/models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl' --attempt-uid 8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878 --out '$REPLAY_JSON' >'$REPLAY_OUT' 2>'$REPLAY_ERR'; printf '%s\n' \$? >'$REPLAY_RC'"

tmux new-session -d -s "$BF16_SMOKE_SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD' && M3_PLACEMENT_TIMEOUT_SECONDS=900 M3_MODEL_INIT_TIMEOUT_SECONDS=10800 TIME_LIMIT=12:00:00 bash pipeline/slurm/run_m3_quality_eval_srun.sh --profile smoke --matrix '$BF16_MATRIX' --run-root '$BF16_ROOT' >'$BF16_OUT' 2>'$BF16_ERR'; printf '%s\n' \$? >'$BF16_RC'"

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

Classify the factual first boundary as exactly one of: `placement`,
`worker start`, `weight load`, `collective communication`, `request generation`,
or `evaluation`. Explain missing artifacts proportionally to how far execution
progressed. Do not infer a strategic verdict.

## Final instruction

Return in `RETURNED_FOR_ANALYSIS`, commit and push the complete evidence packet,
and stop. Do not retry, patch, fall back, launch paired work, or launch BF16
production. Only the planner may interpret the evidence and authorize a later
packet.
