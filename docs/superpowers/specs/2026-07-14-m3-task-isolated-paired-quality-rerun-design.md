# MiniMax-M3 Task-Isolated Paired Quality Rerun Design

**Date:** 2026-07-14
**Scope:** MiniMax-M3 paired GPTQ-versus-AWQ quick quality evaluation
**Status:** Approved for implementation (r3 amendment below supersedes conflicting r2 text)
**Workflow state:** `PLANNER_ANALYSIS`

## Approved r3 amendment: grouped `srun` execution

The executor cluster supports top-level `srun` allocations, not `sbatch`. Replace
the twelve task-isolated array arms with six independent one-node arms, all
launched concurrently from a detached `tmux` controller outside any existing
Slurm allocation:

| Node arm | Ordered work |
| --- | --- |
| `inhouse_gptq/reasoning` | GPQA, then IFEval |
| `inhouse_gptq/broad_math` | MMLU-Pro, then GSM8K, then AIME 2025 |
| `inhouse_gptq/distributional_probe` | 8,192-token probe |
| `cyankiwi_awq/reasoning` | GPQA, then IFEval |
| `cyankiwi_awq/broad_math` | resume saved MMLU-Pro and GSM8K, then AIME 2025 |
| `cyankiwi_awq/distributional_probe` | 8,192-token probe |

Each EvalSuite arm loads its model once and checkpoints after every task. Each
top-level `srun` requests one exclusive 8xH100 node and has an independent
`16:00:00` ceiling. The ceiling is safety headroom, not expected runtime. One
arm's failure must not cancel siblings, and no retry is automatic.

The prior production run saved four complete task checkpoints: GPTQ GPQA
(100/100, 0.28), AWQ GPQA (100/100, 0.24), AWQ MMLU-Pro (100/100, 0.76), and
AWQ GSM8K (100/100, 0.97). Before launch, the r3 packet must verify identical
old/new production manifests, each task aggregate, and 100 unique sample UIDs,
then import only those task aggregates, samples, and generation-health records
into the matching new arm. Existing EvalSuite resume behavior skips them.

GPTQ MMLU-Pro reached only 96/100 and wrote no completed task checkpoint, so it
must rerun. Neither IFEval, AIME, GPTQ GSM8K, nor either distributional probe
completed. Missing or invalid source artifacts are a stop-and-return condition;
a score found only in a log is not a substitute.

The old GSM8K evidence also exposed a normalization defect: 200 stored rows but
only 100 unique stable sample UIDs. Checkpoint writing must collapse identical
rows with the same UID and reject conflicting rows with the same UID. Subtask
namespaces remain part of the UID, preserving legitimate group-task rows.

Implementation reuses `pipeline/slurm/run_m3_quality_eval_srun.sh`, removes the
unusable `sbatch` array scripts, and records the cluster constraint in durable
planner guidance. Acceptance is six dry-run `srun` commands, no `sbatch`, six
nodes maximum, 100 paired samples per task (30 for the 30-item AIME dataset),
one probe per model, validated AWQ MMLU reuse, and protocol-compliant evidence.

Before GPU launch, r3 also writes a fail-closed harness-contract artifact. It
pins lm-eval 0.4.12, resolved task aliases/metrics/few-shot counts, the official
base tokenizer and chat-template hashes, requires both served checkpoint
tokenizers plus a rendered default prompt to match that official source,
adaptive thinking with `</mm:think>`,
greedy 16k generation, and the TP8 vLLM/EP serving contract. This makes the
paired result reflective of the model under a stable standard harness. The
100-sample subset remains a directional paired comparison, not a directly
comparable reproduction of full public leaderboard scores; MiniMax does not
publish an identical five-task recipe, and its model card recommends sampling
parameters for general inference rather than this deterministic benchmark run.

## Decision to make

Determine whether the repaired in-house GPTQ checkpoint preserves enough quality
relative to the cyankiwi AWQ baseline to justify further performance work.

The prior quick rerun did not answer this question. Its four production arms
shared several tasks per model and hit three-hour scheduler limits after producing
only partial results. Raw logs also show that the GPTQ broad arm loaded
successfully and reached 96/100 MMLU-Pro prompts, contrary to the historical
executor report's statement that it timed out during startup. The historical
evidence remains immutable; the correction will be recorded in the new planner
handoff and analysis rather than by editing the returned packet.

## Goals

1. Keep exactly 100 paired samples for every quality task.
2. Isolate each model/task combination so one slow task cannot discard another
   task's completed work.
3. Run no more than six 8xH100 nodes concurrently, leaving approximately five of
   the currently idle nodes available for other work.
4. Give every GPU arm an independent eight-hour scheduler limit.
5. Preserve the existing paired sample manifest, evaluation semantics, 16k
   production generation ceiling, and 8,192-token distributional probe.
6. Make execution copy-ready for the executor and return protocol-compliant raw
   evidence even when one or more arms fail.

## Non-goals

- Changing models, prompts, task aliases, sample seed, sample count, evaluation
  thresholds, or serving topology.
- Treating the 100-sample quick matrix as the final definitive quality verdict.
- Automatically retrying failed or timed-out arms.
- Running speed benchmarks, publishing a result, or adopting GPTQ downstream.
- Rewriting or deleting the pre-protocol evidence packet.

## Chosen architecture

### Preserve the historical matrix

`pipeline/configs/minimax_m3_paired_gptq_awq_quick.yaml` remains unchanged so
historical runs stay reproducible. A new task-isolated quick matrix will contain
six shards:

| Shard | Tasks | Probe |
| --- | --- | --- |
| `gpqa_diamond` | `gpqa_diamond` | no |
| `ifeval` | `ifeval` | no |
| `aime_2025` | `aime_2025` | no |
| `mmlu_pro` | `mmlu_pro` | no |
| `gsm8k` | `gsm8k` | no |
| `distributional_probe` | none | yes |

With two models, the production launch plan contains twelve independent
single-node arms. Each quality sample is selected once per model from the same
seeded manifest, retaining exact paired comparisons.

The new matrix also carries an explicit operational scheduling contract:

```yaml
scheduling:
  max_parallel_arms: 6
  arm_time_limit: "08:00:00"
```

The parsed matrix and generated launch plan expose these values together with
`max_concurrent_nodes: 6`. `total_nodes: 12` continues to describe the sum of
all arm requirements, not a simultaneous reservation.

### Slurm array with a six-arm concurrency cap

The launcher materializes the existing JSON launch plan in the fresh run root,
then submits its twelve production entries as a Slurm array equivalent to:

```bash
sbatch --array=0-11%6 --nodes=1 --ntasks=1 --gpus-per-node=8 \
  --exclusive --time=08:00:00 \
  pipeline/slurm/run_m3_quality_eval_array_arm.sh \
  --plan "$RUN_ROOT/production_launch_plan.json" \
  --run-root "$RUN_ROOT" --matrix "$MATRIX"
```

Each array index selects exactly one immutable launch-plan entry and invokes the
existing arm runner. Slurm may schedule entries in any order; simultaneous
execution is not part of the paired statistical contract. At most six arms may
run at once, and a completed array task releases its node immediately instead of
holding a six-node parent allocation while slower arms finish.

The array job replaces the current production use of a single allocation that
starts every `srun` concurrently. Existing smoke and legacy launch behavior
remain available. The new launcher must print the resolved array mapping and
submission command in dry-run mode before it is allowed to submit work.

The launcher derives the `%6` limit and eight-hour per-arm time from the
committed matrix, while the execution packet repeats and verifies them. They are
not executor-selected overrides. Changing either value requires a new packet
revision.

### Probe-only arm

The current arm runner always invokes EvalSuite, which is invalid for an empty
task list. For the probe-only shard, the runner will:

1. write the normal arm manifest;
2. skip EvalSuite when the resolved task list is empty;
3. write a valid empty `aggregate.json` (`{}`);
4. run the distributional probe once;
5. write `return_code.txt` and `arm_complete.json` normally.

A shard is valid only when it contains at least one task or enables the probe.
This rule prevents accidental no-op arms while making the deliberate probe-only
arm explicit.

## Data and artifact flow

```text
new matrix + prior passing smoke gate
             |
             v
preflight hashes and seeded 100-sample manifest
             |
             v
12-entry immutable production_launch_plan.json
             |
             v
Slurm array 0-11%6 -> one model/shard arm per array task
             |
             v
per-arm manifest, raw logs, aggregate/probe, return code, completion marker
             |
             v
existing model-arm merge -> matrix.json -> gates.json -> report.md
             |
             v
protocol-compliant executor evidence packet
```

The existing `_merge_model_arms` behavior is retained: it combines disjoint task
aggregates and sample records and copies exactly one distributional probe per
model. The merger must accept the probe-only arm's empty aggregate, but still
reject duplicate task results or multiple probe artifacts for one model.

## Smoke-gate reuse

The previous passing smoke result may be reused because running another smoke is
not scientifically useful when evaluation inputs and serving semantics are
unchanged. Reuse is authorized only after a preflight records and verifies exact
equality of:

- sample-manifest hash;
- resolved evaluation-config hash;
- tokenizer hash;
- chat-template hash;
- model paths and model kinds;
- serving topology and relevant vLLM overrides.

The matrix hash is expected to differ because shard layout changes and is not a
reuse equality requirement. The execution packet names the exact prior
`smoke_gate.json` and its source run. A missing field, mismatch, non-passing gate,
or unverifiable source is a stop-and-return condition; the executor does not
choose a substitute smoke artifact.

## Failure handling

- Array arms are independent. Failure, timeout, cancellation, or OOM in one arm
  must not cancel healthy arms.
- The array uses no automatic retries. A retry requires planner analysis, a new
  packet revision, and a fresh result root.
- Every arm writes to a unique model/shard directory. Prior and partial results
  are never overwritten.
- The eight-hour limit applies independently to each array task, not to time
  spent pending in the scheduler.
- Aggregation runs after the array reaches terminal state, including partial
  failure, so it can record missing or invalid arms and preserve a mechanical
  failure report.
- Any missing arm, nonzero return code, false completion marker, malformed
  artifact, provenance mismatch, or failed gate prevents a positive quality
  verdict. Infrastructure failure is reported separately from measured model
  quality.
- The executor captures `sacct`, scheduler stdout/stderr, exact array-to-arm
  mapping, job IDs, nodes, timestamps, elapsed times, states, exit codes, and
  failure signals before returning.

## Planner/executor boundary

The planner will commit a `READY_FOR_EXECUTOR` packet containing:

- the exact base commit and prior smoke-gate path;
- copy-ready preflight, dry-run, submission, monitoring, aggregation, evidence
  packaging, commit, and push commands;
- a fresh run-ID construction command;
- the exact twelve expected arms and six-arm concurrency cap;
- explicit stop conditions, no-retry policy, and prohibited downstream work;
- required small artifacts and path/size/SHA-256 records for large artifacts.

The executor only verifies, submits, monitors, aggregates, and returns evidence.
It does not select a different concurrency level, regroup tasks, extend time
limits, rerun an arm, change a gate, or interpret whether GPTQ should be adopted.

## Implementation surface

Expected implementation changes are deliberately bounded:

1. Add
   `pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml`.
2. Extend matrix validation and launch-plan metadata in
   `pipeline/m3_quality_eval.py` for valid probe-only shards, the explicit
   scheduling contract, and truthful completion-marker validation.
3. Extend `pipeline/slurm/test_m3_quality_eval_arm.sh` to support a probe-only
   arm without invoking EvalSuite.
4. Add `pipeline/slurm/submit_m3_quality_eval_array.sh` to materialize, print,
   dry-run, and submit the array, plus
   `pipeline/slurm/run_m3_quality_eval_array_arm.sh` to select one arm by
   launch-plan index and invoke the existing arm runner. Neither interface asks
   the executor to reconstruct commands.
5. Add focused CPU contract tests in the existing quality-evaluation test
   modules (and a new launcher test module if separation is clearer).
6. Add a planner-to-executor execution packet and planner evidence addendum;
   do not mutate the historical executor report.

## Validation plan

Automated tests will establish that:

- the new matrix loads with six unique, valid shards;
- every quality task appears exactly once per model;
- exactly one probe-only shard exists per model;
- the production plan contains twelve one-node, eight-GPU arms;
- the production submission caps concurrency at six and assigns eight hours per
  array task;
- a shard with neither tasks nor probe is rejected;
- the probe-only runner skips EvalSuite, writes `{}` to `aggregate.json`, runs
  the probe, and writes normal completion evidence;
- merging five task arms plus one probe-only arm per model succeeds;
- duplicate tasks, duplicate probes, incomplete arms, and bad return codes remain
  rejected;
- dry-run output deterministically maps all twelve array indices to arms without
  submitting work;
- shell scripts pass `bash -n` and the focused Python test suite passes.

The final implementation review also runs `git diff --check` and inspects the
generated launch-plan and dry-run commands for exact resource values.

## Acceptance criteria

- One executor command can submit the complete run without dynamic experiment
  design or manual task-to-node assignment.
- No more than six cluster nodes are consumed concurrently by the rerun.
- Each of the five tasks evaluates 100 identical paired samples on AWQ and GPTQ.
- Each model produces exactly one 8,192-token distributional probe.
- A slow or failed arm cannot erase completed evidence from another arm.
- Partial failure produces actionable raw scheduler and arm evidence and cannot
  be mistaken for a quality result.
- Successful aggregation produces the existing decision artifacts without
  changing quality thresholds or statistical semantics.
- Execution stops in `RETURNED_FOR_ANALYSIS`; only the planner decides whether
  the evidence supports further GPTQ performance work.
