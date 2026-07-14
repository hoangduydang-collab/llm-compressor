# MiniMax-M3 Task-Isolated Paired Quality Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a copy-ready, six-node-capped Slurm array run that evaluates 100 paired samples for each MiniMax-M3 quality task in isolated AWQ and GPTQ arms.

**Architecture:** Add a backward-compatible scheduling contract to the quality matrix, then materialize twelve immutable model/shard arms from a new six-shard matrix. A submission script maps those arms onto a Slurm array capped at six concurrent single-node jobs, while the existing arm runner and merger gain the minimal probe-only and completion-integrity behavior needed by the new topology.

**Tech Stack:** Python 3.11, dataclasses, PyYAML, pytest, Bash, Slurm `sbatch` arrays, existing EvalSuite and MiniMax-M3 quality tooling.

## Global Constraints

- Keep `pipeline/configs/minimax_m3_paired_gptq_awq_quick.yaml` unchanged.
- Keep exactly 100 paired samples per quality task with seed 42.
- Use six shards per model: GPQA, IFEval, AIME, MMLU-Pro, GSM8K, and one probe-only shard.
- Cap production at six concurrent exclusive one-node, 8xH100 arms.
- Give each production array task an independent `08:00:00` time limit.
- Preserve the existing 16k production generation limit and 8,192-token probe.
- Do not add automatic retries or authorize downstream performance work.
- Follow `PLANNER_EXECUTOR_PROTOCOL.md`; historical executor evidence is immutable.
- Run shell tests on Windows with `C:\Program Files\Git\bin` prepended to `PATH`.

---

### Task 1: Task-isolated matrix and scheduling contract

**Files:**
- Create: `pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml`
- Modify: `pipeline/m3_quality_eval.py`
- Test: `pipeline/tests/test_m3_quality_eval.py`

**Interfaces:**
- Consumes: existing `MatrixSpec`, `ShardSpec`, `load_matrix()`, and `build_launch_plan()`.
- Produces: `SchedulingSpec(max_parallel_arms: int | None, arm_time_limit: str | None)`, `MatrixSpec.scheduling`, and launch-plan fields `max_parallel_arms`, `max_concurrent_nodes`, and `arm_time_limit`.

- [ ] **Step 1: Write failing matrix and validation tests**

Add tests that load the new matrix and assert:

```python
TASK_ISOLATED_MATRIX = Path(
    "pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml"
)

def test_task_isolated_matrix_has_twelve_arms_and_six_node_cap(tmp_path):
    spec = load_matrix(TASK_ISOLATED_MATRIX)
    assert [shard.name for shard in spec.shards] == [
        "gpqa_diamond", "ifeval", "aime_2025", "mmlu_pro", "gsm8k",
        "distributional_probe",
    ]
    assert [shard.tasks for shard in spec.shards] == [
        ("gpqa_diamond",), ("ifeval",), ("aime_2025",),
        ("mmlu_pro",), ("gsm8k",), (),
    ]
    assert [shard.distributional_probe for shard in spec.shards] == [
        False, False, False, False, False, True,
    ]
    assert spec.sampling["production_samples_per_task"] == 100
    assert spec.scheduling.max_parallel_arms == 6
    assert spec.scheduling.arm_time_limit == "08:00:00"

    gate = tmp_path / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": True}))
    plan = build_launch_plan(spec, profile="production", smoke_gate=gate)
    assert len(plan["arms"]) == 12
    assert plan["total_nodes"] == 12
    assert plan["max_parallel_arms"] == 6
    assert plan["max_concurrent_nodes"] == 6
    assert plan["arm_time_limit"] == "08:00:00"
```

Also write temporary YAML tests proving a shard with neither tasks nor probe,
zero concurrency, and a malformed time limit are rejected with specific
`ValueError` messages.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest pipeline/tests/test_m3_quality_eval.py -k "task_isolated or empty_shard or scheduling" -q
```

Expected: failures because the new matrix, `SchedulingSpec`, validation, and
launch-plan metadata do not exist.

- [ ] **Step 3: Implement the minimal scheduling schema and matrix**

Add:

```python
@dataclass(frozen=True)
class SchedulingSpec:
    max_parallel_arms: int | None = None
    arm_time_limit: str | None = None
```

Parse `scheduling` in `load_matrix()`. Reject shards where both `tasks` is empty
and `distributional_probe` is false; reject non-positive concurrency; validate
non-null time limits with `re.fullmatch(r"\d{2}:\d{2}:\d{2}", value)`.
Old matrices omit `scheduling` and retain `max_parallel_arms == len(arms)`.

For production launch plans compute:

```python
configured = spec.scheduling.max_parallel_arms or len(arms)
max_parallel_arms = min(configured, len(arms))
max_concurrent_nodes = sum(
    sorted((arm["nodes"] for arm in arms), reverse=True)[:max_parallel_arms]
)
```

Smoke plans remain fully parallel. Add the new YAML by copying model, alias,
sampling, probe, and gate semantics from the historical quick matrix, replacing
only shard layout and adding:

```yaml
scheduling:
  max_parallel_arms: 6
  arm_time_limit: "08:00:00"
```

- [ ] **Step 4: Run focused and compatibility tests**

Run:

```powershell
python -m pytest pipeline/tests/test_m3_quality_eval.py -q
```

Expected: all tests pass, including the historical matrix expectations.

- [ ] **Step 5: Commit Task 1**

```powershell
git add pipeline/m3_quality_eval.py pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml pipeline/tests/test_m3_quality_eval.py
git commit -m "feat: add task-isolated M3 quality matrix"
```

---

### Task 2: Completion integrity and probe-only merge

**Files:**
- Modify: `pipeline/m3_quality_eval.py`
- Modify: `pipeline/tests/test_m3_quality_eval.py`

**Interfaces:**
- Consumes: `validate_and_merge(root)` and per-arm `arm_complete.json`.
- Produces: infrastructure failure entries for false completion markers and successful merging of an empty aggregate carrying the model's sole probe.

- [ ] **Step 1: Write failing completion and probe-only merge tests**

Extend `_write_arm()` to accept `task: str | None` and `complete: bool = True`.
When `task is None`, write `{}` to `aggregate.json`, create no sample file, and
write a one-line `distributional_probe.jsonl` record.

Add one test that sets `complete=False` and asserts:

```python
result = validate_and_merge(tmp_path)
assert result["infrastructure_ok"] is False
assert result["failures"] == [{
    "arm": "quant/reasoning", "arm_complete": False,
}]
```

Add another manifest with two shards per model (`gpqa_diamond` and
`distributional_probe`), write one task arm plus one probe-only arm per model,
and assert merge succeeds, the task comparison exists, and both merged model
directories contain `distributional_probe.jsonl`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest pipeline/tests/test_m3_quality_eval.py -k "completion_marker or probe_only" -q
```

Expected: false completion is incorrectly accepted before the implementation.

- [ ] **Step 3: Validate completion truthfully**

After reading `return_code.txt`, load `arm_complete.json` and append:

```python
if _read_json(arm / "arm_complete.json").get("complete") is not True:
    failures.append({"arm": f"{model}/{shard}", "arm_complete": False})
    continue
```

Do not change `_merge_model_arms()` empty-dictionary behavior; the test records
that its existing disjoint merge semantics are the intended contract.

- [ ] **Step 4: Run the focused file**

Run:

```powershell
python -m pytest pipeline/tests/test_m3_quality_eval.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add pipeline/m3_quality_eval.py pipeline/tests/test_m3_quality_eval.py
git commit -m "fix: validate M3 quality arm completion"
```

---

### Task 3: Probe-only arm execution

**Files:**
- Modify: `pipeline/slurm/test_m3_quality_eval_arm.sh`
- Modify: `pipeline/tests/test_m3_quality_eval_runner.py`

**Interfaces:**
- Consumes: `--tasks ""`, `--run-probe 1`, and the existing arm output contract.
- Produces: no EvalSuite invocation, `{}` in `aggregate.json`, one probe invocation, `return_code.txt=0`, and `arm_complete.json.complete=true`.

- [ ] **Step 1: Write a failing behavioral shell-runner test**

Create a temporary executable named `python` earlier on `PATH`. Its Bash body
records an error if called with `-m pipeline.evalsuite.cli`, creates probe output
when called with `-m pipeline.m3_distributional_probe`, and handles the two
`python -` inline-writer calls by writing the requested manifest/completion JSON.
Run `test_m3_quality_eval_arm.sh` with an empty task list and probe enabled.

Assert:

```python
assert result.returncode == 0, result.stderr
assert not (tmp_path / "evalsuite-called").exists()
assert json.loads((arm / "aggregate.json").read_text()) == {}
assert (arm / "distributional_probe.jsonl").is_file()
assert (arm / "return_code.txt").read_text().strip() == "0"
assert json.loads((arm / "arm_complete.json").read_text())["complete"] is True
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
$env:PATH="C:\Program Files\Git\bin;$env:PATH"
python -m pytest pipeline/tests/test_m3_quality_eval_runner.py -k probe_only -q
```

Expected: failure because the current runner invokes EvalSuite with empty tasks.

- [ ] **Step 3: Skip EvalSuite only for the explicit empty task list**

Replace the unconditional eval block with:

```bash
if ((rc == 0)); then
  if [[ -n "$TASKS" ]]; then
    "${eval_cmd[@]}"
    rc=$?
  else
    printf '{}\n' >"$ARM/aggregate.json"
  fi
fi
```

Leave probe ordering, failure propagation, manifests, and completion writers
unchanged.

- [ ] **Step 4: Run all shell-runner contract tests**

Run:

```powershell
$env:PATH="C:\Program Files\Git\bin;$env:PATH"
python -m pytest pipeline/tests/test_m3_quality_eval_runner.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add pipeline/slurm/test_m3_quality_eval_arm.sh pipeline/tests/test_m3_quality_eval_runner.py
git commit -m "feat: support probe-only M3 quality arms"
```

---

### Task 4: Six-concurrent-arm Slurm array launcher

**Files:**
- Create: `pipeline/slurm/submit_m3_quality_eval_array.sh`
- Create: `pipeline/slurm/run_m3_quality_eval_array_arm.sh`
- Create: `pipeline/tests/test_m3_quality_eval_array_runner.py`
- Modify: `pipeline/tests/test_m3_quality_eval_runner.py`

**Interfaces:**
- Consumes: production launch-plan JSON, `SLURM_ARRAY_TASK_ID`, resolved task aliases, and the existing arm runner CLI.
- Produces: deterministic index mapping, a dry-run `sbatch` command with `0-11%6`, one-node/8-GPU exclusivity, `08:00:00`, and an actual submission ID in `array_job_id.txt`.

- [ ] **Step 1: Write failing dry-run and index-selection tests**

The submission test creates a passing smoke gate and invokes:

```python
result = subprocess.run([
    "bash", "pipeline/slurm/submit_m3_quality_eval_array.sh",
    "--matrix", str(TASK_ISOLATED_MATRIX),
    "--run-root", str(tmp_path),
    "--smoke-gate", str(gate), "--dry-run",
], text=True, capture_output=True)
```

Assert return code zero, twelve `index=` mapping lines, and command fragments
`--array=0-11%6`, `--nodes=1`, `--gpus-per-node=8`, `--exclusive`, and
`--time=08:00:00`. Assert no `array_job_id.txt` exists in dry-run mode.

The array-arm test writes a two-entry launch plan and resolved alias file, sets
`SLURM_ARRAY_TASK_ID=1`, substitutes a recording arm runner via
`M3_QUALITY_ARM_RUNNER`, and asserts only entry 1's model, shard, task, topology,
probe flag, and probe-token arguments were forwarded. Add out-of-range and
non-single-node rejection tests.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PATH="C:\Program Files\Git\bin;$env:PATH"
python -m pytest pipeline/tests/test_m3_quality_eval_array_runner.py -q
```

Expected: failure because both launcher scripts are absent.

- [ ] **Step 3: Implement the array arm runner**

`run_m3_quality_eval_array_arm.sh` parses `--plan`, `--run-root`, and `--matrix`,
requires numeric `SLURM_ARRAY_TASK_ID`, reads exactly that JSON entry, resolves
task aliases from `$RUN_ROOT/preflight/resolved_tasks.json`, rejects any arm not
using one node and eight GPUs, and `exec`s `${M3_QUALITY_ARM_RUNNER:-pipeline/slurm/test_m3_quality_eval_arm.sh}` with the complete production arm arguments.

- [ ] **Step 4: Implement the submission wrapper**

`submit_m3_quality_eval_array.sh` parses `--matrix`, `--run-root`,
`--smoke-gate`, and `--dry-run`; requires the preflight-populated run root but
refuses an existing production launch plan or `models/` arm-output tree; runs
`python -m pipeline.m3_quality_eval launch-plan`; prints a stable mapping; and
constructs:

```bash
sbatch --parsable --array="0-$((arm_count - 1))%${max_parallel}" \
  --nodes=1 --ntasks=1 --gpus-per-node=8 --exclusive \
  --time="$arm_time_limit" \
  --output="$RUN_ROOT/logs/production-%A_%a.out" \
  --error="$RUN_ROOT/logs/production-%A_%a.err" \
  pipeline/slurm/run_m3_quality_eval_array_arm.sh \
  --plan "$PLAN" --run-root "$RUN_ROOT" --matrix "$MATRIX"
```

Dry-run prints the shell-escaped command and exits. Actual submission writes the
exact command and parsable job ID under the run root.

- [ ] **Step 5: Run launcher and syntax tests**

Run:

```powershell
$env:PATH="C:\Program Files\Git\bin;$env:PATH"
python -m pytest pipeline/tests/test_m3_quality_eval_array_runner.py pipeline/tests/test_m3_quality_eval_runner.py -q
```

Expected: all tests pass, including `bash -n` for both new scripts.

- [ ] **Step 6: Commit Task 4**

```powershell
git add pipeline/slurm/submit_m3_quality_eval_array.sh pipeline/slurm/run_m3_quality_eval_array_arm.sh pipeline/tests/test_m3_quality_eval_array_runner.py pipeline/tests/test_m3_quality_eval_runner.py
git commit -m "feat: launch M3 quality arms as capped array"
```

---

### Task 5: Planner execution packet and complete verification

**Files:**
- Create: `M3_PAIRED_GPTQ_AWQ_TASK_ISOLATED_HANDOFF.md`
- Modify: `docs/superpowers/plans/2026-07-14-m3-task-isolated-paired-quality-rerun.md`

**Interfaces:**
- Consumes: the implementation commit, prior passing smoke gate, new matrix, launcher, aggregator, and `PLANNER_EXECUTOR_PROTOCOL.md`.
- Produces: one `READY_FOR_EXECUTOR` packet with copy-ready cluster commands and an immutable planner correction to the historical GPTQ broad-arm account.

- [ ] **Step 1: Record the implementation commit and author the packet**

After Tasks 1-4 are committed, record that full SHA as `Base Git commit`. The
packet must include protocol version/state/revision, one decision question,
exact cluster paths/environment, the prior smoke gate and hash-reuse checks,
fresh UTC run ID, preflight, dry-run, submission, monitoring, `sacct`, after-any
aggregation, packaging, no-retry/stop rules, twelve expected arms, six-node cap,
8-hour arm limit, and the full evidence return contract.

The planner addendum must state that raw logs show GPTQ broad loaded in about
296 seconds and reached 96/100 MMLU-Pro prompts; this corrects the prose account
without editing the historical executor packet.

- [ ] **Step 2: Run full local verification**

Run:

```powershell
$env:PATH="C:\Program Files\Git\bin;$env:PATH"
python -m pytest pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py pipeline/tests/test_m3_quality_eval_array_runner.py -q
git diff --check
rg -n "TODO|TBD|<[^>]+>" M3_PAIRED_GPTQ_AWQ_TASK_ISOLATED_HANDOFF.md pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml pipeline/slurm/submit_m3_quality_eval_array.sh pipeline/slurm/run_m3_quality_eval_array_arm.sh
```

Expected: all tests pass, `git diff --check` is silent, and the placeholder scan
has no matches.

- [ ] **Step 3: Review requirements against the approved design**

Confirm from generated dry-run output and files that there are exactly twelve
arms, five 100-sample tasks per model, one 8,192-token probe per model, `%6`
concurrency, one node/eight GPUs per arm, independent eight-hour limits, no
retry, partial-failure evidence, and no downstream authorization.

- [ ] **Step 4: Mark this plan complete and commit the packet**

Change every checkbox in this plan to `[x]`, then run:

```powershell
git add M3_PAIRED_GPTQ_AWQ_TASK_ISOLATED_HANDOFF.md docs/superpowers/plans/2026-07-14-m3-task-isolated-paired-quality-rerun.md
git commit -m "docs: hand off task-isolated M3 quality rerun"
```

- [ ] **Step 5: Verify final repository state**

Run:

```powershell
git status --short
git log -6 --oneline
```

Expected: clean working tree with the design, plan, implementation, tests, and
execution packet committed on `duy-branch`.
