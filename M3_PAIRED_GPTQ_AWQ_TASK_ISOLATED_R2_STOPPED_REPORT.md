# Evidence packet: task-isolated paired quality rerun (r2)

- Protocol version: 1
- State: `RETURNED_FOR_ANALYSIS`
- Packet revision executed: `2026-07-14-r2`
- Expected base commit: `1e2da818a08df09bf1ff0268702bda34bf89ee6e`
- Actual Git revision: `ea53ec3a4981d641749f8fd3635dc40981fe0262`
- Execution classification: `stopped_before_allocation`
- Decision question: Does repaired in-house GPTQ preserve enough quality versus
  cyankiwi AWQ to justify later performance evaluation?

## Factual outcome

The revised workspace condition passed: all Git-visible untracked paths were
under `results/` or `artifacts/`, with no tracked staged or unstaged changes.
The following packet stages passed:

- environment and ancestor checks;
- production preflight;
- smoke-gate reuse check (`reusable: true`);
- twelve-arm dry-run with `--array=0-11%6` and `--time=08:00:00`.

No GPU allocation or evaluation arm started. The actual `sbatch` submission was
rejected by Slurm:

```text
sbatch: error: Batch job submission failed: I/O error writing script/environment to file
```

No `array_job_id.txt` was created, and the scheduler had no jobs for this
packet when checked:

```text
squeue -u hoangduy -o '%.18i %.12T %.10M %.24j %.14R'
             JOBID        STATE       TIME                     NAME NODELIST(REASON)
```

## Run and evidence paths

- Run root:
  `results/m3-quality/20260714T145900Z-m3-paired-gptq-awq-task-isolated-r2`
- Preflight and manifest artifacts are under the run root.
- Smoke reuse result:
  `smoke_reuse_check.json` reports all six checks true.
- Dry-run output:
  `array_dry_run.log` records all twelve expected arms and the required array
  concurrency/time-limit arguments.
- Submission output:
  `submission.log` records the exact Slurm command and I/O error.
- Submission command:
  `submission_command.txt`
- No per-arm records, scheduler stdout/stderr, final `sacct`, or aggregate
  quality artifacts exist because allocation never occurred.

## Exact launch contract

The attempted submission used:

```text
pipeline/slurm/submit_m3_quality_eval_array.sh --matrix pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml --run-root results/m3-quality/20260714T145900Z-m3-paired-gptq-awq-task-isolated-r2 --smoke-gate results/m3-quality/20260714T064000Z-m3-paired-gptq-awq-quick/smoke_gate.json
```

The validated dry-run command was:

```text
sbatch --parsable --array=0-11%6 --nodes=1 --ntasks=1 --gpus-per-node=8 --exclusive --time=08:00:00 --output=results/m3-quality/20260714T145900Z-m3-paired-gptq-awq-task-isolated-r2/logs/production-%A_%a.out --error=results/m3-quality/20260714T145900Z-m3-paired-gptq-awq-task-isolated-r2/logs/production-%A_%a.err pipeline/slurm/run_m3_quality_eval_array_arm.sh --plan results/m3-quality/20260714T145900Z-m3-paired-gptq-awq-task-isolated-r2/production_launch_plan.json --run-root results/m3-quality/20260714T145900Z-m3-paired-gptq-awq-task-isolated-r2 --matrix pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml
```

The final line above is reproduced from the launcher output; the durable
`submission.log` is authoritative for the actual command and error.

## Deviations

The executor wrapper initially stopped on a local filename typo while
constructing the smoke-reuse comparison. It was corrected before the
comparison was rerun. This did not launch jobs or alter packet inputs,
parameters, topology, or time limits.

After the Slurm I/O rejection, no retry, parameter change, result-root reuse,
cleanup, cancellation, or alternative launch was attempted.

## Limited executor interpretation

The failure occurred during Slurm submission, before GPU allocation and before
any model loading or evaluation. It is therefore not evidence about GPTQ/AWQ
quality. The planner must decide whether and how to address the scheduler-side
I/O failure.
