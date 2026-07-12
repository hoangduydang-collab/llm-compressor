# MiniMax-M3 Three-Model Smoke Recovery Handoff

## Outcome

The final three-model smoke attempt did not produce a usable smoke gate.
Production evaluation was not launched.

Run root:

`results/m3-quality/20260712-102451-m3-quality-3model`

Models in scope:

- BF16 MiniMax-M3
- in-house GPTQ portable checkpoint
- cyankiwi AWQ reference

AutoRound was correctly deferred by the new handoff because its mixed-bit
loader requires external repository-specific integration.

## What was fixed and verified

The initial Ray preflight failed because compute hostnames such as `gpu-h113`
are not DNS-resolvable from the nodes. The topology script was updated to:

- derive a routable `10.2.x.x` address using `ip -4 route get`;
- fall back to `hostname -I`;
- share the head IP through the output directory;
- connect the gate to the explicit head IP;
- wait for the expected number of active Ray nodes and GPUs;
- retain rank-local logs and status files.

The evaluation venv was missing Ray. Installed:

```text
ray==2.56.0
msgpack==1.2.1
```

The topology gate then passed:

```json
{
  "ready": true,
  "expected_nodes": 2,
  "alive_nodes": 2,
  "visible_gpus": 16.0
}
```

The relevant preflight and Ray evidence is under:

- `ray_preflight/`
- `ray_debug6/`
- `models/bf16/shards/smoke/ray_runtime/`

Focused runner tests passed after the changes:

```text
4 passed
```

## Final smoke result

The launcher started all three arms with `srun` on four nodes. The Ray gate
passed and the BF16 arm connected to the two-node Ray cluster.

The two quantized arms failed during task evaluation with the same lm-eval
sample-index error:

```text
AssertionError: Elements of --samples should be in the interval [0,k-1]
where k is the number of total examples. In this case, k=717.
```

Affected logs:

- `logs/smoke-inhouse_gptq-smoke.err`
- `logs/smoke-cyankiwi_awq-smoke.err`

Both arms wrote `return_code.txt` containing `1`. They produced partial
artifacts for three tasks, but no complete comparable result; their
`smoke_evidence.json` files record zero probe tokens and zero probe time.
They must not be interpreted as model-quality failures because evaluation
aborted before the configured task set completed.

The BF16 arm remained active for about 35 minutes with no output after
connecting to Ray and remained pending all five tasks. It was cancelled as
job `12798` after producing no scores or samples. Its evidence shows:

- Ray connected at `10:47:22 UTC`;
- `completed_tasks: []`;
- all five tasks remained pending;
- no `arm_complete.json` was produced.

Relevant BF16 files:

- `logs/smoke-bf16-smoke.out`
- `logs/smoke-bf16-smoke.err`
- `models/bf16/shards/smoke/eval_meta.json`
- `models/bf16/shards/smoke/ray_runtime/`

## Diagnosis

There were three separate blockers:

1. Ray topology initially failed on hostname resolution.
2. Ray was absent from the active evaluation venv.
3. After both were fixed, lm-eval rejected the shared sample manifest for
   both quantized arms because at least one task received an index outside its
   actual dataset size (`k=717`).

The BF16 arm additionally appears to hang inside the first evaluation stage
after model/Ray initialization. This is separate from the quantized sample
index failure.

The smoke gate is therefore not a trustworthy quality result, and
`ready_for_production` must remain false. Do not launch production.

## Suggested next investigation

1. Trace how the sample manifest is converted into lm-eval `--samples` for
   each task. Validate indices against each task's resolved leaf size rather
   than applying one global sample index set to every task.
2. Add a preflight assertion that every task-specific sample index is within
   that task's split length, and emit the task name, split, size, and maximum
   index.
3. Re-run only a CPU/minimal task-request construction test before allocating
   GPUs.
4. Investigate BF16's first-task hang separately with a bounded single-task
   run and explicit progress logging around model initialization, request
   construction, and evaluation.
5. Preserve the current run root as evidence; do not overwrite it.

## Exact evidence contract

The pushed run root contains the preflight manifests, resolved evaluation
configuration, checkpoint diagnostics, launch plan, Ray rank logs/status,
per-arm manifests, return codes, partial evidence, and all smoke stdout/stderr.
No model checkpoint payloads are included.
