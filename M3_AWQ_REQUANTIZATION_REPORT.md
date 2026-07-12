# MiniMax-M3 AWQ Re-quantization Progress Report

## Status

Both AWQ repair variants were started, but neither completed. The runs were
terminated by Slurm step timeout while still performing expert smoothing. No
complete checkpoint was produced.

## Runs

| Variant | Output directory | Node | Result |
| --- | --- | --- | --- |
| Offset-norm repair (`awq-offsetfix`) | `artifacts/m3-awq-gptq-prepared/quantized-offsetfix/MiniMax-M3-awq-W4AFP8/20260712-041135` | `gpu-h128` | Incomplete; terminated at 07:51:35 UTC |
| No-smoothing control (`awq-nosmooth`) | `artifacts/m3-awq-gptq-prepared/quantized-nosmooth/MiniMax-M3-awq-W4AFP8/20260712-041135` | `gpu-h127` | Incomplete; terminated at 07:51:40 UTC |

Logs:

- `/mnt/nfs/hoangduy/logs/m3-awq-gptq-prepare/awq-offsetfix.log`
- `/mnt/nfs/hoangduy/logs/m3-awq-gptq-prepare/awq-nosmooth.log`

## Evidence

The offset-fix run reached approximately 59% of smoothing. Its log ends with:

```text
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
slurmstepd-gpu-h128: error: *** STEP 12769.0 ON gpu-h128 CANCELLED AT 2026-07-12T07:51:35 ***
srun: error: Timed out waiting for job step to complete
```

The no-smoothing run reached approximately 88% of smoothing. Its log ends
with:

```text
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
slurmstepd-gpu-h127: error: *** STEP 12768.0 ON gpu-h127 CANCELLED AT 2026-07-12T07:51:40 ***
srun: error: Timed out waiting for job step to complete
```

There is no Python traceback or quantization-specific exception at the end of
either log. The failure is infrastructure/time-limit related, not evidence
that the offset-norm repair or no-smoothing configuration is incorrect.

## Important handling

Do not serve, publish, or analyze either output directory as a valid
checkpoint. They may contain partial files from the interrupted export.
Treat both variants as failed preparation attempts and rerun them with a
wall-time limit large enough to finish smoothing and checkpoint export.

The GPTQ portable preparation is separate and is not affected by these AWQ
timeouts:

`artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123`

## Retry attempt and cancellation analysis

The prescribed retry was launched from a tool-managed background shell on
2026-07-12 at 08:12 UTC with:

```text
test -z "${SLURM_JOB_ID:-}" || { echo "leave parent allocation first"; exit 1; }
unset SRUN_ARGS
TIME_LIMIT=96:00:00 bash pipeline/slurm/run_m3_awq_gptq_prepare_srun.sh
```

The launcher preflight passed (`23 passed`). Slurm reported both allocations
with approximately 3 days 22 hours remaining at the 10:08 UTC progress check,
so the configured 96-hour limit had not expired. Nevertheless, both steps were
cancelled together at approximately 14:30 UTC:

- `awq-offsetfix`: `gpu-h122`, step `12778.0`; smoothing reached 100% and the
  run had entered layer 12 before cancellation.
- `awq-nosmooth`: `gpu-h121`, step `12777.0`; the run was at layer 15 and 31%
  of its 128 smoothing targets.

Both logs end with `srun: error: Timed out waiting for job step to complete`.
No `scancel`, `kill`, or other termination command was issued by the agent.
The launcher process was later absent, and Slurm accounting no longer returned
records for these completed job IDs, so the exact cancellation source cannot
be proven from the available evidence. Because the jobs were started from a
tool-managed background shell, launcher/session teardown remains a plausible
contributor and this launch method should not be reused for the next attempt.

## Recommended next action

Rerun both AWQ variants unchanged from a persistent login-shell context or a
proper detached/batch allocation that survives the controlling session.
Capture `sacct`/`scontrol` metadata before records expire, along with
stdout/stderr and a completion marker. Before serving, verify that each output
contains a complete `config.json`, `model.safetensors.index.json`, all
referenced shards, and a successful quantization completion record. Only then
launch the AWQ/GPTQ serving matrix.
