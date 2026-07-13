# MiniMax-M3 Production Quality Eval Handoff

Standalone handoff for launching the real (production-profile) three-model
quality comparison. This does not supersede or modify the discriminator
handoff (`M3_QUALITY_THREE_MODEL_SMOKE_RECOVERY_HANDOFF.md`); it is the next
phase, gated on that smoke work.

## Progress: smoke passed; production path fixed

The discriminator smoke objective is **met**. In-house GPTQ and cyankiwi AWQ
both completed the smoke profile with valid evidence (`infrastructure_ok`,
`artifacts_valid`, 5 tasks scored, 0 empty, 0 loops, TP8 world size, paired
2,047-token probe, identical sample/probe hashes). See
`M3_3MODEL_GPTQ_AWQ_FINAL_REPORT.md` and `M3_TMUX_SMOKE_FINAL_REPORT.md`.

The only remaining blocker was **BF16**, SIGTERM'd (rc 143) during vLLM init on
the two-node path. Root cause for the *production* eval: the earlier `TP8xPP2`
baseline fix only touched the hardcoded smoke wrapper
(`run_m3_quality_smoke_srun.sh`). The production data path
(matrix schema -> `build_launch_plan` -> `run_m3_quality_eval_srun.sh`) was
pipeline-parallel-blind, so a real run would relaunch BF16 as **TP16xPP1**,
with **no Ray topology preflight** (it was gated on `PROFILE == smoke`) and
**no wall-clock bound** — exactly the conditions that killed BF16.

## Code fix already pushed (`duy-branch`, 2026-07-13)

- `ModelSpec` / `load_matrix` carry `pipeline_parallel_size` (default 1);
- matrix `bf16` is now `tensor_parallel_size: 8` + `pipeline_parallel_size: 2`
  + `distributed_executor_backend: ray`, `nodes: 2`;
- `build_launch_plan` emits `pipeline_parallel_size` per arm;
- smoke-gate `distributed_world_size` check compares against `tp * pp`
  (BF16 reports 16);
- `run_m3_quality_eval_srun.sh` forwards `--pipeline-parallel-size`, runs the
  two-node Ray topology preflight whenever **any** arm needs >1 node (production
  included, not smoke-only), and honors an optional `TIME_LIMIT` -> `srun --time`;
- tests updated; 41 CPU tests pass locally.

No checkpoints, prompts, sample identities, probe corpus, or quantization
settings were changed. Do not change them without recording the deviation.

## Remaining executor tasks (in order)

1. **Pull and revalidate.** `git pull` on `duy-branch`; rerun the CPU contract
   tests:

   ```bash
   cd /mnt/nfs/hoangduy/projects/llm-compressor
   source /mnt/nfs/hoangduy/venvs/quant/bin/activate
   export PYTHONPATH="$PWD"
   python -m pytest -q \
     pipeline/tests/test_m3_quality_eval.py \
     pipeline/tests/test_m3_quality_eval_runner.py \
     pipeline/tests/test_eval_distributional.py
   ```

2. **Regenerate a production-valid smoke gate that includes BF16.** The passing
   GPTQ/AWQ smoke ran through the tmux wrapper, which does **not** emit
   `smoke_gate.json`. Rerun smoke via
   `run_m3_quality_eval_srun.sh --profile smoke` (BF16 now launches TP8xPP2
   behind the Ray gate). Confirm the resulting `smoke_gate.json` has
   `ready_for_production: true` with all three models passing.

3. **Build the repaired-overlay matrix + fresh preflight** in the run root. The
   committed matrix still points `inhouse_gptq` at the raw source, which fails
   the serving-ABI gate. Use the same
   `--add-vllm-shared-expert-ignore --add-vllm-router-ignore` overlay flow as
   the "Fresh preflight" section of the discriminator handoff, producing
   `$RUN_ROOT/repaired_matrix.yaml`. Do not proceed to GPU unless all three
   `preflight/serving_abi/*.json` are `"valid": true`.

4. **Launch the production eval with an explicit time budget** under a detached
   shell (no production tmux wrapper exists yet — use tmux/nohup outside every
   Slurm allocation; `[[ -z "${SLURM_JOB_ID:-}" ]]` must succeed):

   ```bash
   TIME_LIMIT=08:00:00 \
   bash pipeline/slurm/run_m3_quality_eval_srun.sh \
     --profile production --matrix "$MATRIX" --run-root "$RUN_ROOT" \
     --smoke-gate "$RUN_ROOT/smoke_gate.json"
   ```

   Production runs the full suite (mmlu_pro 2,000 samples, `limit: null`
   elsewhere, 16k-token generations, 49,152-token probe). Pick `TIME_LIMIT`
   from the smoke throughput and record it. Confirm the cluster can hold all
   **8 nodes** concurrently (BF16 2 nodes x 2 shards + GPTQ/AWQ 1 node x 2
   shards each); if not, stagger shards and record the deviation.

5. **Distribution comparisons + commit.** Run the comparisons (BF16-vs-GPTQ
   preferred, plus GPTQ-vs-AWQ) per the "Distribution comparisons" section of
   the discriminator handoff, then commit the run root (excluding checkpoints).

## Open decisions for the primary agent

- The exact `TIME_LIMIT` value (depends on full-suite throughput; the lm-eval
  tasks at full sample counts dominate the probe).
- Whether to add a production tmux controller and/or a node-availability
  preflight before the launch.

Keep observations separate from hypotheses. Return the run root, all
`smoke_gate.json` / preflight / manifest artifacts, per-arm evidence and return
codes, distribution comparison JSON, package/GPU identities, and any deviation.
