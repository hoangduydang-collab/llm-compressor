# MiniMax-M3 shared-expert repair validation runbook

## Objective

Validate the config-only vLLM shared-expert ignore alias and close the quality
issue if repaired W4A8 passes canonical offline and HTTP serving. Do not
re-quantize, rewrite shards, enable CUDA graphs, or investigate the second issue.

Confirmed input evidence from matrix `20260711-144120-routed-diagnostics`:

- reference passes; candidate W4A8 and W4A16 fail;
- every candidate rank sees 171 shared tensors and leaves all 171 unmatched;
- both candidates construct zero packed shared parameters;
- all 48 candidate probes have zero shared output; all 48 reference probes are
  nonzero;
- candidate first-MoE inputs match across W4A8/W4A16 and LM-head hashes match
  the reference.

The overlay adds exactly:

```text
re:.*block_sparse_moe[.]shared_experts[.].*
```

Payload files remain symlinks to the source checkpoint. Only copied
`config.json` metadata changes.

## Matrix

| Arm | Interface | Purpose |
| --- | --- | --- |
| `repaired_w4a8_offline` | canonical offline | Prove shared loading/execution and W4A8 quality |
| `repaired_w4a16_offline` | canonical offline | Preserve activation-disabled control |
| `repaired_w4a8_http` | canonical HTTP chat | Prove production-interface quality |

Every arm uses a separate exclusive eight-GPU node, TP8+EP, eager mode, block
size 128, FP8 KV cache, 2048 context, 0.85 utilization, disabled custom
all-reduce, disabled shared-expert auxiliary stream, 64 output tokens,
temperature 0, and thinking disabled. Offline arms enable loader, fingerprint,
and structured MoE diagnostics. HTTP keeps diagnostics off.

Scheduler partition, constraints, NFS roots, and time may be adapted and must be
recorded. Do not change a quality variable. Use `srun`; `sbatch` is unavailable.

## Preflight

```bash
git pull --ff-only origin duy-branch
git status --short
git rev-parse HEAD

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD"

pytest -q \
  pipeline/tests/test_m3_shared_expert_repair.py \
  pipeline/tests/test_m3_shared_expert_repair_runner.py \
  pipeline/tests/test_run_m3_shared_expert_repair_srun.py \
  pipeline/tests/test_m3_routed_diagnostics_runner.py \
  pipeline/tests/test_m3_quality_evidence.py \
  pipeline/tests/test_patch_vllm_m3_serve.py

MATRIX_ID=preflight-shared-repair DRY_RUN=1 \
  bash pipeline/slurm/run_m3_shared_expert_repair_srun.sh
```

The worktree must be clean. Dry-run output must contain exactly three commands,
each with `srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8`, and no `sbatch`.

## Live run

```bash
MATRIX_ID="$(date +%Y%m%d-%H%M%S)-shared-repair"
MATRIX_ID="$MATRIX_ID" \
TIME_LIMIT=02:00:00 \
SRUN_ARGS="--partition=compute" \
bash pipeline/slurm/run_m3_shared_expert_repair_srun.sh \
  2>&1 | tee "/mnt/nfs/hoangduy/logs/m3-shared-repair-$MATRIX_ID-launcher.log"
```

The launcher patches/checks the shared vLLM environment once, launches all
three allocations concurrently, waits for every sibling, rebundles after logs
close, and aggregates even if an arm fails. Runtime troubleshooting and
scheduler adaptation are allowed when recorded; do not implement another model
repair during this run.

## Required evidence and verdicts

```text
results/m3-shared-expert-repair/<matrix_id>/
├── comparison.json
├── repaired_w4a8_offline/
├── repaired_w4a16_offline/
└── repaired_w4a8_http/
```

Each manifest must record source/overlay config hashes, source index hash,
overlay alias, interface/scheme, job/node, code/environment, fixed envelope,
return code, deviations, retries, and retained full-log hashes. Preserve raw
offline reports and HTTP requests/responses. Offline evidence must include all
rank loader summaries, BF16 shared fingerprints, and structured MoE probes.

Interpret only the aggregate classifier:

- `quality_repair_pass`: both repaired W4A8 interfaces and W4A16 pass; all
  offline shared loader/fingerprint/probe gates are healthy;
- `shared_ignore_repair_failed`: shared tensors remain unmatched, runtime shared
  parameters remain packed/zero, or any shared output is zero/dropped;
- `activation_boundary_after_shared_repair`: W4A16 passes while both W4A8
  interfaces fail after healthy shared loading;
- `candidate_interface_disagreement`: W4A8 offline and HTTP differ;
- `post_shared_routed_boundary`: both schemes fail after healthy shared loading;
- `w4a16_overlay_backend_regression`: W4A8 passes but W4A16 fails;
- infrastructure or missing-evidence verdicts are not model conclusions.

## Commit and return

```bash
EVIDENCE_ROOT="results/m3-shared-expert-repair/$MATRIX_ID"
python -m json.tool "$EVIDENCE_ROOT/comparison.json"
find "$EVIDENCE_ROOT" -type f -size +5M -print
rg -n -i 'api[_-]?key|token=|authorization:|bearer ' "$EVIDENCE_ROOT" || true
git add "$EVIDENCE_ROOT"
git commit -m "data: add MiniMax-M3 shared-expert repair $MATRIX_ID"
git push origin duy-branch
```

Return result/code commits, matrix ID, all three `srun` job IDs/nodes, aggregate
and arm outcomes, loader match counts, shared representation/norm summary,
canonical responses, every deviation/retry, missing signals, and full-log
paths/hashes/retention. Stop for analysis; do not resume CUDA graphs yet.
