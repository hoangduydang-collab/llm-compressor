# MiniMax-M3 Quantized Checkpoint Handoff

## Goal and current status

Verify full-calibration MiniMax-M3 W4A8 AWQ/GPTQ checkpoints with the
MiniMax-M3-specific vLLM serve path and identify why generation is garbage.

The checkpoint currently **loads successfully but has not passed quality
verification**. Do not delete any original checkpoint.

## Checkpoints

| Checkpoint | Status |
| --- | --- |
| `artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint` | Original AWQ W4A8; loads, outputs repeated `arring...` garbage. |
| `artifacts/MiniMax-M3-gptq-W4AFP8/20260709-064842/checkpoint` | Original GPTQ W4A8; loads, also outputs garbage. |
| `artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123` | Portable AWQ re-export; statically validated and loads, but still outputs garbage. |
| `/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4` | Working reference model. |

The re-export is 225 GB. It rewrites only Safetensors tensor names; raw tensor
payloads are copied byte-for-byte.

## Confirmed results

1. The original AWQ and GPTQ checkpoints pass structural checkpoint audits and
   load under the M3 vLLM serve setup, but produce nonsensical generation.
2. Serving the original AWQ checkpoint with W4A16 (no runtime FP8 activation
   quantization) still produced `arring...`. W4A8 activation quantization is
   therefore not the sole cause.
3. The original routed-expert tensor layout uses descriptive names:
   `gate_proj`, `down_proj`, and `up_proj`. The installed NVIDIA vLLM M3 loader
   maps only `w1`, `w2`, and `w3` for routed experts:

   - `w1 = gate`
   - `w2 = down`
   - `w3 = up`

4. `pipeline.reexport_minimax_m3_vllm` created a portable layout by renaming
   routed keys:

   - `gate_proj -> w1`
   - `down_proj -> w2`
   - `up_proj -> w3`

   Shared-expert keys intentionally remain descriptive; cyankiwi uses the same
   shared-expert naming.
5. Static re-export validation passed: 5 shards, 67,192 keys, 65,664 routed
   keys renamed. The portable checkpoint completed an 8-H100 vLLM serve with
   `loaded=true`, `rc=0`, and a nonempty output, but the output remained
   garbage:

   ```text
   seringk seringk seringk mempunastast...
   ```

   This proves the routed-key layout mismatch was real but not sufficient to
   explain the quality failure.

## Relevant code changes

- `pipeline/reexport_minimax_m3_vllm.py`
  - Header-only Safetensors re-export utility.
  - Refuses to overwrite a destination, validates transformed index/header
    keys and raw payload byte counts.
- `pipeline/tests/test_reexport_minimax_m3_vllm.py`
  - Tests routed key aliases and payload-preserving shard rewrite.
- `pipeline/slurm/patch_vllm_m3_serve.py`
  - Adds environment-gated `M3_LOAD_AUDIT=1` instrumentation.
  - The audit is injected into vLLM site-packages when `serve_verify` runs.
- `pipeline/serve_verify.py`
  - Installs the optional loader audit before vLLM worker creation.
- `pipeline/tests/test_patch_vllm_m3_serve.py`
  - Covers audit injection/idempotence.

CPU tests run successfully:

```bash
PYTHONPATH="$PWD" pytest -q \
  pipeline/tests/test_reexport_minimax_m3_vllm.py \
  pipeline/tests/test_patch_vllm_m3_serve.py
# 5 passed
```

## Logs and reports

- Original AWQ loader audit:
  `/mnt/nfs/hoangduy/logs/m3-awq-load-audit-srun.log`
- Direct M3 loader audit:
  `/mnt/nfs/hoangduy/logs/m3-awq-loader-audit2-srun.log`
- Re-export:
  `/mnt/nfs/hoangduy/logs/m3-awq-w123-reexport.log`
- Portable checkpoint serve:
  `/mnt/nfs/hoangduy/logs/m3-awq-w123-verify-srun.log`
- Portable checkpoint report:
  `artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123/serve_report.json`

The serve report's `sane_output=true` only means the response is nonempty. It
does not measure semantic quality and is a false positive for these runs.

## Important caveats for further analysis

- Both our checkpoints and cyankiwi place `n_shared_experts=1` under
  `text_config`, not the top-level config. The current audit heuristic warns
  about this, but cyankiwi works, so this warning alone is not root-cause
  evidence.
- The loader audit established that the original routed keys did not reach a
  fused-MoE parameter loader. Its aggregate alias diagnostic should be treated
  as supporting evidence rather than a final proof: the vLLM mapping contains
  full per-expert prefixes, and the audit's presentation is verbose.
- The portable checkpoint's unchanged garbage means the next investigation
  should compare actual loaded parameter values and shared-expert contributions
  against cyankiwi, rather than retrying activation dtype or routed key aliases.
- The existing MoE forward probe is guarded against CUDA graph capture, but
  prior runs did not obtain a useful real-prompt measurement.

## Suggested planner work

1. Design a minimal, low-risk comparison that samples loaded routed and shared
   parameter statistics from cyankiwi versus the portable AWQ checkpoint after
   vLLM construction.
2. Determine whether the shared-expert module is instantiated and receives its
   tensors in both runs, including its contribution during a real prefill.
3. Check whether compressed-tensors W4A8 expert loading supports the exported
   `weight_packed`, `weight_scale`, and `weight_shape` layout equivalently to
   cyankiwi's W4A16 checkpoint.
4. Only after a quality-positive serve should the user be asked again to delete
   the obsolete 225 GB original checkpoint.

## Active next handoff (2026-07-11, shared-expert repair via srun)

Pull the handed-off commit and run `MINIMAX_M3_QUALITY_RUNBOOK.md`. The previous
matrix proved a serve-time naming failure: every candidate rank sees 171 shared
tensors and leaves all 171 unmatched; both candidate schemes create zero packed
shared parameters and all 48 probes have zero shared output. Reference shared
weights and outputs are healthy, candidate first-MoE inputs agree across W4A8
and W4A16, and LM-head hashes match.

The implementation creates an immutable metadata overlay that retains the
Transformers ignore regex and adds the vLLM-native alias
`re:.*block_sparse_moe[.]shared_experts[.].*`. No tensor shard is copied,
rewritten, packed, or re-quantized.

Use `pipeline/slurm/run_m3_shared_expert_repair_srun.sh`. It launches repaired
W4A8 offline diagnostics, repaired W4A16 offline diagnostics, and repaired W4A8
canonical HTTP concurrently on three exclusive eight-GPU nodes through `srun`.
Return and commit the complete compact bundle, then stop for primary-agent
analysis. Do not enable CUDA graphs or begin the second issue in this run.


## Current handoff: AWQ offset-norm repair plus GPTQ control

The layer-boundary matrix localized the first AWQ corruption to layer 8 between
attention output and MoE input. MiniMax-M3's Transformers class is
MiniMaxM3VLRMSNorm, a Gemma-style norm with effective weight 1 + weight.
The existing offset-norm calibration registry did not recognize this class, so
generic AWQ divided the zero-centered raw parameter instead of the effective
weight.

Prepare three checkpoints concurrently using only srun:

    DRY_RUN=1 bash pipeline/slurm/run_m3_awq_gptq_prepare_srun.sh
    bash pipeline/slurm/run_m3_awq_gptq_prepare_srun.sh

This re-exports the existing GPTQ checkpoint and separately quantizes AWQ with
the corrected offset norm and with MLP-input smoothing disabled. After all
three preparation jobs succeed, launch the twelve-node matrix:

    DRY_RUN=1 bash pipeline/slurm/run_m3_awq_gptq_repair_srun.sh
    bash pipeline/slurm/run_m3_awq_gptq_repair_srun.sh

The GPTQ portable checkpoint finishes much earlier than either AWQ rebuild. Do
not wait: launch the early six-node phase while both quantizations continue:

    bash pipeline/slurm/run_m3_gptq_early_srun.sh

Record the printed MATRIX_ID. Once both AWQ variants finish, add only their six
arms to the same evidence directory:

    MATRIX_ID=<printed-id> bash pipeline/slurm/run_m3_awq_repair_finish_srun.sh

The early phase writes comparison_early.json; the finish phase writes the final
comparison.json without rerunning reference, GPTQ, AWQ control, or tensor audit.

Every offline arm probes all sparse layers 3-59. Return the complete
results/m3-awq-gptq-repair/<matrix-id>/ tree, preparation and matrix logs,
checkpoint paths, job/node/return codes, deviations, retries, and retained-log
hashes. Do not start CUDA-graph work.


### Repair the missing staged tensor audit

The first staged run completed all serving arms but its tensor audit artifact was
missing. Pull the classifier/audit fix and rerun only that one allocation:

    MATRIX_ID=20260712-045912-awq-gptq-staged \
      bash pipeline/slurm/rerun_m3_checkpoint_scale_audit_srun.sh

This returns checkpoint_scale_audit.json, its full log and return code, then
regenerates comparison_early.json. Explosion detection is now reference-relative:
the normal approximately 10k residual at layer 5 is no longer a false boundary.


## Current handoff: vLLM-first paired quality matrix (2026-07-12)

### Scope and stop condition

This run addresses the **model-quality issue only**. Do not start serving
throughput/CUDA-graph diagnosis. Compare exactly BF16, in-house GPTQ, cyankiwi
AWQ, and aquaman AutoRound using paired prompts and direct vLLM execution.
A quality-gate failure is a valid experimental result and must be returned; an
infrastructure failure must be reported separately.

The launcher deliberately has two stages:

1. Four concurrent smoke arms use five nodes total (BF16: 2 nodes/16 H100;
   each quantized model: 1 node/8 H100). Each task gets two exact samples,
   generations are capped at 256 tokens, and each model gets a 2,048-token
   distributional probe.
2. Only a passed `smoke_gate.json` unlocks eight concurrent production arms
   using ten nodes total. BF16's two arms each use 2 nodes/16 H100 with Ray;
   the six quantized arms each use 1 node/8 H100.

Use `srun`, not `sbatch`.

### Pull, environment, and dynamic preflight

The AutoRound checkpoint is not preinstalled on the executor cluster. Download
the exact Hugging Face repository to the path already configured in
`pipeline/configs/minimax_m3_quality_matrix.yaml` before running preflight:

```bash
AUTOROUND_REPO=aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx
AUTOROUND_REV=40b00366dcb6bcf16b1710f242b12ce76dd34b9f
AUTOROUND_DIR=/mnt/nfs/hoangduy/hf_assets/aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx

mkdir -p "$(dirname "$AUTOROUND_DIR")"
hf download "$AUTOROUND_REPO" \
  --revision "$AUTOROUND_REV" \
  --local-dir "$AUTOROUND_DIR"
printf '%s\n' "$AUTOROUND_REV" > "$AUTOROUND_DIR/DOWNLOAD_REVISION"
```

Source: <https://huggingface.co/aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx>.
The pinned revision prevents a multi-hour evaluation from silently changing if
the model repository is updated. The model card reports an approximately
176-GiB checkpoint, so check shared-filesystem capacity first. `hf download` is
resumable; rerun the identical command after a network interruption. If the
`hf` command is unavailable, install/update `huggingface_hub` in the executor's
virtual environment (`python -m pip install -U huggingface_hub`) and retry.

Verify every file referenced by the safetensors index rather than relying only
on directory existence:

```bash
python - "$AUTOROUND_DIR" "$AUTOROUND_REV" <<'PYVERIFY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
revision = sys.argv[2]
required = [root / "config.json", root / "model.safetensors.index.json"]
missing = [str(path) for path in required if not path.is_file()]
if not missing:
    index = json.loads(required[1].read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing.extend(str(root / shard) for shard in shards if not (root / shard).is_file())
else:
    shards = []
if missing:
    raise SystemExit("incomplete AutoRound download; missing:\n" + "\n".join(missing[:20]))
recorded = (root / "DOWNLOAD_REVISION").read_text().strip()
if recorded != revision:
    raise SystemExit(f"revision mismatch: expected {revision}, recorded {recorded}")
print(f"AutoRound checkpoint ready: {root} ({len(shards)} indexed shards, revision {revision})")
PYVERIFY
```

The pinned index currently references **36 weight files**: 35 numbered files
(`model-00001-of-00035.safetensors` through
`model-00035-of-00035.safetensors`) plus `model_extra_tensors.safetensors`.
If the verification prints another count, preserve the output and stop for
primary-agent review rather than editing the matrix or selecting a different
checkpoint.

The model card's canonical serving recipe uses OneCompression at tag
`m3-serving-v1` plus repository-specific loader glue. Do not install that plugin
into the shared environment merely to make preflight pass. Run our mandatory
matrix smoke first. If the AutoRound arm fails with an unknown
`autoround_mixed` quantization method, missing OneCompression plugin, or loader
key mismatch, stop before production and return its complete logs and smoke
artifacts. That is an integration incompatibility for the primary agent to
resolve; do not silently switch only this arm to the model card's custom server,
because doing so would invalidate the paired direct-vLLM comparison.

```bash
git pull
cd /mnt/nfs/hoangduy/projects/llm-compressor
source <the-working-vllm-environment>/bin/activate
export PYTHONPATH="$PWD"
RUN_ID="$(date +%Y%m%d-%H%M%S)-m3-quality"
RUN_ROOT="results/m3-quality/$RUN_ID"
MATRIX=pipeline/configs/minimax_m3_quality_matrix.yaml
mkdir -p "$RUN_ROOT"

python -m pipeline.m3_quality_preflight \
  --matrix "$MATRIX" \
  --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/preflight.log"
```

Preflight intentionally performs the runtime-dependent work on the capable
cluster: verifies all checkpoint paths, resolves installed lm-eval task aliases
and leaf splits, creates immutable smoke/production sample manifests, builds
calibration-disjoint 2,048/49,152-token probe corpora, records tokenizer/chat
hashes and Git/software provenance, and writes checkpoint diagnostics.
Do not proceed unless `run_manifest.json`, `preflight/resolved_eval_config.yaml`,
both sample manifests, both probe corpora, `resolved_tasks.json`, and four
checkpoint diagnostic JSONs exist.

If preflight fails because a task alias or dataset split changed, diagnose the
installed lm-eval task registry locally and make the smallest documented fix.
Do not silently substitute a benchmark or reduce production sample counts.

### Mandatory smoke, then parallel production

First inspect the exact allocations without consuming GPUs:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$MATRIX" --run-root "$RUN_ROOT" --dry-run
```

It must print 4 arms and `total_nodes=5`. Then run:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$MATRIX" --run-root "$RUN_ROOT"
SMOKE_RC=$?
python -m json.tool "$RUN_ROOT/smoke_gate.json"
```

Stop and return evidence immediately if `SMOKE_RC != 0` or
`ready_for_production` is not true. Return every smoke stdout/stderr log,
`smoke_report.json`, `smoke_gate.json`, each `smoke_evidence.json`, each arm
manifest/return code, and the preflight artifacts. Include Slurm job IDs,
nodes, environment/version output, exception traces, and any deviation or
retry. Do not launch a five-hour run after a failed smoke.

After smoke passes:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  --smoke-gate "$RUN_ROOT/smoke_gate.json" --dry-run
```

It must print 8 arms and `total_nodes=10`. Launch all arms concurrently:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  --smoke-gate "$RUN_ROOT/smoke_gate.json"
PRODUCTION_RC=$?
```

Do not cancel healthy arms merely because another arm fails; preserve all
completed evidence. If cluster contention prevents ten simultaneous nodes,
keep the same arms and resource topology but start as many independent arms as
available, documenting start/end times and queueing.

### Aggregate and return contract

Run aggregation even when one production arm failed; its infrastructure report
is part of the diagnosis:

```bash
set +e
python -m pipeline.m3_quality_eval aggregate \
  --matrix "$MATRIX" --root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/aggregate.log"
AGGREGATE_RC=${PIPESTATUS[0]}
set -e
```

Commit and push the complete compact evidence tree and this handoff update.
Return all of the following so the primary agent can do the analysis without
asking for a rerun:

- `run_manifest.json`, all preflight manifests/configs/hashes, resolved task
  names and leaf sizes, probe corpora metadata, and checkpoint diagnostics;
- every arm's `arm_manifest.json`, `arm_complete.json`, `return_code.txt`,
  aggregate metrics, normalized per-sample JSONL, generation-health JSON,
  distributional probe JSONL+summary, and stdout/stderr log;
- root `matrix.json`, `gates.json`, `report.md`, `aggregate.log`, smoke report
  and gate;
- exact command lines, Git commit, Python/PyTorch/CUDA/vLLM/lm-eval versions,
  Slurm job IDs/nodes, wall times, exit codes, retries, OOMs, and deviations;
- log/artifact SHA-256 values if any large file must remain outside Git, plus
  its exact durable path. Never replace a missing artifact with a prose summary.

The decision metrics are downstream accuracy, paired accuracy delta and CI,
flip/regression/recovery rates, exact/asymptotic McNemar evidence, score
recovery, teacher-forced NLL/perplexity/top-k drift, generation degeneration,
and checkpoint quantization coverage/scales/saturation. Serving latency and
throughput remain explicitly out of scope until quality is resolved.
