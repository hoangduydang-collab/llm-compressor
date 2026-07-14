# Execution packet: MiniMax-M3 native distributed quantization smoke

- Protocol version: 1
- State: `READY_FOR_EXECUTOR`
- Packet revision: `2026-07-15-r3`
- Planner owner: Codex planner
- Intended executor: cluster executor
- Required fix commit: `4801028f`
- Decision question: Can native eight-rank llm-compressor GPTQ and AWQ process
  representative MiniMax-M3 layers with correct data sharding, shared model
  memory, complete evidence, and enough quantization-time improvement to justify
  a later full run of the quality-selected recipe?

This is the single active packet for MiniMax-M3 quantization speed-up Phase 1.
It supersedes the informal launch commands in `M3_QUANT_SPEEDUP_PLAN.md`; that
document remains the rationale and history.

## Objective and hypothesis

Run native llm-compressor GPTQ and AWQ through the ordinary
`pipeline.run -> oneshot` path under eight initialized ranks. Quantization is
restricted to decoder layers 3, 31, and 59 with the same ignore-list mechanism
used by the existing MiniMax representative configuration. Eight globally
configured calibration samples are deterministically shuffled, partitioned into
one disjoint sample per rank, and hashed.

The hypothesis is that compressed-tensors shared offload keeps host-memory growth
near one model copy while native GPTQ module parallelism and AWQ data parallelism
reduce quantization-only time. This packet collects the evidence needed for the
planner to decide; it does not authorize a full calibration.

## Scope and non-goals

- In scope: one evidence-only GPTQ smoke and one evidence-only AWQ smoke,
  sequentially, using one exclusive 8xH100 node at a time.
- In scope: passive process, timing, host/GPU memory, native quantization metrics,
  model provenance, calibration-partition, and scheduler evidence.
- Not authorized: `pipeline.m3_awq_representative`, its runtime probes/audited
  modifiers, any quality evaluation, any usable checkpoint, a full calibration,
  bespoke expert sharding, code changes during execution, or any retry beyond
  the single fresh-ID r3 run authorized by this packet.
- The current paired GPTQ/AWQ quality evaluation remains independent and running.

## Preconditions and exact environment

- Repository path: `/mnt/nfs/hoangduy/projects/llm-compressor`
- Branch: `duy-branch`
- Environment activation:
  `source /mnt/nfs/hoangduy/venvs/quant/bin/activate`
- Environment file: `/mnt/nfs/hoangduy/env.sh`
- Python path:
  `export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"`
- Model path:
  `/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3`
- Dataset: `HuggingFaceH4/ultrachat_200k`, split `train_sft`; credentials and
  cache access must already work in the quantization environment.
- Scheduler: top-level `srun` only, owned by detached `tmux` from outside an
  allocation.

## Required inputs

| Input | Exact path or identifier | Required validation |
| --- | --- | --- |
| Implementation | Git ancestor `4801028f` | `git merge-base --is-ancestor 4801028f HEAD` |
| Smoke config | `pipeline/configs/minimax_m3_distributed_smoke.yaml` | load with `pipeline.config.load_config` |
| Launcher | `pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh` | focused tests plus `bash -n` |
| Model | `/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3` | directory and `config.json` exist |
| Environment | `/mnt/nfs/hoangduy/venvs/quant/bin/activate` | source succeeds and imports below pass |

## Workspace policy

- Protected paths: all tracked files, the model directory, prior result/log
  roots, and the current paired-quality run.
- Permitted untracked roots inside the repository: existing `results/` and
  `artifacts/` only, provided the fresh run ID does not collide.
- Record and proceed: unrelated pre-existing untracked files only under those two
  roots; record `git status --short` before launch.
- Stop: staged or unstaged tracked changes; untracked files outside the permitted
  roots; a missing input; a run/log/offload-root collision; nested Slurm; or an
  implementation ancestor check failure.

## Resource contract

- Jobs/arms: two (`gptq`, then `awq`).
- Nodes: one per arm; arms run sequentially, never concurrently.
- GPUs: exactly eight H100 GPUs per arm.
- Exclusivity: `srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8`.
- Process layout: one launcher task starts `torchrun --nproc_per_node=8`; eight
  local distributed ranks, one rank per GPU.
- Time limit: `24:00:00` per arm.
- Expected runtime: approximately 2–10 hours per arm including model load; the
  controller may remain alive for up to 48 hours plus queue time.
- Node preflight inside each allocation: exactly eight visible GPUs, at least
  1.2 TB `MemAvailable`, and at least 900 GB free in `/dev/shm`.

## Commands

### Setup and revision verification

```bash
set -euo pipefail
cd /mnt/nfs/hoangduy/projects/llm-compressor
git fetch origin
git checkout duy-branch
git pull --ff-only origin duy-branch
git merge-base --is-ancestor 4801028f HEAD

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

test -z "${SLURM_JOB_ID:-}"
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
git status --short | tee /tmp/m3-ddp-speedup-worktree.txt
git ls-files --others --exclude-standard \
  | awk '!/^(results|artifacts)\//' \
  | tee /tmp/m3-ddp-speedup-untracked-blockers.txt
test ! -s /tmp/m3-ddp-speedup-untracked-blockers.txt

test -f /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3/config.json
test -f pipeline/configs/minimax_m3_distributed_smoke.yaml
test -f pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh
```

### Preflight

```bash
set -euo pipefail
python - <<'PY'
import importlib.metadata as md
import torch
from compressed_tensors.offload import init_dist
from pipeline.config import load_config

cfg = load_config("pipeline/configs/minimax_m3_distributed_smoke.yaml")
assert cfg.model.device_map == "auto_offload"
assert cfg.model.max_memory == {"cpu": 1_000_000_000_000}
assert cfg.calibration.num_samples == 8
assert cfg.calibration.max_seq_length == 512
assert cfg.calibration.sequential_targets == ["MiniMaxM3VLDecoderLayer"]
assert cfg.quantization.sample_generation is False
print({name: md.version(name) for name in (
    "llmcompressor", "compressed-tensors", "torch", "transformers", "datasets"
)})
print("cuda_build", torch.version.cuda)
print("compressed_tensors_init_dist", init_dist)
PY

python -m pytest -q \
  pipeline/tests/test_distributed.py \
  pipeline/tests/test_calibration_partition.py \
  pipeline/tests/test_distributed_quantize_contract.py \
  pipeline/tests/test_metrics.py \
  pipeline/tests/test_m3_distributed_quant_smoke.py
bash -n pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh
```

Expected: config assertions pass, relevant package versions print, all focused
tests pass, and Bash syntax returns zero.

Stop if: any import/assertion/test/syntax check fails. Do not install or upgrade
packages in response; return the exact failure.

### Dry run

```bash
set -euo pipefail
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-ddp-quant-smoke-r3-dry"
DRY_RUN=1 RUN_ID="$RUN_ID" \
  bash pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh \
  | tee "/tmp/$RUN_ID.txt"
test "$(grep -c 'srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8' "/tmp/$RUN_ID.txt")" -eq 2
test "$(grep -c 'torchrun --nproc_per_node=8 -m pipeline.run' "/tmp/$RUN_ID.txt")" -eq 2
test "$(grep -c -- '--evidence-only' "/tmp/$RUN_ID.txt")" -eq 2
! grep -q 'pipeline.m3_awq_representative' "/tmp/$RUN_ID.txt"
```

Expected: exactly two sequential top-level allocations, GPTQ before AWQ, each
containing one eight-rank evidence-only command and no representative harness.

### Launch

```bash
set -euo pipefail
cd /mnt/nfs/hoangduy/projects/llm-compressor
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-ddp-quant-smoke-r3"
RESULT_ROOT="/mnt/nfs/hoangduy/results/m3-distributed-quant-smoke/$RUN_ID"
LOG_ROOT="/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/$RUN_ID"
OFFLOAD_ROOT="/mnt/nfs/hoangduy/offload/m3-distributed-quant-smoke/$RUN_ID"
SESSION="m3-ddp-quant-${RUN_ID}"

test ! -e "$RESULT_ROOT"
test ! -e "$LOG_ROOT"
test ! -e "$OFFLOAD_ROOT"
mkdir -p "$LOG_ROOT"
printf '%s\n' "$RUN_ID" >"$LOG_ROOT/run_id.txt"
git rev-parse HEAD >"$LOG_ROOT/expected_git_commit.txt"

tmux new-session -d -s "$SESSION" \
  "cd '$PWD'; rc=0; RUN_ID='$RUN_ID' RESULT_ROOT='$RESULT_ROOT' LOG_ROOT='$LOG_ROOT' OFFLOAD_ROOT='$OFFLOAD_ROOT' bash pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh >'$LOG_ROOT/controller.log' 2>&1 || rc=\$?; printf '%s\\n' \"\$rc\" >'$LOG_ROOT/controller.rc'"

printf 'RUN_ID=%s\nRESULT_ROOT=%s\nLOG_ROOT=%s\nOFFLOAD_ROOT=%s\nSESSION=%s\n' \
  "$RUN_ID" "$RESULT_ROOT" "$LOG_ROOT" "$OFFLOAD_ROOT" "$SESSION" \
  | tee "/tmp/${RUN_ID}-locations.txt"
```

### Monitoring

Monitoring is non-owning. Do not attach interactively or start another
controller.

```bash
source "/tmp/${RUN_ID}-locations.txt"
tmux has-session -t "$SESSION" 2>/dev/null && \
  tmux capture-pane -pt "$SESSION" -S -100 || true
squeue -u "$USER" -o '%.18i %.12P %.40j %.8T %.10M %.20R'
tail -n 100 "$LOG_ROOT/controller.log" || true
for method in gptq awq; do
  echo "===== $method ====="
  cat "$LOG_ROOT/$method/rc" 2>/dev/null || true
  tail -n 30 "$LOG_ROOT/$method/torchrun.err" 2>/dev/null || true
  tail -n 20 "$LOG_ROOT/$method/resources.log" 2>/dev/null || true
done
```

Wait for `controller.rc`. An arm failure does not cancel or suppress the other
arm; the launcher continues sequentially and returns a nonzero overall result.

### Aggregation and packaging

Run this after `controller.rc` exists, even when it is nonzero:

```bash
set -euo pipefail
source "/tmp/${RUN_ID}-locations.txt"
test -f "$LOG_ROOT/controller.rc"
EVIDENCE_ROOT="results/m3-distributed-quant-speedup/$RUN_ID"
mkdir -p "$EVIDENCE_ROOT"

# Capture scheduler evidence from the exact job/step IDs recorded inside each
# allocation. Scheduler retention is best-effort and does not change run status.
for method in gptq awq; do
  method_logs="$LOG_ROOT/$method"
  job_id="$(awk -F= '$1 == "slurm_job_id" {print $2}' "$method_logs/environment.txt" 2>/dev/null || true)"
  step_id="$(awk -F= '$1 == "slurm_step_id" {print $2}' "$method_logs/environment.txt" 2>/dev/null || true)"
  printf 'slurm_job_id=%s\nslurm_step_id=%s\n' "$job_id" "$step_id" \
    >"$method_logs/scheduler_ids.txt"
  if [[ -n "$job_id" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show job "$job_id" >"$method_logs/scontrol.txt" 2>&1 || true
  else
    printf 'scontrol unavailable or job ID missing\n' >"$method_logs/scontrol.txt"
  fi
  if [[ -n "$job_id" ]] && command -v sacct >/dev/null 2>&1; then
    sacct -j "$job_id" -P \
      --format=JobID,JobName,State,ExitCode,NodeList,AllocTRES,Start,End,Elapsed \
      >"$method_logs/sacct.txt" 2>&1 || true
  else
    printf 'sacct unavailable or job ID missing\n' >"$method_logs/sacct.txt"
  fi
done

RUN_ID="$RUN_ID" RESULT_ROOT="$RESULT_ROOT" LOG_ROOT="$LOG_ROOT" \
EVIDENCE_ROOT="$EVIDENCE_ROOT" python - <<'PY'
import hashlib, json, os, re, shutil
from pathlib import Path

from pipeline.metrics import summarize_quantized_layers

run_id = os.environ["RUN_ID"]
result_root = Path(os.environ["RESULT_ROOT"])
log_root = Path(os.environ["LOG_ROOT"])
evidence_root = Path(os.environ["EVIDENCE_ROOT"])
small_limit = 10 * 1024 * 1024

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}

def mem_available_values(path):
    values = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            match = re.match(r"MemAvailable:\s+(\d+)\s+kB", line)
            if match:
                values.append(int(match.group(1)) * 1024)
    return values

report = {"schema_version": 1, "run_id": run_id, "methods": {}}
for method in ("gptq", "awq"):
    method_logs = log_root / method
    method_results = result_root / method
    run_markers = []
    for pattern in (
        "config.yaml", "metadata.json", "smoke_complete.json",
        "calibration_partition.rank-*.json", "quant_metrics.rank-*.jsonl",
        "model_provenance.rank-*.json",
    ):
        run_markers.extend(method_results.rglob(pattern))
    run_dirs = sorted({path.parent for path in run_markers})
    run_dir = run_dirs[0] if len(run_dirs) == 1 else None
    completions = list(run_dir.glob("smoke_complete.json")) if run_dir else []
    manifests = sorted(run_dir.glob("calibration_partition.rank-*.json")) if run_dir else []
    rows = []
    for path in manifests:
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    bounds = sorted(
        (row.get("start"), row.get("end"))
        for row in rows
        if isinstance(row.get("start"), int) and isinstance(row.get("end"), int)
    )
    ranks = sorted(row.get("rank") for row in rows if isinstance(row.get("rank"), int))
    dist_ranks = sorted(
        row.get("distributed", {}).get("rank")
        for row in rows
        if isinstance(row.get("distributed", {}).get("rank"), int)
    )
    world_sizes = {row.get("distributed", {}).get("world_size") for row in rows}
    local_ranks = sorted(
        row.get("distributed", {}).get("local_rank")
        for row in rows
        if isinstance(row.get("distributed", {}).get("local_rank"), int)
    )
    cuda_devices = sorted(
        row.get("distributed", {}).get("cuda_current_device")
        for row in rows
        if isinstance(row.get("distributed", {}).get("cuda_current_device"), int)
    )
    token_hashes = [row.get("token_ids_sha256") for row in rows]
    global_sample_counts = {row.get("global_num_samples") for row in rows}
    local_sample_counts = [row.get("local_num_samples") for row in rows]
    valid_token_hashes = (
        len(token_hashes) == 8
        and all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in token_hashes
        )
    )

    preflight_mem = mem_available_values(method_logs / "node_preflight.txt")
    runtime_mem = mem_available_values(method_logs / "resources.log")
    peak_host_delta = (
        preflight_mem[0] - min(runtime_mem)
        if preflight_mem and runtime_mem else None
    )
    rc_path = method_logs / "rc"
    rc = int(rc_path.read_text().strip()) if rc_path.is_file() else None
    metrics = sorted(run_dir.glob("quant_metrics.rank-*.jsonl")) if run_dir else []
    provenance = sorted(run_dir.glob("model_provenance.rank-*.json")) if run_dir else []
    layer_work = summarize_quantized_layers(metrics, method=method)
    per_rank_layer_work = [
        {"path": str(path), **summarize_quantized_layers([path], method=method)}
        for path in metrics
    ]
    expected_layers = [3, 31, 59]
    checkpoint_exists = bool(run_dir and (run_dir / "checkpoint").exists())

    checks = {
        "return_code_zero": rc == 0,
        "one_run_directory": len(run_dirs) == 1,
        "one_completion": len(completions) == 1,
        "eight_partition_manifests": len(rows) == 8,
        "ranks_zero_through_seven": ranks == list(range(8)),
        "manifest_and_distributed_ranks_match": dist_ranks == ranks,
        "world_size_is_eight": world_sizes == {8},
        "local_ranks_zero_through_seven": local_ranks == list(range(8)),
        "rank_bounds_cover_global_eight": bounds == [(i, i + 1) for i in range(8)],
        "global_num_samples_is_eight": global_sample_counts == {8},
        "each_rank_has_one_local_sample": local_sample_counts == [1] * 8,
        "token_hashes_are_valid_sha256": valid_token_hashes,
        "eight_distinct_sample_hashes": valid_token_hashes and len(set(token_hashes)) == 8,
        "cuda_bindings_zero_through_seven": cuda_devices == list(range(8)),
        "eight_metrics_files": len(metrics) == 8,
        "metrics_files_nonempty": len(metrics) == 8 and all(path.stat().st_size > 0 for path in metrics),
        "native_work_on_every_rank": len(per_rank_layer_work) == 8 and all(
            item["record_count"] > 0 for item in per_rank_layer_work
        ),
        "native_work_is_exactly_layers_3_31_59": layer_work["layers"] == expected_layers,
        "native_work_names_resolve_to_language_decoder": not layer_work["unresolved_names"],
        "per_rank_work_names_resolve_to_language_decoder": all(
            not item["unresolved_names"] for item in per_rank_layer_work
        ),
        "method_specific_rank_layer_coverage": (
            all(item["layers"] == expected_layers for item in per_rank_layer_work)
            if method == "awq"
            else all(
                item["layers"] and set(item["layers"]).issubset(expected_layers)
                for item in per_rank_layer_work
            )
        ),
        "eight_provenance_files": len(provenance) == 8,
        "provenance_files_nonempty": len(provenance) == 8 and all(path.stat().st_size > 0 for path in provenance),
        "no_partial_checkpoint": not checkpoint_exists,
        "peak_host_delta_under_1_35tb": peak_host_delta is not None and peak_host_delta < 1_350_000_000_000,
    }
    large = []
    for name in ("torchrun.out", "torchrun.err", "resources.log"):
        path = method_logs / name
        if path.is_file():
            large.append(digest(path))

    # Preserve all small result and controller artifacts. Large native metrics
    # stay at their durable absolute path and are represented by size + hash.
    shared_result_files = (
        [run_dir / "config.yaml", run_dir / "metadata.json"] if run_dir else []
    )
    result_files = [
        *shared_result_files, *completions, *manifests, *metrics, *provenance
    ]
    log_files = [
        method_logs / name
        for name in (
            "node_preflight.txt", "environment.txt", "command.txt", "rc",
            "scheduler_ids.txt", "scontrol.txt", "sacct.txt"
        )
    ]
    for category, paths in (("result", result_files), ("log", log_files)):
        destination = evidence_root / method / category
        destination.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if not path.is_file():
                continue
            if path.stat().st_size <= small_limit:
                shutil.copy2(path, destination / path.name)
            elif path not in metrics:
                large.append(digest(path))
    for path in metrics:
        if path.is_file() and path.stat().st_size > small_limit:
            large.append(digest(path))

    report["methods"][method] = {
        "rc": rc,
        "run_dir": str(run_dir) if run_dir else None,
        "discovered_run_directories": [str(path) for path in run_dirs],
        "bounds": bounds,
        "cuda_devices": cuda_devices,
        "native_layer_work": layer_work,
        "per_rank_native_layer_work": per_rank_layer_work,
        "peak_host_memory_delta_bytes": peak_host_delta,
        "checks": checks,
        "mechanical_pass": all(checks.values()),
        "large_artifacts": large,
    }

controller_rc = log_root / "controller.rc"
report["controller_rc"] = int(controller_rc.read_text().strip())
report["mechanical_pass"] = all(
    arm["mechanical_pass"] for arm in report["methods"].values()
)
controller_log = log_root / "controller.log"
if controller_log.is_file():
    report["controller_log"] = digest(controller_log)
    if controller_log.stat().st_size <= small_limit:
        shutil.copy2(controller_log, evidence_root / "controller.log")
else:
    report["controller_log"] = None
(evidence_root / "aggregate.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

cp "$LOG_ROOT/controller.rc" "$EVIDENCE_ROOT/controller.rc"
cp "$LOG_ROOT/run_id.txt" "$EVIDENCE_ROOT/run_id.txt"
cp "$LOG_ROOT/expected_git_commit.txt" "$EVIDENCE_ROOT/expected_git_commit.txt"
git status --short >"$EVIDENCE_ROOT/final_git_status.txt"

# Hash every committed/copied small artifact. Raw large logs remain durable and
# are hashed with absolute paths in aggregate.json.
EVIDENCE_ROOT="$EVIDENCE_ROOT" python - <<'PY'
import hashlib, json, os
from pathlib import Path

root = Path(os.environ["EVIDENCE_ROOT"])
rows = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    if path.name == "small_artifacts.json":
        continue
    payload = path.read_bytes()
    rows.append({
        "path": str(path.relative_to(root)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    })
(root / "small_artifacts.json").write_text(json.dumps(rows, indent=2) + "\n")
PY
cat "$EVIDENCE_ROOT/aggregate.json"
```

The aggregate is mechanical infrastructure evidence. The executor must not turn
the timing records into the strategic decision; the planner will compare native
per-rank metrics against historical single-process logs.

## Expected jobs and independence rules

| Arm | Resources | Expected output | Failure effect |
| --- | --- | --- | --- |
| GPTQ | 1 exclusive node, 8 GPUs/ranks | rank manifests, provenance, metrics, passive logs, completion | Record failure; continue AWQ |
| AWQ | 1 exclusive node, 8 GPUs/ranks | rank manifests, provenance, metrics, passive logs, completion | Record failure; finish and return |

## Success gates and expected artifacts

For each method:

- Return code is zero and exactly one `smoke_complete.json` exists.
- Eight partition manifests cover bounds `(0,1)` through `(7,8)`, have eight
  distinct valid SHA-256 token hashes, report `global_num_samples=8` and
  `local_num_samples=1`, unique ranks/local ranks 0–7 and world size eight,
  agree between manifest and distributed rank, and bind CUDA devices 0–7.
- Eight nonempty model-provenance and eight nonempty native metric files exist.
- Native GPTQ/AWQ work records are nonempty and resolve exclusively to decoder
  layers 3, 31, and 59. Every rank must have native work; every AWQ rank must
  report all three layers, while every GPTQ rank must report a nonempty subset
  and the aggregate must report all three. This parses ordinary native records
  and does not add a runtime probe.
- Every small completion, partition, provenance, metric, environment, command,
  scheduler, preflight, and return-code artifact is copied and SHA-256 indexed
  by `small_artifacts.json`; metric files over 10 MiB stay durable and are
  represented by absolute path, size, and SHA-256 in `aggregate.json`.
- No `checkpoint/` exists.
- Peak host-memory delta from passive sampling is below 1.35 TB. This is a broad
  shared-copy safety gate, not a precise bandwidth benchmark.
- `node_preflight.txt`, `environment.txt`, `command.txt`, `torchrun.out`,
  `torchrun.err`, `resources.log`, and `rc` exist under the durable method log
  root.

Performance target, for planner interpretation: quantization-only timing should
improve over comparable historical single-process layer/module metrics; `>2x` is
the target, not a first-run hard failure threshold. Model loading and saving are
reported separately and excluded from that comparison.

## Allowed adaptations

- None. Queue placement on a different eligible node is not an adaptation.

## Pre-authorized record-and-proceed conditions

- Queue delay: record scheduler state and continue waiting.
- Existing unrelated untracked files under `results/` or `artifacts/`: record
  exact paths and continue if the fresh run root does not collide.
- One arm fails: preserve its evidence and allow the launcher to run the other
  arm exactly as written.

## Pre-authorized retries

- This r3 execution is the single planner-authorized retry of r2, triggered by
  the exact non-source-rank model-load failure recorded below:
  `MiniMaxM3SparseForConditionalGeneration.__init__() got an unexpected keyword
  argument 'tie_word_embeddings'`.
- Required fix: commit `4801028f`, which moves compressed-tensors' injected
  meta-rank setting onto the MiniMax composite config before construction.
- Maximum additional retry count after the r3 launch: 0.
- Fresh run ID required: yes; use the exact r3 run-ID commands above.
- Inputs that must remain unchanged: all model, config, calibration, topology,
  environment, and launcher inputs.

## Stop-and-return conditions

- Revision, workspace, environment, input, import, test, or dry-run preflight
  fails.
- The worker sees other than eight GPUs, less than 1.2 TB available RAM, or less
  than 900 GB free `/dev/shm`.
- Actual topology differs from one node/eight ranks.
- A result/log/offload root already exists.
- Continuing requires editing code/config, changing method/sample/layer scope,
  using a runtime probe, changing scheduler method/topology, or retrying.
- Evidence cannot be preserved without overwriting an existing artifact.

Capture immediate `squeue`, `scontrol`, `sacct`, controller tail, method stderr,
return code, last successful stage, and first failing operation before returning
when a launched allocation exits abnormally.

## Prohibited actions

- Do not edit tracked files before or during execution.
- Do not use the representative-layer runner or enable any runtime/structure/
  fidelity probe.
- Do not change layers, calibration samples, sequence length, recipes, methods,
  topology, time limit, memory gates, or evidence-only behavior.
- Do not save/re-export/serve/evaluate a checkpoint.
- Do not start a full calibration, custom expert-parallel implementation,
  downstream benchmark, retry, or fix.
- Do not delete or overwrite prior results, logs, or offload roots.

## Return contract

Create a factual evidence section at the end of this file using the canonical
executor-to-planner template in `PLANNER_EXECUTOR_PROTOCOL.md`. It must include:

- packet revision, expected and actual Git commits, and execution classification;
- exact commands/overrides, run ID, roots, tmux session, Slurm step IDs, nodes,
  timestamps, elapsed times, topology, terminal states, and return codes;
- package, CUDA, driver, and environment versions from `environment.txt`;
- `aggregate.json` with every per-method check and measurement;
- deviations, record-and-proceed conditions, retries, OOMs, signals,
  cancellations, scheduler failures, missing artifacts, last successful stage,
  and first failing operation, explicitly writing `None` where applicable;
- committed small artifacts under
  `results/m3-distributed-quant-speedup/$RUN_ID/`;
- absolute path, byte size, and SHA-256 for raw `torchrun.out`, `torchrun.err`,
  and `resources.log`, plus any other large log not committed;
- final `git status --short`, pushed evidence commit, and local/remote branch
  synchronization.

## Final instruction

Commit and push the complete evidence packet, set this packet state to
`RETURNED_FOR_ANALYSIS`, and stop. Do not retry, patch, start full calibration,
or launch downstream evaluation/performance work unless a new planner packet
explicitly authorizes it.

## Planner analysis and r3 authorization

The r2 evidence isolates a deterministic compatibility failure before model
weights, calibration, or quantization work. On non-source ranks,
compressed-tensors changes the load to `device_map="meta"` and injects
`tie_word_embeddings=False` to avoid Transformers comparing meta tensors.
Transformers 5.12.1 forwards that otherwise-unconsumed keyword into
`MiniMaxM3SparseForConditionalGeneration.__init__`, whose constructor does not
accept it.

Commit `4801028f` installs a narrow wrapper below compressed-tensors' loader. It
removes the injected keyword only when `config.model_type == "minimax_m3_vl"`
and writes the same value to both the top-level and text configs, preserving the
meta-rank safety intent. Other models and source-rank loads are unchanged. The
regression test reproduces the exact wrapper ordering and fails with the r2
exception without the fix.

Executor instructions for r3:

1. Pull `duy-branch` and require `4801028f` as an ancestor.
2. Run the complete setup, preflight, focused tests, and dry run above. The
   focused suite must include and pass
   `test_minimax_meta_rank_moves_tie_word_embeddings_to_config`.
3. If preflight passes, execute exactly one fresh r3 run using the commands
   above. Do not reuse or overwrite the r2 roots.
4. Preserve and return evidence through the same aggregation contract. Do not
   add a runtime probe, change model/calibration/layer/topology inputs, or retry
   again if r3 fails.

## Executor evidence: 2026-07-14 r2

- Protocol state: `RETURNED_FOR_ANALYSIS`
- Packet revision: `2026-07-15-r2`
- Expected ancestor: `0c8c3186`
- Actual Git commit: `639061684db93a504a0f6e9b636e0230f9815786`
- Execution classification: both evidence-only arms failed during model loading
- Run ID: `20260714T164500Z-m3-ddp-quant-smoke-r2`
- Controller session: `m3-ddp-quant-20260714T164500Z-m3-ddp-quant-smoke-r2`
- Result root:
  `/mnt/nfs/hoangduy/results/m3-distributed-quant-smoke/20260714T164500Z-m3-ddp-quant-smoke-r2`
- Log root:
  `/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/20260714T164500Z-m3-ddp-quant-smoke-r2`
- Evidence root:
  `results/m3-distributed-quant-speedup/20260714T164500Z-m3-ddp-quant-smoke-r2`

### Gates and topology

- Setup, revision, workspace, model-input, and environment checks: passed.
- Config/import assertions: passed.
- Focused tests: `39 passed in 0.55s`.
- Launcher syntax: passed.
- Dry-run: passed; exactly two sequential top-level `srun` commands, each
  requesting one exclusive node with eight GPUs, each invoking eight-rank
  `torchrun` and `--evidence-only`; no representative harness.
- GPTQ: Slurm job `12921`, step `12921.0`, node `h108`, 8 H100 GPUs,
  approximately 67 seconds, return code `1`.
- AWQ: Slurm job `12922`, step `12922.0`, node `h108`, 8 H100 GPUs,
  approximately 68 seconds, return code `1`.
- Controller return code: `1`.
- The unrelated paired quality jobs remained independent and were not changed.

### First failure and last successful stage

Both arms reached node preflight, initialized eight local ranks, and entered
model loading. The first observed operation failure in both arms was:

```text
TypeError: MiniMaxM3SparseForConditionalGeneration.__init__() got an unexpected keyword argument 'tie_word_embeddings'
```

The failure occurred through `pipeline.quantize._load_model_and_tokenizer`,
`linearize.py`, `compressed_tensors.offload.load`, and Transformers
`from_pretrained`. Torchrun terminated the sibling ranks after the first rank
failure. No calibration partition, native quantization, provenance, or
completion stage was reached.

### Environment and artifacts

- Python `3.12.13`
- llmcompressor `0.1.dev3101+g46e6ba4`
- compressed-tensors `0.17.2a20260707`
- torch `2.11.0`, CUDA build `13.0`
- transformers `5.12.1`
- GPUs: eight NVIDIA H100 80GB HBM3, driver `580.126.09`
- Node preflight: `MemAvailable=2079369052 kB`; `/dev/shm` available
  `1075820175360` bytes.
- No OOM, cancellation, retry, or scheduler submission failure was observed.
- No usable checkpoint was produced.
- Missing expected artifacts for both methods: `smoke_complete.json`, eight
  partition manifests, native metric files, and model-provenance files.
- Raw `torchrun.out`, `torchrun.err`, and `resources.log` paths, byte sizes, and
  SHA-256 hashes are recorded in `aggregate.json`.
- The complete committed small evidence tree, including scheduler records,
  environment, commands, return codes, controller output, preflight summary,
  dry-run, and `small_artifacts.json`, is under the evidence root above.

### Deviations and interpretation boundary

No packet deviation, code/config edit, retry, full calibration, checkpoint
save, quality evaluation, or performance run was performed. This is factual
infrastructure/model-load evidence only; it is not a quantization-speed or
model-quality verdict.
