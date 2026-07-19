# Execution packet: Hopper NVFP4 W4A8 Humming proof probe

- Protocol version: 1
- State: PLANNER_ANALYSIS
- Packet revision: 2026-07-19-r1
- Planner owner: Codex planner
- Intended executor: any cluster executor
- Base Git commit: `TO_BE_REPLACED_AFTER_IMPLEMENTATION_COMMIT`
- Decision question: Does the patched Humming A8/E2M1/g16 specialization
  compile on SM90 and satisfy exact K16 scale isolation, numerical,
  packed-memory, and WGMMA instruction gates?

This is the only active packet for this task. There are no older executor
instructions for this specialization.

## Objective and hypothesis

Run one deterministic dense `128 x 128` Humming kernel probe at
`M in [1, 8, 32]`. The hypothesis is that the pinned six-file Humming `0.1.10`
overlay preserves packed E2M1 weights and distinct K16 scales, converts/scales
the B fragment in E4M3 registers, accumulates with Hopper FP8 WGMMA into FP32,
and returns BF16 output matching the exact emulated contract.

The run decides only whether this implementation is correct enough to justify a
later, separately authorized latency experiment.

## Scope and non-goals

- In scope: package/source preflight, overlay installation, one SM90 JIT compile,
  deterministic one-layer correctness cases, layer-transform and per-fragment
  K16 isolation, persistent-byte accounting, cubin/SASS capture, and evidence
  packaging.
- Not authorized: model download, full-model serving, MoE, re-quantization,
  performance benchmarking, quality evaluation, tolerance changes, kernel
  edits, or any retry.

## Preconditions and exact environment

- Repository path: `/mnt/nfs/hoangduy/projects/llm-compressor`
- Branch: `duy-branch`
- Environment activation:
  `source /mnt/nfs/hoangduy/env.sh && source /mnt/nfs/hoangduy/venvs/quant/bin/activate`
- Required environment variables: `HOME=${WORK_ROOT:-/mnt/nfs/hoangduy}` and
  `PYTHONPATH=/mnt/nfs/hoangduy/projects/llm-compressor`
- Required Humming distribution: exactly `humming-kernels==0.1.10`
- Required device: compute capability exactly `(9, 0)`
- Required tools: `srun`, `tmux`, `cuobjdump`, `sha256sum`, Python, PyTorch CUDA

## Required inputs

| Input | Exact path or identifier | Required validation |
| --- | --- | --- |
| Planner code | Base commit above on `duy-branch` | exact revision checks below |
| Humming wheel | quant venv, distribution `humming-kernels` | version and six-file overlay check |
| Probe | `pipeline/hopper_nvfp4_w4a8/gpu_probe.py` | local compile and launcher preflight |
| Scheduler launcher | `pipeline/slurm/run_hopper_nvfp4_w4a8_probe_srun.sh` | `bash -n`; exactly one 1-GPU `srun` |

No checkpoint, model, dataset, tokenizer, or network download is an input.

## Workspace policy

- Protected paths: all tracked repository files, the quant venv except the six
  overlay targets and their named backups, and all prior result roots.
- Permitted untracked roots: none inside the repository before launch. After
  launch, only `pipeline/results/hopper_nvfp4_w4a8/<RUN_ID>/` may be created for
  the returned evidence copy.
- Record and proceed: untracked Python bytecode/cache directories only if
  `git status --porcelain` proves they are ignored; record `git status
  --ignored --short` and continue.
- Stop: any staged/unstaged tracked change, any non-ignored untracked path, a
  pre-existing selected result root, a revision mismatch, or an unknown/partial
  Humming overlay.

## Resource contract

- Nodes: 1
- GPUs per node: 1
- Exclusivity: shared node; one `--gres=gpu:1` allocation
- Task/process layout: one node, one task, one process, no distributed workers
- Concurrency: one probe only
- Time limit: `00:30:00`
- Expected runtime: 5–15 minutes including first JIT compile

## Commands

### Setup and revision verification

```bash
set -euo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
cd "$REPO"
git fetch origin
git checkout duy-branch
git pull --ff-only origin duy-branch
EXPECTED=TO_BE_REPLACED_AFTER_IMPLEMENTATION_COMMIT
ACTUAL="$(git rev-parse HEAD)"
printf 'expected=%s\nactual=%s\n' "$EXPECTED" "$ACTUAL"
test "$ACTUAL" = "$EXPECTED"
git status --porcelain=v1 --untracked-files=all
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

Expected: expected and actual full SHAs are identical; status is empty. Stop
before environment mutation if either condition fails.

### Preflight

```bash
set -euo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"
export PYTHONPATH="$REPO"
cd "$REPO"
python - <<'PY'
import importlib.metadata
assert importlib.metadata.version("humming-kernels") == "0.1.10"
print("humming-kernels=0.1.10")
PY
bash pipeline/slurm/install_humming_nvfp4_w4a8.sh
python pipeline/slurm/patch_humming_nvfp4_w4a8.py --check
python -m compileall -q pipeline/hopper_nvfp4_w4a8 \
  pipeline/slurm/patch_humming_nvfp4_w4a8.py
bash -n pipeline/slurm/install_humming_nvfp4_w4a8.sh
bash -n pipeline/slurm/run_hopper_nvfp4_w4a8_probe_srun.sh
command -v srun
command -v tmux
command -v cuobjdump
command -v sha256sum
```

Expected: every return code is zero, the overlay report says `patched`, and no
GPU allocation has occurred. Stop and return exact stdout/stderr and versions
on any failure.

### Dry run

```bash
set -euo pipefail
cd /mnt/nfs/hoangduy/projects/llm-compressor
test "$(grep -c '^srun ' pipeline/slurm/run_hopper_nvfp4_w4a8_probe_srun.sh)" -eq 1
grep -n -- '--nodes=1\|--ntasks=1\|--gres=gpu:1\|--time=00:30:00' \
  pipeline/slurm/run_hopper_nvfp4_w4a8_probe_srun.sh
grep -n 'M_VALUES = (1, 8, 32)' \
  pipeline/hopper_nvfp4_w4a8/gpu_probe.py
```

Expected: one top-level allocation and the exact topology/shape markers. This
does not submit a scheduler job.

### Launch

```bash
set -euo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SESSION="hopper-nvfp4-w4a8-${STAMP}"
CONTROLLER_LOG="/mnt/nfs/hoangduy/hopper_nvfp4_w4a8_probe/controller-${STAMP}.log"
CONTROLLER_RC="/mnt/nfs/hoangduy/hopper_nvfp4_w4a8_probe/controller-${STAMP}.returncode"
test ! -e "$CONTROLLER_LOG"
test ! -e "$CONTROLLER_RC"
tmux new-session -d -s "$SESSION" \
  "cd '$REPO'; bash pipeline/slurm/run_hopper_nvfp4_w4a8_probe_srun.sh > '$CONTROLLER_LOG' 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > '$CONTROLLER_RC'; exit \"\$rc\""
printf 'SESSION=%s\nCONTROLLER_LOG=%s\nCONTROLLER_RC=%s\n' \
  "$SESSION" "$CONTROLLER_LOG" "$CONTROLLER_RC"
```

The launcher creates a fresh run ID and result root beneath
`/mnt/nfs/hoangduy/hopper_nvfp4_w4a8_probe/`; never supply or reuse an old root.

### Monitoring

```bash
tmux list-sessions | grep 'hopper-nvfp4-w4a8-'
squeue -u "$USER" -o '%.18i %.12P %.28j %.8T %.10M %.6D %R'
tail -n 100 "$CONTROLLER_LOG"
```

Wait for the named tmux session to exit. Do not attach an owning shell, alter
the job, or launch another attempt.

### Aggregation and packaging

```bash
set -euo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
RESULT_ROOT="$(grep '^RESULT_ROOT=' "$CONTROLLER_LOG" | tail -n 1 | cut -d= -f2-)"
test -n "$RESULT_ROOT"
test -d "$RESULT_ROOT"
RUN_ID="$(basename "$RESULT_ROOT")"
EVIDENCE="$REPO/pipeline/results/hopper_nvfp4_w4a8/$RUN_ID"
test ! -e "$EVIDENCE"
mkdir -p "$(dirname "$EVIDENCE")"
cp -a "$RESULT_ROOT" "$EVIDENCE"
cp "$CONTROLLER_LOG" "$EVIDENCE/controller.log"
cp "$CONTROLLER_RC" "$EVIDENCE/controller.returncode"
find "$EVIDENCE" -type f -printf '%P %s bytes\n' | sort
test "$(tr -d '\r\n' < "$EVIDENCE/controller.returncode")" = "0"
test "$(tr -d '\r\n' < "$EVIDENCE/srun.returncode")" = "0"
test "$(tr -d '\r\n' < "$EVIDENCE/validation.returncode")" = "0"
(
  cd "$EVIDENCE"
  sha256sum -c SHA256SUMS
  find . -type f ! -name EVIDENCE_SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > EVIDENCE_SHA256SUMS
  sha256sum -c EVIDENCE_SHA256SUMS
)
python - "$EVIDENCE/probe.json" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
print(json.dumps(p, indent=2, sort_keys=True))
PY
```

If the full result root exceeds 20 MiB, copy all JSON, return-code, text log,
SASS, revision, environment, scheduler, and manifest files; leave larger files
on shared storage and record absolute path, byte size, and SHA-256.

## Expected jobs and independence rules

| Job or arm | Resources | Expected output | Failure effect |
| --- | --- | --- | --- |
| one-layer proof probe | 1 node, 1 H100, 1 task | `probe.json`, SASS, logs, manifest | stop; no retry or downstream work |

## Success gates and expected artifacts

- Overlay/package gate: exact Humming `0.1.10`, all six sources fully patched.
- Hardware gate: `device_capability == [9, 0]`.
- Process gate: controller, scheduler step, probe, and validator return zero.
- Numerical gate: exact-emulation comparison passes at fixed
  `rtol=0.01, atol=0.25` for every M; outputs are entirely finite.
- Isolation gate: changing one K16 scale changes its isolated contribution and
  leaves the adjacent isolated K16 contribution unchanged; sentinel changes
  across every N16 fragment alter exactly their selected output row.
- Load-transform gate: Humming's public layer transform consumes packed-int32
  checkpoint weights, leaves the checkpoint global scale untouched, and emits
  the effective runtime global scale multiplied by exactly eight.
- Determinism gate: repeated launches are byte-identical for every M.
- Memory gate: actual persistent transformed/checkpoint ratio `<= 1.10` and no
  persistent FP8-expanded weight tensor exists.
- Instruction gate: `cuobjdump` succeeds and SASS contains FP8
  `WGMMA.MMA_ASYNC` with E4M3 operands.
- Evidence gate: raw JSON, stdout/stderr, SASS, environment/revision records,
  scheduler/controller output, return codes, and verified SHA-256 manifest exist.

The BF16-dequantized NVFP4 comparison is recorded but is not a tunable gate in
this run.

## Allowed adaptations

- None.

## Pre-authorized record-and-proceed conditions

- Scheduler queue wait without allocation may be recorded and allowed to
  continue within the same controller/run ID.
- Ignored Python cache files may be recorded as described in workspace policy.

## Pre-authorized retries

- Trigger: none
- Maximum retry count: 0
- Fresh run ID required: yes for any later planner-authorized packet
- Inputs that must remain unchanged: all code, thresholds, topology, and seeds

## Stop-and-return conditions

- Any setup/preflight/dry-run failure or unknown/partial source state.
- Device capability other than exactly SM90.
- Any compile, launch, JSON validation, isolation, numerical, determinism,
  memory, SASS, or manifest failure.
- Any missing evidence, nonzero return code, topology mismatch, collision,
  material ambiguity, or need to edit/retry.
- The one probe finishes, whether it passes or fails.

## Prohibited actions

- Do not edit source, thresholds, scales, seeds, shapes, configs, or packet.
- Do not retry, substitute launch methods, add GPUs, or change the time limit.
- Do not download a model or dataset, serve a model, run timing, run quality
  evaluation, run MoE, re-quantize weights, or start follow-on work.
- Do not use `sbatch`.

## Return contract

- Commit the small evidence tree under
  `pipeline/results/hopper_nvfp4_w4a8/<RUN_ID>/` and the factual evidence update
  to this handoff.
- Record exact executed commands, expected/actual revision, environment/package
  and driver versions, scheduler step/node/topology/timings, return codes,
  measurements, gate results, deviations, retries (`none`), and first failure.
- For any artifact left on shared storage, record absolute path, byte size, and
  SHA-256. Never replace raw evidence with a prose summary.
- On a pre-GPU stop, record the exact blocker command, value, path, and return
  code and state explicitly that no scheduler job launched.

## Final instruction

After the bounded run or first stop condition, update this packet to
`RETURNED_FOR_ANALYSIS`, add the protocol evidence fields, commit and push the
complete evidence packet, and stop. Do not retry, patch, time, serve, or launch
downstream work without a new planner packet.
