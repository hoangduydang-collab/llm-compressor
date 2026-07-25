# Execution packet: MiniMax-M3 native Humming W4A8 qualification

- Protocol version: 1
- State: `READY_FOR_EXECUTOR`
- Packet revision: `2026-07-25-r1`
- Planner owner: Codex planner
- Intended executor: any authorized cluster executor
- Base Git commit: `fc31590df3f5b5ee21f64a5e2d804e4223cbb592`
- Required branch: `duy-branch`
- Decision question: Does the pinned Humming path load the in-house GPTQ W4A8 checkpoint on TP8 plus expert parallelism, positively attest indexed Humming MoE selection, and complete ten repeated graphs-on HTTP correctness smokes without non-finite, empty, degenerate, or failed output?

## Objective and hypothesis

Qualify the already-implemented native Humming path before spending cluster time
on the paired performance benchmark. The hypothesis is that vLLM 0.24.0 and
pristine `humming-kernels==0.1.10` can consume the existing compressed-tensors
GPTQ checkpoint directly, preserve packed INT4 group-128 routed-expert weights
and dynamic per-token E4M3 activations, use Humming's indexed MoE GEMMs, and
survive the normal MiniMax-M3 graphs-on TP8/EP serving envelope.

This packet answers compatibility, backend identity, bounded correctness, and
stability only. It does not answer whether Humming is faster than CUTLASS.

## Scope and non-goals

In scope:

- verify the exact Git revision, package pins, pristine Humming wheel contents,
  checkpoint metadata, selector, and effective serving command;
- apply/check the existing MiniMax-M3 vLLM patches plus the optional Humming
  activation-admission patch and dormant MoE probe;
- launch one TP8 plus expert-parallel Humming HTTP server on one exclusive
  8xH100 node with CUDA graphs enabled;
- require positive Humming preflight and backend attestation;
- run ten sequential fixed HTTP correctness probes;
- preserve raw logs, responses, environment, scheduler state, return codes,
  hashes, and the first failure boundary.

Not authorized:

- the performance benchmark or any aiperf workload;
- a CUTLASS comparison, Marlin arm, BF16 arm, quality evaluation, or
  re-quantization;
- a package install, upgrade, downgrade, or venv mutation other than the
  already-approved idempotent vLLM source patches;
- a Humming source patch, the NVFP4 overlay, a CUDA/C++ change, or a custom
  kernel;
- grouped-contiguous Humming GEMM, FP16 accumulation, eager mode, a changed
  checkpoint, changed TP/EP topology, or changed smoke content gate;
- an automatic fallback, retry, resume, or replacement run ID.

## Preconditions and exact environment

- Repository: `/mnt/nfs/hoangduy/projects/llm-compressor`
- Branch: `duy-branch`
- Environment activation:
  `source /mnt/nfs/hoangduy/venvs/quant/bin/activate`
- Scheduler: top-level `srun` owned by a detached `tmux` controller started
  outside Slurm.
- `sbatch` is prohibited.
- Required vLLM version: `0.24.0`
- Required Humming version: `0.1.10`
- Required accelerator: eight H100-class GPUs reporting capability `(9, 0)`.
- Required backend: `M3_W4A8_BACKEND=humming`
- Required MoE GEMM: `VLLM_HUMMING_MOE_GEMM_TYPE=indexed`
- Required accumulation:
  `VLLM_HUMMING_USE_F16_ACCUM=0`
- Required JIT cache namespace:
  `/mnt/nfs/hoangduy/.humming/cache-m3-gptq-w4a8-v1`

## Required inputs

| Input | Exact path or identifier | Required validation |
| --- | --- | --- |
| GPTQ checkpoint | `/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay` | Directory, `config.json`, and `model.safetensors.index.json` exist; CPU ABI and launcher dry-run gates pass |
| Original model metadata/tokenizer | `/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3` | Directory and `config.json` exist |
| HTTP launcher | `pipeline/slurm/run_vllm_http_serve_smoke.sh` | Tracked at the verified revision |
| Patch manager | `pipeline/slurm/patch_vllm_m3_serve.py` | Focused CPU tests pass; `--humming --probe` succeeds inside allocation |
| Preflight/attestor | `pipeline/m3_humming_w4a8.py` | Focused CPU tests pass |
| Smoke client | `pipeline/slurm/smoke_chat_completions.sh` | Tracked at the verified revision |

## Workspace policy

Tracked changes, staged changes, and untracked files other than an optional
repository-root `AGENTS.md` are blocking. `AGENTS.md` is record-and-proceed only
because it is the user-owned instruction file and does not shadow a tracked
path, required input, or output.

Protected paths:

- all tracked repository files;
- both required model/checkpoint directories;
- `/mnt/nfs/hoangduy/venvs/quant` except for the exact idempotent vLLM patches
  applied by `patch_vllm_m3_serve.py --humming --probe`;
- every prior result under
  `/mnt/nfs/hoangduy/results/m3-humming-w4a8-qualification`.

The only permitted fresh output is the one packet-owned result root and the
later small evidence directory under `evidence/m3-humming-w4a8/$RUN_ID`.
Any result-root collision is a stop. Do not delete, rename, reuse, or choose a
replacement ID after a collision.

## Resource contract

- Nodes: 1
- GPUs per node: 8
- Exclusivity: `--exclusive`
- Task layout: one Slurm task, TP8, expert parallelism enabled
- CPU allocation: 192 CPUs
- Concurrency: one server and one sequential client stream; no other packet arm
- Time limit: `03:00:00`
- Readiness timeout: 5,400 seconds
- Expected runtime: 30–120 minutes, including first Humming JIT compilation and
  CUDA-graph capture

## Setup and revision verification

Run from one persistent login shell outside any allocation. Keep this shell for
all later controller commands.

```bash
set -euo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
cd "$REPO"
git switch duy-branch
git pull --ff-only

CODE_BASE=fc31590df3f5b5ee21f64a5e2d804e4223cbb592
ACTUAL_COMMIT="$(git rev-parse HEAD)"
ORIGIN_COMMIT="$(git rev-parse origin/duy-branch)"
test "$ACTUAL_COMMIT" = "$ORIGIN_COMMIT"
git merge-base --is-ancestor "$CODE_BASE" "$ACTUAL_COMMIT"
python - "$CODE_BASE" "$ACTUAL_COMMIT" <<'PY'
import subprocess
import sys

base, actual = sys.argv[1:]
changed = subprocess.check_output(
    ["git", "diff", "--name-only", f"{base}..{actual}"],
    text=True,
).splitlines()
expected = ["M3_HUMMING_W4A8_HANDOFF.md"]
assert changed == expected, {"expected": expected, "actual": changed}
PY

git diff --quiet
git diff --cached --quiet
python - <<'PY'
import subprocess

lines = subprocess.check_output(
    ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    text=True,
).splitlines()
assert lines in ([], ["?? AGENTS.md"]), lines
print({"workspace_status": lines})
PY

test -z "${SLURM_JOB_ID:-}"
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$REPO"
python --version
python - <<'PY'
import importlib.metadata
import vllm

assert vllm.__version__ == "0.24.0", vllm.__version__
version = importlib.metadata.version("humming-kernels")
assert version == "0.1.10", version
print({"vllm": vllm.__version__, "humming-kernels": version})
PY
python - <<'PY'
from pipeline.m3_humming_w4a8 import _distribution_integrity

status, mismatches, overlay = _distribution_integrity()
assert status == "record-matched", (status, mismatches)
assert not mismatches, mismatches
assert overlay is False, "LLMC_NVFP4_W4A8_G16_V1 overlay is installed"
print(
    {
        "humming_source_integrity": status,
        "mismatches": mismatches,
        "nvfp4_overlay": overlay,
    }
)
PY
```

Expected: actual and origin revisions match, the only post-base change is this
handoff, tracked state is clean, the only optional untracked file is
`AGENTS.md`, package pins match, all Humming wheel files match their installed
`RECORD`, and the NVFP4 overlay is absent.

Stop before any allocation if any command fails. A missing/wrong/locally
modified Humming package is blocker evidence; do not install or repair it.

## CPU suite and checkpoint gate

```bash
set -euo pipefail
cd "$REPO"
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
MODEL_ID=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
test -d "$CKPT"
test -f "$CKPT/config.json"
test -f "$CKPT/model.safetensors.index.json"
test -d "$MODEL_ID"
test -f "$MODEL_ID/config.json"

python -m pytest -q \
  pipeline/tests/test_patch_vllm_m3_serve.py \
  pipeline/tests/test_m3_serve_abi.py \
  pipeline/tests/test_m3_humming_w4a8.py \
  pipeline/tests/test_run_vllm_http_serve_smoke.py \
  pipeline/tests/test_perf_eval_humming_contract.py

python -m pipeline.m3_serve_abi \
  --checkpoint "$CKPT" \
  --out /tmp/m3-humming-w4a8-abi.json
python - /tmp/m3-humming-w4a8-abi.json <<'PY'
import json

report = json.load(open("/tmp/m3-humming-w4a8-abi.json"))
assert report["valid"] is True, report["errors"]
routed = report["components"]["routed_experts"]
assert routed["quantized"] > 0, routed
print(report)
PY
```

Expected: the focused suite passes and the existing serving ABI report is
valid with quantized routed experts. `/tmp/m3-humming-w4a8-abi.json` is
read-only preflight evidence and is copied into the fresh result root below.

## Fresh result root and collision gate

```bash
set -euo pipefail
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-humming-w4a8-qualification-r1"
ROOT="/mnt/nfs/hoangduy/results/m3-humming-w4a8-qualification/$RUN_ID"
SESSION="m3-humming-w4a8-${RUN_ID%%-*}"
test ! -e "$ROOT"
! tmux has-session -t "$SESSION" 2>/dev/null
mkdir -p "$ROOT/responses" "$ROOT/scheduler"
cp /tmp/m3-humming-w4a8-abi.json "$ROOT/abi.json"
printf '%s\n' "$ACTUAL_COMMIT" >"$ROOT/actual-commit.txt"
printf '%s\n' "$ROOT" >"$ROOT/result-root.txt"
printf '%s\n' "$SESSION" >"$ROOT/tmux-session.txt"
```

Stop if either the result root or session already exists. Do not choose another
name in this packet.

## Dry run

```bash
set -euo pipefail
cd "$REPO"
M3_W4A8_BACKEND=humming \
PRINT_EFFECTIVE_CONFIG=1 \
CKPT="$CKPT" \
MODEL_ID="$MODEL_ID" \
SERVED_NAME=MiniMaxAI/MiniMax-M3 \
PORT=8000 \
LOG="$ROOT/serve.log" \
PID_FILE="$ROOT/serve.pid" \
bash pipeline/slurm/run_vllm_http_serve_smoke.sh \
  >"$ROOT/effective-config.txt" 2>"$ROOT/effective-config.err"

python - "$ROOT/effective-config.txt" <<'PY'
import sys

text = open(sys.argv[1]).read()
assert "M3_W4A8_BACKEND=humming" in text
assert "VLLM_HUMMING_USE_F16_ACCUM=0" in text
assert "VLLM_HUMMING_MOE_GEMM_TYPE=indexed" in text
assert "cache-m3-gptq-w4a8-v1" in text
argv = next(line for line in text.splitlines() if line.startswith("EFFECTIVE_ARGV:"))
assert argv.count("--quantization humming") == 1, argv
assert "--tensor-parallel-size 8" in argv, argv
assert "--enable-expert-parallel" in argv, argv
assert "--enforce-eager" not in argv, argv
print(argv)
PY
```

Expected: the command selects Humming exactly once, TP8 and expert parallelism
are enabled, indexed GEMM and FP32 accumulation are explicit, the dedicated JIT
cache is selected, and eager mode is absent.

## Node payload preparation

Create the exact packet-owned node script:

```bash
cat >"$ROOT/node-qualification.sh" <<'NODE'
#!/usr/bin/env bash
set -uo pipefail

REPO=/mnt/nfs/hoangduy/projects/llm-compressor
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
MODEL_ID=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
ROOT="${ROOT:?}"
PORT=8000
SERVED_NAME=MiniMaxAI/MiniMax-M3
LOG="$ROOT/serve.log"
PID_FILE="$ROOT/serve.pid"
NODE_RC=1

cleanup() {
  if [ -f "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ]; then
      kill "$pid" 2>/dev/null || true
      sleep 15
      kill -9 -- "-$pid" 2>/dev/null || true
    fi
  fi
  printf '%s\n' "$NODE_RC" >"$ROOT/node.rc"
}
trap cleanup EXIT

cd "$REPO"
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$REPO"
export M3_W4A8_BACKEND=humming
export VLLM_HUMMING_USE_F16_ACCUM=0
export VLLM_HUMMING_MOE_GEMM_TYPE=indexed
export HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming
export M3_LOAD_AUDIT=1
export M3_MOE_PROBE=1
export M3_MOE_PROBE_RECOMPUTE=1
export M3_MOE_PROBE_MAX_TOKENS=256

hostname >"$ROOT/hostname.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$ROOT/node-start-utc.txt"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free \
  --format=csv >"$ROOT/nvidia-smi-before.csv"
python - <<'PY' >"$ROOT/versions.txt"
import importlib.metadata
import torch
import vllm

print("vllm=" + vllm.__version__)
print("humming-kernels=" + importlib.metadata.version("humming-kernels"))
print("torch=" + torch.__version__)
print("cuda=" + str(torch.version.cuda))
print("device_capability=" + repr(tuple(torch.cuda.get_device_capability())))
assert vllm.__version__ == "0.24.0"
assert importlib.metadata.version("humming-kernels") == "0.1.10"
assert tuple(torch.cuda.get_device_capability()) == (9, 0)
PY
rc=$?
if [ "$rc" != 0 ]; then
  echo "runtime version/capability gate rc=$rc" >"$ROOT/first-failure.txt"
  NODE_RC=$rc
  exit "$NODE_RC"
fi

python pipeline/slurm/patch_vllm_m3_serve.py --humming --probe \
  >"$ROOT/patch-apply.log" 2>&1
rc=$?
if [ "$rc" != 0 ]; then
  echo "patch apply rc=$rc" >"$ROOT/first-failure.txt"
  NODE_RC=$rc
  exit "$NODE_RC"
fi
python - <<'PY' >>"$ROOT/patch-apply.log" 2>&1
from pipeline.slurm.patch_vllm_m3_serve import (
    ensure_m3_load_audit,
    ensure_m3_moe_probe,
)

for ensure in (ensure_m3_load_audit, ensure_m3_moe_probe):
    applied = ensure(apply=True)
    checked = ensure(apply=False)
    print(ensure.__name__, "apply:", applied)
    print(ensure.__name__, "check:", checked)
    assert "skipped" not in checked
    assert "NOT injected" not in checked
PY
rc=$?
if [ "$rc" != 0 ]; then
  echo "diagnostic patch apply/check rc=$rc" >"$ROOT/first-failure.txt"
  NODE_RC=$rc
  exit "$NODE_RC"
fi
python pipeline/slurm/patch_vllm_m3_serve.py --check --humming \
  >"$ROOT/patch-check.log" 2>&1
rc=$?
if [ "$rc" != 0 ]; then
  echo "patch check rc=$rc" >"$ROOT/first-failure.txt"
  NODE_RC=$rc
  exit "$NODE_RC"
fi

CKPT="$CKPT" \
MODEL_ID="$MODEL_ID" \
SERVED_NAME="$SERVED_NAME" \
PORT="$PORT" \
LOG="$LOG" \
PID_FILE="$PID_FILE" \
bash pipeline/slurm/run_vllm_http_serve_smoke.sh \
  >"$ROOT/launcher.log" 2>&1
rc=$?
if [ "$rc" != 0 ]; then
  echo "serve launcher rc=$rc" >"$ROOT/first-failure.txt"
  NODE_RC=$rc
  exit "$NODE_RC"
fi

ready=1
for _ in $(seq 1 540); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" \
      -o "$ROOT/models.json" 2>/dev/null; then
    ready=0
    break
  fi
  if ! kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    echo "server died before readiness" >"$ROOT/first-failure.txt"
    break
  fi
  sleep 10
done
if [ "$ready" != 0 ]; then
  NODE_RC=1
  exit "$NODE_RC"
fi
date -u +%Y-%m-%dT%H:%M:%SZ >"$ROOT/ready-utc.txt"

python -m pipeline.m3_humming_w4a8 attest \
  --preflight "$ROOT/serve.log.humming-preflight.json" \
  --log "$LOG" \
  --out "$ROOT/backend-attestation.json"
rc=$?
if [ "$rc" != 0 ]; then
  echo "backend attestation rc=$rc" >"$ROOT/first-failure.txt"
  NODE_RC=$rc
  exit "$NODE_RC"
fi

for index in $(seq -w 1 10); do
  out="$ROOT/responses/smoke-$index.out"
  MODEL="$SERVED_NAME" \
  PORT="$PORT" \
  PROMPT="What is 2+2? Answer briefly." \
  MAX_TOKENS=256 \
  TEMPERATURE=0.0 \
  bash pipeline/slurm/smoke_chat_completions.sh >"$out" 2>&1
  rc=$?
  printf '%s\n' "$rc" >"$ROOT/responses/smoke-$index.rc"
  if [ "$rc" != 0 ]; then
    echo "smoke $index HTTP rc=$rc" >"$ROOT/first-failure.txt"
    NODE_RC=$rc
    exit "$NODE_RC"
  fi
  python - "$out" "$ROOT/responses/smoke-$index.gate.json" <<'PY'
import json
import re
import sys

raw = open(sys.argv[1]).read()
end = raw.rfind("}")
body = None
for match in re.finditer(r"^\{", raw, re.MULTILINE):
    try:
        body = json.loads(raw[match.start() : end + 1])
        break
    except json.JSONDecodeError:
        continue
assert body is not None, "no parseable chat-completion JSON"
message = body["choices"][0]["message"]
content = " ".join(
    part for part in (message.get("content"), message.get("reasoning")) if part
).strip()
assert content, "empty completion"
assert re.search(r"(^|[^0-9])4([^0-9]|$)|four", content, re.IGNORECASE), content[:300]
diversity = 1.0
if len(content) > 200:
    grams = {content[i : i + 8] for i in range(len(content) - 7)}
    diversity = len(grams) / (len(content) - 7)
    assert diversity >= 0.2, (diversity, content[:300])
report = {
    "valid": True,
    "content_chars": len(content),
    "eight_gram_diversity": diversity,
}
open(sys.argv[2], "w").write(json.dumps(report, indent=2) + "\n")
PY
  rc=$?
  if [ "$rc" != 0 ]; then
    echo "smoke $index content gate rc=$rc" >"$ROOT/first-failure.txt"
    NODE_RC=$rc
    exit "$NODE_RC"
  fi
done

grep -q "Capturing CUDA graphs" "$LOG" || {
  echo "CUDA graph capture marker missing" >"$ROOT/first-failure.txt"
  NODE_RC=1
  exit "$NODE_RC"
}
grep -q "M3_LOAD_AUDIT#" "$LOG" || {
  echo "load-audit marker missing" >"$ROOT/first-failure.txt"
  NODE_RC=1
  exit "$NODE_RC"
}
grep -q "M3_MOE_PROBE#" "$LOG" || {
  echo "MoE-probe marker missing" >"$ROOT/first-failure.txt"
  NODE_RC=1
  exit "$NODE_RC"
}
if grep -Eiq \
    "M3_MOE_PROBE_NONFINITE|illegal memory access|CUDA error|Traceback" \
    "$LOG"; then
  echo "fatal/non-finite server-log marker present" >"$ROOT/first-failure.txt"
  NODE_RC=1
  exit "$NODE_RC"
fi
kill -0 "$(cat "$PID_FILE")"
rc=$?
if [ "$rc" != 0 ]; then
  echo "server died after smokes" >"$ROOT/first-failure.txt"
  NODE_RC=$rc
  exit "$NODE_RC"
fi

nvidia-smi --query-gpu=index,name,memory.total,memory.free \
  --format=csv >"$ROOT/nvidia-smi-after.csv"
date -u +%Y-%m-%dT%H:%M:%SZ >"$ROOT/node-end-utc.txt"
NODE_RC=0
exit 0
NODE
chmod 700 "$ROOT/node-qualification.sh"
```

The payload performs no allocation itself and may run only under the one
packet-owned top-level `srun`.

## Launch

```bash
set -euo pipefail
cd "$REPO"
export ROOT
PAYLOAD="cd '$REPO'; srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 --time=03:00:00 --kill-on-bad-exit=1 --job-name=m3-humming-w4a8-r1 --export=ALL,ROOT='$ROOT' bash '$ROOT/node-qualification.sh' >'$ROOT/srun.out' 2>'$ROOT/srun.err'; rc=\$?; printf '%s\n' \$rc >'$ROOT/srun.rc'; exit \$rc"
test -z "${SLURM_JOB_ID:-}"
tmux new-session -d -s "$SESSION" "$PAYLOAD"
date -u +%Y-%m-%dT%H:%M:%SZ >"$ROOT/controller-start-utc.txt"
printf '%s\n' "$PAYLOAD" >"$ROOT/controller-command.txt"
```

Expected: one detached controller owns one top-level exclusive 8xH100 `srun`.
Do not launch a second session.

## Monitoring

Use only non-owning observation:

```bash
tmux capture-pane -pt "$SESSION" -S -200 2>/dev/null || true
tail -n 80 "$ROOT/srun.out" "$ROOT/srun.err" 2>/dev/null || true
squeue -n m3-humming-w4a8-r1 -o "%.18i %.20j %.8T %.10M %.6D %R"
sacct -X --name m3-humming-w4a8-r1 \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,AllocNodes,NodeList%30
```

Unchanged queue state, JIT compilation, or model loading is expected and is not
a retry trigger. When the tmux session ends, continue to aggregation. Do not
relaunch it.

## Expected job and independence rule

| Job | Resources | Expected output | Failure effect |
| --- | --- | --- | --- |
| `m3-humming-w4a8-r1` | 1 exclusive node, 8 H100, one task, TP8 plus EP | preflight, attestation, readiness, 10 responses, diagnostics, return codes | Stop this packet; there are no independent sibling jobs |

## Aggregation and mechanical success gate

After the controller ends:

```bash
set -euo pipefail
date -u +%Y-%m-%dT%H:%M:%SZ >"$ROOT/controller-end-utc.txt"
squeue -n m3-humming-w4a8-r1 -o "%.18i %.20j %.8T %.10M %.6D %R" \
  >"$ROOT/scheduler/squeue-final.txt" 2>&1 || true
sacct -X --name m3-humming-w4a8-r1 \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,AllocNodes,NodeList%30 \
  >"$ROOT/scheduler/sacct-final.txt" 2>&1 || true

set +e
python - "$ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
def read_rc(name):
    path = root / name
    return int(path.read_text().strip()) if path.is_file() else None

preflight_path = root / "serve.log.humming-preflight.json"
attestation_path = root / "backend-attestation.json"
preflight = json.loads(preflight_path.read_text()) if preflight_path.is_file() else None
attestation = (
    json.loads(attestation_path.read_text()) if attestation_path.is_file() else None
)
smokes = []
for index in range(1, 11):
    prefix = root / "responses" / f"smoke-{index:02d}"
    rc_path = prefix.with_suffix(".rc")
    gate_path = prefix.with_suffix(".gate.json")
    smokes.append(
        {
            "index": index,
            "rc": int(rc_path.read_text().strip()) if rc_path.is_file() else None,
            "gate": json.loads(gate_path.read_text()) if gate_path.is_file() else None,
        }
    )
checks = {
    "srun_rc_zero": read_rc("srun.rc") == 0,
    "node_rc_zero": read_rc("node.rc") == 0,
    "preflight_valid": bool(preflight and preflight.get("valid") is True),
    "attestation_valid": bool(
        attestation
        and attestation.get("valid") is True
        and attestation.get("backend") == "humming"
        and attestation.get("gemm_type") == "indexed"
    ),
    "ten_smokes_present": len(smokes) == 10,
    "ten_smokes_valid": all(
        item["rc"] == 0
        and item["gate"]
        and item["gate"].get("valid") is True
        for item in smokes
    ),
    "no_first_failure": not (root / "first-failure.txt").exists(),
}
summary = {
    "schema_version": 1,
    "valid": all(checks.values()),
    "checks": checks,
    "preflight": preflight,
    "attestation": attestation,
    "smokes": smokes,
}
(root / "qualification-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if summary["valid"] else 1)
PY
SUMMARY_RC=$?
set -e
printf '%s\n' "$SUMMARY_RC" >"$ROOT/qualification-summary.rc"

find "$ROOT" -type f \
  ! -name sha256sums.txt \
  ! -name byte-sizes.txt \
  -print0 | sort -z | xargs -0 sha256sum \
  >"$ROOT/sha256sums.txt"
find "$ROOT" -type f \
  ! -name byte-sizes.txt \
  -printf '%s\t%p\n' | sort -k2 \
  >"$ROOT/byte-sizes.txt"
```

The aggregation command is allowed after partial failure. If its Python gate
returns nonzero, preserve `qualification-summary.json`, write the nonzero
`qualification-summary.rc`, and continue only with evidence packaging; do not
rerun any GPU command.

## Success gates and expected artifacts

All are required:

- `srun.rc == 0`
- `node.rc == 0`
- preflight `valid == true`, backend `humming`
- attestation `valid == true`, backend `humming`, GEMM type `indexed`
- server reaches `/v1/models`
- server log contains normal CUDA-graph capture, load-audit, and MoE-probe
  markers
- server log contains no non-finite probe, illegal-memory-access, CUDA-error,
  or traceback marker
- ten smoke commands return zero
- ten smoke content gates are valid, non-empty, answer 4/four, and satisfy the
  bounded degeneracy check
- server remains alive after all smokes
- `qualification-summary.json` has `valid == true`

Expected raw artifacts include:

- `effective-config.txt`
- `abi.json`
- `versions.txt`
- `patch-apply.log` and `patch-check.log`
- `serve.log.humming-preflight.json`
- `serve.log`
- `backend-attestation.json`
- `models.json`
- all response `.out`, `.rc`, and `.gate.json` files
- `launcher.log`, `srun.out`, `srun.err`, `srun.rc`, and `node.rc`
- scheduler records, GPU inventories, timestamps, summary, byte sizes, and
  SHA-256 manifest

## Allowed adaptations

None.

## Pre-authorized record-and-proceed conditions

- Optional untracked repository-root `AGENTS.md`: record it and proceed when it
  is the only workspace status entry.
- Normal queue wait, first-use Humming JIT compilation, or long model load
  within the stated timeouts: preserve observations and continue.

No other condition is record-and-proceed.

## Pre-authorized retries

- Trigger: none
- Maximum retry count: 0
- Fresh run ID required: yes for any future planner-authorized packet
- Inputs that must remain unchanged: checkpoint, model metadata, code revision,
  package versions and source integrity, TP8/EP topology, graphs-on setting,
  indexed GEMM, FP32 accumulation, prompt, generation parameters, and gates

## Stop-and-return conditions

Stop, preserve evidence, package the return, and do not retry when:

- revision, workspace, input, environment, package, source-integrity, CPU test,
  checkpoint ABI, collision, or dry-run preflight fails;
- the actual topology is not one exclusive 8xH100 node with capability `(9, 0)`;
- patch apply/check, Humming preflight, JIT, weight conversion, load, CUDA-graph
  capture, readiness, backend attestation, HTTP, content, diagnostic, or
  server-liveness gate fails;
- a CUTLASS, Marlin, grouped-contiguous, unquantized, eager, or ambiguous
  fallback marker appears;
- continuing requires any unclassified deviation, patch, install, changed
  setting, second allocation, or retry.

Capture `scontrol`, `sacct`, `squeue`, the last 200 server-log lines, process
state, `nvidia-smi`, first-failure text, and all existing partial artifacts
before returning when an abnormal exit occurs.

## Prohibited actions

- Do not use `sbatch`.
- Do not launch the performance benchmark.
- Do not run a second qualification attempt.
- Do not install, reinstall, restore, or patch Humming.
- Do not change vLLM/Humming versions, checkpoint metadata, weights, activation
  contract, model length, KV-cache dtype, TP/EP topology, CUDA-graph mode,
  accumulation, GEMM type, prompt, token cap, or content gate.
- Do not fall back to CUTLASS, Marlin, BF16, unquantized experts, eager mode, or
  a different node type.
- Do not edit tracked implementation code or any benchmark file.
- Do not delete or overwrite prior results.

## Return contract

Create `evidence/m3-humming-w4a8/$RUN_ID` after GPU work ends:

```bash
set -euo pipefail
cd "$REPO"
EVIDENCE="evidence/m3-humming-w4a8/$RUN_ID"
test ! -e "$EVIDENCE"
mkdir -p "$EVIDENCE"
cp "$ROOT/actual-commit.txt" \
   "$ROOT/result-root.txt" \
   "$ROOT/effective-config.txt" \
   "$ROOT/effective-config.err" \
   "$ROOT/abi.json" \
   "$ROOT/controller-command.txt" \
   "$ROOT/versions.txt" \
   "$ROOT/patch-apply.log" \
   "$ROOT/patch-check.log" \
   "$ROOT/serve.log.humming-preflight.json" \
   "$ROOT/backend-attestation.json" \
   "$ROOT/models.json" \
   "$ROOT/launcher.log" \
   "$ROOT/srun.out" \
   "$ROOT/srun.err" \
   "$ROOT/srun.rc" \
   "$ROOT/node.rc" \
   "$ROOT/qualification-summary.json" \
   "$ROOT/qualification-summary.rc" \
   "$ROOT/sha256sums.txt" \
   "$ROOT/byte-sizes.txt" \
   "$EVIDENCE/"
cp -R "$ROOT/responses" "$ROOT/scheduler" "$EVIDENCE/"
if [ -f "$ROOT/first-failure.txt" ]; then
  cp "$ROOT/first-failure.txt" "$EVIDENCE/"
fi

SERVE_BYTES="$(stat -c %s "$ROOT/serve.log")"
if [ "$SERVE_BYTES" -le 5242880 ]; then
  cp "$ROOT/serve.log" "$EVIDENCE/serve.log"
else
  head -n 200 "$ROOT/serve.log" >"$EVIDENCE/serve.head.log"
  tail -n 400 "$ROOT/serve.log" >"$EVIDENCE/serve.tail.log"
fi
printf '%s\t%s\t%s\n' \
  "$ROOT/serve.log" \
  "$SERVE_BYTES" \
  "$(sha256sum "$ROOT/serve.log" | awk '{print $1}')" \
  >"$EVIDENCE/large-artifacts.tsv"
```

Write `$EVIDENCE/executor-return.md` using the full evidence-packet template in
`PLANNER_EXECUTOR_PROTOCOL.md`. It must include:

- expected base and actual executed Git commits;
- exact command, environment, package versions, source-integrity result, and
  topology;
- Slurm job/step ID, node, state, exit code, elapsed time, queue/placement
  events, and controller/node/summary return codes;
- factual preflight, attestation, graph-capture, diagnostic, ten-smoke, and
  liveness results;
- every deviation and record-and-proceed condition;
- first failure and last successful stage;
- all small committed artifacts;
- every large durable artifact with absolute path, byte size, and SHA-256;
- final tracked/untracked repository state;
- `None; returned for planner analysis` under limited interpretation unless a
  bounded factual note is necessary.

Change this handoff's state to `RETURNED_FOR_ANALYSIS`, append a direct pointer
to the evidence return, then run:

```bash
git diff --quiet -- . \
  ':(exclude)M3_HUMMING_W4A8_HANDOFF.md' \
  ':(exclude)evidence/m3-humming-w4a8'
git add M3_HUMMING_W4A8_HANDOFF.md "$EVIDENCE"
git diff --cached --check
git commit -m "evidence: return Humming W4A8 qualification"
git push origin duy-branch
```

If a required artifact does not exist because execution stopped before its
stage, do not fabricate or touch an empty substitute. Adjust the explicit `cp`
list to the artifacts that exist, enumerate every missing artifact and reason
in `executor-return.md`, and include the exact blocker evidence.

## Final instruction

Commit and push the complete evidence packet, set the state to
`RETURNED_FOR_ANALYSIS`, and stop. Do not retry, patch, install, launch
performance work, or make a strategic recommendation. The planner decides
whether the returned evidence authorizes a separate paired CUTLASS-versus-
Humming performance packet.
