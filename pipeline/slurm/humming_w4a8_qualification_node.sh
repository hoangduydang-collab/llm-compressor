#!/usr/bin/env bash
# Node side of the Humming W4A8 correctness qualification for MiniMax-M3.
#
# Promoted verbatim-in-behaviour from the ad hoc script that produced the r3
# pass (evidence/m3-humming-w4a8/20260725T072703Z-*-r3/node-qualification.sh),
# with one thing parameterised: GEMM_TYPE. r3 hardcoded `indexed` because arm 2
# of M3_HOPPER_W4A8_KERNEL_INVESTIGATION.md was still unproven; arm 3
# (grouped_contiguous) is now unblocked and needs the same correctness ladder
# before it is allowed near a stopwatch.
#
# Gates, all fail-closed, in order: declared pack-quantized patch -> runtime
# version/capability -> vLLM M3 serve patches -> serve readiness -> positive
# backend attestation (the requested GEMM strategy specifically) -> 10 repeated
# HTTP chat smokes with content/degeneracy gates -> server-log marker sweep ->
# server still alive.
#
# Env: ROOT (required)  GEMM_TYPE (indexed|grouped|grouped_contiguous)  PORT
set -uo pipefail

REPO=/mnt/nfs/hoangduy/projects/llm-compressor
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
MODEL_ID=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
HUMMING_SITE=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
ROOT="${ROOT:?}"
GEMM_TYPE="${GEMM_TYPE:-indexed}"
PORT="${PORT:-8000}"
SERVED_NAME=MiniMaxAI/MiniMax-M3
LOG="$ROOT/serve.log"
PID_FILE="$ROOT/serve.pid"
NODE_RC=1

mkdir -p "$ROOT/responses"

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

fail() {
  echo "$1" >"$ROOT/first-failure.txt"
  NODE_RC="${2:-1}"
  exit "$NODE_RC"
}

cd "$REPO"
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
# humming-kernels 0.1.10 is side-installed rather than upgraded in the shared
# quant venv; it must win over the venv's pristine 0.1.6.
export PYTHONPATH="$HUMMING_SITE:$REPO"
export M3_W4A8_BACKEND=humming
export VLLM_HUMMING_USE_F16_ACCUM=0
export VLLM_HUMMING_MOE_GEMM_TYPE="$GEMM_TYPE"
export HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming
export M3_LOAD_AUDIT=1
export M3_MOE_PROBE=1
export M3_MOE_PROBE_RECOMPUTE=1
export M3_MOE_PROBE_MAX_TOKENS=256

hostname >"$ROOT/hostname.txt"
printf '%s\n' "$GEMM_TYPE" >"$ROOT/gemm-type.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$ROOT/node-start-utc.txt"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free \
  --format=csv >"$ROOT/nvidia-smi-before.csv"

python pipeline/slurm/patch_humming_ct_input_format.py --site "$HUMMING_SITE" \
  >"$ROOT/humming-patch.log" 2>&1 \
  || fail "humming ct-input patch rc=$?" $?
python pipeline/slurm/patch_humming_ct_input_format.py --site "$HUMMING_SITE" --check \
  >>"$ROOT/humming-patch.log" 2>&1 \
  || fail "humming ct-input patch check rc=$?" $?

python - <<'PY' >"$ROOT/versions.txt" 2>&1
import importlib.metadata
import humming
import torch
import vllm

print("vllm=" + vllm.__version__)
print("humming-kernels=" + importlib.metadata.version("humming-kernels"))
print("humming_path=" + humming.__file__)
print("torch=" + torch.__version__)
print("cuda=" + str(torch.version.cuda))
print("device_count=" + str(torch.cuda.device_count()))
print("device_capability=" + repr(tuple(torch.cuda.get_device_capability())))
assert vllm.__version__ == "0.24.0"
assert importlib.metadata.version("humming-kernels") == "0.1.10"
assert humming.__file__.startswith("/mnt/nfs/hoangduy/venvs/humming-0.1.10-site/")
assert tuple(torch.cuda.get_device_capability()) == (9, 0)
assert torch.cuda.device_count() == 8
PY
[ $? = 0 ] || fail "runtime version/capability gate failed"

python pipeline/slurm/patch_vllm_m3_serve.py --humming --probe \
  >"$ROOT/patch-apply.log" 2>&1 || fail "patch apply rc=$?" $?
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
[ $? = 0 ] || fail "diagnostic patch apply/check failed"
python pipeline/slurm/patch_vllm_m3_serve.py --check --humming \
  >"$ROOT/patch-check.log" 2>&1 || fail "patch check rc=$?" $?

CKPT="$CKPT" \
MODEL_ID="$MODEL_ID" \
SERVED_NAME="$SERVED_NAME" \
PORT="$PORT" \
LOG="$LOG" \
PID_FILE="$PID_FILE" \
bash pipeline/slurm/run_vllm_http_serve_smoke.sh \
  >"$ROOT/launcher.log" 2>&1 || fail "serve launcher rc=$?" $?

ready=1
for _ in $(seq 1 540); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" \
      -o "$ROOT/models.json" 2>/dev/null; then
    ready=0
    break
  fi
  if ! kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    fail "server died before readiness"
  fi
  sleep 10
done
[ "$ready" = 0 ] || fail "server never became ready"
date -u +%Y-%m-%dT%H:%M:%SZ >"$ROOT/ready-utc.txt"

# Attestation is specific to the requested strategy: a grouped run that fell
# back to indexed (vLLM's behaviour for unrecognised values) fails here rather
# than producing numbers attributed to the wrong kernel.
python -m pipeline.m3_humming_w4a8 attest \
  --preflight "$ROOT/serve.log.humming-preflight.json" \
  --log "$LOG" \
  --out "$ROOT/backend-attestation.json" \
  >"$ROOT/attest.log" 2>&1 || fail "backend attestation rc=$?" $?

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
  [ "$rc" = 0 ] || fail "smoke $index HTTP rc=$rc" "$rc"
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
  [ $? = 0 ] || fail "smoke $index content gate failed"
done

grep -q "Capturing CUDA graphs" "$LOG" || fail "CUDA graph capture marker missing"
grep -q "M3_LOAD_AUDIT#" "$LOG" || fail "load-audit marker missing"
grep -q "M3_MOE_PROBE#" "$LOG" || fail "MoE-probe marker missing"
# Runs BEFORE cleanup deliberately: vLLM emits a Traceback/EngineDeadError during
# normal teardown, so sweeping after the kill would flag every healthy run.
if grep -Eiq \
    "M3_MOE_PROBE_NONFINITE|illegal memory access|CUDA error|Traceback" \
    "$LOG"; then
  fail "fatal/non-finite server-log marker present"
fi
kill -0 "$(cat "$PID_FILE")" || fail "server died after smokes"

nvidia-smi --query-gpu=index,name,memory.total,memory.free \
  --format=csv >"$ROOT/nvidia-smi-after.csv"
date -u +%Y-%m-%dT%H:%M:%SZ >"$ROOT/node-end-utc.txt"
NODE_RC=0
exit 0
