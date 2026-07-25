#!/usr/bin/env bash
set -uo pipefail

REPO=/mnt/nfs/hoangduy/projects/llm-compressor
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
MODEL_ID=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
HUMMING_SITE=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
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
# humming-kernels 0.1.10 is side-installed rather than upgraded in the shared
# quant venv; it must win over the venv's pristine 0.1.6.
export PYTHONPATH="$HUMMING_SITE:$REPO"
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
  --out "$ROOT/backend-attestation.json" \
  >"$ROOT/attest.log" 2>&1
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
