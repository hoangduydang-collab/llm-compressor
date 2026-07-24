#!/usr/bin/env bash
# One condition of the shared-experts aux-stream fix matrix on one node (v2).
# v2 over shared_stream_fix_single_condition_node.sh: records and preflights
# LLMC_M3_CAPTURE_SYNC (arm H: restore upstream torch.cuda.graph's device
# synchronize before the breakable-capture pre-cleanup empty_cache).
# Usage: ... <root> <session> <cond> <stream_disable> <fi_ar_mode>
set -uo pipefail

ROOT=${1:?root}; SESSION=${2:?session}; COND=${3:?cond}; SD=${4:?stream_disable}; MODE=${5:?fi_ar_mode}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
VENV_FILE=/mnt/nfs/hoangduy/venvs/quant/lib/python3.12/site-packages/vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py
BCG_FILE=/mnt/nfs/hoangduy/venvs/quant/lib/python3.12/site-packages/vllm/compilation/breakable_cudagraph.py
CASES=async_baseline_1,async_baseline_2,async_baseline_3
export PATH="/mnt/nfs/hoangduy/venvs/quant/bin:$PATH"

mkdir -p "$ROOT"
note() { echo "[fix-1cond $(date -u +%H:%M:%S)] $1" | tee -a "$ROOT/$SESSION-$COND.log"; }
note "host=$(hostname) cond=$COND stream_disable=$SD mode=$MODE capture_sync=${LLMC_M3_CAPTURE_SYNC:-legacy}"

grep -q "llmc M3 cudagraph fused-AR mode switch v2" "$VENV_FILE" || {
  note "FATAL: v2 fused-AR patch missing"; echo "PREFLIGHT_RC=1"; exit 1; }
if [ "${LLMC_M3_CAPTURE_SYNC:-legacy}" = "sync" ]; then
  grep -q "llmc M3 breakable-capture pre-cleanup device sync v1" "$BCG_FILE" || {
    note "FATAL: capture-sync patch missing in breakable_cudagraph.py"
    echo "PREFLIGHT_RC=1"; exit 1; }
fi
note "preflight ok"; echo "PREFLIGHT_RC=0"

RUN_ID="$SESSION-$COND"
RUN_DIR="$ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
cat >"$RUN_DIR/condition.env" <<EOF
condition=$COND
VLLM_DISABLE_SHARED_EXPERTS_STREAM=$SD
LLMC_M3_FI_AR_MODE=$MODE
LLMC_M3_SHARED_RS_MODE=${LLMC_M3_SHARED_RS_MODE:-legacy}
LLMC_M3_CAPTURE_SYNC=${LLMC_M3_CAPTURE_SYNC:-legacy}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<launcher default>}
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256
CUDA_LAUNCH_BLOCKING=unset
MATRIX_CASES=$CASES
EOF

env -u CUDA_LAUNCH_BLOCKING -u TORCH_USE_CUDA_DSA \
    VLLM_DISABLE_SHARED_EXPERTS_STREAM="$SD" \
    LLMC_M3_FI_AR_MODE="$MODE" \
    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256 \
    RESULTS_ROOT="$ROOT" RUN_ID="$RUN_ID" MATRIX_CASES="$CASES" \
    bash "$REPO/pipeline/slurm/test_m3_http_cudagraph_matrix.sh" \
    >"$RUN_DIR/matrix-driver.log" 2>&1
rc=$?
note "condition $COND rc=$rc"

python - "$RUN_DIR/summary.json" "$COND" <<'PY'
import json, sys
try:
    s = json.loads(open(sys.argv[1], encoding="utf-8").read())
except Exception as e:
    print(f"COND_RESULT {sys.argv[2]} error={e}")
    raise SystemExit(1)
ok = sum(1 for t in s["trials"] if t.get("server_ready") and t.get("chat_ok"))
ima = sum(1 for t in s["trials"] if t.get("ima"))
print(f"COND_RESULT {sys.argv[2]} clean={ok}/{len(s['trials'])} ima={ima}")
PY
echo "CONDITION_RC $COND $rc"
