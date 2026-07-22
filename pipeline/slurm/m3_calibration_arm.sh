#!/usr/bin/env bash
# M3 tau2->aiperf agentic-shape calibration (single node, self-contained).
# Serves the in-house GPTQ M3 checkpoint locally (the ship-grade 1-node arm =
# what production would serve), then drives the benchmarks-repo calibration
# launcher: tau2-bench telecom tasks with M3 as the agent (thinking ON) and a
# gateway user-sim (DeepSeek-V4-Pro; the M2.5 recipe used gateway DeepSeek).
# Compute nodes reach the gateway directly (verified HTTP 200 from gpu-h123).
#
# Sequence: serve -> ready -> SMOKE (2 tasks, conc 1; fail-closed wiring gate)
#           -> BASE split (114 tasks, conc 5 = gateway rate cap) -> extract
#           AG_* block -> teardown.
# Outputs:
#   $ROOT/calib/driver.log, ag_block.txt (paste-ready AG_* for the profiles)
#   $TAU2_DIR/data/simulations/m3_calib_{smoke,base}_$CALIB_TS/
# Env: ROOT CALIB_TS  [CKPT PORT USER_MODEL overridable]
set -uo pipefail

ROOT=${ROOT:?}; CALIB_TS=${CALIB_TS:?}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
export TAU2_DIR=${TAU2_DIR:-/mnt/nfs/hoangduy/tau2-bench}
CKPT=${CKPT:-$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay}
PORT=${PORT:-8000}
SERVED_NAME=MiniMaxAI/MiniMax-M3
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
# Telecom turn-0 prompt alone is ~8.2k tokens and vLLM enforces
# prompt + max_tokens <= max_model_len, so 40960 rejected every request
# (8193 + 32768 > 40960 -> all 114 sims infrastructure_error, observed
# 20260722T083736Z). 65536 fits peak ~11k prompt + 32k agent budget and is the
# same ctx the 1-node quality serves used.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}

C=$ROOT/calib
mkdir -p "$C"
note() { echo "[calib $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
note "host=$(hostname) ckpt=$CKPT"

note "serve $CKPT on $PORT"
CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" LOG="$C/serve.log" PID_FILE="$C/serve.pid" \
  bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$C/client.log" 2>&1
rc=$?; [ "$rc" = 0 ] || { note "serve start rc=$rc"; exit 1; }
ready=1
for _ in $(seq 1 540); do
  curl -sf "http://localhost:$PORT/v1/models" -o "$C/models.json" 2>/dev/null && { ready=0; break; }
  kill -0 "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || { note "serve died"; break; }
  sleep 10
done
[ "$ready" = 0 ] || { tail -60 "$C/serve.log" | tee -a "$C/client.log"; exit 1; }

teardown() {
  note "stop serve"
  kill "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 15
  kill -9 -"$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
}

# Shared calibration wiring (run_calibration.sh reads these).
export AGENT_BASE="http://localhost:$PORT/v1" AGENT_MODEL="$SERVED_NAME"
export USER_BASE="https://api-inference.bitdeer.ai/v1"
export USER_MODEL="${USER_MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
export USER_KEY_FILE=/mnt/nfs/hoangduy/secrets/bitdeer_api_key
export THINKING=on DOMAIN=telecom

note "SMOKE: 2 tasks, conc 1 (wiring gate)"
SPLIT=small NUM_TASKS=2 MAX_CONC=1 SAVE_TO="m3_calib_smoke_$CALIB_TS" \
  bash "$BENCH/performance/workloads/calibration/run_calibration.sh" \
  >"$C/smoke.log" 2>&1
rc=$?
# Gate on simulation SUCCESS, not file existence: a results.json full of
# infrastructure_error sims (0 messages) is a failure that must not spend the
# base split (observed 20260722T083736Z: context-window rejects, file non-empty).
smoke_ok=$(python3 - "$TAU2_DIR/data/simulations/m3_calib_smoke_$CALIB_TS/results.json" <<'PY'
import json, sys
try:
    sims = json.load(open(sys.argv[1])).get("simulations", [])
except Exception:
    print(0); raise SystemExit
good = [s for s in sims
        if s.get("termination_reason") != "infrastructure_error"
        and s.get("messages")]
print(1 if sims and len(good) == len(sims) else 0)
PY
)
if [ "$rc" != 0 ] || [ "$smoke_ok" != 1 ]; then
  note "SMOKE FAILED rc=$rc smoke_ok=$smoke_ok (fail-closed: not spending the base split)"
  { grep -m3 -iE "error|exception" "$C/smoke.log"; tail -15 "$C/smoke.log"; } | tee -a "$C/client.log"
  teardown; exit 1
fi
note "smoke OK (all sims terminated normally with messages)"

note "BASE: 114 telecom tasks, conc 5"
SPLIT=base MAX_CONC=5 SAVE_TO="m3_calib_base_$CALIB_TS" \
  bash "$BENCH/performance/workloads/calibration/run_calibration.sh" \
  >"$C/driver.log" 2>&1
rc=$?
note "base run rc=$rc"
tail -5 "$C/driver.log" | tee -a "$C/client.log"

if [ "$rc" = 0 ]; then
  note "extract AG_* shape"
  (cd "$BENCH" && "$TAU2_DIR/.venv/bin/python" \
     performance/workloads/calibration/extract_latency.py \
     --runs-dir "$TAU2_DIR/data/simulations/m3_calib_base_$CALIB_TS" \
     --dump-samples 3) >"$C/ag_block.txt" 2>&1 || rc=$?
  note "AG_* block -> $C/ag_block.txt"
fi

teardown
exit "$rc"
