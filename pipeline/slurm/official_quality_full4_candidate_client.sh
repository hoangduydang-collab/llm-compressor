#!/usr/bin/env bash
# Full four-way, candidate leg — runs ON one allocated 8xH100 node. Serves ONE
# quant arm and drives its full A/B against the shared 2-node BF16 endpoint:
#   1. serve $CKPT on $PORT (vLLM 0.24.0 + M3 overlay, TP8/EP8)
#   2. wait for the BF16 arm's ready marker; pin its IP into a baseline wrapper
#   3. capability probes on both endpoints (text_offset = launch blocker)
#   4. [RUN_BASELINE_GENERAL=1 arm only] launch the baseline's general suite
#      ONCE in the background (standalone orchestrator against the BF16 URL)
#   5. quality.run_ab FULL (no --limit) with --reuse-baseline-general-wait-s:
#      candidate suites run now; the baseline stage reuses step 4's on-disk
#      result instead of re-evaluating BF16 per arm
# Env: ROOT ARM CKPT PORT PROFILE (path under $BENCH) RUN_BASELINE_GENERAL(0/1)
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; CKPT=${CKPT:?}; PORT=${PORT:?}; PROFILE=${PROFILE:?}
RUN_BASELINE_GENERAL=${RUN_BASELINE_GENERAL:-0}
RUN_ID=${RUN_ID:-full4}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
BVENV=/mnt/nfs/hoangduy/venvs/benchmarks
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
SERVED_NAME=MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=65536
BF16_PORT=8001

CLIENT=$ROOT/client-$ARM
mkdir -p "$CLIENT" "$ROOT/profiles/$ARM"
note() { echo "[full4-$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$CLIENT/client.log"; }
gate() { echo "$1=$2" >> "$CLIENT/gates.txt"; note "gate $1=$2"; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

note "host=$(hostname) arm=$ARM ckpt=$CKPT port=$PORT"

note "step 1: serve candidate"
CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" LOG="$CLIENT/serve.log" PID_FILE="$CLIENT/serve.pid" \
  bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$CLIENT/client.log" 2>&1
rc=$?; gate cand_serve_start "$rc"; [ "$rc" = 0 ] || exit 1

note "step 2: candidate readiness poll (max 45 min)"
ready=1
for _ in $(seq 1 270); do
  if curl -sf "http://localhost:$PORT/v1/models" -o "$CLIENT/models.json" 2>/dev/null; then ready=0; break; fi
  if ! kill -0 "$(cat "$CLIENT/serve.pid" 2>/dev/null)" 2>/dev/null; then note "candidate serve died"; break; fi
  sleep 10
done
gate cand_ready "$ready"
[ "$ready" = 0 ] || { tail -60 "$CLIENT/serve.log" | tee -a "$CLIENT/client.log"; exit 1; }

note "step 3: wait for BF16 arm ready marker (max 2h)"
bf16=1
for _ in $(seq 1 720); do
  [ -f "$ROOT/bf16/ready" ] && { bf16=0; break; }
  sleep 10
done
gate bf16_ready "$bf16"; [ "$bf16" = 0 ] || exit 1
BF16_IP=$(cat "$ROOT/bf16/endpoint-ip")
curl -sf "http://$BF16_IP:$BF16_PORT/v1/models" -o "$CLIENT/bf16-models.json"
gate bf16_reachable $?

note "step 4: baseline profile wrapper (id must stay 'minimax-m3-bf16')"
WRAP=$ROOT/profiles/$ARM/minimax-m3-bf16.sh
{ echo "BASE_URL=\"http://$BF16_IP:$BF16_PORT\""
  echo "source \"$BENCH/configs/minimax/minimax-m3-bf16.sh\""; } > "$WRAP"

note "step 5: capability probes"
( cd "$BENCH" && BASE_URL="http://localhost:$PORT" "$BVENV/bin/python" -m quality.probe_endpoint \
    --profile "$PROFILE" --capabilities ) >"$CLIENT/capabilities-candidate.txt" 2>&1
grep -q '\[ok  \] /completions echo text_offset' "$CLIENT/capabilities-candidate.txt"; gate cap_text_offset_candidate $?
( cd "$BENCH" && "$BVENV/bin/python" -m quality.probe_endpoint \
    --profile "$WRAP" --capabilities ) >"$CLIENT/capabilities-bf16.txt" 2>&1
grep -q '\[ok  \] /completions echo text_offset' "$CLIENT/capabilities-bf16.txt"; gate cap_text_offset_bf16 $?

BASE_PID=""
if [ "$RUN_BASELINE_GENERAL" = 1 ]; then
  note "step 6: baseline general suite ONCE (standalone orchestrator, background)"
  ( cd "$BENCH" && PATH="$BVENV/bin:$PATH" "$BVENV/bin/python" -m quality.orchestrator \
      --profile "$WRAP" --out-root "$ROOT/results" --run-id "$RUN_ID" \
      --execute ) >"$CLIENT/baseline_general.log" 2>&1 &
  BASE_PID=$!
fi

note "step 7: quality.run_ab FULL (reuse shared baseline, wait <=12h)"
( cd "$BENCH" && PATH="$BVENV/bin:$PATH" "$BVENV/bin/python" -m quality.run_ab \
    --profile "$PROFILE" \
    --baseline-config "$WRAP" \
    --run-id "$RUN_ID" \
    --out-root "$ROOT/results" \
    --reuse-baseline-general-wait-s 43200 \
    --report --report-out "$ROOT/results/report.$RUN_ID-$ARM.html" ) >"$CLIENT/run_ab.log" 2>&1
rc=$?; gate run_ab "$rc"
tail -20 "$CLIENT/run_ab.log" | tee -a "$CLIENT/client.log"

if [ -n "$BASE_PID" ]; then
  note "step 8: wait for background baseline general"
  wait "$BASE_PID"; gate baseline_general $?
  tail -8 "$CLIENT/baseline_general.log" | tee -a "$CLIENT/client.log"
fi

note "step 9: stop candidate serve"
kill "$(cat "$CLIENT/serve.pid" 2>/dev/null)" 2>/dev/null || true
sleep 15
kill -9 -"$(cat "$CLIENT/serve.pid" 2>/dev/null)" 2>/dev/null || true

note "gates summary:"; tee -a "$CLIENT/client.log" <"$CLIENT/gates.txt"
if grep -qv '=0$' "$CLIENT/gates.txt"; then note "ARM $ARM FAIL"; exit 1; fi
note "ARM $ARM PASS"
