#!/usr/bin/env bash
# Phase 2 candidate+client arm (1x 8xH100 node): serves the in-house W4A8
# candidate on localhost:8000, waits for the BF16 arm's ready marker, then
# drives the official pipeline's A/B (quality.run_ab: general --limit 20 +
# distribution + delta + report) with the candidate on this node and the BF16
# baseline at http://<bf16 endpoint-ip>:8001.
#
# Always touches $ROOT/client-done at exit so the BF16 arm tears down.
# NOT score-comparable to anything: --limit smoke, setup validation only.
set -uo pipefail

ROOT=${1:?usage: official_quality_phase2_client.sh <phase2 root>}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
BVENV=/mnt/nfs/hoangduy/venvs/benchmarks
CKPT=${CKPT:-$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay}
TOKENIZER=${TOKENIZER:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
PORT=${PORT:-8000}
BF16_PORT=${BF16_PORT:-8001}
GENERAL_LIMIT=${GENERAL_LIMIT:-20}
RUN_ID=${RUN_ID:-phase2-smoke}

CLIENT=$ROOT/client
mkdir -p "$CLIENT" "$ROOT/profiles" "$ROOT/results"
note() { echo "[p2-client $(date -u +%H:%M:%S)] $1" | tee -a "$CLIENT/client.log"; }
gate() { echo "$1=$2" >> "$CLIENT/gates.txt"; note "gate $1=$2"; }
trap 'touch "$ROOT/client-done"' EXIT

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

note "step 1: serve candidate W4A8 on localhost:$PORT"
CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$PORT" \
  MAX_MODEL_LEN=65536 LOG="$CLIENT/serve.log" PID_FILE="$CLIENT/serve.pid" \
  bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$CLIENT/client.log" 2>&1
rc=$?; gate cand_serve_start "$rc"; [ "$rc" = 0 ] || exit 1

ready=1
for _ in $(seq 1 270); do
  if curl -sf "http://localhost:$PORT/v1/models" -o "$CLIENT/models.json" 2>/dev/null; then ready=0; break; fi
  kill -0 "$(cat "$CLIENT/serve.pid" 2>/dev/null)" 2>/dev/null || break
  sleep 10
done
gate cand_ready "$ready"
[ "$ready" = 0 ] || { tail -60 "$CLIENT/serve.log" | tee -a "$CLIENT/client.log"; exit 1; }
note "candidate ready"

note "step 2: wait for BF16 arm ready marker (max 2h)"
bf_ok=1
for _ in $(seq 1 720); do
  [[ -f "$ROOT/bf16/ready" && -f "$ROOT/bf16/endpoint-ip" ]] && { bf_ok=0; break; }
  sleep 10
done
gate bf16_ready "$bf_ok"; [ "$bf_ok" = 0 ] || exit 1
BF16_IP=$(<"$ROOT/bf16/endpoint-ip")
BF16_URL="http://$BF16_IP:$BF16_PORT"
curl -sf "$BF16_URL/v1/models" -o "$CLIENT/bf16-models.json"
gate bf16_reachable $?
note "bf16 baseline at $BF16_URL"

note "step 3: baseline profile wrapper (same id 'minimax-m3-bf16', BASE_URL pinned)"
cat >"$ROOT/profiles/minimax-m3-bf16.sh" <<EOF
BASE_URL="$BF16_URL"
source $BENCH/configs/minimax/minimax-m3-bf16.sh
EOF

note "step 4: capability probes (candidate + baseline)"
( cd "$BENCH" && BASE_URL="http://localhost:$PORT" "$BVENV/bin/python" -m quality.probe_endpoint \
    --profile configs/minimax/minimax-m3.sh --capabilities ) >"$CLIENT/capabilities-candidate.txt" 2>&1
( cd "$BENCH" && "$BVENV/bin/python" -m quality.probe_endpoint \
    --profile "$ROOT/profiles/minimax-m3-bf16.sh" --capabilities ) >"$CLIENT/capabilities-bf16.txt" 2>&1
for f in candidate bf16; do
  grep -q '\[ok  \] /completions echo text_offset' "$CLIENT/capabilities-$f.txt"
  gate "cap_text_offset_$f" $?
done

note "step 5: quality.run_ab (general --limit $GENERAL_LIMIT + distribution + delta + report)"
( cd "$BENCH" && "$BVENV/bin/python" -m quality.run_ab \
    --profile configs/minimax/minimax-m3.sh \
    --baseline-config "$ROOT/profiles/minimax-m3-bf16.sh" \
    --run-id "$RUN_ID" \
    --out-root "$ROOT/results" \
    --general-limit "$GENERAL_LIMIT" \
    --report ) >"$CLIENT/run_ab.log" 2>&1
rc=$?; gate run_ab "$rc"
tail -30 "$CLIENT/run_ab.log" | tee -a "$CLIENT/client.log"

note "step 6: stop candidate serve"
kill "$(cat "$CLIENT/serve.pid" 2>/dev/null)" 2>/dev/null || true
sleep 15
kill -9 -"$(cat "$CLIENT/serve.pid" 2>/dev/null)" 2>/dev/null || true

note "gates summary:"
tee -a "$CLIENT/client.log" <"$CLIENT/gates.txt"
if grep -qv '=0$' "$CLIENT/gates.txt"; then note "PHASE2 CLIENT FAIL"; exit 1; fi
note "PHASE2 CLIENT PASS"
