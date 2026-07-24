#!/usr/bin/env bash
# Conc-1 decode TPOT A/B on ONE node: shared-experts stream OFF (production)
# vs stream ON + LLMC_M3_CAPTURE_SYNC=sync (arm H fix, 20260724 RCA).
#
# Interleaved phases A1 B1 A2 B2 (fresh serve per phase) so node/thermal drift
# hits both arms equally. Per phase: aiperf conc-1 (128 in / 1024 out, greedy,
# ignore_eos, 10 requests + 1 warmup) -> steady-state TPOT = inter_token_latency;
# then a 4-prompt greedy quality smoke (content gates + cross-arm comparison).
# A phase whose serve IMAs during capture is retried (<=2), and every attempt
# is recorded — retries on arm B are direct stability evidence against the fix.
#
# Usage: stream_onoff_tpot_ab_node.sh <root> <session>
set -uo pipefail

ROOT=${1:?root}; SESSION=${2:?session}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
PERF_VENV=/mnt/nfs/hoangduy/venvs/perf
QUANT_VENV=/mnt/nfs/hoangduy/venvs/quant
CKPT=/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
SERVED_NAME=MiniMaxAI/MiniMax-M3
BCG_FILE=$QUANT_VENV/lib/python3.12/site-packages/vllm/compilation/breakable_cudagraph.py
PORT=8123
OUT=$ROOT/$SESSION-tpot-ab
mkdir -p "$OUT"
export PATH="$QUANT_VENV/bin:$PATH"
export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

note() { echo "[tpot-ab $(date -u +%H:%M:%S)] $1" | tee -a "$OUT/ab.log"; }
note "host=$(hostname) session=$SESSION"

grep -q "llmc M3 breakable-capture pre-cleanup device sync v1" "$BCG_FILE" || {
  note "FATAL: capture-sync patch missing"; echo "PREFLIGHT_RC=1"; exit 1; }
test -f "$CKPT/config.json" || { note "FATAL: ckpt missing"; echo "PREFLIGHT_RC=1"; exit 1; }
"$PERF_VENV/bin/aiperf" --help >/dev/null 2>&1 || {
  note "FATAL: aiperf missing in perf venv"; echo "PREFLIGHT_RC=1"; exit 1; }
note "preflight ok"; echo "PREFLIGHT_RC=0"

stop_serve() {  # $1 pid-file
  local pid; pid=$(cat "$1" 2>/dev/null) || return 0
  kill "$pid" 2>/dev/null || true; sleep 15
  kill -9 -- -"$pid" 2>/dev/null || true; sleep 5
}

serve_phase() {  # $1 phase-dir  $2 stream_disable  $3 capture_sync -> rc
  local dir=$1 sd=$2 cs=$3
  env -u CUDA_LAUNCH_BLOCKING -u TORCH_USE_CUDA_DSA \
      VLLM_DISABLE_SHARED_EXPERTS_STREAM="$sd" LLMC_M3_CAPTURE_SYNC="$cs" \
      VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256 \
      CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$PORT" \
      MAX_MODEL_LEN=8192 LOG="$dir/serve.log" PID_FILE="$dir/serve.pid" \
      bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$dir/launch.log" 2>&1 || return 1
  for _ in $(seq 1 150); do
    curl -sf "http://localhost:$PORT/v1/models" -o /dev/null 2>/dev/null && return 0
    kill -0 "$(cat "$dir/serve.pid" 2>/dev/null)" 2>/dev/null || return 2
    sleep 5
  done
  return 3
}

run_phase() {  # $1 phase-name  $2 stream_disable  $3 capture_sync
  local phase=$1 sd=$2 cs=$3 attempt rc dir
  for attempt in 1 2 3; do
    dir="$OUT/$phase-a$attempt"; mkdir -p "$dir"
    printf 'phase=%s\nVLLM_DISABLE_SHARED_EXPERTS_STREAM=%s\nLLMC_M3_CAPTURE_SYNC=%s\nattempt=%s\n' \
      "$phase" "$sd" "$cs" "$attempt" >"$dir/phase.env"
    note "phase $phase attempt $attempt (stream_disable=$sd capture_sync=$cs)"
    serve_phase "$dir" "$sd" "$cs"; rc=$?
    if [ "$rc" != 0 ]; then
      local why="not_ready_rc$rc"
      grep -qi "illegal memory access" "$dir/serve.log" 2>/dev/null && why=ima
      note "phase $phase attempt $attempt serve failed ($why)"
      echo "PHASE_ATTEMPT $phase $attempt fail=$why"
      stop_serve "$dir/serve.pid"
      [ "$attempt" = 3 ] && { echo "PHASE_RESULT $phase FAILED"; return 1; }
      continue
    fi
    echo "PHASE_ATTEMPT $phase $attempt ok"

    note "phase $phase aiperf conc-1"
    "$PERF_VENV/bin/aiperf" profile \
      --model "$SERVED_NAME" --url "http://localhost:$PORT" \
      --endpoint-type chat --streaming --tokenizer "$TOKENIZER" \
      --synthetic-input-tokens-mean 128 --synthetic-input-tokens-stddev 0 \
      --prompt-prefix-pool-size 0 --num-dataset-entries 10 \
      --use-legacy-max-tokens \
      --output-tokens-mean 1024 --output-tokens-stddev 0 \
      --extra-inputs '{"temperature":0,"max_tokens":1024,"ignore_eos":true,"min_tokens":1024}' \
      --concurrency 1 --request-count 10 --warmup-request-count 1 \
      --artifact-dir "$dir/aiperf" >"$dir/aiperf.log" 2>&1
    rc=$?
    note "phase $phase aiperf rc=$rc"

    "$QUANT_VENV/bin/python" - "$PORT" "$SERVED_NAME" "$dir/smoke.json" <<'PY'
import json, sys, urllib.request
port, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = [
    "Explain in two sentences why the sky is blue.",
    "List the first 8 prime numbers, comma-separated.",
    "Translate to French: The quick brown fox jumps over the lazy dog.",
    "What is 17 * 23? Answer with the number only.",
]
res = []
for p in prompts:
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 256,
                       "messages": [{"role": "user", "content": p}]}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        txt = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    except Exception as e:
        txt = f"<ERROR {e}>"
    toks = txt.split()
    grams = [" ".join(toks[i:i+8]) for i in range(max(0, len(toks)-7))]
    diversity = len(set(grams)) / len(grams) if grams else 0.0
    res.append({"prompt": p, "text": txt, "ngram8_diversity": round(diversity, 3),
                "degenerate": bool(grams) and diversity < 0.5})
json.dump(res, open(out, "w"), indent=2)
print("SMOKE", "degenerate" if any(r["degenerate"] for r in res) else "ok")
PY
    stop_serve "$dir/serve.pid"
    echo "PHASE_RESULT $phase OK attempt=$attempt aiperf_rc=$rc"
    return 0
  done
}

# Interleaved: A = production (stream OFF, legacy), B = fix (stream ON + sync)
run_phase armA-1 1 legacy
run_phase armB-1 0 sync
run_phase armA-2 1 legacy
run_phase armB-2 0 sync

"$QUANT_VENV/bin/python" - "$OUT" <<'PY' | tee -a "$OUT/ab.log"
import glob, json, os, sys
out = sys.argv[1]

def find_stats(phase):
    for pat in ("**/*aiperf*.json", "**/profile_export*.json"):
        for p in sorted(glob.glob(os.path.join(out, phase + "-a*", "aiperf", pat), recursive=True)):
            try:
                d = json.load(open(p))
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            # Same lookup as benchmarks analyze_perf.py: metric block at top
            # level or under a nested "metrics" dict.
            srcs = [d] + ([d["metrics"]] if isinstance(d.get("metrics"), dict) else [])
            for src in srcs:
                for k in ("inter_token_latency", "itl", "inter token latency",
                          "time_per_output_token", "tpot"):
                    blk = src.get(k)
                    if isinstance(blk, dict):
                        return {"file": p, **{s: blk.get(s) for s in ("avg", "p50", "p90", "p99", "std") if s in blk}}
                    if isinstance(blk, (int, float)):
                        return {"file": p, "avg": blk}
    return None

summary = {"phases": {}}
for phase in ("armA-1", "armB-1", "armA-2", "armB-2"):
    stats = find_stats(phase)
    smokes = sorted(glob.glob(os.path.join(out, phase + "-a*", "smoke.json")))
    smoke = json.load(open(smokes[-1])) if smokes else None
    attempts = len(glob.glob(os.path.join(out, phase + "-a*")))
    summary["phases"][phase] = {
        "itl": stats, "attempts": attempts,
        "smoke_degenerate": any(r["degenerate"] for r in smoke) if smoke else None,
    }

def pool(arm):
    vals = [summary["phases"][p]["itl"]["avg"] for p in summary["phases"]
            if p.startswith(arm) and summary["phases"][p]["itl"] and summary["phases"][p]["itl"].get("avg") is not None]
    return sum(vals) / len(vals) if vals else None

a, b = pool("armA"), pool("armB")
summary["armA_itl_avg"], summary["armB_itl_avg"] = a, b
summary["speedup_pct"] = round((1 - b / a) * 100, 2) if a and b else None
json.dump(summary, open(os.path.join(out, "ab_summary.json"), "w"), indent=2)
print(f"AB_RESULT armA_itl={a} armB_itl={b} speedup_pct={summary['speedup_pct']}")
PY
echo "AB_RC=0"
