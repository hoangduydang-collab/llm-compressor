#!/usr/bin/env bash
# GPQA decoding probe — runs ON one allocated 8xH100 node. Settles with data
# whether greedy decoding (the official suite convention) is pathological for
# MiniMax-M3 on gpqa_diamond_cot_zeroshot vs the vendor-recommended sampling
# (temperature=1.0 top_p=0.95, per the M3 model card / generation_config.json):
#   1. serve the in-house GPTQ W4A8 candidate (vLLM 0.24.0 + M3 overlay)
#   2. lm-eval gpqa_diamond_cot_zeroshot --limit 10, PASS A greedy temp=0.0
#   3. same 10 items, PASS B sampled temp=1.0 top_p=0.95
#   4. per-pass sample analysis: empty-content rate (proxy for thinking that ate
#      the 32k budget), extraction hit rate (filtered answer present), accuracy,
#      content token lengths
# NOT score-comparable to anything: 10-item decoding-behavior probe only.
set -uo pipefail

ROOT=${1:?usage: gpqa_decoding_probe_node.sh <result root>}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BVENV=/mnt/nfs/hoangduy/venvs/benchmarks
CKPT=${CKPT:-$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay}
TOKENIZER=${TOKENIZER:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
LIMIT=${LIMIT:-10}

mkdir -p "$ROOT"
note() { echo "[gpqa-probe $(date -u +%H:%M:%S)] $1" | tee -a "$ROOT/probe.log"; }
gate() { echo "$1=$2" >> "$ROOT/gates.txt"; note "gate $1=$2"; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

note "host=$(hostname) ckpt=$CKPT limit=$LIMIT"

note "step 1: start candidate serve"
CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" LOG="$ROOT/serve.log" PID_FILE="$ROOT/serve.pid" \
  bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$ROOT/probe.log" 2>&1
rc=$?; gate serve_start "$rc"; [ "$rc" = 0 ] || exit 1

note "step 2: readiness poll (max 45 min)"
ready=1
for _ in $(seq 1 270); do
  if curl -sf "http://localhost:$PORT/v1/models" -o "$ROOT/models.json" 2>/dev/null; then ready=0; break; fi
  if ! kill -0 "$(cat "$ROOT/serve.pid" 2>/dev/null)" 2>/dev/null; then note "server process died"; break; fi
  sleep 10
done
gate ready "$ready"
[ "$ready" = 0 ] || { tail -60 "$ROOT/serve.log" | tee -a "$ROOT/probe.log"; exit 1; }

run_pass() {  # $1 pass name, $2 gen_kwargs
  note "lm-eval pass $1 (gen_kwargs: $2)"
  "$BVENV/bin/lm_eval" --model local-chat-completions \
    --model_args "base_url=http://localhost:$PORT/v1/chat/completions,model=$SERVED_NAME,num_concurrent=4,tokenizer=$TOKENIZER,max_length=$MAX_MODEL_LEN" \
    --tasks gpqa_diamond_cot_zeroshot --limit "$LIMIT" --seed 0 --apply_chat_template \
    --log_samples --output_path "$ROOT/$1" \
    --gen_kwargs "$2" >"$ROOT/lm_eval_$1.log" 2>&1
  rc=$?; gate "lm_eval_$1" "$rc"
  [ "$rc" = 0 ] || tail -25 "$ROOT/lm_eval_$1.log" | tee -a "$ROOT/probe.log"
}

run_pass greedy  "temperature=0.0,max_gen_toks=32768"
run_pass sampled "temperature=1.0,top_p=0.95,max_gen_toks=32768"

note "step 4: sample analysis"
"$BVENV/bin/python" - "$ROOT" "$TOKENIZER" <<'PY' >"$ROOT/analysis.txt" 2>&1
import glob, json, sys
from pathlib import Path
root, tok_dir = Path(sys.argv[1]), sys.argv[2]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(tok_dir)
for pass_name in ("greedy", "sampled"):
    hits = sorted(glob.glob(str(root / pass_name / "**" / "samples_*.jsonl"), recursive=True))
    if not hits:
        print(f"{pass_name}: NO SAMPLES FILE"); continue
    all_rows = [json.loads(l) for l in open(hits[-1], encoding="utf-8") if l.strip()]
    # one row per (item, filter); analyze the flexible-extract view, fall back to all
    rows = [r for r in all_rows if r.get("filter") == "flexible-extract"] or all_rows
    print(f"== {pass_name} ({len(rows)} items [filter=flexible-extract], {hits[-1]})")
    empty = extract_ok = correct = cap = 0
    lens = []
    for r in rows:
        resp = (r.get("resps") or [[""]])[0][0] or ""
        n_tok = len(tok(resp, add_special_tokens=False).input_ids)
        lens.append(n_tok)
        if not resp.strip(): empty += 1
        if n_tok >= 32000: cap += 1
        filt = r.get("filtered_resps") or []
        filt0 = filt[0] if filt else ""
        if isinstance(filt0, list): filt0 = filt0[0] if filt0 else ""
        if str(filt0).strip() and str(filt0).strip() not in ("[invalid]",): extract_ok += 1
        em = r.get("exact_match") or r.get("exact_match,flexible-extract") or 0
        correct += int(float(em) > 0)
    lens_s = sorted(lens)
    print(f"  empty content        : {empty}/{len(rows)}")
    print(f"  content>=32k tokens  : {cap}/{len(rows)}  (visible content only; thinking is stripped server-side)")
    print(f"  extraction hit       : {extract_ok}/{len(rows)}")
    print(f"  correct (this filter): {correct}/{len(rows)}")
    print(f"  content tokens min/med/max: {lens_s[0]}/{lens_s[len(lens_s)//2]}/{lens_s[-1]}")
    # repetition heuristic: any 40-char chunk repeated >=8 times in a response
    for i, r in enumerate(rows):
        resp = (r.get("resps") or [[""]])[0][0] or ""
        for j in range(0, max(0, len(resp) - 40), 200):
            if resp.count(resp[j:j+40]) >= 8 and len(resp[j:j+40].strip()) > 10:
                print(f"  REPETITION suspect item {i}: chunk {resp[j:j+40]!r} x{resp.count(resp[j:j+40])}")
                break
PY
gate analysis $?
cat "$ROOT/analysis.txt" | tee -a "$ROOT/probe.log"

note "step 5: stop serve"
kill "$(cat "$ROOT/serve.pid" 2>/dev/null)" 2>/dev/null || true
sleep 15
kill -9 -"$(cat "$ROOT/serve.pid" 2>/dev/null)" 2>/dev/null || true

note "gates summary:"; tee -a "$ROOT/probe.log" <"$ROOT/gates.txt"
if grep -qv '=0$' "$ROOT/gates.txt"; then note "PROBE FAIL"; exit 1; fi
note "PROBE PASS"
