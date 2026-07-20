#!/usr/bin/env bash
# Phase 1 of the official-pipeline (AICloud/benchmarks) migration — runs ON one
# allocated 8xH100 node. Validates the full client path against the in-house
# W4A8 endpoint BEFORE any multi-node BF16 spend:
#   0. fail-closed harness preflight (lm-eval pin, tokenizer/chat-template hashes)
#   1. serve the candidate via run_vllm_http_serve_smoke.sh (vLLM 0.24.0 + M3 overlay)
#   2. readiness poll
#   3. quality.probe_endpoint --capabilities (chat top_logprobs + /completions
#      echo text_offset are the launch-blockers for distribution + mmlu)
#   4. lm-eval smoke: gsm8k --limit 4 (chat path) + mmlu --limit 1 (loglikelihood path)
#   5. stop serve; gates.txt records per-step rc (0 = pass)
#
# NOT score-comparable to anything: --limit smoke, setup validation only.
set -uo pipefail

ROOT=${1:?usage: official_quality_phase1_node.sh <result root>}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
BVENV=/mnt/nfs/hoangduy/venvs/benchmarks
CKPT=${CKPT:-$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay}
TOKENIZER=${TOKENIZER:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}

mkdir -p "$ROOT"
note() { echo "[phase1 $(date -u +%H:%M:%S)] $1" | tee -a "$ROOT/phase1.log"; }
gate() { echo "$1=$2" >> "$ROOT/gates.txt"; note "gate $1=$2"; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

note "host=$(hostname) ckpt=$CKPT"

note "step 0: harness preflight (fail-closed)"
"$BVENV/bin/python" - "$ROOT" "$TOKENIZER" "$CKPT" <<'PY' >>"$ROOT/phase1.log" 2>&1
import hashlib, json, sys
from pathlib import Path
root, tok_dir, ckpt = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
import lm_eval
assert lm_eval.__version__ == "0.4.10", f"lm-eval pin violated: {lm_eval.__version__}"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
tok_files = {}
for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    p = tok_dir / name
    if p.exists():
        tok_files[name] = sha(p)
assert "tokenizer.json" in tok_files or "tokenizer_config.json" in tok_files, \
    f"no tokenizer files under {tok_dir}"
manifest = {
    "purpose": "phase1 setup-validation smoke (official AICloud/benchmarks pipeline)",
    "score_comparable_to_public_benchmarks": False,
    "harness": {"name": "lm-eval", "version": lm_eval.__version__, "backend": "api"},
    "tasks": {"gsm8k": {"path": "chat", "limit": 4, "fewshot": 5},
              "mmlu": {"path": "loglikelihood", "limit": 1, "fewshot": 5}},
    "generation": {"temperature": 0.0, "max_gen_toks": 32768, "seed": 0},
    "serving": {"backend": "vllm-0.24.0+m3-overlay", "topology": "1 node TP8/EP8 mp"},
    "checkpoint": ckpt,
    "tokenizer_dir": str(tok_dir),
    "tokenizer_hashes_sha256": tok_files,
}
(root / "harness_preflight.json").write_text(json.dumps(manifest, indent=2))
print("preflight OK:", json.dumps(tok_files))
PY
rc=$?; gate preflight "$rc"; [ "$rc" = 0 ] || exit 1

note "step 1: start candidate W4A8 HTTP serve (max_model_len=$MAX_MODEL_LEN)"
CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" LOG="$ROOT/serve.log" PID_FILE="$ROOT/serve.pid" \
  bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$ROOT/phase1.log" 2>&1
rc=$?; gate serve_start "$rc"; [ "$rc" = 0 ] || exit 1

note "step 2: readiness poll (max 45 min)"
ready=1
for _ in $(seq 1 270); do
  if curl -sf "http://localhost:$PORT/v1/models" -o "$ROOT/models.json" 2>/dev/null; then ready=0; break; fi
  if ! kill -0 "$(cat "$ROOT/serve.pid" 2>/dev/null)" 2>/dev/null; then note "server process died"; break; fi
  sleep 10
done
gate ready "$ready"
[ "$ready" = 0 ] || { tail -60 "$ROOT/serve.log" | tee -a "$ROOT/phase1.log"; exit 1; }
note "server ready: $(tr -d '\n' <"$ROOT/models.json" | head -c 300)"

note "step 3: capability probe"
( cd "$BENCH" && BASE_URL="http://localhost:$PORT" "$BVENV/bin/python" -m quality.probe_endpoint \
    --profile configs/minimax/minimax-m3.sh --capabilities ) >"$ROOT/capabilities.txt" 2>&1
rc=$?; gate probe_ran "$rc"
cat "$ROOT/capabilities.txt" | tee -a "$ROOT/phase1.log"
grep -q '\[ok  \] chat logprobs / top_logprobs' "$ROOT/capabilities.txt"; gate cap_chat_logprobs $?
grep -q '\[ok  \] /completions echo text_offset' "$ROOT/capabilities.txt"; gate cap_text_offset $?

note "step 4a: lm-eval smoke, chat path (gsm8k --limit 4)"
"$BVENV/bin/lm_eval" --model local-chat-completions \
  --model_args "base_url=http://localhost:$PORT/v1/chat/completions,model=$SERVED_NAME,num_concurrent=4,tokenizer=$TOKENIZER" \
  --tasks gsm8k --output_path "$ROOT/lm_eval_smoke" --seed 0 --apply_chat_template \
  --gen_kwargs temperature=0.0,max_gen_toks=32768 --limit 4 >"$ROOT/lm_eval_gen.log" 2>&1
rc=$?; gate lm_eval_chat "$rc"
[ "$rc" = 0 ] || tail -25 "$ROOT/lm_eval_gen.log" | tee -a "$ROOT/phase1.log"

note "step 4b: lm-eval smoke, loglikelihood path (mmlu --limit 1)"
"$BVENV/bin/lm_eval" --model local-completions \
  --model_args "base_url=http://localhost:$PORT/v1/completions,model=$SERVED_NAME,num_concurrent=4,tokenizer=$TOKENIZER" \
  --tasks mmlu --output_path "$ROOT/lm_eval_smoke" --seed 0 --apply_chat_template \
  --limit 1 >"$ROOT/lm_eval_ll.log" 2>&1
rc=$?; gate lm_eval_loglikelihood "$rc"
[ "$rc" = 0 ] || tail -25 "$ROOT/lm_eval_ll.log" | tee -a "$ROOT/phase1.log"

note "step 5: stop serve"
kill "$(cat "$ROOT/serve.pid" 2>/dev/null)" 2>/dev/null || true
sleep 15
kill -9 -"$(cat "$ROOT/serve.pid" 2>/dev/null)" 2>/dev/null || true

note "gates summary:"
tee -a "$ROOT/phase1.log" <"$ROOT/gates.txt"
if grep -qv '=0$' "$ROOT/gates.txt"; then note "PHASE1 FAIL"; exit 1; fi
note "PHASE1 PASS"
