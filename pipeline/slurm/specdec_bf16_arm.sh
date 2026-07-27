#!/usr/bin/env bash
# One arm of the EAGLE3 BF16 absolute-reference test -- M3_SPECDEC_EAGLE3_PLAN.md
# ("Phase F: drafter against the unquantized target").
#
# Question this answers: is retraining / finetuning the drafter worth it?
#
# CORRECTED FRAMING (this header originally called BF16 the CEILING -- it is not).
# The Inferact EAGLE3 drafter was measured and trained against **MXFP8**, not BF16:
# its README names `MiniMaxAI/MiniMax-M3-MXFP8` as the measurement target, and
# training is pinned by arithmetic (`inference.vllm.tp_size=4` on GB300 leaves
# ~744 GiB per engine -- BF16 M3's 796 GiB does not fit, MXFP8's 414 GiB does).
# Since EAGLE3 consumes the target's hidden states, MXFP8 is the on-distribution
# reference and phase E was the decisive arm. BF16 is a DIFFERENT off-distribution
# target and may legitimately score below MXFP8.
#
# What this arm is for: phase E compared two quantized targets and found no gap, but
# could not exclude both being equally degraded against an unquantized target.
# Spanning the full 4 -> 8 -> 16 bit range excludes it. Phase E's early "~1.35%
# penalty" reading came from its first two cells and did not survive its conc-10
# cells -- do not carry that number forward.
#
# Runs as 2 ranks on 2x 8xH100 (srun --nodes=2 --ntasks=2). BF16 M3 is 796 GiB of
# safetensors and does not fit one node (8x80 GiB), so this needs TP16 over ray --
# the same proven path as official_quality_bf16_http_arm.sh. Rank 0 boots the ray
# head, serves, and runs the client; rank 1 joins ray and idles until rank 0 is done.
#
# DELIBERATE DIVERGENCES from the phase D/E windows, and why each is safe:
#
#   TP16 + ray (vs TP8, single node)  -- forced by the 796 GiB weight footprint.
#       Consequence: BF16 ABSOLUTE speed is NOT comparable to W4AFP8/MXFP8 absolute
#       speed (different topology, cross-node NCCL in the critical path). This arm
#       therefore carries its OWN k=0 control and only within-format ratios are
#       quoted. Accepted length is a model-intrinsic quantity and is unaffected by
#       parallel topology -- that is the metric this phase exists to measure.
#
#   MAX_MODEL_LEN 65536 (vs 131072)  -- the proven BF16 value; weights leave
#       ~22 GiB/GPU for KV at gpu_util 0.9. Safe because phase D measured accepted
#       length to be flat over a 32x prompt-length range (1k/8k/32k), our cells are
#       8k in + 2048 out, and max_model_len changes only KV allocation, not
#       per-step compute at fixed sequence length.
#
#   PATCH_CKPT_CONFIG=0  -- every other arm patches its checkpoint's config from
#       this BF16 directory (it is the MODEL_ID source). Patching it would mutate
#       the pristine reference. The proven BF16 arm never patched it either.
#
# Held constant with phase D/E: the same hash-gated SPEED-Bench prompt bytes, the
# same --random-seed 42, temp 0.6, max_tokens 2048, kv_cache_dtype fp8,
# block_size 128, gpu_util 0.9, expert parallel, disable_custom_all_reduce,
# language_model_only, enable_thinking, and the same 8k-low / 8k-high cells at
# conc 1 and 10.
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; RUN_TS=${RUN_TS:?}
CKPT=${CKPT:?}; PORT=${PORT:?}; SPEC_K=${SPEC_K:?}
DRAFTER=${DRAFTER:-/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
PERF_VENV=/mnt/nfs/hoangduy/venvs/perf
SB_DIR=${SB_DIR:-$REPO/artifacts/aiperf-datasets/speedbench}
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MAX_TOKENS=${MAX_TOKENS:-2048}
TEMP=${TEMP:-0.6}
PRECISION=BF16
READY_MAX=${READY_MAX:-720}          # 10s ticks -> 2h load budget (796 GiB over NFS)

C=$ROOT/arm-$ARM
mkdir -p "$C" "$C/metrics"
rank=${SLURM_PROCID:-0}
note() { echo "[arm-$ARM r$rank $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }

# Same node env as the proven BF16 TP16 arm.
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd "$REPO"
export HOME=/mnt/nfs/hoangduy WORK_ROOT=/mnt/nfs/hoangduy
export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# Cross-node TP16 NCCL/gloo must bind the routable fabric; auto-detect picks an
# unroutable iface on some node pairs. Pin it and keep WARN evidence.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-intranet}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-intranet}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Job-scoped marker: a stale driver-done from an earlier run on the same ROOT makes
# the worker leave ray immediately, and vLLM's 16-GPU placement group then times out.
DRIVER_DONE="$C/ray_runtime/driver-done-${SLURM_JOB_ID:-nojob}"
finish() {
  touch "$DRIVER_DONE" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
}
trap finish EXIT

note "host=$(hostname) rank=$rank spec_k=$SPEC_K mml=$MAX_MODEL_LEN temp=$TEMP"
note "booting ray cluster (TP16 over 2 nodes)"
source pipeline/slurm/test_m3_ray_topology.sh --out "$C/ray_runtime" --keep-alive
set +e   # the topology script enables -e; everything below handles rc explicitly

if ((rank != 0)); then
  note "worker joined ray; idling until driver-done"
  for _ in $(seq 1 86400); do
    [[ -f "$DRIVER_DONE" ]] && break
    sleep 1
  done
  note "worker exiting"
  exit 0
fi

# --- rank 0: ray topology gate (fail closed) ----------------------------------
python -c 'import json,sys; g=json.load(open(sys.argv[1])); sys.exit(0 if g["ready"] else 1)' \
  "$C/ray_runtime/gate.json" \
  || { note "ABORT: ray topology gate not ready"; cat "$C/ray_runtime/gate.json" | tee -a "$C/client.log"; exit 1; }
note "ray gate ready: $(python -c 'import json;g=json.load(open("'"$C"'/ray_runtime/gate.json"));print(g["alive_nodes"],"nodes",g["visible_gpus"],"gpus")')"

if [ "$SPEC_K" -gt 0 ]; then
  test -f "$DRAFTER/config.json" || { note "ABORT drafter missing: $DRAFTER"; exit 1; }
  SPEC_ARG="--speculative-config {\"method\":\"eagle3\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$SPEC_K,\"attention_backend\":\"FLASH_ATTN\"}"
  SPEC_LABEL="eagle3-k$SPEC_K"
else
  SPEC_ARG=""
  SPEC_LABEL="none"
fi
# TP16 requires the ray executor; the smoke helper takes TP via env and the rest here.
export EXTRA_VLLM_ARGS="--distributed-executor-backend ray ${SPEC_ARG}"
printf '%s\n' "$EXTRA_VLLM_ARGS" > "$C/extra-vllm-args.txt"

note "serve BF16 $CKPT on $PORT (TP16/ray)"
CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$PORT" \
  TP=16 MAX_MODEL_LEN="$MAX_MODEL_LEN" PATCH_CKPT_CONFIG=0 \
  LOG="$C/serve.log" PID_FILE="$C/serve.pid" \
  bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$C/client.log" 2>&1 \
  || { note "serve start rc=$?"; tail -60 "$C/serve.log" 2>/dev/null | tee -a "$C/client.log"; exit 1; }

stop_local_serve() {
  note "stop serve"
  kill "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 20
  kill -9 -"$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
}
snap() { curl -sf "http://localhost:$PORT/metrics" -o "$C/metrics/$1.txt" 2>/dev/null || true; }

ready=1
for _ in $(seq 1 "$READY_MAX"); do
  curl -sf "http://localhost:$PORT/v1/models" -o "$C/models.json" 2>/dev/null && { ready=0; break; }
  kill -0 "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || { note "serve died during load"; break; }
  sleep 10
done
[ "$ready" = 0 ] || { tail -80 "$C/serve.log" | tee -a "$C/client.log"; exit 1; }
export BASE_URL="http://localhost:$PORT"
note "endpoint ready"

# --- gates (fail closed) ------------------------------------------------------
# Assert BF16 really served unquantized: no quant method may appear in the banner.
if grep -qiE "quantization[\"' ]*[:=][\"' ]*(humming|mxfp8|compressed-tensors|awq|gptq|fp8)" "$C/serve.log"; then
  note "ABORT: serve.log shows a quantization method on the BF16 arm"
  grep -inE "quantization" "$C/serve.log" | head -20 | tee -a "$C/client.log"
  stop_local_serve; exit 1
fi
grep -iE "quantization|dtype|torch.bfloat16" "$C/serve.log" | head -40 > "$C/quant-boot.log"
# Assert the 16-way topology actually engaged rather than silently falling back.
grep -qiE "tensor_parallel_size['\"]?[:=] ?16|world_size=16" "$C/serve.log" \
  || { note "ABORT: serve.log does not show tensor_parallel_size=16"; stop_local_serve; exit 1; }
grep -iE "tensor_parallel_size|distributed_executor_backend|world_size" "$C/serve.log" | head -20 > "$C/topology-boot.log"
if [ "$SPEC_K" -gt 0 ]; then
  grep -qi "num_speculative_tokens" "$C/serve.log" \
    || { note "ABORT: serve.log shows no speculative config"; stop_local_serve; exit 1; }
  grep -i "speculative\|eagle" "$C/serve.log" | head -40 > "$C/spec-boot.log"
fi

rc_all=0
OUT=$C/speedbench

# $1 cell  $2 concurrency  $3 request count
run_cell() {
  local cell=$1 conc=$2 n=$3
  local file="$SB_DIR/$cell.jsonl"
  test -s "$file" || { note "ABORT: missing staged prompts $file"; rc_all=1; return; }
  note "cell=$cell conc=$conc requests=$n"
  snap "sb-$cell-c$conc-pre"
  # Identical seed to phase D/E so every arm draws the SAME prompts in the same order.
  "$PERF_VENV/bin/aiperf" profile \
      --model "$SERVED_NAME" --url "$BASE_URL" --endpoint-type chat --streaming \
      --tokenizer "$TOKENIZER" \
      --custom-dataset-type single_turn --input-file "$file" \
      --extra-inputs "{\"temperature\":$TEMP,\"max_tokens\":$MAX_TOKENS,\"chat_template_kwargs\":{\"enable_thinking\":true}}" \
      --random-seed 42 \
      --concurrency "$conc" --request-count "$n" --warmup-request-count "$conc" \
      --artifact-dir "$OUT/$cell/conc_$conc" >>"$C/sb-$cell-c$conc.log" 2>&1
  local rc=$?
  snap "sb-$cell-c$conc-post"
  note "cell=$cell conc=$conc rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
}

# Same four cells as phase E, same order.
for cell in 8k-low 8k-high; do run_cell "$cell" 1 40; done
for cell in 8k-low 8k-high; do run_cell "$cell" 10 100; done

for cell in 8k-low 8k-high; do
  [ -d "$OUT/$cell" ] || continue
  "$PERF_VENV/bin/python" "$BENCH/performance/workloads/analyze_perf.py" --run-dir "$OUT/$cell" \
    --mode "sbbf16_$cell" --label "$ARM" --precision "$PRECISION" --gpu 16xH100 \
    --num-gpus 16 --spec-decode "$SPEC_LABEL" >>"$C/analyze-$cell.log" 2>&1 \
    || note "WARN analyze_perf failed for $cell"
done

grep -i "acceptance\|SpecDecoding" "$C/serve.log" > "$C/spec-metrics.log" 2>/dev/null || true
stop_local_serve
note "arm done rc=$rc_all"
exit "$rc_all"
