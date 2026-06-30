#!/bin/bash
# Shared helper for Stage-1 Qwen3-30B-A3B runs.
# Source this AFTER activating the env and cd-ing to the repo root.
# Provides: run_method <method> [scheme] [serve_tp]
#
#   quantize (own process) -> serve-verify (fresh process) -> eval gate
#
# Returns non-zero if any sub-stage failed (but always runs all sub-stages it can).

CONFIG=${CONFIG:-pipeline/configs/qwen3_30b_a3b.yaml}

run_method() {
  local METHOD=$1
  local SCHEME=${2:-${SCHEME:-W4AFP8}}
  # Single-GPU serve by default: TP=1 keeps the expert width (768) un-sharded,
  # so it stays a multiple of 256 for the CUTLASS W4A8 kernel.
  local SERVE_TP=${3:-${SERVE_TP:-1}}
  local rc=0

  echo "================================================================"
  echo "  RUN: method=$METHOD scheme=$SCHEME serve_tp=$SERVE_TP"
  echo "================================================================"

  # 1) Quantize (own process so GPU memory is released before vLLM loads).
  python -m pipeline.run --config "$CONFIG" --stage quantize \
    --set quantization.method="$METHOD" \
    --set quantization.scheme="$SCHEME"
  if [[ $? -ne 0 ]]; then
    echo "[FAIL] quantize method=$METHOD -- skipping serve/eval"
    return 1
  fi

  # Locate the checkpoint just produced (newest run dir for this slug).
  local SLUG="Qwen3-30B-A3B-${METHOD/+/-}-${SCHEME}"
  local CKPT
  CKPT=$(ls -dt artifacts/${SLUG}/*/checkpoint 2>/dev/null | head -1 || true)
  if [[ -z "${CKPT:-}" ]]; then
    echo "[FAIL] no checkpoint found for slug=$SLUG"
    return 1
  fi
  echo "checkpoint: $CKPT"

  # 2) Serve-verify (fresh process, single GPU).
  python -m pipeline.run --config "$CONFIG" --stage serve --checkpoint "$CKPT" \
    --set serve.tensor_parallel_size="$SERVE_TP" \
    --set serve.enable_expert_parallel=false
  [[ $? -ne 0 ]] && { echo "[WARN] serve-verify failed ($METHOD)"; rc=1; }

  # 3) Accuracy gate (informational until eval.baseline is set; see README).
  python -m pipeline.run --config "$CONFIG" --stage eval --checkpoint "$CKPT" \
    --set serve.tensor_parallel_size="$SERVE_TP"
  [[ $? -ne 0 ]] && { echo "[WARN] eval failed ($METHOD)"; rc=1; }

  echo "[done] method=$METHOD -> $(dirname "$CKPT")"
  return $rc
}
