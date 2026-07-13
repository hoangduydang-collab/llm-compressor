#!/usr/bin/env bash
# Run one probe-free MiniMax-M3 production quantization and statically verify it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VARIANT=""
LANE_ROOT=""
CONFIG="pipeline/configs/minimax_m3_full_calib.yaml"
MODEL_ID=""
NUM_SAMPLES=""
MAX_SEQ_LENGTH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --lane-root) LANE_ROOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --max-seq-length) MAX_SEQ_LENGTH="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$VARIANT" && -n "$LANE_ROOT" ]] || {
  echo "--variant and --lane-root are required" >&2
  exit 2
}
if [[ -e "$LANE_ROOT" ]]; then
  echo "refusing non-fresh lane root: $LANE_ROOT" >&2
  exit 2
fi
mkdir -p "$LANE_ROOT"

case "$VARIANT" in
  offsetfix)
    METHOD=awq
    export M3_AWQ_DISABLE_MLP_INPUT_SMOOTH=0
    ;;
  nosmooth)
    METHOD=awq
    export M3_AWQ_DISABLE_MLP_INPUT_SMOOTH=1
    ;;
  quant_only)
    METHOD=quant_only
    export M3_AWQ_DISABLE_MLP_INPUT_SMOOTH=0
    ;;
  *)
    echo "unknown safe variant: $VARIANT" >&2
    exit 2
    ;;
esac

command=(
  python -m pipeline.run --config "$CONFIG" --stage quantize
  --set "quantization.method=$METHOD"
  --set "output_dir=$LANE_ROOT/runs"
)
[[ -z "$MODEL_ID" ]] || command+=(--set "model.id=$MODEL_ID")
[[ -z "$NUM_SAMPLES" ]] || command+=(--set "calibration.num_samples=$NUM_SAMPLES")
[[ -z "$MAX_SEQ_LENGTH" ]] || command+=(--set "calibration.max_seq_length=$MAX_SEQ_LENGTH")
{
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$LANE_ROOT/command.txt"

"${command[@]}"

mapfile -t checkpoints < <(
  find "$LANE_ROOT/runs" -mindepth 3 -maxdepth 3 -type d -name checkpoint | sort
)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "expected exactly one checkpoint, found ${#checkpoints[@]}" >&2
  printf '%s\n' "${checkpoints[@]}" >&2
  exit 3
fi
checkpoint="${checkpoints[0]}"
printf '%s\n' "$checkpoint" >"$LANE_ROOT/checkpoint.path.tmp"
mv "$LANE_ROOT/checkpoint.path.tmp" "$LANE_ROOT/checkpoint.path"

static_rc=0
python -m pipeline.verify_quant_checkpoint --ckpt "$checkpoint" --check-tensors \
  >"$LANE_ROOT/static_checkpoint_verification.log" 2>&1 || static_rc=$?
printf '%s\n' "$static_rc" >"$LANE_ROOT/static_checkpoint_verification.rc.tmp"
mv "$LANE_ROOT/static_checkpoint_verification.rc.tmp" \
  "$LANE_ROOT/static_checkpoint_verification.rc"
if [[ "$static_rc" -ne 0 ]]; then
  echo "static checkpoint verification failed rc=$static_rc checkpoint=$checkpoint" >&2
  exit "$static_rc"
fi

echo "safe lane complete variant=$VARIANT checkpoint=$checkpoint"
