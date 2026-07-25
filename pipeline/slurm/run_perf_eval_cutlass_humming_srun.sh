#!/usr/bin/env bash
# Controller (tmux) for the PAIRED CUTLASS-vs-Humming W4A8 kernel comparison.
#
# One variable: M3_W4A8_BACKEND. Both arms serve the SAME checkpoint
# (gptq-checkpoint-vllm-w123-abi-overlay, the one qualified in
# M3_HUMMING_W4A8_QUALIFICATION_REPORT.md), on the same node type, with the same
# suite, under one shared RUN_TS:
#
#   cutlass  port 8000  M3_W4A8_BACKEND=cutlass  (control)
#   humming  port 8005  M3_W4A8_BACKEND=humming, indexed GEMM, FP32 accum
#
# The control is re-measured fresh in this window. Do NOT compare either arm
# against historical perf numbers: serve defaults have changed since earlier
# passes. Cross-arm comparison WITHIN this run is the valid claim.
#
# Suite: performance/scripts/run_performance.sh with PERF_STRICT=1 —
# reasoning (CONC 1 4 16 64) + agentic warm/cold (CONC 1 4 16 32). M3 profiles
# self-skip non-reasoning (thinking cannot be disabled).
#
# Results: benchmarks/results/minimax-m3-inhouse-<arm>/vllm/perf/*/$RUN_TS
#
# Scope: this run MEASURES. It does not decide adoption; no pass/fail threshold
# is applied to the numbers by design.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
HUMMING_SITE=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-perf-cutlass-humming/$TS}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-perf-cutlass-humming
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-perf-cutlass-humming/latest_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
echo "[controller] run_ts=$RUN_TS"

# --- fail-closed preflight on the login node, before any GPU is allocated ----
fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

test -d "$CKPT" || fail "checkpoint missing: $CKPT"
test -d "$HUMMING_SITE" || fail "humming side-install missing: $HUMMING_SITE"

# The declared pack-quantized admission patch must be present, or every Humming
# worker dies during model init. Cheap to check here; expensive to discover on
# an allocated node.
( cd "$REPO" && PYTHONPATH="$REPO" \
  /mnt/nfs/hoangduy/venvs/quant/bin/python \
    pipeline/slurm/patch_humming_ct_input_format.py \
    --site "$HUMMING_SITE" --check ) >"$ROOT/humming-patch-check.log" 2>&1 \
  || fail "humming ct-input patch missing (see $ROOT/humming-patch-check.log)"

# Pin the versions the Humming arm will actually import.
( PYTHONPATH="$HUMMING_SITE:$REPO" /mnt/nfs/hoangduy/venvs/quant/bin/python - <<'PY'
import importlib.metadata as md
import humming
import vllm

SITE = "/mnt/nfs/hoangduy/venvs/humming-0.1.10-site/"
version = md.version("humming-kernels")
assert vllm.__version__ == "0.24.0", vllm.__version__
assert version == "0.1.10", version
assert humming.__file__.startswith(SITE), humming.__file__
print({
    "vllm": vllm.__version__,
    "humming": version,
    "humming_path": humming.__file__,
})
PY
) >"$ROOT/versions.txt" 2>&1 \
  || fail "version/import gate failed (see $ROOT/versions.txt)"
cat "$ROOT/versions.txt"

printf '%s\n' "$CKPT" >"$ROOT/checkpoint.txt"
( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

# --- arms ------------------------------------------------------------------
# $1 arm  $2 backend  $3 port  $4 quant-recipe label
launch_arm() {
  local arm=$1 backend=$2 port=$3 recipe=$4
  local extra_pythonpath=""
  local -a env_extra=()
  if [ "$backend" = humming ]; then
    extra_pythonpath="$HUMMING_SITE"
    env_extra=(
      VLLM_HUMMING_MOE_GEMM_TYPE=indexed
      VLLM_HUMMING_USE_F16_ACCUM=0
      HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming
    )
  fi
  ROOT="$ROOT" ARM="$arm" MODE=local PROFILE=minimax-m3-inhouse \
  CKPT="$CKPT" PORT="$port" \
  M3_ARM="$arm" MODEL_PATH="$CKPT" ENDPOINT_PORT="$port" QUANT_RECIPE="$recipe" \
  M3_W4A8_BACKEND="$backend" \
  PYTHONPATH="${extra_pythonpath:+$extra_pythonpath:}$REPO" \
  "${env_extra[@]}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=12:00:00 --kill-on-bad-exit=1 --job-name="m3-perf-$arm" \
       --export=ALL \
       bash "$REPO/pipeline/slurm/perf_eval_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  LAST_PID=$!
  echo "[controller] launched $arm backend=$backend port=$port pid=$LAST_PID"
}

launch_arm cutlass-w4afp8 cutlass 8000 gptq-w4afp8-cutlass
CUTLASS_PID=$LAST_PID
launch_arm humming-w4afp8-indexed humming 8005 gptq-w4afp8-humming-indexed
HUMMING_PID=$LAST_PID

rc_all=0
for spec in "cutlass-w4afp8:$CUTLASS_PID" "humming-w4afp8-indexed:$HUMMING_PID"; do
  arm=${spec%%:*}; pid=${spec##*:}
  wait "$pid"; rc=$?
  echo "$rc" > "$ROOT/perf-$arm.rc"
  echo "[controller] perf $arm rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
done

echo "$rc_all" > "$ROOT/controller.rc"
echo "[controller] done rc=$rc_all"
echo "CONTROLLER_RC=$rc_all"
