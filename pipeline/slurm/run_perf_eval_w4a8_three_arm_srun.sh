#!/usr/bin/env bash
# Controller (tmux) for the 3-ARM W4A8 kernel serving comparison on MiniMax-M3.
#
# One variable per arm; identical checkpoint, topology, suite and RUN_TS:
#
#   cutlass-w4afp8            port 8000  CUTLASS W4A8-FP8 MoE          (control)
#   humming-w4afp8-indexed    port 8005  Humming, indexed GEMM         (arm 2)
#   humming-w4afp8-grouped    port 8010  Humming, grouped_contiguous   (arm 3)
#
# Why all three in ONE window. M3_PAIRED_CUTLASS_HUMMING_PERF_REPORT.md measured
# CUTLASS vs Humming-indexed at RUN_TS=20260725T074535Z and had to warn that
# serve defaults have changed since earlier passes, so its numbers are only
# valid *within* that window. Re-measuring the control alongside both Humming
# strategies makes every pairwise comparison here directly supportable instead
# of chained through a stale window.
#
# What differs between the two Humming arms is one env var, and the kernel it
# selects is genuinely different rather than a tuning tweak: humming/tune/sm90.py
# grants use_tma + use_warp_spec + use_mbarrier to every gemm type EXCEPT
# indexed, so arm 3 compiles the warp-specialized kernel (humming_ws.cuh) while
# arm 2 runs cp.async. Block/warp shapes, stages and stream-K are identical --
# see pipeline/m3_humming_gemm_type_probe.py and
# evidence/m3-arm3-gemm-probe/.
#
# Scope: this run MEASURES. No adoption decision and no pass/fail threshold is
# applied to the numbers by design.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
HUMMING_SITE=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-perf-w4a8-three-arm/$TS}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-perf-w4a8-three-arm
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-perf-w4a8-three-arm/latest_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
echo "[controller] run_ts=$RUN_TS"

# --- fail-closed preflight on the login node, before any GPU is allocated ----
fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

test -d "$CKPT" || fail "checkpoint missing: $CKPT"
test -d "$HUMMING_SITE" || fail "humming side-install missing: $HUMMING_SITE"

# The declared pack-quantized admission patch must be present, or every Humming
# worker dies during model init. Cheap here; expensive on an allocated node.
( cd "$REPO" && PYTHONPATH="$REPO" \
  /mnt/nfs/hoangduy/venvs/quant/bin/python \
    pipeline/slurm/patch_humming_ct_input_format.py \
    --site "$HUMMING_SITE" --check ) >"$ROOT/humming-patch-check.log" 2>&1 \
  || fail "humming ct-input patch missing (see $ROOT/humming-patch-check.log)"

# Pin the versions the Humming arms will actually import.
( PYTHONPATH="$HUMMING_SITE:$REPO" /mnt/nfs/hoangduy/venvs/quant/bin/python - <<'PY'
import importlib.metadata as md
import humming
import vllm

SITE = "/mnt/nfs/hoangduy/venvs/humming-0.1.10-site/"
version = md.version("humming-kernels")
assert vllm.__version__ == "0.24.0", vllm.__version__
assert version == "0.1.10", version
assert humming.__file__.startswith(SITE), humming.__file__
print({"vllm": vllm.__version__, "humming": version, "humming_path": humming.__file__})
PY
) >"$ROOT/versions.txt" 2>&1 || fail "version/import gate failed (see $ROOT/versions.txt)"
cat "$ROOT/versions.txt"

printf '%s\n' "$CKPT" >"$ROOT/checkpoint.txt"
( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

# --- arms ------------------------------------------------------------------
declare -A ARM_PID=()

# $1 arm  $2 backend  $3 port  $4 quant-recipe label  $5 humming gemm type ("" for cutlass)
launch_arm() {
  local arm=$1 backend=$2 port=$3 recipe=$4 gemm=$5
  local pythonpath="$REPO"
  # NOTE: these MUST go through `env`. Words produced by array expansion are not
  # parsed as assignment prefixes -- bash would try to execute the first one as a
  # command name ("VLLM_HUMMING_MOE_GEMM_TYPE=indexed: command not found"), which
  # is exactly how an arm silently died in the 20260725T074535Z run.
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "MODE=local" "PROFILE=minimax-m3-inhouse"
    "CKPT=$CKPT" "PORT=$port"
    # M3_ARM (not PROFILE) namespaces RESULTS_ROOT on the benchmarks side:
    # results/minimax-m3-inhouse-$M3_ARM. Distinct per arm or they overwrite.
    "M3_ARM=$arm" "MODEL_PATH=$CKPT" "ENDPOINT_PORT=$port"
    "QUANT_RECIPE=$recipe" "M3_W4A8_BACKEND=$backend"
  )
  if [ "$backend" = humming ]; then
    pythonpath="$HUMMING_SITE:$REPO"
    arm_env+=(
      "VLLM_HUMMING_MOE_GEMM_TYPE=$gemm"
      "VLLM_HUMMING_USE_F16_ACCUM=0"
      "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
    )
  fi
  arm_env+=("PYTHONPATH=$pythonpath")
  env "${arm_env[@]}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=12:00:00 --kill-on-bad-exit=1 --job-name="m3-perf-$arm" \
       --export=ALL \
       bash "$REPO/pipeline/slurm/perf_eval_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s backend=%s gemm=%s port=%s pid=%s launched=%s\n' \
    "$arm" "$backend" "${gemm:-n/a}" "$port" "${ARM_PID[$arm]}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
  echo "[controller] launched $arm backend=$backend gemm=${gemm:-n/a} port=$port pid=${ARM_PID[$arm]}"
}

launch_arm cutlass-w4afp8         cutlass 8000 gptq-w4afp8-cutlass          ""
launch_arm humming-w4afp8-indexed humming 8005 gptq-w4afp8-humming-indexed  indexed
launch_arm humming-w4afp8-grouped humming 8010 gptq-w4afp8-humming-grouped  grouped_contiguous

# Report each arm the moment IT exits, not in launch order. The previous
# controller's sequential `wait` meant a fast-failing arm went unnoticed until
# the slowest finished; and because rc files were keyed only by arm name, a
# stale writer later clobbered a real arm's result. Both are fixed here: poll
# for liveness, and stamp every rc with the pid and time that produced it.
rc_all=0
remaining=${#ARM_PID[@]}
while [ "$remaining" -gt 0 ]; do
  for arm in "${!ARM_PID[@]}"; do
    pid=${ARM_PID[$arm]}
    [ -n "$pid" ] || continue
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      printf 'rc=%s pid=%s arm=%s finished=%s\n' \
        "$rc" "$pid" "$arm" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$ROOT/perf-$arm.rc"
      echo "[controller] perf $arm rc=$rc (pid=$pid)"
      [ "$rc" = 0 ] || rc_all=1
      ARM_PID[$arm]=""
      remaining=$((remaining - 1))
    fi
  done
  [ "$remaining" -gt 0 ] && sleep 60
done

echo "$rc_all" > "$ROOT/controller.rc"
echo "[controller] done rc=$rc_all"
echo "CONTROLLER_RC=$rc_all"
