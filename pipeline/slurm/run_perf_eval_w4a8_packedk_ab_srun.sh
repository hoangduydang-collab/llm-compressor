#!/usr/bin/env bash
# Controller (tmux) for the humming 0.1.10 vs 0.1.11 (packed-K) perf A/B on
# MiniMax-M3 W4A8.
#
# Adoption driver: upstream 0.1.11 auto-enables a packed-K weight layout for
# exactly our config (WGMMA + 8-bit activations + group-128 weight scales) --
# a dequant-throughput optimization. The patched 0.1.11 side-install passed
# correctness qualification (m3-humming-0111-packedk-qual/20260726T032735Z:
# forensics 0/96, full sweep clean; the single w13/4096 det=False is 1-ulp
# stream-K reduce-order wobble, upstream design, see
# pipeline/m3_humming_packedk_det_probe.py). This window measures whether
# packed-K actually helps at serving scale.
#
# Four arms, ONE window, one variable per pair (the humming site); identical
# checkpoint, topology, suite and RUN_TS. CUTLASS is not re-run: the packed-K
# question is humming-vs-humming; the CUTLASS anchor lives in window
# 20260725T122256Z (docs/m3-w4a8-three-arm-perf.md).
#
#   humming-w4afp8-indexed-0110  port 8005  0.1.10 indexed         (baseline)
#   humming-w4afp8-grouped-0110  port 8010  0.1.10 grouped         (baseline)
#   humming-w4afp8-indexed-0111  port 8006  0.1.11 indexed+packedK
#   humming-w4afp8-grouped-0111  port 8011  0.1.11 grouped+packedK
#
# Scope: this run MEASURES. No adoption decision and no pass/fail threshold is
# applied to the numbers by design.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
SITE_0110=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
SITE_0111=/mnt/nfs/hoangduy/venvs/humming-0.1.11-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-perf-w4a8-packedk-ab/$TS}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-perf-w4a8-packedk-ab
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-perf-w4a8-packedk-ab/latest_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
echo "[controller] run_ts=$RUN_TS"

# --- fail-closed preflight on the login node, before any GPU is allocated ----
fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

test -d "$CKPT" || fail "checkpoint missing: $CKPT"
test -d "$SITE_0110" || fail "humming 0.1.10 side-install missing: $SITE_0110"
test -d "$SITE_0111" || fail "humming 0.1.11 side-install missing: $SITE_0111"

# All four declared patches must be present on BOTH side-installs.
for site in "$SITE_0110" "$SITE_0111"; do
  for patch in ct_input_format grouped_expert_bounds tma_store_fence tma_store_commit; do
    ( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
        "pipeline/slurm/patch_humming_$patch.py" \
        --site "$site" --check ) >>"$ROOT/patch-checks.log" 2>&1 \
      || fail "humming $patch patch missing on $site (see $ROOT/patch-checks.log)"
  done
done

# Pin the versions each pair of arms will actually import, and assert the
# 0.1.11 site engages packed-K for our layer config -- the point of the A/B.
( PYTHONPATH="$SITE_0110:$REPO" "$PY" - <<'PY'
import importlib.metadata as md
import humming
import vllm
assert vllm.__version__ == "0.24.0", vllm.__version__
version = md.version("humming-kernels")
assert version == "0.1.10", version
assert humming.__file__.startswith("/mnt/nfs/hoangduy/venvs/humming-0.1.10-site/"), humming.__file__
print({"vllm": vllm.__version__, "humming": version, "humming_path": humming.__file__})
PY
) >"$ROOT/versions-0110.txt" 2>&1 || fail "0.1.10 version/import gate failed (see $ROOT/versions-0110.txt)"
cat "$ROOT/versions-0110.txt"

( PYTHONPATH="$SITE_0111:$REPO" "$PY" - <<'PY'
import importlib.metadata as md
import humming
import vllm
assert vllm.__version__ == "0.24.0", vllm.__version__
version = md.version("humming-kernels")
assert version == "0.1.11", version
assert humming.__file__.startswith("/mnt/nfs/hoangduy/venvs/humming-0.1.11-site/"), humming.__file__
from humming.layer import HummingLayerMeta
from pipeline.m3_humming_grouped_tile_forensics import build_layer_config
meta = HummingLayerMeta(**build_layer_config())
assert getattr(meta, "use_packed_k_layout", False), "packed-K did not engage"
print({"vllm": vllm.__version__, "humming": version,
       "humming_path": humming.__file__, "use_packed_k_layout": True})
PY
) >"$ROOT/versions-0111.txt" 2>&1 || fail "0.1.11 version/import gate failed (see $ROOT/versions-0111.txt)"
cat "$ROOT/versions-0111.txt"

printf '%s\n' "$CKPT" >"$ROOT/checkpoint.txt"
( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

# --- arms ------------------------------------------------------------------
declare -A ARM_PID=()

# $1 arm  $2 port  $3 quant-recipe label  $4 humming gemm type  $5 humming site
launch_arm() {
  local arm=$1 port=$2 recipe=$3 gemm=$4 site=$5
  # NOTE: these MUST go through `env`. Words produced by array expansion are not
  # parsed as assignment prefixes -- bash would try to execute the first one as
  # a command name, which is exactly how an arm silently died in the
  # 20260725T074535Z run.
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "MODE=local" "PROFILE=minimax-m3-inhouse"
    "CKPT=$CKPT" "PORT=$port"
    # M3_ARM (not PROFILE) namespaces RESULTS_ROOT on the benchmarks side:
    # results/minimax-m3-inhouse-$M3_ARM. Distinct per arm or they overwrite.
    "M3_ARM=$arm" "MODEL_PATH=$CKPT" "ENDPOINT_PORT=$port"
    "QUANT_RECIPE=$recipe" "M3_W4A8_BACKEND=humming"
    "VLLM_HUMMING_MOE_GEMM_TYPE=$gemm"
    "VLLM_HUMMING_USE_F16_ACCUM=0"
    "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
    "PYTHONPATH=$site:$REPO"
  )
  env "${arm_env[@]}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=12:00:00 --kill-on-bad-exit=1 --job-name="m3-perf-$arm" \
       --export=ALL \
       bash "$REPO/pipeline/slurm/perf_eval_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s backend=humming gemm=%s site=%s port=%s pid=%s launched=%s\n' \
    "$arm" "$gemm" "$site" "$port" "${ARM_PID[$arm]}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
  echo "[controller] launched $arm gemm=$gemm site=$site port=$port pid=${ARM_PID[$arm]}"
}

launch_arm humming-w4afp8-indexed-0110 8005 gptq-w4afp8-humming-indexed      indexed            "$SITE_0110"
launch_arm humming-w4afp8-grouped-0110 8010 gptq-w4afp8-humming-grouped      grouped_contiguous "$SITE_0110"
launch_arm humming-w4afp8-indexed-0111 8006 gptq-w4afp8-humming-indexed-pk   indexed            "$SITE_0111"
launch_arm humming-w4afp8-grouped-0111 8011 gptq-w4afp8-humming-grouped-pk   grouped_contiguous "$SITE_0111"

# Report each arm the moment IT exits, not in launch order; stamp every rc
# with the pid and time that produced it.
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
