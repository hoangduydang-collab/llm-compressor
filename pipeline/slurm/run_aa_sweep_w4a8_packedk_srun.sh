#!/usr/bin/env bash
# Controller (tmux) for the AA-style perf sweep over the same four humming
# W4A8 arms as the packed-K A/B window (run_perf_eval_w4a8_packedk_ab_srun.sh):
#
#   humming-w4afp8-indexed-0110  port 8005  0.1.10 indexed          compute
#   humming-w4afp8-grouped-0110  port 8010  0.1.10 grouped          compute
#   humming-w4afp8-indexed-0111  port 8006  0.1.11 indexed+packedK  compute
#   humming-w4afp8-grouped-0111  port 8011  0.1.11 grouped+packedK  debug
#
# This is measurement path 2 (AA public-vocabulary numbers: TTFT, output speed,
# natural OSL at input 1k/10k x conc 1/10) alongside -- NOT instead of -- the
# suite-native window 20260726T033158Z, which stays untouched on its own nodes.
# The fourth arm goes to the debug partition (same 8xH100 nodes, no time limit)
# because only three compute nodes were idle; user-authorized.
#
# Scope: this run MEASURES. No adoption decision is applied to the numbers.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
SITE_0110=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
SITE_0111=/mnt/nfs/hoangduy/venvs/humming-0.1.11-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-aa-sweep-w4a8-packedk/$TS}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-aa-sweep-w4a8-packedk
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-aa-sweep-w4a8-packedk/latest_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
echo "[controller] run_ts=$RUN_TS"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

test -d "$CKPT" || fail "checkpoint missing: $CKPT"
test -d "$SITE_0110" || fail "humming 0.1.10 side-install missing: $SITE_0110"
test -d "$SITE_0111" || fail "humming 0.1.11 side-install missing: $SITE_0111"

# aiperf pin: the AA runner + analyzers are coupled to the 0.8 record schema.
aiperf_version=$(/mnt/nfs/hoangduy/venvs/perf/bin/python -c \
  "import importlib.metadata as md; print(md.version('aiperf'))" 2>/dev/null)
case "$aiperf_version" in 0.8.*) ;; *) fail "perf venv aiperf must be 0.8.x, got: ${aiperf_version:-none}";; esac
echo "[controller] aiperf=$aiperf_version"

# All four declared patches must be present on BOTH side-installs.
for site in "$SITE_0110" "$SITE_0111"; do
  for patch in ct_input_format grouped_expert_bounds tma_store_fence tma_store_commit; do
    ( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
        "pipeline/slurm/patch_humming_$patch.py" \
        --site "$site" --check ) >>"$ROOT/patch-checks.log" 2>&1 \
      || fail "humming $patch patch missing on $site (see $ROOT/patch-checks.log)"
  done
done

# The serving preflight must accept both qualified versions (the gate that
# killed the first packed-K A/B launch).
( PYTHONPATH="$REPO" "$PY" - <<'PY'
from pipeline.m3_humming_w4a8 import EXPECTED_HUMMING_VERSIONS
assert "0.1.10" in EXPECTED_HUMMING_VERSIONS, EXPECTED_HUMMING_VERSIONS
assert "0.1.11" in EXPECTED_HUMMING_VERSIONS, EXPECTED_HUMMING_VERSIONS
print({"expected_humming_versions": EXPECTED_HUMMING_VERSIONS})
PY
) >"$ROOT/repo-gate.txt" 2>&1 || fail "repo preflight rejects a qualified version (see $ROOT/repo-gate.txt)"
cat "$ROOT/repo-gate.txt"

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

# $1 arm  $2 port  $3 humming gemm type  $4 humming site  $5 slurm partition
launch_arm() {
  local arm=$1 port=$2 gemm=$3 site=$4 part=$5
  # Assignment-prefix words from array expansion are not parsed as assignments
  # by bash -- they MUST go through `env` (see 20260725T074535Z postmortem).
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "CKPT=$CKPT" "PORT=$port"
    "M3_W4A8_BACKEND=humming"
    "VLLM_HUMMING_MOE_GEMM_TYPE=$gemm"
    "VLLM_HUMMING_USE_F16_ACCUM=0"
    "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
    "PYTHONPATH=$site:$REPO"
  )
  env "${arm_env[@]}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=12:00:00 --kill-on-bad-exit=1 --partition="$part" \
       --job-name="m3-aa-$arm" \
       --export=ALL \
       bash "$REPO/pipeline/slurm/aa_sweep_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s backend=humming gemm=%s site=%s port=%s partition=%s pid=%s launched=%s\n' \
    "$arm" "$gemm" "$site" "$port" "$part" "${ARM_PID[$arm]}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
  echo "[controller] launched $arm gemm=$gemm site=$site port=$port partition=$part pid=${ARM_PID[$arm]}"
}

launch_arm humming-w4afp8-indexed-0110 8005 indexed            "$SITE_0110" compute
launch_arm humming-w4afp8-grouped-0110 8010 grouped_contiguous "$SITE_0110" compute
launch_arm humming-w4afp8-indexed-0111 8006 indexed            "$SITE_0111" compute
launch_arm humming-w4afp8-grouped-0111 8011 grouped_contiguous "$SITE_0111" debug

rc_all=0
remaining=${#ARM_PID[@]}
while [ "$remaining" -gt 0 ]; do
  for arm in "${!ARM_PID[@]}"; do
    pid=${ARM_PID[$arm]}
    [ -n "$pid" ] || continue
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      printf 'rc=%s pid=%s arm=%s finished=%s\n' \
        "$rc" "$pid" "$arm" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$ROOT/aa-$arm.rc"
      echo "[controller] aa $arm rc=$rc (pid=$pid)"
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
