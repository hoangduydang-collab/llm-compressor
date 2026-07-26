#!/usr/bin/env bash
# Relaunch of the two 0.1.11 packed-K arms of the perf A/B window
# 20260726T033158Z (run_perf_eval_w4a8_packedk_ab_srun.sh).
#
# Why a separate controller: the original window's 0.1.11 arms fast-failed at
# serve preflight with HUMMING_VERSION_MISMATCH -- pipeline/m3_humming_w4a8.py
# still pinned EXPECTED_HUMMING_VERSION="0.1.10" (everything else attested
# clean: all four declared patches record-matched). The gate now accepts both
# qualified versions (EXPECTED_HUMMING_VERSIONS). The original controller is
# still running its two 0.1.10 arms and must not be edited mid-run, and its
# poll loop already discarded the failed pids, so the 0.1.11 arms rejoin the
# SAME window (same ROOT, same RUN_TS) from here.
#
# NOTE: the original controller's controller.rc will report rc_all=1 (it saw
# the rc=1 exits). relaunch-0111.rc written here is the authoritative rc for
# the 0.1.11 pair; perf-<arm>.rc files are overwritten on completion.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
SITE_0111=/mnt/nfs/hoangduy/venvs/humming-0.1.11-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python

ROOT=/mnt/nfs/hoangduy/results/m3-perf-w4a8-packedk-ab/20260726T033158Z
export RUN_TS=20260726T033158Z
echo "[relaunch] root=$ROOT run_ts=$RUN_TS"

fail() { echo "[relaunch] ABORT: $1" >&2; echo "$1" >"$ROOT/relaunch-abort.txt"; exit 1; }

test -d "$ROOT" || fail "window root missing: $ROOT"
test -d "$CKPT" || fail "checkpoint missing: $CKPT"
test -d "$SITE_0111" || fail "humming 0.1.11 side-install missing: $SITE_0111"

# The exact gate that killed the first attempt: the serving preflight must now
# accept 0.1.11. Assert it against the repo code the arms will import.
( PYTHONPATH="$REPO" "$PY" - <<'PY'
from pipeline.m3_humming_w4a8 import EXPECTED_HUMMING_VERSIONS
assert "0.1.11" in EXPECTED_HUMMING_VERSIONS, EXPECTED_HUMMING_VERSIONS
print({"expected_humming_versions": EXPECTED_HUMMING_VERSIONS})
PY
) >"$ROOT/relaunch-gate.txt" 2>&1 || fail "repo preflight still rejects 0.1.11 (see $ROOT/relaunch-gate.txt)"
cat "$ROOT/relaunch-gate.txt"

for patch in ct_input_format grouped_expert_bounds tma_store_fence tma_store_commit; do
  ( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
      "pipeline/slurm/patch_humming_$patch.py" \
      --site "$SITE_0111" --check ) >>"$ROOT/relaunch-patch-checks.log" 2>&1 \
    || fail "humming $patch patch missing on $SITE_0111"
done

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
) >"$ROOT/relaunch-versions-0111.txt" 2>&1 || fail "0.1.11 version/import gate failed"
cat "$ROOT/relaunch-versions-0111.txt"

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/relaunch-commit.txt"

declare -A ARM_PID=()

launch_arm() {
  local arm=$1 port=$2 recipe=$3 gemm=$4 site=$5
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "MODE=local" "PROFILE=minimax-m3-inhouse"
    "CKPT=$CKPT" "PORT=$port"
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
       >> "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s backend=humming gemm=%s site=%s port=%s pid=%s relaunched=%s\n' \
    "$arm" "$gemm" "$site" "$port" "${ARM_PID[$arm]}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
  echo "[relaunch] launched $arm gemm=$gemm port=$port pid=${ARM_PID[$arm]}"
}

launch_arm humming-w4afp8-indexed-0111 8006 gptq-w4afp8-humming-indexed-pk indexed            "$SITE_0111"
launch_arm humming-w4afp8-grouped-0111 8011 gptq-w4afp8-humming-grouped-pk grouped_contiguous "$SITE_0111"

rc_all=0
remaining=${#ARM_PID[@]}
while [ "$remaining" -gt 0 ]; do
  for arm in "${!ARM_PID[@]}"; do
    pid=${ARM_PID[$arm]}
    [ -n "$pid" ] || continue
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      printf 'rc=%s pid=%s arm=%s finished=%s relaunch=1\n' \
        "$rc" "$pid" "$arm" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$ROOT/perf-$arm.rc"
      echo "[relaunch] perf $arm rc=$rc (pid=$pid)"
      [ "$rc" = 0 ] || rc_all=1
      ARM_PID[$arm]=""
      remaining=$((remaining - 1))
    fi
  done
  [ "$remaining" -gt 0 ] && sleep 60
done

echo "$rc_all" > "$ROOT/relaunch-0111.rc"
echo "[relaunch] done rc=$rc_all"
echo "RELAUNCH_RC=$rc_all"
