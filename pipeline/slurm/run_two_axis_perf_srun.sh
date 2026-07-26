#!/usr/bin/env bash
# Controller (tmux) for the two-axis perf data-completion window
# (M3_TWO_AXIS_PERF_PLAN.md, user-signed 2026-07-26). 10 GPU nodes + 1 CPU task:
#
# Wave A — axis 2 (quantization; suite + extended AA per serve):
#   gptq-hum-idx-0110   1 node  port 8005  Humming idx 0.1.10 (anchors both axes)
#   awq-r7-hum-idx-0110 1 node  port 8007  Humming idx 0.1.10 (AWQ r7 gate-alpha)
#   cyankiwi            1 node  port 8003  Marlin W4A16 g32 (Humming N/A)
#   mxfp8               1 node  port 8002  native MXFP8 path
#   bf16                2 nodes port 8001  TP16/ray serve + CPU-only client
# Wave B — axis 1 (kernel; AA-only, model fixed = in-house GPTQ):
#   gptq-cutlass        1 node  port 8000  CUTLASS W4A8 MoE
#   gptq-hum-grp-0110   1 node  port 8010
#   gptq-hum-idx-0111   1 node  port 8006  packed-K       (debug partition)
#   gptq-hum-grp-0111   1 node  port 8011  packed-K       (debug partition)
#
# Every serve at MAX_MODEL_LEN=131072 (100k AA cells) except pre-agreed
# fallbacks (mxfp8/cyankiwi -> 40960, 100k cells marked n/a). Arms that exceed
# the idle-node count simply pend in slurm until a slot frees.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
GPTQ_CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
AWQ_CKPT="/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260723T123927Z-m3-ddp-awq-full-r7-gatealpha/awq/MiniMax-M3-awq-W4AFP8/20260723-123953/checkpoint-vllm-w123"
CYAN_CKPT="/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4"
MXFP8_CKPT="/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3-MXFP8"
BF16_CKPT="/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3"
SITE_0110=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
SITE_0111=/mnt/nfs/hoangduy/venvs/humming-0.1.11-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-two-axis-perf/$TS}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-two-axis-perf
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-two-axis-perf/latest_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
# Graph-capture ordering fix must hold on every serve incl. the 2-node BF16
# path (the smoke serve defaults it; the bf16 arm script does not).
export LLMC_M3_CAPTURE_SYNC=sync
echo "[controller] run_ts=$RUN_TS"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

for d in "$GPTQ_CKPT" "$AWQ_CKPT" "$CYAN_CKPT" "$MXFP8_CKPT" "$BF16_CKPT" \
         "$SITE_0110" "$SITE_0111"; do
  test -d "$d" || fail "missing dir: $d"
done

# aiperf pin: AA runner + suite analyzers are coupled to the 0.8 record schema.
aiperf_version=$(/mnt/nfs/hoangduy/venvs/perf/bin/python -c \
  "import importlib.metadata as md; print(md.version('aiperf'))" 2>/dev/null)
case "$aiperf_version" in 0.8.*) ;; *) fail "perf venv aiperf must be 0.8.x, got: ${aiperf_version:-none}";; esac
echo "[controller] aiperf=$aiperf_version"

# All four declared humming patches on BOTH side-installs.
for site in "$SITE_0110" "$SITE_0111"; do
  for patch in ct_input_format grouped_expert_bounds tma_store_fence tma_store_commit; do
    ( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
        "pipeline/slurm/patch_humming_$patch.py" \
        --site "$site" --check ) >>"$ROOT/patch-checks.log" 2>&1 \
      || fail "humming $patch patch missing on $site (see $ROOT/patch-checks.log)"
  done
done

# Serving preflight must accept both qualified humming versions.
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
import humming, vllm
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
import humming, vllm
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

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

# --- BF16 2-node serve (held until client-done) ------------------------------
HOLD_MAX=21600 READY_MAX=540 MAX_MODEL_LEN=131072 \
  srun --exclusive --nodes=2 --ntasks=2 --ntasks-per-node=1 \
     --gpus-per-node=8 --cpus-per-task=192 --time=12:00:00 \
     --kill-on-bad-exit=1 --job-name=m3-2ax-bf16 --export=ALL \
     bash "$REPO/pipeline/slurm/official_quality_bf16_http_arm.sh" "$ROOT" \
     > "$ROOT/bf16-srun.log" 2>&1 &
BF16_JOB=$!
echo "[controller] launched bf16 2-node serve pid=$BF16_JOB"

# --- GPU arms -----------------------------------------------------------------
declare -A ARM_PID=()

# $1 arm  $2 workloads  $3 port  $4 partition  $5 ckpt  $6.. extra env words
launch_arm() {
  local arm=$1 workloads=$2 port=$3 part=$4 ckpt=$5; shift 5
  # Assignment-prefix words from array expansion are not parsed as assignments
  # by bash — they MUST go through `env` (20260725T074535Z postmortem).
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "MODE=local" "WORKLOADS=$workloads"
    "CKPT=$ckpt" "PORT=$port" "RUN_TS=$RUN_TS"
    "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
    "$@"
  )
  env "${arm_env[@]}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=12:00:00 --kill-on-bad-exit=1 --partition="$part" \
       --job-name="m3-2ax-$arm" --export=ALL \
       bash "$REPO/pipeline/slurm/two_axis_perf_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s workloads=%s port=%s partition=%s pid=%s launched=%s\n' \
    "$arm" "$workloads" "$port" "$part" "${ARM_PID[$arm]}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
  echo "[controller] launched $arm workloads=$workloads port=$port partition=$part pid=${ARM_PID[$arm]}"
}

# Wave A — axis 2 (suite + AA)
launch_arm gptq-hum-idx-0110 suite,aa 8005 compute "$GPTQ_CKPT" \
  "PROFILE=minimax-m3-inhouse" "M3_ARM=gptq-hum-idx-0110" "MODEL_PATH=$GPTQ_CKPT" \
  "ENDPOINT_PORT=8005" "QUANT_RECIPE=gptq-w4afp8-humming-indexed" \
  "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=indexed" \
  "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0110:$REPO"
launch_arm awq-r7-hum-idx-0110 suite,aa 8007 compute "$AWQ_CKPT" \
  "PROFILE=minimax-m3-inhouse" "M3_ARM=awq-r7-hum-idx-0110" "MODEL_PATH=$AWQ_CKPT" \
  "ENDPOINT_PORT=8007" "QUANT_RECIPE=awq-w4afp8-r7-humming-indexed" \
  "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=indexed" \
  "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0110:$REPO"
launch_arm cyankiwi suite,aa 8003 compute "$CYAN_CKPT" \
  "PROFILE=minimax-m3-awq-cyankiwi" "MODEL_PATH=$CYAN_CKPT" "ENDPOINT_PORT=8003" \
  "AA_PROFILE=minimax-m3-awq-cyankiwi" "FALLBACK_MML=40960"
launch_arm mxfp8 suite,aa 8002 compute "$MXFP8_CKPT" \
  "PROFILE=minimax-m3-mxfp8" "MODEL_PATH=$MXFP8_CKPT" "ENDPOINT_PORT=8002" \
  "AA_PROFILE=minimax-m3-mxfp8" "FALLBACK_MML=40960"

# Wave B — axis 1 (AA only, model fixed = in-house GPTQ)
launch_arm gptq-cutlass aa 8000 compute "$GPTQ_CKPT" \
  "M3_W4A8_BACKEND=cutlass"
launch_arm gptq-hum-grp-0110 aa 8010 compute "$GPTQ_CKPT" \
  "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=grouped_contiguous" \
  "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0110:$REPO"
launch_arm gptq-hum-idx-0111 aa 8006 debug "$GPTQ_CKPT" \
  "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=indexed" \
  "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0111:$REPO"
launch_arm gptq-hum-grp-0111 aa 8011 debug "$GPTQ_CKPT" \
  "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=grouped_contiguous" \
  "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0111:$REPO"

# BF16 client: CPU-only, suite + AA against the shared 2-node endpoint.
env "ROOT=$ROOT" "ARM=bf16" "MODE=remote" "WORKLOADS=suite,aa" "RUN_TS=$RUN_TS" \
    "PROFILE=minimax-m3-bf16" "AA_PROFILE=minimax-m3-bf16" "BF16_PORT=8001" \
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --time=12:00:00 \
       --job-name=m3-2ax-bf16probe --export=ALL \
       bash "$REPO/pipeline/slurm/two_axis_perf_arm.sh" \
       > "$ROOT/bf16probe-srun.log" 2>&1 &
ARM_PID[bf16]=$!
printf 'arm=bf16 workloads=suite,aa port=8001 partition=cpu pid=%s launched=%s\n' \
  "${ARM_PID[bf16]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
echo "[controller] launched bf16 client pid=${ARM_PID[bf16]}"

rc_all=0
remaining=${#ARM_PID[@]}
while [ "$remaining" -gt 0 ]; do
  for arm in "${!ARM_PID[@]}"; do
    pid=${ARM_PID[$arm]}
    [ -n "$pid" ] || continue
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      printf 'rc=%s pid=%s arm=%s finished=%s\n' \
        "$rc" "$pid" "$arm" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$ROOT/arm-$arm.rc"
      echo "[controller] arm $arm rc=$rc (pid=$pid)"
      [ "$rc" = 0 ] || rc_all=1
      [ "$arm" = bf16 ] && touch "$ROOT/client-done"   # release held BF16 serve
      ARM_PID[$arm]=""
      remaining=$((remaining - 1))
    fi
  done
  [ "$remaining" -gt 0 ] && sleep 60
done

touch "$ROOT/client-done"     # idempotent safety if bf16 client never ran
wait "$BF16_JOB"; echo "[controller] bf16 serve rc=$?"
echo "$rc_all" > "$ROOT/controller.rc"
echo "[controller] done rc=$rc_all"
echo "CONTROLLER_RC=$rc_all"
