#!/usr/bin/env bash
# Controller (tmux) for EAGLE3 spec-dec wave 2 -- M3_SPECDEC_EAGLE3_PLAN.md.
#
# 6 GPU nodes: three phases x two arms (control k=0, k=3), every phase on its OWN
# pair of serves so the phases run on parallel hardware. They deliberately do not
# share a server: a conc-64 load running beside a conc-1 latency cell would
# confound both measurements.
#
#   natural-k0 / natural-k3   ports 8030/8031  ShareGPT real prompts, natural
#                                              output, temp 0.6 then 0, conc 1,10
#   load-k0    / load-k3      ports 8032/8033  reasoning 1k/8k pinned, conc 16,32,64
#   lowconc-k0 / lowconc-k3   ports 8034/8035  reasoning 1k/8k pinned, conc 1,4
#
# k=1 and k=5 are dropped: wave 1 measured both as dominated by k=3.
# Nodes known to carry another user's out-of-band GPU jobs are excluded up front;
# EXCLUDE can be extended at launch.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
GPTQ_CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
DRAFTER=/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3
SHAREGPT="$REPO/artifacts/aiperf-datasets/.cache/aiperf/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
SITE_0110=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python
EXCLUDE=${EXCLUDE:-gpu-h97,gpu-h98,gpu-h101}

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/$TS-wave2}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-specdec-eagle3
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-specdec-eagle3/latest_wave2_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
export LLMC_M3_CAPTURE_SYNC=sync
echo "[controller] run_ts=$RUN_TS exclude=$EXCLUDE"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

for d in "$GPTQ_CKPT" "$SITE_0110" "$DRAFTER"; do
  test -d "$d" || fail "missing dir: $d"
done
# ShareGPT must be pre-staged: the arms run with HF_HUB_OFFLINE=1 and no network.
test -s "$SHAREGPT" || fail "ShareGPT not staged at $SHAREGPT"
echo "[controller] sharegpt=$(du -h "$SHAREGPT" | cut -f1)"

drafter_arch=$("$PY" - <<PY
import json
print(json.load(open("$DRAFTER/config.json"))["architectures"][0])
PY
) || fail "cannot read drafter config"
[ "$drafter_arch" = "LlamaForCausalLMEagle3" ] || fail "unexpected drafter arch: $drafter_arch"

aiperf_version=$(/mnt/nfs/hoangduy/venvs/perf/bin/python -c \
  "import importlib.metadata as md; print(md.version('aiperf'))" 2>/dev/null)
case "$aiperf_version" in 0.8.*) ;; *) fail "perf venv aiperf must be 0.8.x, got: ${aiperf_version:-none}";; esac
echo "[controller] aiperf=$aiperf_version drafter=$drafter_arch"

for patch in ct_input_format grouped_expert_bounds tma_store_fence tma_store_commit; do
  ( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
      "pipeline/slurm/patch_humming_$patch.py" --site "$SITE_0110" --check ) \
      >>"$ROOT/patch-checks.log" 2>&1 \
    || fail "humming $patch patch missing on $SITE_0110"
done
echo "[controller] humming 0.1.10 patches present"

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

declare -A ARM_PID=()

# $1 phase  $2 spec_k  $3 port
launch_arm() {
  local phase=$1 k=$2 port=$3 arm="$1-k$2"
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "RUN_TS=$RUN_TS" "PHASE=$phase" "SPEC_K=$k"
    "CKPT=$GPTQ_CKPT" "PORT=$port" "DRAFTER=$DRAFTER"
    "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
    "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=indexed"
    "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0110:$REPO"
  )
  env "${arm_env[@]}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=06:00:00 --kill-on-bad-exit=1 --partition=compute \
       --exclude="$EXCLUDE" --job-name="m3-w2-$arm" --export=ALL \
       bash "$REPO/pipeline/slurm/specdec_wave2_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s phase=%s spec_k=%s port=%s pid=%s launched=%s\n' \
    "$arm" "$phase" "$k" "$port" "${ARM_PID[$arm]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"$ROOT/arm-provenance.txt"
  echo "[controller] launched $arm port=$port pid=${ARM_PID[$arm]}"
}

launch_arm natural 0 8030
launch_arm natural 3 8031
launch_arm load    0 8032
launch_arm load    3 8033
launch_arm lowconc 0 8034
launch_arm lowconc 3 8035

echo "[controller] all 6 arms launched; waiting"
rc_all=0
for arm in "${!ARM_PID[@]}"; do
  if wait "${ARM_PID[$arm]}"; then
    echo "[controller] $arm rc=0"
  else
    rc=$?
    echo "[controller] $arm rc=$rc"
    echo "$arm rc=$rc" >>"$ROOT/arm-failures.txt"
    rc_all=1
  fi
done

echo "[controller] done rc_all=$rc_all root=$ROOT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/controller-done.txt"
exit "$rc_all"
