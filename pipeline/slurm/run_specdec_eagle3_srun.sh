#!/usr/bin/env bash
# Controller (tmux) for the EAGLE3 spec-dec A/B wave 1 -- M3_SPECDEC_EAGLE3_PLAN.md.
# 4 GPU nodes, one serve each, everything except --speculative-config identical to
# the 20260726T132617Z window's gptq-hum-idx-0110 arm:
#
#   k0-control  1 node  port 8020  no speculative config (in-window control)
#   k1          1 node  port 8021  eagle3 num_speculative_tokens=1
#   k3          1 node  port 8022  eagle3 num_speculative_tokens=3  (recipe value)
#   k5          1 node  port 8023  eagle3 num_speculative_tokens=5
#
# Workload per arm: AA-style sweep 1k,10k x conc 1,10 (natural output).
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
GPTQ_CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
DRAFTER=/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3
SITE_0110=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/$TS}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-specdec-eagle3
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-specdec-eagle3/latest_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
# Graph-capture ordering fix -- required on every M3 serve.
export LLMC_M3_CAPTURE_SYNC=sync
echo "[controller] run_ts=$RUN_TS"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

for d in "$GPTQ_CKPT" "$SITE_0110" "$DRAFTER"; do
  test -d "$d" || fail "missing dir: $d"
done

# Drafter identity gate: the arch string is what makes this loadable by our own
# vLLM (registry.py maps LlamaForCausalLMEagle3 -> llama_eagle3).
drafter_arch=$("$PY" - <<PY
import json
print(json.load(open("$DRAFTER/config.json"))["architectures"][0])
PY
) || fail "cannot read drafter config"
[ "$drafter_arch" = "LlamaForCausalLMEagle3" ] \
  || fail "unexpected drafter architecture: $drafter_arch"
echo "[controller] drafter=$DRAFTER arch=$drafter_arch"
du -sh "$DRAFTER" > "$ROOT/drafter-size.txt" 2>&1 || true

# aiperf pin: the AA runner is coupled to the 0.8 record schema.
aiperf_version=$(/mnt/nfs/hoangduy/venvs/perf/bin/python -c \
  "import importlib.metadata as md; print(md.version('aiperf'))" 2>/dev/null)
case "$aiperf_version" in 0.8.*) ;; *) fail "perf venv aiperf must be 0.8.x, got: ${aiperf_version:-none}";; esac
echo "[controller] aiperf=$aiperf_version"

# All four declared humming patches on the 0.1.10 side-install.
for patch in ct_input_format grouped_expert_bounds tma_store_fence tma_store_commit; do
  ( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
      "pipeline/slurm/patch_humming_$patch.py" \
      --site "$SITE_0110" --check ) >>"$ROOT/patch-checks.log" 2>&1 \
    || fail "humming $patch patch missing on $SITE_0110 (see $ROOT/patch-checks.log)"
done
echo "[controller] humming 0.1.10 patches present"

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

declare -A ARM_PID=()

# $1 arm  $2 spec_k  $3 port
launch_arm() {
  local arm=$1 k=$2 port=$3
  # Assignment-prefix words from array expansion are not parsed as assignments
  # by bash -- they MUST go through `env` (20260725T074535Z postmortem).
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "RUN_TS=$RUN_TS" "SPEC_K=$k"
    "CKPT=$GPTQ_CKPT" "PORT=$port" "DRAFTER=$DRAFTER"
    "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
    "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=indexed"
    "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0110:$REPO"
    "AA_INPUTS=1k,10k" "AA_CONC=1,10"
  )
  env "${arm_env[@]}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=04:00:00 --kill-on-bad-exit=1 --partition=compute \
       --job-name="m3-spec-$arm" --export=ALL \
       bash "$REPO/pipeline/slurm/specdec_eagle3_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s spec_k=%s port=%s pid=%s launched=%s\n' \
    "$arm" "$k" "$port" "${ARM_PID[$arm]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"$ROOT/arm-provenance.txt"
  echo "[controller] launched $arm spec_k=$k port=$port pid=${ARM_PID[$arm]}"
}

launch_arm k0-control 0 8020
launch_arm k1         1 8021
launch_arm k3         3 8022
launch_arm k5         5 8023

echo "[controller] all arms launched; waiting"
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
