#!/usr/bin/env bash
# Controller (tmux) for EAGLE3 spec-dec phase D -- M3_SPECDEC_EAGLE3_PLAN.md.
#
# 2 GPU nodes: control (k=0) vs k=3, each sweeping SPEED-Bench's fixed-ISL buckets
# (1k / 8k / 32k) crossed with entropy tier (low = code/sorting, high = creative
# writing). Answers "does the conc-1 speedup change with longer NATURAL prompts?",
# which wave 1 could only answer on synthetic random tokens and phase A only at
# ~227 tokens.
#
#   phaseD-k0  port 8040
#   phaseD-k3  port 8041
#
# A separate file from run_specdec_wave2_srun.sh on purpose: bash reads a script
# incrementally, so editing a launcher whose controller is still running can
# corrupt the running shell.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
GPTQ_CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
DRAFTER=/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3
SB_DIR="$REPO/artifacts/aiperf-datasets/speedbench"
SITE_0110=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python
EXCLUDE=${EXCLUDE:-gpu-h97,gpu-h98,gpu-h101}

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/$TS-phaseD}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-specdec-eagle3
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-specdec-eagle3/latest_phaseD_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
export LLMC_M3_CAPTURE_SYNC=sync
echo "[controller] run_ts=$RUN_TS exclude=$EXCLUDE"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

for d in "$GPTQ_CKPT" "$SITE_0110" "$DRAFTER" "$SB_DIR"; do
  test -d "$d" || fail "missing dir: $d"
done

# --- SPEED-Bench staging gate (fail closed) -----------------------------------
# ~45% of the public release is masked with a placeholder sentinel and aiperf's
# loader does not filter it. A masked prompt is short and repetitive, so it would
# both break the ISL bucket and inflate acceptance. Assert zero survivors.
test -s "$SB_DIR/manifest.json" || fail "SPEED-Bench not staged: run pipeline/stage_speedbench.py"
CELLS="1k-low 1k-high 8k-low 8k-high 32k-low 32k-high"
for cell in $CELLS; do
  f="$SB_DIR/$cell.jsonl"
  test -s "$f" || fail "missing staged cell: $f"
  n=$(wc -l < "$f")
  [ "$n" -ge 100 ] || fail "$cell has only $n entries (need >=100)"
  if grep -q "FULL BENCHMARK DATA SHOULD BE FETCHED" "$f"; then
    fail "$cell still contains masked placeholder rows"
  fi
done
"$PY" - <<PY >"$ROOT/speedbench-manifest.txt" || fail "cannot read SPEED-Bench manifest"
import hashlib, json
m = json.load(open("$SB_DIR/manifest.json"))
print("dataset:", m["dataset"], "dropped_tier:", m["dropped_tier"])
for name, v in sorted(m["files"].items()):
    h = hashlib.sha256(open("$SB_DIR/" + name, "rb").read()).hexdigest()
    assert h == v["sha256"], "sha256 drift for " + name
    print(f"{name:22s} n={v['entries']:4d} tok_mean={v['tokens_mean']:9.1f} "
          f"tok_med={v['tokens_median']:8} masked_dropped={v['masked_dropped']:4d} sha={h[:12]}")
PY
echo "[controller] SPEED-Bench staged + hashes verified ($(echo $CELLS | wc -w) cells)"

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

# $1 spec_k  $2 port
launch_arm() {
  local k=$1 port=$2 arm="phaseD-k$1"
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "RUN_TS=$RUN_TS" "SPEC_K=$k"
    "CKPT=$GPTQ_CKPT" "PORT=$port" "DRAFTER=$DRAFTER" "SB_DIR=$SB_DIR"
    "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
    "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=indexed"
    "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0110:$REPO"
  )
  env "${arm_env[@]}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=06:00:00 --kill-on-bad-exit=1 --partition=compute \
       --exclude="$EXCLUDE" --job-name="m3-pD-$arm" --export=ALL \
       bash "$REPO/pipeline/slurm/specdec_phaseD_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s spec_k=%s port=%s pid=%s launched=%s\n' \
    "$arm" "$k" "$port" "${ARM_PID[$arm]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"$ROOT/arm-provenance.txt"
  echo "[controller] launched $arm port=$port pid=${ARM_PID[$arm]}"
}

launch_arm 0 8040
launch_arm 3 8041

echo "[controller] both arms launched; waiting"
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
