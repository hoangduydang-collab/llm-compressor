#!/usr/bin/env bash
# Controller (tmux) for the EAGLE3 drafter-compatibility test -- phase E.
#
# Is the Inferact EAGLE3 drafter as compatible with OUR 4-bit W4AFP8 target as with
# the vendor's 8-bit MXFP8 target? 2 arms x 1 node:
#
#   mxfp8-k0  port 8052   vendor MiniMax-M3-MXFP8, native vLLM path (control)
#   mxfp8-k3  port 8053   same + eagle3 k=3
#
# The W4AFP8 side is NOT re-measured: phase D's 8k-low / 8k-high cells at conc 1
# and 10 already ran the identical prompts, seed, sampling and serve config, and
# MXFP8 serves at the same max_model_len=131072 (the two-axis FALLBACK_MML=40960
# was declared but never fired). Phase D's serve banner was diffed against this
# one -- the only non-default args that differ are the checkpoint and
# `quantization: humming`. Acceptance is a model-intrinsic quantity and phase D's
# controls held to 0.5% across cells, so the cross-window comparison is sound.
# Each format still carries its OWN k=0 control, so no speedup ratio crosses
# formats or windows.
#
# A separate file from run_specdec_phaseD_srun.sh on purpose: bash reads a script
# incrementally, so editing a launcher whose controller is still running can
# corrupt the running shell.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
MXFP8_CKPT=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3-MXFP8
DRAFTER=/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3
SB_DIR="$REPO/artifacts/aiperf-datasets/speedbench"
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python
EXCLUDE=${EXCLUDE:-gpu-h97,gpu-h98,gpu-h101}
# Match phase D exactly; MXFP8 is proven to serve at this length.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
# Phase D window whose W4AFP8 8k cells are the comparison arm (recorded for provenance).
W4A8_REF=${W4A8_REF:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T073533Z-phaseD}

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/$TS-format}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-specdec-eagle3
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-specdec-eagle3/latest_format_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
export LLMC_M3_CAPTURE_SYNC=sync
echo "[controller] run_ts=$RUN_TS exclude=$EXCLUDE mml=$MAX_MODEL_LEN"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

for d in "$MXFP8_CKPT" "$DRAFTER" "$SB_DIR" "$W4A8_REF"; do
  test -d "$d" || fail "missing dir: $d"
done
echo "$W4A8_REF" > "$ROOT/w4a8-reference-window.txt"

# --- MXFP8 checkpoint identity gate -------------------------------------------
mx_method=$("$PY" - <<PY
import json
c = json.load(open("$MXFP8_CKPT/config.json"))
print((c.get("quantization_config") or {}).get("quant_method", "none"))
PY
) || fail "cannot read MXFP8 config"
[ "$mx_method" = "mxfp8" ] || fail "MXFP8 ckpt quant_method is '$mx_method', expected mxfp8"

# --- SPEED-Bench staging gate (fail closed, same as phase D) -------------------
# Only the 8k cells are used here, but assert the exact same bytes phase D ran so
# the two windows are prompt-identical.
test -s "$SB_DIR/manifest.json" || fail "SPEED-Bench not staged: run pipeline/stage_speedbench.py"
CELLS="8k-low 8k-high"
for cell in $CELLS; do
  f="$SB_DIR/$cell.jsonl"
  test -s "$f" || fail "missing staged cell: $f"
  n=$(wc -l < "$f")
  [ "$n" -ge 100 ] || fail "$cell has only $n entries (need >=100)"
  if grep -q "FULL BENCHMARK DATA SHOULD BE FETCHED" "$f"; then
    fail "$cell still contains masked placeholder rows"
  fi
done
"$PY" - <<PY >"$ROOT/speedbench-manifest.txt" || fail "SPEED-Bench hash gate failed"
import hashlib, json
m = json.load(open("$SB_DIR/manifest.json"))
for name in ["8k-low.jsonl", "8k-high.jsonl"]:
    v = m["files"][name]
    h = hashlib.sha256(open("$SB_DIR/" + name, "rb").read()).hexdigest()
    assert h == v["sha256"], "sha256 drift for " + name
    print(f"{name:16s} n={v['entries']:4d} tok_mean={v['tokens_mean']:9.1f} sha={h[:12]}")
PY
echo "[controller] SPEED-Bench 8k cells hash-verified (prompt-identical to phase D)"

drafter_arch=$("$PY" - <<PY
import json
print(json.load(open("$DRAFTER/config.json"))["architectures"][0])
PY
) || fail "cannot read drafter config"
[ "$drafter_arch" = "LlamaForCausalLMEagle3" ] || fail "unexpected drafter arch: $drafter_arch"

aiperf_version=$(/mnt/nfs/hoangduy/venvs/perf/bin/python -c \
  "import importlib.metadata as md; print(md.version('aiperf'))" 2>/dev/null)
case "$aiperf_version" in 0.8.*) ;; *) fail "perf venv aiperf must be 0.8.x, got: ${aiperf_version:-none}";; esac
echo "[controller] aiperf=$aiperf_version drafter=$drafter_arch mxfp8_method=$mx_method"

# No Humming patch gate here: both arms run vLLM's native MXFP8 path, so there is
# no side-install to verify. The arm script asserts the mxfp8 path engaged instead.

# Assert the phase D reference actually holds the cells we intend to compare against.
for cell in 8k-low 8k-high; do
  for conc in 1 10; do
    for arm in phaseD-k0 phaseD-k3; do
      f="$W4A8_REF/arm-$arm/speedbench/$cell/conc_$conc/profile_export_aiperf.json"
      test -s "$f" || echo "[controller] NOTE: W4AFP8 reference cell not yet present: $arm/$cell/conc_$conc"
    done
  done
done

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

declare -A ARM_PID=()

# $1 arm  $2 spec_k  $3 port  $4 format  $5 ckpt  $6.. extra env
launch_arm() {
  local arm=$1 k=$2 port=$3 fmt=$4 ckpt=$5; shift 5
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "RUN_TS=$RUN_TS" "SPEC_K=$k" "FORMAT=$fmt"
    "CKPT=$ckpt" "PORT=$port" "DRAFTER=$DRAFTER" "SB_DIR=$SB_DIR"
    "MAX_MODEL_LEN=$MAX_MODEL_LEN"
    "$@"
  )
  env "${arm_env[@]}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=05:00:00 --kill-on-bad-exit=1 --partition=compute \
       --exclude="$EXCLUDE" --job-name="m3-fmt-$arm" --export=ALL \
       bash "$REPO/pipeline/slurm/specdec_format_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s format=%s spec_k=%s port=%s ckpt=%s pid=%s launched=%s\n' \
    "$arm" "$fmt" "$k" "$port" "$ckpt" "${ARM_PID[$arm]}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
  echo "[controller] launched $arm format=$fmt k=$k port=$port pid=${ARM_PID[$arm]}"
}

# MXFP8 uses vLLM's native path: no Humming side-install on PYTHONPATH, no
# --quantization flag, no F16-accum / GEMM-type knobs.
MX_ENV=( "BACKEND=native" "PYTHONPATH=$REPO" )

launch_arm mxfp8-k0 0 8052 mxfp8 "$MXFP8_CKPT" "${MX_ENV[@]}"
launch_arm mxfp8-k3 3 8053 mxfp8 "$MXFP8_CKPT" "${MX_ENV[@]}"

echo "[controller] 2 arms launched; waiting"
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
