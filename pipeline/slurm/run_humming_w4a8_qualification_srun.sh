#!/usr/bin/env bash
# Controller (tmux) for the Humming W4A8 correctness qualification.
#
# Runs one arm on one exclusive 8xH100 node. GEMM_TYPE selects the Humming MoE
# scheduling strategy under test:
#   indexed             investigation arm 2 -- qualified 2026-07-25 (r3)
#   grouped_contiguous  investigation arm 3 -- warp-specialized + TMA kernel
#
# This establishes correctness/attestation only. It says nothing about speed;
# the paired serving benchmark is a separate run.
#
# Usage: GEMM_TYPE=grouped_contiguous bash pipeline/slurm/run_humming_w4a8_qualification_srun.sh
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
HUMMING_SITE=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
GEMM_TYPE="${GEMM_TYPE:-indexed}"

case "$GEMM_TYPE" in
  indexed|grouped|grouped_contiguous) ;;
  *) echo "ABORT: unsupported GEMM_TYPE '$GEMM_TYPE'" >&2; exit 2 ;;
esac

TS=$(date -u +%Y%m%dT%H%M%SZ)
SLUG=$(printf '%s' "$GEMM_TYPE" | tr -d '_')
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-humming-w4a8-qualification/$TS-$SLUG}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-humming-w4a8-qualification
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-humming-w4a8-qualification/latest_root
echo "[controller] root=$ROOT gemm_type=$GEMM_TYPE"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
test -d "$CKPT" || fail "checkpoint missing: $CKPT"
test -d "$HUMMING_SITE" || fail "humming side-install missing: $HUMMING_SITE"

# Cheap on the login node; expensive to discover on an allocated one.
( cd "$REPO" && PYTHONPATH="$REPO" \
  /mnt/nfs/hoangduy/venvs/quant/bin/python \
    pipeline/slurm/patch_humming_ct_input_format.py \
    --site "$HUMMING_SITE" --check ) >"$ROOT/humming-patch-check.log" 2>&1 \
  || fail "humming ct-input patch missing (see $ROOT/humming-patch-check.log)"

printf '%s\n' "$GEMM_TYPE" >"$ROOT/gemm-type.txt"
printf '%s\n' "$CKPT" >"$ROOT/checkpoint.txt"
( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

cd "$REPO"
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
     --time=03:00:00 --kill-on-bad-exit=1 \
     --job-name="m3-humming-$SLUG" --export=ALL \
     env ROOT="$ROOT" GEMM_TYPE="$GEMM_TYPE" \
     bash "$REPO/pipeline/slurm/humming_w4a8_qualification_node.sh" \
     >"$ROOT/srun.out" 2>"$ROOT/srun.err"
rc=$?
printf '%s\n' "$rc" >"$ROOT/srun.rc"
echo "[controller] srun rc=$rc"
echo "CONTROLLER_RC=$rc"
exit "$rc"
