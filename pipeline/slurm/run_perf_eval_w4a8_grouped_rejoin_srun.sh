#!/usr/bin/env bash
# Rejoin the grouped_contiguous arm to an existing 3-arm perf window.
#
# Why this exists. In window 20260725T122256Z the grouped arm failed the suite
# preflight (rc=1 at 12:28:04Z): the kernel's last-expert row count was derived
# from a.size(0), which vLLM oversizes to (M*topk, K), corrupting the tail
# experts' outputs (see pipeline/slurm/patch_humming_grouped_expert_bounds.py
# for the root cause and pipeline/m3_humming_grouped_bounds_probe.py for the
# measurement). The kernel is now fixed and re-qualified
# (20260725T130023Z-groupedcontiguous: attestation valid, comprehension gate
# incl. the tool-call probe that caught the bug passed). CUTLASS and indexed
# from the same window are unaffected -- the patched line sits inside
# `if constexpr (kGemmType == GROUPED_CONTIGUOUS)` -- so relaunching only the
# grouped arm into the same ROOT/RUN_TS completes the 3-way table without
# burning two more node-runs.
#
# Honest caveat, recorded here and in the provenance file: the rejoin overlaps
# the original window rather than starting simultaneously with it. Arms run on
# dedicated exclusive nodes, so there is no cross-arm contention; the shared
# window exists to pin serve defaults and repo state, which this script pins
# explicitly (commit + versions + patch SHAs recorded below).
#
# Attempt 3 (2026-07-25). Attempt 2 completed but its numbers are confounded:
# the kernel corrupted whole output tiles intermittently (1,594 OSL-mismatch
# warnings vs ~10 on the other arms; 169/640 early-EOS requests at reasoning
# conc-64), so requests terminated early and every latency/throughput figure
# is optimistic. Root cause: humming never calls cp.async.bulk.commit_group,
# so every TMA store-group wait is a no-op and the producer overwrites the
# union-aliased epilogue smem while the C store is still reading it (see
# pipeline/slurm/patch_humming_tma_store_commit.py). Fixed by that patch plus
# the PTX-required fence.proxy.async (patch_humming_tma_store_fence.py);
# verified 0/96 bad launches and a fully clean bucket sweep incl. restored
# stream-K determinism (m3-arm3-commit-verify/20260725T162957Z; pre-fix
# baseline 10-11 bad per 48).
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
HUMMING_SITE=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python

ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-perf-w4a8-three-arm/20260725T122256Z}
export RUN_TS=${RUN_TS_OVERRIDE:-20260725T122256Z}
ARM=humming-w4afp8-grouped
PORT=8010

fail() { echo "[rejoin] ABORT: $1" >&2; echo "$1" >"$ROOT/grouped-rejoin-abort.txt"; exit 1; }

test -d "$ROOT" || fail "window root missing: $ROOT"
test -d "$CKPT" || fail "checkpoint missing: $CKPT"
test -d "$HUMMING_SITE" || fail "humming side-install missing: $HUMMING_SITE"

# All four declared patches must be present, fail-closed, before any GPU
# time: the pack-quantized admission patch, the grouped exact-total fix, and
# the two TMA-store fixes (fence + commit-group) attempt 3 exists to carry.
( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
    pipeline/slurm/patch_humming_ct_input_format.py \
    --site "$HUMMING_SITE" --check ) >"$ROOT/grouped-rejoin-ct-patch-check.log" 2>&1 \
  || fail "humming ct-input patch missing"
( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
    pipeline/slurm/patch_humming_grouped_expert_bounds.py \
    --site "$HUMMING_SITE" --check ) >"$ROOT/grouped-rejoin-bounds-patch-check.log" 2>&1 \
  || fail "humming grouped exact-total patch missing"
( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
    pipeline/slurm/patch_humming_tma_store_fence.py \
    --site "$HUMMING_SITE" --check ) >"$ROOT/grouped-rejoin-fence-patch-check.log" 2>&1 \
  || fail "humming tma-store fence patch missing"
( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
    pipeline/slurm/patch_humming_tma_store_commit.py \
    --site "$HUMMING_SITE" --check ) >"$ROOT/grouped-rejoin-commit-patch-check.log" 2>&1 \
  || fail "humming tma-store commit-group patch missing"

( PYTHONPATH="$HUMMING_SITE:$REPO" "$PY" - <<'PY'
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
) >"$ROOT/grouped-rejoin-versions.txt" 2>&1 || fail "version/import gate failed"
cat "$ROOT/grouped-rejoin-versions.txt"

# Preserve the failed attempts' raw evidence instead of letting the retry mix
# with it. Renames only -- nothing is deleted.
if [ -d "$ROOT/perf-$ARM" ] && [ ! -d "$ROOT/perf-$ARM.attempt1-unpatched-kernel" ]; then
  mv "$ROOT/perf-$ARM" "$ROOT/perf-$ARM.attempt1-unpatched-kernel"
fi
if [ -f "$ROOT/perf-$ARM.rc" ] && [ ! -f "$ROOT/perf-$ARM.rc.attempt1" ]; then
  cp "$ROOT/perf-$ARM.rc" "$ROOT/perf-$ARM.rc.attempt1"
fi
# Attempt 2 completed but with corrupted generation (missing TMA commit-group,
# see header); its numbers are confounded by early EOS and must not be mixed
# with attempt 3.
if [ -d "$ROOT/perf-$ARM" ] && [ ! -d "$ROOT/perf-$ARM.attempt2-tma-commit-missing" ]; then
  mv "$ROOT/perf-$ARM" "$ROOT/perf-$ARM.attempt2-tma-commit-missing"
fi
if [ -f "$ROOT/perf-$ARM.rc" ] && [ ! -f "$ROOT/perf-$ARM.rc.attempt2" ]; then
  cp "$ROOT/perf-$ARM.rc" "$ROOT/perf-$ARM.rc.attempt2"
fi
if [ -f "$ROOT/$ARM-rejoin-srun.log" ] && [ ! -f "$ROOT/$ARM-rejoin-srun.log.attempt2" ]; then
  mv "$ROOT/$ARM-rejoin-srun.log" "$ROOT/$ARM-rejoin-srun.log.attempt2"
fi

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/grouped-rejoin-commit.txt"

# Identical env to launch_arm in run_perf_eval_w4a8_three_arm_srun.sh; must go
# through `env` for the same reason documented there.
env "ROOT=$ROOT" "ARM=$ARM" "MODE=local" "PROFILE=minimax-m3-inhouse" \
    "CKPT=$CKPT" "PORT=$PORT" \
    "M3_ARM=$ARM" "MODEL_PATH=$CKPT" "ENDPOINT_PORT=$PORT" \
    "QUANT_RECIPE=gptq-w4afp8-humming-grouped" "M3_W4A8_BACKEND=humming" \
    "VLLM_HUMMING_MOE_GEMM_TYPE=grouped_contiguous" \
    "VLLM_HUMMING_USE_F16_ACCUM=0" \
    "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming" \
    "PYTHONPATH=$HUMMING_SITE:$REPO" \
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
     --time=12:00:00 --kill-on-bad-exit=1 --job-name="m3-perf-$ARM-rejoin" \
     --export=ALL \
     bash "$REPO/pipeline/slurm/perf_eval_arm.sh" \
     > "$ROOT/$ARM-rejoin-srun.log" 2>&1 &
pid=$!
printf 'arm=%s backend=humming gemm=grouped_contiguous port=%s pid=%s launched=%s rejoin=attempt3-tma-commit-fixed\n' \
  "$ARM" "$PORT" "$pid" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
echo "[rejoin] launched $ARM pid=$pid run_ts=$RUN_TS root=$ROOT"

wait "$pid"; rc=$?
printf 'rc=%s pid=%s arm=%s finished=%s attempt=3-tma-commit-fixed\n' \
  "$rc" "$pid" "$ARM" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$ROOT/perf-$ARM.rc"
echo "[rejoin] perf $ARM rc=$rc (pid=$pid)"
echo "REJOIN_RC=$rc"
exit "$rc"
