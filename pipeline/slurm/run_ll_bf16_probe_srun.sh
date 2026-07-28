#!/usr/bin/env bash
# Probe vLLM 0.26.0's cuteDSL ll_bf16 router GEMM at MiniMax-M3's router shape.
#
# Why: the 0.26.0 k=0 baseline (plain M3 W4A8 + Humming, NO spec-dec) died with a CUDA
# illegal memory access during the conc-10 cell after a clean conc-1 cell. The one code
# path that differs between those two cells is the router-gate GEMM tier introduced in
# 0.26.0: M3's router gate is (K=6144, N=128), which appears in NONE of ll_bf16's tables
# (warmup shapes, dotprod tuning, splitk tuning -- all DeepSeek/Inkling). Dispatch sends
# M<=4 to dotprod (conc 1) and M>=5 to split-K with untuned defaults (conc 10).
#
# DeepSeek also reaches splitk(6,4) at M>=7, so the config is not broken in general.
# The variable M3 does not share with any tested shape is N=128 -- half the smallest N
# upstream tuned or warmed (256/264/384). This probe A/Bs N=128 against N=256 and N=264
# at identical M and identical splitk config, which localizes the defect if it is real.
#
# Diagnostic only: 1 GPU, no model load, no checkpoint touched.
set -uo pipefail

REPO=${REPO:-/mnt/nfs/hoangduy/projects/llm-compressor}
PY=${PY:-/mnt/nfs/hoangduy/venvs/serve-026/bin/python}
PROBE=${PROBE:-$REPO/pipeline/diag/ll_bf16_probe.py}
RESULTS=${RESULTS:-/mnt/nfs/hoangduy/results/m3-ll-bf16-probe}
SANITIZER=${SANITIZER:-/usr/local/cuda/bin/compute-sanitizer}

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$RESULTS/$RUN_TS
mkdir -p "$OUT"

test -s "$PROBE" || { echo "FATAL: probe not found at $PROBE" >&2; exit 2; }
test -x "$PY"    || { echo "FATAL: python not found at $PY" >&2; exit 2; }

{
  echo "run_ts       : $RUN_TS"
  echo "probe        : $PROBE"
  echo "python       : $PY"
  echo "vllm         : $("$PY" -c 'import vllm; print(vllm.__version__)' 2>&1)"
  echo "commit       : $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null)"
} | tee "$OUT/provenance.txt"

cat > "$OUT/step.sh" <<STEP
set -uo pipefail
echo "=== node \$(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv | head -3

echo
echo "=== PASS 1: plain (correctness + raise detection, all shapes) ==="
"$PY" "$PROBE"
rc_plain=\$?
echo "PASS 1 rc=\$rc_plain"

echo
echo "=== PASS 2: compute-sanitizer memcheck (M3 shape, splitk M values only) ==="
# Sanitizer is slow, so restrict to the suspect path. Best-effort: the sanitizer here is
# a CUDA 12.4 build against a cu130 torch, so a launch failure is not itself evidence.
if [ -x "$SANITIZER" ]; then
  PROBE_M=5,10 "$SANITIZER" --tool memcheck --target-processes application-only \\
    "$PY" "$PROBE" 2>&1 | tail -60
  echo "PASS 2 rc=\$?"
else
  echo "SKIPPED: no compute-sanitizer at $SANITIZER"
fi

echo
echo "OVERALL plain rc=\$rc_plain"
exit \$rc_plain
STEP

echo "launching srun (1 GPU, 20 min cap) -> $OUT/probe.log"
srun --exclusive=user --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 \
     --time=00:20:00 --kill-on-bad-exit=1 --partition=compute \
     --job-name=ll-bf16-probe --export=ALL \
     bash "$OUT/step.sh" 2>&1 | tee "$OUT/probe.log"
rc=$?

echo
echo "probe rc=$rc  log: $OUT/probe.log"
printf 'rc=%s finished=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/done.txt"
echo "$OUT" > "$RESULTS/latest.txt"
exit $rc
