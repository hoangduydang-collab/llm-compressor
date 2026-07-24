#!/usr/bin/env bash
# Shared-experts aux-stream fix matrix — runs ON one allocated 8xH100 node.
#
# Follow-up to the 20260710 RCA (artifacts/m3-cudagraph-shared-stream/.../
# RCA_REPORT.md, classification "narrowed"): tests the refined root-cause
# hypothesis that the capture IMA is the FlashInfer trtllm fused AR+RMSNorm
# (fail-open v1 guard -> live inside captured graphs, PDL, first joins the
# ladder at the ~0.5MB/42-token gate = capture ~44/51) conflicting with the
# shared-experts aux stream. Four conditions x 3 trials, same fixed contract
# as the RCA (cyankiwi ckpt, TP8, EP, 8192/0.9, graphs+breakable, async CUDA):
#
#   sanity-streamOFF-legacy   stream off, fused AR live      (prod; expect 3/3)
#   repro-streamON-legacy     stream on,  fused AR live      (expect >=1 IMA)
#   fix-streamON-ncclgraphs   stream on,  NCCL under capture (fix candidate)
#   probe-streamON-pdloff     stream on,  fused AR, PDL off under capture
#
# Requires the v2 fused-AR patch (LLMC_M3_FI_AR_MODE) in the quant venv.
# Usage: shared_stream_fix_matrix_node.sh <result root> <session id>
set -uo pipefail

ROOT=${1:?usage: shared_stream_fix_matrix_node.sh <result root> <session id>}
SESSION=${2:?missing session id}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
VENV_FILE=/mnt/nfs/hoangduy/venvs/quant/lib/python3.12/site-packages/vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py
CASES=async_baseline_1,async_baseline_2,async_baseline_3

mkdir -p "$ROOT"
note() { echo "[fix-matrix $(date -u +%H:%M:%S)] $1" | tee -a "$ROOT/matrix.log"; }

# The underlying matrix harness calls bare `python` (summary step); make the
# quant venv's interpreter visible for it and for everything downstream.
export PATH="/mnt/nfs/hoangduy/venvs/quant/bin:$PATH"

note "host=$(hostname) session=$SESSION"

# Fail-closed preflight: v2 patch must be installed; do NOT auto-apply here
# (serve launcher would, but a mismatch means the venv is not what we tested).
if ! grep -q "llmc M3 cudagraph fused-AR mode switch v2" "$VENV_FILE"; then
  note "FATAL: v2 fused-AR patch missing in venv; run patch_vllm_m3_serve.py"
  echo "PREFLIGHT_RC=1"; exit 1
fi
if grep -q "get_current_vllm_config()" "$VENV_FILE"; then
  note "FATAL: v1 fail-open guard still present in venv"
  echo "PREFLIGHT_RC=1"; exit 1
fi
test -f /mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4/config.json || {
  note "FATAL: cyankiwi checkpoint missing"; echo "PREFLIGHT_RC=1"; exit 1
}
note "preflight ok (v2 patch present, v1 gone, ckpt present)"
echo "PREFLIGHT_RC=0"

run_condition() {  # $1 cond-name  $2 stream-disable  $3 fi-ar-mode
  local cond=$1 sd=$2 mode=$3
  local run_id="$SESSION-$cond"
  local run_dir="$ROOT/$run_id"
  mkdir -p "$run_dir"
  cat >"$run_dir/condition.env" <<EOF
condition=$cond
VLLM_DISABLE_SHARED_EXPERTS_STREAM=$sd
LLMC_M3_FI_AR_MODE=$mode
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256
CUDA_LAUNCH_BLOCKING=unset
MATRIX_CASES=$CASES
EOF
  note "condition $cond start (stream_disable=$sd fi_ar_mode=$mode)"
  env -u CUDA_LAUNCH_BLOCKING -u TORCH_USE_CUDA_DSA \
      VLLM_DISABLE_SHARED_EXPERTS_STREAM="$sd" \
      LLMC_M3_FI_AR_MODE="$mode" \
      VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256 \
      RESULTS_ROOT="$ROOT" RUN_ID="$run_id" MATRIX_CASES="$CASES" \
      bash "$REPO/pipeline/slurm/test_m3_http_cudagraph_matrix.sh" \
      >"$run_dir/matrix-driver.log" 2>&1
  local rc=$?
  note "condition $cond done rc=$rc"
  echo "CONDITION_RC $cond $rc"
}

# Sanity canary first: if the known-good production config fails on this
# node/stack, stop before burning the other conditions.
PYBIN=/mnt/nfs/hoangduy/venvs/quant/bin/python

run_condition sanity-streamOFF-legacy 1 legacy
if "$PYBIN" - "$ROOT/$SESSION-sanity-streamOFF-legacy/summary.json" <<'PY'
import json, sys
s = json.loads(open(sys.argv[1], encoding="utf-8").read())
bad = [t["case"] for t in s["trials"] if not (t.get("server_ready") and t.get("chat_ok"))]
sys.exit(1 if bad else 0)
PY
then
  note "sanity condition clean; proceeding"
else
  note "FATAL: sanity (streamOFF-legacy) not clean on this node — stack/node drift, aborting"
  echo "MATRIX_ABORTED=sanity_failed"; exit 1
fi

run_condition repro-streamON-legacy 0 legacy
run_condition fix-streamON-ncclgraphs 0 nccl_graphs
run_condition probe-streamON-pdloff 0 pdl_off

# Comparison artifact (machine-readable, same shape as the RCA one).
"$PYBIN" - "$ROOT" "$SESSION" <<'PY' | tee -a "$ROOT/matrix.log"
import json, re, sys
from pathlib import Path

root, session = Path(sys.argv[1]), sys.argv[2]
conds = [
    "sanity-streamOFF-legacy",
    "repro-streamON-legacy",
    "fix-streamON-ncclgraphs",
    "probe-streamON-pdloff",
]
rows = []
for cond in conds:
    run_dir = root / f"{session}-{cond}"
    sfile = run_dir / "summary.json"
    if not sfile.exists():
        rows.append({"condition": cond, "error": "no summary.json"})
        continue
    summary = json.loads(sfile.read_text(encoding="utf-8"))
    for t in summary["trials"]:
        log = run_dir / t["case"] / "serve.log"
        progress = None
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
            hits = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", text) if int(b) == 51]
            progress = list(hits[-1]) if hits else None
        rows.append({
            "condition": cond, "case": t["case"], "verdict": t.get("verdict"),
            "server_ready": t.get("server_ready"), "chat_ok": t.get("chat_ok"),
            "ima": t.get("ima"), "last_capture_progress": progress,
        })

def clean(cond):
    rs = [r for r in rows if r.get("condition") == cond and "case" in r]
    return len(rs) == 3 and all(r["server_ready"] and r["chat_ok"] for r in rs)

def any_ima(cond):
    return any(r.get("ima") for r in rows if r.get("condition") == cond)

verdict = {
    "sanity_clean": clean("sanity-streamOFF-legacy"),
    "repro_ima": any_ima("repro-streamON-legacy"),
    "fix_clean": clean("fix-streamON-ncclgraphs"),
    "pdl_off_clean": clean("probe-streamON-pdloff"),
}
if verdict["sanity_clean"] and verdict["repro_ima"] and verdict["fix_clean"]:
    verdict["classification"] = "CONFIRMED: fused-AR x aux-stream capture conflict"
elif verdict["sanity_clean"] and not verdict["repro_ima"]:
    verdict["classification"] = "INCONCLUSIVE: flaky repro did not fire (bounded rerun allowed)"
else:
    verdict["classification"] = "NOT CONFIRMED: see rows"

out = {"session": session, "root": str(root),
       "hypothesis": "flashinfer fused AR+RMSNorm x shared-experts aux stream x capture",
       "verdict": verdict, "rows": rows}
p = root / f"{session}-fix-comparison.json"
p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(verdict, indent=2))
print(f"COMPARISON={p}")
PY

note "matrix complete"
echo "MATRIX_RC=0"
