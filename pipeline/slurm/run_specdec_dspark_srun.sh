#!/usr/bin/env bash
# Controller for the DSpark axis-1 (draft depth k) window.
#
# Runs from a persistent detached tmux session and holds ONE top-level `srun`
# allocation (this cluster does not accept sbatch). Every gate below runs BEFORE the
# allocation is taken, on the login node, so a misconfiguration costs seconds rather
# than a node-hour.
#
# NEW ENVIRONMENT: this is the first window on venvs/serve-026 (vLLM 0.26.0 with
# humming 0.1.10 merged in-venv). PYTHONPATH therefore carries NO humming side dir --
# see docs/m3-serve-venv-026.md. Cross-venv comparisons against the 0.23.1 EAGLE3
# numbers are not within-window and must be labelled as such.
set -uo pipefail

REPO=/mnt/nfs/hoangduy/projects/llm-compressor
PY=/mnt/nfs/hoangduy/venvs/serve-026/bin/python
SITE=/mnt/nfs/hoangduy/venvs/serve-026/lib/python3.12/site-packages
RESULTS=/mnt/nfs/hoangduy/results/m3-specdec-dspark
DRAFTER=/mnt/nfs/hoangduy/hf_assets/nvidia/MiniMax-M3-DSpark
GPTQ_CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
SB_DIR="$REPO/artifacts/aiperf-datasets/speedbench"
NODE=${NODE:?set NODE, e.g. NODE=gpu-h123}
PORT_BASE=${PORT_BASE:-8120}
ONLY=${ONLY:-}
ARM=dspark

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT:-$RESULTS/$RUN_TS-k-sweep}
mkdir -p "$ROOT"
fail() { echo "[controller] GATE FAILED: $1" | tee -a "$ROOT/gates.log"; exit 1; }
say()  { echo "[controller] $1" | tee -a "$ROOT/gates.log"; }

say "root=$ROOT node=$NODE port_base=$PORT_BASE venv=serve-026"

# ============================ SERVE LIST ======================================
# label;k;cells
#
# Order carries the statistics:
#   * D-k8-a runs FIRST and is the SMOKE serve -- k=8 is NVIDIA's published
#     reference config, so a layout failure shows up in ~20 min and stops the arm.
#   * k=8 and k=6 each get two replicates spread across the window.
#   * k=0 controls bracket both ends, so the pair also measures window drift.
#   * k=10 and k=12 probe ABOVE the trained block_size of 8. num_speculative_tokens
#     is a free serve-time int (vLLM sizes the query block from it, it is not clamped
#     to the checkpoint), but the drafter was trained at 8, so these two points test
#     whether acceptance saturates or degrades past training.
SERVES=$(cat <<'EOF'
D-k8-a;8;8k-low 8k-high
D-k0-a;0;8k-low 8k-high
D-k6-a;6;8k-low 8k-high
D-k4-a;4;8k-low 8k-high
D-k10-a;10;8k-low 8k-high
D-k8-b;8;8k-low 8k-high
D-k12-a;12;8k-low 8k-high
D-k6-b;6;8k-low 8k-high
D-k0-b;0;8k-low 8k-high
EOF
)
SMOKE_LABEL=D-k8-a
n_serves=$(printf '%s\n' "$SERVES" | grep -c ';')
[ "$n_serves" = 9 ] || fail "serve list parsed $n_serves entries, expected 9"
printf '%s\n' "$SERVES" > "$ROOT/serve-list.txt"
say "serve list: $n_serves serves, smoke=$SMOKE_LABEL"

# ============================ PRE-FLIGHT GATES ================================
for d in "$GPTQ_CKPT" "$DRAFTER" "$SB_DIR"; do
  [ -d "$d" ] || fail "missing directory: $d"
done

# --- staged prompts: same bytes as every EAGLE3 phase, so the workload is shared ---
say "staged prompt sha gate"
for cell in 8k-low 8k-high; do
  [ -s "$SB_DIR/$cell.jsonl" ] || fail "missing staged prompts $SB_DIR/$cell.jsonl"
done
low_sha=$(sha256sum "$SB_DIR/8k-low.jsonl"  | cut -c1-12)
high_sha=$(sha256sum "$SB_DIR/8k-high.jsonl" | cut -c1-12)
[ "$low_sha"  = bfcf60739f43 ] || fail "8k-low prompts changed ($low_sha != bfcf60739f43)"
[ "$high_sha" = 325e2a9dc34f ] || fail "8k-high prompts changed ($high_sha != 325e2a9dc34f)"
say "prompts match the EAGLE3 windows (8k-low $low_sha, 8k-high $high_sha)"

# --- drafter identity + THE aux-layer resolution gate -------------------------
# The aux-layer check is the one that matters: if resolution fails, vLLM silently
# falls back to the target's default 3-layer EAGLE3 set and only acceptance shows it.
# We assert the resolution here, offline, using vLLM's own resolver.
say "DSpark drafter identity + aux-layer resolution gate"
"$PY" - "$DRAFTER" <<'PY' >"$ROOT/drafter-identity.txt" 2>&1 || fail "drafter identity gate failed"
import json, sys, hashlib
from types import SimpleNamespace
from transformers import AutoConfig
from safetensors import safe_open
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import get_eagle3_aux_layers_from_config

d = sys.argv[1]
raw = json.load(open(f"{d}/config.json"))
print("config sha256 :", hashlib.sha256(open(f"{d}/config.json","rb").read()).hexdigest()[:16])

assert raw["architectures"] == ["Qwen3DSparkModel"], raw["architectures"]
assert raw["model_type"] == "qwen3", raw["model_type"]
assert raw["block_size"] == 8, raw["block_size"]
assert raw["vocab_size"] == 200064, raw["vocab_size"]
assert raw["num_hidden_layers"] == 6, raw["num_hidden_layers"]
assert raw["hidden_size"] == 6144, raw["hidden_size"]
assert "quantization_config" not in raw, "drafter unexpectedly quantized"
df = raw["dflash_config"]
assert df["target_layer_ids"] == [1, 12, 23, 35, 46, 57], df["target_layer_ids"]
assert df["mask_token_id"] == 200063, df["mask_token_id"]
assert df["markov_rank"] == 256 and df["markov_head_type"] == "vanilla"
assert df["use_swa"] is True and df["swa_window_size"] == 1024
assert df["causal"] is False, df["causal"]
assert df["projector_type"] == "dspark", df["projector_type"]
print("arch          :", raw["architectures"][0])
print("block_size    :", raw["block_size"], "(trained draft block; k is free at serve time)")
print("aux target ids:", df["target_layer_ids"])
print("swa / causal  :", df["swa_window_size"], "/", df["causal"])

# weights: 6144 x (6*6144) projector proves six aux streams; bf16 lm_head present
f = safe_open(f"{d}/model.safetensors", "pt")
keys = set(f.keys())
assert "fc.weight" in keys and "lm_head.weight" in keys, sorted(keys)[:8]
assert tuple(f.get_slice("fc.weight").get_shape()) == (6144, 36864), \
    f.get_slice("fc.weight").get_shape()
assert tuple(f.get_slice("lm_head.weight").get_shape()) == (200064, 6144)
assert "embed_tokens.weight" not in keys, "drafter should share the target embedding"
for k in ("markov_head.markov_w1.weight", "markov_head.markov_w2.weight"):
    assert k in keys, k
print("tensors       :", len(keys))
print("fc.weight     : (6144, 36864) == 6144 x 6*6144 -> six aux streams confirmed")
print("embed_tokens  : absent (shares the target's), lm_head present (bf16)")

# THE gate: vLLM's own resolver, on this exact config.
hf = AutoConfig.from_pretrained(d)
aux = get_eagle3_aux_layers_from_config(SimpleNamespace(
    draft_model_config=SimpleNamespace(hf_config=hf)))
assert aux == (2, 13, 24, 36, 47, 58), f"aux resolution changed: {aux}"
assert getattr(hf, "eagle_config", None) is None, \
    "an eagle_config dict would overwrite the dflash_config aux ids in the legacy runner"
print("aux resolved  :", aux, "(dflash +1 semantics, FROM CONFIG not model default)")

# Draft layout depends on these being ABSENT (see the runtime gate for the two
# upstream getattr spellings). Present-but-true would flip us to the 1+N fill-in
# block and silently collapse acceptance.
assert getattr(hf, "dspark_bonus_anchor", False) is False, "bonus-anchor layout requested"
assert getattr(hf, "sample_from_anchor", True) is True, "anchor-as-first disabled"
print("draft layout  : anchor-as-first (k query slots), neither override present")
PY
cat "$ROOT/drafter-identity.txt"
say "drafter verified"

# --- serve-026 patch stack: apply then re-check, so a fresh venv is handled ------
say "vLLM + humming patch gates on serve-026"
for p in patch_vllm_m3_serve patch_vllm_humming_lmhead patch_vllm_eagle3_lmhead_pad; do
  ( cd "$REPO" && "$PY" "pipeline/slurm/$p.py" ) >>"$ROOT/patch-checks.log" 2>&1 \
    || fail "$p apply failed"
  ( cd "$REPO" && "$PY" "pipeline/slurm/$p.py" --check ) >>"$ROOT/patch-checks.log" 2>&1 \
    || fail "$p still not applied after apply"
done
for p in ct_input_format grouped_expert_bounds tma_store_fence tma_store_commit; do
  ( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
      "pipeline/slurm/patch_humming_$p.py" --site "$SITE" --check ) \
      >>"$ROOT/patch-checks.log" 2>&1 \
    || fail "humming $p patch missing on $SITE"
done
say "all 7 patch edits verified"

# --- runtime capability gates -------------------------------------------------
say "runtime gates (dspark support, kernel registry, humming merged in-venv)"
"$PY" - <<'PY' >"$ROOT/runtime-gates.txt" 2>&1 || fail "runtime gates failed"
import importlib.metadata as md
from typing import get_args
import vllm, torch, humming

print("vllm           :", vllm.__version__)
print("torch          :", torch.__version__, torch.version.cuda)
hk = md.version("humming_kernels")
print("humming_kernels:", hk, "at", humming.__file__)
assert hk == "0.1.10", hk
assert "serve-026" in humming.__file__ and "site-packages" in humming.__file__, \
    "humming must be the in-venv copy, not a PYTHONPATH side-install"

from vllm.config.speculative import SpeculativeMethod
def flat(t):
    for a in get_args(t):
        if isinstance(a, str):
            yield a
        else:
            yield from flat(a)
methods = set(flat(SpeculativeMethod))
assert "dspark" in methods, sorted(methods)
print("dspark method  : present")

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkModel
import inspect
src = inspect.getsource(DSparkSpeculator.__init__)

# DRAFT LAYOUT. Our checkpoint declares neither knob, so a getattr default decides
# whether the query block is N slots (anchor-as-first) or 1+N (fill-in, anchor is a
# bonus token). Getting this wrong does not raise -- it collapses acceptance. The two
# upstream spellings differ, so assert the EFFECTIVE outcome, not one spelling:
#   0.26.0 release : sample_from_anchor = not getattr(hf, "dspark_bonus_anchor", False)
#   upstream main   : sample_from_anchor =     getattr(hf, "sample_from_anchor", True)
# Both give anchor-as-first for a config carrying neither key.
found = [s for s in ('"dspark_bonus_anchor", False', '"sample_from_anchor", True') if s in src]
assert found, (
    "DSparkSpeculator no longer selects the draft layout by either known getattr "
    "default; re-verify the layout before trusting acceptance"
)
print("dspark classes : DSparkSpeculator + Qwen3DSparkModel import")
print("draft layout   : anchor-as-first via", found[0], "-> query slots == k")

from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
assert "humming" in QUANTIZATION_METHODS
from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
from vllm.platforms import PlatformEnum
names = [k.__name__ for k in _POSSIBLE_KERNELS[PlatformEnum.CUDA]]
print("CUDA MP order  :", " > ".join(names))
# Recorded, not asserted as an ordering: this window uses no drafter kernel lever
# (bf16 drafter has no quantized linears). 0.26.0 demoted Humming to last; see
# docs/m3-serve-venv-026.md.
PY
cat "$ROOT/runtime-gates.txt"

aiperf_version=$(/mnt/nfs/hoangduy/venvs/perf/bin/aiperf --version 2>&1 | head -1)
printf 'aiperf=%s\n' "$aiperf_version" > "$ROOT/harness-version.txt"
say "harness: $aiperf_version"

# --- node health: slurm 'idle' is unreliable in both directions ---------------
say "probing $NODE for 8 GPUs each >= 70 GiB free"
r=$(timeout 120 srun --nodes=1 --ntasks=1 --nodelist="$NODE" --time=00:01:00 \
      --partition=compute --job-name="dspark-probe" \
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk '{c++; s+=$1; if ($1 < 70000) bad++}
           END {printf "%d:%d:%s", c, int(s/1024), (bad || c != 8 ? "BAD" : "ok")}')
case "$r" in
  *:ok) say "node $NODE: ${r%%:*} GPUs, $(echo "$r" | cut -d: -f2) GiB free" ;;
  *)    fail "node $NODE failed GPU probe: ${r:-no response}" ;;
esac

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

# ============================ LAUNCH ==========================================
arm_env=(
  "ROOT=$ROOT" "ARM=$ARM" "RUN_TS=$RUN_TS"
  "CKPT=$GPTQ_CKPT" "PORT_BASE=$PORT_BASE" "SB_DIR=$SB_DIR"
  "DRAFTER=$DRAFTER" "SERVES=$SERVES" "ONLY=$ONLY" "SMOKE_LABEL=$SMOKE_LABEL"
  "BACKEND=humming"
  "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
  "VLLM_HUMMING_MOE_GEMM_TYPE=indexed" "VLLM_HUMMING_USE_F16_ACCUM=0"
  "LLMC_M3_CAPTURE_SYNC=sync"
  # run_vllm_http_serve_smoke.sh sources this venv. Its default is `quant`
  # (vLLM 0.24.0), which has no dspark method at all, so this override is what makes
  # the window possible -- not a convenience.
  "SERVE_VENV=/mnt/nfs/hoangduy/venvs/serve-026"
  # The Humming W4A8 preflight is qualified against vLLM 0.24.0 and fail-closes on
  # anything else (reason VLLM_VERSION_MISMATCH). 0.26.0 cannot be qualified without
  # first running on it, so this names that exact version as provisional. It unlocks
  # 0.26.0 and nothing else, and every preflight/attestation artifact from this window
  # carries details.vllm_version_provisional=true plus a VLLM_VERSION_PROVISIONAL
  # advisory. Numbers from this window are NOT qualified Humming numbers until
  # 0.26.0 moves into QUALIFIED_VLLM_VERSIONS with a citation.
  #
  # THIS WINDOW GENERATES THAT CITATION: the D-k0-a / D-k0-b controls are Humming
  # k=0 on 0.26.0 over the identical staged prompts that the h114 window measured on
  # 0.24.0 the same day (136.8 tok/s conc 1; 75.2 8k-low / 80.3 8k-high at conc 10).
  # Agreement there is the same-workload, same-day, runtime-only comparison that
  # qualifies the W4A8 path; divergence is a finding in its own right.
  "LLMC_HUMMING_PROVISIONAL_VLLM=0.26.0"
  # No humming side dir: 0.1.10 is installed in serve-026 itself.
  "PYTHONPATH=$REPO"
)
env "${arm_env[@]}" \
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
     --time=08:00:00 --kill-on-bad-exit=1 --partition=compute \
     --nodelist="$NODE" --job-name="m3-dspark-$ARM" --export=ALL \
     bash "$REPO/pipeline/slurm/specdec_dspark_arm.sh" \
     > "$ROOT/$ARM-srun.log" 2>&1 &
ARM_PID=$!
printf 'arm=%s serves=%s port_base=%s node=%s pid=%s only=%s smoke=%s venv=serve-026 launched=%s\n' \
  "$ARM" "$n_serves" "$PORT_BASE" "$NODE" "$ARM_PID" "${ONLY:-<all>}" "$SMOKE_LABEL" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
say "launched $ARM on $NODE pid=$ARM_PID"
wait "$ARM_PID"
say "arm exited rc=$?"
