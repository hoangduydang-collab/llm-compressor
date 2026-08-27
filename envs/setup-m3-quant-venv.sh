#!/usr/bin/env bash
# Reproduce the MiniMax-M3 quantization environment.
#
#   envs/setup-m3-quant-venv.sh [venv-dir]
#
# The reference is envs/m3-quant-freeze.txt, copied verbatim from the retired
# cluster at /data/hoangduy.dang/nfs-hoangduy/venvs/freeze-manifests/quant.txt.
# Header of that file:
#     # venv: quant
#     # python: Python 3.12.13
#     # frozen: 2026-07-31
#
# WHY THIS EXISTS. GLM-5.2 was being quantized on the sglang image's stack
# (transformers 5.12.1) while the gate ran on 5.14.1 and M3 was validated on
# 5.14.1. A GLM-vs-M3 comparison across different library versions is confounded
# on exactly the axis the M3 record has a whole "Transformers 5.14.1 upgrade
# assessment" section about. Same manifest for both models, or the comparison is
# not a comparison.
#
# TWO TIERS, because a faithful install is not possible everywhere:
#   full     every pin, as frozen. Linux + CUDA only.
#   planner  the subset that determines quantization SEMANTICS (torch,
#            transformers, compressed-tensors, accelerate, tokenizers,
#            safetensors, datasets, numpy, huggingface_hub, ...) with the
#            CUDA/serving-only packages dropped. For the CPU-only planner box,
#            which runs gates and unit tests and never touches a GPU.
# Pass MODE=full or MODE=planner; default is chosen from the platform.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FREEZE="$HERE/m3-quant-freeze.txt"
VENV="${1:-$HERE/../.venv-m3-quant}"
PY="${PY:-python3.12}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$FREEZE" ]] || die "missing $FREEZE"

case "$(uname -s)" in
  Linux) DEFAULT_MODE=full ;;
  *)     DEFAULT_MODE=planner ;;
esac
MODE="${MODE:-$DEFAULT_MODE}"

command -v "$PY" >/dev/null 2>&1 || die "$PY not found; M3 used Python 3.12.13"
echo "==> interpreter : $("$PY" --version 2>&1)  (M3: Python 3.12.13)"
echo "==> mode        : $MODE"
echo "==> venv        : $VENV"

# Packages that cannot install on a CPU-only / non-Linux box. Dropping any of
# these changes NOTHING about quantization numerics -- they are CUDA kernels,
# serving runtimes, and their transitive NVIDIA wheels.
PLANNER_EXCLUDE_RE='^(nvidia-|triton|flashinfer|torch_c_dlpack_ext|vllm|sglang|xformers|flash-attn|flash_attn|deep_gemm|cuda-|pynvml|nvitop)'

python_bin() { echo "$VENV/bin/python"; }
[[ -d "$VENV/Scripts" ]] && python_bin() { echo "$VENV/Scripts/python.exe"; }

if [[ ! -d "$VENV" ]]; then
  "$PY" -m venv "$VENV" || die "venv creation failed"
fi
# Resolve the interpreter path for either layout (POSIX bin/, Windows Scripts/).
VPY="$VENV/bin/python"
[[ -x "$VPY" ]] || VPY="$VENV/Scripts/python.exe"
[[ -x "$VPY" ]] || die "cannot find the venv interpreter under $VENV"

"$VPY" -m pip install --quiet --upgrade pip setuptools wheel \
  || die "pip bootstrap failed"

REQ="$VENV/.requirements.txt"
if [[ "$MODE" == "full" ]]; then
  grep -vE '^\s*#' "$FREEZE" > "$REQ"
  echo "==> installing ALL $(wc -l < "$REQ") pins verbatim"
else
  grep -vE '^\s*#' "$FREEZE" | grep -vE "$PLANNER_EXCLUDE_RE" > "$REQ"
  DROPPED="$VENV/.excluded.txt"
  grep -vE '^\s*#' "$FREEZE" | grep -E "$PLANNER_EXCLUDE_RE" > "$DROPPED" || true
  echo "==> installing $(wc -l < "$REQ") pins; EXCLUDED $(wc -l < "$DROPPED") CUDA/serving-only:"
  sed 's/^/      - /' "$DROPPED"
fi

# One shot with the exact pins. Not --no-deps: we want pip to verify the pinned
# set is mutually consistent rather than silently assembling a different one.
if ! "$VPY" -m pip install -r "$REQ"; then
  echo
  echo "==> the pinned set did not install cleanly. This is expected on some"
  echo "    platforms and is the point where we stop and look, rather than"
  echo "    quietly relaxing pins. Retrying the SEMANTIC core alone so the box"
  echo "    is at least usable, and reporting what could not be satisfied."
  CORE="$VENV/.core.txt"
  grep -E '^(torch|transformers|compressed-tensors|accelerate|tokenizers|safetensors|datasets|numpy|huggingface_hub|pydantic|loguru|pyyaml)==' "$FREEZE" > "$CORE"
  "$VPY" -m pip install -r "$CORE" || die "even the semantic core failed; stop here"
  echo "==> core installed; FULL PIN SET NOT REPRODUCED -- treat this venv as approximate"
fi

# The M3 venv's repair to transformers' sharded offloaded save_pretrained was
# an in-place source edit, which `pip freeze` cannot record -- so installing
# the manifest reproduces the version and drops the patch. Re-apply it here or
# every rebuilt venv silently reintroduces the bug.
echo
echo "==> re-applying the transformers sharded-save hotfix"
"$VPY" "$HERE/hotfix-transformers-sharded-save.py" \
  || die "transformers sharded-save hotfix failed; see the marker note above"

echo
echo "==> installed versions vs the M3 manifest"
"$VPY" -m pip install --quiet -e "$HERE/.." 2>&1 | tail -2 || true
"$VPY" - "$FREEZE" <<'PY'
import sys, importlib.metadata as md
want = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#") or "==" not in line:
        continue
    n, v = line.split("==", 1)
    want[n.lower().replace("_", "-")] = v
check = ["torch", "transformers", "compressed-tensors", "accelerate",
         "huggingface-hub", "tokenizers", "safetensors", "datasets", "numpy"]
print(f"  {'package':22s} {'wanted':28s} {'installed':28s} match")
ok = True
for name in check:
    w = want.get(name, "-")
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        got = "NOT INSTALLED"
    same = got == w
    ok &= same or w == "-"
    print(f"  {name:22s} {w:28s} {got:28s} {'OK' if same else 'DIFFERS'}")
print()
print("VERDICT:", "matches M3 on the semantic core" if ok
      else "DIVERGES from M3 -- see rows marked DIFFERS")
PY
echo
echo "activate with:  source $VENV/bin/activate   (or $VENV/Scripts/activate)"
