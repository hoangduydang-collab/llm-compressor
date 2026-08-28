#!/usr/bin/env bash
# Render and launch a GLM-5.2 distributed PTQ arm on the Rancher cluster.
#
#   pipeline/k8s/launch-quant-glm52.sh --method gptq --gpus 4
#   pipeline/k8s/launch-quant-glm52.sh --method awq  --gpus 4 --config <path>
#   pipeline/k8s/launch-quant-glm52.sh --method gptq --dry-run
#
# World size == --gpus. See the template header for why 4 is correct today and
# why it costs throughput but not correctness.
#
# This wrapper exists for the reason scripts/gpu-launch-multinode.sh exists:
# checking capacity by hand is necessary but not sufficient. It gates on REAL
# free GPUs (via ../../scripts/gpu-free.sh, which loops all six project
# namespaces — the `evaluation` namespace alone sees only a fraction of the
# cluster's GPU usage) and pins the code to a commit so the run is reproducible.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TMPL="$HERE/quantize-glm52.yaml.tmpl"
GPU_FREE="$(cd "$REPO_ROOT/.." && pwd)/scripts/gpu-free.sh"

METHOD=""
GPUS=4
CONFIG=""
REPO_REF=""
RUN_TAG=""
DRY_RUN=0
EVIDENCE_ONLY=0

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)   METHOD="${2:-}"; shift 2 ;;
    --gpus)     GPUS="${2:-}"; shift 2 ;;
    --config)   CONFIG="${2:-}"; shift 2 ;;
    --ref)      REPO_REF="${2:-}"; shift 2 ;;
    --run-tag)  RUN_TAG="${2:-}"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --evidence-only) EVIDENCE_ONLY=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$METHOD" in
  gptq|awq) ;;
  *) die "--method must be gptq or awq (got '${METHOD}')" ;;
esac
[[ "$GPUS" =~ ^[1-8]$ ]] || die "--gpus must be 1..8 (got '$GPUS')"

# Default config per method. The AWQ config is a separate file so the two arms
# cannot share an output_dir by accident.
if [[ -z "$CONFIG" ]]; then
  case "$METHOD" in
    gptq) CONFIG="pipeline/configs/glm52_distributed_w4afp8_smoke.yaml" ;;
    awq)  CONFIG="pipeline/configs/glm52_distributed_w4afp8_awq_smoke.yaml" ;;
  esac
fi
[[ -f "$REPO_ROOT/$CONFIG" ]] || die "config not found in repo: $CONFIG"

# --- pin the code -----------------------------------------------------------
# The pod clones a commit, not a branch: a branch would let the tree move
# between the two arms and make them incomparable.
if [[ -z "$REPO_REF" ]]; then
  REPO_REF="$(cd "$REPO_ROOT" && git rev-parse HEAD)"
  if ! (cd "$REPO_ROOT" && git diff --quiet && git diff --cached --quiet); then
    echo "WARNING: working tree has uncommitted changes; the pod will run" >&2
    echo "         $REPO_REF, which does NOT include them." >&2
  fi
  if ! (cd "$REPO_ROOT" && git branch -r --contains "$REPO_REF" 2>/dev/null | grep -q .); then
    die "commit $REPO_REF is not pushed; the pod clones from GitHub and could not fetch it"
  fi
fi

[[ -n "$RUN_TAG" ]] || RUN_TAG="$(date -u +%Y%m%dt%H%M%Sz)"
JOB="quant-glm52-${METHOD}-${RUN_TAG}"

# --- capacity gate ----------------------------------------------------------
# Fail closed. gpu-free.sh exits non-zero and prints nothing if it cannot read a
# namespace, so an unreadable namespace can never look like free capacity.
if [[ -x "$GPU_FREE" || -f "$GPU_FREE" ]]; then
  echo "==> checking real free GPU capacity"
  if ! REPORT="$(bash "$GPU_FREE" 2>&1)"; then
    echo "$REPORT" >&2
    die "occupancy check failed; refusing to launch blind"
  fi
  echo "$REPORT" | sed -n '/NODE  *USED/,$p'
  LARGEST="$(echo "$REPORT" | sed -n 's/.*largest single node *: *\([0-9]\+\).*/\1/p' | head -1)"
  [[ -n "$LARGEST" ]] || die "could not parse largest-free-node from occupancy report"
  if (( LARGEST < GPUS )); then
    die "largest single free node has $LARGEST GPU(s); need $GPUS. Not launching."
  fi
  echo "    largest single free node: $LARGEST (need $GPUS) OK"
else
  echo "WARNING: $GPU_FREE not found; launching without a capacity gate" >&2
fi

# --- render -----------------------------------------------------------------
# Rendered into the repo and handed to kubectl as a RELATIVE path, deliberately.
# mktemp gives a POSIX path (/tmp/...) that a Windows kubectl.exe cannot resolve
# ("the path ... does not exist"), and absolute Git-Bash paths (/c/Users/...) are
# no better. A relative path sidesteps path translation on every platform.
RENDER_DIR=".k8s-rendered"
cd "$REPO_ROOT"
mkdir -p "$RENDER_DIR"
RENDERED="$RENDER_DIR/${JOB}.yaml"
sed -e "s|@@METHOD@@|${METHOD}|g" \
    -e "s|@@GPUS@@|${GPUS}|g" \
    -e "s|@@CONFIG@@|${CONFIG}|g" \
    -e "s|@@REPO_REF@@|${REPO_REF}|g" \
    -e "s|@@RUN_TAG@@|${RUN_TAG}|g" \
    -e "s|@@EVIDENCE_ONLY@@|${EVIDENCE_ONLY}|g" \
    "$TMPL" > "$RENDERED"

grep -q '@@' "$RENDERED" && die "unsubstituted token remains: $(grep -o '@@[A-Z_]*@@' "$RENDERED" | sort -u | tr '\n' ' ')"

echo
echo "==> job      : $JOB"
echo "==> method   : $METHOD   world_size: $GPUS"
echo "==> config   : $CONFIG"
echo "==> code ref : $REPO_REF"
echo "==> manifest : $RENDERED"

if (( DRY_RUN )); then
  echo "==> --dry-run: not applying. Review the manifest above."
  exit 0
fi

kubectl apply -f "$RENDERED"
echo
echo "follow:  kubectl logs -n evaluation -f job/$JOB"
echo "stop:    kubectl delete job -n evaluation $JOB"
