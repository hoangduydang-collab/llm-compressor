#!/usr/bin/env bash
# Render and, after explicit operator approval, submit the GLM-5.3 EP GPTQ chain.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
GPU_FREE="$(cd "$REPO_ROOT/.." && pwd)/scripts/gpu-free.sh"
RUN_TAG=""
REPO_REF=""
NODE=""
DRY_RUN=0

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-tag) RUN_TAG="${2:-}"; shift 2 ;;
    --ref) REPO_REF="${2:-}"; shift 2 ;;
    --node) NODE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$RUN_TAG" ]] || RUN_TAG="$(date -u +%Y%m%dt%H%M%sz)"
[[ -n "$REPO_REF" ]] || REPO_REF="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$REPO_REF" =~ ^[0-9a-f]{40}$ ]] || die "--ref must be a full commit SHA"
if ! git -C "$REPO_ROOT" branch -r --contains "$REPO_REF" 2>/dev/null | grep -q .; then
  die "commit $REPO_REF is not on a remote branch; push before launching"
fi

RENDER_DIR="$REPO_ROOT/.k8s-rendered"
mkdir -p "$RENDER_DIR"
RENDERED="$RENDER_DIR/glm53-ep-gptq-${RUN_TAG}.yaml"
BODY="$RENDER_DIR/glm53-ep-gptq-${RUN_TAG}.body.sh"

render_args=(
  --run-tag "$RUN_TAG"
  --ref "$REPO_REF"
  --out "$RENDERED"
)
if [[ -n "$NODE" ]]; then
  render_args+=(--node "$NODE")
fi
python "$HERE/render_glm53_ep_gptq_chain.py" "${render_args[@]}"

python - "$RENDERED" "$BODY" <<'PY'
import pathlib, sys, yaml
doc = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
body = doc["spec"]["template"]["spec"]["containers"][0]["args"][0]
pathlib.Path(sys.argv[2]).write_text(body, encoding="utf-8", newline="\n")
PY
[[ -s "$BODY" ]] || die "rendered container script is empty"
bash -n "$BODY" || die "rendered container script is not valid bash"
grep -q '@@' "$RENDERED" && die "rendered manifest contains a placeholder"

echo "==> authoritative Rancher occupancy"
REPORT="$(bash "$GPU_FREE" --verify)"
echo "$REPORT"
LARGEST="$(printf '%s\n' "$REPORT" |
  sed -n 's/.*largest single node *: *\([0-9]\+\).*/\1/p' | head -1)"
[[ -n "$LARGEST" ]] || die "could not parse largest schedulable pod size"
(( LARGEST >= 8 )) || die "no completely free eight-GPU node; largest is $LARGEST"

echo "==> job: glm53-ep-gptq-$RUN_TAG"
echo "==> ref: $REPO_REF"
echo "==> image: docker.io/lmsysorg/sglang@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155"
echo "==> request: 8 GPUs, 16 CPU, 700Gi RAM, 220Gi ephemeral"
echo "==> failure: hold 24h; success: release immediately"
echo "==> manifest: $RENDERED"

if (( DRY_RUN )); then
  echo "==> dry run only; not applying"
  exit 0
fi

kubectl apply -f "$RENDERED"
echo "follow: kubectl logs -n evaluation -f job/glm53-ep-gptq-$RUN_TAG"
