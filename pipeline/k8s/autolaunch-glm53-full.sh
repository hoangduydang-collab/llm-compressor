#!/usr/bin/env bash
# Wait for the GLM-5.3-BF16 download to finish, verify the artifact, take gpu03
# back from our hold pod, and launch the full AWQ W4AFP8 run there.
#
# Authorized by the user on 2026-08-28 ("you don't need my permission, just do it,
# I'll be busy at that point"). Written as a script rather than done by hand
# because the trigger is ~1.5 h out and the launch should not depend on anyone
# being at a keyboard.
#
# WHAT IT REFUSES TO DO. Every precondition is checked and a failure ABORTS
# without launching, leaving a reason in the log. A 10-hour run started on a bad
# premise costs more than a missed window:
#
#   1. The download job must report Complete. Not "the directory looks big" --
#      the job's own completion condition.
#   2. 282 shards present. The FP8 release has 141; 141 shards would mean we are
#      about to quantize the wrong artifact (already happened once: 64 GB of the
#      FP8 release was downloaded before the mistake was caught).
#   3. config.json must have NO quantization_config. AWQ cannot consume a
#      block-scaled FP8 checkpoint -- llm-compressor has no weight_scale_inv read
#      path anywhere in src/ -- and this is the cheap way to prove we have BF16.
#   4. The 1-layer smoke must have SUCCEEDED. It exists to validate the gates this
#      run depends on, and it has already earned its keep once: it caught an
#      unresolvable AWQ mapping that 8 passing unit tests missed, which would
#      have failed this run after the full model load.
#   5. No other quantization job of ours may be Running, so two runs cannot
#      contend for the same cephfs offload directory.
#
# The hold pod is deleted only AFTER every check passes, immediately before the
# launch, to keep the window where gpu03 is unclaimed as small as possible.
set -uo pipefail

NS=evaluation
DOWNLOAD_JOB=hd-download-glm53-bf16
HOLD_POD=duy-gpu03-hold
NODE=aicloud-infermesh-test-ca-gpu03
CONFIG=pipeline/configs/glm53_distributed_w4afp8_awq_full.yaml
GPUS=8
SNAP_GLOB='/mnt/cephfs/.hf-cache/models--zai-org--GLM-5.3-BF16/snapshots/*'
EXPECT_SHARDS=282
SMOKE_JOB="${SMOKE_JOB:-}"          # set by the caller; checked if non-empty
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"   # 6 h ceiling

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
abort() { log "ABORT: $*"; log "gpu03 hold left in place; nothing launched."; exit 1; }

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || abort "not in a git repo"

log "waiting for job/${DOWNLOAD_JOB} to complete (poll ${POLL_SECONDS}s, ceiling ${MAX_WAIT_SECONDS}s)"
waited=0
while :; do
  phase=$(kubectl get job "$DOWNLOAD_JOB" -n "$NS" \
            -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)
  failed=$(kubectl get job "$DOWNLOAD_JOB" -n "$NS" \
            -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)
  [ "$phase" = "True" ] && { log "download job Complete"; break; }
  [ "$failed" = "True" ] && abort "download job reports Failed"
  [ "$waited" -ge "$MAX_WAIT_SECONDS" ] && abort "download did not complete within the ceiling"
  sleep "$POLL_SECONDS"
  waited=$((waited + POLL_SECONDS))
done

# --- verify the artifact, using the hold pod (it mounts the same cephfs PVC) ---
log "verifying the downloaded artifact"
verify=$(kubectl exec "$HOLD_POD" -n "$NS" -- sh -c "
  S=\$(ls -d ${SNAP_GLOB} 2>/dev/null | head -1)
  [ -n \"\$S\" ] || { echo 'NO_SNAPSHOT'; exit 0; }
  echo \"shards=\$(ls -1 \$S/*.safetensors 2>/dev/null | wc -l)\"
  python3 -c \"
import json
cfg = json.load(open('\$S/config.json'))
print('quantized=' + str(cfg.get('quantization_config') is not None))
print('layers=' + str(cfg.get('num_hidden_layers')))
print('arch=' + ','.join(cfg.get('architectures', [])))
\"
" 2>&1) || abort "artifact verification could not run: $verify"

log "artifact: $(echo "$verify" | tr '\n' ' ')"
echo "$verify" | grep -q NO_SNAPSHOT && abort "no snapshot directory under the HF cache"
shards=$(echo "$verify" | sed -n 's/^shards=//p')
[ "$shards" = "$EXPECT_SHARDS" ] || abort "expected ${EXPECT_SHARDS} shards, found '${shards}' (141 would mean the FP8 release)"
echo "$verify" | grep -q '^quantized=False' || abort "config.json HAS a quantization_config; this is not the BF16 release"
echo "$verify" | grep -q '^layers=78' || abort "unexpected num_hidden_layers"
echo "$verify" | grep -q 'arch=GlmMoeDsaForCausalLM' || abort "unexpected architecture"

# --- the smoke must have passed -----------------------------------------------
if [ -n "$SMOKE_JOB" ]; then
  smoke=$(kubectl get job "$SMOKE_JOB" -n "$NS" \
            -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)
  [ "$smoke" = "True" ] || abort "smoke job ${SMOKE_JOB} has not completed successfully (got '${smoke}'); it validates the gates this run depends on"
  log "smoke job ${SMOKE_JOB} Complete"
else
  log "WARNING: no SMOKE_JOB set, so the gate-validation precondition is unchecked"
fi

# --- no competing run ---------------------------------------------------------
running=$(kubectl get pods -n "$NS" -l purpose=distributed-ptq \
            --field-selector status.phase=Running -o name 2>/dev/null | wc -l)
[ "$running" -eq 0 ] || abort "${running} distributed-ptq pod(s) still Running; they would share the cephfs offload dir"

# --- take gpu03 and launch ----------------------------------------------------
log "all preconditions pass; releasing ${HOLD_POD} and launching"
kubectl delete pod "$HOLD_POD" -n "$NS" --wait=true 2>&1 | while read -r l; do log "  $l"; done

bash pipeline/k8s/launch-quant-glm52.sh \
  --method awq --gpus "$GPUS" --node "$NODE" --config "$CONFIG" 2>&1 |
  while read -r l; do log "  $l"; done

log "launched. Follow with: kubectl logs -n ${NS} -f job/\$(kubectl get jobs -n ${NS} -l purpose=distributed-ptq --sort-by=.metadata.creationTimestamp -o name | tail -1 | cut -d/ -f2)"
log "NOTE: gpu03's hold is gone. The quantization job now holds those 8 GPUs; when"
log "      it ends they are free for anyone. Re-apply k8s/gpu03-hold.yaml to keep them."
