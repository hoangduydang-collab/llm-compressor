# Rancher Full GPU-Node Holder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Queue a priority-zero Pod that can bind only when one current eight-GPU Rancher node is completely free, then hold all eight GPUs under a renewable 48-hour lease.

**Architecture:** Use Kubernetes' atomic resource scheduler instead of polling. A bare Pod requests all eight GPUs, targets GPU workers, and tolerates their taint; while every eligible node has capacity eight, the request cannot fit on a partially free node. A timestamp file inside the container implements a renewable 48-hour lease.

**Tech Stack:** Kubernetes v1.34, Rancher `infermesh-test-my`, PowerShell, Git Bash, `scripts/gpu-free.sh`

## Global Constraints

- Namespace: `evaluation`; Pod name: `hoangduy-full-node-hold`.
- Reserve exactly eight GPUs and no deliberate CPU, RAM, RDMA, host-path, or privileged resources.
- Use `restartPolicy: Never` and the admitted priority-zero policy. Rancher rejects
  Pod-specified `preemptionPolicy: Never`; all current GPU pods are priority zero,
  so this holder cannot preempt them.
- Expire 48 hours after container start or the latest lease renewal, with a
  five-minute maximum expiry-check delay.
- Revalidate that every eligible GPU node has capacity exactly eight immediately before creation.
- Run `scripts/gpu-free.sh --verify`; abort if namespace accounting and Rancher accounting disagree.
- Do not delete or replace an existing resource with the same name.
- The owner approved the revised renewable GPU claim after reviewing the exact manifest on 2026-09-09.

---

### Task 1: Create and verify the queued holder

**Files:**
- Reference: `scripts/gpu-free.sh`
- Reference: `docs/superpowers/specs/2026-09-09-rancher-full-node-holder-design.md`
- Create in cluster: `Pod/evaluation/hoangduy-full-node-hold`

**Interfaces:**
- Consumes: Kubernetes node capacities, GPU scheduling state, and the approved Pod manifest.
- Produces: one Pending or scheduled holder Pod; manual release remains `kubectl delete pod -n evaluation hoangduy-full-node-hold`.

- [ ] **Step 1: Revalidate the full-node invariant and current occupancy**

Run:

```powershell
$raw = kubectl get nodes -l 'node-role.kubernetes.io/remote-worker' -o json
if ($LASTEXITCODE -ne 0) { throw 'Cannot read eligible GPU nodes.' }
$nodes = ($raw | ConvertFrom-Json).items
if ($nodes.Count -eq 0) { throw 'No eligible GPU nodes found.' }
$caps = $nodes | ForEach-Object {
  [pscustomobject]@{
    Name = $_.metadata.name
    Capacity = [int]$_.status.capacity.'nvidia.com/gpu'
  }
}
$caps | Format-Table -AutoSize
if (($caps | Where-Object Capacity -ne 8).Count -ne 0) {
  throw 'Eligible GPU nodes are not uniformly eight-GPU nodes; refusing to queue.'
}
& 'C:\Program Files\Git\bin\bash.exe' scripts/gpu-free.sh --verify
if ($LASTEXITCODE -ne 0) { throw 'Authoritative GPU check failed.' }
$existing = kubectl get pod -n evaluation hoangduy-full-node-hold --ignore-not-found -o name
if ($LASTEXITCODE -ne 0) { throw 'Existing holder check failed.' }
if ($existing) { throw "Holder already exists: $existing" }
```

Expected: all nine eligible nodes report capacity `8`; both occupancy reports agree; no existing holder Pod is returned. If any condition fails, stop without creating anything.

- [ ] **Step 2: Validate the manifest through the API server without persisting it**

Run:

```powershell
@'
apiVersion: v1
kind: Pod
metadata:
  name: hoangduy-full-node-hold
  namespace: evaluation
  labels:
    app: full-node-hold
    owner: hoangduy
spec:
  restartPolicy: Never
  nodeSelector:
    node-role.kubernetes.io/remote-worker: ""
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  containers:
    - name: hold
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          touch /tmp/lease-renewed
          while true; do
            now=$(date +%s)
            renewed=$(stat -c %Y /tmp/lease-renewed)
            [ $((now-renewed)) -ge 172800 ] && exit 0
            sleep 300
          done
      resources:
        requests:
          nvidia.com/gpu: "8"
        limits:
          nvidia.com/gpu: "8"
'@ | kubectl create --dry-run=server -f -
```

Expected: `pod/hoangduy-full-node-hold created (server dry run)`.

- [ ] **Step 3: Create the holder**

Run:

```powershell
@'
apiVersion: v1
kind: Pod
metadata:
  name: hoangduy-full-node-hold
  namespace: evaluation
  labels:
    app: full-node-hold
    owner: hoangduy
spec:
  restartPolicy: Never
  nodeSelector:
    node-role.kubernetes.io/remote-worker: ""
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  containers:
    - name: hold
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          touch /tmp/lease-renewed
          while true; do
            now=$(date +%s)
            renewed=$(stat -c %Y /tmp/lease-renewed)
            [ $((now-renewed)) -ge 172800 ] && exit 0
            sleep 300
          done
      resources:
        requests:
          nvidia.com/gpu: "8"
        limits:
          nvidia.com/gpu: "8"
'@ | kubectl create -f -
```

Expected: `pod/hoangduy-full-node-hold created`.

- [ ] **Step 4: Verify scheduling and the no-partial-node contract**

Run:

```powershell
kubectl get pod -n evaluation hoangduy-full-node-hold `
  -o custom-columns='NAME:.metadata.name,PHASE:.status.phase,NODE:.spec.nodeName,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu,PRIORITY:.spec.priority'
kubectl describe pod -n evaluation hoangduy-full-node-hold
& 'C:\Program Files\Git\bin\bash.exe' scripts/gpu-free.sh --verify
```

Expected while the cluster remains full: `PHASE=Pending`, `NODE=<none>`, `GPU=8`, and `PRIORITY=0`. If it binds during creation, the assigned node must show this Pod holding `8/8` GPUs in the verified report. A scheduled Pod in image startup or pull backoff still holds the requested GPUs and must be reported as acquired, not free.

- [ ] **Step 5: Record renewal and manual release operations**

To reset a Running holder's lease to another 48 hours, run:

```powershell
kubectl exec -n evaluation hoangduy-full-node-hold -- touch /tmp/lease-renewed
```

Expected: command exits zero. Verify the timestamp with
`kubectl exec -n evaluation hoangduy-full-node-hold -- stat -c %y /tmp/lease-renewed`.

Do not release during creation. When the owner asks to release the node, first confirm intent and then run:

```powershell
kubectl delete pod -n evaluation hoangduy-full-node-hold
```

Expected: the Pod is deleted and a fresh `scripts/gpu-free.sh --verify` report reflects the released GPUs.
