# Rancher full GPU-node holder

Date: 2026-09-09
Status: Design approved by owner on 2026-09-09
Namespace: `evaluation`

## Goal

Queue one Kubernetes workload that remains Pending until an entire eight-GPU
Rancher node is available, then reserves all eight GPUs for a renewable 48-hour
lease. It must not take a partially free node.

## Baseline and decision

The cluster's established holder pattern is a priority-zero bare Pod that requests
all eight `nvidia.com/gpu` resources and tolerates the GPU-node taint. Use that
scheduler contract rather than an external polling loop. All nine current GPU
nodes expose exactly eight GPUs, so an atomic request for eight cannot fit on a
partially free node.

Rancher's priority admission controller rejects a Pod-specified
`preemptionPolicy: Never`, and this account cannot create a non-preempting
PriorityClass. All current GPU workloads are priority zero, so the holder cannot
preempt them because Kubernetes preempts only lower-priority workloads. This is
the established cluster holder pattern, not an absolute future guarantee if
negative-priority GPU workloads are introduced.

Use a bare Pod rather than a Job because a Job controller could recreate a holder
after its Pod is deleted. Do not use a `gpu-free.sh` watcher because the
check-to-create interval introduces a race that the Kubernetes scheduler already
solves atomically.

## Workload contract

- Name: `evaluation/hoangduy-full-node-hold`
- Image and command: `busybox:1.36`, running a five-minute lease-check loop
- Resources: requests and limits of exactly eight `nvidia.com/gpu`; no deliberate
  CPU, RAM, RDMA, host-path, or privileged reservation
- Placement: `node-role.kubernetes.io/remote-worker` nodes with the
  `nvidia.com/gpu:NoSchedule` toleration
- Preemption: admitted cluster default, priority zero
- Lifecycle: `restartPolicy: Never`; 48 hours from container start or the latest
  renewal, with release within five minutes after expiry

The Pod may remain Pending indefinitely, and the lease starts only when its
container starts. A successful bind means the scheduler found eight unallocated
GPUs on one current eight-GPU node. Renewing touches `/tmp/lease-renewed`, resetting
the 48-hour interval.

## Verification and operations

Before creation, run `scripts/gpu-free.sh --verify` and retain its output as the
authoritative occupancy snapshot. After creation:

1. Confirm the Pod exists and is Pending or Running.
2. If Pending, confirm it has no assigned node and requests eight GPUs.
3. If Running, record the assigned node and rerun `scripts/gpu-free.sh --verify`;
   that node must report `8/8` used by this Pod.
4. Treat script failure or disagreement with Rancher accounting as untrusted.

Renew while Running with:

```bash
kubectl exec -n evaluation hoangduy-full-node-hold -- touch /tmp/lease-renewed
```

Kubernetes does not allow increasing an existing Pod's
`activeDeadlineSeconds`, so the renewable file timestamp is deliberate.

Release early only with explicit owner intent:

```bash
kubectl delete pod -n evaluation hoangduy-full-node-hold
```

## Assumptions and removal

The full-node guarantee depends on every eligible GPU node having capacity eight,
as verified on 2026-09-09. Revalidate this before recreating the holder; if
heterogeneous larger nodes are added, an eight-GPU request no longer proves the
selected node was completely free. Delete the Pod when the reservation is no
longer needed.
