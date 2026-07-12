# MiniMax-M3 Representative AWQ Diagnostic

## Snapshot

Snapshot taken on 2026-07-12 at approximately 17:10 UTC from run
`20260712T164100Z-m3-awq-representative`.

The six arms were launched under detached `tmux`, with one
`srun --exclusive --nodes=1` allocation per arm. The earlier same-node
host-RAM/OOM failure was therefore avoided.

| Variant | Layer | Status | Result artifact |
| --- | ---: | --- | --- |
| `offsetfix` | 8 | Failed, `rc=1` | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/offsetfix-layer8` |
| `offsetfix` | 31 | Running at snapshot | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/offsetfix-layer31` |
| `offsetfix` | 59 | Running at snapshot | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/offsetfix-layer59` |
| `nosmooth` | 8 | Failed, `rc=1` | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/nosmooth-layer8` |
| `nosmooth` | 31 | Running at snapshot | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/nosmooth-layer31` |
| `nosmooth` | 59 | Failed, `rc=1` | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/nosmooth-layer59` |

At the snapshot, Slurm still showed three running allocations and the
controller `tmux` session was alive. The diagnostic must not yet be
aggregated or treated as complete.

## Failure evidence

All three failed arms reached the AWQ error-metric calculation and raised:

```text
avg_reduction = sum(reductions) / len(reductions)
ZeroDivisionError: division by zero
```

The traceback is in the AWQ transform metric code at
`llmcompressor/modifiers/transform/awq/base.py:846`. This is a deterministic
empty-metric failure, not the previous node-co-location `rc=137` failure.
The logs do not establish yet whether the empty reduction list is caused by
layer selection, calibration coverage, or a metric guard missing from the
AWQ implementation.

## Durable logs and artifacts

The complete logs remain outside Git at:

`/mnt/nfs/hoangduy/logs/m3-awq-representative/20260712T164100Z-m3-awq-representative/`

They total 275,479 bytes. The controller log and six arm logs were hashed at
this snapshot:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `controller.log` | 1,098 | `ec3f0ef1a70e73c74746aa88b7e3393878ccaa8dd02a63e24bb798f002d34d76` |
| `offsetfix-layer8.log` | 46,499 | `0cfbc04be1ad60d44debee6fb2b505ffa4100801c6ce93e407ef8cf2542cb817` |
| `offsetfix-layer31.log` | 48,256 | `f682dbd554c0f6f0517582f217f5665692d57fcd89eccd136d77be96cc739e7c` |
| `offsetfix-layer59.log` | 47,916 | `7cbb09279ff886bf1576ed6f1dcb7006f746345876641f0b9193797129b50c7f` |
| `nosmooth-layer8.log` | 41,663 | `cde05621c9ade940c0c4ca82d86f2c5ef2d60107c0c10c07489a7b4ff93ec4b6` |
| `nosmooth-layer31.log` | 47,561 | `fbaa6a6d020ee8e178f738021bf2333b078bc56eb8a9233c18698d0fe10d16e3` |
| `nosmooth-layer59.log` | 42,486 | `e05b680d40b16bd9f97e648cbc1753edc582dfc1e90ba29779bde90902b7f936` |

The compact result roots, including `start.json` and completed-arm `rc` files,
remain at:

`/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/`

The controller log records the six arm launch PIDs and their corresponding
log paths. The running arms may add files or change their logs after this
snapshot; recompute the hashes after completion.

## Next action

Wait for the three active arms to finish. Then collect their final return
codes and evidence, inspect whether the same empty-metric condition occurs at
layers 31 or 59, and only then decide whether to add a guarded zero-sample
handling fix or revise the representative diagnostic. Do not launch either
full AWQ rebuild until this diagnostic has been explained.
