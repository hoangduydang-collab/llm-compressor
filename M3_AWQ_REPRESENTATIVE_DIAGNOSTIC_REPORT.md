# MiniMax-M3 Representative AWQ Diagnostic

## Final status

Final state verified on 2026-07-12 at approximately 17:15 UTC from run
`20260712T164100Z-m3-awq-representative`.

The six arms were launched under detached `tmux`, with one
`srun --exclusive --nodes=1` allocation per arm. The earlier same-node
host-RAM/OOM failure was therefore avoided.

| Variant | Layer | Status | Result artifact |
| --- | ---: | --- | --- |
| `offsetfix` | 8 | Failed, `rc=1` | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/offsetfix-layer8` |
| `offsetfix` | 31 | Failed, `rc=1` | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/offsetfix-layer31` |
| `offsetfix` | 59 | Failed, `rc=1` | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/offsetfix-layer59` |
| `nosmooth` | 8 | Failed, `rc=1` | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/nosmooth-layer8` |
| `nosmooth` | 31 | Failed, `rc=1` | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/nosmooth-layer31` |
| `nosmooth` | 59 | Failed, `rc=1` | `/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/nosmooth-layer59` |

The detached controller finished at `2026-07-12T17:11:58+00:00` with `rc=1`.
Slurm has no active arm allocations and the tmux session is gone. The compact
matrix report classifies all six arms as `infrastructure_failure` and gives an
overall `incomplete` verdict because no arm produced quality evidence.

## Failure evidence

All six arms reached the AWQ error-metric calculation and raised:

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
| `controller.log` | 1,098 | `566e79905efbeae6f86b3136ba7a484ebc3796f0aac6d21cf393a052464ea80c` |
| `offsetfix-layer8.log` | 46,499 | `0cfbc04be1ad60d44debee6fb2b505ffa4100801c6ce93e407ef8cf2542cb817` |
| `offsetfix-layer31.log` | 48,256 | `ddb2751387b920dd84b280dd500b5d8388b23933da7513d941c3fcf6b061e529` |
| `offsetfix-layer59.log` | 47,916 | `6f67207473ad587b8260f3d3a530e02c9827d71bc683f9c5af09dc2d8edf96b8` |
| `nosmooth-layer8.log` | 41,663 | `cde05621c9ade940c0c4ca82d86f2c5ef2d60107c0c10c07489a7b4ff93ec4b6` |
| `nosmooth-layer31.log` | 47,561 | `6bbdc287300ab373f8305d96ea039272f489f7aa11643f063c78245e5afe6a66` |
| `nosmooth-layer59.log` | 42,486 | `e05b680d40b16bd9f97e648cbc1753edc582dfc1e90ba29779bde90902b7f936` |

The compact result roots, including `start.json` and completed-arm `rc` files,
remain at:

`/mnt/nfs/hoangduy/results/m3-awq-representative/20260712T164100Z-m3-awq-representative/`

The controller log records the six arm launch PIDs, corresponding log paths,
per-arm return codes, aggregate return code, and finish time. The final
compact result files include `matrix.json`, `report.md`, `start.json`, and
`rc` for every arm.

## Next action

Treat this diagnostic as a blocked AWQ calibration-path experiment, not as
evidence that either recipe passed or failed model quality. Investigate why
the representative-layer setup produces an empty `reductions` list, add a
regression test for that case, and rerun the diagnostic only after the
calibration/metric lifecycle is corrected. Do not launch either full AWQ
rebuild until this failure is explained.
