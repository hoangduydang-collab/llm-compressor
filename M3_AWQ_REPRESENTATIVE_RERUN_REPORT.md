# MiniMax-M3 AWQ Representative-Layer Rerun

## Verdict

The six-arm diagnostic completed, but produced no usable AWQ mapping evidence.
The result is an `incomplete` infrastructure/calibration-path failure, not a
quality pass or a quality failure.

Run:

```text
20260713T043659Z-m3-awq-representative
```

Controller:

```text
started: 2026-07-13T04:37:04Z
finished: 2026-07-13T05:04:20Z
controller rc: 1
```

All six arms returned `rc=1`:

| Arm | Status | Resolved | Completed | Skipped | Unprocessed |
|---|---|---:|---:|---:|---:|
| `offsetfix-layer8` | infrastructure failure | 129 | 0 | 0 | 129 |
| `offsetfix-layer31` | infrastructure failure | 129 | 0 | 0 | 129 |
| `offsetfix-layer59` | infrastructure failure | 129 | 0 | 0 | 129 |
| `nosmooth-layer8` | infrastructure failure | 128 | 0 | 0 | 128 |
| `nosmooth-layer31` | infrastructure failure | 128 | 0 | 0 | 128 |
| `nosmooth-layer59` | infrastructure failure | 128 | 0 | 0 | 128 |

The compact matrix confirms:

```json
{
  "pass": 0,
  "quality_failure": 0,
  "infrastructure_failure": 6,
  "missing": 0,
  "verdict": "incomplete"
}
```

## Failure

Every arm reached AWQ initialization and lifecycle finalization, but completed
zero mapping grid searches:

```text
AWQ produced no completed mapping metrics; skipped_mappings=0
RuntimeError: AWQ completed zero mapping grid searches:
resolved=128/129 skipped=0 unprocessed=128/129
```

The runner raised this error from
`pipeline/m3_awq_representative.py:693` after reading each arm's
`lifecycle.json`. This is the same empty-completion class as the previous
representative attempt, although the old `ZeroDivisionError` is fixed: the AWQ
metric logger now returns safely and the harness reports the lifecycle defect
explicitly.

Representative `lifecycle.json` artifacts were written for all six arms. They
show nonzero resolved mappings but zero completed and skipped mappings, leaving
all resolved mappings unprocessed. No `arm.json` quality evidence was produced.

## Infrastructure status

The tmux controller survived and waited for all six arms. The arms were
launched with `srun --exclusive --nodes=1`; at startup they occupied six
distinct nodes:

```text
offsetfix-layer8   gpu-h123
offsetfix-layer31  gpu-h113
offsetfix-layer59  gpu-h114
nosmooth-layer8    gpu-h115
nosmooth-layer31   gpu-h116
nosmooth-layer59   gpu-h117
```

There was no node-collision OOM, nested-Slurm rejection, tool interruption, or
Slurm timeout. The failure is therefore inside the representative AWQ
calibration/lifecycle path, not the detached launcher or allocation topology.

## Durable evidence

Logs:

```text
/mnt/nfs/hoangduy/logs/m3-awq-representative/20260713T043659Z-m3-awq-representative/
```

Results:

```text
/mnt/nfs/hoangduy/results/m3-awq-representative/20260713T043659Z-m3-awq-representative/
```

The controller log, six arm logs, `matrix.json`, and `report.md` are retained
at those paths. Their SHA-256 values are recorded below:

| Artifact | SHA-256 |
|---|---|
| `controller.log` | `1c77b87ac53ed3ecd2525efdcdd1f2e968f0b6eaf851315eb462b7e31b2c4978` |
| `offsetfix-layer8.log` | `e156b08d67a062742a2f12b7495e2cd7bf47bd3e23846849d48ea981b5aae69d` |
| `offsetfix-layer31.log` | `75a39907ce08f98cd51c8d7bed664beafde9a4e05dcd8eb9e6bc5f4bbe278c9e` |
| `offsetfix-layer59.log` | `a78297c8e7804c516cd27009a6c2a547c89208b197ee5d410ac6132d2820f5b2` |
| `nosmooth-layer8.log` | `5e5d0ecfa50b99a4700c847fcb01a00d3c25214bc1519abaa59af95955bee94d` |
| `nosmooth-layer31.log` | `40bfc21c4bf376def008b3a2222d2bf76c3267dc433dbd5a0e2b6da82da2746c` |
| `nosmooth-layer59.log` | `d6e3a47e865dbe32b299143e4b9218e45a40ea9121fb8c94fa1373e7b196cbb9` |
| `matrix.json` | `5d079339019c99828c3eee3de66045354215c9bb58e7178ccb06bd0a7926aefa` |
| `report.md` | `40b409e503bf351b4e064b571cb8494cab4afceca69d48b72484c5c6eb17ed7d` |

## Next action

Do not start full AWQ re-quantization. The representative harness still needs
to explain why calibration reaches resolved mappings but never executes a
mapping grid search. The next planner should inspect:

- `pipeline/m3_awq_representative.py` around lifecycle capture and the
  `completed_mapping_count` guard;
- `src/llmcompressor/modifiers/transform/awq/base.py` around mapping execution,
  skipped metrics, and `_log_error_metrics`;
- the per-arm `lifecycle.json` and logs above.

The unrelated BF16 production-evaluation commits were fetched from
`origin/duy-branch` but were not pulled into the worktree because an existing
uncommitted quality-matrix change would have been overwritten. They are not
part of this report.
