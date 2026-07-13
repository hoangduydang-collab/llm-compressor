# MiniMax-M3 tmux Smoke Progress Snapshot

Run root:

`results/m3-quality/20260712T162609Z-m3-quality-smoke-tmux`

This snapshot was taken while the detached tmux controller was still running.
No live job was interrupted.

## Completed evidence

- Fresh CPU preflight completed successfully before launch.
- GPTQ and AWQ 2,047-token teacher-forced probes completed.
- Both probes used corpus hash
  `f1e6e4a3c7323bf0d43cd0a670adce667b8e1e0cdc7982879298ef41afdb0704`.
- GPTQ probe elapsed time: 7.37 seconds.
- AWQ probe elapsed time: 8.78 seconds.
- Ray placement diagnostic completed with a timeout waiting for its placement
  group driver; its gate, status, node, rank, and stop artifacts are included.
- BF16 attempted after Ray and failed before model launch because the runner
  expected `ray_preflight/gate.json`, while the placement diagnostic wrote
  `ray_placement/topology/gate.json`.

Included evidence covers preflight metadata, controller launch metadata,
Ray/BF16 logs, Ray topology artifacts, GPTQ/AWQ probe JSONL and summaries,
arm manifests, partial aggregates, samples, and generation-health files
available at snapshot time. The repaired checkpoint itself is excluded.

## Still running at snapshot

The GPTQ and AWQ Slurm arm steps were still active, so their current logs and
partial artifacts are not final results. The tmux controller had not written
`controller.rc`.

Unrelated AWQ representative-layer jobs on `gpu-h123` were not inspected,
modified, cancelled, or included in this run package.

## Executor-side work outside the planner handoff

The executor created a fresh run root and repeated the CPU preflight so the
corrupted/interrupted earlier run root was not reused. The executor also
performed the tmux dry-run and verified the detached session before launch.
These are operational safeguards only; no model, prompt, probe, or resource
configuration was changed.
