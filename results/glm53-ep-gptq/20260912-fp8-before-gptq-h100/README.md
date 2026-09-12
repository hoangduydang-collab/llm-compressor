# FP8-before-GPTQ two-H100 qualification

User authorized qualification on 2026-09-12. Source revision: `d90ea79c49b4e62d314cc8d9eba190fdfba0ddf4`.
Reuses the earlier local H100 controller/worker. Five fixed tiny GLM NCCL cases:
DDP/EP, weight-only disk, legacy mixed disk, early-FP8 parity, early-FP8 disk.
One node, two non-MIG H100s, 4 CPUs, 16 GiB host RAM, 15-minute GPU runtime,
eight-hour queue cap. No full-model inputs or runtime/quality evaluation.

Launch from detached tmux `glm53_fp8_h100_20260912`. Worker uses `git archive`
into node-local storage and the existing pinned `.venv-prequant` environment.
No test thresholds change. Zero skipped tests and all five passes are required.
One submission; failures stop this attempt and retain raw logs, XML, environment,
scheduler records and per-rank diagnostics. Any repair is a new attempt.

EP4/EP8 representative execution is pending Rancher access: no `kubectl` or
`/mnt/cephfs` source mount is accessible from the planner workspace.
