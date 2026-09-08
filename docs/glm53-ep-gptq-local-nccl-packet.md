# Local two-A100 NCCL diagnostic contract

Scope: the approved small two-GPU integration gate, executed locally by the
implementing agent under `FULL_STACK_AGENT_PROTOCOL.md`. The separate remote
executor retains the real-width and full-run work.

Decision: do the tiny real-GLM EP tests preserve numerical behavior and survive
NCCL, actual disk offload, collective packed saving and reload?

Resource limit: one x86_64 node (`xgph12`), two A100-40 GPUs, four CPU cores,
16 GiB host memory, 15-minute allocation limit, 30-second immediate placement
limit. Three cases run sequentially; expected runtime is several minutes. No
full-model download, calibration corpus, serving or quality benchmark is involved.

Launcher: `srun` from detached tmux session `glm53_ep_nccl_20260909`.
Working directory: `/home/e/e1129930/llm-compressor`; interpreter:
`.venv-prequant/bin/python`. The controller records the exact committed revision
and requires no code/config/test diff before allocation. NVIDIA and Python
preflight must report two visible CUDA GPUs before pytest begins.

Command inside the allocation:

```bash
env -u SLURM_STEP_ID OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-prequant/bin/python -m pytest -q -rs -o tmp_path_retention_policy=all --basetemp="/tmp/pytest-of-e1129930/glm53-ep-nccl-${SLURM_JOB_ID}" tests/llmcompressor/modifiers/gptq/test_expert_parallel_nccl.py
```

The tests spawn two ranks, so there is one Slurm task and no outer torchrun.
Existing fixture gates are fixed in the committed tests: Hessian/count and
intermediate-output comparisons, GPTQ packed-save/reload, finite scales/outputs,
and mixed-modifier noninterference. Pure GPTQ compares DDP/EP. Mixed FP8-rest
has no legacy DDP comparator because that baseline overwrites GPTQ scales.

Stop on preflight or test failure; capture raw logs and scheduler state. Do not
change tolerances, replace a skipped test with a pass, or escalate to a larger
model. There is no automatic retry. Any repair must retain the same test design
and be recorded under a fresh attempt and implementation revision.

Evidence root: `results/glm53-ep-gptq/20260909-local-nccl/` (scripts prepared before launch).
The controller records revision, raw stdout/stderr, exit status, environment,
job/node/GPU identity and scheduler state. Code stays protected during the run.
Actual checkpoint artifacts remain in the job-specific pytest directory on its
compute node; retain small comparison/manifest records in Git.

This is a bounded local diagnostic, not a ready full-run packet for Rancher.

Result: job 830661 failed before worker startup; all three GPU gates remain
pending. See the [attempt record](../results/glm53-ep-gptq/20260909-local-nccl/README.md).
