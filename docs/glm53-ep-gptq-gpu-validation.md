# GLM-5.3 EP GPTQ GPU validation, 9 September 2026

Status: **all three bounded two-T4 NCCL tests passed**, plus all 10 affected CPU tests.
This is the active local diagnostic record. Representative H100 and full-model
execution remain separate, unqualified gates.

## Launch investigation

| Job | Purpose | Result |
| --- | --- | --- |
| 830661 | Original A100 allocation | Queued allocation exited after 10 seconds, before worker startup. |
| 830714 | Reproduce inherited environment with hostname | Task rejected: inherited CPU mask lies outside allocated cores. |
| 830715 | Clean environment hostname probe | Same 10-second queued allocation failure as the original; no task. |
| 830721 | Clean environment, cores binding, immediate placement | Worker started; hardware preflight failed. A100-40 resources were two MIG slices on one physical A100-80. |
| 830729 | Same small suite on two full T4s | Hardware passed; pytest setup failed because its temporary parent directory was absent. |
| 830738 | T4 suite after temporary-directory repair | Parity/save/reload passed; two disk cases failed in initial cache setup with NCCL timeouts. |
| 830817 | T4 suite after CPU-qualified setup/export repairs | **3 passed**, 119.83 seconds in pytest; allocation completed in 3 min 1 s with exit 0. |

Confirmed launcher repairs: start `srun` with a clean environment (reuse the
repo's `env -i` pattern) and explicit `--cpu-bind=cores`. Immediate placement
(`--immediate=1`) successfully starts work from this host. The original queued
allocation callback failure is reproduced but its underlying cause is not proven.
The short hostname resolves to loopback, which is a possible clue, not a diagnosis.
Node logs are not readable, and SSH probes to the node/login host timed out.
No cluster configuration was changed.

Slurm exposes CPU binding through its input environment; see
[official srun documentation](https://slurm.schedmd.com/srun.html).
NVIDIA documents that [MIG does not support NCCL](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/deployment-considerations.html).
Do not infer physical GPU count from scheduler GRES names.

## Repairs found during validation

- Prepare both pytest's node-local temporary parent and the disk offload directory.
- Linearized experts already have CPU caches. Transition through CT's existing
  `remove_module_offload(..., onload_tensors=True)` before installing disk caches.
- Add CPU coverage of the exact two disk lifecycle cases, so GPU-only skips cannot
  hide setup or serialization errors.
- The installed packer can return a non-contiguous view for a width of 16. CT's
  disk writer passes it straight to safetensors, which rejects it. A scoped adapter
  around compression makes only the current serialized tensor contiguous, using
  the original writer. Values, packing arithmetic, dtypes and shapes are unchanged;
  the writer is restored on exceptions. Installed packages are unchanged.
  Current upstream [DiskCache](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/offload/cache/disk.py)
  still delegates directly to `save_file`; the existing CT `patch_attr` supplies
  scope/restoration. No new packer, offload cache or checkpoint writer was built.
- The pre-save reference forward now uses `disable_offloading()`, as calibration
  does. CT's weight QDQ temporarily patches an onloaded tensor; disk-backed reads
  must reuse that tensor for the reference to actually apply FP8 weight QDQ.
  The original bare forward was not a valid quantized disk reference. No numerical
  tolerance or target recipe was changed.

CPU verification: **10 passed** in 33.67 seconds (complete oneshot lifecycle file
and packed-cache tests). Four save-coordination tests also passed in the prior
combined run. Intermediate failure logs and the passing log are retained in
[CPU disk evidence](../results/glm53-ep-gptq/20260909-cpu-disk/).

## Final bounded GPU attempt

Controller and worker: [retry 4](../results/glm53-ep-gptq/20260909-local-nccl-retry4/).
Detached tmux, top-level srun, one x86_64 node `xgpf11`, two full T4 GPUs, four CPU
cores, 16 GiB host memory, 15-minute cap, immediate placement. Exact revision is
recorded by the controller; source/config/tests must be clean before allocation.
The same three tests retain the original 60-second process-group timeout,
240-second per-case process deadline, numerical tolerances and save/reload checks.
Passed-test stdout is retained too. No full checkpoint or corpus is loaded.

This tests small-model NCCL correctness and persistence on T4. It does not measure
representative H100 memory, runtime or full-model quantization quality. No full
H100/H200 pair was available during the resource check; MIG devices were excluded.

Job **830817**, revision **2b51c7fe**, completed with pytest/worker/controller exit
codes all zero: **3 passed**, no skips, in 119.83 seconds. Scheduler elapsed time
was 3 minutes 1 second including startup/preflight. Both ranks reached packed save
and reload-forward completion in every case. The pure-GPTQ fixture compared 21
packed tensors per rank and recorded zero differing packed words; this is not a
bitwise-equivalence guarantee for larger models or other hardware.

The [raw pytest log](../results/glm53-ep-gptq/20260909-local-nccl-retry4/pytest.log),
[scheduler accounting](../results/glm53-ep-gptq/20260909-local-nccl-retry4/sacct.txt),
[environment](../results/glm53-ep-gptq/20260909-local-nccl-retry4/environment.json),
and [per-rank markers](../results/glm53-ep-gptq/20260909-local-nccl-retry4/artifacts/)
are retained. All task GPU allocations ended; no queued retry remains.

## Next local gate

The owner authorized queuing the same tiny suite on two full H100s. See the
[H100 queue contract](glm53-ep-gptq-h100-queue.md) and the
[shared three-objective brief](../GLM53_QUANTIZATION_OBJECTIVES.md).
