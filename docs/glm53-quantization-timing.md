# GLM-5.3 quantization timing

Objective 3 uses the existing structured phase logs to show where AWQ/GPTQ wall
time goes. The report is diagnostic evidence; it does not by itself establish
that a slow phase is caused by shared storage.

## Collection and overhead

Existing phase records already cover loading, tracing, calibration, propagation,
FP8 preparation, GPTQ reduction/solve/publication, save, MTP assembly and offline
verification. New timing boundaries cover sequential compression callbacks,
AWQ mapping searches, MTP preflight, checkpoint side artifacts and existing
barrier waits.

New boundaries use `compression_phase(..., collect_snapshot=False)`. They record
paired monotonic timestamps and identity without extra resource-counter reads.
They do not add CUDA synchronization, CUDA events, tensor copies, collectives,
barriers, per-token hooks or timers inside the AWQ grid-search loop. Existing
resource snapshots and quantization operations are unchanged.

Report aggregation runs after the source rank closes its quantization capture.
It reads logs and writes reports, never model/checkpoint tensors. Automatic
aggregation has a **64 MiB total input budget**. Larger runs write a small
deferred-report notice and the offline command without parsing log payloads,
so multi-gigabyte traces do not add a long report-generation tail to the job.
The offline CLI has no input-size cap. It scans the
expected rank paths once and never waits or polls for peers. Therefore the
automatic report may be partial if another rank is still finishing. Regenerate
it after the launcher exits for the final all-rank view. This also works for
failed or interrupted runs. A report-generation error is a warning and does not
replace a quantization failure or turn successful quantization into a failure.

For operator-level investigation of a phase that remains unexplained, reuse
[PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
on a bounded representative workload. Its optional shape/stack tracing adds
overhead; it is not enabled by this change.

## Generate the final report

From the repository root, set `run_dir` to an existing run directory. After the
quantization launcher terminates, use its existing Python environment:

```bash
python -m pipeline.timing_report "$run_dir"/quant_metrics.rank-*.jsonl \
  --output-prefix "$run_dir/timing_report"
```

For a single-process run:

```bash
python -m pipeline.timing_report "$run_dir/quant_metrics.jsonl" \
  --output-prefix "$run_dir/timing_report"
```

The reporter needs only the Python standard library. It writes JSON for analysis
and Markdown for review. Markdown emphasizes layer/subgraph rollups and the
slowest groups; detailed per-module records remain in JSON. Preserve the raw logs, launch configuration, dependency
versions, storage/offload paths, shard layout and prefetch settings alongside it.
Missing rank files and incomplete spans are evidence gaps, not zero durations.

## Interpret the report

- Per-phase rank durations identify the slowest observed rank and duration spread.
  Different ranks can own different amounts of work; spread is not a direct
  measurement of communication latency.
- Module identities identify decoder layers where names permit. Subgraph indices
  remain separate identities: tracing does not guarantee one subgraph per layer.
- Durations are host wall time including waits and asynchronous GPU launch time.
  They are not isolated GPU kernel execution time. Timed barrier calls include
  their actual wait without adding a new barrier.
- Concurrent rank work and nested phase durations overlap. Do not add phase rows
  to derive total runtime. The elapsed estimate uses observed node envelopes;
  monotonic timestamps from different nodes are never compared directly.
- Process I/O counters do not measure unique bytes from CephFS. RSS and GPU
  allocator observations are boundary samples, not per-phase peaks. Missing or
  reset counters remain unavailable. Raw logs retain the fuller cgroup/node
  observations for follow-up analysis.
- Failed spans can be paired and complete as evidence. Evidence completeness
  does not mean quantization succeeded.

## Local overhead validation

The reproducible benchmark is
`results/glm53-timing/20260912-cpu/benchmark_overhead.py`. It compares a context
manager baseline, new lightweight spans and existing snapshot spans with a real
Loguru JSONL sink, a warmup and repeated trials. Run it in the existing
quantization environment and choose a fresh output directory:

```bash
PYTHONPATH=src:. python results/glm53-timing/20260912-cpu/benchmark_overhead.py \
  --output-dir /tmp/glm53-timing-overhead
```

This is a CPU observer benchmark with buffered file writes, not a full GPU run
or a CephFS throughput benchmark. Local results are preserved beside the script in `overhead.json`. Five measured
trials after warmup gave median costs of **216.0 microseconds per lightweight
span** (two real JSONL records), **0.48 microseconds** for the baseline, and
**1,816.6 microseconds** for the existing snapshot span. An illustrative 100,000
added lightweight spans would cost **21.6 seconds**, or **0.060% of a 10-hour
run**, on this host. This extrapolation is not a measured full-run slowdown. Full-run overhead on Rancher remains to be checked
against the executor's actual phase evidence; no new GPU allocation is required
for report generation.

The local complete report took **0.23 seconds** for **4.47 MB / 4,002 boundary
records**, measured separately from the observer benchmark. The automatic size
budget follows this measured cost; it is a byte limit, not a hard wall-time
guarantee for arbitrary storage. Example JSON/Markdown and the compressed raw
benchmark log are preserved under `results/glm53-timing/20260912-cpu/`.

The operation-order check in `results/glm53-timing/20260912-cpu/` compares the
AWQ smoothing, sequential walk and quantization work functions against the
previous branch revision. Removing only timer wrappers leaves their work ASTs
identical, including call order. Runtime tests cover reporting integration and
exceptions separately.

Validation: **56 distinct CPU tests passed**, including real native AWQ export,
MTP pipeline ordering, sequential helpers, overlap/rank/partial-log reporting and
report-failure isolation. Raw logs and JUnit are in the evidence directory.
PyTorch emitted existing `torch.jit.script_method` deprecation warnings; final
runs had no failures or errors.
