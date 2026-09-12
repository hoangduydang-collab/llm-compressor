# CPU timing evidence

These measurements cover the observer and offline report, not GLM quantization.

- `overhead.json`: final raw trial timings and medians; `overhead-initial.json`
  preserves the earlier measurement before snapshot object compatibility changed.
- `benchmark_overhead.py`: reproducible standalone benchmark.
- `benchmark-manifest.json`: command, scope and decompressed raw sample hash.
- `lightweight-sample.jsonl.gz`: one complete measured trial, including its root
  span and 2,000 lightweight child spans (4,002 boundaries).
- `example-report.json` and `example-report.md`: report generated from that trial.
- `report-generation.json`: aggregation runtime measured separately.

Reconstruct the raw file for offline inspection:

```bash
gzip -dc results/glm53-timing/20260912-cpu/lightweight-sample.jsonl.gz > /tmp/glm53-timing-sample.jsonl
python -m pipeline.timing_report /tmp/glm53-timing-sample.jsonl --output-prefix /tmp/glm53-timing-example
```

Raw logs retain the original host/rank identity. The absolute source paths in the
example identify the measurement invocation; the compressed sample above makes
the input reproducible after temporary files are cleaned up.

`validation.json` records exact source hashes and the 56 distinct passing tests.
The final focused/integration stdout and JUnit are preserved. The earlier
integration teardown error came from a simultaneous documentation commit
triggering the repository file-count guard; the identical suite passed after
concurrent writes stopped. `operation-order.json` and its script record the
quantization-work AST comparison.
