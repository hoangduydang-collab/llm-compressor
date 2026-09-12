"""Measure observer overhead with a real JSONL sink; no model or GPU work.

Run from repo root with PYTHONPATH=src:. and the quantization Python environment.
The output directory determines the filesystem under test. Preserve raw samples.
"""

import argparse
import json
import platform
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

from loguru import logger

from llmcompressor.utils.metric_logging import compression_phase
from pipeline.metrics import capture_quant_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spans", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logger.remove()  # standalone process; no console formatting in timed loops
    results = {}
    for mode in ("baseline", "lightweight", "existing_snapshot"):
        count = args.spans if mode != "existing_snapshot" else min(args.spans, 200)
        samples = []
        for trial in range(args.trials + 1):
            path = args.output_dir / f"{mode}-{trial}.jsonl"
            with (
                capture_quant_metrics(path),
                compression_phase(
                    "quantize_run", collect_snapshot=False, benchmark=True
                ),
            ):
                started = time.perf_counter_ns()
                for _ in range(count):
                    context = (
                        nullcontext()
                        if mode == "baseline"
                        else compression_phase(
                            "awq_search",
                            collect_snapshot=mode == "existing_snapshot",
                            module="model.layers.3.mlp.experts.0.up_proj",
                        )
                    )
                    with context:
                        pass
                elapsed = (time.perf_counter_ns() - started) / 1e9
            samples.append(
                {
                    "trial": trial,
                    "warmup": trial == 0,
                    "spans": count,
                    "elapsed_s": elapsed,
                    "us_per_span": elapsed * 1e6 / count,
                    "jsonl_bytes": path.stat().st_size,
                }
            )
        results[mode] = {
            "samples": samples,
            "median_us_per_span": statistics.median(
                sample["us_per_span"] for sample in samples if not sample["warmup"]
            ),
        }
    extra_us = max(
        0,
        results["lightweight"]["median_us_per_span"]
        - results["baseline"]["median_us_per_span"],
    )
    report = {
        "scope": "CPU observer only; no CUDA initialization, model or GPU work; "
        "real Loguru JSONL sink, buffered writes without fsync; "
        "not a shared-storage or full-quantization benchmark",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "output_dir": str(args.output_dir.resolve()),
        "results": results,
        "lightweight_incremental_us_per_span": extra_us,
        "illustrative_100000_added_spans_s": extra_us * 100000 / 1e6,
        "illustrative_fraction_of_10h_percent": extra_us * 100000 / 1e6 / 36000 * 100,
    }
    (args.output_dir / "overhead.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
