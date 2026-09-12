"""Offline reports for structured quantization phase JSONL evidence."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import sys
from pathlib import Path

from pipeline.metrics import (
    _decoder_layer,
    _iter_records,
    _union_duration,
    summarize_phases,
)

_IO_COUNTERS = ("read_bytes", "write_bytes", "rchar", "wchar")
_GPU_GAUGES = (
    "allocated_bytes",
    "reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)


def _compact_snapshot(snapshot):
    if not isinstance(snapshot, dict) or not snapshot:
        return None
    process = snapshot.get("process", {})
    memory = process.get("memory", {})
    io = process.get("io", {})
    gpu = snapshot.get("gpu", {})
    return {
        "process": {
            "scope": process.get("scope"),
            "memory": {
                name: memory.get(name) for name in ("rss_bytes", "peak_rss_bytes")
            },
            "io": {
                "available": io.get("available", False),
                "counters": {
                    name: io.get("counters", {}).get(name) for name in _IO_COUNTERS
                },
            },
        },
        "gpu": {
            "available": gpu.get("available", False),
            **{name: gpu.get(name) for name in _GPU_GAUGES},
        },
    }


def _compact_boundary(extra):
    return {
        name: extra.get(name)
        for name in (
            "phase",
            "timestamp_ns",
            "parent_span_id",
            "status",
            "error_type",
            "identity",
        )
    } | {"snapshot": _compact_snapshot(extra.get("snapshot"))}


def _rank_label(key):
    return f"{key[0]}:{key[1]}"


def _snapshot_observation(start, end):
    before, after = start.get("snapshot"), end.get("snapshot")
    if not before or not after:
        return {"available": False, "reason": "boundary snapshot unavailable"}
    bp, ap = before["process"], after["process"]
    result = {
        "process_io_scope": bp.get("scope") or ap.get("scope"),
        "process_io_delta": {},
        "rss": {},
        "gpu_allocator": {
            "available": bool(
                before["gpu"].get("available") and after["gpu"].get("available")
            )
        },
    }
    for name in _IO_COUNTERS:
        lo = bp["io"]["counters"].get(name) if bp["io"]["available"] else None
        hi = ap["io"]["counters"].get(name) if ap["io"]["available"] else None
        result["process_io_delta"][name] = (
            hi - lo
            if isinstance(lo, int) and isinstance(hi, int) and hi >= lo
            else None
        )
    for name in ("rss_bytes", "peak_rss_bytes"):
        result["rss"][name] = {
            "start": bp["memory"].get(name),
            "end": ap["memory"].get(name),
        }
    for name in _GPU_GAUGES:
        result["gpu_allocator"][name] = {
            "start": before["gpu"].get(name),
            "end": after["gpu"].get(name),
        }
    result["available"] = (
        any(value is not None for value in result["process_io_delta"].values())
        or any(
            pair["start"] is not None or pair["end"] is not None
            for pair in result["rss"].values()
        )
        or result["gpu_allocator"]["available"]
    )
    if not result["available"]:
        result["reason"] = "resource counters unavailable"
    return result


def _duration_stats(rank_durations):
    values = list(rank_durations.values())
    if not values:
        return {}
    slowest = max(rank_durations, key=rank_durations.get)
    return {
        "observed_rank_count": len(values),
        "slowest_rank": slowest,
        "min_duration_s": min(values),
        "median_duration_s": statistics.median(values),
        "max_duration_s": max(values),
        "spread_s": max(values) - min(values),
    }


def _finish_group(group):
    intervals = group.pop("rank_intervals")
    rank_durations = {rank: _union_duration(items) for rank, items in intervals.items()}
    group["rank_durations_s"] = rank_durations
    group.update(_duration_stats(rank_durations))
    group["status"] = "failed" if group.get("failed_spans") else "complete"
    return group


def build_timing_report(paths) -> dict:
    """Build a compact report from explicit rank logs without importing torch."""
    source_paths = [str(Path(path)) for path in paths]
    canonical = summarize_phases(source_paths)
    starts, ends = {}, {}
    parser_diagnostics = {}
    for raw_path in source_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for record in _iter_records(path, diagnostics=parser_diagnostics):
            extra = record.get("extra", {})
            event, span_id = extra.get("event"), extra.get("span_id")
            if event not in {"phase_start", "phase_end"} or not span_id:
                continue
            key = (extra.get("node"), extra.get("rank"), span_id)
            (starts if event == "phase_start" else ends).setdefault(key, []).append(
                _compact_boundary(extra)
            )

    duplicate_boundaries = sorted(
        key[2]
        for key in starts.keys() | ends.keys()
        if len(starts.get(key, [])) > 1 or len(ends.get(key, [])) > 1
    )
    run_counts = {}
    for key, boundaries in starts.items():
        if boundaries and boundaries[0]["phase"] == "quantize_run":
            rank = _rank_label(key)
            run_counts[rank] = run_counts.get(rank, 0) + len(boundaries)
    multiple_runs = {rank: count for rank, count in run_counts.items() if count != 1}

    groups, layer_rollups, phase_intervals = {}, {}, {}
    resource_observations = []
    for key in starts.keys() & ends.keys():
        if len(starts[key]) != 1 or len(ends[key]) != 1:
            continue
        start, end = starts[key][0], ends[key][0]
        lo, hi = start["timestamp_ns"], end["timestamp_ns"]
        if (
            not isinstance(lo, int)
            or not isinstance(hi, int)
            or hi < lo
            or start["phase"] != end["phase"]
            or start["parent_span_id"] != end["parent_span_id"]
        ):
            continue
        identity = start["identity"] if isinstance(start["identity"], dict) else {}
        rank, interval, phase = _rank_label(key), (lo, hi), start["phase"]
        phase_intervals.setdefault(phase, {}).setdefault(rank, []).append(interval)
        identity_json = json.dumps(identity, sort_keys=True, default=str)
        group = groups.setdefault(
            (phase, identity_json),
            {
                "phase": phase,
                "identity": identity,
                "decoder_layer": _decoder_layer(str(identity.get("module", ""))),
                "rank_intervals": {},
                "failed_spans": 0,
                "completed_spans": 0,
            },
        )
        group["rank_intervals"].setdefault(rank, []).append(interval)
        group["completed_spans"] += 1
        group["failed_spans"] += int(end["status"] == "error")
        if group["decoder_layer"] is not None:
            rollup = layer_rollups.setdefault(
                (phase, group["decoder_layer"]),
                {
                    "phase": phase,
                    "decoder_layer": group["decoder_layer"],
                    "rank_intervals": {},
                    "failed_spans": 0,
                    "completed_spans": 0,
                },
            )
            rollup["rank_intervals"].setdefault(rank, []).append(interval)
            rollup["completed_spans"] += 1
            rollup["failed_spans"] += int(end["status"] == "error")
        observation = _snapshot_observation(start, end)
        if observation["available"]:
            resource_observations.append(
                {"phase": phase, "identity": identity, "rank": rank, **observation}
            )

    rendered_groups = [_finish_group(group) for group in groups.values()]
    rendered_groups.sort(
        key=lambda group: (
            str(group["phase"]),
            json.dumps(group["identity"], sort_keys=True),
        )
    )
    rendered_rollups = [_finish_group(group) for group in layer_rollups.values()]
    rendered_rollups.sort(
        key=lambda group: (str(group["phase"]), group["decoder_layer"])
    )
    phase_rows = []
    for phase, info in sorted(canonical.get("phases", {}).items()):
        rank_durations = {
            rank: _union_duration(items)
            for rank, items in phase_intervals.get(phase, {}).items()
        }
        phase_rows.append({"phase": phase, **_duration_stats(rank_durations), **info})
    detail_groups = [
        group
        for group in rendered_groups
        if any(key in group["identity"] for key in ("module", "subgraph"))
    ]
    slow_groups = sorted(
        detail_groups + rendered_rollups,
        key=lambda group: group.get("max_duration_s", 0),
        reverse=True,
    )[:20]

    ambiguities = {
        "duplicate_span_boundaries": duplicate_boundaries,
        "multiple_or_missing_run_roots_by_observed_rank": multiple_runs,
        "malformed_or_truncated_lines": parser_diagnostics.get("malformed_lines", 0),
    }
    evidence_complete = bool(canonical.get("complete")) and not any(
        (
            duplicate_boundaries,
            multiple_runs,
            ambiguities["malformed_or_truncated_lines"],
        )
    )
    return {
        "schema_version": 1,
        "source_paths": source_paths,
        "evidence_complete": evidence_complete,
        "automatic_report_note": "A source-rank report may be partial until regenerated after the launcher exits.",  # noqa: E501
        "timing_scope": "Rank durations include waits. Nested phases and rank work must not be added as elapsed. Clock values are compared only within a node. Rank imbalance does not identify collective latency.",  # noqa: E501
        "resource_scope": "Process I/O is not unique shared-storage traffic. Missing or reset counters are unavailable; RSS and GPU allocator values are boundary gauges.",  # noqa: E501
        "summary": canonical,
        "ambiguities": ambiguities,
        "phases": phase_rows,
        "groups": rendered_groups,
        "decoder_layer_rollups": rendered_rollups,
        "slow_groups": slow_groups,
        "resource_observations": resource_observations,
    }


def _fmt(value):
    return "unavailable" if value is None else f"{value:.6f}"


def _markdown(report):
    state = "complete" if report["evidence_complete"] else "incomplete or ambiguous"
    lines = [
        "# Quantization timing report",
        "",
        f"Evidence status: **{state}**",
        "",
        report["automatic_report_note"],
        "",
        report["timing_scope"],
        "",
        report["resource_scope"],
        "",
        "## Source evidence",
        "",
    ]
    lines.extend(f"- `{path}`" for path in report["source_paths"])
    lines += [
        "",
        "## Phase totals",
        "",
        "| Phase | Ranks | Slowest rank | Min (s) | Median (s) | Max (s) | Spread (s) |",  # noqa: E501
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for phase in report["phases"]:
        lines.append(
            f"| {phase['phase']} | {phase.get('observed_rank_count', 0)} | {phase.get('slowest_rank', 'unavailable')} | {_fmt(phase.get('min_duration_s'))} | {_fmt(phase.get('median_duration_s'))} | {_fmt(phase.get('max_duration_s'))} | {_fmt(phase.get('spread_s'))} |"  # noqa: E501
        )
    lines += [
        "",
        "## Per-layer and subgraph groups",
        "",
        "| Phase | Identity | Decoder layer | Ranks | Max (s) | Spread (s) | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    displayed_groups = [
        group for group in report["groups"] if "subgraph" in group["identity"]
    ] + report["decoder_layer_rollups"]
    for group in displayed_groups:
        identity = (
            "decoder-layer rollup"
            if "decoder_layer" in group and "identity" not in group
            else json.dumps(group["identity"], sort_keys=True)
        )
        identity = identity.replace("|", "\\|")
        layer = group.get("decoder_layer")
        layer_display = layer if layer is not None else "n/a"
        rank_count = group.get("observed_rank_count", 0)
        max_duration = _fmt(group.get("max_duration_s"))
        lines.append(
            f"| {group['phase']} | `{identity}` | {layer_display} | "
            f"{rank_count} | {max_duration} | "
            f"{_fmt(group.get('spread_s'))} | {group['status']} |"
        )
    lines += ["", "## Slowest layer and subgraph groups", ""]
    for group in report["slow_groups"]:
        identity = group.get("identity", {"decoder_layer": group.get("decoder_layer")})
        lines.append(
            f"- `{group['phase']}` `{json.dumps(identity, sort_keys=True)}`: "
            f"{_fmt(group.get('max_duration_s'))} s max, "
            f"{_fmt(group.get('spread_s'))} s spread"
        )
    lines += ["", "## Incomplete, failed, or ambiguous evidence", ""]
    summary = report["summary"]
    issues = {
        "missing paths": summary.get("missing_paths", []),
        "missing ranks": summary.get("missing_ranks", []),
        "open spans": summary.get("unmatched_starts", []),
        "orphan ends": summary.get("unmatched_ends", []),
        "invalid spans": summary.get("invalid_spans", []),
        "duplicate boundaries": report["ambiguities"]["duplicate_span_boundaries"],
        "multiple/missing roots": report["ambiguities"][
            "multiple_or_missing_run_roots_by_observed_rank"
        ],
        "malformed/truncated lines": report["ambiguities"][
            "malformed_or_truncated_lines"
        ],
        "failed spans": sum(group["failed_spans"] for group in report["groups"]),
    }
    lines.extend(
        f"- {name}: `{json.dumps(value, sort_keys=True)}`"
        for name, value in issues.items()
    )
    lines += ["", "## Resource observations", ""]
    if report["resource_observations"]:
        shown = report["resource_observations"][:20]
        total_observations = len(report["resource_observations"])
        lines.append(
            f"Showing {len(shown)} of {total_observations} observations; "
            "the JSON report and raw sources retain the complete evidence."
        )
        lines.extend(
            f"- `{item['phase']}` {item['rank']} "
            f"{json.dumps({k: v for k, v in item.items() if k not in {'phase', 'rank'}}, sort_keys=True)}"  # noqa: E501
            for item in shown
        )
    else:
        lines.append("- No usable paired boundary resource observations were present.")  # noqa: E501
    return "\n".join(lines) + "\n"


def write_timing_report(paths, output_prefix) -> dict:
    report = build_timing_report(paths)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args(argv)
    report = write_timing_report(args.logs, args.output_prefix)
    print(
        f"wrote {args.output_prefix.with_suffix('.json')} and {args.output_prefix.with_suffix('.md')} ({'complete' if report['evidence_complete'] else 'incomplete'})"  # noqa: E501
    )
    return 0


def write_automatic_timing_report(
    paths, output_prefix, max_input_bytes=64 * 1024 * 1024
) -> dict:
    """Bound source-rank aggregation cost; leave large logs for the offline CLI."""
    source_paths = [str(Path(path)) for path in paths]
    sizes = {}
    total_bytes = 0
    for raw_path in source_paths:
        try:
            size = Path(raw_path).stat().st_size
        except OSError:
            sizes[raw_path] = None
        else:
            sizes[raw_path] = size
            total_bytes += size
    if total_bytes <= max_input_bytes:
        return write_timing_report(source_paths, output_prefix)

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "pipeline.timing_report",
            *source_paths,
            "--output-prefix",
            str(prefix),
        ]
    )
    report = {
        "schema_version": 1,
        "status": "deferred",
        "reason": "automatic input byte budget exceeded",
        "source_paths": source_paths,
        "input_sizes_bytes": sizes,
        "total_input_bytes": total_bytes,
        "max_input_bytes": max_input_bytes,
        "offline_command": command,
        "evidence_complete": False,
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(
        "# Quantization timing report\n\n"
        "Automatic aggregation was **deferred** because the observed logs "
        f"total {total_bytes} bytes, above the {max_input_bytes}-byte budget.\n\n"
        "Run after the launcher exits:\n\n"
        f"```sh\n{command}\n```\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    raise SystemExit(main())
