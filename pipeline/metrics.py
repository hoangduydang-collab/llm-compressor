"""Capture llm-compressor's in-process quantization metrics per run.

llm-compressor logs internal metrics through loguru. They arrive two ways:

  - GPTQ: native ``Quantizing <module>`` work records plus per-module
    reconstruction ``error`` and ``time`` at the ``METRIC`` level.
  - AWQ: a structured per-mapping summary at ``DEBUG`` level, in a single line
    ``"AWQ per-mapping error metrics: {... 'metrics': [{'best_error':..,
    'reduction':..}, ...]}"`` (AWQ's live ``best_error`` is only a tqdm postfix,
    which never reaches loguru, so we rely on this end-of-run summary instead).

This module tees native metrics and structured phase records to
``<run_dir>/quant_metrics.jsonl`` from load through save
and summarizes them (GPTQ error/time distribution; AWQ best_error/reduction
distribution) for ``metadata.json``.
"""

import ast
import json
import re
import statistics
from contextlib import contextmanager
from pathlib import Path

# GPTQ logs messages like "error 1724.52" and "time 0.49s".
_ERROR_RE = re.compile(r"\berror\s+([0-9.eE+\-]+)")
_TIME_RE = re.compile(r"\btime\s+([0-9.]+)\s*s\b")
_GPTQ_QUANTIZING_RE = re.compile(r"^Quantizing\s+(\S+)\s+using\s+\d+\s+samples$")
_DECODER_LAYER_RE = re.compile(
    r"^(?:model[.]language_model|language_model|model)[.]layers[.](\d+)(?:[.]|$)"
)

# AWQ logs a single structured DEBUG line with this prefix.
_AWQ_PREFIX = "AWQ per-mapping error metrics:"


def _is_captured_message(record) -> bool:
    """Keep native GPTQ work/metric records and AWQ structured metrics."""
    if record.get("extra", {}).get("event") in {"phase_start", "phase_end"}:
        return True
    if record["level"].name == "METRIC":
        return True
    message = str(record["message"])
    return message.startswith(_AWQ_PREFIX) or bool(_GPTQ_QUANTIZING_RE.match(message))


@contextmanager
def capture_quant_metrics(path):
    """Tee native GPTQ/AWQ work and metric records to ``path`` (JSON lines).

    Adds an extra sink (does not disturb existing console logging) and removes
    it on exit. The sink level is DEBUG so the AWQ line passes the level gate,
    but the content filter keeps the file to just the metric records.

    The sink is registered through llm-compressor's external-sink registry so it
    survives ``configure_logger()`` resets: on distributed runs ``oneshot`` calls
    ``configure_distributed_logger()`` internally, whose ``logger.remove()``
    would otherwise silently disconnect this sink and leave the evidence file
    empty (observed in the r6 distributed smoke: native GPTQ METRIC records on
    stdout, zero records captured).
    """
    from loguru import logger as _loguru

    from llmcompressor.logger import add_external_sink, remove_external_sink

    path = Path(path)
    try:
        _loguru.level("METRIC")
    except ValueError:
        _loguru.level("METRIC", no=38)

    sink_id = add_external_sink(
        str(path),
        level="DEBUG",
        serialize=True,
        filter=_is_captured_message,
    )
    try:
        yield path
    finally:
        # idempotent: survives llm-compressor replacing/removing handlers
        remove_external_sink(sink_id)


def _iter_records(path: Path, *, diagnostics: dict | None = None):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if diagnostics is not None:
                    diagnostics["malformed_lines"] = (
                        diagnostics.get("malformed_lines", 0) + 1
                    )
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("record", {}), dict):
                if diagnostics is not None:
                    diagnostics["malformed_lines"] = (
                        diagnostics.get("malformed_lines", 0) + 1
                    )
                continue
            yield obj.get("record", {})


def _iter_messages(path: Path):
    for record in _iter_records(path):
        yield record.get("message", "")


def _union_duration(intervals):
    """Count overlapping/nested wall intervals once, on one rank's clock."""
    end = None
    total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total / 1e9


def summarize_phases(paths) -> dict:
    """Summarize paired evidence, without summing nested spans as elapsed time.

    Critical-path elapsed is the maximum observed per-node wall envelope,
    including staggered starts across ranks on the same node. It includes gaps
    and waits. Evidence is complete only with the declared world-size rank
    coverage and every span enclosed by a paired quantize_run root; successful
    work status is separate. Monotonic clocks are never compared across nodes.
    Summed rank work unions intervals on each rank (including waits),
    so concurrent ranks contribute separately but nested phases do not.
    """
    starts, ends = {}, {}
    missing_paths = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            missing_paths.append(str(path))
            continue
        for record in _iter_records(path):
            extra = record.get("extra", {})
            event = extra.get("event")
            if event in {"phase_start", "phase_end"} and extra.get("span_id"):
                key = (extra.get("node"), extra.get("rank"), extra["span_id"])
                # Boundary snapshots remain in raw JSONL; retaining them all
                # here would duplicate potentially large full-run evidence.
                (starts if event == "phase_start" else ends)[key] = {
                    field: extra.get(field)
                    for field in (
                        "phase",
                        "timestamp_ns",
                        "parent_span_id",
                        "world_size",
                        "status",
                    )
                }
    if not starts and not ends:
        return {"available": False}

    ranks, nodes, phases = {}, {}, {}
    invalid = []
    valid_keys = set()
    for key in starts.keys() & ends.keys():
        start, end = starts[key], ends[key]
        lo, hi = start.get("timestamp_ns"), end.get("timestamp_ns")
        if (
            not isinstance(lo, int)
            or not isinstance(hi, int)
            or hi < lo
            or end.get("phase") != start.get("phase")
            or end.get("parent_span_id") != start.get("parent_span_id")
        ):
            invalid.append(key[2])
            continue
        valid_keys.add(key)
        rank = f"{key[0]}:{key[1]}"
        interval = (lo, hi)
        ranks.setdefault(rank, []).append(interval)
        nodes.setdefault(key[0], []).append(interval)
        phase = phases.setdefault(
            start["phase"],
            {
                "completed_spans": 0,
                "failed_spans": 0,
                "rank_intervals": {},
            },
        )
        phase["completed_spans"] += 1
        phase["failed_spans"] += int(end.get("status") == "error")
        phase["rank_intervals"].setdefault(rank, []).append(interval)

    for phase in phases.values():
        durations = [_union_duration(v) for v in phase.pop("rank_intervals").values()]
        phase["summed_rank_work_s"] = sum(durations)
        phase["max_rank_observed_s"] = max(durations)

    def envelope(intervals):
        return (max(hi for _, hi in intervals) - min(lo for lo, _ in intervals)) / 1e9

    rank_elapsed = [envelope(intervals) for intervals in ranks.values()]
    node_elapsed = [envelope(intervals) for intervals in nodes.values()]
    world_sizes = [r.get("world_size") for r in (*starts.values(), *ends.values())]
    world_size = (
        world_sizes[0]
        if all(
            isinstance(size, int) and not isinstance(size, bool) and size > 0
            for size in world_sizes
        )
        and len(set(world_sizes)) == 1
        else None
    )
    observed = {key[1] for key in valid_keys}
    expected = set(range(world_size)) if world_size is not None else set()
    roots = {
        key
        for key in valid_keys
        if starts[key].get("phase") == "quantize_run"
        and starts[key].get("parent_span_id") is None
    }
    missing_parents = set()
    unenclosed = []
    for key in valid_keys:
        ancestor = key
        visited = set()
        while ancestor not in roots:
            if ancestor not in valid_keys or ancestor in visited:
                break
            visited.add(ancestor)
            parent = starts[ancestor].get("parent_span_id")
            if parent is None:
                break
            ancestor = (*key[:2], parent)
            if ancestor not in valid_keys:
                missing_parents.add(parent)
        if ancestor not in roots:
            unenclosed.append(key[2])
    paired_complete = starts.keys() == ends.keys() and not invalid
    complete = (
        paired_complete
        and not missing_paths
        and world_size is not None
        and observed == expected
        and not unenclosed
        and {key[1] for key in roots} == expected
    )
    return {
        "available": True,
        "complete": complete,
        "paired_complete": paired_complete,
        "expected_world_size": world_size,
        "missing_ranks": sorted(expected - observed),
        "unexpected_ranks": sorted(observed - expected) if world_size else [],
        "missing_run_ranks": sorted(expected - {key[1] for key in roots}),
        "missing_parent_spans": sorted(missing_parents),
        "unenclosed_spans": sorted(unenclosed),
        "missing_paths": missing_paths,
        "unmatched_starts": sorted(k[2] for k in starts.keys() - ends.keys()),
        "unmatched_ends": sorted(k[2] for k in ends.keys() - starts.keys()),
        "invalid_spans": sorted(invalid),
        "observed_ranks": sorted(ranks),
        "critical_path_elapsed_s": max(node_elapsed) if node_elapsed else None,
        "max_rank_elapsed_s": max(rank_elapsed) if rank_elapsed else None,
        "summed_rank_work_s": (
            sum(_union_duration(v) for v in ranks.values()) if ranks else None
        ),
        "scope": "observed paired spans only; critical path is the maximum "
        "node envelope, not synchronized multi-node elapsed; wall time includes waits; "
        "phase totals overlap and must not be added; "
        "process I/O is not unique filesystem traffic",
        "phases": phases,
    }


def _decoder_layer(name: str) -> int | None:
    match = _DECODER_LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def summarize_quantized_layers(paths, *, method: str) -> dict:
    """Summarize native GPTQ/AWQ work records by decoder layer.

    GPTQ contributes its existing ``Quantizing <module>`` INFO records; AWQ
    contributes its existing structured per-mapping summary. This is passive
    log parsing and does not install runtime hooks or alter quantization.
    """
    if method not in {"gptq", "awq"}:
        raise ValueError(f"unsupported quantization method: {method!r}")

    names: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for message in _iter_messages(path):
            if method == "gptq":
                match = _GPTQ_QUANTIZING_RE.match(message)
                if match:
                    names.append(match.group(1))
            else:
                parsed = _parse_awq_metrics(message)
                if parsed is not None:
                    names.extend(
                        str(item["layer_name"])
                        for item in parsed
                        if isinstance(item, dict) and item.get("layer_name")
                    )

    layers: set[int] = set()
    unresolved: set[str] = set()
    for name in names:
        layer = _decoder_layer(name)
        if layer is None:
            unresolved.add(name)
        else:
            layers.add(layer)
    return {
        "method": method,
        "record_count": len(names),
        "layers": sorted(layers),
        "unresolved_names": sorted(unresolved),
    }


def _dist(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
        "min": min(values),
    }


def _parse_awq_metrics(msg: str):
    """Parse the AWQ structured-metrics line into a list of per-layer dicts."""
    if not msg.startswith(_AWQ_PREFIX):
        return None
    payload = msg[len(_AWQ_PREFIX) :].strip()
    try:
        data = ast.literal_eval(payload)
    except (ValueError, SyntaxError):
        return None
    if isinstance(data, dict):
        return data.get("metrics", [])
    return None


def summarize_quant_metrics(path) -> dict:
    """Parse the captured JSONL into a compact summary dict."""
    path = Path(path)
    if not path.exists():
        return {"available": False}

    errors: list[float] = []  # GPTQ per-module error
    times: list[float] = []  # GPTQ per-module time
    awq_metrics: list[dict] = []  # AWQ per-mapping metrics

    for msg in _iter_messages(path):
        parsed = _parse_awq_metrics(msg)
        if parsed is not None:
            awq_metrics = parsed  # one summary line per run; last wins
            continue
        m = _ERROR_RE.search(msg)
        if m:
            try:
                errors.append(float(m.group(1)))
            except ValueError:
                pass
        t = _TIME_RE.search(msg)
        if t:
            try:
                times.append(float(t.group(1)))
            except ValueError:
                pass

    summary: dict = {"available": True, "num_module_errors": len(errors)}
    if errors:
        summary["error"] = _dist(errors)
    if times:
        summary["time_s"] = {
            "total": sum(times),
            "mean": statistics.fmean(times),
            "count": len(times),
        }

    if awq_metrics:
        best_errors = [
            float(m["best_error"])
            for m in awq_metrics
            if isinstance(m.get("best_error"), (int, float))
        ]
        reductions = [
            float(m["reduction"])
            for m in awq_metrics
            if isinstance(m.get("reduction"), (int, float))
        ]
        awq_summary: dict = {"num_layers": len(awq_metrics)}
        if best_errors:
            awq_summary["best_error"] = _dist(best_errors)
        if reductions:
            awq_summary["reduction"] = _dist(reductions)
        summary["awq"] = awq_summary

    summary["phases"] = summarize_phases([path])
    return summary
