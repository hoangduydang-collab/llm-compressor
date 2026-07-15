"""Capture llm-compressor's in-process quantization metrics per run.

llm-compressor logs internal metrics through loguru. They arrive two ways:

  - GPTQ: native ``Quantizing <module>`` work records plus per-module
    reconstruction ``error`` and ``time`` at the ``METRIC`` level.
  - AWQ: a structured per-mapping summary at ``DEBUG`` level, in a single line
    ``"AWQ per-mapping error metrics: {... 'metrics': [{'best_error':..,
    'reduction':..}, ...]}"`` (AWQ's live ``best_error`` is only a tqdm postfix,
    which never reaches loguru, so we rely on this end-of-run summary instead).

This module tees both to ``<run_dir>/quant_metrics.jsonl`` during calibration
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
_DECODER_LAYER_RE = re.compile(r"(?:^|[.])language_model[.]layers[.](\d+)(?:[.]|$)")

# AWQ logs a single structured DEBUG line with this prefix.
_AWQ_PREFIX = "AWQ per-mapping error metrics:"


def _is_captured_message(record) -> bool:
    """Keep native GPTQ work/metric records and AWQ structured metrics."""
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
    """
    from loguru import logger as _loguru

    path = Path(path)
    try:
        _loguru.level("METRIC")
    except ValueError:
        _loguru.level("METRIC", no=38)

    sink_id = _loguru.add(
        str(path),
        level="DEBUG",
        serialize=True,
        filter=_is_captured_message,
    )
    try:
        yield path
    finally:
        try:
            _loguru.remove(sink_id)
        except ValueError:
            # llm-compressor may replace/remove Loguru handlers internally.
            # Cleanup of this passive evidence sink must therefore be idempotent.
            pass


def _iter_messages(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj.get("record", {}).get("message", "")


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

    return summary
