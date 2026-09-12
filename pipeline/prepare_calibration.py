"""Prepare deterministic, CPU-tokenized mixed calibration bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import random
import shutil
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

import yaml

from pipeline.calibration_bundle import (
    BUNDLE_SCHEMA_VERSION,
    DATA_FILENAME,
    MANIFEST_FILENAME,
    sha256_file,
    tokenizer_identity,
)
from pipeline.swe_chat import (
    NormalizedSweSession,
    SweAssistantAnchor,
    normalize_swe_chat_session,
)

_FORMATS = {"messages", "text", "swe_chat"}
_TOP_LEVEL_KEYS = {"num_samples", "max_seq_length", "seed", "sources"}


class _IneligibleDocument(Exception):
    """A valid source item that cannot provide one full useful window."""


_SOURCE_KEYS = {
    "name",
    "weight",
    "dataset_id",
    "dataset_split",
    "dataset_config_name",
    "dataset_revision",
    "dataset_data_files",
    "format",
    "column",
    "id_column",
}


@dataclass(frozen=True)
class SourceConfig:
    name: str
    weight: float
    dataset_id: str
    dataset_split: str = "train"
    dataset_config_name: str | None = None
    dataset_revision: str | None = None
    dataset_data_files: Any = None
    format: str = "messages"
    column: str | None = None
    id_column: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("source name must be a non-empty string")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise ValueError(f"source {self.name!r} weight must be numeric")
        if not math.isfinite(float(self.weight)) or self.weight <= 0:
            raise ValueError(f"source {self.name!r} weight must be positive and finite")
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError(
                f"source {self.name!r} dataset_id must be a non-empty string"
            )
        if not isinstance(self.dataset_split, str) or not self.dataset_split:
            raise ValueError(
                f"source {self.name!r} dataset_split must be a non-empty string"
            )
        for field_name in ("dataset_config_name", "dataset_revision"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(
                    f"source {self.name!r} {field_name} must be a non-empty string"
                )
        if self.dataset_data_files is not None and not isinstance(
            self.dataset_data_files, (str, list, dict)
        ):
            raise ValueError(
                f"source {self.name!r} dataset_data_files must be a string, "
                "list, or mapping"
            )
        if self.format not in _FORMATS:
            raise ValueError(
                f"source {self.name!r} format must be one of {sorted(_FORMATS)}"
            )
        if self.column is None:
            object.__setattr__(
                self, "column", "text" if self.format == "text" else "messages"
            )
        if not isinstance(self.column, str) or not self.column:
            raise ValueError(f"source {self.name!r} column must be a non-empty string")
        if self.id_column is None and self.format == "swe_chat":
            object.__setattr__(self, "id_column", "session_id")
        if self.id_column is not None and (
            not isinstance(self.id_column, str) or not self.id_column
        ):
            raise ValueError(
                f"source {self.name!r} id_column must be a non-empty string"
            )
        if (
            self.format == "swe_chat"
            and self.dataset_id == "SALT-NLP/SWE-chat"
            and self.dataset_config_name != "conversations"
        ):
            raise ValueError(
                f"SWE-chat source {self.name!r} must explicitly set "
                "dataset_config_name: conversations"
            )


@dataclass(frozen=True)
class MixConfig:
    num_samples: int
    max_seq_length: int
    seed: int
    sources: tuple[SourceConfig, ...]

    def __post_init__(self) -> None:
        for field_name in ("num_samples", "max_seq_length", "seed"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        object.__setattr__(self, "sources", tuple(self.sources))
        if not self.sources:
            raise ValueError("sources must contain at least one source")
        if any(not isinstance(source, SourceConfig) for source in self.sources):
            raise ValueError("every source must be a SourceConfig")
        names = [source.name for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def load_mix_config(path: str | Path) -> MixConfig:
    """Read and strictly validate a mixed-calibration YAML file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw = _mapping(raw, "mix config")
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown keys in mix config: {sorted(unknown)}")
    missing = _TOP_LEVEL_KEYS - set(raw)
    if missing:
        raise ValueError(f"missing keys in mix config: {sorted(missing)}")
    raw_sources = raw["sources"]
    if not isinstance(raw_sources, list):
        raise ValueError("sources must be a list")
    sources = []
    for index, source_value in enumerate(raw_sources):
        source = _mapping(source_value, f"source {index}")
        unknown = set(source) - _SOURCE_KEYS
        if unknown:
            raise ValueError(f"unknown keys in source {index}: {sorted(unknown)}")
        missing = {"name", "weight", "dataset_id"} - set(source)
        if missing:
            raise ValueError(f"missing keys in source {index}: {sorted(missing)}")
        sources.append(SourceConfig(**source))
    return MixConfig(
        num_samples=raw["num_samples"],
        max_seq_length=raw["max_seq_length"],
        seed=raw["seed"],
        sources=tuple(sources),
    )


def allocate_quotas(
    num_samples: int, sources: list[SourceConfig] | tuple[SourceConfig, ...]
) -> list[int]:
    """Allocate an exact sample budget using deterministic largest remainder."""
    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, int)
        or num_samples <= 0
    ):
        raise ValueError("num_samples must be a positive integer")
    if not sources:
        raise ValueError("sources must not be empty")
    weights = [Decimal(str(source.weight)) for source in sources]
    total = sum(weights)
    exact = [Decimal(num_samples) * weight / total for weight in weights]
    quotas = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in exact]
    remaining = num_samples - sum(quotas)
    order = sorted(
        range(len(sources)), key=lambda index: (-(exact[index] - quotas[index]), index)
    )
    for index in order[:remaining]:
        quotas[index] += 1
    starved = [sources[index].name for index, quota in enumerate(quotas) if quota == 0]
    if starved:
        raise ValueError(
            "positive-weight sources rounded to zero windows: " + ", ".join(starved)
        )
    return quotas


def _load_source(source: SourceConfig):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": source.dataset_split}
    if source.dataset_revision is not None:
        kwargs["revision"] = source.dataset_revision
    if source.dataset_data_files is not None:
        kwargs["data_files"] = source.dataset_data_files
    if source.dataset_config_name is None:
        return load_dataset(source.dataset_id, **kwargs)
    return load_dataset(source.dataset_id, source.dataset_config_name, **kwargs)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_token_ids(tokens: Any) -> list[int]:
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        if len(tokens) != 1:
            raise ValueError("tokenizer returned an unexpected batch")
        tokens = tokens[0]
    normalized = []
    for token in tokens:
        if isinstance(token, bool):
            raise ValueError("tokenizer returned a boolean token ID")
        try:
            token_id = operator.index(token)
        except TypeError as exc:
            raise ValueError("tokenizer returned a non-integer token ID") from exc
        if token_id < 0:
            raise ValueError("tokenizer returned a negative token ID")
        normalized.append(token_id)
    return normalized


def _tokenize_text(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return _normalize_token_ids(encoded["input_ids"])


def _tokenize_text_with_offsets(
    tokenizer, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize canonical SWE text once and require character offset mappings."""
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise ValueError(
            "SWE-chat format requires a tokenizer that supports "
            "return_offsets_mapping=True"
        ) from exc
    if "offset_mapping" not in encoded:
        raise ValueError(
            "SWE-chat format requires a tokenizer that supports "
            "return_offsets_mapping=True"
        )

    tokens = _normalize_token_ids(encoded["input_ids"])
    raw_offsets = encoded["offset_mapping"]
    if hasattr(raw_offsets, "tolist"):
        raw_offsets = raw_offsets.tolist()
    if (
        raw_offsets
        and isinstance(raw_offsets[0], (list, tuple))
        and raw_offsets[0]
        and isinstance(raw_offsets[0][0], (list, tuple))
    ):
        if len(raw_offsets) != 1:
            raise ValueError("tokenizer returned an unexpected offset batch")
        raw_offsets = raw_offsets[0]

    offsets = []
    for offset in raw_offsets:
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError("tokenizer returned a malformed offset mapping")
        try:
            start, end = (operator.index(value) for value in offset)
        except TypeError as exc:
            raise ValueError("tokenizer returned a non-integer offset") from exc
        if start < 0 or end < start or end > len(text):
            raise ValueError("tokenizer returned an out-of-range offset mapping")
        offsets.append((start, end))
    if len(offsets) != len(tokens):
        raise ValueError("tokenizer returned mismatched tokens and offset mappings")
    return tokens, offsets


def _render_messages(tokenizer, messages: list[dict[str, Any]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise ValueError("tokenizer chat template must render to text")
    return rendered


def _anchor_text(
    messages: list[dict[str, Any]], anchor: SweAssistantAnchor
) -> str:
    message = messages[anchor.message_index]
    if anchor.field != "tool_call_function_name":
        value = message[anchor.field]
    else:
        value = message["tool_calls"][anchor.tool_call_index]["function"]["name"]
    if not isinstance(value, str):
        raise ValueError("SWE-chat assistant anchor must be text")
    return value


def _set_anchor_text(
    messages: list[dict[str, Any]], anchor: SweAssistantAnchor, value: str
) -> None:
    message = messages[anchor.message_index]
    if anchor.field != "tool_call_function_name":
        message[anchor.field] = value
    else:
        message["tool_calls"][anchor.tool_call_index]["function"]["name"] = value


def _anchor_markers(
    canonical_text: str, anchor: SweAssistantAnchor
) -> tuple[str, str]:
    """Derive deterministic alphanumeric sentinels absent from canonical text."""
    attempt = 0
    while True:
        key = (
            f"{hashlib.sha256(canonical_text.encode('utf-8')).hexdigest()}:"
            f"{anchor.message_index}:{anchor.field}:{anchor.tool_call_index}:{attempt}"
        )
        digest = hashlib.sha256(key.encode("ascii")).hexdigest().upper()
        start = f"SWECHATANCHORSTART{digest}X"
        end = f"SWECHATANCHOREND{digest}X"
        if start not in canonical_text and end not in canonical_text:
            return start, end
        attempt += 1


def _canonical_anchor_span(
    session: NormalizedSweSession,
    anchor: SweAssistantAnchor,
    tokenizer,
    canonical_text: str,
) -> tuple[int, int]:
    """Locate one assistant field in the canonical full-session rendering.

    The marked rendering must become byte-for-byte identical to the canonical
    rendering after the two sentinels are removed. This rejects templates that
    omit, transform, duplicate, or relocate the selected assistant field.
    """
    value = _anchor_text(session.messages, anchor)
    content_start = len(value) - len(value.lstrip())
    content_end = len(value.rstrip())
    if content_start >= content_end:
        raise ValueError("SWE-chat assistant anchor is empty after trimming")

    start_marker, end_marker = _anchor_markers(canonical_text, anchor)
    marked_messages = deepcopy(session.messages)
    marked_value = (
        value[:content_start]
        + start_marker
        + value[content_start:content_end]
        + end_marker
        + value[content_end:]
    )
    _set_anchor_text(marked_messages, anchor, marked_value)
    marked_text = _render_messages(tokenizer, marked_messages)
    if marked_text.count(start_marker) != 1 or marked_text.count(end_marker) != 1:
        raise ValueError(
            "SWE-chat chat template must preserve assistant anchor markers exactly"
        )
    start = marked_text.index(start_marker)
    marked_end = marked_text.index(end_marker)
    if marked_end < start + len(start_marker):
        raise ValueError(
            "SWE-chat chat template reordered assistant anchor markers"
        )
    unmarked_text = (
        marked_text[:start]
        + marked_text[start + len(start_marker) : marked_end]
        + marked_text[marked_end + len(end_marker) :]
    )
    if unmarked_text != canonical_text:
        raise ValueError(
            "SWE-chat chat template changed canonical rendering around "
            "assistant anchor markers"
        )
    end = marked_end - len(start_marker)
    if start >= end:
        raise ValueError("SWE-chat assistant anchor rendered no substantive text")
    return start, end


def _standard_document(
    source: SourceConfig, row: dict[str, Any], tokenizer
) -> tuple[list[int], str]:
    if source.column not in row:
        raise ValueError(
            f"source {source.name!r} row is missing column {source.column!r}"
        )
    content = row[source.column]
    if source.format == "messages":
        if not isinstance(content, list) or not content:
            raise _IneligibleDocument
        text = _render_messages(tokenizer, content)
    else:
        if not isinstance(content, str):
            raise ValueError(f"source {source.name!r} text must be a string")
        text = content
    return _tokenize_text(tokenizer, text), _canonical_hash(content)


def _window(
    tokens: list[int], length: int, rng: random.Random
) -> tuple[list[int], int]:
    if len(tokens) < length:
        raise _IneligibleDocument
    offset = rng.randint(0, len(tokens) - length)
    return tokens[offset : offset + length], offset


def _swe_window(
    session: NormalizedSweSession,
    tokenizer,
    length: int,
    rng: random.Random,
) -> tuple[list[int], dict[str, Any]]:
    """Return a full-length window with verified assistant-field overlap.

    Character and token spans are half-open positions in the one canonical full
    session rendering. anchor_turns lists every original turn merged into the
    selected assistant message.
    """
    if not session.anchors:
        raise _IneligibleDocument
    anchor = rng.choice(session.anchors)
    canonical_text = _render_messages(tokenizer, session.messages)
    anchor_char_start, anchor_char_end = _canonical_anchor_span(
        session, anchor, tokenizer, canonical_text
    )
    full_tokens, token_offsets = _tokenize_text_with_offsets(tokenizer, canonical_text)
    if len(full_tokens) < length:
        raise _IneligibleDocument

    overlapping_tokens = [
        index
        for index, (start, end) in enumerate(token_offsets)
        if end > anchor_char_start and start < anchor_char_end
    ]
    if not overlapping_tokens:
        raise ValueError(
            "SWE-chat assistant anchor has no overlapping tokenizer offsets"
        )
    anchor_token_start = overlapping_tokens[0]
    anchor_token_end = overlapping_tokens[-1] + 1
    offset = max(
        0,
        min(anchor_token_start - length // 3, len(full_tokens) - length),
    )
    if not any(offset <= index < offset + length for index in overlapping_tokens):
        raise ValueError("SWE-chat window does not overlap its assistant anchor")

    return full_tokens[offset : offset + length], {
        "offset": offset,
        "anchor_message_index": anchor.message_index,
        "anchor_turns": list(session.message_turns[anchor.message_index]),
        "anchor_field": anchor.field,
        "anchor_tool_call_index": anchor.tool_call_index,
        "anchor_char_span": [anchor_char_start, anchor_char_end],
        "anchor_token_span": [anchor_token_start, anchor_token_end],
    }


def _sample_standard_source(
    source: SourceConfig,
    dataset,
    quota: int,
    length: int,
    rng: random.Random,
    tokenizer,
) -> list[tuple[dict[str, list[int]], dict[str, Any]]]:
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected = []
    seen_document_ids: set[str] = set()
    for index in indices:
        row = dataset[index]
        raw_document_id = (
            row.get(source.id_column) if source.id_column is not None else None
        )
        document_id = str(
            index
            if raw_document_id is None or raw_document_id == ""
            else raw_document_id
        )
        if document_id in seen_document_ids:
            continue
        seen_document_ids.add(document_id)
        try:
            tokens, content_hash = _standard_document(source, row, tokenizer)
            window, offset = _window(tokens, length, rng)
        except _IneligibleDocument:
            continue
        except Exception as exc:
            raise ValueError(
                f"source {source.name!r} document {document_id!r} is malformed: {exc}"
            ) from exc
        selected.append(
            (
                {"input_ids": window, "attention_mask": [1] * length},
                {
                    "source": source.name,
                    "document_id": document_id,
                    "offset": offset,
                    "anchor_message_index": None,
                    "anchor_turns": None,
                    "anchor_field": None,
                    "anchor_tool_call_index": None,
                    "anchor_char_span": None,
                    "anchor_token_span": None,
                    "content_sha256": content_hash,
                },
            )
        )
        if len(selected) == quota:
            return selected
    raise ValueError(
        f"source {source.name!r} supplied {len(selected)} eligible windows; "
        f"quota is {quota}"
    )


def _swe_session_index(
    source: SourceConfig, dataset
) -> dict[str, list[tuple[int, int]]]:
    required = {source.id_column, "turn_number"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"SWE-chat source {source.name!r} is missing columns {sorted(missing)}"
        )
    lightweight = dataset.select_columns(sorted(required))
    sessions: dict[str, list[tuple[int, int]]] = {}
    for index, row in enumerate(lightweight):
        session_id = row[source.id_column]
        turn = row["turn_number"]
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("SWE-chat session ID must be a non-empty string")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
            raise ValueError("SWE-chat turn_number must be a non-negative integer")
        sessions.setdefault(session_id, []).append((turn, index))
    return sessions


def _sample_swe_source(
    source: SourceConfig,
    dataset,
    quota: int,
    length: int,
    rng: random.Random,
    tokenizer,
) -> list[tuple[dict[str, list[int]], dict[str, Any]]]:
    sessions = _swe_session_index(source, dataset)
    session_ids = list(sessions)
    rng.shuffle(session_ids)
    selected = []
    for session_id in session_ids:
        indices = [index for _, index in sorted(sessions[session_id])]
        rows = [dataset[index] for index in indices]
        try:
            normalized = normalize_swe_chat_session(rows)
            tokens, anchor_metadata = _swe_window(
                normalized, tokenizer, length, rng
            )
        except _IneligibleDocument:
            continue
        except Exception as exc:
            raise ValueError(
                f"source {source.name!r} session {session_id!r} is malformed: {exc}"
            ) from exc
        selected.append(
            (
                {"input_ids": tokens, "attention_mask": [1] * length},
                {
                    "source": source.name,
                    "document_id": session_id,
                    **anchor_metadata,
                    "content_sha256": _canonical_hash(rows),
                },
            )
        )
        if len(selected) == quota:
            return selected
    raise ValueError(
        f"source {source.name!r} supplied {len(selected)} eligible sessions; "
        f"quota is {quota}"
    )


def _source_random(seed: int, source_index: int, source_name: str) -> random.Random:
    return random.Random(f"{seed}:{source_index}:{source_name}")


def _write_bundle(
    output: Path,
    rows: list[dict[str, list[int]]],
    manifest: dict[str, Any],
) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        data_path = temporary / DATA_FILENAME
        with data_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        manifest["data_sha256"] = sha256_file(data_path)
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_calibration_bundle(
    config: MixConfig,
    output: str | Path,
    tokenizer,
    *,
    tokenizer_revision: str | None = None,
) -> dict[str, Any]:
    """Prepare all source windows and publish a verified bundle atomically."""
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output {output_path}")
    quotas = allocate_quotas(config.num_samples, config.sources)
    combined = []
    source_records = []
    realized_counts: dict[str, int] = {}
    for source_index, (source, quota) in enumerate(zip(config.sources, quotas)):
        dataset = _load_source(source)
        rng = _source_random(config.seed, source_index, source.name)
        if source.format == "swe_chat":
            selected = _sample_swe_source(
                source, dataset, quota, config.max_seq_length, rng, tokenizer
            )
        else:
            selected = _sample_standard_source(
                source, dataset, quota, config.max_seq_length, rng, tokenizer
            )
        combined.extend(selected)
        realized_counts[source.name] = len(selected)
        source_records.append(
            {
                **asdict(source),
                "quota": quota,
                "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            }
        )

    random.Random(f"{config.seed}:combined").shuffle(combined)
    rows = [row for row, _ in combined]
    samples = [metadata for _, metadata in combined]
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "num_samples": config.num_samples,
        "max_seq_length": config.max_seq_length,
        "seed": config.seed,
        "tokenizer": tokenizer_identity(tokenizer, revision=tokenizer_revision),
        "sources": source_records,
        "realized_counts": realized_counts,
        "samples": samples,
    }
    _write_bundle(output_path, rows, manifest)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_mix_config(args.config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )
    prepare_calibration_bundle(
        config,
        args.output,
        tokenizer,
        tokenizer_revision=args.tokenizer_revision,
    )


if __name__ == "__main__":
    main()
