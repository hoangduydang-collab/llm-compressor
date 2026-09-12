"""Validation and loading for prepared fixed-token calibration bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA_VERSION = 1
DATA_FILENAME = "data.jsonl"
MANIFEST_FILENAME = "manifest.json"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _backend_description(tokenizer) -> str | None:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        return None
    description = to_str()
    try:
        parsed = json.loads(description)
    except (TypeError, json.JSONDecodeError):
        return description
    # Runtime encode settings can be mutated temporarily and do not describe
    # intrinsic normalization or tokenization behavior.
    parsed.pop("truncation", None)
    parsed.pop("padding", None)
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tokenizer_identity(tokenizer, *, revision: str | None = None) -> dict[str, Any]:
    """Return a stable fingerprint of behavior that changes tokenized windows."""
    vocab = tokenizer.get_vocab()
    material = {
        "vocabulary": sorted(
            (str(token), int(index)) for token, index in vocab.items()
        ),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "backend": _backend_description(tokenizer),
    }
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    resolved_revision = (
        init_kwargs.get("_commit_hash") or init_kwargs.get("revision") or revision
    )
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "revision": resolved_revision,
        "sha256": hashlib.sha256(_json_bytes(material)).hexdigest(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_plain_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an integer")
    return value


def load_calibration_bundle(
    directory: str | Path,
    tokenizer,
    num_samples: int,
    max_seq_length: int,
):
    """Verify a prepared bundle completely, then return it as a Dataset."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILENAME
    data_path = directory / DATA_FILENAME
    if not manifest_path.is_file() or not data_path.is_file():
        raise ValueError(
            f"prepared calibration bundle {directory} must contain "
            f"{MANIFEST_FILENAME} and {DATA_FILENAME}"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid calibration manifest: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported calibration bundle schema_version")

    recorded_hash = manifest.get("data_sha256")
    if not isinstance(recorded_hash, str) or sha256_file(data_path) != recorded_hash:
        raise ValueError("calibration data SHA-256 does not match manifest")

    manifest_samples = _require_plain_int(
        manifest.get("num_samples"), "manifest num_samples"
    )
    manifest_length = _require_plain_int(
        manifest.get("max_seq_length"), "manifest max_seq_length"
    )
    if manifest_samples != num_samples:
        raise ValueError(
            f"prepared calibration num_samples mismatch: bundle={manifest_samples}, "
            f"config={num_samples}"
        )
    if manifest_length != max_seq_length:
        raise ValueError(
            "prepared calibration max_seq_length mismatch: "
            f"bundle={manifest_length}, config={max_seq_length}"
        )

    expected_identity = manifest.get("tokenizer")
    actual_identity = tokenizer_identity(tokenizer)
    if (
        not isinstance(expected_identity, dict)
        or expected_identity.get("sha256") != actual_identity["sha256"]
    ):
        raise ValueError(
            "prepared calibration tokenizer identity does not match "
            "the loaded tokenizer"
        )

    rows = []
    try:
        with data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {
                    "input_ids",
                    "attention_mask",
                }:
                    raise ValueError(
                        f"calibration row {line_number} must contain only "
                        "input_ids and attention_mask"
                    )
                input_ids = row["input_ids"]
                attention_mask = row["attention_mask"]
                if not isinstance(input_ids, list) or len(input_ids) != max_seq_length:
                    raise ValueError(
                        f"calibration row {line_number} input_ids must have "
                        f"length {max_seq_length}"
                    )
                if any(
                    isinstance(token, bool) or not isinstance(token, int) or token < 0
                    for token in input_ids
                ):
                    raise ValueError(
                        f"calibration row {line_number} has invalid token IDs"
                    )
                if (
                    not isinstance(attention_mask, list)
                    or len(attention_mask) != max_seq_length
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value != 1
                        for value in attention_mask
                    )
                ):
                    raise ValueError(
                        f"calibration row {line_number} attention_mask "
                        "must be all ones"
                    )
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid calibration data: {exc}") from exc

    if len(rows) != num_samples:
        raise ValueError(
            f"prepared calibration row count mismatch: data={len(rows)}, "
            f"config={num_samples}"
        )

    from datasets import Dataset

    return Dataset.from_list(rows)
