"""Exact sample manifests for paired checkpoint evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SampleManifest:
    schema_version: int
    seed: int
    tasks: dict[str, dict[str, tuple[int, ...]]]
    sha256: str
    metadata: dict[str, Any]


def _canonical_payload(data: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in data.items() if key != "sha256"}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def manifest_sha256(data: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(data)).hexdigest()


def stable_sample_uid(task: str, subtask: str, doc_id: object) -> str:
    payload = json.dumps(
        [task, subtask, doc_id],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_indices(task: str, subtask: str, values: object) -> tuple[int, ...]:
    if not isinstance(values, list):
        raise ValueError(f"sample indices for {task}/{subtask} must be a list")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise ValueError(
            f"sample indices for {task}/{subtask} must be non-negative integers"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"sample indices for {task}/{subtask} contain duplicates")
    return tuple(sorted(values))


def load_sample_manifest(path: str | Path) -> SampleManifest:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("sample manifest must be a JSON object")
    if data.get("schema_version") != 1:
        raise ValueError(
            "unsupported sample manifest schema_version: "
            f"{data.get('schema_version')!r}"
        )
    seed = data.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("sample manifest seed must be an integer")
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, dict) or not raw_tasks:
        raise ValueError("sample manifest tasks must be a non-empty object")

    tasks: dict[str, dict[str, tuple[int, ...]]] = {}
    for task, raw_subtasks in raw_tasks.items():
        if (
            not isinstance(task, str)
            or not isinstance(raw_subtasks, dict)
            or not raw_subtasks
        ):
            raise ValueError("each sample manifest task must map to a non-empty object")
        tasks[task] = {
            subtask: _validate_indices(task, subtask, values)
            for subtask, values in raw_subtasks.items()
            if isinstance(subtask, str)
        }
        if len(tasks[task]) != len(raw_subtasks):
            raise ValueError(f"sample manifest {task!r} has a non-string subtask name")

    digest = manifest_sha256(data)
    declared = data.get("sha256")
    if declared is not None and declared != digest:
        raise ValueError(
            f"sample manifest sha256 mismatch: declared={declared} computed={digest}"
        )

    metadata = {
        key: value
        for key, value in data.items()
        if key not in {"schema_version", "seed", "tasks", "sha256"}
    }
    return SampleManifest(1, seed, tasks, digest, metadata)


def sample_map_for_task(
    manifest: SampleManifest,
    task_name: str,
) -> dict[str, list[int]] | None:
    subtasks = manifest.tasks.get(task_name)
    if subtasks is None:
        return None
    return {name: list(indices) for name, indices in subtasks.items()}


def _leaf_seed(seed: int, leaf: str) -> int:
    digest = hashlib.sha256(leaf.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:8], "big")


def build_stratified_indices(
    sizes: dict[str, int],
    total: int,
    seed: int,
) -> dict[str, list[int]]:
    if not sizes or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 0
        for size in sizes.values()
    ):
        raise ValueError("sizes must be a non-empty mapping of non-negative integers")
    available = sum(sizes.values())
    if total <= 0 or total > available:
        raise ValueError(f"total must be between 1 and {available}, got {total}")

    quotas = {name: total * size / available for name, size in sizes.items()}
    allocations = {name: int(quota) for name, quota in quotas.items()}
    remainder = total - sum(allocations.values())
    ranked = sorted(
        sizes, key=lambda name: (-(quotas[name] - allocations[name]), name)
    )
    for name in ranked[:remainder]:
        allocations[name] += 1

    selected: dict[str, list[int]] = {}
    for name in sorted(sizes):
        count = allocations[name]
        rng = random.Random(_leaf_seed(seed, name))
        selected[name] = (
            sorted(rng.sample(range(sizes[name]), count)) if count else []
        )
    return selected
