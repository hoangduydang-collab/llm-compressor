"""Tests for exact paired evaluation sample manifests."""

from __future__ import annotations

import json

import pytest

from pipeline.evalsuite.sampling import (
    build_stratified_indices,
    load_sample_manifest,
    sample_map_for_task,
    stable_sample_uid,
)


def test_stratified_indices_are_deterministic_and_proportional():
    sizes = {"mmlu_pro_math": 100, "mmlu_pro_history": 50}

    first = build_stratified_indices(sizes, total=30, seed=42)
    second = build_stratified_indices(sizes, total=30, seed=42)

    assert first == second
    assert len(first["mmlu_pro_math"]) == 20
    assert len(first["mmlu_pro_history"]) == 10
    assert sum(len(indices) for indices in first.values()) == 30
    assert first["mmlu_pro_math"] == sorted(set(first["mmlu_pro_math"]))
    assert all(0 <= index < 100 for index in first["mmlu_pro_math"])


def test_stratified_indices_distribute_remainder_deterministically():
    result = build_stratified_indices({"a": 1, "b": 1, "c": 1}, total=2, seed=9)
    assert {name: len(indices) for name, indices in result.items()} == {
        "a": 1,
        "b": 1,
        "c": 0,
    }


@pytest.mark.parametrize("total", [0, 4])
def test_stratified_indices_reject_invalid_total(total):
    with pytest.raises(ValueError, match="total"):
        build_stratified_indices({"a": 3}, total=total, seed=42)


def test_manifest_roundtrip_and_task_lookup(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 42,
                "tasks": {
                    "mmlu_pro": {
                        "mmlu_pro_math": [1, 3],
                        "mmlu_pro_history": [0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_sample_manifest(path)

    assert len(manifest.sha256) == 64
    assert sample_map_for_task(manifest, "mmlu_pro") == {
        "mmlu_pro_math": [1, 3],
        "mmlu_pro_history": [0],
    }
    assert sample_map_for_task(manifest, "gpqa_diamond") is None


def test_manifest_rejects_declared_hash_mismatch(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 42,
                "tasks": {"mmlu_pro": {"mmlu_pro_math": [1]}},
                "sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sha256"):
        load_sample_manifest(path)


def test_stable_sample_uid_namespaces_group_subtasks():
    math_uid = stable_sample_uid("mmlu_pro", "mmlu_pro_math", 0)
    history_uid = stable_sample_uid("mmlu_pro", "mmlu_pro_history", 0)

    assert math_uid != history_uid
    assert math_uid == stable_sample_uid("mmlu_pro", "mmlu_pro_math", 0)
    assert len(math_uid) == 64
