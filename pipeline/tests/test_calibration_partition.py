import hashlib
import json

import pytest

from pipeline.calibration import (
    CalibrationPartition,
    calibration_partition_manifest,
    partition_bounds,
)


def test_partition_bounds_cover_nondivisible_global_set():
    bounds = [partition_bounds(10, rank, 3) for rank in range(3)]

    assert bounds == [(0, 3), (3, 6), (6, 10)]
    assert [index for start, end in bounds for index in range(start, end)] == list(
        range(10)
    )


def test_partition_bounds_cover_fewer_samples_than_ranks():
    bounds = [partition_bounds(3, rank, 4) for rank in range(4)]

    assert bounds == [(0, 0), (0, 1), (1, 2), (2, 3)]


@pytest.mark.parametrize(
    ("num_samples", "rank", "world_size", "message"),
    [
        (-1, 0, 1, "num_samples"),
        (8, -1, 2, "rank"),
        (8, 2, 2, "rank"),
        (8, 0, 0, "world_size"),
    ],
)
def test_partition_bounds_reject_invalid_inputs(num_samples, rank, world_size, message):
    with pytest.raises(ValueError, match=message):
        partition_bounds(num_samples, rank, world_size)


def test_partition_manifest_hashes_local_token_ids_stably():
    dataset = [
        {"input_ids": [1, 2], "attention_mask": [1, 1]},
        {"input_ids": [3], "attention_mask": [1]},
    ]
    partition = CalibrationPartition(
        global_num_samples=5,
        rank=1,
        world_size=2,
        start=2,
        end=5,
    )

    manifest = calibration_partition_manifest(dataset, partition)

    expected_hash = hashlib.sha256(
        json.dumps([[1, 2], [3]], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert manifest == {
        "schema_version": 1,
        "global_num_samples": 5,
        "local_num_samples": 2,
        "rank": 1,
        "world_size": 2,
        "start": 2,
        "end": 5,
        "token_ids_sha256": expected_hash,
    }


def test_partition_manifest_rejects_missing_input_ids():
    partition = CalibrationPartition(1, 0, 1, 0, 1)

    with pytest.raises(ValueError, match="input_ids"):
        calibration_partition_manifest([{"text": "not tokenized"}], partition)
