"""Regression test: under distributed calibration, each rank's DataLoader must
yield its own disjoint shard of the dataset.

The old implementation partitioned inside ``_make_sampler`` by rebinding a
local variable, so the sampler's shard-relative indices were applied to the
full dataset by the DataLoader and every rank silently calibrated on the same
first-partition rows (bit-identical losses across ranks; effective calibration
set shrunk by the world size).
"""

from unittest.mock import patch

import pytest
from datasets import Dataset

import llmcompressor.datasets.utils as du
from llmcompressor.args import DatasetArguments

NUM_ROWS = 8


@pytest.fixture
def tokenized_dataset():
    # fixed-length rows, mirroring pre-tokenized calibration data
    return Dataset.from_dict({"input_ids": [[i] * 16 for i in range(NUM_ROWS)]})


def _rows_for_rank(dataset, rank, world_size):
    args = DatasetArguments(
        num_calibration_samples=NUM_ROWS,
        shuffle_calibration_samples=False,
        batch_size=1,
    )
    with (
        patch.object(du.dist, "is_initialized", return_value=True),
        patch.object(du.dist, "get_rank", return_value=rank),
        patch.object(du.dist, "get_world_size", return_value=world_size),
        patch.object(
            du.dist,
            "all_gather_object",
            side_effect=lambda out, h: out.__setitem__(
                slice(None), [h] * world_size
            ),
        ),
    ):
        loader = du.format_calibration_data(args, dataset, processor=None)
        return sorted(int(batch["input_ids"][0][0]) for batch in loader)


@pytest.mark.parametrize("world_size", [2, 4])
def test_ranks_get_disjoint_complete_shards(tokenized_dataset, world_size):
    shards = [
        _rows_for_rank(tokenized_dataset, rank, world_size)
        for rank in range(world_size)
    ]

    seen = [row for shard in shards for row in shard]
    assert sorted(seen) == list(range(NUM_ROWS)), (
        "union of rank shards must cover every sample exactly once, "
        f"got {sorted(seen)}"
    )
    for a in range(world_size):
        for b in range(a + 1, world_size):
            assert set(shards[a]).isdisjoint(shards[b]), (
                f"ranks {a} and {b} overlap: {shards[a]} vs {shards[b]}"
            )


def test_non_distributed_uses_full_dataset(tokenized_dataset):
    args = DatasetArguments(
        num_calibration_samples=NUM_ROWS,
        shuffle_calibration_samples=False,
        batch_size=1,
    )
    loader = du.format_calibration_data(args, tokenized_dataset, processor=None)
    rows = sorted(int(batch["input_ids"][0][0]) for batch in loader)
    assert rows == list(range(NUM_ROWS))
