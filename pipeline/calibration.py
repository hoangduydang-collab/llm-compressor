"""Build the calibration dataset used by ``oneshot``.

Factored out of the example scripts (load -> chat-template -> tokenize). Returns
a tokenized ``datasets.Dataset`` ready to hand to ``oneshot(dataset=...)``.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass

from pipeline.config import CalibrationConfig


@dataclass(frozen=True)
class CalibrationPartition:
    """Rank-local slice of one globally configured calibration set."""

    global_num_samples: int
    rank: int
    world_size: int
    start: int
    end: int


def partition_bounds(
    num_samples: int, rank: int, world_size: int
) -> tuple[int, int]:
    """Return non-overlapping floor-partition bounds for one rank."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must satisfy 0 <= rank < world_size, got {rank}")
    start = num_samples * rank // world_size
    end = num_samples * (rank + 1) // world_size
    return start, end


def _distributed_rank_world_size() -> tuple[int, int]:
    """Read initialized rank metadata without importing torch for local runs."""
    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        environment_world_size = int(raw_world_size)
    except ValueError as exc:
        raise RuntimeError(
            f"WORLD_SIZE must be an integer, got {raw_world_size!r}"
        ) from exc
    if environment_world_size <= 1:
        return 0, 1

    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "WORLD_SIZE > 1 but torch.distributed is not initialized; "
            "pipeline.run must initialize distributed state before calibration"
        )
    return dist.get_rank(), dist.get_world_size()


def calibration_partition_manifest(dataset, partition: CalibrationPartition) -> dict:
    """Describe and hash the exact token IDs consumed by one rank."""
    token_rows: list[list[int]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if "input_ids" not in sample:
            raise ValueError("calibration sample is missing input_ids")
        token_rows.append([int(token) for token in sample["input_ids"]])
    token_hash = hashlib.sha256(
        json.dumps(token_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        **asdict(partition),
        "local_num_samples": len(dataset),
        "token_ids_sha256": token_hash,
    }


def build_calibration_dataset_with_partition(
    cal: CalibrationConfig, tokenizer
):
    """Load/tokenize calibration data and return its rank-local partition."""
    from datasets import load_dataset

    split = f"{cal.dataset_split}[:{cal.num_samples}]"
    ds = load_dataset(cal.dataset_id, split=split)
    ds = ds.shuffle(seed=cal.seed)

    global_num_samples = len(ds)
    rank, world_size = _distributed_rank_world_size()
    start, end = partition_bounds(global_num_samples, rank, world_size)
    partition = CalibrationPartition(
        global_num_samples=global_num_samples,
        rank=rank,
        world_size=world_size,
        start=start,
        end=end,
    )
    if world_size > 1:
        ds = ds.select(range(start, end))

    column_names = ds.column_names
    has_messages = "messages" in column_names
    has_text = "text" in column_names

    def preprocess(example):
        if has_messages:
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"], tokenize=False
                )
            }
        if has_text:
            return {"text": example["text"]}
        # Fall back to the first string column.
        first = column_names[0]
        return {"text": str(example[first])}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=cal.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds, partition


def build_calibration_dataset(cal: CalibrationConfig, tokenizer):
    """Load, format and tokenize the calibration set described by ``cal``."""
    dataset, _ = build_calibration_dataset_with_partition(cal, tokenizer)
    return dataset
