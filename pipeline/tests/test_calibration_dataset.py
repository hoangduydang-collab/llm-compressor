from __future__ import annotations

import sys
from types import SimpleNamespace

from pipeline import calibration
from pipeline.config import CalibrationConfig


def test_local_data_files_are_forwarded_to_datasets(monkeypatch):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return ["row"]

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )
    cfg = CalibrationConfig(
        dataset_id="json",
        dataset_split="train",
        dataset_data_files="pipeline/fixtures/calibration.jsonl",
        num_samples=8,
    )

    assert calibration._load_raw_dataset(cfg) == ["row"]
    assert calls == [
        (
            ("json",),
            {
                "split": "train[:8]",
                "data_files": "pipeline/fixtures/calibration.jsonl",
            },
        )
    ]


def test_remote_dataset_omits_data_files_keyword(monkeypatch):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return ["row"]

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
    )
    cfg = CalibrationConfig(
        dataset_id="HuggingFaceH4/ultrachat_200k",
        dataset_split="train_sft",
        num_samples=256,
    )

    assert calibration._load_raw_dataset(cfg) == ["row"]
    assert calls == [
        (
            ("HuggingFaceH4/ultrachat_200k",),
            {"split": "train_sft[:256]"},
        )
    ]
