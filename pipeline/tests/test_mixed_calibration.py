from __future__ import annotations

import json

import datasets as hf_datasets
import pytest
from datasets import Dataset

from pipeline import calibration
from pipeline.calibration_bundle import (
    load_calibration_bundle,
    sha256_file,
    tokenizer_identity,
)
from pipeline.config import CalibrationConfig, load_config
from pipeline.prepare_calibration import (
    MixConfig,
    SourceConfig,
    allocate_quotas,
    load_mix_config,
    prepare_calibration_bundle,
)
from pipeline.swe_chat import normalize_swe_chat_session


class TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    chat_template = "tiny-v1"
    special_tokens_map = {"bos_token": "<bos>", "eos_token": "<eos>"}
    init_kwargs = {"revision": "tok-rev"}

    def __init__(self, *, vocab_extra: str = ""):
        self.vocab_extra = vocab_extra

    def get_vocab(self):
        return {"<bos>": 0, "<eos>": 1, "x": 2, self.vocab_extra: 3}

    def apply_chat_template(self, messages, *, tokenize=False, **_kwargs):
        rendered = []
        for message in messages:
            value = f"<{message['role']}>"
            if message.get("reasoning_content"):
                value += f"<think>{message['reasoning_content']}</think>"
            value += str(message.get("content") or "")
            for call in message.get("tool_calls", []):
                function = call["function"]
                value += f"<call:{call['id']}:{function['name']}:"
                value += json.dumps(function["arguments"], sort_keys=True) + ">"
            if message.get("tool_call_id"):
                value += f"<result:{message['tool_call_id']}>"
            rendered.append(value)
        text = "".join(rendered)
        return self(text)["input_ids"] if tokenize else text

    def __call__(self, text, **_kwargs):
        return {
            "input_ids": [2 + (ord(char) % 211) for char in text],
            "attention_mask": [1] * len(text),
        }


def _rows(prefix: str, count: int = 6, size: int = 120):
    return [
        {"messages": [{"role": "user", "content": f"{prefix}-{i}-" + "x" * size}]}
        for i in range(count)
    ]


def _install_fake_datasets(monkeypatch, datasets_by_id):
    calls = []

    def load_dataset(dataset_id, config_name=None, **kwargs):
        calls.append((dataset_id, config_name, kwargs))
        rows = datasets_by_id[dataset_id]
        columns = set().union(*(row.keys() for row in rows))
        normalized = [{key: row.get(key) for key in columns} for row in rows]
        return Dataset.from_list(normalized)

    monkeypatch.setattr(hf_datasets, "load_dataset", load_dataset)
    return calls


def test_largest_remainder_quotas_are_exact_and_reject_starved_sources():
    sources = [
        SourceConfig(name="a", weight=2, dataset_id="a"),
        SourceConfig(name="b", weight=1, dataset_id="b"),
        SourceConfig(name="c", weight=1, dataset_id="c"),
    ]
    assert allocate_quotas(10, sources) == [5, 3, 2]
    with pytest.raises(ValueError, match="rounded to zero"):
        allocate_quotas(2, sources)


def test_mix_config_parses_defaults_and_rejects_unknown_keys(tmp_path):
    path = tmp_path / "mix.yaml"
    path.write_text(
        "num_samples: 3\nmax_seq_length: 8\nseed: 7\nsources:\n"
        "  - name: chat\n    weight: 1\n    dataset_id: local\n",
        encoding="utf-8",
    )
    config = load_mix_config(path)
    assert config == MixConfig(
        num_samples=3,
        max_seq_length=8,
        seed=7,
        sources=(SourceConfig(name="chat", weight=1, dataset_id="local"),),
    )

    path.write_text(path.read_text() + "surprise: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_mix_config(path)


def test_prepare_mixed_bundle_has_exact_counts_and_is_deterministic(
    monkeypatch, tmp_path
):
    calls = _install_fake_datasets(
        monkeypatch,
        {"generic": _rows("generic"), "coding": _rows("coding")},
    )
    config = MixConfig(
        num_samples=5,
        max_seq_length=32,
        seed=17,
        sources=(
            SourceConfig(name="generic", weight=3, dataset_id="generic"),
            SourceConfig(
                name="coding",
                weight=2,
                dataset_id="coding",
                dataset_config_name="cfg",
                dataset_revision="source-rev",
                dataset_data_files="rows.jsonl",
            ),
        ),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_calibration_bundle(
        config, first, TinyTokenizer(), tokenizer_revision="tok-rev"
    )
    prepare_calibration_bundle(
        config, second, TinyTokenizer(), tokenizer_revision="tok-rev"
    )

    first_data = (first / "data.jsonl").read_bytes()
    assert first_data == (second / "data.jsonl").read_bytes()
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    rows = [json.loads(line) for line in first_data.splitlines()]
    assert len(rows) == 5
    assert all(len(row["input_ids"]) == 32 for row in rows)
    assert all(row["attention_mask"] == [1] * 32 for row in rows)
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["realized_counts"] == {"coding": 2, "generic": 3}
    assert len(manifest["samples"]) == 5
    assert calls[1] == (
        "coding",
        "cfg",
        {"split": "train", "revision": "source-rev", "data_files": "rows.jsonl"},
    )


def test_bundle_loader_rejects_corruption_and_tokenizer_mismatch(monkeypatch, tmp_path):
    _install_fake_datasets(monkeypatch, {"only": _rows("only")})
    output = tmp_path / "bundle"
    config = MixConfig(
        num_samples=2,
        max_seq_length=16,
        seed=1,
        sources=(SourceConfig(name="only", weight=1, dataset_id="only"),),
    )
    prepare_calibration_bundle(config, output, TinyTokenizer())

    with pytest.raises(ValueError, match="tokenizer identity"):
        load_calibration_bundle(output, TinyTokenizer(vocab_extra="changed"), 2, 16)

    with (output / "data.jsonl").open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_calibration_bundle(output, TinyTokenizer(), 2, 16)


def test_prepared_bundle_partitions_cover_rows_without_retokenizing(
    monkeypatch, tmp_path
):
    _install_fake_datasets(monkeypatch, {"only": _rows("only", count=8)})
    output = tmp_path / "bundle"
    prepare_calibration_bundle(
        MixConfig(
            num_samples=7,
            max_seq_length=12,
            seed=9,
            sources=(SourceConfig(name="only", weight=1, dataset_id="only"),),
        ),
        output,
        TinyTokenizer(),
    )
    expected = load_calibration_bundle(output, TinyTokenizer(), 7, 12)["input_ids"]
    covered = []
    for rank in range(3):
        monkeypatch.setattr(
            calibration, "_distributed_rank_world_size", lambda rank=rank: (rank, 3)
        )
        dataset, partition = calibration.build_calibration_dataset_with_partition(
            CalibrationConfig(
                prepared_dataset=str(output), num_samples=7, max_seq_length=12
            ),
            TinyTokenizer(),
        )
        assert partition.global_num_samples == 7
        covered.extend(dataset["input_ids"])
    assert covered == expected


def test_pipeline_config_accepts_prepared_dataset(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "model:\n  id: example/model\ncalibration:\n"
        "  prepared_dataset: /data/calibration\n",
        encoding="utf-8",
    )
    assert load_config(path).calibration.prepared_dataset == "/data/calibration"


def test_swe_session_reconstructs_tools_and_rejects_bad_calls():
    rows = [
        {
            "session_id": "s",
            "turn_number": 0,
            "role": "user",
            "turn_type": "user_prompt",
            "content": "fix it",
        },
        {
            "session_id": "s",
            "turn_number": 1,
            "role": "assistant",
            "turn_type": "assistant_thinking",
            "content": "inspect first",
        },
        {
            "session_id": "s",
            "turn_number": 2,
            "role": "assistant",
            "turn_type": "assistant_response",
            "content": "I will inspect.",
        },
        {
            "session_id": "s",
            "turn_number": 3,
            "role": "tool_use",
            "turn_type": "tool_use",
            "content": "",
            "tool_name": "read_file",
            "tool_call_id": "call-1",
            "tool_input_json": '{"path":"a.py"}',
        },
        {
            "session_id": "s",
            "turn_number": 4,
            "role": "tool_result",
            "turn_type": "tool_result",
            "content": "contents",
            "tool_call_id": "call-1",
        },
        {
            "session_id": "s",
            "turn_number": 5,
            "role": "metadata",
            "turn_type": "progress",
            "content": "drop me",
        },
    ]
    normalized = normalize_swe_chat_session(rows)
    assert normalized.messages == [
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": "I will inspect.",
            "reasoning_content": "inspect first",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "a.py"}},
                }
            ],
        },
        {"role": "tool", "content": "contents", "tool_call_id": "call-1"},
    ]
    assert normalized.message_turns == [0, 3, 4]

    broken = [dict(rows[0]), dict(rows[3], tool_input_json="[]")]
    with pytest.raises(ValueError, match="dictionary"):
        normalize_swe_chat_session(broken)


def test_swe_structural_window_selects_later_meaningful_work(monkeypatch, tmp_path):
    rows = [
        {
            "session_id": "s",
            "turn_number": 0,
            "role": "metadata",
            "turn_type": "system_event",
            "content": "boilerplate" * 50,
        },
        {
            "session_id": "s",
            "turn_number": 1,
            "role": "user",
            "turn_type": "user_prompt",
            "content": "please diagnose",
        },
        {
            "session_id": "s",
            "turn_number": 2,
            "role": "tool_use",
            "turn_type": "tool_use",
            "content": "",
            "tool_name": "read_log",
            "tool_call_id": "early-call",
            "tool_input_json": '{"path":"large.log"}',
        },
        {
            "session_id": "s",
            "turn_number": 3,
            "role": "tool_result",
            "turn_type": "tool_result",
            "content": "unhelpful tool output " + "z" * 400,
            "tool_call_id": "early-call",
        },
        {
            "session_id": "s",
            "turn_number": 4,
            "role": "user",
            "turn_type": "user_prompt",
            "content": "Find the actual bug.",
        },
        {
            "session_id": "s",
            "turn_number": 5,
            "role": "assistant",
            "turn_type": "assistant_thinking",
            "content": (
                "The parser drops the final record because the loop exits "
                "before flushing state."
            ),
        },
        {
            "session_id": "s",
            "turn_number": 6,
            "role": "assistant",
            "turn_type": "assistant_response",
            "content": (
                "Patch the finalizer and add a regression test for the missing record."
            ),
        },
    ]
    _install_fake_datasets(monkeypatch, {"swe": rows})
    output = tmp_path / "swe-bundle"
    prepare_calibration_bundle(
        MixConfig(
            num_samples=1,
            max_seq_length=80,
            seed=1,
            sources=(
                SourceConfig(
                    name="swe",
                    weight=1,
                    dataset_id="swe",
                    format="swe_chat",
                    id_column="session_id",
                ),
            ),
        ),
        output,
        TinyTokenizer(),
    )
    manifest = json.loads((output / "manifest.json").read_text())
    sample = manifest["samples"][0]
    assert sample["anchor_turn"] == 6
    assert sample["offset"] > 0


def test_tokenizer_identity_is_stable_and_tracks_template_and_vocab():
    first = tokenizer_identity(TinyTokenizer(), revision="rev")
    assert first == tokenizer_identity(TinyTokenizer(), revision="rev")
    assert (
        first["sha256"]
        != tokenizer_identity(TinyTokenizer(vocab_extra="new"), revision="rev")[
            "sha256"
        ]
    )


def test_shortage_fails_without_publishing_partial_bundle(monkeypatch, tmp_path):
    _install_fake_datasets(
        monkeypatch,
        {"short": [{"text": "tiny"}, {"text": "also tiny"}]},
    )
    output = tmp_path / "must-not-exist"
    config = MixConfig(
        num_samples=1,
        max_seq_length=64,
        seed=1,
        sources=(
            SourceConfig(name="short", weight=1, dataset_id="short", format="text"),
        ),
    )
    with pytest.raises(ValueError, match="eligible windows; quota is 1"):
        prepare_calibration_bundle(config, output, TinyTokenizer())
    assert not output.exists()


def test_loader_validates_dimensions_tokens_and_masks_after_byte_hash(tmp_path):
    output = tmp_path / "bundle"
    data_path = output / "data.jsonl"
    manifest_path = output / "manifest.json"
    output.mkdir()
    rows = [{"input_ids": [2, 3], "attention_mask": [1, 1]}]
    data_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "num_samples": 1,
        "max_seq_length": 2,
        "tokenizer": tokenizer_identity(TinyTokenizer()),
        "data_sha256": sha256_file(data_path),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="max_seq_length mismatch"):
        load_calibration_bundle(output, TinyTokenizer(), 1, 3)

    rows[0]["input_ids"][0] = -1
    data_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    manifest["data_sha256"] = sha256_file(data_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid token IDs"):
        load_calibration_bundle(output, TinyTokenizer(), 1, 2)

    rows[0]["input_ids"][0] = 2
    rows[0]["attention_mask"][0] = 0
    data_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    manifest["data_sha256"] = sha256_file(data_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="attention_mask must be all ones"):
        load_calibration_bundle(output, TinyTokenizer(), 1, 2)


def test_local_json_swe_source_uses_hugging_face_loader_without_named_config(
    tmp_path,
):
    rows_path = tmp_path / "swe.jsonl"
    rows = [
        {
            "session_id": "local-session",
            "turn_number": 0,
            "role": "user",
            "turn_type": "user_prompt",
            "content": "Fix the parser and explain the regression.",
            "tool_name": None,
            "tool_call_id": None,
            "tool_input_json": None,
        },
        {
            "session_id": "local-session",
            "turn_number": 1,
            "role": "assistant",
            "turn_type": "assistant_response",
            "content": "The final buffered record needs an explicit flush. " * 8,
            "tool_name": None,
            "tool_call_id": None,
            "tool_input_json": None,
        },
    ]
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "bundle"
    prepare_calibration_bundle(
        MixConfig(
            num_samples=1,
            max_seq_length=32,
            seed=2,
            sources=(
                SourceConfig(
                    name="local-swe",
                    weight=1,
                    dataset_id="json",
                    dataset_data_files=str(rows_path),
                    format="swe_chat",
                ),
            ),
        ),
        output,
        TinyTokenizer(),
    )
    loaded = load_calibration_bundle(output, TinyTokenizer(), 1, 32)
    assert len(loaded) == 1


def test_existing_output_is_rejected_before_loading_sources(monkeypatch, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()

    def must_not_load(*_args, **_kwargs):
        raise AssertionError("dataset loader must not be called")

    monkeypatch.setattr(hf_datasets, "load_dataset", must_not_load)
    config = MixConfig(
        num_samples=1,
        max_seq_length=8,
        seed=1,
        sources=(SourceConfig(name="source", weight=1, dataset_id="source"),),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_calibration_bundle(config, output, TinyTokenizer())


def test_explicit_document_ids_limit_each_document_to_one_window(monkeypatch, tmp_path):
    rows = [
        {"doc_id": "duplicate", "text": "a" * 100},
        {"doc_id": "duplicate", "text": "b" * 100},
        {"doc_id": "unique", "text": "c" * 100},
    ]
    _install_fake_datasets(monkeypatch, {"documents": rows})
    output = tmp_path / "bundle"
    config = MixConfig(
        num_samples=3,
        max_seq_length=16,
        seed=4,
        sources=(
            SourceConfig(
                name="documents",
                weight=1,
                dataset_id="documents",
                format="text",
                id_column="doc_id",
            ),
        ),
    )
    with pytest.raises(ValueError, match="supplied 2 eligible windows; quota is 3"):
        prepare_calibration_bundle(config, output, TinyTokenizer())
    assert not output.exists()
