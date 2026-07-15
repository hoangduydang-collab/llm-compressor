"""CPU-only tests for the exact MiniMax-M3 empty-output replay contract."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import pipeline.m3_empty_output_replay as replay
from pipeline.m3_empty_output_replay import (
    REPLAY_CAPS,
    load_replay_attempt,
    postprocess_stages,
    run_controls,
)

ROW = {
    "attempt_uid": "8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878",
    "task": "mmlu_pro",
    "subtask": "mmlu_pro_economics",
    "doc_id": 45,
    "generation_seed": 1234,
    "response": "",
    "generation_arguments": [[
        "rendered prompt",
        {
            "until": ["Question:"],
            "max_gen_toks": 256,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 1234,
        },
    ]],
}


def _write_rows(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_runtime_config(path: Path, *, backend: str = "vllm") -> None:
    path.write_text(
        "model:\n"
        "  id: model\n"
        "eval:\n"
        f"  backend: {backend}\n"
        "  enable_thinking: true\n"
        '  think_end_token: "</mm:think>"\n'
        "  gen_kwargs:\n"
        "    temperature: 1.0\n"
        "    top_p: 0.95\n"
        "    do_sample: true\n"
        "    max_gen_toks: 16384\n",
        encoding="utf-8",
    )


def test_load_replay_attempt_returns_exact_normalized_request(tmp_path: Path):
    path = tmp_path / "attempts.jsonl"
    _write_rows(path, [ROW])

    attempt = load_replay_attempt(path, ROW["attempt_uid"])

    assert attempt.prompt == "rendered prompt"
    assert attempt.prompt_sha256 == hashlib.sha256(b"rendered prompt").hexdigest()
    assert attempt.generation_kwargs["max_gen_toks"] == 256
    assert attempt.source_row == ROW


@pytest.mark.parametrize("matching_rows", [0, 2])
def test_load_replay_attempt_requires_exactly_one_matching_uid(
    tmp_path: Path,
    matching_rows: int,
):
    path = tmp_path / "attempts.jsonl"
    _write_rows(path, [ROW] * matching_rows)

    with pytest.raises(ValueError, match="exactly one"):
        load_replay_attempt(path, ROW["attempt_uid"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "mmlu"),
        ("subtask", "mmlu_pro_history"),
        ("doc_id", 46),
        ("generation_seed", 4321),
    ],
)
def test_load_replay_attempt_rejects_wrong_attempt_identity(
    tmp_path: Path,
    field: str,
    value: object,
):
    path = tmp_path / "attempts.jsonl"
    row = copy.deepcopy(ROW)
    row[field] = value
    _write_rows(path, [row])

    with pytest.raises(ValueError, match=field):
        load_replay_attempt(path, ROW["attempt_uid"])


def test_load_replay_attempt_requires_original_empty_response(tmp_path: Path):
    path = tmp_path / "attempts.jsonl"
    row = copy.deepcopy(ROW)
    row["response"] = "generated text"
    _write_rows(path, [row])

    with pytest.raises(ValueError, match="response"):
        load_replay_attempt(path, ROW["attempt_uid"])


@pytest.mark.parametrize(
    "generation_arguments",
    [
        None,
        [],
        [["rendered prompt", ROW["generation_arguments"][0][1]]] * 2,
        [["rendered prompt"]],
        [["rendered prompt", ROW["generation_arguments"][0][1], "extra"]],
        [[123, ROW["generation_arguments"][0][1]]],
        [["rendered prompt", None]],
    ],
)
def test_load_replay_attempt_rejects_malformed_generation_arguments(
    tmp_path: Path,
    generation_arguments: object,
):
    path = tmp_path / "attempts.jsonl"
    row = copy.deepcopy(ROW)
    row["generation_arguments"] = generation_arguments
    _write_rows(path, [row])

    with pytest.raises(ValueError, match="generation_arguments"):
        load_replay_attempt(path, ROW["attempt_uid"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("until", ["Answer:"]),
        ("max_gen_toks", 255),
        ("do_sample", False),
        ("temperature", 0.9),
        ("top_p", 1.0),
        ("seed", 1235),
    ],
)
def test_load_replay_attempt_rejects_changed_pinned_generation_setting(
    tmp_path: Path,
    field: str,
    value: object,
):
    path = tmp_path / "attempts.jsonl"
    row = copy.deepcopy(ROW)
    row["generation_arguments"][0][1][field] = value
    _write_rows(path, [row])

    with pytest.raises(ValueError, match="generation"):
        load_replay_attempt(path, ROW["attempt_uid"])


def test_postprocess_stages_records_thinking_and_task_stop_boundaries():
    stages = postprocess_stages(
        "<mm:think>reasoning</mm:think>Question: hidden",
        think_end_token="</mm:think>",
        until=["Question:"],
    )

    assert stages == {
        "raw_text": "<mm:think>reasoning</mm:think>Question: hidden",
        "after_thinking": "Question: hidden",
        "after_task_stops": "",
        "thinking_marker_present": True,
        "matched_stop": "Question:",
    }


def test_postprocess_stages_uses_last_marker_lstrip_and_sequential_stops():
    stages = postprocess_stages(
        "old</mm:think>discard</mm:think>  answer END ignored STOP tail",
        think_end_token="</mm:think>",
        until=["", "STOP", "END"],
    )

    assert stages["after_thinking"] == "answer END ignored STOP tail"
    assert stages["after_task_stops"] == "answer "
    assert stages["matched_stop"] == "END"
    assert stages["thinking_marker_present"] is True


def test_postprocess_stages_preserves_text_when_no_markers_match():
    stages = postprocess_stages(
        "  plain answer",
        think_end_token="</mm:think>",
        until=["Question:"],
    )

    assert stages["after_thinking"] == "plain answer"
    assert stages["after_task_stops"] == "plain answer"
    assert stages["thinking_marker_present"] is False
    assert stages["matched_stop"] is None


def test_run_controls_uses_fixed_cap_order_and_reports_postprocessing(
    tmp_path: Path,
):
    path = tmp_path / "attempts.jsonl"
    _write_rows(path, [ROW])
    attempt = load_replay_attempt(path, ROW["attempt_uid"])
    calls = []

    def generate(replay_attempt, cap):
        calls.append((replay_attempt, cap))
        return {"raw_text": f"answer {cap}Question: hidden", "tokens": cap // 2}

    controls = run_controls(attempt, generate)

    assert [cap for _, cap in calls] == list(REPLAY_CAPS) == [256, 16384]
    assert all(replay_attempt is attempt for replay_attempt, _ in calls)
    assert [control["max_gen_toks"] for control in controls] == [256, 16384]
    assert [control["tokens"] for control in controls] == [128, 8192]
    assert [
        control["postprocessing"]["after_task_stops"] for control in controls
    ] == ["answer 256", "answer 16384"]


def test_classify_controls_identifies_thinking_only_at_smoke_cap():
    controls = [
        {
            "max_gen_toks": 256,
            "raw_text": "<mm:think>reasoning</mm:think>",
            "token_ids": [1, 2, 3],
            "token_count": 3,
            "finish_reason": "stop",
            "stop_reason": 2,
            "postprocessing": {
                "raw_text": "<mm:think>reasoning</mm:think>",
                "after_thinking": "",
                "after_task_stops": "",
                "thinking_marker_present": True,
                "matched_stop": None,
            },
        },
        {
            "max_gen_toks": 16384,
            "raw_text": "<mm:think>reasoning</mm:think>The answer is (C).",
            "token_ids": [1, 2, 3, 4],
            "token_count": 4,
            "finish_reason": "stop",
            "stop_reason": 2,
            "postprocessing": {
                "raw_text": "<mm:think>reasoning</mm:think>The answer is (C).",
                "after_thinking": "The answer is (C).",
                "after_task_stops": "The answer is (C).",
                "thinking_marker_present": True,
                "matched_stop": None,
            },
        },
    ]

    assert replay.classify_controls(controls) == {
        "kind": "thinking_only_at_smoke_cap",
        "smoke_processed_empty": True,
        "production_processed_empty": False,
    }


def test_build_replay_report_preserves_identity_raw_fields_and_versions(
    tmp_path: Path,
):
    samples = tmp_path / "samples.jsonl"
    config = tmp_path / "eval.yaml"
    model = tmp_path / "checkpoint"
    _write_rows(samples, [ROW])
    config.write_text("committed config\n", encoding="utf-8")
    attempt = load_replay_attempt(samples, ROW["attempt_uid"])
    calls = []

    def generate(replay_attempt, cap):
        calls.append((replay_attempt, cap))
        return {
            "raw_text": "<mm:think>reasoning</mm:think>",
            "token_ids": [1, 2, 3],
            "token_count": 3,
            "finish_reason": "stop",
            "stop_reason": 2,
        }

    versions = {"python": "3.test", "lm_eval": "0.4.12", "vllm": "test"}
    report = replay.build_replay_report(
        attempt,
        config_path=config,
        model_path=model,
        generate=generate,
        versions=versions,
    )

    assert [cap for _, cap in calls] == [256, 16384]
    assert all(replay_attempt is attempt for replay_attempt, _ in calls)
    assert report["schema_version"] == 1
    assert report["attempt_uid"] == ROW["attempt_uid"]
    assert report["task"] == "mmlu_pro"
    assert report["subtask"] == "mmlu_pro_economics"
    assert report["doc_id"] == 45
    assert report["generation_seed"] == 1234
    assert report["prompt_sha256"] == attempt.prompt_sha256
    assert report["checkpoint_path"] == str(model)
    assert report["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert report["fixed_caps"] == [256, 16384]
    assert report["versions"] == versions
    assert report["controls"][0]["token_ids"] == [1, 2, 3]
    assert report["controls"][0]["finish_reason"] == "stop"
    assert report["controls"][0]["stop_reason"] == 2
    assert report["classification"] == {
        "kind": "processed_empty_unclassified",
        "smoke_processed_empty": True,
        "production_processed_empty": True,
    }


def test_write_replay_report_replaces_temporary_file_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    output = tmp_path / "report.json"
    replacements = []
    original_replace = Path.replace

    def recording_replace(source, target):
        replacements.append((source, target))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    replay.write_replay_report(output, {"schema_version": 1})

    assert replacements == [(tmp_path / "report.json.tmp", output)]
    assert json.loads(output.read_text(encoding="utf-8")) == {"schema_version": 1}
    assert not (tmp_path / "report.json.tmp").exists()


def test_cli_parser_exposes_only_approved_inputs():
    parser = replay._build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }

    assert options == {"--config", "--model", "--samples", "--attempt-uid", "--out"}
    assert "--min-tokens" not in options
    assert "--max-gen-toks" not in options


def test_validation_before_model_load(tmp_path: Path):
    config = tmp_path / "invalid.yaml"
    _write_runtime_config(config, backend="sglang")
    loader_calls = []

    def model_loader(cfg, model_path):
        loader_calls.append((cfg, model_path))
        raise AssertionError("model loader must not run")

    with pytest.raises(ValueError, match="backend"):
        replay.load_raw_vllm_generator(
            config,
            tmp_path / "checkpoint",
            model_loader=model_loader,
            version_getter=lambda name: "0.4.12",
        )

    assert loader_calls == []


def test_runtime_metadata_validation_before_model_load(tmp_path: Path):
    config = tmp_path / "eval.yaml"
    _write_runtime_config(config)
    loader_calls = []

    def version_getter(name):
        if name in {"lm_eval", "lm-eval"}:
            return "0.4.12"
        raise replay.importlib.metadata.PackageNotFoundError(name)

    def model_loader(cfg, model_path):
        loader_calls.append((cfg, model_path))
        raise AssertionError("model loader must not run")

    with pytest.raises(replay.importlib.metadata.PackageNotFoundError):
        replay.load_raw_vllm_generator(
            config,
            tmp_path / "checkpoint",
            model_loader=model_loader,
            version_getter=version_getter,
        )

    assert loader_calls == []


def test_raw_vllm_runtime_uses_pinned_adapter_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = tmp_path / "eval.yaml"
    samples = tmp_path / "samples.jsonl"
    _write_runtime_config(config)
    _write_rows(samples, [ROW])
    attempt = load_replay_attempt(samples, ROW["attempt_uid"])
    truncate_calls = []
    sampling_params = []

    def maybe_truncate(token_ids, **kwargs):
        truncate_calls.append((token_ids, kwargs))
        return token_ids, kwargs["max_gen_toks"]

    class SamplingParams:
        def __init__(self, **kwargs):
            sampling_params.append(kwargs)

    utils_module = types.ModuleType("lm_eval.models.utils")
    utils_module.maybe_truncate = maybe_truncate
    monkeypatch.setitem(sys.modules, "lm_eval", types.ModuleType("lm_eval"))
    monkeypatch.setitem(
        sys.modules, "lm_eval.models", types.ModuleType("lm_eval.models")
    )
    monkeypatch.setitem(sys.modules, "lm_eval.models.utils", utils_module)
    vllm_module = types.ModuleType("vllm")
    vllm_module.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)

    class FakeLm:
        eot_token_id = 99
        max_gen_toks = 4096
        max_length = 65536
        truncation_side = "left"

        def __init__(self):
            self.generate_calls = []
            self.clean_calls = 0

        def tok_decode(self, token_id):
            assert token_id == 99
            return "<eos>"

        def tok_encode(self, prompt):
            assert prompt == "rendered prompt"
            return [10, 11]

        def modify_gen_kwargs(self, kwargs, **defaults):
            assert defaults == {"eos": "<eos>", "default_max_gen_toks": 4096}
            cap = kwargs.pop("max_gen_toks")
            until = kwargs.pop("until") + ["<eos>"]
            kwargs.pop("do_sample")
            return kwargs, until, cap

        def _model_generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            completion = types.SimpleNamespace(
                text="<mm:think>reasoning</mm:think>",
                token_ids=(1, 2, 3),
                finish_reason="stop",
                stop_reason=2,
            )
            return [types.SimpleNamespace(outputs=[completion])]

        def clean(self):
            self.clean_calls += 1

    lm = FakeLm()
    loader_calls = []

    def model_loader(cfg, model_path):
        loader_calls.append((cfg, model_path))
        return lm

    report = replay.run_raw_vllm_replay(
        attempt,
        config_path=config,
        model_path=tmp_path / "checkpoint",
        model_loader=model_loader,
        version_getter=lambda name: {
            "lm_eval": "0.4.12",
            "lm-eval": "0.4.12",
            "vllm": "0.test",
        }[name],
    )

    assert len(loader_calls) == 1
    assert [call[1]["max_gen_toks"] for call in truncate_calls] == [256, 16384]
    assert [params["max_tokens"] for params in sampling_params] == [256, 16384]
    assert all(params["stop"] == ["<eos>"] for params in sampling_params)
    assert report["controls"][0]["token_ids"] == [1, 2, 3]
    assert report["controls"][0]["token_count"] == 3
    assert report["versions"]["lm_eval"] == "0.4.12"
    assert report["versions"]["vllm"] == "0.test"
    assert lm.clean_calls == 1


def test_cli_returns_nonzero_and_writes_no_report_for_invalid_source(tmp_path: Path):
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "report.json"
    _write_rows(samples, [ROW])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline.m3_empty_output_replay",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--model",
            str(tmp_path / "checkpoint"),
            "--samples",
            str(samples),
            "--attempt-uid",
            "wrong-uid",
            "--out",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "expected exactly one row" in completed.stderr
    assert not output.exists()
