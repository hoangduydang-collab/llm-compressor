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

class _EosTokenizer:
    """Fake of the lm-eval VLLM wrapper's `.tokenizer`: decode(eot_id) -> eos."""

    def __init__(self, expected_id: int):
        self._expected_id = expected_id

    def decode(self, token_id):
        assert token_id == self._expected_id
        return "<eos>"


EVIDENCE_PATH = (
    Path(__file__).parents[2]
    / "results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/models"
    / "inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl"
)
PINNED_ATTEMPT_UID = (
    "8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878"
)
PINNED_PROMPT_SHA256 = (
    "006f5eef8c151c3e7418a047e8a171a33bad2d77fad989a8b7f1552648d60b93"
)


def _load_evidence_prompt() -> str:
    rows = [json.loads(line) for line in EVIDENCE_PATH.read_text().splitlines()]
    row = next(row for row in rows if row.get("attempt_uid") == PINNED_ATTEMPT_UID)
    return row["generation_arguments"][0][0]


PINNED_PROMPT = _load_evidence_prompt()

ROW = {
    "attempt_uid": PINNED_ATTEMPT_UID,
    "task": "mmlu_pro",
    "subtask": "mmlu_pro_economics",
    "doc_id": 45,
    "generation_seed": 1234,
    "response": "",
    "generation_arguments": [[
        PINNED_PROMPT,
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

    assert attempt.prompt == PINNED_PROMPT
    assert attempt.prompt_sha256 == PINNED_PROMPT_SHA256
    assert hashlib.sha256(PINNED_PROMPT.encode()).hexdigest() == PINNED_PROMPT_SHA256
    assert attempt.generation_kwargs["max_gen_toks"] == 256
    assert attempt.source_row == ROW


def test_load_replay_attempt_rejects_unpinned_requested_uid(tmp_path: Path):
    path = tmp_path / "attempts.jsonl"
    row = copy.deepcopy(ROW)
    row["attempt_uid"] = "not-the-pinned-attempt"
    _write_rows(path, [row])

    with pytest.raises(ValueError, match="pinned attempt UID"):
        load_replay_attempt(path, row["attempt_uid"])


def test_load_replay_attempt_rejects_rendered_prompt_drift(tmp_path: Path):
    path = tmp_path / "attempts.jsonl"
    row = copy.deepcopy(ROW)
    row["generation_arguments"][0][0] += " drift"
    _write_rows(path, [row])

    with pytest.raises(ValueError, match="prompt SHA-256"):
        load_replay_attempt(path, PINNED_ATTEMPT_UID)


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


def test_raw_vllm_generator_uses_tokenizer_decode_not_tok_decode(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression: the raw generator must derive eos via lm.tokenizer.decode,
    mirroring lm-eval's VLLM.generate_until. The pinned VLLM wrapper has no
    tok_decode method, so the earlier tok_decode call raised AttributeError on
    GPU after the model had already loaded. Guard it with a fake lm that (like
    the real wrapper) exposes tokenizer.decode but no tok_decode.
    """
    fake_vllm = types.ModuleType("vllm")

    class _SamplingParams:
        def __init__(self, max_tokens, stop, **kwargs):
            self.max_tokens = max_tokens
            self.stop = stop
            self.kwargs = kwargs

    fake_vllm.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    class _Out:
        token_ids = [10, 11]
        text = "answer"
        finish_reason = "stop"
        stop_reason = 2

    class _Req:
        outputs = [_Out()]

    class _FakeLM:
        # Deliberately no `tok_decode`; a regression to it would AttributeError.
        tokenizer = _EosTokenizer(7)
        eot_token_id = 7
        max_gen_toks = 256
        max_length = 4096
        truncation_side = "left"

        def modify_gen_kwargs(self, kwargs, *, eos, default_max_gen_toks):
            return ({"temperature": 0.0}, [eos], kwargs["max_gen_toks"])

        def tok_encode(self, prompt):
            return [1, 2, 3]

        def _model_generate(self, *, requests, generate, sampling_params):
            assert generate is True
            assert sampling_params[0].stop == ["<eos>"]
            return [_Req()]

    generator = replay._RawVllmGenerator(_FakeLM(), versions={})
    attempt = replay.ReplayAttempt(
        attempt_uid="x",
        prompt="p",
        prompt_sha256="s",
        generation_kwargs={"until": ["<eos>"]},
        source_row={},
    )

    result = generator(attempt, 256)

    assert result["raw_text"] == "answer"
    assert result["token_ids"] == [10, 11]
    assert result["token_count"] == 2
    assert result["finish_reason"] == "stop"
    assert result["effective_generation_arguments"]["effective_max_tokens"] == 256


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


def test_write_replay_report_rejects_non_finite_json(tmp_path: Path):
    output = tmp_path / "report.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        replay.write_replay_report(output, {"stop_reason": float("nan")})

    assert not output.exists()
    assert not (tmp_path / "report.json.tmp").exists()


def test_json_safe_scalar_stringifies_non_finite_float():
    assert replay._json_safe_scalar(float("nan")) == "nan"
    assert replay._json_safe_scalar(float("inf")) == "inf"
    assert replay._json_safe_scalar(float("-inf")) == "-inf"


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


def test_cli_rejects_resolved_output_samples_alias_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    samples = tmp_path / "samples.jsonl"
    _write_rows(samples, [ROW])
    source_bytes = samples.read_bytes()
    model_load_calls = []

    def unexpected_replay(*args, **kwargs):
        model_load_calls.append((args, kwargs))
        return {"schema_version": 1}

    monkeypatch.setattr(replay, "run_raw_vllm_replay", unexpected_replay)
    monkeypatch.chdir(tmp_path)

    try:
        with pytest.raises(ValueError, match="different files"):
            replay.main(
                [
                    "--config",
                    "eval.yaml",
                    "--model",
                    "checkpoint",
                    "--samples",
                    str(samples.resolve()),
                    "--attempt-uid",
                    ROW["attempt_uid"],
                    "--out",
                    "samples.jsonl",
                ]
            )
    finally:
        assert samples.read_bytes() == source_bytes

    assert model_load_calls == []


@pytest.mark.parametrize(
    "relative_output",
    [
        "diagnostics/samples/replay.jsonl",
        "diagnostics/aggregate.json",
        "diagnostics/matrix.json",
        "diagnostics/gates.json",
        "diagnostics/manifest.json",
        "diagnostics/generation-health.json",
    ],
)
def test_cli_rejects_benchmark_artifact_targets_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_output: str,
):
    run_root = tmp_path / "run"
    samples = (
        run_root
        / "models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl"
    )
    samples.parent.mkdir(parents=True)
    _write_rows(samples, [ROW])
    replay_calls = []
    monkeypatch.setattr(
        replay,
        "run_raw_vllm_replay",
        lambda *args, **kwargs: replay_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="diagnostic replay sidecar"):
        replay.main(
            [
                "--config",
                str(tmp_path / "eval.yaml"),
                "--model",
                str(tmp_path / "checkpoint"),
                "--samples",
                str(samples),
                "--attempt-uid",
                PINNED_ATTEMPT_UID,
                "--out",
                str(run_root / relative_output),
            ]
        )

    assert replay_calls == []


@pytest.mark.parametrize(
    "relative_output",
    [
        "diagnostics/empty-output-replay.json",
        "replays/empty-output-replay.json",
    ],
)
def test_cli_permits_benchmark_diagnostic_sidecar_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_output: str,
):
    run_root = tmp_path / "run"
    samples = (
        run_root
        / "models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl"
    )
    samples.parent.mkdir(parents=True)
    _write_rows(samples, [ROW])
    replay_calls = []
    monkeypatch.setattr(
        replay,
        "run_raw_vllm_replay",
        lambda *args, **kwargs: replay_calls.append((args, kwargs))
        or {"schema_version": 1},
    )

    assert replay.main(
        [
            "--config",
            str(tmp_path / "eval.yaml"),
            "--model",
            str(tmp_path / "checkpoint"),
            "--samples",
            str(samples),
            "--attempt-uid",
            PINNED_ATTEMPT_UID,
            "--out",
            str(run_root / relative_output),
        ]
    ) == 0

    assert len(replay_calls) == 1
    assert json.loads((run_root / relative_output).read_text()) == {
        "schema_version": 1
    }


def test_cli_rejects_symlink_alias_to_benchmark_artifact_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    run_root = tmp_path / "run"
    samples = (
        run_root
        / "models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl"
    )
    samples.parent.mkdir(parents=True)
    _write_rows(samples, [ROW])
    aggregate = run_root / "aggregate.json"
    aggregate.write_text('{"scientific": true}\n', encoding="utf-8")
    diagnostics = run_root / "diagnostics"
    diagnostics.mkdir()
    alias = diagnostics / "empty-output-replay.json"
    try:
        alias.symlink_to(aggregate)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    replay_calls = []
    monkeypatch.setattr(
        replay,
        "run_raw_vllm_replay",
        lambda *args, **kwargs: replay_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="diagnostic replay sidecar"):
        replay.main(
            [
                "--config",
                str(tmp_path / "eval.yaml"),
                "--model",
                str(tmp_path / "checkpoint"),
                "--samples",
                str(samples),
                "--attempt-uid",
                PINNED_ATTEMPT_UID,
                "--out",
                str(alias),
            ]
        )

    assert replay_calls == []
    assert aggregate.read_text(encoding="utf-8") == '{"scientific": true}\n'


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
        return token_ids, kwargs["max_gen_toks"] - 1

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

        # Real lm-eval VLLM wrapper exposes tokenizer.decode, not tok_decode.
        tokenizer = _EosTokenizer(99)

        def tok_encode(self, prompt):
            assert prompt == PINNED_PROMPT
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
    assert [params["max_tokens"] for params in sampling_params] == [255, 16383]
    assert all(params["stop"] == ["<eos>"] for params in sampling_params)
    assert report["controls"][0]["token_ids"] == [1, 2, 3]
    assert report["controls"][0]["token_count"] == 3
    effective_arguments = [
        control["effective_generation_arguments"] for control in report["controls"]
    ]
    assert effective_arguments == [
        {
            "normalized_sampling_kwargs": {
                "temperature": 1.0,
                "top_p": 0.95,
                "seed": 1234,
            },
            "original_stops": ["Question:"],
            "effective_stops": ["Question:", "<eos>"],
            "model_stops": ["<eos>"],
            "effective_max_tokens": 255,
        },
        {
            "normalized_sampling_kwargs": {
                "temperature": 1.0,
                "top_p": 0.95,
                "seed": 1234,
            },
            "original_stops": ["Question:"],
            "effective_stops": ["Question:", "<eos>"],
            "model_stops": ["<eos>"],
            "effective_max_tokens": 16383,
        },
    ]
    json.dumps(report["controls"], allow_nan=False)
    assert report["versions"]["lm_eval"] == "0.4.12"
    assert report["versions"]["vllm"] == "0.test"
    assert lm.clean_calls == 1


def test_raw_vllm_runtime_cleans_up_when_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    samples = tmp_path / "samples.jsonl"
    _write_rows(samples, [ROW])
    attempt = load_replay_attempt(samples, ROW["attempt_uid"])

    utils_module = types.ModuleType("lm_eval.models.utils")
    utils_module.maybe_truncate = lambda token_ids, **kwargs: (
        token_ids,
        kwargs["max_gen_toks"],
    )
    monkeypatch.setitem(sys.modules, "lm_eval", types.ModuleType("lm_eval"))
    monkeypatch.setitem(
        sys.modules, "lm_eval.models", types.ModuleType("lm_eval.models")
    )
    monkeypatch.setitem(sys.modules, "lm_eval.models.utils", utils_module)
    vllm_module = types.ModuleType("vllm")
    vllm_module.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)

    class FailingLm:
        eot_token_id = 99
        max_gen_toks = 4096
        max_length = 65536
        truncation_side = "left"

        tokenizer = _EosTokenizer(99)

        def __init__(self):
            self.clean_calls = 0

        def tok_encode(self, prompt):
            return [10, 11]

        def modify_gen_kwargs(self, kwargs, **defaults):
            cap = kwargs.pop("max_gen_toks")
            kwargs.pop("until")
            kwargs.pop("do_sample")
            return kwargs, ["<eos>"], cap

        def _model_generate(self, **kwargs):
            raise RuntimeError("generation failed")

        def clean(self):
            self.clean_calls += 1

    lm = FailingLm()
    generator = replay._RawVllmGenerator(
        lm,
        versions={"python": "test", "lm_eval": "0.4.12", "vllm": "test"},
    )
    monkeypatch.setattr(
        replay,
        "load_raw_vllm_generator",
        lambda *args, **kwargs: generator,
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        replay.run_raw_vllm_replay(
            attempt,
            config_path=tmp_path / "eval.yaml",
            model_path=tmp_path / "checkpoint",
        )

    assert lm.clean_calls == 1


def test_raw_vllm_generator_falls_back_to_cleanup():
    class CleanupOnlyLm:
        def __init__(self):
            self.cleanup_calls = 0

        def cleanup(self):
            self.cleanup_calls += 1

    lm = CleanupOnlyLm()
    generator = replay._RawVllmGenerator(lm, versions={})

    generator.close()
    generator.close()

    assert lm.cleanup_calls == 1


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
    assert "pinned attempt UID" in completed.stderr
    assert not output.exists()
