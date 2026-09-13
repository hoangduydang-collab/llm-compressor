import json

import pytest

from pipeline import aa_lcr_v11 as A
from pipeline.tests.test_aa_lcr_v11_dataset import (
    prepare_synthetic_100_question_fixture,
)


def test_help_lists_all_aa_lcr_phases(capsys):
    with pytest.raises(SystemExit, match="0"):
        A.main(["--help"])

    output = capsys.readouterr().out
    assert "prepare" in output
    assert "generate" in output
    assert "judge" in output
    assert "summarize" in output
    assert "run" in output


def test_plan_only_validates_local_fixture_without_endpoint_or_judge(
    tmp_path, monkeypatch
):
    prepared = prepare_synthetic_100_question_fixture(tmp_path, monkeypatch)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps({"served_model": "glm-5.3-w4afp8"}))
    monkeypatch.setattr(A, "prepare_dataset", lambda _path: prepared)
    monkeypatch.setattr(
        A, "fetch_server_identity", lambda _url: pytest.fail("endpoint")
    )
    monkeypatch.setattr(A, "build_openai_client", lambda: pytest.fail("judge"))

    assert A.main(
        [
            "prepare",
            "--run-id",
            "local-plan",
            "--work-dir",
            str(tmp_path / "work"),
            "--endpoint-identity-file",
            str(identity_path),
            "--plan-only",
        ]
    ) == 0
    assert (tmp_path / "work" / "local-plan" / "plan.json").is_file()


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--repeats", "0", "repeats must be between 1 and 3"),
        ("--repeats", "4", "repeats must be between 1 and 3"),
        ("--limit", "0", "limit must be between 1 and 100"),
        ("--limit", "101", "limit must be between 1 and 100"),
    ],
)
def test_cli_rejects_invalid_population_values(option, value, message, capsys):
    with pytest.raises(SystemExit, match="2") as error:
        A.main(["prepare", "--run-id", "test-run", option, value, "--plan-only"])

    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_requires_run_id():
    with pytest.raises(SystemExit, match="2"):
        A.main(["prepare", "--plan-only"])


def test_run_preflight_failure_precedes_candidate_generation(tmp_path, monkeypatch):
    prepared = A.PreparedDataset("revision", (), {}, "prompt")
    checkpoint = object()
    calls: list[str] = []
    monkeypatch.setattr(A, "_prepare_run", lambda _args: (prepared, checkpoint))
    monkeypatch.setattr(A, "_ensure_published_result", lambda *_args: False)
    monkeypatch.setattr(
        A, "build_openai_client", lambda: calls.append("client") or object()
    )

    def fail_preflight(_client):
        calls.append("preflight")
        raise A.JudgeProtocolError("preflight failed")

    monkeypatch.setattr(A, "preflight_judge", fail_preflight)
    monkeypatch.setattr(
        A, "generate_missing", lambda *_args, **_kwargs: calls.append("generate")
    )
    monkeypatch.setattr(A, "judge_missing", lambda *_args: calls.append("judge"))

    with pytest.raises(A.JudgeProtocolError, match="preflight failed"):
        A._run_phase(
            A._parser().parse_args(["run", "--run-id", "run-r1", "--judge-preflight"])
        )

    assert calls == ["client", "preflight"]


def test_judge_preflight_failure_precedes_judgment(tmp_path, monkeypatch):
    prepared = A.PreparedDataset("revision", (), {}, "prompt")
    checkpoint = object()
    calls: list[str] = []
    monkeypatch.setattr(A, "_prepare_run", lambda _args: (prepared, checkpoint))
    monkeypatch.setattr(A, "_ensure_published_result", lambda *_args: False)
    monkeypatch.setattr(
        A, "build_openai_client", lambda: calls.append("client") or object()
    )

    def fail_preflight(_client):
        calls.append("preflight")
        raise A.JudgeProtocolError("preflight failed")

    monkeypatch.setattr(A, "preflight_judge", fail_preflight)
    monkeypatch.setattr(A, "judge_missing", lambda *_args: calls.append("judge"))

    with pytest.raises(A.JudgeProtocolError, match="preflight failed"):
        A._run_phase(
            A._parser().parse_args(
                ["judge", "--run-id", "judge-r1", "--judge-preflight"]
            )
        )

    assert calls == ["client", "preflight"]


def test_generate_phase_does_not_construct_judge_client(monkeypatch):
    prepared = A.PreparedDataset("revision", (), {}, "prompt")
    checkpoint = object()
    calls: list[str] = []
    monkeypatch.setattr(A, "_prepare_run", lambda _args: (prepared, checkpoint))
    monkeypatch.setattr(A, "_ensure_published_result", lambda *_args: False)
    monkeypatch.setattr(
        A, "build_openai_client", lambda: pytest.fail("generate needs no judge key")
    )
    monkeypatch.setattr(
        A, "generate_missing", lambda *_args, **_kwargs: calls.append("generate")
    )

    args = A._parser().parse_args(["generate", "--run-id", "generate-r1"])
    assert A._run_phase(args) == 0
    assert calls == ["generate"]
