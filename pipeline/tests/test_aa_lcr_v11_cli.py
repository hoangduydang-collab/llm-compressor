import json
from types import SimpleNamespace

import pytest

from pipeline import aa_lcr_v11 as A
from pipeline.tests.test_aa_lcr_v11_candidate import seeded_checkpoint
from pipeline.tests.test_aa_lcr_v11_checkpoint import preflight_audit
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
    identity_path.write_text(
        json.dumps(
            {
                "served_model": A.CANDIDATE_MODEL,
                "expected_served_model": A.CANDIDATE_MODEL,
                "observed_served_model": A.CANDIDATE_MODEL,
            }
        )
    )
    monkeypatch.setattr(A, "prepare_dataset", lambda _path: prepared)
    monkeypatch.setattr(
        A, "fetch_server_identity", lambda _url: pytest.fail("endpoint")
    )
    monkeypatch.setattr(A, "build_openai_client", lambda: pytest.fail("judge"))

    assert (
        A.main(
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
        )
        == 0
    )
    assert (tmp_path / "work" / "local-plan" / "plan.json").is_file()


def test_plan_only_rejects_wrong_candidate_identity(tmp_path, monkeypatch):
    prepared = prepare_synthetic_100_question_fixture(tmp_path, monkeypatch)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(
        json.dumps(
            {
                "served_model": "glm-5.3-w4afp8-shadow",
                "expected_served_model": A.CANDIDATE_MODEL,
                "observed_served_model": "glm-5.3-w4afp8-shadow",
            }
        )
    )
    monkeypatch.setattr(A, "prepare_dataset", lambda _path: prepared)

    with pytest.raises(SystemExit, match="2"):
        A.main(
            [
                "prepare",
                "--run-id",
                "bad-plan",
                "--work-dir",
                str(tmp_path / "work"),
                "--endpoint-identity-file",
                str(identity_path),
                "--plan-only",
            ]
        )

    assert not (tmp_path / "work" / "bad-plan" / "run.sqlite").exists()


def test_cli_defaults_candidate_sampling_to_aa_reasoning_convention():
    args = A._parser().parse_args(["prepare", "--run-id", "aa-default"])

    assert args.candidate_temperature == 0.6
    assert args.candidate_top_p == 1.0


def test_cli_accepts_phala_sampling():
    args = A._parser().parse_args(
        [
            "prepare",
            "--run-id",
            "phala",
            "--candidate-temperature",
            "1.0",
            "--candidate-top-p",
            "0.95",
        ]
    )

    assert args.candidate_temperature == 1.0
    assert args.candidate_top_p == 0.95


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("-0.1", "candidate-temperature must be between 0 and 2"),
        ("2.1", "candidate-temperature must be between 0 and 2"),
        ("nan", "candidate-temperature must be a finite number"),
        ("1.1", "candidate-top-p must be between 0 and 1"),
    ],
)
def test_cli_rejects_invalid_candidate_sampling(value, message, capsys):
    flag = (
        "--candidate-top-p"
        if "top-p" in message
        else "--candidate-temperature"
    )
    with pytest.raises(SystemExit, match="2"):
        A.main(
            [
                "prepare",
                "--run-id",
                "bad-sampling",
                flag,
                value,
                "--plan-only",
            ]
        )

    assert message in capsys.readouterr().err


def test_prepare_run_pins_overridden_temperature_in_contract(tmp_path, monkeypatch):
    prepared = prepare_synthetic_100_question_fixture(tmp_path, monkeypatch)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(
        json.dumps(
            {
                "served_model": A.CANDIDATE_MODEL,
                "expected_served_model": A.CANDIDATE_MODEL,
                "observed_served_model": A.CANDIDATE_MODEL,
            }
        )
    )
    monkeypatch.setattr(A, "prepare_dataset", lambda _path: prepared)

    args = A._parser().parse_args(
        [
            "prepare",
            "--run-id",
            "t1-plan",
            "--work-dir",
            str(tmp_path / "work"),
            "--endpoint-identity-file",
            str(identity_path),
            "--candidate-temperature",
            "1.0",
            "--candidate-top-p",
            "0.95",
            "--plan-only",
        ]
    )
    _prepared, checkpoint = A._prepare_run(args)

    assert checkpoint.contract.candidate_temperature == 1.0
    assert checkpoint.contract.candidate_top_p == 0.95


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
    checkpoint = SimpleNamespace(preflight_audit=lambda: None)
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
    checkpoint = SimpleNamespace(preflight_audit=lambda: None)
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


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--run-id", "partial", "--limit", "1", "--repeats", "1"],
        ["run", "--run-id", "partial", "--limit", "99", "--repeats", "3"],
        ["run", "--run-id", "partial", "--limit", "100", "--repeats", "2"],
        ["run", "--run-id", "bad-canary", "--canary", "--limit", "2", "--repeats", "1"],
        ["run", "--run-id", "bad-canary", "--canary", "--limit", "1", "--repeats", "2"],
        [
            "run",
            "--run-id",
            "bad-canary",
            "--canary",
            "--limit",
            "1",
            "--repeats",
            "1",
            "--plan-only",
        ],
        [
            "generate",
            "--run-id",
            "bad-phase",
            "--canary",
            "--limit",
            "1",
            "--repeats",
            "1",
        ],
    ],
)
def test_run_rejects_nonstandard_population_before_network(argv, monkeypatch):
    monkeypatch.setattr(
        A, "_prepare_run", lambda _args: pytest.fail("network preparation started")
    )

    with pytest.raises(SystemExit, match="2"):
        A.main(argv)


def test_canary_preflights_then_runs_one_unit_without_publication(
    tmp_path, monkeypatch, capsys
):
    prepared = A.PreparedDataset("revision", (), {}, "prompt")
    calls: list[str] = []
    checkpoint = SimpleNamespace(
        preflight_audit=lambda: None,
        record_preflight_audit=lambda _audit: calls.append("persist-preflight")
    )
    monkeypatch.setattr(A, "_prepare_run", lambda _args: (prepared, checkpoint))
    monkeypatch.setattr(A, "_ensure_published_result", lambda *_args: False)
    monkeypatch.setattr(
        A, "build_openai_client", lambda: calls.append("client") or object()
    )
    monkeypatch.setattr(
        A, "preflight_judge", lambda _client: calls.append("preflight") or {}
    )
    monkeypatch.setattr(
        A, "generate_missing", lambda *_args, **_kwargs: calls.append("generate")
    )
    monkeypatch.setattr(A, "judge_missing", lambda *_args: calls.append("judge"))
    monkeypatch.setattr(
        A, "publish_results", lambda *_args: pytest.fail("canary published a headline")
    )
    monkeypatch.setattr(
        A,
        "_canary_status",
        lambda *_args: calls.append("status")
        or {
            "mode": "canary",
            "status": "complete",
            "run_id": "canary-r1",
            "candidate_count": 1,
            "judgment_count": 1,
            "published": False,
        },
    )

    args = A._parser().parse_args(
        [
            "run",
            "--run-id",
            "canary-r1",
            "--work-dir",
            str(tmp_path),
            "--canary",
            "--limit",
            "1",
            "--repeats",
            "1",
        ]
    )
    assert A._run_phase(args) == 0

    assert calls == [
        "client",
        "preflight",
        "persist-preflight",
        "generate",
        "judge",
        "status",
    ]
    assert json.loads(capsys.readouterr().out) == {
        "mode": "canary",
        "status": "complete",
        "run_id": "canary-r1",
        "candidate_count": 1,
        "judgment_count": 1,
        "published": False,
    }


def test_canary_resume_reuses_immutable_preflight_audit(monkeypatch, capsys):
    prepared = A.PreparedDataset("revision", (), {}, "prompt")
    audit = preflight_audit()
    checkpoint = SimpleNamespace(
        preflight_audit=lambda: audit,
        record_preflight_audit=lambda _audit: pytest.fail("rewrote preflight"),
    )
    calls: list[str] = []
    monkeypatch.setattr(A, "_prepare_run", lambda _args: (prepared, checkpoint))
    monkeypatch.setattr(A, "build_openai_client", lambda: object())
    monkeypatch.setattr(
        A, "preflight_judge", lambda _client: pytest.fail("repeated preflight API call")
    )
    monkeypatch.setattr(
        A, "generate_missing", lambda *_args, **_kwargs: calls.append("generate")
    )
    monkeypatch.setattr(A, "judge_missing", lambda *_args: calls.append("judge"))
    monkeypatch.setattr(
        A,
        "_canary_status",
        lambda *_args: {"mode": "canary", "status": "complete"},
    )

    args = A._parser().parse_args(
        [
            "run",
            "--run-id",
            "canary-r1",
            "--canary",
            "--limit",
            "1",
            "--repeats",
            "1",
        ]
    )
    assert A._run_phase(args) == 0

    assert calls == ["generate", "judge"]
    capsys.readouterr()


def test_full_run_preflights_before_generation_without_optional_flag(
    monkeypatch,
):
    prepared = A.PreparedDataset("revision", (), {}, "prompt")
    calls: list[str] = []
    checkpoint = SimpleNamespace(
        preflight_audit=lambda: None,
        record_preflight_audit=lambda _audit: calls.append("persist-preflight"),
    )
    monkeypatch.setattr(A, "_prepare_run", lambda _args: (prepared, checkpoint))
    monkeypatch.setattr(A, "_ensure_published_result", lambda *_args: False)
    monkeypatch.setattr(A, "build_openai_client", lambda: object())
    monkeypatch.setattr(
        A, "preflight_judge", lambda _client: calls.append("preflight") or {}
    )
    monkeypatch.setattr(
        A, "generate_missing", lambda *_args, **_kwargs: calls.append("generate")
    )
    monkeypatch.setattr(A, "judge_missing", lambda *_args: calls.append("judge"))
    monkeypatch.setattr(
        A, "publish_results", lambda *_args: calls.append("publish")
    )

    assert A._run_phase(A._parser().parse_args(["run", "--run-id", "full-r1"])) == 0

    assert calls == [
        "preflight",
        "persist-preflight",
        "generate",
        "judge",
        "publish",
    ]


def test_canary_status_points_to_complete_durable_checkpoint(tmp_path):
    checkpoint = seeded_checkpoint(tmp_path)
    checkpoint.record_preflight_audit(preflight_audit())
    checkpoint.record_candidate(
        A.CandidateRecord(
            question_id=1,
            repeat_index=0,
            http_status=200,
            raw_response={},
            usage={},
            finish_reason="stop",
            content="answer",
            reasoning_content="reasoning",
            retry_count=0,
            started_at_utc="2026-09-13T00:00:00Z",
            completed_at_utc="2026-09-13T00:00:01Z",
            error=None,
        )
    )
    checkpoint.record_judgment(
        A.JudgmentRecord(
            question_id=1,
            repeat_index=0,
            judge_contract_hash=A.validate_judge_contract(checkpoint.contract),
            raw_response={},
            verdict="CORRECT",
            retry_count=0,
            started_at_utc="2026-09-13T00:00:02Z",
            completed_at_utc="2026-09-13T00:00:03Z",
            error=None,
        )
    )
    args = A._parser().parse_args(
        [
            "run",
            "--run-id",
            "canary-r1",
            "--canary",
            "--limit",
            "1",
            "--repeats",
            "1",
        ]
    )

    status = A._canary_status(checkpoint, args)

    assert status["status"] == "complete"
    assert status["checkpoint"] == str(checkpoint.path)
    assert status["preflight_recorded"] is True
    assert status["candidate_count"] == 1
    assert status["judgment_count"] == 1
    assert status["judge_terminal_failure_count"] == 0
    assert status["published"] is False
