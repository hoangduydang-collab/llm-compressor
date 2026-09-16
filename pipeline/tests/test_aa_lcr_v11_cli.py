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


def test_cli_defaults_candidate_sampling_to_the_lab_override_branch():
    # AA's generic default is 0.6/1.0, but its own rule defers to the model
    # creator when the lab publishes a config, and Z.ai publishes 1.0/0.95.
    args = A._parser().parse_args(["prepare", "--run-id", "aa-default"])

    assert args.candidate_temperature == 1.0
    assert args.candidate_top_p == 0.95


def test_cli_accepts_aa_generic_sampling():
    args = A._parser().parse_args(
        [
            "prepare",
            "--run-id",
            "aa-generic",
            "--candidate-temperature",
            "0.6",
            "--candidate-top-p",
            "1.0",
        ]
    )

    assert args.candidate_temperature == 0.6
    assert args.candidate_top_p == 1.0


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
            "generic-plan",
            "--work-dir",
            str(tmp_path / "work"),
            "--endpoint-identity-file",
            str(identity_path),
            "--candidate-temperature",
            "0.6",
            "--candidate-top-p",
            "1.0",
            "--plan-only",
        ]
    )
    _prepared, checkpoint = A._prepare_run(args)

    assert checkpoint.contract.candidate_temperature == 0.6
    assert checkpoint.contract.candidate_top_p == 1.0
    assert not checkpoint.contract.uses_public_methodology_sampling


def test_cli_defaults_output_cap_and_concurrency_to_the_aa_recipe():
    args = A._parser().parse_args(["prepare", "--run-id", "aa-default"])

    assert args.candidate_max_tokens == 131_072
    assert args.candidate_concurrency == 2
    assert args.candidate_long_attempt_seconds == 900


def test_cli_accepts_a_doubled_output_cap_and_a_single_slot():
    args = A._parser().parse_args(
        [
            "prepare",
            "--run-id",
            "cap-2x",
            "--candidate-max-tokens",
            "262144",
            "--candidate-concurrency",
            "1",
            "--candidate-long-attempt-seconds",
            "1200",
        ]
    )

    assert args.candidate_max_tokens == 262_144
    assert args.candidate_concurrency == 1
    assert args.candidate_long_attempt_seconds == 1200


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--candidate-max-tokens", "0", "candidate-max-tokens must be between"),
        ("--candidate-max-tokens", "524289", "candidate-max-tokens must be between"),
        ("--candidate-concurrency", "0", "candidate-concurrency must be between"),
        ("--candidate-concurrency", "3", "candidate-concurrency must be between"),
    ],
)
def test_cli_rejects_out_of_range_cap_and_concurrency(option, value, message, capsys):
    with pytest.raises(SystemExit, match="2"):
        A.main(["prepare", "--run-id", "test-run", option, value, "--plan-only"])

    assert message in capsys.readouterr().err


def test_prepare_run_pins_overridden_cap_and_gate_in_contract(tmp_path, monkeypatch):
    prepared = prepare_synthetic_100_question_fixture(tmp_path, monkeypatch)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(
        json.dumps(
            {
                "served_model": A.CANDIDATE_MODEL,
                "expected_served_model": A.CANDIDATE_MODEL,
                "observed_served_model": A.CANDIDATE_MODEL,
                "max_total_num_tokens": 598848,
            }
        )
    )
    monkeypatch.setattr(A, "prepare_dataset", lambda _path: prepared)

    args = A._parser().parse_args(
        [
            "prepare",
            "--run-id",
            "cap-2x-plan",
            "--work-dir",
            str(tmp_path / "work"),
            "--endpoint-identity-file",
            str(identity_path),
            "--candidate-max-tokens",
            "262144",
            "--candidate-long-attempt-seconds",
            "1200",
            "--plan-only",
        ]
    )
    _prepared, checkpoint = A._prepare_run(args)

    assert checkpoint.contract.candidate_max_tokens == 262_144
    assert checkpoint.contract.candidate_long_attempt_seconds == 1200
    assert not checkpoint.contract.uses_public_methodology_sampling


def _budget_dataset(prompt_tokens: int) -> A.PreparedDataset:
    return A.PreparedDataset(
        revision="revision",
        questions=(),
        file_sha256={},
        prompt_sha256="b" * 64,
        question_input_tokens={1: (prompt_tokens, prompt_tokens)},
    )


def _budget_contract(prompt_tokens: int, pool: int, cap: int) -> A.RunContract:
    return A.build_run_contract(
        _budget_dataset(prompt_tokens),
        candidate_model=A.CANDIDATE_MODEL,
        served_model=A.CANDIDATE_MODEL,
        endpoint_deployment_identity={"max_total_num_tokens": pool},
        code_revision="test",
        candidate_max_tokens=cap,
    )


def test_build_run_contract_rejects_a_cap_the_kv_pool_cannot_hold():
    with pytest.raises(A.CheckpointConflictError, match="max_total_num_tokens"):
        _budget_contract(prompt_tokens=114_611, pool=598_848, cap=524_288)


def test_build_run_contract_accepts_the_doubled_cap_on_the_two_node_pool():
    contract = _budget_contract(prompt_tokens=114_611, pool=598_848, cap=262_144)

    assert contract.candidate_max_tokens == 262_144


def test_build_run_contract_skips_the_budget_check_without_a_reported_pool():
    contract = A.build_run_contract(
        _budget_dataset(114_611),
        candidate_model=A.CANDIDATE_MODEL,
        served_model=A.CANDIDATE_MODEL,
        endpoint_deployment_identity={"tp_size": 16},
        code_revision="test",
        candidate_max_tokens=524_288,
    )

    assert contract.candidate_max_tokens == 524_288


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
