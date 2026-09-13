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
