"""CPU tests for the NVIDIA gpqa_diamond_aa_v3 client contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pipeline.aa_gpqa_v3 import (
    DEFAULT_VENV,
    MAX_NEW_TOKENS,
    NEMO_EVALUATOR_PIN,
    PACKAGE_PIN,
    REQUEST_TIMEOUT,
    TASK_NAME,
    TEMPERATURE,
    TOP_P,
    TaskMissingError,
    build_run_eval_argv,
    require_task,
    write_manifest,
    write_run_config,
)


def test_pin_and_task_are_the_inspected_wheel():
    assert PACKAGE_PIN == "nvidia-simple-evals==26.3"
    assert NEMO_EVALUATOR_PIN == "nemo-evaluator>=0.1.51,<0.3"
    assert TASK_NAME == "gpqa_diamond_aa_v3"
    assert DEFAULT_VENV == "/mnt/cephfs/hoangduy/venvs/nvidia-simple-evals-26.3"


def test_require_task_accepts_nemo_evaluator_ls_line():
    require_task("* gpqa_diamond_aa_v3 (in simple_evals)\n* gpqa_diamond_aa_v2\n")


def test_require_task_rejects_stale_readme_listing():
    stale = "* gpqa_diamond (in simple_evals)\n* gpqa_diamond_aa_v2 (in simple_evals)\n"
    with pytest.raises(TaskMissingError, match="gpqa_diamond_aa_v3"):
        require_task(stale)


def test_write_run_config_glm_max_overrides_and_thinking(tmp_path: Path):
    path = tmp_path / "run.yml"
    cfg = write_run_config(
        path,
        url="http://127.0.0.1:30000/v1/chat/completions",
        model_id="glm",
        output_dir="/tmp/aa-gpqa-v3",
    )
    on_disk = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert on_disk == cfg
    assert cfg["config"]["type"] == TASK_NAME
    params = cfg["config"]["params"]
    assert params["temperature"] == TEMPERATURE == 0.6
    assert params["top_p"] == TOP_P == 1.0
    assert params["max_new_tokens"] == MAX_NEW_TOKENS == 65536
    assert params["request_timeout"] == REQUEST_TIMEOUT == 3600
    assert params["limit_samples"] is None
    assert params["extra"]["n_samples"] == 5
    ep = cfg["target"]["api_endpoint"]
    assert ep["url"] == "http://127.0.0.1:30000/v1/chat/completions"
    assert ep["model_id"] == "glm"
    assert ep["type"] == "chat"
    assert ep["adapter_config"]["params_to_add"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }


def test_write_run_config_canary_limit(tmp_path: Path):
    path = tmp_path / "canary.yml"
    cfg = write_run_config(
        path,
        url="http://127.0.0.1:30000/v1/chat/completions",
        model_id="glm",
        output_dir="/tmp/aa-gpqa-v3",
        limit_samples=2,
    )
    assert cfg["config"]["params"]["limit_samples"] == 2


def test_build_run_eval_argv_uses_run_config_not_eval_type_flag():
    argv = build_run_eval_argv(
        "/mnt/cephfs/hoangduy/venvs/nvidia-simple-evals-26.3/bin/nemo-evaluator",
        "/tmp/run.yml",
    )
    assert argv == [
        "/mnt/cephfs/hoangduy/venvs/nvidia-simple-evals-26.3/bin/nemo-evaluator",
        "run_eval",
        "--run_config",
        "/tmp/run.yml",
    ]


def test_parse_limit_empty_is_formal():
    from pipeline.aa_gpqa_v3 import parse_limit
    assert parse_limit(None) is None
    assert parse_limit("") is None
    assert parse_limit("2") == 2


def test_write_manifest_records_pin_and_honesty(tmp_path: Path):
    path = tmp_path / "manifest.json"
    man = write_manifest(
        path,
        arm="ours",
        run_id="glm53-aa",
        url="http://127.0.0.1:30000/v1/chat/completions",
        model_id="glm",
        limit_samples=None,
    )
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == man
    assert man["package"] == PACKAGE_PIN
    assert man["task"] == TASK_NAME
    assert man["n_samples"] == 5
    assert man["temperature"] == 0.6
    assert man["top_p"] == 1.0
    assert man["max_new_tokens"] == 65536
    assert man["request_timeout"] == 3600
    assert man["enable_thinking"] is True
    assert man["score_is_artificial_analysis"] is False
    assert "clone" in man["honesty"].lower()
