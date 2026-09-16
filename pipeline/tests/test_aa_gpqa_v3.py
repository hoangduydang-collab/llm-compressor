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
    # Z.ai's recommended config: the lab-override branch of AA's sampling rule.
    assert params["temperature"] == TEMPERATURE == 1.0
    assert params["top_p"] == TOP_P == 0.95
    assert params["max_new_tokens"] == MAX_NEW_TOKENS == 131072
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


def test_write_run_config_diag_overrides(tmp_path: Path):
    path = tmp_path / "diag.yml"
    cfg = write_run_config(
        path,
        url="http://svc/v1/chat/completions",
        model_id="glm-5.3-w4afp8",
        output_dir="/tmp/aa-caphit70",
        max_new_tokens=262144,
        request_timeout=14400,
        parallelism=2,
    )
    params = cfg["config"]["params"]
    assert params["max_new_tokens"] == 262144
    assert params["request_timeout"] == 14400
    assert params["parallelism"] == 2
    assert params["temperature"] == 1.0
    assert params["extra"]["n_samples"] == 5

    path = tmp_path / "canary.yml"
    cfg = write_run_config(
        path,
        url="http://127.0.0.1:30000/v1/chat/completions",
        model_id="glm",
        output_dir="/tmp/aa-gpqa-v3",
        limit_samples=2,
    )
    assert cfg["config"]["params"]["limit_samples"] == 2


def test_write_run_config_sampling_is_overridable(tmp_path: Path):
    """Reproducing the Sep-11 contract must be possible without editing code."""
    cfg = write_run_config(
        tmp_path / "sep11.yml",
        url="http://127.0.0.1:30000/v1/chat/completions",
        model_id="glm",
        output_dir="/tmp/aa-gpqa-v3",
        temperature=0.6,
        top_p=1.0,
    )
    params = cfg["config"]["params"]
    assert params["temperature"] == 0.6
    assert params["top_p"] == 1.0


def test_canary_reduces_repeats_and_is_marked_non_formal(tmp_path: Path):
    cfg = write_run_config(
        tmp_path / "canary.yml",
        url="http://127.0.0.1:30000/v1/chat/completions",
        model_id="glm",
        output_dir="/tmp/aa-canary",
        n_samples=1,
    )
    # All 198 items, one repeat — not a first-N `limit_samples` subset.
    assert cfg["config"]["params"]["extra"]["n_samples"] == 1
    assert cfg["config"]["params"]["limit_samples"] is None

    man = write_manifest(tmp_path / "canary.json", arm="ours", n_samples=1)
    assert man["n_samples"] == 1
    assert man["is_formal_aa_protocol"] is False


def test_manifest_formal_only_when_full_items_and_repeats(tmp_path: Path):
    assert write_manifest(tmp_path / "a.json")["is_formal_aa_protocol"] is True
    # A limited item subset is not formal even at 5 repeats.
    man = write_manifest(tmp_path / "b.json", limit_samples=40)
    assert man["is_formal_aa_protocol"] is False


def test_manifest_labels_zai_lab_override_branch(tmp_path: Path):
    man = write_manifest(tmp_path / "m.json", arm="ours")
    assert man["temperature"] == 1.0
    assert man["top_p"] == 0.95
    assert "Z.ai" in man["sampling_provenance"]
    # A Z.ai-config run is NOT poolable with the Sep-11 formal scores.
    assert man["comparable_to_sep11_formal_198_c8"] is False


def test_manifest_labels_aa_default_branch(tmp_path: Path):
    man = write_manifest(tmp_path / "m.json", arm="ours", temperature=0.6, top_p=1.0)
    assert man["sampling_provenance"] == (
        "AA published default (no lab override applied)"
    )
    assert man["comparable_to_sep11_formal_198_c8"] is True


def test_manifest_flags_a_config_that_is_neither(tmp_path: Path):
    man = write_manifest(tmp_path / "m.json", temperature=0.7, top_p=0.8)
    assert man["sampling_provenance"].startswith("custom:")
    assert man["comparable_to_sep11_formal_198_c8"] is False


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
    assert man["temperature"] == 1.0
    assert man["top_p"] == 0.95
    assert man["max_new_tokens"] == 131072
    assert man["request_timeout"] == 3600
    assert man["enable_thinking"] is True
    assert man["score_is_artificial_analysis"] is False
    assert "clone" in man["honesty"].lower()
