"""The save preflight: prove the save works before paying for calibration.

The checkpoint save is one all-or-nothing operation at the end of a multi-hour
run. GLM-5.2's invalid generation_config was rejected by a validator that runs
at the END of save_pretrained, costing two 4-GPU runs ~11 GPU-hours between
them for zero artifacts.

These tests pin the failure modes the preflight is supposed to convert from
"six hours then nothing" into "fails immediately", and -- just as important --
that it cleans up after itself, since it writes into the real run directory on
a shared PVC.
"""

import pytest

from pipeline.quantize import assert_checkpoint_save_preflight


class FakeConfig:
    def __init__(self, name, fail=None):
        self.name = name
        self.fail = fail
        self.saved_to = []

    def save_pretrained(self, directory):
        if self.fail is not None:
            raise self.fail
        self.saved_to.append(directory)
        from pathlib import Path

        Path(directory, f"{self.name}.json").write_text("{}", encoding="utf-8")


class FakeModel:
    def __init__(self, config=None, generation_config=None):
        if config is not None:
            self.config = config
        if generation_config is not None:
            self.generation_config = generation_config


def test_passes_and_reports_what_it_serialized(tmp_path, capsys):
    model = FakeModel(FakeConfig("config"), FakeConfig("generation_config"))
    assert_checkpoint_save_preflight(model, tmp_path)

    out = capsys.readouterr().out
    assert "save preflight OK" in out
    assert "config.json" in out and "generation_config.json" in out


def test_leaves_no_probe_directory_behind(tmp_path):
    """It writes into the real run dir on a shared PVC; residue is not ok."""
    model = FakeModel(FakeConfig("config"), FakeConfig("generation_config"))
    assert_checkpoint_save_preflight(model, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cleans_up_even_when_it_fails(tmp_path):
    model = FakeModel(FakeConfig("config", fail=ValueError("boom")))
    with pytest.raises(RuntimeError):
        assert_checkpoint_save_preflight(model, tmp_path)
    assert list(tmp_path.iterdir()) == [], "probe dir survived a failure"


# --------------------------------------------------------------------------
# the failure this exists for
# --------------------------------------------------------------------------


def test_generation_config_validation_failure_is_caught(tmp_path):
    """GLM-5.2's exact failure: the validator that runs at the end of a save."""
    bad = FakeConfig(
        "generation_config",
        fail=ValueError(
            "GenerationConfig is invalid: \n- `top_p`: `do_sample` is not set "
            "to `True`. However, `top_p` is set to `0.95`"
        ),
    )
    model = FakeModel(FakeConfig("config"), bad)

    with pytest.raises(RuntimeError) as excinfo:
        assert_checkpoint_save_preflight(model, tmp_path)

    message = str(excinfo.value)
    assert "model.generation_config.save_pretrained failed" in message
    assert "AFTER calibration" in message, "must say why this gate exists"
    assert "top_p" in message, "must surface the underlying cause"


def test_model_config_failure_is_caught_too(tmp_path):
    model = FakeModel(FakeConfig("config", fail=TypeError("unserializable")))
    with pytest.raises(RuntimeError, match=r"model\.config\.save_pretrained failed"):
        assert_checkpoint_save_preflight(model, tmp_path)


def test_missing_configs_are_tolerated(tmp_path):
    """A model without either config must not be treated as a failure."""
    assert_checkpoint_save_preflight(FakeModel(), tmp_path)


def test_none_configs_are_skipped(tmp_path):
    model = FakeModel()
    model.config = None
    model.generation_config = None
    assert_checkpoint_save_preflight(model, tmp_path)


# --------------------------------------------------------------------------
# filesystem problems
# --------------------------------------------------------------------------


def test_creates_a_missing_run_dir(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    model = FakeModel(FakeConfig("config"))
    assert_checkpoint_save_preflight(model, target)
    assert target.is_dir()


def test_unwritable_run_dir_is_reported(tmp_path, monkeypatch):
    """A read-only mount or full volume must fail here, not after 6 hours."""
    model = FakeModel(FakeConfig("config"))

    real_write_bytes = type(tmp_path).write_bytes

    def refuse(self, data):
        if self.name == "probe.bin":
            raise OSError(28, "No space left on device")
        return real_write_bytes(self, data)

    monkeypatch.setattr(type(tmp_path), "write_bytes", refuse)

    with pytest.raises(RuntimeError, match="cannot write to"):
        assert_checkpoint_save_preflight(model, tmp_path)


def test_undeletable_run_dir_parent_is_reported(tmp_path, monkeypatch):
    import pathlib

    def refuse_mkdir(self, parents=False, exist_ok=False):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(pathlib.Path, "mkdir", refuse_mkdir)
    with pytest.raises(RuntimeError, match="cannot create run dir"):
        assert_checkpoint_save_preflight(FakeModel(), tmp_path / "sub")
