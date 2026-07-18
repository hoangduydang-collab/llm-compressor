"""Fail-closed gate for the transformers offloaded-save path.

transformers 5.14.0/5.14.1 (and upstream main as of 2026-07-18) ship the
per-shard weight-format revert with a broken sharded weight-map update
(``weight_map.update({k: basename} for k in ...)`` — dict.update over a
generator of 1-element dicts raises ValueError, masked by the broad except
as the "unlucky sharding" RuntimeError). Every sharded offloaded
original-format save crashes at the end of shard 1. We hotfix the venv line;
the gate refuses to spend GPU hours if a venv rebuild dropped the hotfix.
"""

import pytest

from pipeline.quantize import (
    _offloaded_save_health,
    assert_transformers_offloaded_save_healthy,
)

_PRE_514 = """
        state_dict = revert_weight_conversion(model_to_save, state_dict)
        for shard_file, tensor_names in shards:
            safe_save_file(shard_state_dict, filename, metadata=metadata)
"""

_514_BROKEN = """
        for shard_file, tensor_names in shards:
            shard_state_dict = revert_weight_conversion(model_to_save, shard_state_dict)
            if state_dict_split.is_sharded:
                weight_map.update({k: os.path.basename(shard_file)} for k in shard_state_dict.keys())
            safe_save_file(shard_state_dict, filename, metadata=metadata)
"""

_514_HOTFIXED = """
        for shard_file, tensor_names in shards:
            shard_state_dict = revert_weight_conversion(model_to_save, shard_state_dict)
            if state_dict_split.is_sharded:
                weight_map.update({k: os.path.basename(shard_file) for k in shard_state_dict.keys()})
            safe_save_file(shard_state_dict, filename, metadata=metadata)
"""


def test_health_classification():
    assert _offloaded_save_health(_PRE_514) == "shimmed"
    assert _offloaded_save_health(_514_BROKEN) == "broken"
    assert _offloaded_save_health(_514_HOTFIXED) == "healthy"


def test_installed_transformers_passes_gate(capsys):
    """The venv this suite runs in must never be in the broken state: either
    pre-5.14 (shims own the save) or 5.14+ with the hotfix applied."""
    assert_transformers_offloaded_save_healthy()
    out = capsys.readouterr().out
    assert "offloaded-save gate OK" in out


def test_gate_raises_on_broken_source(monkeypatch):
    import inspect

    import transformers.modeling_utils as mu

    real_getsource = inspect.getsource

    def fake_getsource(obj):
        if obj is mu.PreTrainedModel.save_pretrained:
            return _514_BROKEN
        return real_getsource(obj)

    monkeypatch.setattr(inspect, "getsource", fake_getsource)
    with pytest.raises(RuntimeError, match="weight_map update is the known-broken"):
        assert_transformers_offloaded_save_healthy()
