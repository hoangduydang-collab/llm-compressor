"""Tests for _resolve_weight_index, the VMA gate's index lookup.

The gate that this feeds exists to stop a 3h NCCL broadcast timeout that produces
no checkpoint. It went inert on the first GLM-5.2 run because `model.id` is a hub
repo id, not a directory, so `Path(model_id)/"model.safetensors.index.json"` was a
relative path that could never exist and the gate skipped itself with a message
that read like a property of the model rather than a bug in the lookup.
"""

import json
from pathlib import Path

import pytest

from pipeline.quantize import _resolve_weight_index

INDEX = "model.safetensors.index.json"


def _write_index(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / INDEX
    p.write_text(json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}))
    return p


def test_local_directory_is_found(tmp_path, monkeypatch):
    """A path-style model.id must still resolve, and without touching the cache."""
    monkeypatch.chdir(tmp_path)
    _write_index(tmp_path / "models" / "glm52")
    got = _resolve_weight_index("models/glm52")
    assert got is not None
    assert got.is_file()
    assert got.name == INDEX


def test_repo_id_resolves_through_the_hf_cache(tmp_path, monkeypatch):
    """The regression: a hub repo id must resolve via the cache, not as a path."""
    monkeypatch.chdir(tmp_path)
    cached = _write_index(tmp_path / "hfcache" / "snapshots" / "abc123")

    # _resolve_weight_index imports try_to_load_from_cache INSIDE the function,
    # so it re-resolves the module attribute on every call and this patch takes
    # effect without needing to reach into pipeline.quantize's namespace.
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo_id, filename: str(cached),
    )

    got = _resolve_weight_index("zai-org/GLM-5.2")
    assert got == cached


def test_returns_none_when_nothing_is_cached(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda repo_id, filename: None
    )
    assert _resolve_weight_index("zai-org/GLM-5.2") is None


def test_cached_no_exist_sentinel_is_not_mistaken_for_a_path(tmp_path, monkeypatch):
    """try_to_load_from_cache returns a sentinel OBJECT when upstream is known not
    to have the file. Truth-testing the return value would make a single-shard
    model look like a cache hit.

    Deliberately stands in a plain object rather than importing
    huggingface_hub's private _CACHED_NO_EXIST: the contract under test is "only
    a str is a path", which holds for any non-str sentinel and does not break
    when the library moves or renames that symbol.
    """
    monkeypatch.chdir(tmp_path)
    import huggingface_hub

    sentinel = object()
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo_id, filename: sentinel,
    )
    assert _resolve_weight_index("zai-org/GLM-5.2") is None


def test_stale_cache_entry_pointing_at_a_deleted_file(tmp_path, monkeypatch):
    """A cache hit whose blob has since been removed must not be returned."""
    monkeypatch.chdir(tmp_path)
    import huggingface_hub

    missing = tmp_path / "hfcache" / "snapshots" / "gone" / INDEX
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo_id, filename: str(missing),
    )
    assert _resolve_weight_index("zai-org/GLM-5.2") is None


def test_cache_lookup_errors_do_not_propagate(tmp_path, monkeypatch):
    """A gate must not crash the run before quantization starts."""
    monkeypatch.chdir(tmp_path)
    import huggingface_hub

    def boom(repo_id, filename):
        raise ValueError("Repo id must be in the form 'repo_name' or 'namespace/repo_name'")

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", boom)
    assert _resolve_weight_index("not a valid id!!") is None


def test_local_directory_wins_over_cache(tmp_path, monkeypatch):
    """A relative model_id yields a relative path, so compare resolved paths."""
    monkeypatch.chdir(tmp_path)
    local = _write_index(tmp_path / "local" / "glm52")
    other = _write_index(tmp_path / "cache" / "glm52")
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda repo_id, filename: str(other)
    )
    got = _resolve_weight_index("local/glm52")
    assert got is not None
    assert got.resolve() == local.resolve()
    assert got.resolve() != other.resolve()


@pytest.mark.parametrize("model_id", ["", "   "])
def test_degenerate_ids_are_handled(model_id, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda repo_id, filename: None
    )
    assert _resolve_weight_index(model_id) is None
