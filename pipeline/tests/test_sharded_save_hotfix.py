"""The transformers sharded-save hotfix: transform, semantics, and drift.

The whole repair is the position of one brace, and its verdict is computed by a
classifier that now exists in two places -- ``pipeline.quantize`` (runtime gate)
and ``envs/hotfix-transformers-sharded-save.py`` (which must run before the repo
is installed, so it cannot import the first). A brace in the wrong place, or
those two copies disagreeing, silently reintroduces a bug that costs a full
calibration before it surfaces. Both are pinned here.
"""

import importlib.util
from pathlib import Path

import pytest

from pipeline.quantize import _offloaded_save_health

HOTFIX_PATH = (
    Path(__file__).resolve().parents[2] / "envs" / "hotfix-transformers-sharded-save.py"
)

REVERT = "shard_state_dict = revert_weight_conversion(model_to_save, shard_state_dict)"


def load_hotfix():
    """Import the hotfix script by path: envs/ is not an importable package."""
    spec = importlib.util.spec_from_file_location("_hotfix", HOTFIX_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hotfix():
    return load_hotfix()


def broken_source():
    """The 5.14.1 shape, reproduced closely enough for both classifiers."""
    return (
        "def save_pretrained(self):\n"
        "    if is_offloaded and save_original_format:\n"
        "        try:\n"
        f"            {REVERT}\n"
        "            if state_dict_split.is_sharded:\n"
        "                weight_map.update({k: os.path.basename(shard_file)}"
        " for k in shard_state_dict.keys())  # ty: ignore[unresolved-attribute]\n"
        "        except Exception:\n"
        "            raise RuntimeError('unlucky sharding')\n"
    )


def test_hotfix_file_exists():
    assert HOTFIX_PATH.is_file(), f"missing {HOTFIX_PATH}"


# --------------------------------------------------------------------------
# The transform
# --------------------------------------------------------------------------


def test_replacement_flips_the_verdict_to_healthy(hotfix):
    src = broken_source()
    assert hotfix.health(src) == "broken"
    patched = src.replace(hotfix.BROKEN, hotfix.FIXED, 1)
    assert hotfix.health(patched) == "healthy"


def test_replacement_is_recognised_by_the_runtime_gate_too(hotfix):
    """The gate that actually aborts runs must agree with the patcher."""
    src = broken_source()
    assert _offloaded_save_health(src) == "broken"
    patched = src.replace(hotfix.BROKEN, hotfix.FIXED, 1)
    assert _offloaded_save_health(patched) == "healthy"


def test_marker_occurs_exactly_once_in_the_broken_shape(hotfix):
    """The patcher refuses to act on any other count, so this must hold."""
    assert broken_source().count(hotfix.BROKEN) == 1


def test_patched_line_is_valid_python_and_keeps_the_trailing_comment(hotfix):
    patched = broken_source().replace(hotfix.BROKEN, hotfix.FIXED, 1)
    line = next(li for li in patched.splitlines() if "weight_map.update" in li)
    assert "# ty: ignore[unresolved-attribute]" in line
    compile(line.strip(), "<patched>", "eval")


def test_replacement_preserves_the_dict_value_expression(hotfix):
    """os.path.basename(shard_file) must survive; only the brace moves."""
    patched = broken_source().replace(hotfix.BROKEN, hotfix.FIXED, 1)
    assert "{k: os.path.basename(shard_file) for k in" in patched


# --------------------------------------------------------------------------
# Semantics: why the brace matters at all
# --------------------------------------------------------------------------


def test_broken_form_raises_and_fixed_form_maps_every_key():
    keys = {"a.weight": 1, "b.weight": 2}

    with pytest.raises(ValueError, match="length 1"):
        {}.update({k: "shard-1"} for k in keys)

    weight_map = {}
    weight_map.update({k: "shard-1" for k in keys})
    assert weight_map == {"a.weight": "shard-1", "b.weight": "shard-1"}


def test_broken_form_raises_even_for_a_single_key():
    """Not a sharding-luck problem: one key is enough to blow up."""
    with pytest.raises(ValueError):
        {}.update({k: "shard-1"} for k in {"only.weight": 1})


# --------------------------------------------------------------------------
# Drift between the two classifier copies
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, expected",
    [
        ("def save_pretrained(self):\n    pass\n", "shimmed"),
        (broken_source(), "broken"),
        (broken_source().replace(
            "{k: os.path.basename(shard_file)} for k in shard_state_dict.keys()",
            "{k: os.path.basename(shard_file) for k in shard_state_dict.keys()}",
        ), "healthy"),
    ],
    ids=["pre-5.14", "5.14-unpatched", "5.14-patched"],
)
def test_both_classifiers_agree(hotfix, source, expected):
    assert hotfix.health(source) == expected
    assert _offloaded_save_health(source) == expected


def test_installed_transformers_is_patched(hotfix):
    """The environment this test runs in must itself be repaired.

    Skipped rather than failed on a pre-5.14 install, where the repo's save
    shims own the path and there is nothing to patch.
    """
    import inspect

    from transformers import modeling_utils as mu

    verdict = hotfix.health(inspect.getsource(mu.PreTrainedModel.save_pretrained))
    if verdict == "shimmed":
        pytest.skip("pre-5.14 transformers: save shims cover this path")
    assert verdict == "healthy", (
        "installed transformers has the sharded-save bug; run "
        "envs/hotfix-transformers-sharded-save.py"
    )
