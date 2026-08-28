"""The keep-bf16 ignore assertion in verify_quant_checkpoint must be
model-parameterized, and must not be disableable into a vacuous pass.

Why this exists: the assertion was a hardcoded MiniMax-M3 list. On a healthy
GLM-5.2 checkpoint it fails five times (no vision_tower, multi_modal_projector,
patch_merge, indexer or block_sparse_moe exist to ignore), and because
pipeline/quantize.py::assert_quant_checkpoint_verified raises on a non-zero
return, that turns a good ~7-hour run into an apparent catastrophic failure.
Same defect class as the scale audit's hardcoded norm-gain form.
"""

from __future__ import annotations

import json

import pytest

from pipeline.verify_quant_checkpoint import (
    _EXPECTED_IGNORE_SUBSTR,
    _GLM52_EXPECTED_IGNORE_SUBSTR,
    _IGNORE_PRESETS,
    main,
    verify,
)

# The ignore list a GLM-5.2 run actually persists, from
# pipeline/configs/glm52_distributed_w4afp8_awq_smoke.yaml. The last entry is the
# partial-layer sampling restriction a smoke adds.
GLM52_PERSISTED_IGNORE = [
    "lm_head",
    "re:.*mlp[.]gate$",
    "re:.*mlp[.]shared_experts[.].*",
    "re:.*self_attn[.].*",
    "re:.*layers[.][0-2][.].*",
    "re:.*layers[.]78[.].*",
    "re:.*model[.]layers[.](?!(?:3|42)(?:[.]|$))[0-9]+(?:[.]|$).*",
]

M3_PERSISTED_IGNORE = [
    "lm_head",
    "re:.*vision_tower.*",
    "re:.*multi_modal_projector.*",
    "re:.*patch_merge.*",
    "re:.*mlp[.]gate$",
    "re:.*mlp[.]shared_experts[.].*",
    "re:.*block_sparse_moe.*",
    "re:.*self_attn[.]indexer[.].*",
    "re:.*layers[.][0-2][.].*",
]


def _write_ckpt(tmp_path, ignore):
    """Minimal checkpoint that lets verify() run to a clean return.

    config.json drives section 1 (the ignore assertion under test). A weight
    index is also required or _load_weight_keys raises before section 1's output
    can be judged; the index deliberately contains no expert projections, so
    section 4 reports "no routed-expert projections detected" and returns early.
    These tests therefore assert on the ignore-specific FAIL lines, never on the
    exit code, which is non-zero by construction.
    """
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "format": "pack-quantized",
                    "ignore": list(ignore),
                    "config_groups": {},
                }
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001.safetensors",
                    "model.norm.weight": "model-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    return ckpt


def _ignore_failures(capsys):
    out = capsys.readouterr().out
    return [
        line
        for line in out.splitlines()
        if "expected ignore pattern containing" in line
    ]


# --- the bug being fixed ----------------------------------------------------

def test_m3_preset_false_fails_on_glm52_checkpoint(tmp_path, capsys):
    """Characterizes the defect: the M3 default rejects a healthy GLM-5.2
    ignore list. If this ever stops failing, the presets have been conflated."""
    ckpt = _write_ckpt(tmp_path, GLM52_PERSISTED_IGNORE)
    verify(ckpt, check_tensors=False)
    fails = _ignore_failures(capsys)
    assert len(fails) == 5, fails
    for token in ("vision_tower", "multi_modal_projector", "patch_merge",
                  "block_sparse_moe", "indexer"):
        assert any(token in f for f in fails), (token, fails)


def test_glm52_preset_accepts_glm52_checkpoint(tmp_path, capsys):
    ckpt = _write_ckpt(tmp_path, GLM52_PERSISTED_IGNORE)
    verify(ckpt, check_tensors=False,
           expect_ignore=_GLM52_EXPECTED_IGNORE_SUBSTR)
    assert _ignore_failures(capsys) == []


def test_m3_preset_still_accepts_m3_checkpoint(tmp_path, capsys):
    """Backward compatibility: the historical behaviour is unchanged when no
    expectation is passed."""
    ckpt = _write_ckpt(tmp_path, M3_PERSISTED_IGNORE)
    verify(ckpt, check_tensors=False)
    assert _ignore_failures(capsys) == []


# --- the preset must not be a way to switch the gate off -------------------

def test_glm52_preset_rejects_a_checkpoint_missing_a_glm_requirement(tmp_path, capsys):
    """The GLM preset is a real gate, not a rubber stamp: drop shared_experts
    from the recipe and it must fail."""
    ignore = [p for p in GLM52_PERSISTED_IGNORE if "shared_experts" not in p]
    ckpt = _write_ckpt(tmp_path, ignore)
    verify(ckpt, check_tensors=False,
           expect_ignore=_GLM52_EXPECTED_IGNORE_SUBSTR)
    fails = _ignore_failures(capsys)
    assert len(fails) == 1 and "shared_experts" in fails[0], fails


@pytest.mark.parametrize("dropped", ["lm_head", "self_attn", "mlp[.]gate$"])
def test_glm52_preset_catches_each_dropped_requirement(tmp_path, capsys, dropped):
    ignore = [p for p in GLM52_PERSISTED_IGNORE
              if dropped.replace("[.]", ".").rstrip("$") not in p.replace("[.]", ".")]
    ckpt = _write_ckpt(tmp_path, ignore)
    verify(ckpt, check_tensors=False,
           expect_ignore=_GLM52_EXPECTED_IGNORE_SUBSTR)
    assert any(dropped in f for f in _ignore_failures(capsys))


def test_empty_expectation_is_rejected(tmp_path):
    """An empty list would silently make the keep-bf16 check vacuous, which is
    worse than the wrong-model failure it would be used to work around."""
    ckpt = _write_ckpt(tmp_path, GLM52_PERSISTED_IGNORE)
    with pytest.raises(ValueError, match="vacuous"):
        verify(ckpt, check_tensors=False, expect_ignore=[])


# --- CLI wiring -------------------------------------------------------------

def test_cli_preset_selects_glm52(tmp_path, capsys):
    ckpt = _write_ckpt(tmp_path, GLM52_PERSISTED_IGNORE)
    main(["--ckpt", str(ckpt), "--expect-ignore-preset", "glm52"])
    assert _ignore_failures(capsys) == []


def test_cli_defaults_to_m3(tmp_path, capsys):
    """Default must stay m3 so existing M3 callers are unaffected."""
    ckpt = _write_ckpt(tmp_path, GLM52_PERSISTED_IGNORE)
    main(["--ckpt", str(ckpt)])
    assert len(_ignore_failures(capsys)) == 5


def test_cli_explicit_overrides_preset(tmp_path, capsys):
    ckpt = _write_ckpt(tmp_path, GLM52_PERSISTED_IGNORE)
    main(["--ckpt", str(ckpt), "--expect-ignore-preset", "m3",
          "--expect-ignore", "lm_head", "--expect-ignore", "self_attn"])
    assert _ignore_failures(capsys) == []


def test_cli_rejects_unknown_preset(tmp_path):
    ckpt = _write_ckpt(tmp_path, GLM52_PERSISTED_IGNORE)
    with pytest.raises(SystemExit):
        main(["--ckpt", str(ckpt), "--expect-ignore-preset", "llama"])


# --- preset content ---------------------------------------------------------

def test_presets_registered_and_nonempty():
    assert set(_IGNORE_PRESETS) == {"m3", "glm52"}
    for name, preset in _IGNORE_PRESETS.items():
        assert preset, name


def test_glm52_preset_excludes_the_partial_layer_sampling_regex():
    """The layer-restriction regex is a smoke sampling choice, not a keep-bf16
    requirement; asserting it would make the preset reject a full run."""
    assert not any("(?!" in p for p in _GLM52_EXPECTED_IGNORE_SUBSTR)


def test_glm52_preset_omits_m3_only_modules():
    for token in ("vision_tower", "multi_modal_projector", "patch_merge",
                  "block_sparse_moe", "indexer"):
        assert not any(token in p for p in _GLM52_EXPECTED_IGNORE_SUBSTR), token
    assert any(token in p for p in _EXPECTED_IGNORE_SUBSTR
               for token in ("vision_tower",))
