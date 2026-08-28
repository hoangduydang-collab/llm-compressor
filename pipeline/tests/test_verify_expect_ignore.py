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


# Unquantized modules a GLM-5.2 checkpoint really has. The coverage check reads
# these, so a fixture without them makes every token vacuous and every assertion
# below pass for the wrong reason -- which is exactly what happened when the check
# moved from pattern text to coverage.
GLM52_MODULES = (
    "lm_head.weight",
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.0.self_attn.indexer.wk.weight",
    "model.layers.1.mlp.gate_proj.weight",
    "model.layers.3.mlp.gate.weight",
    "model.layers.3.mlp.shared_experts.gate_proj.weight",
    "model.layers.3.self_attn.o_proj.weight",
)


def _write_ckpt(tmp_path, ignore, modules=None):
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
    weight_map = {
        name: "model-00001.safetensors" for name in (modules or GLM52_MODULES)
    }
    weight_map["model.embed_tokens.weight"] = "model-00001.safetensors"
    weight_map["model.norm.weight"] = "model-00001.safetensors"
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return ckpt


def _ignore_failures(capsys):
    out = capsys.readouterr().out
    return [
        line
        for line in out.splitlines()
        if "NOT covered by any ignore entry" in line
    ]


# --- the bug being fixed ----------------------------------------------------

def test_m3_preset_no_longer_false_fails_on_a_glm_checkpoint(tmp_path, capsys):
    """The defect this file was written for is now fixed BY CONSTRUCTION.

    It used to fail five times, because the check asked whether the ignore list's
    TEXT contained 'vision_tower', 'multi_modal_projector', 'patch_merge',
    'block_sparse_moe' and 'indexer'. A GLM checkpoint has none of those modules,
    so there was nothing for an ignore entry to protect and the demand was
    meaningless.

    Since 2026-08-28 the check asks whether every UNQUANTIZED MODULE of each
    component is covered by some ignore entry, so a component with no such modules
    is vacuous. Picking the right preset is still good hygiene -- it documents what
    a recipe intends -- but it is no longer load-bearing for correctness, which is
    the better place for this to have landed."""
    ckpt = _write_ckpt(tmp_path, GLM52_PERSISTED_IGNORE)
    verify(ckpt, check_tensors=False)
    assert _ignore_failures(capsys) == []


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
    """Default must stay m3 so existing M3 callers are unaffected.

    Asserted on the PRESET rather than on a failure count. It used to assert 5
    failures, which only worked while the check compared pattern text; under
    coverage the M3 tokens a GLM checkpoint has no modules for are vacuous, so the
    default produces no failures here. The default itself still matters, so this
    checks the thing that has to hold.
    """
    ckpt = _write_ckpt(tmp_path, GLM52_PERSISTED_IGNORE)
    main(["--ckpt", str(ckpt)])
    assert _ignore_failures(capsys) == []
    # and the M3 preset is genuinely enforced where the modules DO exist
    m3_dir = tmp_path / "m3"
    m3_dir.mkdir()
    m3_ckpt = _write_ckpt(
        m3_dir,
        ["lm_head"],
        modules=("lm_head.weight", "vision_tower.layers.0.attn.qkv.weight"),
    )
    main(["--ckpt", str(m3_ckpt)])
    fails = _ignore_failures(capsys)
    assert any("vision_tower" in f for f in fails), fails


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
