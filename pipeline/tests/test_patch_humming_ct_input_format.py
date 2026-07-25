"""CPU-only tests for the Humming compressed-tensors input-format patch."""

from __future__ import annotations

import pytest

from pipeline.slurm.patch_humming_ct_input_format import (
    ANCHOR,
    PATCHED,
    RELATIVE_TARGET,
    apply_patch,
    classify,
    main,
)

PRISTINE_BODY = f'''@dataclasses.dataclass(kw_only=True)
class CompressedTensorsInputSchema(BaseInputSchema):
    def __post_init__(self):
{ANCHOR} "input_global_scale" if "nvfp4" in self.format else "input_scale"
'''


def write_site(tmp_path, body):
    target = tmp_path / RELATIVE_TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return tmp_path


def test_classify_distinguishes_the_three_states():
    assert classify(PRISTINE_BODY) == "unpatched"
    assert classify(PRISTINE_BODY.replace(ANCHOR, PATCHED)) == "patched"
    assert classify("def unrelated(): pass\n") == "unknown"


def test_apply_adds_pack_quantized_and_is_idempotent(tmp_path):
    site = write_site(tmp_path, PRISTINE_BODY)

    status, first_digest = apply_patch(site, apply=True)
    assert status == "patched"

    body = (site / RELATIVE_TARGET).read_text(encoding="utf-8")
    assert '"pack-quantized",' in body
    # The other four formats must survive untouched.
    for fmt in (
        "int-quantized",
        "float-quantized",
        "naive-quantized",
        "nvfp4-pack-quantized",
        "mxfp4-pack-quantized",
    ):
        assert f'"{fmt}",' in body
    # The scale-key line must be preserved verbatim.
    assert 'if "nvfp4" in self.format else "input_scale"' in body

    status, second_digest = apply_patch(site, apply=True)
    assert status == "already patched"
    assert second_digest == first_digest


def test_check_mode_never_writes(tmp_path):
    site = write_site(tmp_path, PRISTINE_BODY)

    status, _ = apply_patch(site, apply=False)

    assert status == "NOT patched"
    assert (site / RELATIVE_TARGET).read_text(encoding="utf-8") == PRISTINE_BODY


def test_check_exit_codes(tmp_path, capsys):
    site = write_site(tmp_path, PRISTINE_BODY)

    assert main(["--site", str(site), "--check"]) == 1
    assert main(["--site", str(site)]) == 0
    assert main(["--site", str(site), "--check"]) == 0
    assert "already patched" in capsys.readouterr().out


def test_unknown_content_refuses_to_guess(tmp_path):
    site = write_site(tmp_path, "class Something: pass\n")

    with pytest.raises(SystemExit, match="refusing to guess"):
        apply_patch(site, apply=True)


def test_missing_target_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="target not found"):
        apply_patch(tmp_path, apply=True)
