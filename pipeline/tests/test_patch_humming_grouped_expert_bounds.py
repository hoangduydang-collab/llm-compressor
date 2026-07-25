"""CPU-only tests for the Humming grouped_contiguous exact-total patch."""

from __future__ import annotations

import pytest

from pipeline.m3_humming_w4a8 import DECLARED_PATCH_SHA256
from pipeline.slurm.patch_humming_grouped_expert_bounds import (
    ANCHOR,
    PATCHED,
    RELATIVE_TARGET,
    apply_patch,
    classify,
    main,
)

PRISTINE_BODY = f"""      if constexpr (
          ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS) {{
{ANCHOR}
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumExperts - 1, kNumThreads); i++) {{
          uint32_t index = kNumThreads * i + threadIdx.x;
        }}
      }}
"""


def write_site(tmp_path, body):
    target = tmp_path / RELATIVE_TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return tmp_path


def test_classify_distinguishes_the_three_states():
    assert classify(PRISTINE_BODY) == "unpatched"
    assert classify(PRISTINE_BODY.replace(ANCHOR, PATCHED)) == "patched"
    assert classify("__global__ void unrelated() {}\n") == "unknown"


def test_apply_uses_expert_offsets_and_is_idempotent(tmp_path):
    site = write_site(tmp_path, PRISTINE_BODY)

    status, first_digest = apply_patch(site, apply=True)
    assert status == "patched"

    body = (site / RELATIVE_TARGET).read_text(encoding="utf-8")
    # The whole point: the last expert's count no longer depends on shape_m.
    assert "shape_m - smem.expert_offset[kNumExperts - 1]" not in body
    assert (
        "smem.expert_offset[kNumExperts] - smem.expert_offset[kNumExperts - 1]" in body
    )

    status, second_digest = apply_patch(site, apply=True)
    assert status == "already patched"
    assert second_digest == first_digest


def test_check_reports_unpatched_without_writing(tmp_path):
    site = write_site(tmp_path, PRISTINE_BODY)

    status, _ = apply_patch(site, apply=False)
    assert status == "NOT patched"
    assert (site / RELATIVE_TARGET).read_text(encoding="utf-8") == PRISTINE_BODY


def test_unknown_content_refuses_to_guess(tmp_path):
    site = write_site(tmp_path, "some other scheduler entirely\n")

    with pytest.raises(SystemExit):
        apply_patch(site, apply=True)


def test_missing_target_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        apply_patch(tmp_path, apply=False)


def test_main_check_exit_codes(tmp_path, capsys):
    site = write_site(tmp_path, PRISTINE_BODY)

    assert main(["--site", str(site), "--check"]) == 1
    assert main(["--site", str(site)]) == 0
    assert main(["--site", str(site), "--check"]) == 0
    assert "already patched" in capsys.readouterr().out


def test_patched_file_is_declared_in_the_integrity_gate():
    """A patch the integrity gate does not know about must fail closed.

    If this target were missing from DECLARED_PATCH_SHA256, every Humming
    preflight would report the side-install as tampered and abort.
    """
    assert RELATIVE_TARGET in DECLARED_PATCH_SHA256
    assert len(DECLARED_PATCH_SHA256[RELATIVE_TARGET]) == 64
