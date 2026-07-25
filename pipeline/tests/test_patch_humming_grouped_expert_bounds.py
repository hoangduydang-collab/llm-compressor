"""CPU-only tests for the Humming grouped_contiguous exact-total patch."""

from __future__ import annotations

import pytest

from pipeline.m3_humming_w4a8 import DECLARED_PATCH_SHA256
from pipeline.slurm.patch_humming_grouped_expert_bounds import (
    RELATIVE_TARGET,
    VARIANTS,
    apply_patch,
    classify,
    main,
)

VARIANT_IDS = [label for label, _, _ in VARIANTS]


def pristine_body(anchor: str) -> str:
    return f"""      if constexpr (
          ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS) {{
{anchor}
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


@pytest.mark.parametrize("label,anchor,patched", VARIANTS, ids=VARIANT_IDS)
def test_classify_distinguishes_the_three_states(label, anchor, patched):
    body = pristine_body(anchor)
    assert classify(body) == "unpatched"
    assert classify(body.replace(anchor, patched)) == "patched"
    assert classify("__global__ void unrelated() {}\n") == "unknown"


@pytest.mark.parametrize("label,anchor,patched", VARIANTS, ids=VARIANT_IDS)
def test_apply_uses_expert_offsets_and_is_idempotent(tmp_path, label, anchor, patched):
    site = write_site(tmp_path, pristine_body(anchor))

    status, first_digest = apply_patch(site, apply=True)
    assert status == f"patched ({label} content)"

    body = (site / RELATIVE_TARGET).read_text(encoding="utf-8")
    # The whole point: the last expert's count no longer depends on shape_m.
    assert "shape_m - " not in body
    assert "expert_offset[kNumExperts] - " in body

    status, second_digest = apply_patch(site, apply=True)
    assert status == "already patched"
    assert second_digest == first_digest


@pytest.mark.parametrize("label,anchor,patched", VARIANTS, ids=VARIANT_IDS)
def test_check_reports_unpatched_without_writing(tmp_path, label, anchor, patched):
    body = pristine_body(anchor)
    site = write_site(tmp_path, body)

    status, _ = apply_patch(site, apply=False)
    assert status == "NOT patched"
    assert (site / RELATIVE_TARGET).read_text(encoding="utf-8") == body


def test_unknown_content_refuses_to_guess(tmp_path):
    site = write_site(tmp_path, "some other scheduler entirely\n")

    with pytest.raises(SystemExit):
        apply_patch(site, apply=True)


def test_missing_target_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        apply_patch(tmp_path, apply=False)


def test_main_check_exit_codes(tmp_path, capsys):
    site = write_site(tmp_path, pristine_body(VARIANTS[0][1]))

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
    declared = DECLARED_PATCH_SHA256[RELATIVE_TARGET]
    # One post-patch hash per supported upstream release (0.1.10 and 0.1.11,
    # whose pristine scheduler.cuh differ).
    assert len(declared) == len(VARIANTS)
    assert all(len(digest) == 64 for digest in declared)
