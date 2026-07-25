"""CPU-only tests for the Humming TMA-store commit-group patch."""

from __future__ import annotations

import pytest

from pipeline.m3_humming_w4a8 import DECLARED_PATCH_SHA256
from pipeline.slurm.patch_humming_tma_store_commit import (
    RELATIVE_TARGET,
    VARIANTS,
    apply_patch,
    classify,
    main,
)

VARIANT_IDS = [label for label, _, _ in VARIANTS]


def pristine_body(anchor: str) -> str:
    return (
        "  CUDA_INLINE\n"
        "  void write_tma(uint32_t slice_id, uint32_t slice_count) {\n"
        "    if (block_idx < count) {\n"
        + anchor
        + "\n    }\n  }\n"
    )


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
def test_apply_commits_every_store_issue_and_is_idempotent(
    tmp_path, label, anchor, patched
):
    site = write_site(tmp_path, pristine_body(anchor))

    status, first_digest = apply_patch(site, apply=True)
    assert status == f"patched ({label} content)"

    body = (site / RELATIVE_TARGET).read_text(encoding="utf-8")
    # The whole point: each of the three TMA store/reduce issuances is now
    # committed into a bulk async-group, so wait_group actually waits.
    assert body.count("tma_commit_store_group();") == 3
    for op in ("tma_store_2d(", "tma_reduce_add_2d("):
        pos = 0
        while (pos := body.find(op, pos)) != -1:
            commit_pos = body.find("tma_commit_store_group();", pos)
            next_op = min(
                p
                for p in (
                    body.find("tma_store_2d(", pos + 1),
                    body.find("tma_reduce_add_2d(", pos + 1),
                    len(body),
                )
                if p != -1
            )
            assert commit_pos != -1 and commit_pos < next_op, (
                f"{op} at {pos} has no commit before the next store"
            )
            pos += 1

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
    site = write_site(tmp_path, "some other epilogue entirely\n")

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
    """A patch the integrity gate does not know about must fail closed."""
    assert RELATIVE_TARGET in DECLARED_PATCH_SHA256
    declared = DECLARED_PATCH_SHA256[RELATIVE_TARGET]
    # One post-patch hash per supported upstream release (0.1.10 and 0.1.11,
    # whose pristine gmem_writer.cuh differ).
    assert len(declared) == len(VARIANTS)
    assert all(len(digest) == 64 for digest in declared)
