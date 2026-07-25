"""CPU-only tests for the Humming TMA-store commit-group patch."""

from __future__ import annotations

import pytest

from pipeline.m3_humming_w4a8 import DECLARED_PATCH_SHA256
from pipeline.slurm.patch_humming_tma_store_commit import (
    ANCHOR,
    PATCHED,
    RELATIVE_TARGET,
    apply_patch,
    classify,
    main,
)

PRISTINE_BODY = (
    "  CUDA_INLINE\n"
    "  void write_tma(uint32_t slice_id, uint32_t slice_count) {\n"
    "    if (block_idx < count) {\n"
    + ANCHOR
    + "\n    }\n  }\n"
)


def write_site(tmp_path, body):
    target = tmp_path / RELATIVE_TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return tmp_path


def test_classify_distinguishes_the_three_states():
    assert classify(PRISTINE_BODY) == "unpatched"
    assert classify(PRISTINE_BODY.replace(ANCHOR, PATCHED)) == "patched"
    assert classify("__global__ void unrelated() {}\n") == "unknown"


def test_apply_commits_every_store_issue_and_is_idempotent(tmp_path):
    site = write_site(tmp_path, PRISTINE_BODY)

    status, first_digest = apply_patch(site, apply=True)
    assert status == "patched"

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


def test_check_reports_unpatched_without_writing(tmp_path):
    site = write_site(tmp_path, PRISTINE_BODY)

    status, _ = apply_patch(site, apply=False)
    assert status == "NOT patched"
    assert (site / RELATIVE_TARGET).read_text(encoding="utf-8") == PRISTINE_BODY


def test_unknown_content_refuses_to_guess(tmp_path):
    site = write_site(tmp_path, "some other epilogue entirely\n")

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
    """A patch the integrity gate does not know about must fail closed."""
    assert RELATIVE_TARGET in DECLARED_PATCH_SHA256
    assert len(DECLARED_PATCH_SHA256[RELATIVE_TARGET]) == 64
