"""CPU-only tests for the Humming TMA-store proxy-fence patch."""

from __future__ import annotations

import pytest

from pipeline.m3_humming_w4a8 import DECLARED_PATCH_SHA256
from pipeline.slurm.patch_humming_tma_store_fence import (
    PAIRS,
    RELATIVE_TARGET,
    apply_patch,
    classify,
    main,
)

PRISTINE_BODY = (
    "#pragma once\n\n"
    + PAIRS[0][1]
    + "\n               \" [%0, {%2, %3}], [%1];\"\n"
    + "               :\n"
    + "               : \"l\"(gmem_int_desc), \"r\"(smem_int_ptr)\n"
    + "               : \"memory\");\n};\n\n"
    + PAIRS[1][1]
    + "\n               \" [%0, {%2, %3}], [%1];\"\n"
    + "               :\n"
    + "               : \"l\"(gmem_int_desc), \"r\"(smem_int_ptr)\n"
    + "               : \"memory\");\n};\n"
)


def write_site(tmp_path, body):
    target = tmp_path / RELATIVE_TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return tmp_path


def test_classify_distinguishes_the_three_states():
    assert classify(PRISTINE_BODY) == "unpatched"
    patched_body = PRISTINE_BODY
    for _, anchor, patched in PAIRS:
        patched_body = patched_body.replace(anchor, patched)
    assert classify(patched_body) == "patched"
    assert classify("__global__ void unrelated() {}\n") == "unknown"


def test_apply_fences_both_store_wrappers_and_is_idempotent(tmp_path):
    site = write_site(tmp_path, PRISTINE_BODY)

    status, first_digest = apply_patch(site, apply=True)
    assert status == "patched"

    body = (site / RELATIVE_TARGET).read_text(encoding="utf-8")
    # The whole point: both smem->gmem bulk-tensor copies are preceded by the
    # cross-proxy fence that makes generic-proxy smem writes async-visible.
    assert body.count('fence.proxy.async.shared::cta') == 2
    for op in (
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group",
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group",
    ):
        fence_pos = body.rfind("fence.proxy.async.shared::cta", 0, body.find(op))
        assert fence_pos != -1, f"no fence before {op}"

    status, second_digest = apply_patch(site, apply=True)
    assert status == "already patched"
    assert second_digest == first_digest


def test_check_reports_unpatched_without_writing(tmp_path):
    site = write_site(tmp_path, PRISTINE_BODY)

    status, _ = apply_patch(site, apply=False)
    assert status == "NOT patched"
    assert (site / RELATIVE_TARGET).read_text(encoding="utf-8") == PRISTINE_BODY


def test_unknown_content_refuses_to_guess(tmp_path):
    site = write_site(tmp_path, "some other tma header entirely\n")

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
    declared = DECLARED_PATCH_SHA256[RELATIVE_TARGET]
    # tma.cuh is byte-identical upstream in 0.1.10 and 0.1.11, so a single
    # post-patch hash covers both releases.
    assert len(declared) == 1
    assert all(len(digest) == 64 for digest in declared)
