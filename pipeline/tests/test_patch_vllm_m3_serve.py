"""CPU-only tests for persistent MiniMax-M3 vLLM diagnostics."""

from pipeline.slurm.patch_vllm_m3_serve import (
    _LOAD_AUDIT_BLOCK,
    _patch_append_load_audit,
)


def test_load_audit_patch_is_env_gated_and_covers_both_expert_layouts():
    source = """
class MiniMaxM3MoE:
    pass


class MiniMaxM3Model:
    pass
"""

    patched, changed, found = _patch_append_load_audit(source)

    assert found
    assert changed
    assert 'M3_LOAD_AUDIT' in patched
    assert 'gate_proj' in patched
    assert 'up_proj' in patched
    assert 'down_proj' in patched
    assert '"w1"' in patched
    assert '"w2"' in patched
    assert '"w3"' in patched


def test_load_audit_instruments_the_m3_loader_alias_boundary():
    source = """
class MiniMaxM3MoE:
    pass


class MiniMaxM3Model:
    pass
"""

    patched, changed, found = _patch_append_load_audit(source)

    assert found
    assert changed
    assert 'MiniMaxM3Model' in patched
    assert 'mapping_aliases' in patched
    assert 'unsupported_routed' in patched
    assert 'scope=model' in patched


def test_load_audit_patch_is_idempotent():
    source = """
class MiniMaxM3MoE:
    pass


class MiniMaxM3Model:
    pass
"""

    patched, _, _ = _patch_append_load_audit(source)
    repatched, changed, found = _patch_append_load_audit(patched)

    assert found
    assert not changed
    assert repatched == patched


def test_load_audit_block_contains_bounded_parameter_fingerprints():
    assert 'M3_PARAM_FINGERPRINT' in _LOAD_AUDIT_BLOCK
    assert 'M3_PARAM_FINGERPRINT#' in _LOAD_AUDIT_BLOCK
    assert 'M3_PARAM_FINGERPRINT_SUMMARY#' in _LOAD_AUDIT_BLOCK
    assert '_llmc_fp_max_samples = 256' in _LOAD_AUDIT_BLOCK
    compile(_LOAD_AUDIT_BLOCK, "<m3-load-audit>", "exec")
