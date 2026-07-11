"""CPU-only tests for persistent MiniMax-M3 vLLM diagnostics."""

from pipeline.slurm.patch_vllm_m3_serve import (
    _BOUNDARY_BLOCK,
    _LOAD_AUDIT_BLOCK,
    _PROBE_BLOCK,
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


def test_parameter_fingerprint_uses_correct_top_level_class_and_safe_sampling():
    assert (
        'globals().get("MiniMaxM3SparseForConditionalGeneration")'
        in _LOAD_AUDIT_BLOCK
    )
    assert (
        '"MiniMaxM3SparseForConditionalGeneration"\n'
        '            "MiniMax'
        not in _LOAD_AUDIT_BLOCK
    )
    assert "linspace(" not in _LOAD_AUDIT_BLOCK
    assert "index_select(" not in _LOAD_AUDIT_BLOCK
    assert "[::_stride][:_sample_count]" in _LOAD_AUDIT_BLOCK


def test_moe_probe_covers_canonical_prefill_and_emits_structured_digests():
    assert "M3_MOE_PROBE_MAX_TOKENS" in _PROBE_BLOCK
    assert '"rank"' in _PROBE_BLOCK
    assert '"input_sample_sha256"' in _PROBE_BLOCK
    assert '"output_sample_sha256"' in _PROBE_BLOCK
    assert '"routed_sample_sha256"' in _PROBE_BLOCK
    assert "M3_MOE_PROBE# %s" in _PROBE_BLOCK
    compile(_PROBE_BLOCK, "<m3-moe-probe>", "exec")


def test_explicit_diagnostic_setup_errors_are_not_swallowed():
    marker = "raise  # explicit diagnostics must fail loudly"

    assert marker in _LOAD_AUDIT_BLOCK
    assert marker in _PROBE_BLOCK


def test_layer_boundary_probe_is_layer_resolved_bounded_and_capture_safe():
    for boundary in (
        "decoder_input_hidden",
        "decoder_input_residual",
        "attention_input",
        "attention_output",
        "moe_input",
        "moe_output",
        "decoder_output_hidden",
        "decoder_output_residual",
    ):
        assert boundary in _BOUNDARY_BLOCK
    assert 'M3_LAYER_BOUNDARY_LAYERS' in _BOUNDARY_BLOCK
    assert '"layer"' in _BOUNDARY_BLOCK
    assert '"sample_sha256"' in _BOUNDARY_BLOCK
    assert '"finite_fraction"' in _BOUNDARY_BLOCK
    assert "is_current_stream_capturing" in _BOUNDARY_BLOCK
    assert "layer_id" in _BOUNDARY_BLOCK
    assert "M3_LAYER_BOUNDARY# %s" in _BOUNDARY_BLOCK
    compile(_BOUNDARY_BLOCK, "<m3-layer-boundary>", "exec")


def test_router_is_included_in_load_and_parameter_audits():
    assert "block_sparse_moe.gate" in _LOAD_AUDIT_BLOCK
    assert '"moe_router"' in _LOAD_AUDIT_BLOCK
