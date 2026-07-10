"""Unit tests for MiniMax-M3 CUDA-graph evidence classifier (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.m3_cudagraph_evidence import classify_files, classify_log


def test_server_ready_with_chat_ok():
    log = "INFO: Application startup complete\nUvicorn running\n"
    rec = classify_log(log, chat_ok=True, graphs_on=True, debug_cudagraph=False)
    assert rec.verdict == "server_ready"
    assert rec.server_ready is True
    assert rec.ima is False


def test_masked_pass_under_debug_cudagraph():
    log = "Application startup complete\n"
    rec = classify_log(log, chat_ok=True, graphs_on=True, debug_cudagraph=True)
    assert rec.verdict == "masked_pass"
    assert any("DEBUG_CUDAGRAPH" in n or "LAUNCH_BLOCKING" in n for n in rec.notes)


def test_moe_symbol_outranks_empty_cache():
    log = """
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 16/51
File ".../breakable_cudagraph.py", line 1, in _capture
    torch.cuda.empty_cache()
CUDA error: an illegal memory access was encountered
finalizeMoeRoutingKernel<<<...>>> illegal address
"""
    rec = classify_log(log, graphs_on=True, debug_cudagraph=False)
    assert rec.verdict == "graph_ima_moe"
    assert "finalizeMoeRoutingKernel" in rec.moe_symbols
    assert rec.empty_cache_near_ima is True
    assert any("empty_cache near IMA" in n for n in rec.notes)


def test_collective_outranks_generic_graph_frames():
    log = """
Capturing CUDA graphs: 43/51
breakable_cudagraph._capture
capture_end()
CUDA error: an illegal memory access was encountered
fused_allreduce_gemma_rms_norm
"""
    rec = classify_log(log, graphs_on=True)
    assert rec.verdict == "graph_ima_collective"
    assert "fused_allreduce_gemma_rms_norm" in rec.collective_symbols


def test_memory_lifetime_when_only_empty_cache():
    log = """
Capturing CUDA graphs: 20/51
File "breakable_cudagraph.py", line 10, in _capture
    torch.cuda.empty_cache()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
"""
    rec = classify_log(log, graphs_on=True)
    assert rec.verdict == "graph_ima_memory_lifetime"
    assert rec.empty_cache_near_ima is True


def test_unclassified_ima():
    log = """
Capturing CUDA graphs: 5/51
CUDA error: an illegal memory access was encountered
Worker failed
"""
    rec = classify_log(log, graphs_on=True)
    assert rec.verdict == "graph_ima_unclassified"


def test_graphs_off_failed():
    log = "CUDA error: an illegal memory access was encountered\n"
    rec = classify_log(log, graphs_on=False, debug_cudagraph=False)
    assert rec.verdict == "graphs_off_failed"


def test_startup_without_chat_is_not_full_pass_note():
    log = "Application startup complete\n"
    rec = classify_log(log, chat_ok=None, graphs_on=True, debug_cudagraph=False)
    assert rec.verdict == "server_ready"
    assert any("chat_ok not provided" in n for n in rec.notes)


def test_ready_but_chat_failed_is_inconclusive():
    log = "Application startup complete\n"
    rec = classify_log(log, chat_ok=False, graphs_on=True, debug_cudagraph=False)
    assert rec.verdict == "inconclusive"


def test_faulting_kernel_overrides_log_symbols():
    log = "CUDA error: an illegal memory access\nempty_cache()\n"
    rec = classify_log(
        log,
        graphs_on=True,
        faulting_kernel="finalizeMoeRoutingKernel",
    )
    assert rec.verdict == "graph_ima_moe"


def test_compile_fallback_inconclusive():
    log = "torch.compile ... does not support it\nAssertionError: Compilation mode should be\n"
    rec = classify_log(log, graphs_on=True)
    assert rec.verdict == "inconclusive"


def test_classify_files_reads_chat_json(tmp_path: Path):
    log = tmp_path / "serve.log"
    chat = tmp_path / "chat.json"
    meta = tmp_path / "meta.json"
    log.write_text("Application startup complete\n", encoding="utf-8")
    chat.write_text(
        json.dumps({"choices": [{"message": {"content": "Paris"}}]}),
        encoding="utf-8",
    )
    meta.write_text(
        json.dumps(
            {
                "graphs_on": True,
                "debug_cudagraph": False,
                "case_name": "async_baseline",
            }
        ),
        encoding="utf-8",
    )
    rec = classify_files(log, chat_path=chat, meta=json.loads(meta.read_text()))
    assert rec.verdict == "server_ready"
    assert rec.chat_ok is True
    assert rec.case_name == "async_baseline"


def test_moe_beats_collective_if_both_present():
    # Precedence: MoE first in the if-chain.
    log = """
illegal memory access
finalizeMoeRoutingKernel
fused_allreduce_gemma_rms_norm
"""
    rec = classify_log(log, graphs_on=True)
    assert rec.verdict == "graph_ima_moe"
