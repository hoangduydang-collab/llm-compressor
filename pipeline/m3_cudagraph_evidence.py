"""Classify MiniMax-M3 HTTP CUDA-graph IMA evidence from serve logs.

Pure text/JSON analysis — no GPU, no vLLM import. Used by the cluster matrix
runner (``pipeline/slurm/test_m3_http_cudagraph_matrix.sh``) and unit tests.

Outcomes (``verdict``):

- ``server_ready`` — Application startup complete (+ optional chat ok).
- ``masked_pass`` — ready only under ``CUDA_LAUNCH_BLOCKING`` / DEBUG_CUDAGRAPH.
- ``graph_ima_moe`` — IMA with MoE routing/finalize/CUTLASS symbols.
- ``graph_ima_collective`` — IMA with fused AR / NCCL / all-reduce symbols.
- ``graph_ima_memory_lifetime`` — IMA with empty_cache / workspace realloc hints
  and no stronger MoE/collective symbol.
- ``graph_ima_unclassified`` — IMA during/around graph capture, no named class.
- ``graphs_off_failed`` — failure while enforce_eager / graphs off.
- ``inconclusive`` — unsupported path, compile fallback, or insufficient signal.

Precedence for IMA classification: MoE symbol > collective symbol >
memory-lifetime hint > unclassified. A deferred ``empty_cache`` frame alone
never outranks a named MoE/collective symbol elsewhere in the log.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --- symbol tables -----------------------------------------------------------

_MOE_SYMBOLS = (
    "finalizeMoeRoutingKernel",
    "finalize_moe_routing",
    "topk_softmax",
    "topkGating",
    "moe_unpermute",
    "cutlass_moe",
    "grouped_gemm",
    "cutlass_grouped",
    "FusedMoE",
    "blockExpertPrefixSum",
    "moe_align_block_size",
)

_COLLECTIVE_SYMBOLS = (
    "fused_allreduce_gemma_rms_norm",
    "fused_allreduce",
    "custom_all_reduce",
    "flashinfer_all_reduce",
    "ncclAllReduce",
    "ProcessGroupNCCL",
    "tensor_model_parallel_all_reduce",
    "all_reduce.cuh",
)

_MEMORY_HINTS = (
    "empty_cache",
    "WorkspaceManager",
    "workspace realloc",
    "stale buffer",
    "stale pointer",
    "graph pool",
)

_IMA_RE = re.compile(
    r"illegal memory access|cudaErrorIllegalAddress|CUDA error: an illegal",
    re.IGNORECASE,
)
_READY_RE = re.compile(r"Application startup complete", re.IGNORECASE)
_CAPTURE_RE = re.compile(
    r"Capturing CUDA graphs|Graph capturing finished|cudagraph",
    re.IGNORECASE,
)
_BREAKABLE_RE = re.compile(
    r"breakable_cudagraph|VLLM_USE_BREAKABLE_CUDAGRAPH|Auto-enabling "
    r"VLLM_USE_BREAKABLE_CUDAGRAPH",
    re.IGNORECASE,
)
_COMPILE_FALLBACK_RE = re.compile(
    r"does not support it|Compilation mode should be|falling back to eager|"
    r"enforce_eager|cudagraph_mode.*NONE",
    re.IGNORECASE,
)
_CAPTURE_END_RE = re.compile(r"capture_end", re.IGNORECASE)


@dataclass
class EvidenceRecord:
    """One trial's classified evidence."""

    verdict: str
    server_ready: bool = False
    chat_ok: bool | None = None
    ima: bool = False
    graphs_on: bool | None = None
    debug_cudagraph: bool | None = None
    breakable_cudagraph_mentioned: bool = False
    capture_seen: bool = False
    moe_symbols: list[str] = field(default_factory=list)
    collective_symbols: list[str] = field(default_factory=list)
    memory_hints: list[str] = field(default_factory=list)
    capture_end_seen: bool = False
    empty_cache_near_ima: bool = False
    first_ima_line: str | None = None
    core_dump_path: str | None = None
    core_dump_exists: bool | None = None
    faulting_kernel: str | None = None
    notes: list[str] = field(default_factory=list)
    case_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_symbols(text: str, symbols: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    lower = text.lower()
    for sym in symbols:
        if sym.lower() in lower:
            found.append(sym)
    return found


def _first_ima_line(text: str) -> str | None:
    for line in text.splitlines():
        if _IMA_RE.search(line):
            return line.strip()[:500]
    return None


def _empty_cache_near_ima(text: str, window: int = 40) -> bool:
    lines = text.splitlines()
    ima_idxs = [i for i, ln in enumerate(lines) if _IMA_RE.search(ln)]
    if not ima_idxs:
        return False
    for i in ima_idxs:
        lo, hi = max(0, i - window), min(len(lines), i + window + 1)
        chunk = "\n".join(lines[lo:hi])
        if "empty_cache" in chunk:
            return True
    return False


def classify_log(
    log_text: str,
    *,
    chat_ok: bool | None = None,
    graphs_on: bool | None = None,
    debug_cudagraph: bool | None = None,
    core_dump_path: str | None = None,
    faulting_kernel: str | None = None,
    case_name: str | None = None,
) -> EvidenceRecord:
    """Classify a serve log (+ optional chat / core-dump metadata)."""
    notes: list[str] = []
    server_ready = bool(_READY_RE.search(log_text))
    ima = bool(_IMA_RE.search(log_text))
    capture_seen = bool(_CAPTURE_RE.search(log_text))
    breakable = bool(_BREAKABLE_RE.search(log_text))
    capture_end = bool(_CAPTURE_END_RE.search(log_text))
    moe = _find_symbols(log_text, _MOE_SYMBOLS)
    coll = _find_symbols(log_text, _COLLECTIVE_SYMBOLS)
    mem = _find_symbols(log_text, _MEMORY_HINTS)
    empty_near = _empty_cache_near_ima(log_text)
    first_ima = _first_ima_line(log_text) if ima else None

    dump_exists: bool | None = None
    if core_dump_path:
        # Expand printf-style %h.%p.%t by globbing the directory prefix.
        p = Path(core_dump_path)
        if "%" in p.name:
            matches = list(p.parent.glob(p.name.split("%")[0] + "*")) if p.parent.exists() else []
            dump_exists = bool(matches)
            if matches:
                core_dump_path = str(matches[0])
        else:
            dump_exists = p.exists()

    # Explicit graphs_on from trial config wins; else infer from log / flags.
    if graphs_on is None:
        if "enforce_eager" in log_text.lower() and re.search(
            r"enforce[_-]eager[=:\s]+(true|1|True)", log_text
        ):
            graphs_on = False
        elif capture_seen:
            graphs_on = True

    verdict: str

    if faulting_kernel:
        fk = faulting_kernel.lower()
        if any(s.lower() in fk for s in _MOE_SYMBOLS):
            verdict = "graph_ima_moe"
            notes.append(f"faulting_kernel={faulting_kernel}")
        elif any(s.lower() in fk for s in _COLLECTIVE_SYMBOLS):
            verdict = "graph_ima_collective"
            notes.append(f"faulting_kernel={faulting_kernel}")
        else:
            verdict = "graph_ima_unclassified"
            notes.append(f"faulting_kernel={faulting_kernel} (unmapped)")
    elif server_ready and not ima:
        if chat_ok is False:
            verdict = "inconclusive"
            notes.append("server ready but chat failed")
        elif debug_cudagraph:
            verdict = "masked_pass"
            notes.append("ready under DEBUG_CUDAGRAPH/CUDA_LAUNCH_BLOCKING — not a root fix")
        else:
            verdict = "server_ready"
            if chat_ok is None:
                notes.append("server ready; chat_ok not provided")
            elif chat_ok:
                notes.append("server ready + chat ok")
    elif ima:
        if graphs_on is False:
            verdict = "graphs_off_failed"
            notes.append("IMA while graphs off — not graph-capture-specific")
        elif moe:
            verdict = "graph_ima_moe"
            notes.append(f"MoE symbols: {', '.join(moe)}")
        elif coll:
            verdict = "graph_ima_collective"
            notes.append(f"collective symbols: {', '.join(coll)}")
        elif empty_near or (mem and ("empty_cache" in mem or capture_end)):
            verdict = "graph_ima_memory_lifetime"
            notes.append("empty_cache / workspace / capture_end memory-lifetime hints")
        else:
            verdict = "graph_ima_unclassified"
            notes.append("IMA without named MoE/collective/memory symbol")
        if empty_near and verdict != "graph_ima_memory_lifetime":
            notes.append("empty_cache near IMA (deferred report; lower precedence)")
    elif graphs_on is False and not server_ready:
        verdict = "graphs_off_failed"
        notes.append("graphs-off trial did not become ready")
    elif _COMPILE_FALLBACK_RE.search(log_text) and not server_ready:
        verdict = "inconclusive"
        notes.append("compile/breakable fallback or unsupported path")
    else:
        verdict = "inconclusive"
        notes.append("insufficient signal (no ready, no IMA)")

    return EvidenceRecord(
        verdict=verdict,
        server_ready=server_ready,
        chat_ok=chat_ok,
        ima=ima,
        graphs_on=graphs_on,
        debug_cudagraph=debug_cudagraph,
        breakable_cudagraph_mentioned=breakable,
        capture_seen=capture_seen,
        moe_symbols=moe,
        collective_symbols=coll,
        memory_hints=mem,
        capture_end_seen=capture_end,
        empty_cache_near_ima=empty_near,
        first_ima_line=first_ima,
        core_dump_path=core_dump_path,
        core_dump_exists=dump_exists,
        faulting_kernel=faulting_kernel,
        notes=notes,
        case_name=case_name,
    )


def classify_files(
    log_path: str | Path,
    *,
    chat_path: str | Path | None = None,
    meta: dict[str, Any] | None = None,
) -> EvidenceRecord:
    """Load log (+ optional chat JSON / meta) and classify."""
    log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    meta = dict(meta or {})
    chat_ok = meta.get("chat_ok")
    if chat_path is not None and Path(chat_path).exists():
        raw = Path(chat_path).read_text(encoding="utf-8", errors="replace")
        try:
            blob = json.loads(raw)
            chat_ok = bool(
                blob.get("choices")
                or blob.get("ok")
                or (isinstance(blob.get("chat_ok"), bool) and blob["chat_ok"])
            )
            if "error" in blob and not blob.get("choices"):
                chat_ok = False
        except json.JSONDecodeError:
            # curl json.tool success usually has "choices"
            chat_ok = '"choices"' in raw and "error" not in raw.lower()

    return classify_log(
        log_text,
        chat_ok=chat_ok,
        graphs_on=meta.get("graphs_on"),
        debug_cudagraph=meta.get("debug_cudagraph"),
        core_dump_path=meta.get("core_dump_path"),
        faulting_kernel=meta.get("faulting_kernel"),
        case_name=meta.get("case_name"),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="vLLM serve log path")
    ap.add_argument("--chat", type=Path, default=None, help="optional chat JSON")
    ap.add_argument("--meta", type=Path, default=None, help="optional trial meta JSON")
    ap.add_argument("-o", "--output", type=Path, default=None, help="write JSON here")
    args = ap.parse_args(argv)

    meta: dict[str, Any] = {}
    if args.meta and args.meta.exists():
        meta = json.loads(args.meta.read_text(encoding="utf-8"))

    rec = classify_files(args.log, chat_path=args.chat, meta=meta)
    blob = json.dumps(rec.to_dict(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(blob + "\n", encoding="utf-8")
    print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
