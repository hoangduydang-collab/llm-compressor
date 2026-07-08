#!/usr/bin/env python
"""Persistently patch the installed vLLM to serve MiniMax-M3 W4AFP8 (W4A8 MoE).

Unlike ``pipeline/vllm_m3_patches.py`` (an in-process monkeypatch used by
``serve_verify``), this edits the vLLM source files in the active venv **once** so
that any launch path -- including the production ``vllm serve`` HTTP server --
works without a runtime hook.

Two edits (see BUGS_AND_FIXES.md "W4A8 MoE ... SWIGLUOAI_UNINTERLEAVE"):

  1. fused_moe/experts/cutlass_moe.py
     Add ``MoEActivation.SWIGLUOAI_UNINTERLEAVE`` to
     ``CutlassExpertsW4A8Fp8._supports_activation`` (the only tuple-form
     ``_supports_activation`` with exactly SILU/GELU/SWIGLUOAI).

  2. fused_moe/activation.py
     In ``apply_moe_activation``'s ``SWIGLUOAI_UNINTERLEAVE`` branch, default the
     clamp scalars to the M3/gpt-oss SwiGLU-OAI constants when the W4A8 call site
     passes none (it does), instead of asserting.

  3. model_executor/layers/fused_allreduce_gemma_rms_norm.py
     When CUDA graphs are enabled, skip FlashInfer fused AR in
     ``_can_use_flashinfer`` (NCCL fallback — graph-capturable). See BUGS_AND_FIXES.md
     "CUDA graph capture".

  4. model_executor/layers/fused_moe/router/base_router.py
     ``nan_to_num`` on ``router_logits`` in ``RouterBase._select_experts`` (the
     template method every router subclass funnels through, right before
     ``_compute_routing``; padding NaNs → duplicate/OOB expert IDs → W4A8 MoE IMA;
     vLLM #39288 / #39391).

     NOTE: this replaced an earlier edit to ``MoERunner._apply_quant_method`` — a
     **dead path** for M3 W4AFP8 (it uses the modular ``FusedMoEModularKernel`` /
     ``router/*``, not ``MoERunner``), which is why the IMA at capture 16/51 never
     moved despite that patch verifying as applied. See BUGS_AND_FIXES.md
     "CUDA graph capture".

Idempotent: re-running is a no-op. Fails loudly if the expected code is not found
(so a vLLM upgrade that changes these files can't silently leave a broken serve).

Usage:
    python pipeline/slurm/patch_vllm_m3_serve.py            # apply
    python pipeline/slurm/patch_vllm_m3_serve.py --check    # report only, exit 1 if unpatched

Removal criteria: delete this script and revert once a vLLM release serves M3
W4A8 (SwiGLU-OAI uninterleaved) natively.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# M3 / gpt-oss SwiGLU-OAI constants: gate*sigmoid(alpha*gate)*(up+beta), clamped.
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0

_MARK = "llmc M3 W4A8 SWIGLUOAI_UNINTERLEAVE patch"
_CG_AR_MARK = "llmc M3 cudagraph: skip FlashInfer fused AR"
_CG_MOE_MARK = "llmc M3 cudagraph: nan_to_num router_logits in _select_experts"


def _vllm_dir() -> Path:
    import vllm

    return Path(vllm.__file__).resolve().parent


# FlashInfer >= 0.6.10 restored the finalizeMoeRoutingKernel bounds check that was
# dropped in 0.5.3 (flashinfer#2762). Missing it => padding tokens during CUDA
# graph capture index out-of-bounds in the MoE finalize -> deterministic IMA
# (vLLM #35706 / #42906). This is a *separate* suspect from the router NaN patch:
# it affects the flashinfer-backed MoE finalize, not vLLM's native W4A8 grouped
# GEMM. Report it so a stale quant-venv flashinfer is caught immediately.
_FLASHINFER_MIN_SAFE = (0, 6, 10)


def _report_flashinfer_version() -> None:
    try:
        import flashinfer  # type: ignore

        ver = getattr(flashinfer, "__version__", "?")
        print(f"flashinfer {ver}")
        parts = re.findall(r"\d+", str(ver))[:3]
        if len(parts) == 3:
            tup = tuple(int(p) for p in parts)
            if tup < _FLASHINFER_MIN_SAFE:
                print(
                    f"  WARNING: flashinfer {ver} < 0.6.10 lacks the "
                    "finalizeMoeRoutingKernel bounds-check fix (flashinfer#2762 / "
                    "vLLM #42906). If the CUDA-graph IMA is in a flashinfer MoE "
                    "finalize (confirm with CUDA_LAUNCH_BLOCKING=1), upgrade: "
                    '"$UV" pip install -U "flashinfer-python>=0.6.11.post2"'
                )
    except Exception as exc:  # noqa: BLE001
        print(f"flashinfer: not importable ({exc})")


def _patch_supports_activation(text: str) -> tuple[str, bool, bool]:
    """Add SWIGLUOAI_UNINTERLEAVE to the W4A8 tuple-form _supports_activation.

    Returns (new_text, changed, found).
    """
    # The W4A8 kernel is the ONLY class using a tuple (parentheses) with exactly
    # these three members; every other _supports_activation uses a list.
    pattern = re.compile(
        r"(?P<head>return\s+activation\s+in\s+\(\s*\n"
        r"(?P<ind>[ \t]+)MoEActivation\.SILU,[ \t]*\n"
        r"[ \t]+MoEActivation\.GELU,[ \t]*\n"
        r"[ \t]+MoEActivation\.SWIGLUOAI,[ \t]*\n)"
        r"(?P<close>[ \t]*\))",
        re.MULTILINE,
    )
    m = pattern.search(text)
    if m is None:
        # Either already patched (enum present) or layout changed.
        if "MoEActivation.SWIGLUOAI_UNINTERLEAVE" in text:
            return text, False, True
        return text, False, False

    ind = m.group("ind")
    injected = (
        m.group("head")
        + f"{ind}MoEActivation.SWIGLUOAI_UNINTERLEAVE,\n"
        + m.group("close")
    )
    return text[: m.start()] + injected + text[m.end() :], True, True


def _patch_apply_activation(text: str) -> tuple[str, bool, bool]:
    """Replace the SWIGLUOAI_UNINTERLEAVE assert with a clamp-scalar default."""
    assert_line = re.compile(
        r"^(?P<indent>[ \t]+)assert clamp_limit is not None,"
        r'\s*"SWIGLUOAI_UNINTERLEAVE requires clamp_limit"\s*$',
        re.MULTILINE,
    )
    if _MARK in text:
        return text, False, True

    m = assert_line.search(text)
    if m is None:
        return text, False, False

    indent = m.group("indent")
    replacement = (
        f"{indent}if clamp_limit is None:  # {_MARK}\n"
        f"{indent}    clamp_limit, alpha, beta = "
        f"{SWIGLU_LIMIT}, {SWIGLU_ALPHA}, {SWIGLU_BETA}"
    )
    return text[: m.start()] + replacement + text[m.end() :], True, True


def _patch_fused_ar_cudagraph(text: str) -> tuple[str, bool, bool]:
    """Skip FlashInfer fused AR when CUDA graphs are on; use NCCL fallback."""
    if _CG_AR_MARK in text:
        return text, False, True

    anchor = (
        'def _can_use_flashinfer(hidden_states: torch.Tensor, tp_size: int) -> tuple[bool, int]:\n'
        '    """Whether the flashinfer fused path applies; returns (ok, max_token_num)."""'
    )
    if anchor not in text:
        return text, False, False

    injection = (
        "    # llmc M3 cudagraph: FlashInfer fused AR+RMSNorm is not capturable on TP8\n"
        "    # (illegal memory access at capture_end; vLLM #46253). Use NCCL fallback.\n"
        f"    # {_CG_AR_MARK}\n"
        "    try:\n"
        "        from vllm.config import get_current_vllm_config\n"
        "\n"
        "        vc = get_current_vllm_config()\n"
        "        if vc is not None and not vc.enforce_eager:\n"
        "            return False, 0\n"
        "    except Exception:\n"
        "        pass\n"
    )
    new_text = text.replace(anchor, anchor + "\n" + injection, 1)
    return new_text, True, True


def _patch_moe_router_cudagraph(text: str) -> tuple[str, bool, bool]:
    """Sanitize NaN router logits at the real MoE routing entry (cudagraph padding).

    Injects ``router_logits = torch.nan_to_num(...)`` in ``RouterBase._select_experts``
    (``fused_moe/router/base_router.py``) — the template method every router
    subclass funnels through — right before it delegates to ``_compute_routing``.
    This is where vLLM maintainers pointed for the #39288 class of IMA (padding
    tokens → NaN/garbage logits → duplicate/OOB expert IDs → CUTLASS MoE out-of-
    bounds during graph capture). One edit covers fused_topk / grouped_topk / bias
    / custom routers.

    The anchor is the ``topk_weights, topk_ids = self._compute_routing(`` call,
    which appears once. Insert the sanitizer on the line before it, at the same
    indentation.
    """
    if _CG_MOE_MARK in text:
        return text, False, True

    pattern = re.compile(
        r"(?P<ind>[ \t]+)topk_weights, topk_ids = self\._compute_routing\(",
    )
    m = pattern.search(text)
    if m is None:
        return text, False, False

    ind = m.group("ind")
    injection = (
        f"{ind}# llmc M3 cudagraph: padding tokens -> NaN/garbage router logits ->\n"
        f"{ind}# duplicate/OOB expert IDs -> W4A8 CUTLASS MoE illegal memory access\n"
        f"{ind}# during graph capture (vLLM #39288 / #39391). No-op on real logits.\n"
        f"{ind}# {_CG_MOE_MARK}\n"
        f"{ind}router_logits = torch.nan_to_num(\n"
        f"{ind}    router_logits, nan=0.0, posinf=0.0, neginf=0.0\n"
        f"{ind})\n"
    )
    new_text = text[: m.start()] + injection + text[m.start() :]
    return new_text, True, True


def _apply(path: Path, patch_fn, check_only: bool, *, fatal: bool = True) -> bool:
    """Return True if the file is patched (already or newly)."""
    text = path.read_text(encoding="utf-8")
    new_text, changed, found = patch_fn(text)
    if not found:
        msg = f"ERROR: expected code not found in {path} (vLLM layout changed?)"
        if fatal:
            print(msg)
            sys.exit(2)
        print(msg)
        return False
    if changed and not check_only:
        path.write_text(new_text, encoding="utf-8")
        print(f"patched: {path}")
    elif changed and check_only:
        print(f"UNPATCHED: {path}")
    else:
        print(f"already patched: {path}")
    return not changed


def _patch_targets(vllm_dir: Path) -> list[tuple[str, Path, object]]:
    return [
        ("W4A8 SWIGLU support", vllm_dir / "model_executor/layers/fused_moe/experts/cutlass_moe.py", _patch_supports_activation),
        ("W4A8 SWIGLU clamp", vllm_dir / "model_executor/layers/fused_moe/activation.py", _patch_apply_activation),
        ("cudagraph fused AR", vllm_dir / "model_executor/layers/fused_allreduce_gemma_rms_norm.py", _patch_fused_ar_cudagraph),
        ("cudagraph MoE router", vllm_dir / "model_executor/layers/fused_moe/router/base_router.py", _patch_moe_router_cudagraph),
    ]


def ensure_vllm_m3_patches(*, apply: bool = True) -> None:
    """Apply (if needed) and verify all four persistent vLLM M3 serve patches.

  vLLM worker subprocesses are spawned fresh — in-process monkeypatches in
  ``serve_verify`` do **not** reach ``Worker_TP*``. This must edit site-packages.

  Raises RuntimeError if any patch cannot be applied or verified.
    """
    vllm_dir = _vllm_dir()
    missing_files: list[str] = []
    unpatched: list[str] = []
    for label, path, patch_fn in _patch_targets(vllm_dir):
        if not path.exists():
            missing_files.append(str(path))
            continue
        ok = _apply(path, patch_fn, check_only=not apply, fatal=False)
        if not ok:
            unpatched.append(label)

    if missing_files:
        raise RuntimeError(
            "vLLM M3 serve files not found (wrong vLLM build?). Missing:\n  "
            + "\n  ".join(missing_files)
            + "\nInstall: bash pipeline/slurm/install_vllm_m3_serve.sh"
        )
    if unpatched:
        raise RuntimeError(
            "vLLM M3 serve patches missing in site-packages: "
            + ", ".join(unpatched)
            + ". Run: python pipeline/slurm/patch_vllm_m3_serve.py"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if unpatched")
    args = ap.parse_args()

    vllm_dir = _vllm_dir()
    for label, path, _ in _patch_targets(vllm_dir):
        if not path.exists():
            print(f"ERROR: {path} not found ({label}); is this the W4A8-MoE vLLM build?")
            return 2

    import vllm

    print(f"vLLM {getattr(vllm, '__version__', '?')} at {vllm_dir}")
    _report_flashinfer_version()
    results = [
        _apply(path, patch_fn, args.check)
        for _, path, patch_fn in _patch_targets(vllm_dir)
    ]

    if args.check:
        already = all(results)
        print("STATUS:", "patched" if already else "NOT patched")
        return 0 if already else 1

    print(
        "\nDone. Recompile of C++/CUDA is NOT required (pure-Python edits).\n"
        "Re-run after any vLLM reinstall. Then serve normally, e.g.:\n"
        "  vllm serve <ckpt> --tensor-parallel-size 8 --enable-expert-parallel ..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
