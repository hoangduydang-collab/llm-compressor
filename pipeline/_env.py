"""Environment safeguards for serving/eval.

Some clusters set ``$HOME`` to a non-writable path (e.g. ``/home/<user>`` when
the real working dir is on NFS). vLLM/FlashInfer create a JIT workspace under
``$HOME`` (``~/.cache/flashinfer``) at import time, which then fails with
``PermissionError``. Call ``ensure_writable_caches()`` BEFORE importing vLLM so
those caches land in a writable location.
"""

import os
import tempfile
from pathlib import Path


def _is_writable(path: str) -> bool:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return os.access(path, os.W_OK)
    except Exception:
        return False


def ensure_writable_caches() -> dict:
    """Redirect HOME / FlashInfer workspace to a writable dir if needed.

    Returns a dict describing what was set, for logging.
    """
    changed: dict = {}

    home = os.environ.get("HOME", "")
    if not home or not _is_writable(home):
        fallback = (
            os.environ.get("WORK_ROOT")
            or os.environ.get("TMPDIR")
            or tempfile.gettempdir()
        )
        os.environ["HOME"] = fallback
        home = fallback
        changed["HOME"] = fallback

    if not os.environ.get("FLASHINFER_WORKSPACE_DIR"):
        ws = str(Path(home) / "cache" / "flashinfer")
        os.environ["FLASHINFER_WORKSPACE_DIR"] = ws
        changed["FLASHINFER_WORKSPACE_DIR"] = ws
    Path(os.environ["FLASHINFER_WORKSPACE_DIR"]).mkdir(parents=True, exist_ok=True)

    return changed


def apply_sglang_compat_env() -> dict[str, str]:
    """SGLang eval fallbacks for clusters without a working NVCC toolkit.

    Must run before ``import lm_eval`` / ``import sglang`` so DeepGEMM picks
    NVRTC and SGLang reads ``SGLANG_ENABLE_JIT_DEEPGEMM`` at import time.
    """
    applied: dict[str, str] = {}

    def _set(key: str, value: str) -> None:
        if os.environ.get(key) != value:
            os.environ[key] = value
            applied[key] = value

    _set("FLASHINFER_USE_CUDA_NORM", "1")
    # SGLang 0.5.x env name (SGL_ENABLE_JIT_DEEPGEMM is not read).
    _set("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
    # DSA indexer still calls deep_gemm directly; NVRTC avoids a broken nvcc.
    _set("SGL_DG_USE_NVRTC", "1")
    _set("DG_JIT_USE_NVRTC", "1")

    return applied
