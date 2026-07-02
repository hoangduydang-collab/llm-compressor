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
    _set("SGLANG_DG_USE_NVRTC", "1")
    _set("DG_JIT_USE_NVRTC", "1")

    return applied


def preflight_sglang_deepgemm() -> list[str]:
    """Warn when nvcc is missing or likely too old for DeepGEMM DSA kernels."""
    import re
    import shutil
    import subprocess

    msgs: list[str] = []
    nvcc = shutil.which("nvcc")
    if not nvcc:
        try:
            from torch.utils.cpp_extension import CUDA_HOME

            if CUDA_HOME:
                candidate = Path(CUDA_HOME) / "bin" / "nvcc"
                if candidate.is_file():
                    nvcc = str(candidate)
        except Exception:
            pass
    if not nvcc:
        msgs.append(
            "nvcc not found (GLM-5.2 DSA indexer JIT-compiles DeepGEMM at first forward). "
            "Load a CUDA toolkit module and export CUDA_HOME, e.g. "
            "module load cuda/12.6 && export CUDA_HOME=$CUDA_HOME."
        )
        return msgs
    try:
        proc = subprocess.run(
            [nvcc, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        text = proc.stdout + proc.stderr
        match = re.search(r"release (\d+)\.(\d+)", text)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            if major < 12 or (major == 12 and minor < 4):
                msgs.append(
                    f"nvcc {major}.{minor} at {nvcc} is likely too old for DeepGEMM "
                    f"(__int128 PTX / st.shared.b128). Use CUDA toolkit >= 12.4 "
                    f"(12.8+ recommended to match torch)."
                )
        msgs.append(f"DeepGEMM will JIT with nvcc: {nvcc}")
    except Exception as exc:
        msgs.append(f"could not run nvcc --version ({exc})")
    return msgs
