"""Environment safeguards for serving/eval.

Some clusters set ``$HOME`` to a non-writable path (e.g. ``/home/<user>`` when
the real working dir is on NFS). vLLM/FlashInfer create a JIT workspace under
``$HOME`` (``~/.cache/flashinfer``) at import time, which then fails with
``PermissionError``. Call ``ensure_writable_caches()`` BEFORE importing vLLM so
those caches land in a writable location.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# DeepGEMM's __int128 PTX (st.shared.b128 + constraint "q") needs nvcc >= 12.9.
DEEPGEMM_MIN_NVCC = (12, 9)


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


def _parse_nvcc_release(text: str) -> tuple[int, int] | None:
    match = re.search(r"release (\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _nvcc_version(nvcc_path: str) -> tuple[int, int] | None:
    try:
        proc = subprocess.run(
            [nvcc_path, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return _parse_nvcc_release(proc.stdout + proc.stderr)
    except Exception:
        return None


def _iter_nvcc_candidates() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    def add(path: str | Path | None) -> None:
        if not path:
            return
        resolved = str(Path(path).resolve())
        if resolved not in seen and Path(resolved).is_file():
            seen.add(resolved)
            ordered.append(resolved)

    add(os.environ.get("DG_JIT_NVCC_COMPILER"))
    add(shutil.which("nvcc"))
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME:
            add(Path(CUDA_HOME) / "bin" / "nvcc")
    except Exception:
        pass

    try:
        import site

        roots = list(site.getsitepackages())
        user = site.getusersitepackages()
        if user:
            roots.append(user)
        for root in roots:
            add(Path(root) / "nvidia" / "cuda_nvcc" / "bin" / "nvcc")
    except Exception:
        pass

    return ordered


def ensure_deepgemm_nvcc(
    min_version: tuple[int, int] = DEEPGEMM_MIN_NVCC,
) -> dict[str, str]:
    """Pick nvcc >= 12.9 for DeepGEMM JIT (sets ``DG_JIT_NVCC_COMPILER``)."""
    applied: dict[str, str] = {}
    best_path: str | None = None
    best_ver: tuple[int, int] | None = None

    for path in _iter_nvcc_candidates():
        ver = _nvcc_version(path)
        if ver is None or ver < min_version:
            continue
        if best_ver is None or ver > best_ver:
            best_path, best_ver = path, ver

    if best_path:
        if os.environ.get("DG_JIT_NVCC_COMPILER") != best_path:
            os.environ["DG_JIT_NVCC_COMPILER"] = best_path
            applied["DG_JIT_NVCC_COMPILER"] = best_path
        # Prefer nvcc 12.9+ over NVRTC for DSA paged_mqa kernels.
        for key, value in (
            ("DG_JIT_USE_NVRTC", "0"),
            ("SGLANG_DG_USE_NVRTC", "0"),
        ):
            if os.environ.get(key) != value:
                os.environ[key] = value
                applied[key] = value

    return applied


def apply_sglang_compat_env() -> dict[str, str]:
    """SGLang eval fallbacks for clusters without a working NVCC toolkit.

    Must run before ``import lm_eval`` / ``import sglang`` so DeepGEMM picks
    the right compiler and SGLang reads env at import time.
    """
    applied: dict[str, str] = {}

    def _set(key: str, value: str) -> None:
        if os.environ.get(key) != value:
            os.environ[key] = value
            applied[key] = value

    _set("FLASHINFER_USE_CUDA_NORM", "1")
    _set("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
    applied.update(ensure_deepgemm_nvcc())
    if "DG_JIT_NVCC_COMPILER" not in applied:
        # Last resort when only an old system nvcc exists.
        _set("SGLANG_DG_USE_NVRTC", "1")
        _set("DG_JIT_USE_NVRTC", "1")

    return applied


def preflight_sglang_deepgemm() -> list[str]:
    """Warn when nvcc is missing or too old for DeepGEMM DSA kernels."""
    msgs: list[str] = []
    candidates = _iter_nvcc_candidates()
    if not candidates:
        msgs.append(
            "nvcc not found. GLM-5.2 DSA needs DeepGEMM JIT with nvcc >= 12.9. "
            "Install: source /mnt/nfs/hoangduy/env.sh && "
            '"$UV" pip install "nvidia-cuda-nvcc-cu12==12.9.86"'
        )
        return msgs

    best_path: str | None = None
    best_ver: tuple[int, int] | None = None
    for path in candidates:
        ver = _nvcc_version(path)
        if ver is None:
            continue
        note = f"nvcc {ver[0]}.{ver[1]} at {path}"
        if ver < DEEPGEMM_MIN_NVCC:
            msgs.append(
                f"{note} is too old for DeepGEMM (__int128 PTX needs >= 12.9)."
            )
            continue
        if best_ver is None or ver > best_ver:
            best_path, best_ver = path, ver

    if best_path:
        msgs.append(
            f"DeepGEMM will use nvcc {best_ver[0]}.{best_ver[1]} at {best_path}"
        )
    elif not any("too old" in m for m in msgs):
        msgs.append("could not parse nvcc version from candidate paths")

    if best_path is None:
        msgs.append(
            'Fix: source /mnt/nfs/hoangduy/env.sh && '
            '"$UV" pip install \\"nvidia-cuda-nvcc-cu12==12.9.86\\" in sglang-eval, '
            "then rm -rf ~/.cache/deep_gemm/tmp and re-run compile_deep_gemm / eval."
        )

    return msgs
