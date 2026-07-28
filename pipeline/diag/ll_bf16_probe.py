"""Probe vLLM 0.26.0's cuteDSL ll_bf16 router GEMM at MiniMax-M3's router shape.

Hypothesis under test: M3's router gate is (K=6144, N=128), which appears in none of
ll_bf16's tables -- not _LL_BF16_WARMUP_MODEL_SHAPES, not _TUNED_DOTPROD_MAX_M, not
_TUNED_CONFIGS (all DeepSeek/Inkling shapes). Dispatch sends M<=4 to the dot-product
kernel and M>=5 to a split-K kernel with untuned defaults (split_k=6, num_stages=4).
Our conc-1 serve (M=1, dotprod) ran clean; our conc-10 serve (M~10, split-K) IMA'd.

Run plain for correctness, and under `compute-sanitizer --tool memcheck` for OOB.
Exits non-zero if any shape errors or exceeds the correctness tolerance.
"""

import os
import sys
import traceback

import torch

M3 = (6144, 128)  # MiniMax-M3: hidden_size, num_local_experts   <- untuned everywhere
DSV3 = (7168, 256)  # DeepSeek-V3 control: present in _TUNED_DOTPROD_MAX_M
INKLING = (6144, 264)  # present in the warmup list

M_VALUES = [int(v) for v in os.environ.get("PROBE_M", "1,2,3,4,5,6,8,10,12,16").split(",")]
SHAPES = {"M3": M3, "DSV3(control)": DSV3, "INKLING(control)": INKLING}
# bf16 accumulation in a split-K reduction: generous but still catches garbage.
TOL = float(os.environ.get("PROBE_TOL", "0.15"))


def main() -> int:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        is_available,
        ll_bf16_gemm,
        ll_bf16_gemm_kernel,
    )

    if not is_available():
        print("FATAL: cuteDSL unavailable in this venv; nothing to probe")
        return 2

    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"tolerance {TOL}  M values {M_VALUES}\n")

    failures = []
    for name, (K, N) in SHAPES.items():
        w = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
        for M in M_VALUES:
            key = ll_bf16_gemm_kernel.dispatch(M=M, K=K, N=N)
            backend = key.backend
            tag = f"{name:18s} K={K:5d} N={N:4d} M={M:3d} -> {backend:8s}"
            x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
            try:
                out = ll_bf16_gemm(x, w)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                print(f"{tag}  RAISED {type(exc).__name__}: {exc}")
                traceback.print_exc()
                failures.append((name, M, backend, f"raised {type(exc).__name__}"))
                # A CUDA error poisons the context; no point continuing.
                if "CUDA" in type(exc).__name__ or "cuda" in str(exc).lower():
                    print("\nCUDA context poisoned -- aborting remaining shapes.")
                    _summary(failures)
                    return 1
                continue

            ref = (x.float() @ w.float().t()).to(out.dtype)
            denom = ref.abs().max().clamp_min(1e-6)
            err = ((out - ref).abs().max() / denom).item()
            bad = (not torch.isfinite(out).all().item()) or err > TOL
            print(f"{tag}  rel_err={err:.4g}  finite={torch.isfinite(out).all().item()}"
                  f"{'   <<< FAIL' if bad else ''}")
            if bad:
                failures.append((name, M, backend, f"rel_err={err:.4g}"))

    return _summary(failures)


def _summary(failures) -> int:
    print()
    if not failures:
        print("RESULT: all shapes clean (no raise, finite, within tolerance)")
        return 0
    print(f"RESULT: {len(failures)} failing configuration(s):")
    for name, M, backend, why in failures:
        print(f"  {name} M={M} backend={backend}: {why}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
