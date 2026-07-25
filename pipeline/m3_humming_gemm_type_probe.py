"""Arm-3 preflight: what does Humming's ``grouped_contiguous`` MoE GEMM compile to?

Context. The paired CUTLASS-vs-Humming serving benchmark
(``M3_PAIRED_CUTLASS_HUMMING_PERF_REPORT.md``) measured Humming *indexed* only.
It won decode by 13-26% but lost TTFT under load (up to +31%). Reading
``humming/tune/sm90.py`` gives a named mechanism for that shape::

    if gemm_type != GemmType.INDEXED:
        config["use_warp_spec"] = True
        config["use_tma"] = True
        config["use_mbarrier"] = True

Indexed is *structurally denied* TMA and warp specialization, because it gathers
its A rows through ``sorted_ids`` and therefore has no contiguous tile to hand a
TMA descriptor. So the arm we benchmarked runs a cp.async, non-warp-specialized
kernel on Hopper -- while ``grouped_contiguous`` would compile the canonical
TMA + producer/consumer pipeline.

This probe answers, with **no serving run and no 8-GPU allocation**, three
questions that decide whether arm 3 is worth cluster time:

1. Does our reconstructed layer meta exactly reproduce the kernels the qualified
   run actually compiled?  (Guard against reasoning about a fictional config.)
2. What does the resolved kernel config differ by, indexed vs grouped?
3. Does grouped actually *compile and load* for every shape_m bucket -- i.e. does
   warp-spec + TMA + 4 stages fit SM90's 227 KB shared memory?  ``Sm90Heuristics``
   overrides the base ``get_config`` and performs **no** smem check, so an
   oversized config would fail at cubin load with no Python guard.

Question 1 is answered by cache-key identity: point ``HUMMING_CACHE_DIR`` at a
*copy* of the qualified JIT cache and confirm indexed adds zero new entries.
A byte-identical generated source hashes to an existing directory; any deviation
in the meta writes a new one.

What this probe does NOT establish: any performance claim. A kernel that
compiles with TMA and warp specialization is not thereby faster. Only a paired
serving measurement can say that.

Usage (under a 1-GPU srun; cubin load needs a CUDA context)::

    python -m pipeline.m3_humming_gemm_type_probe --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

# The two routed-expert GEMMs of MiniMax-M3 under TP8 + expert parallelism.
# hidden_size=6144, moe intermediate=3072, 128 global experts / 8 ranks = 16
# local experts holding the full intermediate (EP shards experts, not columns).
#
# w13 fuses gate+up, so shape_n = 2 * 3072; w2 projects back to hidden.
HIDDEN_SIZE = 6144
MOE_INTERMEDIATE = 3072
LOCAL_EXPERTS = 16

# Field values below are transcribed from the #defines of a kernel the qualified
# r3 run actually compiled (cache-m3-gptq-w4a8-v1), not guessed:
#   NUM_EXPERTS 16   INPUT_SCALE_GROUP_SIZE 0   WEIGHT_SCALE_GROUP_SIZE 128
#   HAS_ZERO_POINT 0 IS_GROUP_WEIGHT_SCALE 1    HAS_INPUT_SCALE 1
#   MmaType::WGMMA   WeightScaleType::GROUP     PAD_SHAPE_{N,K} 0
_BASE_META: dict[str, Any] = {
    "num_experts": LOCAL_EXPERTS,
    # Unsigned: humming's check_dtype asserts ``not b_dtype.is_signed`` for the
    # packed-weight path (compressed-tensors W4 arrives offset-encoded).
    "b_dtype": "uint4",
    "a_dtype": "float8e4m3",
    "c_dtype": "bfloat16",
    "input_scale_group_size": 0,
    "weight_scale_group_size": 128,
    "weight_scale_group_size_n": 0,
    "use_int_weight_scale": False,
    "use_fused_e8m0_scale": False,
    "has_zero_point": False,
    "is_fp_zero_point": False,
    "has_bias": False,
}

SUBLAYERS: dict[str, dict[str, Any]] = {
    "w13": {**_BASE_META, "shape_n": 2 * MOE_INTERMEDIATE, "shape_k": HIDDEN_SIZE},
    "w2": {**_BASE_META, "shape_n": HIDDEN_SIZE, "shape_k": MOE_INTERMEDIATE},
}

# Matches the perf run's Humming policy (VLLM_HUMMING_USE_F16_ACCUM=0).
COMPUTE_BASE: dict[str, Any] = {"use_f16_accum": False, "use_batch_invariant": False}

GEMM_TYPES = ("indexed", "grouped_contiguous")

# Config keys worth diffing across gemm types. Everything else is either shape
# bookkeeping or derived.
_DIFF_KEYS = (
    "block_shape",
    "warp_shape",
    "num_stages",
    "num_ctas_per_sm",
    "use_stream_k",
    "use_warp_spec",
    "use_tma",
    "use_tma_a",
    "use_tma_as",
    "use_tma_b",
    "use_tma_c",
    "use_tma_bs",
    "use_mbarrier",
    "use_cp_async",
    "multi_cast_size_a",
    "num_threads",
    "num_math_threads",
    "num_load_threads",
)


def heuristics_buckets(meta_kwargs: dict[str, Any], gemm_type: str) -> list[Any]:
    """The (min_shape_m, max_shape_m, config) buckets vLLM would hand Humming.

    Mirrors ``HummingMethod.get_default_tuning_configs``, which calls
    ``get_heuristics_config`` with no ``shape_m`` and -- note -- leaves
    ``use_m_major_input_scale`` at its default of False. That default is why
    ``use_tma_as`` can never be enabled on the MoE path: the only writer,
    ``_apply_m_major_input_scale``, requires GemmType.DENSE anyway.
    """
    from humming.layer import HummingLayerMeta
    from humming.tune import get_heuristics_config

    meta = HummingLayerMeta(**meta_kwargs)
    return get_heuristics_config(meta=meta, gemm_type=gemm_type, **COMPUTE_BASE)


def resolved_kernel_config(
    meta_kwargs: dict[str, Any], gemm_type: str, bucket: Any
) -> dict[str, Any]:
    """Run a bucket's config through HummingKernel's own post-init resolution.

    The heuristics dict is not the final word: ``ComputeConfig.__post_init__``
    defaults ``use_tma_as`` to False *before* propagating ``use_tma`` to the
    other ``use_tma_*`` flags, and ``HummingKernel.__post_init__`` forces
    ``use_m_major_input_scale`` on for channel-wise input scales. Construct the
    real object so the reported flags are the ones the kernel is compiled with.
    """
    from humming.jit.runtime import KernelRuntime
    from humming.kernel.humming import HummingKernel

    _, _, tuning = bucket
    tuning = dict(tuning)
    tuning.pop("num_sms", None)

    # HummingKernel.__post_init__ ends by calling KernelRuntime.__post_init__,
    # which generates and NVRTC-compiles the kernel. We want only the flag
    # resolution above that call, so stub the runtime half out rather than
    # reimplementing the override logic (which is the very thing under test).
    original = KernelRuntime.__post_init__
    KernelRuntime.__post_init__ = lambda self: None  # type: ignore[method-assign]
    try:
        kernel = HummingKernel(
            **{**meta_kwargs, **COMPUTE_BASE, "gemm_type": gemm_type, **tuning}
        )
    finally:
        KernelRuntime.__post_init__ = original  # type: ignore[method-assign]

    out: dict[str, Any] = {}
    for key in _DIFF_KEYS:
        if hasattr(kernel, key):
            value = getattr(kernel, key)
            out[key] = list(value) if isinstance(value, tuple) else value
    return out


def _set_cache_dir(path: Path) -> None:
    """Point Humming's JIT cache at ``path``.

    ``humming.utils.jit.get_humming_cache_dir`` is ``@lru_cache(maxsize=1)``, so
    setting the environment variable alone silently keeps writing to whichever
    directory was resolved first. The cache must be invalidated too.
    """
    from humming.utils import jit as humming_jit

    os.environ["HUMMING_CACHE_DIR"] = str(path)
    humming_jit.get_humming_cache_dir.cache_clear()
    resolved = humming_jit.get_humming_cache_dir()
    assert Path(resolved) == path, (
        f"cache dir did not take effect: {resolved} != {path}"
    )


def _count_entries(cache_dir: Path) -> set[str]:
    if not cache_dir.is_dir():
        return set()
    return {p.name for p in cache_dir.iterdir() if p.is_dir() and p.name != "launcher"}


def compile_all(
    meta_kwargs: dict[str, Any], gemm_type: str, sublayer: str
) -> dict[str, Any]:
    """Compile + load every bucket via the real vLLM entry point.

    ``prepare_kernels`` is what ``HummingMethod.forward_layer`` reaches; calling
    it here exercises NVRTC codegen *and* ``load_cubin()``, so a shared-memory
    overflow surfaces as a load failure rather than silently passing a
    source-only check.
    """
    from humming.kernel.humming import HummingKernel

    cache_dir = Path(os.environ["HUMMING_CACHE_DIR"])
    before = _count_entries(cache_dir)
    started = time.monotonic()
    error: str | None = None
    try:
        HummingKernel.prepare_kernels(
            layer_config=json.dumps(meta_kwargs),
            compute_config=json.dumps({**COMPUTE_BASE, "gemm_type": gemm_type}),
        )
    except Exception as exc:  # noqa: BLE001 - the failure mode is the result
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    after = _count_entries(cache_dir)

    return {
        "sublayer": sublayer,
        "gemm_type": gemm_type,
        "ok": error is None,
        "error": error,
        "seconds": round(elapsed, 1),
        "cache_entries_before": len(before),
        "cache_entries_new": len(after - before),
        "new_entry_names": sorted(after - before)[:4],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory for probe artifacts")
    parser.add_argument(
        "--qualified-cache",
        default="/mnt/nfs/hoangduy/.humming/cache-m3-gptq-w4a8-v1",
        help="JIT cache the qualified indexed run populated (copied, never written)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="config resolution only; no NVRTC and no CUDA context",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"sublayers": {}, "compile": [], "identity": {}}

    # --- config resolution (no GPU, no NVRTC) ------------------------------
    for sublayer, meta_kwargs in SUBLAYERS.items():
        entry: dict[str, Any] = {"meta": meta_kwargs, "gemm_types": {}}
        for gemm_type in GEMM_TYPES:
            buckets = heuristics_buckets(meta_kwargs, gemm_type)
            block_m = [b[2]["block_shape"][0] for b in buckets]
            entry["gemm_types"][gemm_type] = {
                "num_buckets": len(buckets),
                "block_shape_m": block_m,
                # Representative buckets: smallest (decode-ish), a mid tile, and
                # the largest (prefill-ish). Flags are bucket-invariant here but
                # reporting three guards against assuming that.
                "resolved": {
                    str(buckets[i][2]["block_shape"][0]): resolved_kernel_config(
                        meta_kwargs, gemm_type, buckets[i]
                    )
                    for i in (0, len(buckets) // 2, len(buckets) - 1)
                },
            }
        report["sublayers"][sublayer] = entry

    if not args.skip_compile:
        # ``prepare_kernels`` calls torch.cuda.current_device() and loads cubins
        # with cuModuleLoad in the main thread. Without a context established
        # here first, that fails with CUDA_ERROR_INVALID_CONTEXT.
        import torch

        if not torch.cuda.is_available():
            print("ERROR: --out requires a GPU for cubin load; use --skip-compile")
            return 2
        torch.cuda.set_device(0)
        torch.zeros(1, device="cuda").sum().item()
        report["device"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }

        # --- identity check: indexed must be a pure cache hit --------------
        # Copy so a wrong meta cannot pollute the qualified cache.
        probe_indexed = out / "cache-probe-indexed"
        if probe_indexed.exists():
            shutil.rmtree(probe_indexed)
        shutil.copytree(args.qualified_cache, probe_indexed)
        _set_cache_dir(probe_indexed)
        for sublayer, meta_kwargs in SUBLAYERS.items():
            report["compile"].append(compile_all(meta_kwargs, "indexed", sublayer))

        indexed_rows = [r for r in report["compile"] if r["gemm_type"] == "indexed"]
        new_total = sum(r["cache_entries_new"] for r in indexed_rows)
        # "Zero new entries" only proves cache identity if the run actually got
        # far enough to compile. A crash before codegen also writes nothing, so
        # require every indexed row to have succeeded.
        all_ok = all(r["ok"] for r in indexed_rows)
        report["identity"] = {
            "indexed_new_cache_entries": new_total,
            "indexed_all_ok": all_ok,
            "meta_reproduces_qualified_kernels": all_ok and new_total == 0,
            "note": (
                "valid only when indexed_all_ok is true; a pre-codegen failure "
                "also produces zero new entries"
            ),
        }

        # --- the actual question: does grouped compile and load? ----------
        # Start from the qualified cache's prebuilt launcher extension so the
        # probe measures kernel compilation, not a torch cpp_extension rebuild.
        # Grouped configs hash to fresh directories, so nothing collides.
        probe_grouped = out / "cache-probe-grouped"
        if probe_grouped.exists():
            shutil.rmtree(probe_grouped)
        probe_grouped.mkdir(parents=True, exist_ok=True)
        launcher = Path(args.qualified_cache) / "launcher"
        if launcher.is_dir():
            shutil.copytree(launcher, probe_grouped / "launcher")
        _set_cache_dir(probe_grouped)
        for sublayer, meta_kwargs in SUBLAYERS.items():
            report["compile"].append(
                compile_all(meta_kwargs, "grouped_contiguous", sublayer)
            )

    (out / "probe-report.json").write_text(json.dumps(report, indent=2, default=str))

    # --- human-readable summary ------------------------------------------
    print("=== resolved kernel config: indexed vs grouped_contiguous ===")
    for sublayer, entry in report["sublayers"].items():
        idx = entry["gemm_types"]["indexed"]
        grp = entry["gemm_types"]["grouped_contiguous"]
        meta = entry["meta"]
        print(f"\n[{sublayer}] shape_n={meta['shape_n']} shape_k={meta['shape_k']}")
        print(f"  buckets: indexed={idx['num_buckets']} grouped={grp['num_buckets']}")
        print(f"  block_shape_m: {idx['block_shape_m'][0]}..{idx['block_shape_m'][-1]}")
        for block_m, icfg in idx["resolved"].items():
            gcfg = grp["resolved"][block_m]
            diffs = {
                k: (icfg.get(k), gcfg.get(k))
                for k in _DIFF_KEYS
                if icfg.get(k) != gcfg.get(k)
            }
            print(
                f"  block_m={block_m}: "
                + ", ".join(f"{k} {a}->{b}" for k, (a, b) in diffs.items())
            )

    if report["compile"]:
        print("\n=== compile + cubin load ===")
        for row in report["compile"]:
            status = "OK " if row["ok"] else "FAIL"
            print(
                f"  {status} {row['sublayer']:>4} {row['gemm_type']:<19} "
                f"{row['seconds']:>6.1f}s new_cache_entries={row['cache_entries_new']}"
                + (f"  {row['error']}" if row["error"] else "")
            )
        ident = report["identity"]
        if not ident["indexed_all_ok"]:
            verdict = "INCONCLUSIVE (indexed run failed before codegen)"
        else:
            verdict = "YES" if ident["meta_reproduces_qualified_kernels"] else "NO"
        print(
            f"\n  meta reproduces qualified indexed kernels: {verdict} "
            f"({ident['indexed_new_cache_entries']} new entries; 0 expected)"
        )

    print(f"\nreport: {out / 'probe-report.json'}")
    failed = [r for r in report["compile"] if not r["ok"]]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
