"""Phase-1 premise/risk benchmark for the MiniMax-M3 expert-scatter speed-up.

This does NOT quantize a real model. It answers the two questions that gate the
whole `M3_QUANT_SPEEDUP_PLAN` before any production scatter code is written:

  1. PREMISE: does per-expert GPTQ-shaped work underutilize a single H100?
     (watch `nvidia-smi dmon -s u` during the serial phase.)
  2. RISK (plan  6): does dispatching per-expert work across cuda:0..N in ONE
     process give real wall-clock parallelism, or does the Python-driven GPTQ
     column loop (GIL + kernel-launch overhead) eat the win?

The workload is a faithful *proxy* of llm-compressor's `quantize_weight`: the
[in,in] Hessian Cholesky-inverse plus the blocked column-update loop that
dominates GPTQ cost -- including the per-column Python loop that is exactly the
GIL concern. It is deliberately recipe-agnostic and needs no model download, so
it can run the moment the AWQ GPUs are freed. Real bit-parity is a separate
Phase-2 gate, not this script's job.

Run (executor, on a freed 8-GPU node):
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 \
        python -m pipeline.bench_expert_scatter --experts 128
Self-test the workload logic on CPU (planner, no GPU):
    python -m pipeline.bench_expert_scatter --self-test
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

DEFAULT_OUT = Path("results/m3-expert-scatter-bench/expert_scatter_bench.json")


def _log(msg: str = "") -> None:
    """Print to stdout with an immediate flush so srun log capture never loses
    lines to buffering when the job is killed or redirected."""
    print(msg, flush=True)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    _log(f"WROTE {path.resolve()}")

# MiniMax-M3 per-expert dims (configuration_minimax_m3_vl.py): hidden=6144,
# per-expert intermediate=3072. Each expert quantizes two Linears:
#   gate_up_proj: weight [2*inter, hidden] -> Hessian [hidden, hidden]
#   down_proj:    weight [hidden, inter]   -> Hessian [inter, inter]
HIDDEN = 6144
INTER = 3072


def gptq_like_quantize(
    weight: torch.Tensor, hessian: torch.Tensor, blocksize: int = 128, percdamp: float = 0.01
) -> torch.Tensor:
    """FLOP- and control-flow-faithful proxy of GPTQ quantize_weight.

    Mirrors the real inner loop: damp + Cholesky-inverse of H, then a blocked
    column loop with a per-column Python step (round + error propagation). The
    per-column Python loop is the serialization the concurrency test probes.
    """
    W = weight.to(torch.float32).clone()
    num_columns = W.shape[1]
    H = hessian.to(torch.float32).clone()
    damp = percdamp * torch.mean(torch.diag(H))
    idx = torch.arange(num_columns, device=W.device)
    H[idx, idx] += damp
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    for i1 in range(0, num_columns, blocksize):
        i2 = min(i1 + blocksize, num_columns)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.round(w)  # fake-quant stand-in (nearest int)
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return W


def _make_expert_task(device: torch.device, dtype: torch.dtype, blocksize: int):
    """Return a callable that builds one expert's weights+Hessians on `device`
    and runs the two-Linear quantize, timing pure-Python setup vs CUDA work."""

    def task(seed: int) -> dict:
        g = torch.Generator(device="cpu").manual_seed(seed)
        t_setup0 = time.perf_counter()
        gate_up = torch.randn(2 * INTER, HIDDEN, generator=g, dtype=torch.float32).to(device, dtype)
        down = torch.randn(HIDDEN, INTER, generator=g, dtype=torch.float32).to(device, dtype)
        # Hessian = X^T X from a calibration activation slab (PSD by construction).
        xg = torch.randn(2048, HIDDEN, generator=g, dtype=torch.float32).to(device)
        h_gate = xg.T @ xg
        xd = torch.randn(2048, INTER, generator=g, dtype=torch.float32).to(device)
        h_down = xd.T @ xd
        setup_s = time.perf_counter() - t_setup0

        t_k0 = time.perf_counter()
        gptq_like_quantize(gate_up, h_gate, blocksize)
        gptq_like_quantize(down, h_down, blocksize)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        kernel_s = time.perf_counter() - t_k0
        return {"setup_s": setup_s, "kernel_s": kernel_s}

    return task


def run_benchmark(n_experts: int, blocksize: int, dtype: torch.dtype, out_path: Path) -> dict:
    n_dev = torch.cuda.device_count()
    if n_dev == 0:
        raise SystemExit("no CUDA devices; use --self-test for a CPU logic check")
    results: dict = {
        "schema_version": 1,
        "status": "running",
        "node": platform.node(),
        "devices": n_dev,
        "device_names": [torch.cuda.get_device_name(d) for d in range(n_dev)],
        "experts": n_experts,
        "dtype": str(dtype),
        "blocksize": blocksize,
        "dims": {"hidden": HIDDEN, "inter": INTER},
        "torch_version": torch.__version__,
    }
    # Persist a partial record up front so even a crash mid-run leaves evidence.
    _write_json(out_path, results)
    _log(f"devices={n_dev}  experts={n_experts}  dtype={dtype}  blocksize={blocksize}")
    _log(f"per-expert: gate_up[{2*INTER},{HIDDEN}] + down[{HIDDEN},{INTER}]\n")

    try:
        # --- Serial baseline: every expert on cuda:0 (the current behavior). ---
        dev0 = torch.device("cuda:0")
        serial_task = _make_expert_task(dev0, dtype, blocksize)
        torch.cuda.synchronize(dev0)
        t0 = time.perf_counter()
        per_expert = [serial_task(seed) for seed in range(n_experts)]
        serial_s = time.perf_counter() - t0
        setup = sum(p["setup_s"] for p in per_expert)
        kernel = sum(p["kernel_s"] for p in per_expert)
        results["serial"] = {
            "total_s": serial_s,
            "python_setup_s": setup,
            "cuda_s": kernel,
            "python_fraction": setup / (setup + kernel),
        }
        _write_json(out_path, results)
        _log(f"[serial  cuda:0] {serial_s:7.2f}s total  "
             f"(python-setup {setup:6.2f}s / cuda {kernel:6.2f}s = "
             f"{100*setup/(setup+kernel):.0f}% python)")
        _log("  -> watch `nvidia-smi dmon -s u`: low SM% here confirms the premise\n")

        # --- Parallel: scatter experts round-robin across all GPUs via threads. ---
        tasks = {d: _make_expert_task(torch.device(f"cuda:{d}"), dtype, blocksize) for d in range(n_dev)}
        for d in range(n_dev):
            torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_dev) as pool:
            list(pool.map(lambda s: tasks[s % n_dev](s), range(n_experts)))
        parallel_s = time.perf_counter() - t0
        speedup = serial_s / parallel_s
        real = speedup >= 0.5 * n_dev
        verdict = (
            "real parallelism -> thread-pool expert-scatter is viable; proceed to Phase 2"
            if real
            else "sub-half-ceiling -> Python/launch overhead is eating the win; "
            "plan 6 fallback (per-device CUDA streams / ProcessPool) needed"
        )
        results.update({
            "scatter": {"total_s": parallel_s},
            "speedup": speedup,
            "ceiling": n_dev,
            "real_parallelism": real,
            "verdict": verdict,
            "status": "ok",
        })
        _write_json(out_path, results)
        _log(f"[scatter {n_dev} GPU] {parallel_s:7.2f}s total")
        _log(f"\nspeedup: {speedup:.2f}x  (ceiling {n_dev}x)")
        _log(f"VERDICT: {verdict}")
    except Exception as exc:  # persist the failure instead of vanishing
        results["status"] = "error"
        results["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(out_path, results)
        raise
    return results


def self_test() -> None:
    """Validate workload determinism/shape on CPU at tiny dims (no GPU needed)."""
    torch.manual_seed(0)
    W = torch.randn(32, 64)
    X = torch.randn(128, 64)
    H = X.T @ X
    out1 = gptq_like_quantize(W, H, blocksize=16)
    out2 = gptq_like_quantize(W, H, blocksize=16)
    assert out1.shape == W.shape, out1.shape
    assert torch.equal(out1, out2), "workload must be deterministic"
    assert torch.isfinite(out1).all(), "workload produced non-finite values"
    _log("self-test OK: workload is shape-preserving, deterministic, finite")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"JSON results path (default: {DEFAULT_OUT})",
    )
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    results = run_benchmark(args.experts, args.blocksize, dtype, args.out)
    # Also emit the machine-readable summary to stdout as a fallback if the file
    # is on a path the caller can't reach.
    _log("\n=== RESULTS JSON ===")
    _log(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
