"""Phase-1 premise/risk benchmark for the MiniMax-M3 expert-scatter speed-up.

This does NOT quantize a real model. It answers the two questions that gate the
whole `M3_QUANT_SPEEDUP_PLAN` before any production scatter code is written:

  1. PREMISE: does per-expert GPTQ-shaped work underutilize a single H100?
     (watch `nvidia-smi dmon -s u` during the serial phase.)
  2. RISK (plan 6): which dispatch mechanism actually parallelizes per-expert
     quantization across cuda:0..N? Measures three, apples-to-apples:
       - serial    : all experts on cuda:0 (today's behavior)
       - threads   : one thread per GPU, one shared interpreter
       - processes : one process per GPU, own interpreter (no shared GIL)

The workload is a faithful *proxy* of llm-compressor's `quantize_weight`: the
[in,in] Hessian Cholesky-inverse plus the blocked column-update loop that
dominates GPTQ cost -- including the per-column Python loop that is exactly the
GIL concern.

IMPORTANT (schema v2): timing covers ONLY the GPU-resident quant compute.
Synthetic weight/Hessian generation is done BEFORE the timed region, because in
the real pipeline weights are already on-GPU (placed by accelerate) and the
Hessian is already accumulated during the calibration forward -- so timing data
fabrication (as schema v1 did) understated parallelism and is fixed here. Real
bit-parity of the production path is a separate Phase-2 gate, not this script.

Run (executor, on a freed 8-GPU node):
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 \
        python -m pipeline.bench_expert_scatter --experts 128 --mode all
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


def _make_resident_payload(device: torch.device, dtype: torch.dtype, seed: int) -> dict:
    """One expert's weights + Hessians, already on `device`. Stands in for data
    the real pipeline has ALREADY produced by quant time (weights placed by
    accelerate, Hessian accumulated during the calibration forward). Generating
    it is therefore NOT part of the work we time -- it is done before timing."""
    g = torch.Generator().manual_seed(seed)
    gate_up = torch.randn(2 * INTER, HIDDEN, generator=g, dtype=torch.float32).to(device, dtype)
    down = torch.randn(HIDDEN, INTER, generator=g, dtype=torch.float32).to(device, dtype)
    xg = torch.randn(2048, HIDDEN, generator=g, dtype=torch.float32).to(device)
    xd = torch.randn(2048, INTER, generator=g, dtype=torch.float32).to(device)
    h_gate, h_down = xg.T @ xg, xd.T @ xd
    del xg, xd
    return {"gate_up": gate_up, "down": down, "h_gate": h_gate, "h_down": h_down, "device": device}


def _quantize_resident(payload: dict, blocksize: int) -> None:
    """Quantize one expert's two projections from GPU-resident tensors. This is
    the ONLY thing the benchmark times (gptq_like_quantize clones its inputs, so
    the payload is reusable across calls)."""
    dev = payload["device"]
    gptq_like_quantize(payload["gate_up"], payload["h_gate"], blocksize)
    gptq_like_quantize(payload["down"], payload["h_down"], blocksize)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


# ----- process-per-GPU mode (own interpreter per rank -> no shared GIL) -----
# One OS process per GPU, launched explicitly (NOT via ProcessPoolExecutor, which
# lazily reuses a single worker for fast tasks and would leave most GPUs idle --
# a measurement artifact). This is also the real torchrun model: rank r pins
# cuda:r, holds its expert shard, and quantizes it independently.


def _proc_worker(rank: int, n_local: int, blocksize: int, dtype_name: str, result_q) -> None:
    import torch as _t

    device = _t.device(f"cuda:{rank}")
    _t.cuda.set_device(device)
    dtype = getattr(_t, dtype_name)
    payload = _make_resident_payload(device, dtype, seed=rank)  # pre-timed data
    _quantize_resident(payload, blocksize)  # warm up kernels/JIT
    _t.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_local):
        _quantize_resident(payload, blocksize)
    _t.cuda.synchronize(device)
    result_q.put((rank, time.perf_counter() - t0))


def run_benchmark(
    n_experts: int, blocksize: int, dtype: torch.dtype, out_path: Path, mode: str
) -> dict:
    n_dev = torch.cuda.device_count()
    if n_dev == 0:
        raise SystemExit("no CUDA devices; use --self-test for a CPU logic check")
    results: dict = {
        "schema_version": 2,
        "status": "running",
        "node": platform.node(),
        "devices": n_dev,
        "device_names": [torch.cuda.get_device_name(d) for d in range(n_dev)],
        "experts": n_experts,
        "dtype": str(dtype),
        "blocksize": blocksize,
        "mode": mode,
        "dims": {"hidden": HIDDEN, "inter": INTER},
        "torch_version": torch.__version__,
        "note": "times ONLY GPU-resident quant compute; synthetic data-gen is "
                "excluded (unlike schema v1), matching the real pipeline where "
                "weights/Hessians already exist at quant time",
    }
    _write_json(out_path, results)
    _log(f"devices={n_dev}  experts={n_experts}  dtype={dtype}  mode={mode}")
    _log(f"per-expert: gate_up[{2*INTER},{HIDDEN}] + down[{HIDDEN},{INTER}]  "
         f"(compute-only timing)\n")
    want = {"serial", "threads", "processes"} if mode == "all" else {mode}

    try:
        # === SERIAL: all experts on cuda:0, data pre-resident (current behavior). ===
        serial_s = None
        if "serial" in want:
            payloads = [_make_resident_payload(torch.device("cuda:0"), dtype, s) for s in range(n_experts)]
            torch.cuda.synchronize(0)
            t0 = time.perf_counter()
            for pl in payloads:
                _quantize_resident(pl, blocksize)
            serial_s = time.perf_counter() - t0
            del payloads
            torch.cuda.empty_cache()
            results["serial"] = {"compute_s": serial_s}
            _write_json(out_path, results)
            _log(f"[serial   1 GPU] {serial_s:7.2f}s  (compute only)")

        # === THREADS: experts pre-placed across GPUs, quantized via a thread pool. ===
        if "threads" in want:
            placed = [_make_resident_payload(torch.device(f"cuda:{i % n_dev}"), dtype, s)
                      for i, s in enumerate(range(n_experts))]
            for d in range(n_dev):
                torch.cuda.synchronize(d)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n_dev) as pool:
                list(pool.map(lambda pl: _quantize_resident(pl, blocksize), placed))
            threads_s = time.perf_counter() - t0
            del placed
            torch.cuda.empty_cache()
            sp = (serial_s / threads_s) if serial_s else None
            results["threads"] = {"compute_s": threads_s, "speedup_vs_serial": sp}
            _write_json(out_path, results)
            _log(f"[threads {n_dev} GPU] {threads_s:7.2f}s"
                 + (f"  speedup {sp:.2f}x" if sp else ""))

        # === PROCESSES: one worker per GPU, own interpreter -> no shared GIL. ===
        if "processes" in want:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            result_q = ctx.Queue()
            dtype_name = str(dtype).split(".")[-1]
            # split n_experts across ranks (remainder to the low ranks)
            base, extra = divmod(n_experts, n_dev)
            counts = [base + (1 if r < extra else 0) for r in range(n_dev)]
            t0 = time.perf_counter()
            procs = [
                ctx.Process(target=_proc_worker,
                            args=(r, counts[r], blocksize, dtype_name, result_q))
                for r in range(n_dev) if counts[r] > 0
            ]
            for p in procs:
                p.start()
            per_rank = [result_q.get() for _ in procs]  # collect before join
            for p in procs:
                p.join()
            processes_wall_s = time.perf_counter() - t0
            # steady-state parallel compute = slowest rank's loop (spawn/CUDA-init
            # is a one-time cost in production, not paid per layer)
            max_rank_compute_s = max(t for _, t in per_rank)
            sp_wall = (serial_s / processes_wall_s) if serial_s else None
            sp_compute = (serial_s / max_rank_compute_s) if serial_s else None
            results["processes"] = {
                "wall_s": processes_wall_s,
                "max_rank_compute_s": max_rank_compute_s,
                "per_rank_compute_s": {r: t for r, t in sorted(per_rank)},
                "speedup_wall_vs_serial": sp_wall,
                "speedup_compute_vs_serial": sp_compute,
                "note": "wall includes one-time spawn + CUDA init; "
                        "speedup_compute (vs slowest rank's loop) is the "
                        "steady-state per-layer parallelism the production run sees",
            }
            _write_json(out_path, results)
            _log(f"[procs   {n_dev} GPU] wall {processes_wall_s:7.2f}s  "
                 f"compute {max_rank_compute_s:7.2f}s"
                 + (f"  speedup(compute) {sp_compute:.2f}x" if sp_compute else ""))

        # === Verdict from the best real-parallel speedup we measured. ===
        # processes: use steady-state compute speedup (spawn is one-time in prod).
        candidates = [
            results.get("threads", {}).get("speedup_vs_serial") or 0.0,
            results.get("processes", {}).get("speedup_compute_vs_serial") or 0.0,
        ]
        best_sp = max(candidates, default=0.0)
        results["best_speedup_vs_serial"] = best_sp
        results["ceiling"] = n_dev
        results["real_parallelism"] = best_sp >= 0.5 * n_dev
        results["verdict"] = (
            f"best {best_sp:.2f}x of {n_dev}x ceiling -> viable; proceed to Phase 2 wiring"
            if best_sp >= 0.5 * n_dev
            else f"best {best_sp:.2f}x of {n_dev}x ceiling -> insufficient; "
            "reconsider mechanism before wiring"
        )
        results["status"] = "ok"
        _write_json(out_path, results)
        _log(f"\nVERDICT: {results['verdict']}")
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
        "--mode", choices=["all", "serial", "threads", "processes"], default="all",
        help="which dispatch modes to measure (default: all three)",
    )
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
    results = run_benchmark(args.experts, args.blocksize, dtype, args.out, args.mode)
    # Also emit the machine-readable summary to stdout as a fallback if the file
    # is on a path the caller can't reach.
    _log("\n=== RESULTS JSON ===")
    _log(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
