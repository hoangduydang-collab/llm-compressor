# Serving venv `serve-026` — vLLM 0.26.0 with humming merged in

> ## STATUS 2026-07-28: **NOT QUALIFIED — do not publish numbers from this venv**
>
> First cluster window on `serve-026` (gpu-h123, job 13422) hit **two independent
> blockers**. Both are 0.26.0-side; neither is a DSpark config error and neither is a
> humming-merge error (patched humming sources are byte-identical to the validated
> `humming-0.1.10-site`, and the JIT cache key includes per-file `.cuh` mtimes, so the
> patched headers are genuinely compiled).
>
> 1. ~~**k=0 baseline is unstable.**~~ **RESOLVED 2026-07-28 — root-caused and fixed.**
>    Plain M3 W4A8 + Humming, `speculative_config=None`, Model Runner **V1**: CUDA
>    *illegal memory access* on any mixed prefill+decode batch. Root cause is an
>    **upstream 0.26.0 regression**, not our overlay: `nvidia/model.py` allocates the
>    shared `topk_indices_buffer` token-major `[T, H, K]` for the rewritten SM100 MSA
>    top-k, while the Triton indexer and Triton attend — the impls selected on every
>    non-SM100 GPU — still slice it head-major. At TP8 `num_index_heads == 1`, so
>    `buf[:, nd:, :]` with `nd >= 1` drops the head axis and yields an empty view with a
>    shifted base pointer. Full analysis and the fix:
>    **`m3-026-topk-buffer-layout.md`**. Verified 20/20 requests on the reproducer that
>    previously gave 0/10 three times.
>
>    The hypothesis recorded here originally — our `breakable_cudagraph.py` overlay edit
>    losing its effect — was **wrong**: the enforce-eager bisect arm crashed too, so
>    cudagraphs were never involved. So were two later hypotheses (the new cuteDSL
>    `ll_bf16` router GEMM; the packed KV layout / in-kernel fp8 dequant). See the
>    falsified list in that doc.
>
>    Note the fix currently lives only in the venv plus a standalone fixer script — it is
>    **not yet a `patch_vllm_m3_serve.py` target**, so rebuilding `serve-026` loses it.
> 2. **DSpark cannot run on M3 at all** — see `m3-dspark-blockers-026.md`.
>
> The qualification decision is **still no, but for one remaining reason instead of two.**
> `QUALIFIED_VLLM_VERSIONS` stays `("0.24.0",)`; `LLMC_HUMMING_PROVISIONAL_VLLM=0.26.0`
> remains the only way in, and it stamps `VLLM_VERSION_PROVISIONAL` into every
> attestation — which it did correctly here.
>
> **What qualification now waits on.** With blocker 1 fixed, 0.26.0 can host
> measurements again, so the citation this venv needs is finally obtainable: the
> `D-k0-a` / `D-k0-b` Humming k=0 controls over the identical staged prompts that the
> h114 window measured on 0.24.0 the same day (136.8 tok/s conc 1; 75.2 8k-low / 80.3
> 8k-high at conc 10). Agreement there is the same-workload, same-day, runtime-only
> comparison that qualifies the W4A8 path; divergence is a finding in its own right.
> Re-run in flight 2026-07-28: `results/m3-specdec-dspark/20260728T094710Z-k-sweep`
> (gpu-h105, job 13428), with `D-k0-a` promoted to smoke serve so the citation is banked
> before any DSpark leg can abort the window. **Do not flip
> `QUALIFIED_VLLM_VERSIONS` until those controls are in and agree** — blocker 1 being
> fixed is necessary, not sufficient.

**Purpose.** Intended as the go-forward M3 serving environment, replacing the
`quant` venv + `PYTHONPATH` humming side-install with a single self-contained venv.
It is the *only* environment in which DSpark speculative decoding can run at all
(0.24.0 lacks the method) — but see the status block above.

Built 2026-07-28. Path: `/mnt/nfs/hoangduy/venvs/serve-026`.

## Why 0.26.0

DSpark spec-dec (`nvidia/MiniMax-M3-DSpark`) requires a vLLM that has the method.
Verified by source, not by version string:

| vLLM | `DSparkModelTypes` in `config/speculative.py` |
|---|---|
| 0.23.x | absent |
| **0.24.0** (our `quant` venv — the actual serving venv) | **absent** — verified in the installed venv, not just upstream |
| **0.25.0** | **present** — first release with DSpark |
| 0.26.0 | present (latest stable; 0.26.1 does not exist) |

Upstream layout: `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` +
`vllm/model_executor/models/qwen3_dspark.py`. Both live in the post-0.23 worker
tree, so this is a genuine two-minor-version move, not a cherry-pick.

**Which venv actually serves.** `run_vllm_http_serve_smoke.sh` sources its venv, and
the default is **`quant`** (vLLM 0.24.0, `humming_kernels` 0.1.6 installed and shadowed
to 0.1.10 at serve time by `PYTHONPATH`). That is the environment every published M3
serving number came from — not the `serve` venv, which holds an unrelated 0.23.1 build.
The move is therefore **0.24.0 → 0.26.0**.

**The de-risk that made it cheap:** vLLM 0.26.0 pins `torch==2.11.0`, exactly what
`quant` already runs (`2.11.0+cu130`, CUDA 13.0). PyPI resolved the same `+cu130`
build. No torch bump, no CUDA rebuild.

## Contents

| | |
|---|---|
| Python | 3.12.13 (`/mnt/nfs/hoangduy/python/cpython-3.12-linux-x86_64-gnu`) |
| Builder | `/mnt/nfs/hoangduy/uv/uv` 0.11.8 |
| vLLM | 0.26.0 (PyPI wheel, `cp38-abi3-manylinux_2_28_x86_64`) |
| torch | 2.11.0+cu130 / CUDA 13.0 |
| triton | 3.6.0 |
| transformers | 5.14.1 |
| flashinfer | 0.6.14 |
| **humming_kernels** | **0.1.10, installed in-venv** |

Install log: `/mnt/nfs/hoangduy/venvs/serve-026-install.log`

## The humming merge

`humming_kernels` is a pure `py3-none-any` wheel (CUDA sources are JIT-compiled at
runtime), so it needs no side-install to carry a patched copy. 0.1.10 is now a normal
dependency of this venv.

Select it with **`SERVE_VENV=/mnt/nfs/hoangduy/venvs/serve-026`**, the override added to
`run_vllm_http_serve_smoke.sh` (default stays `quant`, so existing launchers are
unaffected).

**Launchers using `serve-026` must NOT prepend a humming site dir to `PYTHONPATH`.**
The old form was `PYTHONPATH=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site:$REPO`,
which shadowed the venv's own 0.1.6. Now it is just `PYTHONPATH=$REPO`.

Equivalence was proven, not assumed: after re-applying the four humming patches to
the venv copy, `diff -rq --exclude=__pycache__` against the validated
`humming-0.1.10-site` tree reports **no differences**, and all four patch digests
match:

| patch | file | sha256 (first 16) |
|---|---|---|
| `ct_input_format` | `humming/schema/compressed_tensors.py` | `8e2ab300b595e98f` |
| `grouped_expert_bounds` | `humming/include/humming/scheduler.cuh` | `befa01f9758df24e` |
| `tma_store_fence` | `humming/include/humming/utils/ptx/tma.cuh` | `2ad7d5339d730d4a` |
| `tma_store_commit` | `humming/include/humming/epilogue/gmem_writer.cuh` | `3e135b55f3753245` |

## Rebuilding from scratch

```sh
export UV_CACHE_DIR=/mnt/nfs/hoangduy/.cache/uv
PY312=/mnt/nfs/hoangduy/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
/mnt/nfs/hoangduy/uv/uv venv --python "$PY312" /mnt/nfs/hoangduy/venvs/serve-026
/mnt/nfs/hoangduy/uv/uv pip install --python /mnt/nfs/hoangduy/venvs/serve-026/bin/python \
    "vllm==0.26.0" "humming_kernels==0.1.10"

PY=/mnt/nfs/hoangduy/venvs/serve-026/bin/python
SITE=/mnt/nfs/hoangduy/venvs/serve-026/lib/python3.12/site-packages

# vLLM overlays — these resolve vLLM through the running interpreter, so they MUST
# be invoked with the serve-026 python or they will patch the wrong venv.
"$PY" pipeline/slurm/patch_vllm_m3_serve.py          # M3 W4A8 serving overlay (7 edits)
"$PY" pipeline/slurm/patch_vllm_humming_lmhead.py    # ParallelLMHead fix
"$PY" pipeline/slurm/patch_vllm_eagle3_lmhead_pad.py # vocab-pad lever

# humming patches — default --site is the OLD side-install, so pass --site explicitly.
for p in ct_input_format grouped_expert_bounds tma_store_fence tma_store_commit; do
  PYTHONPATH=. "$PY" "pipeline/slurm/patch_humming_$p.py" --site "$SITE"
done
```

Every one of these is idempotent and supports `--check` (non-zero exit if unpatched);
launcher pre-flight gates should apply-then-recheck so a fresh venv is handled but a
failed apply still fails closed.

All seven applied cleanly on 0.26.0 — the source targets survived two minor releases.
Notably `humming_utils.prepare_humming_layer` still carries the unguarded
`has_bias=layer.has_bias` read at 0.26.0, so the `ParallelLMHead` fix is still ours to
maintain (upstream has not taken it).

## Two behaviour changes to know about

### 1. Humming was demoted to last in the CUDA kernel priority

| | order |
|---|---|
| 0.24.0 (`quant`, our baseline) | Cutlass-W4A8 > Machete > AllSpark > Marlin > **Humming** > Conch > Exllama > Triton-W4A16 |
| 0.26.0 | Cutlass-W4A8 > Machete > AllSpark > Marlin > Conch > Exllama > Triton-W4A16 > **Humming** |

This **silently breaks the EAGLE3 axis-2 cell definitions**, which reached Humming by
*subtraction*. Measured with `choose_mp_linear_kernel(..., compute_capability=90)` on
the drafter's unpadded `lm_head` config (W4A16 uint4b8, group 128, symmetric,
`[6144, 25008]` per rank, bf16 activations):

| `VLLM_DISABLED_KERNELS` | 0.24.0 selected | 0.26.0 selects |
|---|---|---|
| *(none)* | Marlin | Marlin — unchanged |
| `MarlinLinearKernel` | **Humming** | **TritonW4A16** — wrong cell |
| `MarlinLinearKernel,MacheteLinearKernel` | **Humming** | **TritonW4A16** — wrong cell |
| `MarlinLinearKernel,ConchLinearKernel,ExllamaLinearKernel,TritonW4A16LinearKernel` | — | **Humming** ✅ |

Both orders and all three selections above were measured in the installed venvs, so
the published EAGLE3 axis-2 cells were correct as run on 0.24.0; the demotion happened
between 0.24.0 and 0.26.0.

The fall-through lands on generic Triton rather than Conch because `conch-triton-kernels`
is not installed and Exllama accepts fp16 activations only. So the old two-cell recipe
does not merely pick a different good kernel — it silently substitutes the correctness
fallback, which would read as a large Humming regression.

Use the four-name disable list above. And note that a gate asserting only the *relative*
order `Machete > Marlin > Humming` passes regardless — it did here — so assert the
intended kernel **by name from the serve log**, never by inferring it from the disable
list.

### 1b. Kernel eligibility on H100 / bf16 (why most of the registry is untestable)

`can_implement` verdicts for the same drafter `lm_head` config:

| kernel | verdict | reason |
|---|---|---|
| Cutlass-W4A8 | no | FP8 (e4m3) activations only |
| Machete | no (yes if padded) | out-features must be divisible by 128; 25008 % 128 == 48 |
| AllSpark | **no** | "does not support device_capability = 90" — never available on H100 |
| Marlin | yes | the default |
| Conch | no *(unlockable)* | `conch-triton-kernels` not installed — a pip install, not a node |
| Exllama | **no** | float16 activations only; we serve bf16 |
| Triton-W4A16 | yes | generic fallback / correctness floor |
| Humming | yes | |

AllSpark and Exllama are structurally out on this hardware/dtype, so no amount of
cluster time can turn them into cells. Conch is the only genuinely new candidate, and
`LLMC_EAGLE3_LMHEAD_PAD=1024` remains the only way to make Machete eligible.

### 1c. New in 0.26.0: cuteDSL low-latency BF16 gate GEMM

`kernels/linear/cute_dsl/ll_bf16.py` is new, and `GateLinear` (the MoE router gate) now
dispatches to it as **tier 1 on SM90+ when M ≤ 16, bf16 in, K % 8 == 0**. M3 has 60 MoE
layers, so this GEMM runs 60× per forward. It is automatic — no env flag.

It applies at conc 1 (M = 1), at conc 10 (M = 10), and to spec-dec verify at conc 1
(M = k+1 ≤ 16), but **not** to conc-10 verify at k=5 (M = 60 > 16). So it should help
exactly the low-concurrency regime and drop out under load.

This is not an axis to sweep — it is part of the 0.26.0 baseline, and one concrete
reason a 0.26.0 EAGLE3 control need not reproduce its 0.23.1 counterpart.

Also new but not useful here: `int4_emulation_moe.py` (dequantizes int4 → BF16 at load
and runs plain Triton BF16 experts — a compatibility fallback that discards the memory
benefit of W4), `rdna_hybrid_w4a16.py` (AMD), `trtllm_lora_moe.py` (LoRA).

### 2. `minimax_m3_mtp` exists as a method but we have no weights for it

vLLM 0.26.0 lists `minimax_m3_mtp` among its speculative methods, which would be a
free third drafter needing no external checkpoint. It is not usable here: neither
`MiniMaxAI/MiniMax-M3` (23 416 tensors) nor our in-house
`gptq-checkpoint-vllm-w123-abi-overlay` (67 192 tensors, layers 0–59) contains any
`mtp` or `eh_proj` weights. Revisit only if a checkpoint variant that ships the MTP
module appears.

## What stays frozen

Do **not** upgrade or re-point these:

- **`/mnt/nfs/hoangduy/venvs/quant`** (vLLM `0.24.0`, `humming_kernels` 0.1.6) — the
  actual serving venv and the provenance of every published EAGLE3 / two-axis /
  packed-K number, *and* the qualified quantization environment. Mutating it would make
  already-reported results unreproducible.
- **`/mnt/nfs/hoangduy/venvs/serve`** (vLLM `0.23.1rc1.dev643+gf41e8ddc9`) — an older
  build not used by the serving path; left alone.
- **`/mnt/nfs/hoangduy/venvs/humming-0.1.10-site`** and `humming-0.1.11-site` — the
  validated side-installs those runs loaded, and the reference trees the merge above is
  diffed against.

Cross-venv comparisons are therefore **not** within-window. Any claim that pits a
`serve-026` measurement against a `serve` measurement carries a runtime change on top of
whatever else varied, and must say so.
