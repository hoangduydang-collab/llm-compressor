# Serving venv `serve-026` — vLLM 0.26.0 with humming merged in

**Purpose.** The go-forward M3 serving environment. It replaces the
`serve` venv + `PYTHONPATH` humming side-install with a single self-contained venv,
and it is the *only* environment in which DSpark speculative decoding can run.

Built 2026-07-28. Path: `/mnt/nfs/hoangduy/venvs/serve-026`.

## Why 0.26.0

DSpark spec-dec (`nvidia/MiniMax-M3-DSpark`) requires a vLLM that has the method.
Verified by source, not by version string:

| vLLM | `DSparkModelTypes` in `config/speculative.py` |
|---|---|
| 0.23.x (our `serve` venv) | absent |
| 0.24.0, 0.24.1 | absent |
| **0.25.0** | **present** — first release with DSpark |
| 0.26.0 | present (latest stable; 0.26.1 does not exist) |

Upstream layout: `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` +
`vllm/model_executor/models/qwen3_dspark.py`. Both live in the post-0.23 worker
tree, so this is a genuine two-minor-version move, not a cherry-pick.

**The de-risk that made it cheap:** vLLM 0.26.0 pins `torch==2.11.0`, which is
exactly what the `serve` venv already runs (`2.11.0+cu130`, CUDA 13.0). PyPI
resolved the same `+cu130` build. No torch bump, no CUDA rebuild.

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
| 0.23.1 | Cutlass-W4A8 > Machete > AllSpark > Marlin > **Humming** > Conch > Exllama > Triton-W4A16 |
| 0.26.0 | Cutlass-W4A8 > Machete > AllSpark > Marlin > Conch > Exllama > Triton-W4A16 > **Humming** |

This **silently breaks the EAGLE3 axis-2 cell definitions**, which reached Humming by
*subtraction*:

| cell | 0.23.1 `VLLM_DISABLED_KERNELS` | what it now selects on 0.26.0 |
|---|---|---|
| Machete×8 + Humming | `MarlinLinearKernel` | Machete×8 + **Conch** — wrong |
| Humming×9 | `MarlinLinearKernel,MacheteLinearKernel` | **Conch**×9 — wrong |

On 0.26.0 those disable lists must also exclude `ConchLinearKernel`,
`ExllamaLinearKernel` and `TritonW4A16LinearKernel`. A gate that only asserts the
*relative* order `Machete > Marlin > Humming` passes anyway — it did here — so assert
the intended kernel by name from the serve log instead of inferring it from the
disable list.

### 2. `minimax_m3_mtp` exists as a method but we have no weights for it

vLLM 0.26.0 lists `minimax_m3_mtp` among its speculative methods, which would be a
free third drafter needing no external checkpoint. It is not usable here: neither
`MiniMaxAI/MiniMax-M3` (23 416 tensors) nor our in-house
`gptq-checkpoint-vllm-w123-abi-overlay` (67 192 tensors, layers 0–59) contains any
`mtp` or `eh_proj` weights. Revisit only if a checkpoint variant that ships the MTP
module appears.

## What stays frozen

Do **not** upgrade or re-point these:

- **`/mnt/nfs/hoangduy/venvs/serve`** (vLLM `0.23.1rc1.dev643+gf41e8ddc9`) — the
  provenance of every published EAGLE3 / two-axis / packed-K number. Mutating it would
  make already-reported results unreproducible.
- **`/mnt/nfs/hoangduy/venvs/humming-0.1.10-site`** and `humming-0.1.11-site` — the
  validated side-installs those runs loaded, and the reference trees the merge above is
  diffed against.
- **`/mnt/nfs/hoangduy/venvs/quant`** — stays on `humming_kernels` 0.1.6; the
  quantization path is qualified against that version.

Cross-venv comparisons are therefore **not** within-window. Any claim that pits a
`serve-026` measurement against a `serve` measurement carries a runtime change on top of
whatever else varied, and must say so.
