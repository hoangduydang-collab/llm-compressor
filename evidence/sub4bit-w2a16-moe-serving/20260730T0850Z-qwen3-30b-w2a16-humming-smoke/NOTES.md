# Sub-4-bit (W2A16) compressed-tensors MoE serving smoke test — PASS

Date: 2026-07-30. Slurm job 13464 on gpu-h104 (1×H100), tmux `w2a16smoke`.

## What was proven

A 2-bit weight-only **compressed-tensors pack-quantized MoE** checkpoint serves
end-to-end on vLLM via Humming kernels — the exact export format our
AutoRound/llm-compressor pipeline produces. Coherent generations at temp 0
("42", "Paris.", fluent MoE definition); machine gate `SMOKE_RESULT: PASS`
(see `run.log`, full server log in `serve.log`).

Backend-selection evidence (serve.log):

- dense 2-bit Linear → `Using HummingLinearKernel for CompressedTensorsWNA16`
- routed experts → `Using CompressedTensorsWNA16MarlinMoEMethod` →
  `Using 'HUMMING' WNA16 MoE backend` (indexed GEMM)

## Model

`Yi30/Qwen3-30B-A3B-Instruct-2507-W2A16-G128-AutoRound-LLMC-200-Testing-Only`
(local: `/mnt/nfs/hoangduy/hf_assets/Yi30/Qwen3-30B-A3B-Instruct-2507-W2A16-G128-AutoRound-LLMC`,
9.2 GB). Qwen3MoeForCausalLM, 128 experts/top-8. quant: int2, group 128,
symmetric, `pack-quantized`, targets all Linear (routers + lm_head ignored).
Produced with AutoRoundModifier (iters=200, "testing-only" calibration — fine
for kernel smoke, NOT a quality reference). Published by yiliu30 (Intel), the
author of vLLM PR #48918, as that PR's test artifact.

## Serving stack (all steps reproducible)

1. **venv `serve-sub4`** = `cp -a` clone of `serve-026` (vLLM 0.26.0,
   torch 2.11, humming-kernels 0.1.10 in-venv) with:
   - shebangs/`bin/*` rewritten serve-026 → serve-sub4 (else console scripts
     load the *unpatched* venv);
   - **vLLM PR #48918** ("[CT] Support Humming for WNA16 MoE", open/unmerged,
     455-line pure-Python diff, copy in `pr48918.diff`) applied onto
     site-packages. 3 of 5 files applied clean with `git apply`; 2 files
     (`fused_humming_moe.py`, `quant_utils.py`) conflicted with our local M3
     W4AFP8 patches and were ported by hand (marked
     `PR #48918 port` in comments).
2. **humming main** side-install (`0.1.11.post25+ga9747951e`) at
   `/mnt/nfs/hoangduy/venvs/humming-main-site`, used via `PYTHONPATH`
   (shadows the in-venv 0.1.10). Needed: the compressed-tensors schema fix
   (humming PR #42) is in no tagged release.

## Environment gotchas hit (all fixed in launch.sh)

1. "idle" slurm node had ~38 GiB residue on GPU 0 → run with
   `--gpu-memory-utilization 0.45` (model is only 9 GB).
2. humming JIT → `libnvrtc-builtins.so.13.0` not found → `LD_LIBRARY_PATH`
   must include `$VENV/lib/python3.12/site-packages/nvidia/cu13/lib`.
3. humming launcher extension build → "Ninja is required" → `PATH` must
   include `$VENV/bin`; launcher was pre-compiled from the login box into the
   NFS `HUMMING_CACHE_DIR` (same cache reused by nodes).

## Upstream state at time of test (verified 2026-07-30)

- vLLM PR #48918: open, `needs-rebase`, 0 approvals; missed 0.25.x/0.26.0.
- Dense sub-4-bit (PR #46389, bits 2/3/5/6/7, symmetric+group only): merged,
  shipped in vLLM 0.25.0.
- humming v0.1.11 (2026-07-15) is the latest tag; CT schema fix merged to
  main 13h later. PyPI wheel alone is not enough for this path.

## What this does NOT show

- No quality claim (200-iter calibration model, 3 probes).
- No performance claim (shared node, first-run JIT, no benchmark).
- Asymmetric/actorder variants untested (vLLM humming path is symmetric-only;
  asym PR #46528 still open).

Next steps if we pursue: quantize our own W2A16/W3A16 MoE with the fork's
AutoRoundModifier (needs LLMC main past PR #2895 for sub-4-bit schemes),
paired-subset quality eval vs the 4-bit arm, and a per-user tok/s comparison
at matched concurrency.
