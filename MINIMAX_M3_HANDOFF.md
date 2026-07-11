# MiniMax-M3 Quantized Checkpoint Handoff

## Goal and current status

Verify full-calibration MiniMax-M3 W4A8 AWQ/GPTQ checkpoints with the
MiniMax-M3-specific vLLM serve path and identify why generation is garbage.

The checkpoint currently **loads successfully but has not passed quality
verification**. Do not delete any original checkpoint.

## Checkpoints

| Checkpoint | Status |
| --- | --- |
| `artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint` | Original AWQ W4A8; loads, outputs repeated `arring...` garbage. |
| `artifacts/MiniMax-M3-gptq-W4AFP8/20260709-064842/checkpoint` | Original GPTQ W4A8; loads, also outputs garbage. |
| `artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123` | Portable AWQ re-export; statically validated and loads, but still outputs garbage. |
| `/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4` | Working reference model. |

The re-export is 225 GB. It rewrites only Safetensors tensor names; raw tensor
payloads are copied byte-for-byte.

## Confirmed results

1. The original AWQ and GPTQ checkpoints pass structural checkpoint audits and
   load under the M3 vLLM serve setup, but produce nonsensical generation.
2. Serving the original AWQ checkpoint with W4A16 (no runtime FP8 activation
   quantization) still produced `arring...`. W4A8 activation quantization is
   therefore not the sole cause.
3. The original routed-expert tensor layout uses descriptive names:
   `gate_proj`, `down_proj`, and `up_proj`. The installed NVIDIA vLLM M3 loader
   maps only `w1`, `w2`, and `w3` for routed experts:

   - `w1 = gate`
   - `w2 = down`
   - `w3 = up`

4. `pipeline.reexport_minimax_m3_vllm` created a portable layout by renaming
   routed keys:

   - `gate_proj -> w1`
   - `down_proj -> w2`
   - `up_proj -> w3`

   Shared-expert keys intentionally remain descriptive; cyankiwi uses the same
   shared-expert naming.
5. Static re-export validation passed: 5 shards, 67,192 keys, 65,664 routed
   keys renamed. The portable checkpoint completed an 8-H100 vLLM serve with
   `loaded=true`, `rc=0`, and a nonempty output, but the output remained
   garbage:

   ```text
   seringk seringk seringk mempunastast...
   ```

   This proves the routed-key layout mismatch was real but not sufficient to
   explain the quality failure.

## Relevant code changes

- `pipeline/reexport_minimax_m3_vllm.py`
  - Header-only Safetensors re-export utility.
  - Refuses to overwrite a destination, validates transformed index/header
    keys and raw payload byte counts.
- `pipeline/tests/test_reexport_minimax_m3_vllm.py`
  - Tests routed key aliases and payload-preserving shard rewrite.
- `pipeline/slurm/patch_vllm_m3_serve.py`
  - Adds environment-gated `M3_LOAD_AUDIT=1` instrumentation.
  - The audit is injected into vLLM site-packages when `serve_verify` runs.
- `pipeline/serve_verify.py`
  - Installs the optional loader audit before vLLM worker creation.
- `pipeline/tests/test_patch_vllm_m3_serve.py`
  - Covers audit injection/idempotence.

CPU tests run successfully:

```bash
PYTHONPATH="$PWD" pytest -q \
  pipeline/tests/test_reexport_minimax_m3_vllm.py \
  pipeline/tests/test_patch_vllm_m3_serve.py
# 5 passed
```

## Logs and reports

- Original AWQ loader audit:
  `/mnt/nfs/hoangduy/logs/m3-awq-load-audit-srun.log`
- Direct M3 loader audit:
  `/mnt/nfs/hoangduy/logs/m3-awq-loader-audit2-srun.log`
- Re-export:
  `/mnt/nfs/hoangduy/logs/m3-awq-w123-reexport.log`
- Portable checkpoint serve:
  `/mnt/nfs/hoangduy/logs/m3-awq-w123-verify-srun.log`
- Portable checkpoint report:
  `artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123/serve_report.json`

The serve report's `sane_output=true` only means the response is nonempty. It
does not measure semantic quality and is a false positive for these runs.

## Important caveats for further analysis

- Both our checkpoints and cyankiwi place `n_shared_experts=1` under
  `text_config`, not the top-level config. The current audit heuristic warns
  about this, but cyankiwi works, so this warning alone is not root-cause
  evidence.
- The loader audit established that the original routed keys did not reach a
  fused-MoE parameter loader. Its aggregate alias diagnostic should be treated
  as supporting evidence rather than a final proof: the vLLM mapping contains
  full per-expert prefixes, and the audit's presentation is verbose.
- The portable checkpoint's unchanged garbage means the next investigation
  should compare actual loaded parameter values and shared-expert contributions
  against cyankiwi, rather than retrying activation dtype or routed key aliases.
- The existing MoE forward probe is guarded against CUDA graph capture, but
  prior runs did not obtain a useful real-prompt measurement.

## Suggested planner work

1. Design a minimal, low-risk comparison that samples loaded routed and shared
   parameter statistics from cyankiwi versus the portable AWQ checkpoint after
   vLLM construction.
2. Determine whether the shared-expert module is instantiated and receives its
   tensors in both runs, including its contribution during a real prefill.
3. Check whether compressed-tensors W4A8 expert loading supports the exported
   `weight_packed`, `weight_scale`, and `weight_shape` layout equivalently to
   cyankiwi's W4A16 checkpoint.
4. Only after a quality-positive serve should the user be asked again to delete
   the obsolete 225 GB original checkpoint.

## Active next handoff (2026-07-11, canonical chat matrix)

Run the four-node matrix in `MINIMAX_M3_QUALITY_RUNBOOK.md` from the handed-off
commit. The previous sequential reference result ruled out batching but exposed
that the quality harness used invalid bare completions: arithmetic emitted the
configured EOS token `200020`, and the France prompt continued raw text. This is
consistent with the user's known-good canonical HTTP chat serving and is not
evidence that cyankiwi is corrupt.

The new offline path applies the official MiniMax-M3 chat template. The matrix
runs reference/candidate through both offline and HTTP canonical chat on four
separate eager-mode nodes, with diagnostics off and identical quality settings.
Use `pipeline/slurm/submit_m3_chat_quality_matrix.sh`; do not manually combine
arms on one node. After all jobs finish, aggregate with
`python -m pipeline.m3_chat_quality aggregate`, inspect the compact bundle,
commit it, and push it through Git.

Do not repair the candidate, re-quantize, enable CUDA graphs, or resume the
second issue in this run. Runtime scheduler adaptation is welcome when recorded;
quality-variable changes require a new related attempt.
