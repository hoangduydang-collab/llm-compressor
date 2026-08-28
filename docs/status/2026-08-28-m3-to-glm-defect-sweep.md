# M3 → GLM-5.2/5.3 defect sweep

2026-08-28. Systematic pass over the 40+ documented failures in
`BUGS_AND_FIXES.md`, classified by the only question that matters operationally:
**does the M3 fix protect GLM, or is it gated on detecting M3?**

Motivation: the uncompensated-router defect was a bug M3 had already solved,
tested, and documented, and it shipped into a GLM checkpoint anyway because the
fix lived behind an M3 gate. That is a *class*, not an incident, so this sweep
looks for every other member of the class.

Two new serve-blocking or audit-blinding findings came out of it (§2.2, §2.4).

---

## 1. M3 fixes that DO protect GLM (verified model-agnostic)

| M3 defect | Fix location | Why it covers GLM |
|---|---|---|
| AWQ grid search hijacked by FP8-schemed balance layers (r8a) | `modifiers/transform/awq/base.py::_is_grid_search_targeted` | Keys off the scheme's *type* (float vs int), not the model. Regression `test_fp8_mixed_recipe.py` passes. |
| Smoothing-scale degeneracy on dead norm channels (r4) | `m3_checkpoint_scale_audit.py::compensation_error` | Explicitly handles both norm forms: "offset form: base weight exactly -1 → gain 0 …; plain form: base weight exactly 0". Generalized when the norm-offset parameter was added. |
| AWQ smoothing fold lost under disk offload (r2) | `modeling/offset_norm.py` write-back + source-rank persistence in `quantize.py` | Fix is in the offload/write-back path, which every model shares. |
| `pin_memory` / weights-side VMA exhaustion | VMA budget gate in `quantize.py` | Computes the projection from the live model; `M3_SKIP_VMA_GUARD` is only the bypass name. Confirmed firing on GLM: "VMA gate OK: ~2508 planned shm segments + 24000 slack ≤ 262144". |
| Offloaded-save health (tie detection on buffers, meta-tensor renames, r11/r12) | transformers hotfix + `assert_transformers_offloaded_save_healthy` | Gate ran on GLM: "offloaded-save gate OK: transformers path is healthy". |
| `DistributedDiskCache.update_offload` write race (r10) | patched in `quantize.py` | Applied unconditionally; log shows it on GLM runs. |
| llm-compressor pruning ignore patterns for unquantized modules | `quantize.py::_persist_ignore_to_config` | Model-agnostic, and its docstring already warned that mishandling `mlp.gate` gives "broken routing → garbage output". |

**Worth noting about that last row:** the codebase already documented that the
router is special and that getting it wrong breaks routing — in the *serving*
context. Nobody connected it to the *smoothing* context. Both were about
`mlp.gate`; both had the same consequence.

---

## 2. M3 fixes that do NOT protect GLM (M3-gated) — actionable

### 2.1 AWQ mappings — the router defect (FIXED today)

`register_minimax_m3_awq_mappings()` is called only inside
`if patch_minimax_m3_for_text_calibration(model):`. GLM therefore resolved
through the generic static registry, which had dropped `mlp.gate` from the
balance set. M3's own mapping keeps the router and scopes patterns to sparse
layers 3-59 (`_M3_SPARSE_LAYER`), with a test asserting "router must be
balanced".

Fixed by `build_mla_mixed_dense_moe_mappings` (commit `5cc823b5`), which ports
M3's layer-scoping technique. Validated in-cluster: `3 dense + 75 MoE layers`,
router in the balance set, layer-3 mapping resolved with 0 skips.

### 2.2 NEW FINDING — the r8 "served garbage with a passing smoke" defect is latent in GLM-5.2

M3's r8 root cause: the saved `quantization_config.ignore` carried broad
quant-layout regexes, and **vLLM checks ignore BEFORE targets**, so every FP8
module served as "unquantized" — raw fp8 bits cast into bf16 with scales
silently dropped, producing `"omensomens…"` while the smoke returned RC=0.

**GLM-5.2's saved checkpoint has the identical overlap.** `_persist_ignore_to_config`
appends the recipe's regexes to the serialized list, so the shipped config ends
with (lines 56834-56839 of its `ignore`):

```
re:.*mlp[.]gate$
re:.*mlp[.]shared_experts[.].*
re:.*self_attn[.].*
re:.*layers[.][0-2][.].*
re:.*layers[.]78[.].*
re:.*model[.]layers[.](?!(?:3|42)(?:[.]|$))[0-9]+(?:[.]|$).*
```

and every `fp8_dynamic_targets` entry is matched by one of them:

| fp8 target | shadowed by |
|---|---|
| `layers.(0\|3).self_attn.(q_a_proj\|q_b_proj\|kv_a_proj_with_mqa\|kv_b_proj\|o_proj)` | `re:.*self_attn[.].*` |
| `layers.3.mlp.shared_experts.(gate_proj\|up_proj\|down_proj)` | `re:.*mlp[.]shared_experts[.].*` |
| `layers.0.mlp.(gate_proj\|up_proj\|down_proj)` | `re:.*layers[.][0-2][.].*` |

The M3 remedy is `pipeline/reexport_minimax_m3_vllm.py --fp8-serve-fix`, which
rewrites targets and ignores into serve layout and runs a fail-closed
storage-vs-scheme audit. **There is no GLM equivalent** — `ls pipeline/ | grep
reexport` returns only the M3 file.

Confidence: high but not yet engine-verified. The regex overlap is exact and the
M3 precedent was verified against vLLM's own `should_ignore_layer`. Before any
serve or eval of a GLM W4AFP8 checkpoint, run that same matcher against our
config. Note also a first-pass check that looked for *concrete* fp8 module names
in the ignore list came back clean and was misleading — the regexes cover them.

### 2.3 Serve-config patching has no GLM path

`ensure_minimax_m3_vllm_serve_config(ckpt, model_id)` is M3-only. Whatever
config rewrites a GLM W4AFP8 checkpoint needs for its serving engine, nothing
currently performs them.

### 2.4 NEW FINDING — the smooth-fold gate audits only layers 3-59, so GLM layers 60-77 are unchecked

`quantize.py:1276`: `gate_layers = list(range(3, 60))` when
`M3_DIAGNOSTIC_LAYERS` is unset. That is M3's sparse range. GLM-5.2/5.3 have 78
layers (MoE layers 3-77), so **18 of 75 MoE layers are never audited** by the
fold-consistency gate. A fold lost only in that tail — exactly the failure mode
of r2/r3/r7 — would pass silently.

Should derive the range from the model (or from the recipe's targeted layers)
rather than a constant. Until then, set `M3_DIAGNOSTIC_LAYERS` explicitly for
GLM runs.

### 2.5 Checkpoint verifier preset — partially fixed

`verify_quant_checkpoint.py` now takes `--expect-ignore-preset glm52`, but the
in-run caller `quantize.py::assert_quant_checkpoint_verified` still uses the M3
default. The router-fix validation run exited 1 for exactly this reason: its only
failures were `vision_tower`, `multi_modal_projector`, `patch_merge`,
`block_sparse_moe` — modules GLM does not have. One-line wiring fix outstanding.

---

## 3. M3 defects that genuinely do not apply to GLM (with the reason)

- **AWQ `up_proj → down_proj` fold not function-preserving.** M3's expert
  activation is gpt-oss style, `h = (clamp(up, ±7) + 1.0) * glu`, which is affine
  and clamped in `up`, so the fold changes the function per channel (~5-33% RMS
  perturbation, ~10x the int4 rounding error; the cause of M3's reasoning
  non-termination). GLM's experts are plain SwiGLU — `act_fn(gate) * up`, no
  clamp, no `+1` — so the same fold is a pure reparameterization. Our own config
  comment says so, and the M3 post-mortem records that GLM-5.2 AWQ W4AFP8 was
  clean on this axis. **Keeping `up_proj → down_proj` in the GLM mapping is
  correct.**
- **`image_token_id` calibration failure, `image_features.numel()` FX trace
  failure, `preprocessor_config.json` / `img_token_compression_config` serve
  failures, vision-tower name collisions.** All MiniMax-M3-VL multimodal
  specifics. GLM-5.2/5.3 are text-only in our path.
- **`LinearExperts2D has no attribute swiglu_limit`, r7 gate-alpha fold,
  `Unsupported activation: silu. Only swigluoai is supported`,
  `SWIGLUOAI_UNINTERLEAVE` CUTLASS backend patches.** All consequences of M3's
  SwiGLU-OAI activation. GLM uses silu/SwiGLU.
- **`MinimaxM3QKVParallelLinearWithIndexer` q/k/v+indexer GEMM fusion blocking
  fp8 qkv.** A property of vLLM's M3 plugin, not of GLM.
- **FlashInfer fused all-reduce hang, `gemma_rmsnorm` CuTe-DSL JIT abort,
  shared-experts aux-stream CUDA-graph IMA.** M3 serve-path issues; they would
  need re-testing per engine for GLM but are not inherited defects.

---

## 4. Recurring classes, independent of model

These are the patterns worth internalizing, since each produced multiple
incidents:

1. **A model-gated fix silently does not protect the next model.** Root cause of
   §2.1 and §2.2. Every M3-conditional branch is a place where the next
   architecture starts from the pre-fix state.
2. **"Has a quantization scheme" ≠ "wants AWQ smoothing"** (r8a), and **"is not
   quantized" ≠ "needs no compensation"** (the router). Both are the same error:
   using quantization status as a proxy for a different property.
3. **A `quantization_config` is a serving contract written in the serve model's
   namespace.** Every rename between quant layout and disk layout must rewrite
   targets *and* ignores (r8).
4. **Smoke tests must assert on content, not transport.** r8 returned RC=0 while
   emitting pure repetition. This is why `pipeline/sample_output_check.py` now
   judges output rather than printing it.
5. **Thresholds calibrated on one model mislead on another.** The dequant
   residual bound 0.25 comes from M3 r9 full calibration (healthy 0.09-0.12); the
   GLM 32-sample smoke reads 0.323 and that is not, by itself, evidence of a
   defect.
6. **Gates run under the wrong architectural constant do not merely weaken — they
   can invert.** Auditing GLM's fold with M3's offset norm form made the router
   look healthy (6.75e-3) and the shared experts look broken (0.232), i.e.
   exactly backwards.
7. **A first-pass check that comes back clean deserves suspicion when it is cheap
   to make it wrong.** Three times today: `grep '\n'` (reduced to `n`), the
   rotated log read as complete, and the concrete-names-only ignore check in
   §2.2.

---

## 5. Can we AWQ-quantize the already-FP8 GLM-5.3 to W4A8?

**Not directly from the FP8 release, but yes from the BF16 release — which is what
we are downloading.**

`zai-org/GLM-5.3` ships block-wise FP8: `quant_method: "fp8"`, `fmt: "e4m3"`,
`weight_block_size: [128, 128]`, `activation_scheme: "dynamic"`, plus a
`modules_to_not_convert` list. Architecture is identical to GLM-5.2
(`GlmMoeDsaForCausalLM`, 78 layers, hidden 6144, 256 routed experts, top-8,
`first_k_dense_replace: 3`, 1 MTP layer), so every mapping and recipe carries
over.

Why not straight from FP8:

- AWQ must read weights as bf16 to compute per-channel smoothing scales and pack
  INT4. Block-FP8 weights need dequantizing first (`W_bf16 = W_fp8 × scale_inv`
  broadcast over each 128×128 block).
- **llm-compressor has no support for that.** No `FineGrainedFP8`, no
  `weight_scale_inv` handling anywhere in `src/`. It would be new numerics code
  in the exact area where every defect this week originated.

`zai-org/GLM-5.3-BF16` exists: **1.51 TB, 282 shards, no `quantization_config`,
bf16** — structurally identical to GLM-5.2 (1403 GiB, 282 shards), so the
pipeline runs unchanged. Measured download 136 MB/s, ETA ~3 h.

**One honest caveat on quality.** The BF16 release is an upcast of natively-FP8
weights, not a higher-fidelity original — Z.ai state the FP8 release is the
reference rather than a lossy derivative. So GLM-5.3's effective weight precision
is already ~FP8 (e4m3: 4 exponent, 3 mantissa bits, per-128×128 block scale)
before we start. Quantizing that to INT4 g128 is a *smaller* additional step than
GLM-5.2's true-bf16 → INT4, but the headroom AWQ has to work with is also
smaller, and some detail is already gone. Whether W4A8-from-FP8 lands closer to or
further from its baseline than W4A8-from-bf16 did is an empirical question for the
eval harness; it should not be assumed to transfer from the GLM-5.2 result.
