# GLM-5.2 distributed PTQ: what carries over from MiniMax-M3

MiniMax-M3 was our first correct large-scale quantization, and `BUGS_AND_FIXES.md`
records **31 structured incidents** from it. This document classifies every one of
them for GLM-5.2 so the GPTQ and AWQ arms do not rediscover them.

The classification is deliberately three-way, because "we hit this on M3" is not by
itself a reason to expect it on GLM-5.2 — and treating it as one wastes effort on
impossible failures while hiding the ones that genuinely transfer:

- **[A] Does not transfer** — the mechanism depends on something M3 has and
  GLM-5.2 does not. Verified against source, not assumed.
- **[B] Transfers** — the mechanism is method- or infrastructure-level. Verify the
  fix is present *before* spending GPU time.
- **[C] New GLM-5.2 risk** — found while writing this, no M3 analogue.

Model facts used throughout, read from the released `config.json` and
`transformers v5.14.1` `modeling_glm_moe_dsa.py` (not from memory):

| | GLM-5.2 | MiniMax-M3 |
|---|---|---|
| modality | **text-only** | vision-language |
| RMSNorm | `self.weight * h` (**plain**) | `(1 + weight) * h` (**offset**) |
| expert activation | `act_fn(gate) * up` (**plain SwiGLU**, `hidden_act=silu`) | `(clamp(up, ±7) + 1.0) * glu` (gpt-oss style) |
| attention | MLA (`q_a/q_b/kv_a_proj_with_mqa/kv_b`) | MLA + BF16 indexer |
| dense layers | **0,1,2** (`first_k_dense_replace=3`) | — |
| experts | 256 routed + 1 shared, `moe_intermediate_size=2048` | — |
| AWQ mappings | **registered upstream** (`GlmMoeDsaForCausalLM`) | hand-authored in-repo |

---

## Required order (from `BUGS_AND_FIXES.md`)

Do not reorder these. Each one exists because something expensive was learned late.

1. **pre-quantization gate** (CPU-only, no GPU) — `python -m pipeline.prequant_compatibility`
2. **representative-layer canary** — required for expensive/new recipes, which this is
3. full quantization
4. post-quantization serving ABI gate
5. runtime smoke
6. quality evaluation

---

## [A] Does NOT transfer — with the evidence

Nine M3 incidents are structurally impossible on GLM-5.2. Recorded so nobody
re-litigates them.

| M3 incident | Why it cannot happen on GLM-5.2 |
|---|---|
| **AWQ up→down fold not function-preserving** — cost 24 pts GPQA via reasoning non-termination | The fold `up_rows /= s`, `down_cols *= s` is exact iff down's input is *linear* in `up`. GLM: `act_fn(gate) * up` → linear in `up`. M3's `(clamp(up,±7)+1.0)*glu` is affine **and clamped**, so folding changed the function per-channel. `BUGS_AND_FIXES.md` says so directly: *"Qwen3/GLM use plain SwiGLU … the identical mapping is exactly legal there"*, and cites GLM-5.2's own AWQ W4AFP8 as clean. **Verified**: zero matches for `clamp\|swiglu_limit\|swiglu_beta\|1.702\|alpha` in the whole GLM modeling file. |
| **AWQ offset-RMSNorm smoothing** (the open investigation at the top of the file) | Needs a norm whose forward is `(1+weight)·h`. `GlmMoeDsaRMSNorm.forward` returns `self.weight * hidden_states`. No `CalibrationOffsetNorm` adapter is required, and the `missing_offset_norm_adapter` gate is not applicable. |
| `LinearExperts2D has no attribute swiglu_limit` | M3's `_apply_gate` read config-derived scalars. GLM's experts carry no `swiglu_*` attributes at all. |
| `image_token_id` FX-trace failure | VL-only. GLM-5.2 is text-only. |
| `image_features.numel()` FX-trace failure | VL-only. |
| missing `preprocessor_config.json` | VL-only (`ensure_vl_processor_artifacts`). |
| `img_token_compression_config` missing | VL vision-config coercion. GLM's config is flat — no `text_config`/`vision_config` nesting. |
| `Unsupported activation: silu. Only swigluoai is supported` + CUTLASS `SWIGLUOAI_UNINTERLEAVE` | Both are the `swigluoai` activation. GLM is plain `silu`. |
| shared-experts aux-stream CUDA-graph IMA, FlashInfer `gemma_rmsnorm` JIT, custom all-reduce hang | M3 serving-path specifics tied to its norm/expert layout. |

**One generic lesson survives from A**, though: M3's `linearize_moe` OOM (RSS ~1980 GiB) came from *config dtype*, not from the model — `text_config` had no `dtype`, so experts materialized in fp32. GLM's config is flat, so the M3 coercion bug can't recur, **but the failure mode can**: I already hit `MoEConfig.from_config` reading `config.dtype` (not parameter dtype) while testing the GLM linearize path. Assert `config.dtype is torch.bfloat16` before linearize regardless.

---

## [B] TRANSFERS — verify before burning GPU time

These are method- or infrastructure-level. Ranked by risk for *our* recipe, which
is the relevant filter: **W4AFP8 = int4 routed experts + FP8_DYNAMIC attention /
shared experts / dense MLPs, under DDP, with `offload_folder` set.** That is the
exact combination that produced M3's r7/r8 series of bugs.

### HIGH — our recipe hits all four preconditions

| M3 incident | Fix site | Why it threatens GLM-5.2 |
|---|---|---|
| **NCCL deadlock from rank-sharded FP8 weight-qparam updates** (r8 smoke) | `QuantizationModifier.on_sequential_epoch_end` — weights replicated across ranks, so weight observation must not be rank-sharded | Needs distributed + FP8_DYNAMIC mixed recipe. **We have exactly that.** A deadlock hangs the run indefinitely rather than failing — the worst failure shape for a long job. |
| **Global pack-quantized override corrupts FP8 group at save** (r8 smoke v2) | `_stamp_mixed_precision_formats(model)` sets `scheme.format` per group | Needs mixed int4+FP8 at save. **We have that.** M3's symptom: run completes, checkpoint is silently wrong. |
| **AWQ grid search hijacked by FP8-schemed balance layers** (r8a smoke) | `_is_grid_search_targeted` in AWQ base | Needs AWQ + FP8-schemed balance layers. **The AWQ arm has that.** Our GLM config FP8-targets `self_attn.*` and `shared_experts.*`, which are exactly the balance layers `_deepseek_mappings` reaches. |
| **AWQ smoothing fold / qparam broadcast lost under disk offload** (r2, r3, r7 post-mortems) | `CalibrationOffsetNorm.restore`; `_make_fold_consumer` writes the composed scale through the offload path | The *site* was M3's offset norm, but the *class* is generic: a fold or qparam written to a module whose params live in `offload_folder` is silently discarded. **Our GLM configs set `offload_folder`.** Three separate M3 runs died this way — it is the single most repeated failure in the record. |

### MEDIUM — generic to large offloaded/DDP runs

| M3 incident | Note for GLM-5.2 |
|---|---|
| `pin_memory` VMA exhaustion in `IntermediatesCache._offload_value` | Fork removed pinning entirely. Confirm the patch is still present — this is an *upstream* regression we carry a local fix for, so a rebase can silently reintroduce it. |
| DDP weights-side VMA exhaustion (per-tensor shm) | Fixed 2026-07-17. Same rebase caveat. |
| `DistributedDiskCache.update_offload` write race | DDP + disk offload. We have both. |
| offloaded-save tie detection crashes on buffers | Offloaded save. We have it. |
| offloaded-save revert renames meta tensors before materialization | Offloaded save. We have it. |
| **AWQ smoothing-scale degeneracy on dead norm channels** | Root cause described as *"identical upstream"* in `AWQModifier._compute_best_scale`. Any model can have dead norm channels; GLM is not exempt. Applies to the AWQ arm. |
| post-quant `SAMPLE GENERATION` hangs, gating the save | **Already mitigated** — both GLM configs set `sample_generation: false`. Keep it that way; on a 743 B offloaded model 64 tokens streams every expert off disk. |
| vLLM worker init: fused q/k/v + indexer packed GEMM vs a BF16 indexer | GLM-5.2 also keeps the whole indexer BF16 (we ignore `self_attn.*`). Serving-side, but the same shape — expect it at gate 4, not during quantization. |
| r8 mixed int4+FP8 served garbage **with a passing smoke** | Process requirement, not code: the smoke must assert output *quality*, not HTTP 200. This is why gate 5 exists separately from gate 4. |
| Detached launcher `kill $(cat PID_FILE)` missed the worker (stale `$!`) | Kubernetes makes the specific bug moot, but the lesson stands for the new launcher: the recorded handle must be the thing that actually dies. |
| Pre-quantization gate: meta-device MoE linearization offload | Needed for gate 1 to run *at all* on any MoE model — `linear_experts.py` skips the offload loop when `offload_device == meta`. GLM's gate run depends on it. |

---

## [C] NEW GLM-5.2 risk, found writing this

**AWQ mapping includes `mlp.gate` but GLM-5.2 has three routerless dense layers.**

`GlmMoeDsaForCausalLM` → `_deepseek_mappings`, whose MoE-input mapping is:

```python
AWQMapping(
    "re:.*post_attention_layernorm$",
    ["re:.*mlp.gate$", "re:.*gate_proj$", "re:.*up_proj$"],
)
```

Directly below it in the same file, `_glm4_moe_lite_mappings` exists *specifically*
to drop `mlp.gate`, with this comment:

> GLM-4.7-Flash … has a mixed dense/MoE architecture: layer 0 is dense
> (`first_k_dense_replace=1`) … The dense layer 0 has no `mlp.gate` router, so we
> cannot include `mlp.gate` in the balance layers (**it would break per-layer
> grouping in `match_modules_set`**).

**GLM-5.2 has `first_k_dense_replace = 3`** — layers 0, 1, 2 are dense and have no
router, yet it is pointed at the mapping that *includes* `mlp.gate`.

Counter-evidence worth stating: `DeepseekV3ForCausalLM` also uses
`_deepseek_mappings` and DeepSeek-V3 likewise has `first_k_dense_replace=3`, so
either `match_modules_set` tolerates the absent gate or this latent bug has simply
never been exercised. **Unresolved — do not guess.** Gate 1 resolves it for free:
`prequant_compatibility` runs the real AWQ mapping resolver on a meta model with no
GPU and no weights. If resolution breaks, the fix is a GLM-5.2 entry mirroring
`_glm4_moe_lite_mappings`.

This is exactly the M3 lesson generalizing correctly: M3's *"AWQ fails on default
smooth-layer mappings"* was the same category of failure (registry/architecture
mismatch), even though the specific mapping and cause differ.

---

## Pre-launch checklist

```bash
# Gate 1 — CPU-only, no GPU, no weights. Run for BOTH arms.
python -m pipeline.prequant_compatibility \
  --config pipeline/configs/glm52_distributed_w4afp8_smoke.yaml \
  --output artifacts/preflight/glm52-gptq.json
python -m pipeline.prequant_compatibility \
  --config pipeline/configs/glm52_distributed_w4afp8_awq_smoke.yaml \
  --output artifacts/preflight/glm52-awq.json

# Target/regex gate (complements gate 1: catches dead FP8 patterns)
python -m pipeline.quant_target_preflight \
  --config pipeline/configs/glm52_distributed_w4afp8_smoke.yaml \
  --arch-config <snapshot>/config.json
```

### HIGH-fix presence — verified 2026-08-27 on `duy-branch`

All four are local fixes layered over upstream, so a rebase can drop them
silently. Re-run these greps after any rebase; all four currently PASS:

| fix | evidence |
|---|---|
| `_stamp_mixed_precision_formats` defined **and called** | `pipeline/quantize.py:466` (def), `:925` (call) |
| `_is_grid_search_targeted` in AWQ modifier | `src/llmcompressor/modifiers/transform/awq/base.py:132` (def), `:487` (used in the smooth/balance guard) |
| weight observation replicated, not rank-sharded | `src/llmcompressor/modifiers/quantization/quantization/base.py:82` — *"Weights are replicated across DDP ranks, so weight observation is …"* |
| **no** `pin_memory()` in `IntermediatesCache._offload_value` | `src/llmcompressor/pipelines/cache.py` — sole `pin_memory` occurrence is a docstring at `:86` about `dataloader_num_workers`, not in the offload path |

```bash
# after any rebase, all four must still hold
grep -rn "_stamp_mixed_precision_formats" pipeline/quantize.py
grep -rn "_is_grid_search_targeted" src/llmcompressor/modifiers/transform/awq/base.py
grep -rn "replicated across DDP ranks" src/llmcompressor/modifiers/quantization/quantization/base.py
grep -n "pin_memory" src/llmcompressor/pipelines/cache.py   # docstring only
```

The disk-offload fold/qparam-loss class (the fourth HIGH item) has **no single
grep** — it was three separate M3 runs with three fix sites. Gate 1 does not
cover it either, since it never runs a forward. That is what the gate-2
representative-layer canary is for: quantize a couple of layers with
`offload_folder` set and confirm the written scales survive a reload.

And two config-level carry-overs already in place — keep them:
`sample_generation: false`, and `moe_calibrate_all_experts: true` with
`sequential_targets: ["GlmMoeDsaDecoderLayer"]`.
