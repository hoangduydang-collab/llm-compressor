# Cross-layer predictor + codebook residual for the GLM-5.3 KV latent

Status: DESIGN, written 2026-10-09. Not an execution packet; nothing has been run.
Idea: the user's. Write-up and evidence links: planner session 2026-10-06 to 2026-10-09.

## Idea

Two predictors, chained, then a scalar quantizer on what is left:

1. **Cross-layer prediction (AQUA-KV style).** Predict token t's latent at layer l from the
   same token's *dequantized* latent at layer l-1: `p = A_l · ĉ_t^(l-1) + b_l`.
2. **Codebook on the prediction residual (ours).** `r1 = c_t^l - p`, nearest centroid
   `C[k]` (k-means fit on r1, with the zero centroid), `r2 = r1 - C[k]`.
3. **Scalar quantizer on r2** (NVFP4 or INT, as in the existing arms).

Decode: `ĉ_t^l = A_l · ĉ_t^(l-1) + b_l + C[k] + Q(r2)`. The predictor costs no stored bits;
only `k` and `Q(r2)` are stored, as today.

## Motivation

The two methods exploit different redundancy, and each is strong where the other is weak.

* **Codebook = similarity across tokens within a layer.** It captures 88-98% of latent
  energy in L0-3, but only 26-60% past L4 (residual median 0.67 of energy L4-77;
  ~75% at L30-66). Source: `20260925T1717Z-glm53-w4afp8-L0-77-full/layers/*.json`.
  Likely cause: deep latents carry each token's own context, so tokens stop clustering.
* **Cross-layer = similarity of the same token across layers.** MiniCache
  (arXiv 2405.14366, Llama/Mixtral): "KV cache in the shallow layers exhibits low
  similarity, whereas those in the middle-to-deep layers are very similar to one another."
  Both facts follow from the residual stream h^l = h^(l-1) + Δ^l: deep h is token-unique,
  but each deep layer changes it only a little.
* Our own data also shows deep-layer context sharing: neighbour (lag-1) correlation of the
  centred latent is ~0 in L0-6, 0.14-0.17 in L9-24, 0.31-0.37 in L36-60, 0.43 in L72-77.

Hope: the two are complementary, and the combination beats each alone. If both remove
roughly independent energy, residual energy should multiply: ρ_combined ≈ ρ_pred × ρ_cb.

## Prior art (checked 2026-10-06/09)

* AQUA-KV (arXiv 2501.19392): linear predictor per layer, inputs are reconstructed
  (dequantized) previous-layer pre-RoPE keys/values; one layer back is enough; layer 0
  unpredicted (3-4 bit); residual quantized with HIGGS. Llama-3.2-3B 2-bit: Wiki2 PPL
  7.47 (HIGGS) -> 7.03; without the key predictor 7.50. Previous-layer values explain
  >1/2 of value variance. Per-layer explained variance is only in App. B Fig. 5 (not
  discussed in text). GQA models only.
* MiniCache: merges adjacent middle-to-deep layers (shared direction + per-layer norms).
* No combination of a cross-layer predictor with a codebook residual found; nothing on
  MLA latents; nothing aligned with DSA IndexShare.

## Serving fit: DSA IndexShare

GLM-5.3 runs its own top-k indexer only on layers 0, 1, 2 and every 4th from 6 to 74
(21 "full" layers); the other 57 are "shared" and reuse the previous full layer's
top-2048 tokens. Verified 2026-10-09 from the released `zai-org/GLM-5.3` `config.json`
(`indexer_types`, 78 entries; `index_topk` 2048, `kv_lora_rank` 512). Do not derive it
from `index_topk_freq` (transformers' default differs; see
`pipeline/configs/glm52_distributed_w4afp8_full.yaml:109`). The harness threads each
window's top-k from layer to layer (`pipeline/kv_anchor_deep.py:534-537`, saved in the
resume state), so shared layers see the right tokens.

* Inside a group (e.g. 6-9), every layer reads the same 2048 tokens, so the rows of
  ĉ^(l-1) needed by layer l are already decoded (keep a ~2 MB/seq BF16 buffer, or re-read
  the codes). No scattered extra reads.
* Selecting layers start a new token set, so code them directly (no predictor), like
  video keyframes. The prediction chain is at most 3 layers.

## MLA specifics

* The 512-d latent is pre-RoPE by construction (AQUA found pre-RoPE predicts better).
  Keep the 64-d RoPE key part out of the predictor input.
* One latent serves as key and value, so AQUA's "current keys -> values" term has no
  counterpart; only the cross-layer term applies.
* Each layer projects h through its own W_DKV, so raw latents of adjacent layers need not
  look alike; A_l must learn the map. Its quality is the open question.
* Encoder must predict from ĉ^(l-1) (dequantized), not the exact latent, so encoder and
  decoder agree and error does not compound.

## Test plan

Harness: `pipeline/kv_anchor_deep.py` (streams layers in order; keep the previous
layer's per-arm decoded latents per window).

Layer groups (selecting layer first): 2-5 (shallow, codebook strong), 6-9, 34-37, 58-61.

Fit (training windows only): A_l by ridge regression on the *arm's own* dequantized
ĉ^(l-1), sequentially within a group; codebook k-means on r1 (with zero centroid,
attention-weighted w2 as in the current best arm).

Arms, matched bits (~4.5 b with NVFP4 residual; also INT4/INT3 residual variants):

| Arm | Predictor | Codebook | Residual |
|---|---|---|---|
| `sglang_nvfp4` | - | - | NVFP4 |
| current best `H-cb4096w2z-nvfp4\|tiny` | - | yes | NVFP4 |
| pred-only | yes | - | NVFP4 |
| **combined** | yes | yes (on r1) | NVFP4 |
| oracle pred (diagnostic) | exact c^(l-1) input | yes | NVFP4 |

Report per layer, AA-LCR eval and generated splits:
1. Predictor explained variance (exact and dequantized input).
2. Codebook capture on c vs on r1.
3. Latent rel-MSE and attention-output rel-MSE; ratio vs NVFP4.

Success: in deep groups (34-37, 58-61) the combined arm beats both the codebook-only and
the predictor-only arm, and beats NVFP4 by clearly more than today's ~1.5x; in the shallow
group it is no worse than codebook-only. Failure / stop: predictor explained variance
under ~20% in deep layers (then it cannot beat the codebook arm).

Cost: one short single-GPU run; can share the run with the planned per-layer latent
covariance + M_K/M_V sensitivity dump (see project notes on FSQ / transform coding).
