# KV anchor diagnostic, GLM-5.3 W4AFP8 (served EP-GPTQ checkpoint), layers 0-12

Job `hd-kv-anchor-glm53-deep-diag` (gpu06, 1 GPU), `--act-fp8 --layers 13 --diag --lexico-every 0`.
Source: `/mnt/cephfs/hoangduy/results/kv-anchor/glm53-w4afp8-deep-diag-20260925t133436z/`.
Latent = `kv_a_layernorm` output (512 of the 576 cached values per token per layer; RoPE key left exact).
Windows start 1024 tokens into each document, no BOS/chat template.

## Findings

1. Attention sink appears at layer 7 and saturates by layer 9: token 0 receives
   0.2-0.4% of attention at L0-6, 14-16% at L7-8, 49-50% at L9-12.
   Its latent shrinks at the same time: norm / median-token norm 0.4-1.1 (L0-6), 0.17 (L7), 0.018-0.036 (L9-12).
2. Anchors hurt near-zero tokens. The nearest centroid (or the global mean) has typical norm, so the residual
   is ~ -P and its quantization error is many times the token's own size. Token-0 latent rel MSE at L10:
   direct-ch-int2 1.0, cb4096-tok-int2 28, cb4096-ch-int2 99, gmean-ch-int2 98. Direct cannot exceed ~1.
3. Keeping token 0 exact (`|keep1`) cuts anchored attention error ~4x at L9-12 but does not close the gap:
   L12 attn rel MSE cb4096-ch-int2|keep1 17.4, cb4096-tok-int2|keep1 6.6, direct-ch-int2|keep1 0.33.
   Unexplained remainder; candidate = other small-norm, heavily attended tokens (unmeasured).
4. Attention amplification of latent error grows with depth for every arm (attn/latent rel MSE:
   fp8_tok 0.011 at L0 -> 1.1 at L6; direct-ch-int4 0.002 -> 1.7), before the sink forms.
5. Before the sink (L4-7) `tok` grouping beats `ch` after anchoring: L6 cb4096-tok-int3 attn 6.9e-3 vs
   direct-ch-int4 2.0e-2.

## Next (not yet run)

Zero centroid in the codebook (tiny tokens fall back to direct), count of small-norm tokens and the attention
they receive, `kv_a_layernorm` gains and pre-norm norms for those tokens.
