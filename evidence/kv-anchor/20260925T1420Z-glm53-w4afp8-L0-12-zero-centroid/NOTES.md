# KV anchor + zero centroid, GLM-5.3 W4AFP8, layers 0-12

Job `hd-kv-anchor-glm53-deep-diag` rerun (gpu06, 1 GPU), commit b82005f5, `--layers 13 --diag --lexico-every 0`.
Source: `/mnt/cephfs/hoangduy/results/kv-anchor/glm53-w4afp8-deep-diag-20260925t142037z/`.
Latent dump (layers 3/6/9/12, same setup, identical layer numbers): `.../glm53-w4afp8-deep-dump-20260925t143016z/dump/`,
statistics from `pipeline/kv_residual_stats.py` in `residual_stats_cb4096_dump-20260925t143016z.json`.

## Zero centroid fixes the blow-up

Attention-output rel MSE, cb4096-ch-int2 -> cb4096z-ch-int2: L5 3.3e-2 -> 1.9e-2, L8 0.72 -> 0.070,
L10 41 -> 0.15, L12 67 -> 0.26 (direct-ch-int2 0.35). No anchored arm loses to direct at equal bits.

cb4096z-tok-int4 (4.15 b) beats fp8_unscaled (8.0 b) at every layer 0-12 (L12 5.7e-3 vs 8.7e-3);
fp8_tok (8.03 b) is 1.5-4x better from L4. Missing baseline: direct-tok-int{2,3,4}.

## Tiny, heavily attended tokens

* latent norm < 0.1x median (tokens >= 1): L6 0.5% of tokens take 56% of attention, L8 0.7% / 68%,
  L9-12 one token per window (position 2-13) takes 41-48%, on top of token 0's 49-50%.
* cause: kv_a_layernorm gains -- min |gain| 0.0000-0.0002 vs median 0.002-0.2; token 0's pre-norm RMS is
  1.1-8.7x the median token's and 160-680x sqrt(eps), so the norm input is not small.
* L9/L12 sinks form two tight groups (token 0 and the early second sink, cosine ~0.96 within, ~-0.8 across):
  2 dedicated centroids fit on half the windows reconstruct held-out sinks at 3.6% / 5.9% rel MSE with no
  residual. L6 tiny tokens are diverse (4 centroids: 24%).

## Residual after the anchor (cb4096z, eval windows)

* ||x-P|| / ||x|| median: L3 0.11, L6 0.42, L9 0.59, L12 0.60 -- not near zero past L4.
* spread across tokens (||r|| / median ||r||) p99: L3 3.1, L6 3.0, L9 1.7, L12 1.7.
* crest factor max|v|/rms(v), p50 (p99): ch groups raw 2.5-3.0 (3.7-4.4) -> residual 3.1-3.9 (4.7-6.6);
  tok groups raw 5.6 (9.2) at L3 -> 4.4 (8.5), deep layers 2.8 (4.0) unchanged. Anchoring makes ch groups
  peakier and tok groups no worse, matching tok > ch in attention error.
