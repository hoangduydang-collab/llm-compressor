# KV anchor: attention-aware objective, GLM-5.3 W4AFP8, layers 0-12

Job `hd-kv-anchor-glm53-deep-diag` (gpu06, 1 GPU), commit 1a57a6ee, `--layers 13 --diag --lexico-every 4 --dump-layers 6,9,12`.
Source: `/mnt/cephfs/hoangduy/results/kv-anchor/glm53-w4afp8-deep-diag-20260925t151725z/` (dumps for L6/9/12 there).
Goal: ~4-bit error comparable to served FP8; 2-bit is insight only.

## 4-bit (attention-output rel MSE, AA-LCR eval windows)

| L | fp8_tok 8.03b | direct-tok-int4 4.12b | cb4096z-tok-int4 4.15b | w2 | zs | z\|tiny |
|---|---|---|---|---|---|---|
| 0 | 7.7e-6 | 4.4e-4 | 4.2e-6 | 3.3e-6 | 4.2e-6 | 4.2e-6 |
| 3 | 8.8e-5 | 4.6e-3 | 6.5e-5 | 6.4e-5 | 6.5e-5 | 6.5e-5 |
| 5 | 2.8e-4 | 2.0e-3 | 1.1e-3 | 6.0e-4 | 6.2e-4 | 6.6e-4 |
| 8 | 3.9e-4 | 2.3e-3 | 1.3e-3 | 8.1e-4 | 1.3e-3 | 7.3e-4 |
| 10 | 1.1e-3 | 6.1e-3 | 4.3e-3 | 2.1e-3 | 2.1e-3 | 1.7e-3 |
| 12 | 1.5e-3 | 8.8e-3 | 5.7e-3 | 2.8e-3 | 2.8e-3 | 2.3e-3 |

* Attention-weighted k-means (w2: mean squared attention weight per training token) and keeping tiny tokens
  (<0.1x training-median norm) exact each cut deep-layer 4-bit error 30-60%; best arm is 1.5-2.3x fp8_tok from L4,
  equal or better at L0-3.
* Anchor vs no anchor at equal grouping (tok): 2.5-3.9x lower error in deep layers.
* Lexico (s=45, 2.14 b) vs cb4096w2-tok-int2 (2.15 b): L0 1.2e-4 vs 1.9e-5, L4 1.4e-2 vs 7.7e-3, L8 3.4e-2 vs
  2.0e-2, L12 0.13 vs 0.051.

Missing here (next run `hd-kv-anchor-glm53-deep-4bit`): SGLang FP8-tile / NVFP4 / MX-FP4, Hadamard arms,
codebook + NVFP4 residual.
