# KV 4-bit baselines vs codebook arms, GLM-5.3 W4AFP8, layers 0-12

Job `hd-kv-anchor-glm53-deep-4bit` (gpu06, 1 GPU), commit 61fc4783 + log-line change, `--layers 13 --lexico-every 0`.
Source: `/mnt/cephfs/hoangduy/results/kv-anchor/glm53-w4afp8-deep-4bit-20260925t161400z/`.
`sglang_fp8_tile128` and `sglang_fp4_mx16` are bit-exact vs SGLang v0.5.17 reference quantizers; `sglang_nvfp4`
follows the NVFP4 recipe with a per-window global scale.

Attention-output rel MSE, AA-LCR eval windows (generated-text split in layers/*.json, same ordering, smaller gaps):

| L | FP8 served 8.25b | NVFP4 4.5b | H+NVFP4 | MXFP4-16 4.5b | H+INT4 4.12b | cb+z+INT4 \|tiny 4.15b | cb w2 INT4 | H+cb+z+NVFP4 4.52b |
|---|---|---|---|---|---|---|---|---|
| 0 | 4.7e-6 | 5.9e-5 | 5.5e-5 | 5.6e-4 | 1.6e-4 | 4.2e-6 | 3.3e-6 | 3.4e-6 |
| 2 | 1.5e-5 | 2.6e-4 | 1.9e-4 | 1.1e-3 | 3.1e-4 | 5.2e-5 | 4.8e-5 | 2.4e-5 |
| 4 | 1.1e-4 | 2.7e-3 | 2.6e-3 | 4.1e-3 | 2.9e-3 | 8.4e-4 | 8.2e-4 | 7.0e-4 |
| 6 | 1.6e-4 | 3.6e-3 | 3.5e-3 | 5.2e-3 | 4.0e-3 | 1.6e-3 | 1.6e-3 | 1.5e-3 |
| 8 | 1.0e-4 | 1.7e-3 | 1.6e-3 | 4.2e-3 | 2.0e-3 | 7.3e-4 | 8.1e-4 | 1.1e-3 |
| 10 | 2.6e-4 | 5.2e-3 | 4.9e-3 | 1.1e-2 | 6.0e-3 | 1.7e-3 | 2.1e-3 | 3.9e-3 |
| 12 | 3.2e-4 | 7.6e-3 | 7.0e-3 | 1.7e-2 | 8.2e-3 | 2.3e-3 | 2.8e-3 | 5.3e-3 |

* Every 4-bit arm is far from served FP8: NVFP4 is 13-27x worse; our best 6-10x (eval), 9-16x (gen) from L4.
  vLLM reports NVFP4 ~= FP8 KV downstream on GLM-5.2, so attention-output MSE parity with FP8 is not the bar.
* Best codebook arm vs NVFP4 at <= 4.5 b: 11-18x lower error L0-3, 2.1-3.9x L4-12 (eval); 4.6-9x / 1.3-2.6x (gen:
  codebook fit on AA-LCR documents, shifted on the model's own reasoning text).
* MX-FP4 (power-of-two scales, SGLang fp4_mx_block16) is 1.5-3x worse than NVFP4 -- consistent with its reported
  AIME25 drop on DeepSeek-R1. Hadamard helps NVFP4 only 5-40% (RMSNorm latent has few outliers).
* Deep layers: codebook + INT4 + tiny-exact is best; codebook + NVFP4 residual wins L1-3 but lacks the attention
  weighting / tiny exemption (combination untested).
