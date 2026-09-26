# KV codebook vs served 4-bit / 8-bit formats, GLM-5.3 W4AFP8, all 78 layers

Job `hd-kv-anchor-glm53-deep` (gpu06, 1 GPU, 6h41m), commit 60c98658, `--lexico-every 0 --gen-train 8`.
Source: `/mnt/cephfs/hoangduy/results/kv-anchor/glm53-w4afp8-deep-20260925t171725z/`.
Codebook fit on 15 AA-LCR + 8 generated (GLM-5.3 GPQA reasoning) windows; held out: 15 AA-LCR + 8 generated.

Best arm at every layer: `H-cb4096w2z-nvfp4|tiny` (4.52 b) -- attention-weighted 4096-centroid codebook + zero
centroid, Hadamard-rotated NVFP4 residual, tokens under 0.1x training-median norm kept exact (<= 0.7% of tokens).
It beats the INT4-residual variant (`cb4096w2z-tok-int4|tiny`, 4.15 b) in 78/78 layers.

Attention-output rel MSE (AA-LCR eval; generated split in layers/*.json, within ~10%):

| L | FP8 served 8.25b | NVFP4 4.5b | best 4.52b | NVFP4 / best | best / FP8 |
|---|---|---|---|---|---|
| 0 | 4.7e-6 | 5.9e-5 | 2.9e-6 | 20x | 0.6x |
| 6 | 1.6e-4 | 3.6e-3 | 1.4e-3 | 2.6x | 8.8x |
| 12 | 3.2e-4 | 7.6e-3 | 1.9e-3 | 4.0x | 5.9x |
| 24 | 3.7e-4 | 6.3e-3 | 3.3e-3 | 1.9x | 9.0x |
| 36 | 8.2e-4 | 9.3e-3 | 6.4e-3 | 1.5x | 7.8x |
| 48 | 9.7e-4 | 1.2e-2 | 8.2e-3 | 1.5x | 8.5x |
| 60 | 7.3e-4 | 9.1e-3 | 5.7e-3 | 1.6x | 7.8x |
| 72 | 7.3e-4 | 1.3e-2 | 9.1e-3 | 1.4x | 12.6x |
| 77 | 4.4e-4 | 7.3e-3 | 4.4e-3 | 1.7x | 10.0x |

* L4-77: NVFP4/best median 1.59x (eval) / 1.54x (gen), range 1.17-3.97x; best/FP8 median ~9x, max 15x.
  Past ~L30 the gain settles at 1.3-1.7x. Generated-text gap closed once generated windows joined the fit.
* 3-bit variant (`cb4096w2z-tok-int3|tiny`, 3.15 b) is ~2x NVFP4's error (median; beats it in 3-5/78 layers),
  so no bit saving vs NVFP4 at equal error.
* Keeping tiny tokens exact helps plain NVFP4 only where sinks dominate (L12 7.6e-3 -> 4.3e-3).
