# --propagate smoke run, GLM-5.3 W4AFP8, layers 0-3

SkyPilot managed job 22 (`hoangduy-kv-prop-smoke`), test-ca-01, 1x H100, image
lmsysorg/sglang:v0.5.17 (torch 2.11.0+cu130, transformers 5.12.1), code at commit 847511c3
(`kv_anchor_deep.py` sha256 dd040b74...). Args: `--weights w4afp8 --act-fp8 --gen-train 8
--propagate --layers 4`. Output: `/cephfs/hoangduy/kv-prop/glm53-w4afp8-prop-smoke-20261012t080405z/`.
23 train (15 AA-LCR + 8 generated) / 15 AA-LCR eval + 8 generated windows x 4096 tokens.

* Gates 1, 2, 3 PASS on GPU. Layer-0 numbers equal the 2026-09-25 single-layer run (no upstream
  error yet): fp8 4.67e-6, nvfp4 5.94e-5, cb 2.96e-6 attention rel-MSE.
* Cost: dense layers 44-59 s; first MoE layer 168 s, of which 133 s is load (CephFS read +
  dequant + one-layer checkpoint round trip); compute ~35 s. Peak GPU memory 54.0 GB.
  Full 78-layer estimate ~3.5-4 h on 1 GPU, load-bound.
* DSA top-k overlap vs exact at layer 2 (full): 0.995-0.996 for every arm.

Layer 3 (first "shared" DSA layer, predicted from layer 2), AA-LCR eval:

| arm | bits | local latent rel | local attn rel | attn vs exact | hidden drift |
|---|---|---|---|---|---|
| fp8 | 8.25 | 4.03e-4 | 1.7e-5 | 2.75e-4 | 3.15e-4 |
| nvfp4 | 4.50 | 8.50e-3 | 4.8e-4 | 8.16e-4 | 4.58e-4 |
| cb | 4.52 | 1.58e-4 | 3.3e-5 | 2.80e-4 | 3.04e-4 |
| pred | 4.50 | 3.03e-4 | 4.6e-5 | 3.56e-4 | 4.16e-4 |
| predcb | 4.52 | 8.6e-5 | 2.7e-5 | 2.76e-4 | 3.04e-4 |

* Predictor explained variance from the decoded layer-2 latent: 0.965 held-out (pred), 0.972
  (predcb); the codebook then removes 65% of the prediction residual (cb_resid 0.35). The
  codebook alone leaves 1.7% at L3 (cb_resid 0.017).
* Total error vs exact at L3 is dominated by upstream drift from L0-2, where cb / predcb / fp8
  are equal; deep layers are the real test.
* prop_final after 4 layers is a code-path check only (KL 5e-4 to 7e-4, top-1 0.95-0.97).
