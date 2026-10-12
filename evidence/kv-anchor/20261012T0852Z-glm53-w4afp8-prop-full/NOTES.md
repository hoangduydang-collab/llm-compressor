# KV codecs with propagated error, all 78 layers — GLM-5.3 W4AFP8

Two SkyPilot managed jobs on test-ca-01, 1x H100 each, image lmsysorg/sglang:v0.5.17,
`--weights w4afp8 --act-fp8 --gen-train 8 --propagate`; 23 train (15 AA-LCR + 8 generated) /
15 AA-LCR eval + 8 generated held-out windows x 4096 tokens.

| job | arms | code | output (/cephfs/hoangduy/kv-prop/...) | wall | peak GPU |
|---|---|---|---|---|---|
| 24 | fp8, nvfp4, cb, pred, predcb | 847511c3 | glm53-w4afp8-prop-full-20261012t085211z | 3 h 33 min | 57.7 GB |
| 26 | fp8, ulp, int2, cb2, pred2, predcb2 | 9928236b | glm53-w4afp8-prop-2bit-20261012t115821z | 3 h 37 min | 56.9 GB |

fp8 reproduces bit-for-bit across the two jobs (KL 0.0152 / top-1 0.9474 / ppl 4.642 on eval,
0.0166 / 0.9460 / 2.140 on gen), so the runs can be compared directly.

## End of model (after layer 77): next-token KL vs the exact stream

`ulp` (+1 BF16 ulp on 1% of latent values, latent error ~1e-7) measures how much the model itself
amplifies a negligible KV perturbation (MoE top-8 routing, DSA top-k, FP8 activation rounding).
Excess = KL - floor; floor = ulp (eval 0.0152; gen 0.0171 = mean of ulp 0.0176 and fp8 0.0166,
whose gap, 0.001, is a rough noise scale for these KLs).

| arm | bits | KL eval | excess eval | KL gen | excess gen | top-1 eval | ppl eval (exact 4.643) | ppl gen (exact 2.140) |
|---|---|---|---|---|---|---|---|---|
| fp8 | 8.25 | 0.0152 | 0.000 | 0.0166 | ~0 | 0.947 | 4.642 | 2.140 |
| ulp | 16 | 0.0152 | (floor) | 0.0176 | (floor) | 0.946 | 4.642 | 2.141 |
| predcb | 4.52 | 0.0173 | 0.0021 | 0.0197 | 0.0026 | 0.943 | 4.642 | 2.142 |
| cb | 4.52 | 0.0175 | 0.0023 | 0.0202 | 0.0031 | 0.942 | 4.653 | 2.147 |
| pred | 4.50 | 0.0194 | 0.0042 | 0.0217 | 0.0046 | 0.940 | 4.644 | 2.146 |
| nvfp4 | 4.50 | 0.0204 | 0.0052 | 0.0233 | 0.0062 | 0.939 | 4.647 | 2.146 |
| predcb2 | 2.16 | 0.0576 | 0.042 | 0.0751 | 0.058 | 0.899 | 4.705 | 2.231 |
| cb2 | 2.16 | 0.0750 | 0.060 | 0.1076 | 0.090 | 0.887 | 4.767 | 2.283 |
| pred2 | 2.13 | 0.1259 | 0.111 | 0.1431 | 0.126 | 0.858 | 4.871 | 2.393 |
| int2 | 2.13 | 0.1742 | 0.159 | 0.1943 | 0.177 | 0.837 | 5.076 | 2.574 |

* Served FP8 KV is indistinguishable from the floor: its KL is all model amplification.
* ~4 bit: predcb's excess KL is ~2.5x below NVFP4's (eval 0.0021 vs 0.0052, gen 0.0026 vs 0.0062).
  predcb vs cb (0.0021 vs 0.0023, 0.0026 vs 0.0031) is within the ~0.001 noise scale: not a
  demonstrated gain at 4 bits.
* 2 bit: the combination is clearly complementary. predcb2's excess is 30% / 36% below cb2's,
  2.6x / 2.2x below pred2's and 3.8x / 3.1x below plain INT2's; perplexity +1.3% eval / +4.3% gen
  (cb2 +2.7% / +6.7%, int2 +9.3% / +20%). Still ~8x NVFP4's excess, so 2-bit is insight only.
* No confidence intervals: one pass over ~94k held-out tokens per split, no bootstrap.

## Per layer: attention-output rel-MSE vs the exact stream (AA-LCR eval, from the job logs)

| L | type | fp8 | ulp | nvfp4 | cb | pred | predcb | int2 | cb2 | pred2 | predcb2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | full | 4.7e-6 | 1.2e-6 | 5.9e-5 | 3.0e-6 | 5.5e-5 | 2.9e-6 | 4.0e-2 | 2.1e-5 | 4.0e-2 | 2.0e-5 |
| 7 | shared | 1.30e-3 | 1.26e-3 | 3.59e-3 | 1.92e-3 | 2.22e-3 | 1.66e-3 | 0.105 | 1.63e-2 | 5.2e-2 | 1.03e-2 |
| 15 | shared | 8.5e-3 | 8.7e-3 | 1.50e-2 | 1.05e-2 | 1.12e-2 | 9.9e-3 | 0.160 | 0.070 | 0.105 | 0.052 |
| 47 | shared | 0.118 | 0.119 | 0.145 | 0.131 | 0.136 | 0.127 | 0.527 | 0.345 | 0.468 | 0.281 |
| 63 | shared | 0.098 | 0.098 | 0.125 | 0.111 | 0.113 | 0.107 | 0.482 | 0.326 | 0.402 | 0.255 |
| 77 | shared | 3.16e-2 | 3.14e-2 | 4.56e-2 | 3.77e-2 | 4.08e-2 | 3.64e-2 | 0.302 | 0.122 | 0.202 | 0.107 |

* Local (this layer's codec only) FP8 is 12-28x better than NVFP4 at every depth, but the
  accumulated total is only 1.3-3x apart: latent drift entering each layer is common to all arms
  (job 25 read-out, layers 3-49). ulp == fp8 to 2-3 digits from L7 on.
* DSA top-k overlap vs exact falls from ~0.998 (L6) to ~0.98 (L38) for every arm.
* At L3 the predictor explains 96.5% of held-out latent variance; mid-depth pred alone ties cb
  (e.g. L27-29 at 4 bits, L25/28/29 at 2 bits).

Per-layer JSONs (local error, drift, predictor explained variance, codebook residual, top-k
overlap) are in `prop_layers/` of each output directory; `prop_final.json` holds the table above.
