# KV-cache anchor probe — GLM-5.3 MLA latent, layers 0–3, AA-LCR text

**Verdict:** positional (WebP / neighbour) prediction is dead — the latent has no
temporal correlation. Content-based anchors (nearest of 4096 k-means centroids)
are strong: codebook + INT2 at 2.15 bits/value matches plain INT4 (4.125) on
attention-output error in layers 0, 2, 3; codebook + INT4 (~4.15) matches
per-token-scaled FP8 (8.03). **Early layers only** — deep layers are the open
question.

- Model: `zai-org/GLM-5.3-BF16` snapshot `304b8051…`, depth-truncated to layers
  0–3 (`build_subset_checkpoint --layers 4`, 24.17 GiB), transformers 5.12.1,
  eager attention. Config: kv_lora_rank 512, rope 64, 64 heads, index_topk 2048,
  indexer_types[0:4] = full, full, full, shared.
- Data: AA-LCR v1.1 extracted text (pinned, sha-checked by `aa_lcr_v11`), one
  4096-token window per document set at offset 1024; 15 sets fit the global
  mean + codebooks, the other 15 are evaluated. DSA clustering on 4 eval docs
  × 32,768 tokens.
- Script `pipeline/kv_anchor_probe.py`, job `pipeline/k8s/hd-kv-anchor-glm53-early.yaml`,
  1× H100 on ca-gpu06, 29 min. Raw: `/mnt/cephfs/hoangduy/results/kv-anchor/glm53-L0-3-20260925t081504z/`.
- Gates passed: tiny-model preflight (real layer types, real save/load path);
  re-running each layer's attention with its own latent reproduces the output
  exactly (0.0); chunked layer-0 indexer matches the module (overlap 1.0).

## 1. No temporal structure

| | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| lag-1 corr (centred) | −0.039 | −0.049 | −0.151 | −0.036 |
| lag 2–256 corr, max \|ρ\| | 0.019 | 0.023 | 0.017 | 0.034 |
| first-token anchor, N=16: residual / gmean residual | 1.84 | 1.84 | 1.87 | 1.83 |
| chunk-mean anchor, N=4 (1−1/N = 0.75) | 0.76 | 0.76 | 0.81 | 0.75 |

Token-shuffled control reproduces every chunk-anchor number. The chunk-mean
"gain" is the self-inclusion artefact (1−1/N), not prediction.

## 2. Content anchors

| residual / gmean residual | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| cb256 | 0.456 | 0.210 | 0.101 | 0.035 |
| cb4096 | 0.180 | 0.072 | 0.043 | 0.017 |
| rank1 per 16-token chunk (+fp16 α) | 0.890 | 0.747 | 0.437 | 0.219 |

Attention-output relative MSE (best scale grouping per arm):

| bits/value | arm | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| 2.125 | direct INT2 | 0.097% | 1.83% | 4.78% | 5.14% |
| 2.148 | cb4096 + INT2 | 0.001% | 0.044% | 0.053% | 0.056% |
| 3.125 | direct INT3 | 0.010% | 0.151% | 0.629% | 0.632% |
| 3.148 | cb4096 + INT3 | 0.000% | 0.008% | 0.008% | 0.011% |
| 4.125 | direct INT4 | 0.002% | 0.023% | 0.052% | 0.086% |
| 4.148 | cb4096 + INT4 | 0.000% | 0.002% | 0.002% | 0.003% |
| 8.000 | fp8 unscaled cast | 0.003% | 0.162% | 0.107% | 0.091% |
| 8.031 | fp8, per-token fp16 scale | 0.000% | 0.002% | 0.002% | 0.004% |

Overheads counted: residual scales (16 bits / 128 values), 12-bit index per
token, codebook 4096×512 fp16 = 4 MB per layer (not in bits/value; ~0.3 GB for
78 layers).

## 3. DSA selection clustering (layer 0, 32k context)

Distinct chunks touched by the top-2048 selection, relative to a uniformly
random selection of the same size: 0.80–0.86 (N=16), 0.90–0.94 (N=64). Barely
clustered — positional anchors would pay extra reads; codebook anchors do not
(shared table, per-token index).

## Caveats

- Layers 0–3 only. Layer 0's latent is a function of the current token alone;
  clustering *strengthens* through layer 3 but deep, contextual layers are untested.
- 4096-token windows: DSA's top-2048 covers about half the context, so attention
  errors are from a near-dense regime.
- Only the 512-d latent is compressed; the 64-d RoPE key part is untouched.
- "fp8 unscaled" is a plain e4m3 cast; SGLang's actual fp8 KV scaling was not checked.
