# ~4-bit KV-cache methods: survey for the GLM-5.3 MLA latent (2026-09-25)

Compiled by a research sub-agent from papers, repos and framework PRs. Items marked UNVERIFIED rest on abstracts or
secondary sources; framework PR numbers are author-reported and not independently reproduced.

## Most relevant results

1. **vLLM `nvfp4_ds_mla`** (NVIDIA, merged 2026-09-01, SM100): 512-d latent as FP4 E2M1 with E4M3 scale per 16
   values (4.5 b), RoPE FP8; GLM-5.2 AIME25 .927->.938, GPQA .899->.896, AA-LCR .696->.697 vs FP8 KV (64 repeats);
   DeepSeek-V3.2 similar. <https://github.com/vllm-project/vllm/pull/51724>
2. **SAW-INT4** (Together AI): block-Hadamard-128 + per-token/per-head asymmetric INT4 (~4.25 b); GLM-4.7 (GQA)
   77.95 vs 77.89 BF16; naive INT4 collapses (Qwen3-4B 0 vs 73.8 rotated); k-means up to 2048 centroids adds little.
   <https://arxiv.org/html/2604.19157v1>, <https://github.com/togethercomputer/saw-int4>
3. **TensorRT-LLM NVFP4 KV**: Qwen3-Coder-480B vs FP8 MMLU-Pro 78.1->77.4, RULER-64K 95.5->94.6; beats MXFP4 by
   ~4pp MMLU on Llama-3.3-70B. <https://developer.nvidia.com/blog/?p=109878>; LMSYS SGLang NVFP4 KV tests
   <https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache>
4. Counter-example: **SGLang power-of-two-scale FP4** (`fp4_e2m1`, now `fp4_mx_block16`): DeepSeek-R1 AIME25
   .493->.400, GPQA .770->.727 vs FP8. <https://github.com/sgl-project/sglang/pull/10078>

## Other methods (short)

| Method | Bits, granularity | Outliers / sinks | Reported ~4-bit | Impl. | MLA fit |
|---|---|---|---|---|---|
| SGLang DSA FP8 (our served baseline) | 8.25; fp32 scale per 128-tile per token; RoPE BF16 | - | reference | FlashMLA/SGLang | native |
| QuaRot | 4.25, asym, group 128 | Hadamard K,V | Llama-2-7B ppl 5.47->5.51 | research | yes (fold) |
| KVQuant | ~4.3 incl. 1% sparse; keys per-channel pre-RoPE, non-uniform | dense+sparse; first token FP16 | LLaMA-7B 5.68->5.69 | CUDA | grid + sink trick |
| KIVI | ~5.0 at "4-bit"; K per-channel, V per-token | recent 128 tokens FP16 | ~lossless | HF | partial |
| LMDeploy int4 | per-token per-head asym | - | InternLM2.5 CEVAL 78.06->77.05 | TurboMind | no MLA |
| IntactKV | pivot tokens lossless | sinks | gains (abstract) | code | yes |
| TurboQuant | 3.5; rotation + Lloyd-Max | rotation | LongBench = full (Llama-3.1-8B) | vLLM (no QJL) | MSE stage yes |
| CQ | 2 b, coupled channels | Fisher k-means | beats KVQuant-2b | research | yes |
| Lexico | 4k-atom dictionary + OMP | - | 90-95% GSM8K at 15-25% mem | repo | closest to ours |
| AQUA-KV | 2-2.5; cross-layer predictor | - | <1% rel ppl | code | test cross-layer |
| GEAR | 4-bit + low-rank + sparse | sparse | near-lossless 4-bit | code | yes |
| KVTuner | per-layer K/V bits | - | lossless at 3.25/4.0 | offline | per-layer |
| DeepSeek-V4.1-Flash | E2M1 + E4M3/16, no global scale | QAT; SWA KV FP8 | "marginal" | in-house | MLA-like |
| SnapMLA (FP8) | per-token FP8 latent, RoPE BF16 | - | DS-V3.1 AIME-24 93.65 vs 93.85 | research | MLA |

Also: RotateKV, SpinQuant, QServe (SmoothAttention), Atom, ZipCache, MiKV, SKVQ, WKVQuant, QAQ, QJL, PolarQuant,
CommVQ, BitDecoding, llama.cpp q4_0 (DeepSeek ppl +3.0%), MHA2MLA (latent Int4 -0.5% LongBench, UNVERIFIED details).

## Implications for our probe

* Practical bar at 4 bits = NVFP4 on the latent (matches FP8 downstream on GLM-5.2 despite higher MSE); a 4-bit
  scalar codec will not match FP8 MSE without ~1.5 b of coding gain. Target: attention error <= NVFP4 at <= 4.5 b,
  then a downstream check.
* Arms added (commit after 1a57a6ee): SGLang fp8_tile128 / fp4_mx16 (bit-exact vs SGLang reference) / nvfp4,
  Hadamard + NVFP4, Hadamard + INT4 tok, codebook + NVFP4 residual (+/- Hadamard).
* Untested leads: divide out kv_a_layernorm gains before quantizing (fold into W_UK/W_UV); metric-aware whitening
  (W_UK^T Sigma_q W_UK, W_UV^T W_O^T W_O W_UV); per-layer mixed precision; keep sinks + a recent window exact.
