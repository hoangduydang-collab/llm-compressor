# Aug 31 – Sep 4

## Duy

### What I worked on

**GLM-5.3 W4AFP8, end to end.** Our own quantization of GLM-5.3, from the BF16
release through to a checkpoint SGLang actually serves, plus the paired quality
evaluation against the only independent W4AFP8 release of this model. This is the
day-0 capability we set out to have for GLM-5.3, and it is the first time our
pipeline has produced a serveable checkpoint on a family other than MiniMax-M3.

**The evaluation below is our in-house instrument, by design.** It is cheap, it is
paired, and it answers the question we actually needed answered this week — is our
checkpoint as good as the alternative. It was never meant to produce publishable
absolutes. Getting those is next week's job, and it needs a different harness and
considerably more eval time.

Also wrote the collaborator handoff so anyone on the eval side can host either
checkpoint without me — `docs/glm53-w4afp8-rancher-evaluation.md`.

### Quantization report — GLM-5.3

**Format: W4AFP8**, 394.6 GB.

- Routed experts (57,600 projections) **int4**; attention/MLA, shared experts and
  dense MLPs **block-FP8** 128×128; router, `lm_head`, embeddings and norms at
  source precision. MTP layer 78 grafted from BF16.
- **No scope divergence from upstream.** We had meant to keep the DSA indexer at
  source precision, but SGLang cannot serve that, so the conversion puts `wk` /
  `wq_b` back to FP8 — exactly Z.ai's own split. Cost: one unloadable conversion
  plus a 19-shard repatch.

**Method: AWQ**, our llm-compressor fork, 8×H100 on one node.

- Source is `zai-org/GLM-5.3-BF16` (1.51 TB, 282 shards) — AWQ needs BF16 to
  compute smoothing scales. Calibration `ultrachat_200k`, **256 × 2048 tokens**,
  8× the canonical AWQ recipe, all experts calibrated.
- Scope is pinned by a gate that diffs our recipe against the vendor's released
  weight index, so "are we quantizing the right components" now has an answer
  rather than an intention.

**Time: ~20.5 h for the quantization run** (job start 2026-08-28 22:44:53Z →
checkpoint written 2026-08-29 19:20Z), then ~3.5 h to make it serveable.

| Component | Wall | Basis |
|---|---|---|
| Load + dispatch | ~1 h | estimate |
| Calibration walk, 78 layers | ~13–15 h | estimate (dominant) |
| Save, 394.6 GB / 8 shards | ~4 h | estimate |
| **Quantization total** | **~20.5 h** | **measured** |
| Offline post-save gate replay | ~30 min | estimate |
| SGLang W4AFP8 conversion (→ 40 shards) | 2 h 26 m | measured |
| DSA-indexer FP8 repatch, 19 of 40 shards | ~32 min | measured |
| MTP layer-78 graft (hardlinked, +5.35 GB) | ~15 min | estimate |

- Only the quantization total was logged for this run; the three phase rows are
  apportioned from GLM-5.2 measurements on the same storage.
- **Storage-bound, not compute-bound** — CephFS reads at 31 MB/s single-stream
  against 1632 MB/s local NVMe. Calibration volume is not the lever (doubling
  tokens costs 1.85×); the load and the save are.
- **The job reported `Failed` and the work had survived** — the kubelet was lost
  for 3h27m while containerd kept running, and the save finished an hour before
  the kill. Replaying the rank-0 tail offline recovered all ~20.5 h.

### Key results — quality evaluation (the highlight)

Full seven-task suite, both arms same node, same harness, same protocol — only
the weights differ. Peer arm is **PhalaCloud/GLM-5.3-W4AFP8**, an independent
third-party quantization of the same model.

| Task | Ours (W4AFP8) | PhalaCloud (W4AFP8) |
|---|---:|---:|
| GSM8K exact match | **97.65%** | 97.19% |
| IFEval strict prompt | 89.65% | **90.76%** |
| GPQA Diamond CoT | **61.11%** | 55.56% |
| MMLU | **86.67%** | 86.66% |
| ARC Challenge acc_norm | 68.77% | **69.80%** |
| HellaSwag acc_norm | **89.37%** | 89.29% |
| TruthfulQA MC2 | **62.99%** | 62.50% |

- **Ours is at least as good as the independent release on 5 of 7 tasks**, and the
  two it loses are within ~1 point. No task shows a regression pattern.
- Numerics agree with that: int4 dequant residual median **0.102** against our
  0.40 bound (min 0.074, max 0.108), and **no `up_proj` anomaly** — it is the
  *best* of the three projections, once the residual check fits the AWQ fold
  correctly.
- Against BF16 over a 4-layer slice: top-1 agreement **0.875**, mean |Δlogprob|
  **0.050** — quantization noise, not an encoding fault.
- All three post-save gates pass: smooth-fold (75 MoE layers, 297 attention-side
  consumers), quant-verify (57,600 int4 Linears exact; FP8 attn 390 / shared 225 /
  dense 9), serve-ignore (58,224 modules, no shadowing).
- It serves: 394 GB loads at tp=8 in ~37 min off CephFS with `W4AFp8MoEMethod`
  active, reasoning parsing and logprobs both verified.

**What this instrument does and does not do — known from the start:**

- **Good for the go/no-go, not publishable as absolutes.** The protocols are ours,
  not any public recipe's (GPQA is one greedy sample, not avg@k) — which is what
  buys a fast, paired, same-node read.
- **Peer-vs-peer, not quant-vs-original.** No BF16 or FP8 GLM-5.3 baseline fits an
  8×H100 node, so a defect shared by both arms would be invisible.
- **Long-context is untested** — the whole suite is short-context, and the FP8 DSA
  indexer is what a long-context test would stress. AA-LCR closes this.
- An earlier in-house run's 33.51% MMLU was the harness templating loglikelihood
  prompts, not the model. Fixed; the table above is the corrected run.

### Plan for next week

**Rerun quality on AA methodology** — AA's own prompt, repeats and extraction, run
through NVIDIA's packaged loader — so the scores sit next to AA's published GLM-5.3
figures. It costs far more eval time than this week's suite, which is why the cheap
instrument went first.

- **GPQA Diamond, AA methodology**: 198 items × 5 repeats, pass@1, AA reasoning
  decode. Compare against AA's published GLM-5.3 (max) ~91.7%.
- **AA-LCR**, closing the long-context gap the short-context suite leaves — and the
  first real test of the FP8 DSA indexer.
- **BFCL** (function calling) is **Zhou Yu's**, who already has an agentic
  benchmark running — we supply both endpoints and the serve recipe, not a second
  harness.
- **Triage the remaining six tasks** for AA coverage; anything without it stays
  labelled internal-paired.
- **Enable `log_samples`** so the paired arms can measure flip rate, the sharpest
  instrument we have and missing last time.

**Constraints:** one 8-GPU node, ~37 min of load per arm, and thinking traces up to
128K output — so ours-then-PhalaCloud is sequential, a two-day item per benchmark.
Reports get labelled AA *methodology*, not an Artificial Analysis listing.
