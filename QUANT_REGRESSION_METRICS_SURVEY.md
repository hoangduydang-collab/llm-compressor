# Quantization-Regression Metrics: Literature & Industry Survey

*Compiled 2026-07-21 from a three-track web-research sweep (distributional
fidelity / behavioral degradation / industry practice). Purpose: extend the
official-eval pipeline's metric set beyond {accuracy delta, answer flip-rate,
token-distribution agreement + truncated KL, completion-token spend +
budget-exhaustion rate}. Implementation home: the benchmarks repo
(`quality/general/usage*.py`, distribution suite); this document is the
planning reference.*

*Verification note: numeric findings were extracted from abstracts/HTML
fetches by research subagents, not full PDF reads. Treat exact percentages as
directionally reliable; spot-check the primary PDF before quoting in external
reports. Several arXiv IDs are 2025–2026 publications.*

## The one-line summary

The literature's strongest theme is exactly what we found empirically on
MiniMax-M3 AWQ: **aggregate accuracy hides quantization damage**. The
best-evidenced hidden channels are (a) per-item answer flips at flat aggregate
score [1], (b) token-distribution divergence [1][4][5], and (c)
reasoning-token inflation / non-termination [21][22][23]. Our pipeline
already covers all three; the survey adds established names, reference
numbers, and an adoption shortlist.

## Key external validation of our token-spend metric

- "Quantized Reasoning Models Think They Need to Think Longer, but They Do
  Not" [21]: 3-bit AWQ on MATH-500 drops 85.6%→47.0% while CoT length
  inflates 5.2K→23.4K tokens; up to **52% of quantized-model failures reach
  the right answer inside the CoT but never emit it** (our GPQA empty-answer
  mechanism, published). CoT-length-vs-accuracy ρ = −0.73. A logit penalty on
  overthinking markers cut CoT length 12–23%.
- "Quantization Inflates Reasoning" [22]: defines the **CoT Token Inflation
  Ratio** (quant/FP reasoning-token ratio) — our `token_spend_ratio` is this
  metric — plus semantic-repetition rate in the trace.
- "Quantization Hurts Reasoning?" [23] (COLM 2025): degradation scales with
  task difficulty and response length; W8A8/W4A16 near-lossless, sub-4-bit
  cliffs. Counterpoint: length amplification is method/precision-specific,
  not universal.
- Industry (Red Hat/Neural Magic reasoning-model evals [59][62a]) flags
  thinking-token budget effects but publishes no metric for them — our
  exhausted-rate tracking is ahead of published industry practice here.

## Metric taxonomy (what exists, what it catches, cost)

### Tier 1 — log-only, cheap, adopt/reframe now

| Metric | Sources | Catches | Status for us |
|---|---|---|---|
| **Recovery rate** (quant/base score per task) | Neural Magic convention [59][62a] (>500K evals: 8-bit 99.9%, 4-bit ~98.9% avg recovery); MLPerf 99%/99.9% tiers [62c] | nothing new — the standard reporting/gating frame | reframe our deltas as per-task recovery % in delta/report |
| **Flips / Correctness Agreement** | [1] (flips = correct↔incorrect only, excl. inc→inc; 13.6% flips at 0–2% acc delta; KLD–flips Spearman ≈0.98); [46] (CA) | instance-level churn at flat aggregate | = our planned per-task flip-rate; use the [1] definition, report both directions |
| **CoT Token Inflation Ratio** | [22] | hidden test-time-compute cost | shipped (`token_spend_ratio`) |
| **Non-termination / budget-exhaustion rate** | [21] failure taxonomy | scoring-invisible collapse (our AWQ GPQA case) | shipped (`exhausted_rate`) |
| **Overthinking-marker frequency** ("wait", "alternatively"… in think text) | [21] | rumination distinct from raw length | needs reasoning TEXT stored (we keep only char counts) |
| **seq-rep-n / distinct-n on think text** | [28][26][29]; quant application is a literature gap (closest: [30], which found 4-bit AWQ/GPTQ largely preserves degeneration behavior) | repetition loops inside CoT | needs reasoning text |
| **Right-answer-not-emitted rate** | [21] (≤52% of failures) | termination/format failure vs true reasoning failure | scan think text on failed items; needs reasoning text |

### Tier 2 — needs an extra eval pass or logits

| Metric | Sources | Notes |
|---|---|---|
| **KLD percentile ladder, Same-Top-P, Mean/RMS Δp** | llama.cpp `--kl-divergence` [4]; community interpretation bands [62f]: KLD <1e-4 ≈ identical, 1e-2–5e-2 typical 4-bit, >1e-1 substantial | upgrade for our distribution suite (mean-level stats today). Caution [61]: KLD–flips correlation collapses near baseline ("silent zone") — don't gate on KLD alone when deltas are small |
| **Calibration drift: ECE/ACE + JS divergence** | [8] (quantization lowers confidence on already-uncertain items; direction model-dependent); canonical ECE [7]; estimator criticisms [10][11]; LLM-confidence caveat [12] | computable from MC-task logprobs already collected on the completions path (+ --log_samples) |
| **Length-stratified accuracy (RULER)** | [40]; Red Hat: W4A16 >99.5% recovery ≤64K but 85–88% at 128K [62b]; [41]: up to 59% long-context drop, strongly model-dependent | new task family; relevant to our 65K-ctx serving |
| **Multi-turn delta (MT-Bench turn-2)** | [54]: KV8→KV3 dropped turn-2 judge score ~4× more than turn-1 | damage concentrates in later turns; single-turn QA blind |
| **Self-consistency spread / pass@k gap under T>0** | self-consistency origin [33]; closest quant pairing [34] (code-only; robustness direction NOT uniform — 51.6% of adversarial trials had quant MORE robust); no direct quant study exists (gap) | k reruns; control the backend confound [37] first |
| **Agentic end-to-end success** | ACBench [53] (ICML 2025): 4-bit keeps tool-call syntax (−1–3%) but drops end-to-end agent success 10–15% | expensive rollouts; no BFCL×quant paper exists (gap) |
| **Safety/refusal drift** | [48] (quant mostly preserves bias/toxicity; pruning doesn't); [49] ("quality is not a safety proxy": PPL/ROUGE preservation ≠ refusal/jailbreak preservation) | judge/classifier pass; lower priority for internal-quality goal |
| **Per-item bias/long-tail flips: PIE / CIE** | [50][51] (compression over-indexes on long-tail examples); LLM-era replications [52]: 5–16% per-item flips at <2% aggregate loss | derivable from base-vs-quant outputs alone |

### Tier 3 — white-box (we own the weights; ties to ABI/pre-quant gates, project goal 5)

| Metric | Sources | Notes |
|---|---|---|
| **MoE router stability** (Jaccard of top-k expert sets, FP vs quant) | VSRAQ [38]; EAQuant [39] | directly relevant to M3; candidate static gate. Needs router logits — not API-reachable |
| **Per-layer error propagation / SQNR / weighted RMSE** | error propagation [17]; SQNR in FP8 design [14]; kurtosis targeting [18]; outlier channels [19] | diagnosis of WHERE damage lives; not serving-path |
| **Cosine similarity of hidden states** | used widely [15] but [16] shows weak correlation with real degradation | treat as unreliable proxy |
| **Speculative-decoding acceptance rate** (quant as draft, BF16 as target) | mature vLLM machinery repurposed as a corpus-scale distribution-match scalar (no direct paper — our observation) | one serve with spec-decode gives a single fidelity number |

### Perplexity: keep only as a cheap smoke signal

PPL deltas are dominated by high-confidence tokens and miss decision flips
(llama.cpp maintainers [3]; the GPTQ paper itself pairs PPL with task eval
[2]); aggregate correlation with benchmarks is decent (|r|≈0.79, up to 0.93
on GSM8K [56]) but reliability degrades with input length [6], and
sampling-tail risk can rise while PPL stays flat [5]. Never gate on PPL.

## Adoption shortlist (recommended order)

1. **Recovery-% column** in delta/report (reframing, zero new data) [59][62c].
2. **Per-task flips** ([1] definition) — CPU-only from lm_cache via
   --log_samples replay; report both flip directions.
3. **Store reasoning text** (optional flag in usage capture or probe mode) →
   unlocks overthinking-marker rate, seq-rep-n, right-answer-not-emitted
   [21][22][28] — the three best reasoning-forensics metrics, all log-only.
4. **Distribution-suite upgrade**: Same-Top-P + KLD percentiles + Δp stats
   (llama.cpp vocabulary [4]), silent-zone caveat documented [61].
5. **ECE drift** from MC-task logprobs (data already on disk) [7][8].
6. **MoE router Jaccard** as a white-box gate [38][39] (feeds goal 5).
7. Later: RULER length-stratified pass [40]; multi-turn delta [54];
   spec-decode acceptance-rate probe.

## Cross-cutting cautions from the survey

- **Multi-metric or nothing**: Red Hat's own guidance — "some evaluation
  results will go up and others will go down" [62a]; Fireworks argues MMLU is
  too noisy to rank quant schemes and uses KL instead [62d]. Single-number
  gates mislead.
- **Backend confound** [37]: switching serving backends alone moves reasoning
  accuracy up to 16.6pp. Our paired same-vLLM design is the right control;
  never compare scores across engines.
- **Flips direction ≠ damage**: a large share of flips can be
  incorrect→correct [1]; report both directions.
- **Thresholds**: MLPerf's per-benchmark 99%/99.9% tiers [62c] are the only
  formal gate convention; Neural Magic's ≥99% recovery [59] is self-reported
  convention. If we formalize a ship gate, state it MLPerf-style (per-task,
  documented, reproducible).
- **Hard cliffs exist**: KV-cache INT2 collapses HumanEval to 0% while
  INT4/INT8 show no real loss [55]; 2-bit AWQ degenerates toward naive RTN
  [32]. Sweeps must include the cliff region, not interpolate across it.

## References

Papers (arXiv):

1. A. Dutta, S. Krishnan, N. Kwatra, R. Ramjee (Microsoft Research),
   "Accuracy is Not All You Need," 2024. arXiv:2407.09141.
   https://arxiv.org/abs/2407.09141
2. E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh, "GPTQ: Accurate
   Post-Training Quantization for Generative Pre-trained Transformers," 2022.
   arXiv:2210.17323. https://arxiv.org/abs/2210.17323
3. llama.cpp maintainer discussion on perplexity's limits for quant quality,
   ggml-org/llama.cpp discussion #4110.
   https://github.com/ggml-org/llama.cpp/discussions/4110
4. llama.cpp KL-divergence tooling (`llama-perplexity --kl-divergence`):
   PR #5076 (ikawrakow) and `tools/perplexity/perplexity.cpp`. Reports mean
   KLD ± unc., percentile ladder, Mean/RMS Δp, Same-Top-P, PPL ratio.
   https://github.com/ggerganov/llama.cpp/pull/5076
5. "ReQAT" (tail-mass inflation under quantization; ρ tail-mass ratio), 2026.
   arXiv:2606.15682. https://arxiv.org/abs/2606.15682
6. "Rethinking Perplexity" (PPL reliability deteriorates with input length),
   2026. arXiv:2602.04099. https://arxiv.org/abs/2602.04099
7. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On Calibration of Modern
   Neural Networks," 2017. arXiv:1706.04599. https://arxiv.org/abs/1706.04599
8. "When Quantization Affects Confidence of Large Language Models?," 2024.
   arXiv:2405.00632. https://arxiv.org/abs/2405.00632
9. Calibration of quantized LLMs via Adaptive Calibration Error, 2025.
   arXiv:2508.16785. https://arxiv.org/abs/2508.16785
10. J. Nixon et al., "Measuring Calibration in Deep Learning" (ACE; ECE
    binning sensitivity), 2019. arXiv:1904.01685.
    https://arxiv.org/abs/1904.01685
11. A. Kumar et al., "Verified Uncertainty Calibration" (plug-in ECE bias),
    2019. arXiv:1909.10155. https://arxiv.org/abs/1909.10155
12. Chat-LLM softmax confidence is overconfident / weakly correlated with
    accuracy, 2024. arXiv:2402.13213. https://arxiv.org/abs/2402.13213
13. Pruning harms calibration, quantization near-neutral (pre-LLM setting),
    2023. arXiv:2308.14969. https://arxiv.org/abs/2308.14969
14. "FP8 Quantization: The Power of the Exponent" (SQNR-driven format
    design), 2022. arXiv:2208.09225. https://arxiv.org/abs/2208.09225
15. Cosine-similarity layer-importance metrics for compression:
    arXiv:2403.19135 (2024), arXiv:2603.17354 (2026).
16. Cosine similarity correlates weakly with actual degradation, 2026.
    arXiv:2605.14075. https://arxiv.org/abs/2605.14075
17. "Quantization Error Propagation" (per-layer error compounding in PTQ),
    2025. arXiv:2504.09629. https://arxiv.org/abs/2504.09629
18. "KurTail" (kurtosis-targeted rotation for quantization), 2025.
    arXiv:2503.01483. https://arxiv.org/abs/2503.01483
19. G. Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
    Quantization for LLMs," 2022. arXiv:2211.10438.
    https://arxiv.org/abs/2211.10438
20. Z. Liu et al., "SpinQuant: LLM Quantization with Learned Rotations,"
    2024. arXiv:2405.16406. https://arxiv.org/abs/2405.16406
21. "Quantized Reasoning Models Think They Need to Think Longer, but They Do
    Not," 2026. arXiv:2606.00206. https://arxiv.org/abs/2606.00206
22. "Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of
    Low-Bit Reasoning Models," 2026. arXiv:2606.25519.
    https://arxiv.org/abs/2606.25519
23. "Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning
    Models," COLM 2025. arXiv:2504.04823. https://arxiv.org/abs/2504.04823
24. Trace-level 2-bit failure modes (repetitive loops, delayed commitment,
    unclosed segments), 2026. arXiv:2606.02011.
    https://arxiv.org/abs/2606.02011
25. "Quantization Meets Reasoning" (step-aligned CoT error taxonomy), 2025.
    arXiv:2505.11574. https://arxiv.org/abs/2505.11574
26. J. Li et al., "A Diversity-Promoting Objective Function for Neural
    Conversation Models" (distinct-n), 2015. arXiv:1510.03055.
    https://arxiv.org/abs/1510.03055
27. Y. Zhu et al., "Texygen: A Benchmarking Platform for Text Generation
    Models" (self-BLEU), 2018. arXiv:1802.01886.
    https://arxiv.org/abs/1802.01886
28. S. Welleck et al., "Neural Text Generation with Unlikelihood Training"
    (seq-rep-n), 2019. arXiv:1908.04319. https://arxiv.org/abs/1908.04319
29. A. Holtzman et al., "The Curious Case of Neural Text Degeneration," 2019.
    arXiv:1904.09751. https://arxiv.org/abs/1904.09751
30. Correlation-dimension degeneration detector under quantization
    (mean abs. change 0.14 for 4-bit AWQ/GPTQ), NeurIPS 2025.
    arXiv:2510.21258. https://arxiv.org/abs/2510.21258
31. "LFQ" (block-PTQ matches PPL but degrades long generative/CoT tasks;
    logit-distribution misalignment), 2026. arXiv:2605.29756.
    https://arxiv.org/abs/2605.29756
32. PTQ-Bench (cross-bitwidth/architecture/modality robustness; 3-bit
    practical floor; 2-bit AWQ→RTN collapse), 2025. arXiv:2502.13178.
    https://arxiv.org/abs/2502.13178
33. X. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in
    Language Models," 2022. arXiv:2203.11171.
    https://arxiv.org/abs/2203.11171
34. "Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code
    Generation" (RRS; robustness direction not uniform), 2025.
    arXiv:2506.22776. https://arxiv.org/abs/2506.22776
35. K. Zhu et al., "PromptRobust: Towards Evaluating the Robustness of LLMs
    on Adversarial Prompts," 2023. arXiv:2306.04528.
    https://arxiv.org/abs/2306.04528
36. Quantization vs adversarial/natural-corruption robustness (opposite
    signs): arXiv:2304.03968 (2023); RobustMQ, arXiv:2308.02350 (2023).
37. Backend-swap confound (engine change alone shifts DeepSeek-R1/GSM8K
    accuracy up to 16.6pp; Disagreement Rate / Divergence Index / LogProb
    RMSE), 2026. arXiv:2605.19537. https://arxiv.org/abs/2605.19537
38. "VSRAQ" (MoE router stability, Jaccard of top-k expert sets), 2026.
    arXiv:2606.05688. https://arxiv.org/abs/2606.05688
39. "EAQuant" (expert-aware MoE quantization), 2025. arXiv:2506.13329.
    https://arxiv.org/abs/2506.13329
40. C.-P. Hsieh et al. (NVIDIA), "RULER: What's the Real Context Size of
    Your Long-Context Language Models?," 2024. arXiv:2404.06654.
    https://arxiv.org/abs/2404.06654
41. 4-bit long-context degradation (up to 59%, non-English worse,
    model-dependent), 2025. arXiv:2505.20276.
    https://arxiv.org/abs/2505.20276
42. KV-cache quantization eval line: KIVI arXiv:2402.02750; GEAR
    arXiv:2403.05527 (explicit autoregressive error-compounding claim); QAQ
    arXiv:2403.04643; Coupled Quantization arXiv:2405.03917; KVarN
    arXiv:2606.03458 (retrieval 100%→85% over 100→600 lines at 2-bit KV).
43. "The Quantization Trap" (multi-step GSM8K loss ~doubles from 1–2-step to
    3–4-step under 4-bit), 2026. arXiv:2602.13595.
    https://arxiv.org/abs/2602.13595
44. J. Zhou et al., "Instruction-Following Evaluation for Large Language
    Models" (IFEval), 2023. arXiv:2311.07911.
    https://arxiv.org/abs/2311.07911
45. "A Comprehensive Evaluation of Quantized Instruction-Tuned LLMs up to
    405B," 2024. arXiv:2409.11055 (IFEval disproportionately
    quantization-sensitive). https://arxiv.org/abs/2409.11055
46. "The Illusion of Equivalency" (Correctness Agreement), 2026.
    arXiv:2607.08734. https://arxiv.org/abs/2607.08734
47. "What Makes Quantization for Large Language Models Hard?," AAAI 2024.
    arXiv:2403.06408. https://arxiv.org/abs/2403.06408
48. "Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM
    Compression," EMNLP Findings 2024. arXiv:2407.04965.
    https://arxiv.org/abs/2407.04965
49. "Quality Is Not a Safety Proxy Under Quantization" (refusal/jailbreak
    metrics vs quality metrics), 2026. arXiv:2606.10154.
    https://arxiv.org/abs/2606.10154
50. S. Hooker et al., "What Do Compressed Deep Neural Networks Forget?" (PIE),
    2019. arXiv:1911.05248. https://arxiv.org/abs/1911.05248
51. S. Hooker et al., "Characterising Bias in Compressed Models" (CIE), 2020.
    arXiv:2010.03058. https://arxiv.org/abs/2010.03058
52. LLM-era per-item flip / bias-flip replications: arXiv:2605.15208,
    arXiv:2605.08137, arXiv:2509.15206, arXiv:2602.06181 (5–16% per-item
    flips at <2% aggregate loss; ~21% bias-classification flips at flat
    aggregate bias score).
53. "ACBench: Benchmarking Agent Capabilities of Compressed LLMs," ICML 2025.
    arXiv:2505.19433. https://arxiv.org/abs/2505.19433
54. S. Li et al., "Evaluating Quantized Large Language Models," ICML 2024.
    arXiv:2402.18158 (5-dimension taxonomy: basic NLP, emergent abilities,
    trustworthiness, dialogue, long-context; MT-Bench turn-2 finding).
    https://arxiv.org/abs/2402.18158
55. "LLMC: Benchmarking Large Language Model Quantization" (calibration ×
    algorithm × format; KV-INT2 HumanEval collapse; kurtosis/cosine/KL
    diagnostics), 2024. arXiv:2405.06001. https://arxiv.org/abs/2405.06001
56. "A Comprehensive Evaluation of Quantization Strategies for Large Language
    Models," ACL Findings 2024. arXiv:2402.16775 (PPL–benchmark |r|≈0.79;
    bias direction inconsistent). https://arxiv.org/abs/2402.16775
57. "On the Generalization Ability of Quantized LLMs" (calibration-set
    distribution match not always optimal), 2024. arXiv:2406.12928.
    https://arxiv.org/abs/2406.12928
58. INT4 vs MXFP4/NVFP4 (rotation/scaling tricks don't transfer to FP4),
    2025. arXiv:2507.17417. https://arxiv.org/abs/2507.17417
59. E. Kurtić et al. (Neural Magic), "'Give Me BF16 or Give Me Death'?
    Accuracy-Performance Trade-Offs in LLM Quantization," 2024.
    arXiv:2411.02355. https://arxiv.org/abs/2411.02355
60. P. Liang et al., "Holistic Evaluation of Language Models" (HELM), 2022.
    arXiv:2211.09110; https://crfm.stanford.edu/helm — 7 axes (accuracy,
    calibration incl. Platt-scaled variants, robustness, fairness, bias,
    toxicity, efficiency incl. observed/denoised/idealized runtime).
61. KLD "silent zone" — KLD–flips correlation collapses near baseline, 2026.
    arXiv:2606.19558. https://arxiv.org/abs/2606.19558

Web / industry sources:

- 62a. Red Hat Developer, "We ran over half a million evaluations on
  quantized LLMs" (Oct 2024) — ≥99% avg recovery claim; and the Aug 2025
  follow-up ("holistic view... some results go up and others down"); and the
  DeepSeek-R1-Distill quantization eval (Mar 2025).
  https://developers.redhat.com/articles/2024/10/17/
- 62b. Red Hat Developer, long-context (RULER) quantized-Llama-3.1 study —
  W4A16 recovery >99.5% ≤64K, 85–88% at 128K.
- 62c. MLCommons, MLPerf Inference Rules
  (`mlcommons/inference_policies/inference_rules.adoc`) — per-benchmark
  99%/99.9%-of-reference accuracy tiers; generation-length compliance ±10%;
  quantization permitted if reproducible + calibration-data-restricted.
  https://github.com/mlcommons/inference_policies
- 62d. Fireworks AI, "Quantization" blog — argues MMLU too noisy to rank
  quant schemes; uses KL divergence.
  https://fireworks.ai/blog/fireworks-quantization
- 62e. Together AI, "Together Inference Engine 2" — HELM (1000×3 trials, 17
  tasks) + AlpacaEval 2.0 to support FP8 quality claims.
  https://together.ai/blog/together-inference-engine-2
- 62f. smcleod.net (2026), KLD interpretation bands for GGUF quants.
- 62g. Unsloth, "Dynamic 2.0 GGUFs" docs — custom 5-shot MMLU harness
  (±0.1pt of official) + before/after KLD tables.
  https://unsloth.ai/docs (dynamic GGUF methodology)
- 62h. SambaNova, "Does reduced precision hurt?" — 15 Eval-Gauntlet tasks +
  HumanEval/MBPP/AlpacaEval vs Groq; up to 9pp CoQA delta claim.
  https://sambanova.ai/blog/does-reduced-precision-hurt
- 62i. NVIDIA TensorRT-LLM docs — per-model MMLU deltas for FP8 (e.g.
  Falcon-180B 70.3 vs 70.4).
- 62j. llama.cpp community Bradley–Terry human-preference benchmark for quant
  levels, ggml-org/llama.cpp discussion #5962.
  https://github.com/ggml-org/llama.cpp/discussions/5962
