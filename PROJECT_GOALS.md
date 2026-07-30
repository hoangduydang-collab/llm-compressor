# Project goals — MiniMax-M3 quantization & evaluation

This is the durable, repo-authoritative record of the project's long-term goals so
any fresh planner/executor/full-stack agent shares the same north star. Task-level
handoffs (`*_HANDOFF.md`, `*_PLAN.md`) carry the current work; this file carries
*why*. Keep the status markers current; do not delete a goal when it completes —
mark it `DONE` with a pointer to the evidence.

Last reviewed: 2026-07-29.

## Long-term goals

1. **Fast parallel quantization** — *Work in progress.*
   Speed up quantization with parallelization for **AWQ and GPTQ**, targeting a
   **full quantization process in 4–8 hours**. See `M3_QUANT_SPEEDUP_PLAN.md` and
   the distributed/multi-GPU calibration path (llm-compressor multi-GPU + GPTQModel;
   do not rebuild bespoke expert-parallel code — see the CLAUDE.md worked example).

2. **Complete the evaluation pipeline** — *Work in progress.*
   A pipeline to compare our **in-house quantized model** against **other existing
   quantized models** and the **original unquantized (BF16) baseline**. Covers the
   paired GPTQ-vs-AWQ reasoning eval, the BF16 companion baseline, fail-closed
   harness/gate contracts, and honest raw-evidence returns. See
   `M3_PRODUCTION_EVAL_HANDOFF.md` and the plans/specs under `docs/superpowers/`.

3. **Working AWQ quantized model** — *DONE (2026-07-20); quality-competitive as of
   2026-07-23 (r6) on the two measured tasks.*
   **Serve `r6` or `r7`, not `r5`.** Two separate defects were fixed in sequence:

   - *Corruption (r4 → r5, 2026-07-20).* An AWQ smoothing-scale degeneracy on dead
     norm channels (M3 layers 8/10–13 have a post-attention norm gain of exactly
     −1.0); fixed in `_grid_search_scales` (e87eef77, upstream-candidate) with
     regression tests and a hardened fail-closed gate suite. r5 evidence
     (512×2048): smooth-fold gate OK on all 57 layers, quant-verify + risk-layer
     dequant sampling OK, coherent TP8 vLLM serving, passing official smoke eval
     (`ready_for_production: true`, 0 empty outputs / 0 loops). Evidence:
     `results/m3-quality/20260720T134946Z-m3-inhouse-awq-r5-smoke/`.
   - *Reasoning non-termination (r5 → r6/r7, 2026-07-23/24).* r5 then failed the
     **paired 64k quality eval**: GPQA recovery 71.7%, token spend 2.19×, budget
     exhaustion 38.9% vs a 12.6% BF16 floor — runaway thinking loops that burn the
     budget and emit nothing. Root cause was AWQ folding smoothing scales through
     M3's **up→down** path, which is not function-preserving across the clamped GLU
     `(clamp(up)+1)·glu`. `r6` drops that fold; `r7` reintroduces down-side
     smoothing through the gate path, where it is function-preserving. Both
     restore near-baseline behaviour: GPQA recovery **98.7% / 104.4%**, IFEval
     **98.6% / 95.7%**, spend 1.13×/0.93× and 1.17×/1.25×, exhaustion
     14.7%/10.6%. Under plain greedy, r6's non-termination on the r5 exhausted
     subset is 46% — better than r5 achieved *with* sampling (49%). Evidence:
     `results/m3-official-quality/20260723T153532Z-tok64k-awqr6/`,
     `…/20260724T053810Z-tok64k-awqr7/`,
     `results/m3-sampling-probe/20260723T125813Z-r6/`.

   Checkpoints and per-arm provenance: `docs/m3-benchmark-arms.md`
   (r5 is retained there as historical — do not serve it). Post-mortems in
   `BUGS_AND_FIXES.md`; collaborator-facing writeup in
   `M3_OFFICIAL_QUALITY_RESULTS.html`.

   **Remaining step:** the r6/r7 evals cover GPQA-Diamond and IFEval only. A
   seven-task breadth run (the `full4` protocol) has only ever been done for r5, so
   in-house GPTQ still holds the sole full-breadth shipping verdict. Closing that is
   goal-2 territory.

4. **Generalize to any quantization method** — *Future work.*
   Extend the pipeline beyond AWQ and GPTQ to arbitrary quantization methods.

5. **Generalize the gates to any model family** — *Future work.*
   Generalize the serving-ABI gate and the pre-quantization static gate so they
   apply to any model family, not just MiniMax-M3.

6. **Packed NVFP4 W4A8 fallback on Hopper** — *Planned: benchmark-gated.*
   Enable officially released NVFP4 checkpoints to run on H100/H200 with packed
   four-bit weights in GPU memory, dynamic FP8 activations, in-kernel E2M1-to-FP8
   conversion and scaling, FP8 Tensor Core multiplication, FP32 accumulation,
   and BF16 output. Build on the existing vLLM NVFP4/Marlin loaders and NVIDIA
   CUTLASS SM90 mixed-input WGMMA machinery rather than writing a kernel from
   scratch. First compare a dense proof of concept against upstream NVFP4 W4A16
   Marlin and load-expanded W8A8; proceed to MoE production work only if the
   dense path passes correctness and target-workload throughput gates. See
   `docs/superpowers/specs/2026-07-19-hopper-packed-nvfp4-w4a8-fallback-design.md`.

7. **Native Humming W4A8 serving on Hopper** — *DONE (2026-07-26).*
   Humming's existing GPTQ W4A8 path is qualified and adopted for the in-house
   MiniMax-M3 checkpoint on H100. The qualified stack preserves packed INT4
   group-128 weights, dynamic per-token E4M3 activations, TP8 plus expert
   parallelism, graphs-on serving, and the production benchmark contract.
   Indexed Humming 0.1.10 passed fail-closed backend attestation, correctness,
   and stability qualification, then beat CUTLASS by about 34% at concurrency 1
   in the paired serving benchmark. It is now the default kernel for the
   serve-ready `gptq-base` arm. See `M3_HUMMING_W4A8_QUALIFICATION_REPORT.md`,
   `docs/m3-two-axis-perf.md`, and
   `docs/superpowers/specs/2026-07-25-minimax-m3-humming-native-w4a8-backend-design.md`.

## Current session focus

**Goals 1 and 2.** Goal 7 is complete. The remaining near-term work is improving
the save/export portion of distributed quantization and completing the missing
seven-task breadth evaluation for the quality-clean AWQ recipe. Goal 6 remains
an approved benchmark-gated future project.
