# Project goals — MiniMax-M3 quantization & evaluation

This is the durable, repo-authoritative record of the project's long-term goals so
any fresh planner/executor/full-stack agent shares the same north star. Task-level
handoffs (`*_HANDOFF.md`, `*_PLAN.md`) carry the current work; this file carries
*why*. Keep the status markers current; do not delete a goal when it completes —
mark it `DONE` with a pointer to the evidence.

Last reviewed: 2026-07-16.

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

3. **Working AWQ quantized model** — *Planned: after goal 1, or in parallel with it.*
   Produce a correct AWQ-quantized model by fixing the current AWQ bug.

4. **Generalize to any quantization method** — *Future work.*
   Extend the pipeline beyond AWQ and GPTQ to arbitrary quantization methods.

5. **Generalize the gates to any model family** — *Future work.*
   Generalize the serving-ABI gate and the pre-quantization static gate so they
   apply to any model family, not just MiniMax-M3.

## Current session focus

**Goals 1 and 2.** Everything else is context, not the immediate objective.
