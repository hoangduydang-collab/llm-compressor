# MiniMax-M3 Quantization & Evaluation · Program Overview

> **Updated 31 July 2026.** Web version:
> [`automatic-quantization-pipeline-progress.html`](automatic-quantization-pipeline-progress.html).

Automatic quantization is hard because three systems must agree — a new model
architecture, a quantization algorithm, and an inference engine — and the
agreement must be proven, not assumed. Our `llm-compressor` fork is the control
plane; MiniMax-M3 is the first case study. Each goal is a sub-task checklist;
finished sub-tasks carry the week they landed, so weekly progress reads directly
off this page.

**Seven goals, tracked as week-stamped sub-tasks. Two complete: a
quality-competitive in-house AWQ model (3) and 34%-faster native W4A8 serving
(7). Active: parallel quantization (1), the evaluation pipeline (2), a third
quant method (4).**

**Contents:** [Seven goals](#seven-long-term-goals) ·
[Current focus](#current-focus-goals-1-2-and-4) · [Weekly log](#weekly-log) ·
[Field notes](#field-notes)


## Seven long-term goals

### Goal 1 — Fast parallel quantization · Work in progress · [field note](goals/goal-1-fast-parallel-quantization.md)
Full quantization run in 4–8 hours, any method.
- [x] 1a · Distributed calibration — AWQ **7 h 22 m**, GPTQ **3 h 14 m**, all gates green, one node `wk Jul 20–26`
- [x] 1b · Multi-GPU calibration correctness fix `wk Jul 27–Aug 02`
- [x] 1c · Proven on a second model + third method — 30B MoE in **1 h 40 m** `wk Jul 27–Aug 02`
- [ ] 1d · Distributed save/export

### Goal 2 — Evaluation pipeline · Work in progress · [field note](goals/goal-2-temporary-evaluation-pipeline.md)
One fail-closed harness: our model vs existing quants vs the unquantized baseline.
- [x] 2a · Paired harness + smoke gate live `wk Jul 20–26`
- [x] 2b · GPTQ validated on all seven tasks — 97–101% recovery `wk Jul 20–26`
- [x] 2c · Serving-performance report (ten arms) `wk Jul 20–26`
- [x] 2d · Second model family — GLM-5.2, three arms `wk Jul 20–26`
- [x] 2e · Second quant track — 2-bit vs baseline A/B `wk Jul 27–Aug 02`
- [x] 2f · Collaborator guide, live-verified `wk Jul 27–Aug 02`
- [ ] 2g · Seven-task run for the fixed AWQ model
- [x] 2h · Speculative-decoding tuning study — up to 2.5× faster decoding, no quality cost `wk Jul 27–Aug 02`

### Goal 3 — Working AWQ quantized model · Done (quality-competitive 2026-07-23)
Shipped broken twice, fixed twice; the current recipe (`r6`) is within ~1% of
baseline. **Serve r6 or r7, never r5.**
- [x] 3a · Checkpoint corruption root-caused and fixed `wk Jul 20–26`
- [x] 3b · Runaway-reasoning defect root-caused; r6 fixes it (GPQA 98.7%, IFEval 98.6%) `wk Jul 20–26`

### Goal 4 — Generalize to any quant method · Work in progress
Extend beyond AWQ and GPTQ. First new method: AutoRound, on a 2-bit track (30B MoE).
- [x] 4a · Quantize→serve loop closed — 2-bit checkpoint serves coherently `wk Jul 27–Aug 02`
- [x] 4b · First quality A/B — no-ship; the gate blocked it as designed `wk Jul 27–Aug 02`
- [ ] 4c · Retry with longer tuning + re-eval

### Goal 5 — Generalize gates to any model family · Future work
Make the compatibility gates work for any model family, not just MiniMax-M3.

### Goal 6 — Packed NVFP4 W4A8 on Hopper · Planned, benchmark-gated · [field note](goals/goal-6-hopper-packed-nvfp4-w4a8.md)
Run vendor NVFP4 checkpoints efficiently on current (Hopper) GPUs — weights stay
packed at 4 bits, compute in FP8. Design done; investment gated on a benchmark proof.
- [ ] 6a · Dense proof of concept — the gate for all further work

### Goal 7 — Native Humming W4A8 serving · Done 2026-07-26 · [field note](goals/goal-7-native-humming-w4a8.md)
A faster serving kernel (Humming W4A8), qualified and adopted for the in-house
model on H100.
- [x] 7a · Qualified — attestation, correctness, stability `wk Jul 20–26`
- [x] 7b · Adopted — ~34% faster for a single user; now the default kernel `wk Jul 20–26`

## Current focus: Goals 1, 2 and 4

Next up: **4c** (2-bit retry), **2g** (seven-task run for the fixed AWQ model),
**1d** (distributed save/export). While the owner is away,
[`M3_COLLABORATOR_GUIDE.md`](../M3_COLLABORATOR_GUIDE.md) (live-verified
2026-07-31) is the front door for collaborators.

## Weekly log

One line per achievement, newest first. IDs point at the sub-tasks above; the
durable copy lives in `PROJECT_GOALS.md`.

### wk 2026-07-27 – 08-02
- **2h** · Speculative decoding tuned on the 4-bit model: 1.2–2.5× faster decoding depending on content type, no quality cost.
- **1c / 4a** · Third quant method onboarded: 30B MoE quantized in 1 h 40 m, serves coherently.
- **4b / 2e** · First 2-bit quality A/B — no-ship verdict; retry queued.
- **2f** · Collaborator guide live-verified; handoff-ready.
- **1b** · Multi-GPU calibration correctness fix.
- Ops · goal tracking restructured into week-stamped sub-tasks; ~591 GB workspace archive launched.

### wk 2026-07-20 – 07-26
- **1a** · Speed target met: full quantization in 7 h 22 m (goal: 4–8 h).
- **3a / 3b** · Both AWQ defects fixed; model near-baseline. **Goal 3 complete.**
- **2a / 2b** · GPTQ model validated on all seven tasks (97–101% recovery).
- **2c** · Serving-performance report published (ten arms).
- **2d** · Evaluation proven on a second model family (GLM-5.2).
- **7a / 7b** · Faster kernel qualified + adopted (+34% single-user). **Goal 7 complete.**

### wk 2026-07-13 – 07-19
- First four-way quality comparison on the interim harness (superseded by 2b).
- Distributed-run rehearsals; execution protocol signed.

## Field notes

Each note records the goal's boundary, evidence, and handoffs (md + html pairs):

- [Goal 1 — Fast parallel quantization](goals/goal-1-fast-parallel-quantization.md)
- [Goal 2 — Evaluation pipeline](goals/goal-2-temporary-evaluation-pipeline.md)
- [Goal 6 — Packed NVFP4 W4A8 on Hopper](goals/goal-6-hopper-packed-nvfp4-w4a8.md)
- [Goal 7 — Native Humming W4A8 serving](goals/goal-7-native-humming-w4a8.md)
