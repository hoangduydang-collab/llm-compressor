# Goal 1 · Fast Parallel Quantization — Field Note

> **Work in progress** · web version:
> [`goal-1-fast-parallel-quantization.html`](goal-1-fast-parallel-quantization.html).
> [← Program overview](../automatic-quantization-pipeline-progress.md)

**Contents:** [Objective](#objective) · [Sub-tasks](#sub-tasks) ·
[Result](#result) · [Boundary](#boundary) · [Evidence](#evidence)

The core target landed for both methods: a full AWQ run in **7 h 22 m** and a
full GPTQ run in **3 h 14 m**, both inside the 4–8 hour window — and the
parallel path has since quantized a second model with a third method in
**1 h 40 m**. Open: distributed save/export.

## Objective

A complete MiniMax-M3 quantization run in **4–8 hours**, including a saved
checkpoint that survives correctness gates — using the fork's existing
distributed calibration paths rather than bespoke parallel code.

## Sub-tasks


- [x] 1a · Distributed calibration — full AWQ run, all gates green, **7 h 22 m** on one 8-GPU node `wk Jul 20–26`
- [x] 1b · Multi-GPU calibration correctness fix (all GPUs had been calibrating on the same data) `wk Jul 27–Aug 02`
- [x] 1c · Proven beyond MiniMax-M3 and beyond AWQ/GPTQ — a 30B MoE quantized in **1 h 40 m** with a third method `wk Jul 27–Aug 02`
- [ ] 1d · Distributed save/export

## Result

The full-calibration AWQ run finished in **7 h 22 m on one 8×H100 node** —
inside the target — covering all 57 MoE layers and 21,888 expert weight
matrices: ~19 min to load the ~920 GB model, 6 h 40 m of calibration (≈7 min per
layer), 19 min to save and run the in-line gates. The checkpoint cleared every
fail-closed gate, served coherently, and passed the smoke evaluation. An
earlier run had been correctly *rejected* by those gates (a scale-search defect
on dead channels, since fixed and a candidate for upstream contribution) — the
gates catching it is the system working.

**Distributed runs to date** (newest first):

| Model | Method · bits | Hardware | Wall time | Output | When |
|---|---|---|---|---|---|
| Qwen3-30B MoE | AutoRound · 2-bit | 8×H100, 1 node | **1 h 40 m** | 8.6 GB checkpoint, serves coherently | wk Jul 27–Aug 02 |
| MiniMax-M3 (~460B MoE) | GPTQ · 4-bit | 8×H100, 1 node | **3 h 14 m** | 215 GB checkpoint + export variants | wk Jul 20–26 |
| MiniMax-M3 (~460B MoE) | AWQ · 4-bit | 8×H100, 1 node | **7 h 22 m** | 225 GB checkpoint, all gates green | wk Jul 20–26 |

Between the two runs, a subtle correctness defect was found and fixed
(sub-task 1b): every GPU had been calibrating on the *same* data slice instead
of its own. Outputs looked fine — which is exactly why it needed a code-level
audit, not just output checks.

## Boundary

This goal is about parallelization throughput. Producing a *correct* checkpoint
is Goal 3's contract; measuring its quality is Goal 2's.

## Evidence

- Run artifacts: `results/m3-distributed-awq-full/20260720T060340Z-m3-ddp-awq-full-r5-deadchan/`; smoke eval `results/m3-quality/20260720T134946Z-m3-inhouse-awq-r5-smoke/`
- Sub-tasks 1b/1c: `evidence/sub4bit-w2a16-moe-quant/20260730T1005Z-qwen3-30b-a3b-w2a16-ddp8/`, commit `ddd4f9f9`
- Post-mortems: [`BUGS_AND_FIXES.md`](../../BUGS_AND_FIXES.md) · Plan: [`M3_QUANT_SPEEDUP_PLAN.md`](../../M3_QUANT_SPEEDUP_PLAN.md) · Contract: [`PROJECT_GOALS.md`](../../PROJECT_GOALS.md)
