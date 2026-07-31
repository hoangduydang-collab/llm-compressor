# Project goals — MiniMax-M3 quantization & evaluation

This is the durable, repo-authoritative record of the project's long-term goals so
any fresh planner/executor/full-stack agent shares the same north star. Task-level
handoffs (`*_HANDOFF.md`, `*_PLAN.md`) carry the current work; this file carries
*why*.

**Structure convention (since 2026-07-31):** each goal is a durable aim with an
explicit **sub-task list**. Sub-tasks are the unit of progress — new ones may be
added at any time without rewriting the goal's status history, and every finished
sub-task carries the week it landed (`wk MM-DD–MM-DD`) so weekly achievement is
visible at a glance. A goal is `DONE` when its current sub-task list is complete;
adding a later sub-task reopens the list, not the definition. Do not delete a goal
or a finished sub-task — evidence pointers stay.

**When something lands, update three places (one minute total):**
1. tick its sub-task checkbox under the goal (or append a new `Ng — title` line —
   next letter, one bold title, evidence pointer, `*(done DATE, wk …)*` stamp);
2. add a one-liner to the **Weekly log** below under the current week (create the
   week heading if it is the first entry that week);
3. mirror both in `docs/automatic-quantization-pipeline-progress.html` (goal card
   `<li>` + weekly-log line — marked extension points in the HTML).

Last reviewed: 2026-07-31.

## Long-term goals

1. **Fast parallel quantization** — speed up quantization with parallelization for
   AWQ and GPTQ (and, via goal 4, any method), targeting a **full quantization
   process in 4–8 hours**. See `M3_QUANT_SPEEDUP_PLAN.md` and the
   distributed/multi-GPU calibration path (llm-compressor multi-GPU + GPTQModel;
   do not rebuild bespoke expert-parallel code — see the CLAUDE.md worked example).

   - [x] **1a — Distributed AWQ/GPTQ calibration path integrated** (`torchrun` DDP
     through `pipeline.run`): full-calibration AWQ run, all gates green, in
     **7 h 22 m** on one 8×H100 node — inside the 4–8 h target.
     *(done 2026-07-20, wk 07-20–07-26)*
   - [x] **1b — DDP rank-partition correctness fix**: the partition was applied to
     a local variable only, so all ranks calibrated on the same rows; fixed in
     ddd4f9f9. *(done 2026-07-30, wk 07-27–08-02)*
   - [x] **1c — DDP proven beyond M3 and beyond AWQ/GPTQ**: 8×H100 DDP AutoRound
     quantized Qwen3-30B-A3B (W2A16) end to end in **1 h 40 m**. Evidence:
     `evidence/sub4bit-w2a16-moe-quant/20260730T1005Z-qwen3-30b-a3b-w2a16-ddp8/`.
     *(done 2026-07-30, wk 07-27–08-02)*
   - [ ] **1d — Distributed save/export**: improve the save/export portion of
     distributed quantization.

2. **Complete the evaluation pipeline** — a pipeline to compare our **in-house
   quantized model** against **other existing quantized models** and the
   **original unquantized (BF16) baseline**, with fail-closed harness/gate
   contracts and honest raw-evidence returns. See `M3_PRODUCTION_EVAL_HANDOFF.md`
   and the plans/specs under `docs/superpowers/`.

   - [x] **2a — Fail-closed paired harness + official smoke gate** (the in-house
     AWQ checkpoint cleared it: `ready_for_production: true`).
     *(done 2026-07-20, wk 07-20–07-26)*
   - [x] **2b — Seven-task breadth verdict for in-house GPTQ** vs BF16 / MXFP8 /
     cyankiwi: recovery 97.4–101.1% on all seven tasks, spend within 2% of BF16.
     `M3_OFFICIAL_QUALITY_RESULTS.html`. *(done 2026-07-23, wk 07-20–07-26)*
   - [x] **2c — Two-axis perf report** (kernel axis × model axis, ten arms, one
     controller): `docs/m3-two-axis-perf.md`, `M3_OFFICIAL_PERF_RESULTS.html`.
     *(done 2026-07-26, wk 07-20–07-26)*
   - [x] **2d — Second model family**: paired three-arm GLM-5.2 eval (BF16 vs
     community W4AFP8 checkpoints on SGLang) — Phala's AWQ-produced W4AFP8 is
     clean; the M3 AWQ pathology does not replicate.
     `GLM52_OFFICIAL_EVAL_RESULTS.html`. *(done 2026-07-23, wk 07-20–07-26)*
   - [x] **2e — Second quant track**: W2A16 vs BF16 paired quality A/B (GPQA +
     IFEval, paired greedy, harness check PASS). Evidence:
     `evidence/sub4bit-w2a16-moe-quality/20260731T0300Z-qwen3-30b-w2a16-vs-bf16-gpqa-ifeval/`.
     *(done 2026-07-31, wk 07-27–08-02)*
   - [x] **2f — Collaborator front door**: `M3_COLLABORATOR_GUIDE.md`, audited
     against owner docs and with every Part-A command executed end-to-end on real
     hardware. *(done 2026-07-31, wk 07-27–08-02)*
   - [ ] **2g — Seven-task breadth run for the quality-clean AWQ r6 recipe** (the
     `full4` protocol has only ever been run for r5; in-house GPTQ still holds the
     sole full-breadth shipping verdict).

3. **Working AWQ quantized model** — *DONE (all current sub-tasks complete);
   quality-competitive as of 2026-07-23 (r6).* **Serve `r6` or `r7`, not `r5`.**
   Checkpoints and per-arm provenance: `docs/m3-benchmark-arms.md` (r5 retained as
   historical — do not serve it). Post-mortems in `BUGS_AND_FIXES.md`;
   collaborator-facing writeup in `M3_OFFICIAL_QUALITY_RESULTS.html`.

   - [x] **3a — Checkpoint corruption root-caused and fixed** (r4 → r5): an AWQ
     smoothing-scale degeneracy on dead norm channels (M3 layers 8/10–13 have a
     post-attention norm gain of exactly −1.0); fixed in `_grid_search_scales`
     (e87eef77, upstream-candidate) with regression tests and a hardened
     fail-closed gate suite. r5 then passed all 57-layer gates, served coherently
     on TP8, and cleared the official smoke eval. Evidence:
     `results/m3-quality/20260720T134946Z-m3-inhouse-awq-r5-smoke/`.
     *(done 2026-07-20, wk 07-20–07-26)*
   - [x] **3b — Reasoning non-termination root-caused and fixed** (r5 → r6/r7): r5
     failed the paired 64k quality eval (GPQA recovery 71.7%, token spend 2.19×,
     budget exhaustion 38.9% vs a 12.6% BF16 floor — runaway thinking loops).
     Root cause: AWQ folding smoothing scales through M3's **up→down** path,
     which is not function-preserving across the clamped GLU `(clamp(up)+1)·glu`.
     `r6` drops that fold; `r7` reintroduces down-side smoothing through the gate
     path, where it is function-preserving. Result: GPQA recovery **98.7% /
     104.4%**, IFEval **98.6% / 95.7%**, spend 1.13×/0.93×, exhaustion
     14.7%/10.6%. Evidence:
     `results/m3-official-quality/20260723T153532Z-tok64k-awqr6/`,
     `…/20260724T053810Z-tok64k-awqr7/`,
     `results/m3-sampling-probe/20260723T125813Z-r6/`.
     *(done 2026-07-24, wk 07-20–07-26)*

   The r6/r7 seven-task breadth evaluation is tracked as sub-task **2g** (goal-2
   territory).

4. **Generalize to any quantization method** — extend the pipeline beyond AWQ and
   GPTQ to arbitrary quantization methods. First concrete method: **AutoRound**
   (sub-4-bit W2A16 track on Qwen3-30B-A3B-Instruct-2507).

   - [x] **4a — AutoRound quant→serve loop closed**: DDP AutoRound quant (LLMC
     main + auto-round 0.14.1) produced an 8.6 GB CT pack-quantized int2-g128
     checkpoint that serves coherently on vLLM 0.26 + PR#48918 port + Humming
     main. Evidence: `evidence/sub4bit-w2a16-moe-quant/`.
     *(done 2026-07-30, wk 07-27–08-02)*
   - [x] **4b — W2A16 quality A/B at `iters=200`** — verdict **NO-SHIP** (GPQA
     −0.30, IFEval −0.39 vs BF16; greedy repetition loops, IFEval token spend
     3.44×). A completed measurement with a negative outcome, not a failed task.
     Evidence: `evidence/sub4bit-w2a16-moe-quality/`.
     *(done 2026-07-31, wk 07-27–08-02)*
   - [ ] **4c — `iters=1000` requant** and paired re-eval (next candidate step).
   - [ ] **4d — Third method** beyond AWQ/GPTQ/AutoRound (unscheduled).

5. **Generalize the gates to any model family** — *Future work.*
   Generalize the serving-ABI gate and the pre-quantization static gate so they
   apply to any model family, not just MiniMax-M3. (No sub-tasks started; the
   GLM-5.2 eval under 2d exercised the eval side on a second family, but the
   gates themselves remain M3-specific.)

6. **Packed NVFP4 W4A8 fallback on Hopper** — *Planned: benchmark-gated.*
   Enable officially released NVFP4 checkpoints to run on H100/H200 with packed
   four-bit weights in GPU memory, dynamic FP8 activations, in-kernel E2M1-to-FP8
   conversion and scaling, FP8 Tensor Core multiplication, FP32 accumulation,
   and BF16 output. Build on the existing vLLM NVFP4/Marlin loaders and NVIDIA
   CUTLASS SM90 mixed-input WGMMA machinery rather than writing a kernel from
   scratch. See
   `docs/superpowers/specs/2026-07-19-hopper-packed-nvfp4-w4a8-fallback-design.md`.

   - [ ] **6a — Dense proof of concept** vs upstream NVFP4 W4A16 Marlin and
     load-expanded W8A8; MoE production work proceeds only if this passes
     correctness and target-workload throughput gates.

7. **Native Humming W4A8 serving on Hopper** — *DONE (all current sub-tasks
   complete).* Humming's existing GPTQ W4A8 path qualified and adopted for the
   in-house MiniMax-M3 checkpoint on H100: packed INT4 group-128 weights, dynamic
   per-token E4M3 activations, TP8 + expert parallelism, graphs-on serving, and
   the production benchmark contract. See
   `M3_HUMMING_W4A8_QUALIFICATION_REPORT.md` and
   `docs/superpowers/specs/2026-07-25-minimax-m3-humming-native-w4a8-backend-design.md`.

   - [x] **7a — Qualification**: indexed Humming 0.1.10 passed fail-closed backend
     attestation, correctness, and stability qualification.
     *(done 2026-07-26, wk 07-20–07-26)*
   - [x] **7b — Adoption**: beat CUTLASS by ~34% at concurrency 1 in the paired
     serving benchmark (`docs/m3-two-axis-perf.md`); now the default kernel for
     the serve-ready `gptq-base` arm. *(done 2026-07-26, wk 07-20–07-26)*

## Weekly log

One line per achievement, newest week first. Details, numbers, and evidence live
in the referenced sub-tasks above (and the per-goal field notes under
`docs/goals/`) — this list only answers "what moved this week".

### wk 2026-07-27 – 08-02
- **1c / 4a** — First new method onboarded (AutoRound): DDP W2A16 quant of a 30B
  MoE in 1 h 40 m; checkpoint serves coherently.
- **4b** — W2A16 `iters=200` quality A/B measured: NO-SHIP verdict; `iters=1000`
  requant queued as 4c.
- **2e** — Paired A/B protocol proven on a second quant track (W2A16 vs BF16).
- **2f** — Collaborator guide finished and live-verified end-to-end; handoff-ready.
- **1b** — DDP rank-partition correctness fix (all ranks had calibrated on the
  same rows).
- Ops — goals tracking restructured into week-stamped sub-tasks; curated ~591 GB
  workspace archive to the ai-lab jump host launched
  (`/mnt/nfs/hoangduy/workspace-transfer/`).

### wk 2026-07-20 – 07-26
- **1a** — Distributed calibration hit the 4–8 h target: full AWQ run in 7 h 22 m.
- **3a / 3b** — Both AWQ defects root-caused and fixed; r6 is quality-competitive
  (GPQA 98.7%, IFEval 98.6%). Goal 3 complete.
- **2a / 2b** — Official smoke gate cleared, then the seven-task breadth verdict
  for in-house GPTQ (recovery 97.4–101.1%).
- **2c** — Two-axis perf report (kernel × model, ten arms).
- **2d** — Second model family: three-arm GLM-5.2 eval.
- **7a / 7b** — Humming W4A8 qualified and adopted (+34% at concurrency 1).
  Goal 7 complete.

### wk 2026-07-13 – 07-19
- Stopgap four-way quality comparison (in-house GPTQ / cyankiwi / MXFP8 / BF16)
  on the seeded-subset harness — historical, superseded by 2b.
- DDP full-run rehearsals r1–r3; planner–executor protocol and the distributed
  speedup execution packet signed.

## Current session focus

**Goals 1, 2 and 4.** Goals 3 and 7 are complete. The owner is away for a few
weeks starting 2026-08; `M3_COLLABORATOR_GUIDE.md` (live-verified 2026-07-31) is
the continuity anchor for the evaluation-side collaborators, and a curated
~590 GB workspace archive (both servable M3 checkpoints + all code/evidence) is
being mirrored to the ai-lab jump host
(`/mnt/nfs/hoangduy/workspace-transfer/README.md`). Near-term work queue: **4c**
(W2A16 `iters=1000` requant), **2g** (seven-task breadth run for AWQ r6), and
**1d** (distributed save/export). Goal 6 remains an approved benchmark-gated
future project.
