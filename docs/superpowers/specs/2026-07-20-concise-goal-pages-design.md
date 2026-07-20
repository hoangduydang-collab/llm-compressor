# Concise goal pages design

**Date:** 2026-07-20  
**Audience:** Busy collaborators seeking a quick research update  
**Target:** About three minutes per page

## Editorial contract

Keep the existing visual design, navigation, status labels, and evidence links. Rewrite
the body of each goal page around decisions and results, targeting roughly 350–500
words plus one essential table or checklist. Use collaborator-facing language and
omit personal debugging history, obsolete approaches, internal run chronology, and
details that do not change a collaborator's understanding or next decision.

Claims must remain accurate and clearly distinguish measured results from targets or
future plans. Keep a short source section so readers can inspect the detailed reports.

## Goal 1: fast parallel quantization

Structure the page as:

1. Objective: complete AWQ/GPTQ quantization in 4–8 hours.
2. Approach: use the fork's existing distributed calibration path.
3. Results and status: retain the 71-minute three-layer smoke, 852 GB peak host RSS,
   1,152 verified quantized Linears per method, and the explicit caveat that no full
   end-to-end result exists yet.
4. Next steps: finish the full run, measure speedup, and run checkpoint and serving
   quality gates.

Remove the initial failure to use available GPUs, the bespoke implementation detour,
the thread-pool benchmark, and the r2–r9 debugging chronology.

## Goal 2: temporary evaluation pipeline

Structure the page as:

1. Purpose: paired quantization-fidelity comparison, not public leaderboard scoring.
2. Core results: retain one four-model score table and a brief interpretation.
3. Harness setup: retain the framework/version, model set, tasks and subset sizes,
   three seeds, generation settings, and serving topology in compact form.
4. Limitations and next step: note seeded subsets and the pending merged comparison
   report.

Remove the duplicate BF16-delta table, obsolete harness-era warning, raw hashes,
run-by-run completion history, BF16 troubleshooting narrative, exact checkpoint
paths, shell launch details, and extended reproduction checklist.

## Goal 6: packed NVFP4 W4A8 on Hopper

Present this only as a future-work plan:

1. Motivation: retain packed four-bit weights on Hopper while using FP8 Tensor Cores.
2. Proposed direction: extend the existing Humming path, with Marlin W4A16 as the
   fail-closed fallback.
3. Decision gates: summarize correctness, memory, and useful performance improvement.
4. Staged plan: one-layer probe, dense proof of concept, serving qualification, then
   MoE work only if prior stages pass.

State prominently that no hardware results exist. Remove detailed kernel mechanics,
conversion equations, effort estimates, prototype file inventory, analytical memory
arithmetic, and exhaustive numeric gate thresholds.

## Verification

Check that each page:

- can be read in about three minutes;
- leads with outcome and current status;
- preserves all retained numbers exactly;
- labels plans and unmeasured claims clearly;
- contains working navigation and source links;
- remains valid responsive HTML without changing the established visual theme.
