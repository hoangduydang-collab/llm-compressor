# Goal status boundaries design

**Date:** 2026-07-20  
**Scope:** HTML research updates only  
**Audience:** Busy collaborators

## Status model

The HTML research update will present:

- Goal 1, fast parallel quantization: **Done**
- Goal 2, evaluation pipeline: **Done**
- Goal 3, working AWQ quantized model: **Work in progress**
- Goal 6: unchanged as planned and benchmark-gated

`PROJECT_GOALS.md` must remain untouched. Because its status markers will temporarily
differ from the HTML presentation, the program overview must stop claiming that it
exactly mirrors the file's current statuses. It may continue linking to
`PROJECT_GOALS.md` for the durable goal definitions.

## Program overview

Update `docs/automatic-quantization-pipeline-progress.html` to:

- add a green `Done` status treatment;
- mark Goal 1 and Goal 2 as done;
- mark Goal 3 as work in progress;
- describe Goals 1 and 2 as completed foundations;
- make Goal 3 the only current session focus;
- remove wording that calls Goal 3 planned or folded into Goal 1;
- retain existing links, layout, and Goal 6 status.

Goal 3 remains a non-linked card. Creating a dedicated Goal 3 page is out of scope.

## Goal 1 page

Update `docs/goals/goal-1-fast-parallel-quantization.html` to:

- show a `Done` status in the hero;
- lead with completion of the parallelization objective: the distributed path met
  the 4–8 hour target with a seven-hour full calibration;
- retain the failed AWQ correctness-gate result as an honest boundary, not as evidence
  that parallelization is unfinished;
- replace "current status" and "acceptance path" language with a scope-boundary
  handoff to Goal 3;
- state that producing a correct full-calibration AWQ checkpoint, serving it, and
  evaluating it are Goal 3 work;
- remove footer wording that says the full-calibration target is unachieved.

## Goal 2 page

Update `docs/goals/goal-2-temporary-evaluation-pipeline.html` to:

- show a `Done` status in the hero;
- state that the working deliverable is the reproducible, fail-closed paired
  evaluation pipeline;
- retain the completed four-model comparison and harness setup as completion evidence;
- replace the "Next" section with an "Available for use" boundary;
- classify generating additional merged reports and evaluating future checkpoints as
  use of the completed pipeline, not unfinished Goal 2 development;
- keep limitations about seeded subsets and public-leaderboard comparability.

## Verification

Verify that:

- only the three HTML files above change;
- Goal 1 and Goal 2 each display `Done`;
- Goal 3 displays `Work in progress` on the overview;
- the overview names Goal 3 as the sole current focus;
- no HTML wording says Goal 1 or Goal 2 remains open;
- `PROJECT_GOALS.md` is byte-for-byte unchanged;
- all three pages parse and all local links resolve.
