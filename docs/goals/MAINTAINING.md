# Maintaining the PM-facing goal pages

Internal instructions for whoever updates the pages under `docs/` — **never put
this kind of content in the pages themselves**; they are written for PM readers.

## The page set

| Page | Pair |
|---|---|
| Program overview | `docs/automatic-quantization-pipeline-progress.{html,md}` |
| Goal 1 field note | `docs/goals/goal-1-fast-parallel-quantization.{html,md}` |
| Goal 2 field note (carries all results tables) | `docs/goals/goal-2-temporary-evaluation-pipeline.{html,md}` |
| Goal 6 field note | `docs/goals/goal-6-hopper-packed-nvfp4-w4a8.{html,md}` |
| Goal 7 field note | `docs/goals/goal-7-native-humming-w4a8.{html,md}` |

Source of truth: `PROJECT_GOALS.md` (repo root, engineer-grade). Every page is
an `.html`/`.md` **twin pair with identical content** — edit both or neither.

## Rules

1. **Audience is the PM.** Plain outcome language; define any metric at first
   use (e.g. TPOT); no repo jargon, commit hashes, agent/planner talk, or
   notes-to-self. Engineering detail belongs in `PROJECT_GOALS.md` and the
   owner docs the pages link to.
2. **Sub-tasks are the unit of progress.** IDs (`1a`, `2g`, …) come from
   `PROJECT_GOALS.md` and must match everywhere. To record progress: tick the
   item + add a week stamp (`wk MM-DD–MM-DD`, the ISO Mon–Sun week it landed).
   New sub-tasks append with the next letter; never delete finished ones.
3. **Weekly log**: one line per achievement under the current week's heading,
   **newest week first**, in `PROJECT_GOALS.md` and both overview twins.
4. **Latest results first.** New results tables go into the goal-2 pair's
   results sections, above older ones; superseded results move to (or stay in)
   the Historical section, clearly marked not-comparable.
5. **Numbers must trace to a source.** Every figure comes from
   `M3_OFFICIAL_QUALITY_RESULTS.html`, `M3_OFFICIAL_PERF_RESULTS.html`, or an
   evidence directory — re-check against the source when copying, and keep the
   "Updated <date>" stamps current.
6. **Every page keeps a Contents block** at the top; update it when sections
   change (html: the `<nav class="toc">`; md: the `**Contents:**` line).

## Update walkthrough (when something lands)

1. `PROJECT_GOALS.md`: tick the sub-task, stamp the week, add a weekly-log line.
2. Overview pair: mirror the sub-task tick + weekly-log line (goal cards in the
   html, checklists in the md).
3. The goal's field-note pair: mirror the sub-task; add results
   tables/paragraphs if there are new numbers (goal 2 for quality/perf results).
4. Sanity-check: html tag balance, internal links, and TOC anchors — a small
   parser script for this lives in the git history of these pages (commit
   `a096977d`); numbers-vs-source diff as in commit `28a14b19`.
