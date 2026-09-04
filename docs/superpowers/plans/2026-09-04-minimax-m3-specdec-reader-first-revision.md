# Reader-First MiniMax-M3 Spec-Dec Note Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the MiniMax-M3 synthetic-versus-real prompt note to a concise decision memo for collaborators investigating a differing result.

**Architecture:** Replace the current setup-first narrative with a conclusion-first reader path. Keep one result table, one prompt-construction table, a short discrepancy checklist, and a compact non-sensitive artifact handoff; link out for all deeper detail.

**Tech Stack:** Markdown and the existing MiniMax-M3 EAGLE3 primary report.

## Global Constraints

- Modify only `docs/minimax-m3-specdec-synthetic-vs-real-prompts.md`.
- Keep the note near 100 lines.
- Preserve 2.450, 2.473, +0.9%, 1.72×, 1.81×, the cross-window caveat, and the private archive host `138.252.188.36`.
- Do not expose the owner-specific archive path or repeat the legacy runbook.
- Point to the primary report and design packet for full setup and reproduction detail.

---

### Task 1: Replace the setup-first note with a collaborator decision memo

**Files:**
- Modify: `docs/minimax-m3-specdec-synthetic-vs-real-prompts.md`
- Reference: `docs/m3-specdec-eagle3.md:125-135, 170-240`
- Reference: `M3_SPECDEC_EAGLE3_PLAN.md:114-150`

**Interfaces:**
- Consumes: documented Wave 1/Wave 2 results and migrated-artifact verification.
- Produces: a concise, linkable memo that lets a collaborator assess whether prompt source plausibly explains their result.

- [ ] **Step 1: Put the finding first**

Start with a three-sentence `## Takeaway` that states synthetic acceptance
2.450 versus natural ShareGPT acceptance 2.473 (+0.9%, treated as noise), the
cross-window instrumentation caveat, and the conclusion that prompt source
alone is not a supported explanation for a differing result.

- [ ] **Step 2: Keep the minimum evidence**

Add `## What was compared` with one two-row table for synthetic random-token
AA prompts versus first-turn ShareGPT text. Add `## Results` with one table:

```markdown
| Workload | Accepted length | Per-position acceptance | k=0 → k=3 tok/s/user | Decode speedup |
|---|---:|---|---:|---:|
| Synthetic AA-style, 1k input / conc-1 | 2.450* | 0.70 / 0.46 / 0.29 | 137.5 → 236.0 | 1.72× |
| Natural ShareGPT, conc-1 | 2.473 | 0.690 / 0.459 / 0.324 | 137.9 → 249.8 | 1.81× |
```

Explain in one footnote that 2.450 is Wave 1's arm-level log-derived mean,
while 2.473 is Wave 2's per-cell counter delta.

- [ ] **Step 3: Guide comparison and evidence inspection**

Add `## What to compare next` listing output policy, temperature, k,
target/drafter pair, and topology. State the directly observed output-shape
correction: forced 8k continuation raised acceptance 2.473 → 3.286 (+33%).

Add `## Inspect the evidence` with the private host, owner-access instruction,
window names, two aggregator names, and the Wave 1 limitation: its raw logs
re-aggregate acceptance but the external AA throughput summary was not
migrated; use preserved `aggregate.json` for that speed row.

- [ ] **Step 4: Validate concise reader-first content**

Run:

```powershell
$note = 'docs/minimax-m3-specdec-synthetic-vs-real-prompts.md'
(Get-Content $note | Measure-Object -Line).Lines
rg -n "^## (Takeaway|What was compared|Results|What to compare next|Inspect the evidence)$" $note
rg -n "2\.450|2\.473|1\.72×|1\.81×|cross-window|138\.252\.188\.36" $note
git diff --check -- $note
```

Expected: approximately 100 lines; all five reader-oriented sections and all
required evidence/caveat values appear; no whitespace errors.

- [ ] **Step 5: Commit and push**

Run:

```powershell
git add -- docs/minimax-m3-specdec-synthetic-vs-real-prompts.md
@'
docs(specdec): make prompt comparison reader-first

Condense the synthetic-versus-real MiniMax-M3 speculative-decoding note into a
collaborator decision memo and move deep reproduction detail to source links.
'@ | git commit -F -
git push origin duy-branch
```

Expected: one documentation commit pushed without staging unrelated work.
