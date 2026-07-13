# Automatic Quantization Roadmap Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the near-term roadmap follow the pipeline dependency: smoke qualification before full-calibration checkpoint production.

**Architecture:** Update the ordered roadmap in the Markdown and self-contained HTML as equivalent views of the same article. Preserve the remaining roadmap order and all presentation styles.

**Tech Stack:** Markdown, semantic HTML, Python standard-library HTML parsing, Git text checks.

## Global Constraints

- Item 1 must qualify a recipe through all-layer smoke quantization with embedded probes.
- Item 2 must run full-calibration quantization for qualified recipes to produce working in-house MiniMax-M3 checkpoints.
- Items 3–6 must retain their current order and meaning.
- The revised first two items must use identical wording in Markdown and HTML; existing concise HTML wording for items 3–6 must remain unchanged.

---

### Task 1: Reorder and clarify the near-term roadmap

**Files:**
- Modify: `docs/automatic-quantization-pipeline-progress.md:181`
- Modify: `docs/automatic-quantization-pipeline-progress.html:434`

**Interfaces:**
- Consumes: the dependency sequence approved in `docs/superpowers/specs/2026-07-13-automatic-quantization-progress-blog-design.md`.
- Produces: matching six-item roadmaps whose first two items encode smoke → full calibration → working checkpoint.

- [x] **Step 1: Replace the first two Markdown items**

Use this exact sequence:

```markdown
1. Run all-layer smoke quantization with embedded probes to qualify each recipe before committing to a full-calibration run.
2. Run full-calibration quantization for qualified recipes to produce working in-house MiniMax-M3 checkpoints, starting with repaired GPTQ and a healthy AWQ path.
```

- [x] **Step 2: Mirror the sequence in HTML**

Use these exact list items:

```html
<li>Run all-layer smoke quantization with embedded probes to qualify each recipe before committing to a full-calibration run.</li>
<li>Run full-calibration quantization for qualified recipes to produce working in-house MiniMax-M3 checkpoints, starting with repaired GPTQ and a healthy AWQ path.</li>
```

- [x] **Step 3: Validate sequence and structure**

Parse both files and assert that the smoke item precedes the full-calibration item, both formats contain six roadmap items, the first two normalized item texts match, the HTML still contains its semantic ordered list, and items 3–6 remain unchanged from `HEAD` in each format.

Run:

```powershell
git diff --check
git diff -- docs/automatic-quantization-pipeline-progress.md docs/automatic-quantization-pipeline-progress.html
```

Expected: exit code zero; only the first two roadmap items change.

- [x] **Step 4: Commit the synchronized roadmap**

```powershell
git add -- docs/automatic-quantization-pipeline-progress.md docs/automatic-quantization-pipeline-progress.html docs/superpowers/plans/2026-07-14-automatic-quantization-roadmap-order.md
git commit -m "docs: order quantization roadmap by dependency"
```

Expected: one documentation commit containing both article formats and this plan.
