# Automatic Quantization Progress Editorial Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the approved terminology, GPTQ lifecycle-row, and direct Marlin-description revisions consistently to the Markdown and HTML engineering update.

**Architecture:** Keep the Markdown and self-contained HTML as equivalent views of one article. Change reader-facing copy only; retain historical report filenames, links, and implementation identifiers so repository references remain valid.

**Tech Stack:** Markdown, Mermaid, semantic HTML, inline CSS, Python standard-library HTML parsing, Git text checks.

## Global Constraints

- Use **post-quantization gate** as the reader-facing name; do not use Application Binary Interface or ABI as an article concept.
- Explain at first mention that the gate checks whether checkpoint metadata and tensors match inference-engine expectations.
- Represent the original and metadata-repaired in-house GPTQ artifact in one lifecycle row.
- Present verified FP4 Marlin behavior directly, without “correction,” “previous shorthand,” or similar revision-history framing.
- Do not rename historical files or code identifiers containing `ABI`.

---

### Task 1: Revise the Markdown article

**Files:**
- Modify: `docs/automatic-quantization-pipeline-progress.md`

**Interfaces:**
- Consumes: the approved editorial rules in `docs/superpowers/specs/2026-07-13-automatic-quantization-progress-blog-design.md`.
- Produces: canonical article copy for terminology and evidence-state parity with the HTML artifact.

- [x] **Step 1: Replace the reader-facing ABI terminology**

Change the first checkpoint-boundary explanation to define the post-quantization gate in plain language. Update the Mermaid label, plain-text pipeline, Layer 3 copy, MiniMax evidence prose, status table, and reference labels so article copy no longer calls the boundary an ABI. Preserve repository link destinations whose filenames contain `ABI`.

- [x] **Step 2: Consolidate the GPTQ lifecycle**

Replace the separate “Original in-house GPTQ” and “Metadata-repaired in-house GPTQ” table rows with one “In-house GPTQ” row. State that the same checkpoint first failed with 228 namespace/ignore mismatches, then passed after a tensor-preserving metadata overlay and produced coherent smoke generations. Give the combined row the final status “Working candidate.”

- [x] **Step 3: Present Marlin behavior directly**

Remove revision-history phrases such as “initial shorthand” and “needs one correction.” State directly that vLLM's FP4 Marlin path uses packed FP4 weights with higher-precision activations on SM80+ hardware, including Hopper, and that this is an NVFP4-weight/A16-style execution path rather than checkpoint conversion.

- [x] **Step 4: Verify Markdown terminology and structure**

Run:

```powershell
rg -n -i "application binary interface|serving ABI|post-quant.*ABI|correction:|initial shorthand|needs one correction" docs/automatic-quantization-pipeline-progress.md
rg -n "Post-quantization gate|In-house GPTQ|FP4 Marlin|```mermaid" docs/automatic-quantization-pipeline-progress.md
```

Expected: the first command returns no reader-facing matches; the second finds the new gate label, one consolidated GPTQ row, the direct Marlin description, and the Mermaid block.

### Task 2: Revise and validate the HTML companion

**Files:**
- Modify: `docs/automatic-quantization-pipeline-progress.html`

**Interfaces:**
- Consumes: the canonical wording produced in Task 1.
- Produces: a self-contained visual companion with equivalent terminology, evidence states, links, and factual claims.

- [x] **Step 1: Mirror the approved copy and table structure**

Replace reader-facing ABI labels with “post-quantization gate,” add the plain-language first-use explanation, consolidate the two GPTQ rows into one lifecycle row with “Working candidate,” and replace the correction callout heading with a neutral heading such as “FP4 Marlin on Hopper.” Keep the existing callout styling and link targets.

- [x] **Step 2: Parse and audit the HTML**

Run a Python standard-library parser that asserts: semantic landmarks remain present; IDs are unique; no external scripts, images, or stylesheets were added; exactly one “In-house GPTQ” evidence row is present; “post-quantization gate” and “FP4 Marlin on Hopper” are present; and reader-facing “Application Binary Interface,” “serving ABI,” and “Correction:” are absent.

Expected: `artifact_validation: passed`.

- [x] **Step 3: Check cross-format parity and Git cleanliness**

Run:

```powershell
rg -n -i "post-quantization gate|in-house GPTQ|FP4 Marlin on Hopper" docs/automatic-quantization-pipeline-progress.md docs/automatic-quantization-pipeline-progress.html
git diff --check
git diff -- docs/automatic-quantization-pipeline-progress.md docs/automatic-quantization-pipeline-progress.html
```

Expected: both formats contain the three revised concepts; `git diff --check` exits zero; the diff contains only the approved editorial and table-structure changes.

- [x] **Step 4: Commit the article revisions**

```powershell
git add -- docs/automatic-quantization-pipeline-progress.md docs/automatic-quantization-pipeline-progress.html docs/superpowers/plans/2026-07-14-automatic-quantization-progress-editorial-revision.md
git commit -m "docs: clarify quantization pipeline terminology"
```

Expected: one documentation commit containing the two synchronized article artifacts and this implementation plan.
