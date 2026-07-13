# Automatic Quantization Progress Blog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce evidence-backed Markdown and self-contained HTML engineering updates that explain the automatic quantization pipeline, its three protection layers, current MiniMax-M3 results, and near-term work.

**Architecture:** One canonical article outline drives two hand-authored renderings: portable Markdown for repository collaboration and a standalone editorial HTML document for presentation. Repository reports support internal observations; official documentation, source code, hardware documentation, and original papers support upstream claims. Both versions use the same claim boundaries and references.

**Tech Stack:** Markdown, Mermaid, semantic HTML5, embedded CSS, minimal vanilla JavaScript, Python standard-library validation, local browser rendering.

## Global Constraints

- Audience is internal engineering collaborators.
- Use the approved **Engineering field notes** visual direction.
- Keep repository observations, upstream facts, and proposed work visibly distinct.
- Treat the source-model representation and exported checkpoint as the components we adapt to fixed quantizer and inference-engine contracts.
- Expand Application Binary Interface (ABI) on first use.
- Make the three-layer control system the article's centerpiece.
- Layer 2 is an all-layer smoke quantization with embedded probes, not a representative-layer canary.
- Describe small smoke evaluations as diagnostics, not statistically meaningful benchmark conclusions.
- Keep multimodal calibration and custom runtime kernel implementation out of scope.
- Do not expose private absolute paths, node names, credentials, or unpublished collaborator identities.
- The HTML file must work without external assets, packages, fonts, or network access.

---

### Task 1: Build the evidence ledger

**Files:**
- Create: `docs/automatic-quantization-pipeline-progress.md`
- Reference: `M3_PREQUANT_REAL_CLI_FAILURE_REPORT.md`
- Reference: `docs/quantization-static-serving-preflight-status-and-roadmap.md`
- Reference: `M3_STATIC_ABI_GATE_REPORT.md`
- Reference: `M3_GPTQ_REPAIRED_ABI_PREFLIGHT_REPORT.md`
- Reference: `M3_3MODEL_GPTQ_AWQ_FINAL_REPORT.md`
- Reference: `M3_AWQ_REQUANTIZATION_REPORT.md`
- Reference: `M3_AWQ_REPRESENTATIVE_RERUN_REPORT.md`
- Reference: `pipeline/prequant_compatibility.py`
- Reference: `pipeline/m3_guarded_full.py`

**Interfaces:**
- Consumes: repository reports and primary upstream URLs.
- Produces: a claim ledger embedded at the end of the Markdown article under `## Evidence and references`.

- [ ] **Step 1: Extract repository-backed claims**

Record only claims needed by the article: the two compatibility boundaries, the current scope of both static gates, the 228-error GPTQ ABI failure, the tensor-preserving overlay, coherent GPTQ/AWQ-control smoke results, unresolved in-house AWQ status, and all-layer guarded diagnostics.

- [ ] **Step 2: Verify upstream claims**

Use primary sources to verify: llm-compressor's advertised formats and algorithms; AWQ's activation-aware scaling/smoothing mechanism; GPTQ's second-order post-training quantization basis; vLLM's Marlin and NVFP4 hardware paths; NVIDIA Hopper versus Blackwell low-precision capabilities; and multimodal calibration considerations. If official evidence does not support runtime NVFP4-to-W4A16 conversion through Marlin, explicitly correct that claim in the article.

- [ ] **Step 3: Classify each claim**

Use the labels `Observed in this fork`, `Supported upstream`, and `Proposed work`. Do not let a proposal appear in a paragraph as an implemented capability.

- [ ] **Step 4: Run a claim-language scan**

Run:

```powershell
rg -n -i "proven|universal|model-agnostic|quality pass|production-ready|NVFP4.*Marlin|Marlin.*NVFP4" docs/automatic-quantization-pipeline-progress.md
```

Expected: every match is either sourced, scoped, or explicitly marked as a goal/hypothesis.

### Task 2: Write the Markdown engineering update

**Files:**
- Create: `docs/automatic-quantization-pipeline-progress.md`

**Interfaces:**
- Consumes: Task 1 evidence ledger.
- Produces: canonical headings, prose, Mermaid diagram, status table, near-term plan, and reference links used by Task 3.

- [ ] **Step 1: Write the narrative**

Use this exact section order:

```text
Title and status deck
The goal
The compatibility gap
The three-layer pipeline
What MiniMax-M3 taught us
Problem 2: multimodality
Problem 3: official-checkpoint fallback
Near-term plan
Evidence and references
```

- [ ] **Step 2: Add the pipeline visualization**

Create a Mermaid flowchart that shows: intake; pre-quant static gate; all-layer smoke quantization with embedded probes and early termination; full-calibration run; post-quant checkpoint ABI gate; inference-engine smoke; paired evaluation; and publish/hold decisions. Include a plain-text summary directly below for renderers without Mermaid support.

- [ ] **Step 3: Add current-status and evidence tables**

Include a compact status table separating implemented, proven for MiniMax-M3, unresolved, and future work. Include a MiniMax case table with direct GPTQ, repaired GPTQ, external AWQ control, and in-house AWQ W4AFP8 as distinct rows.

- [ ] **Step 4: Validate Markdown structure**

Run:

```powershell
rg -n "^#|^```mermaid|Observed in this fork|Supported upstream|Proposed work|Application Binary Interface" docs/automatic-quantization-pipeline-progress.md
```

Expected: one H1, the approved section order, one Mermaid block, all three evidence labels, and ABI expanded on first use.

### Task 3: Build the standalone HTML article

**Files:**
- Create: `docs/automatic-quantization-pipeline-progress.html`

**Interfaces:**
- Consumes: Task 2 headings, claims, citations, tables, and pipeline stages.
- Produces: a standalone HTML5 document with no external runtime dependencies.

- [ ] **Step 1: Create the semantic document shell**

Use `<header>`, `<nav aria-label="Article sections">`, `<main>`, `<article>`, `<section>`, `<figure>`, `<table>`, `<aside>`, and `<footer>`. Add matching section anchors for the Markdown headings and a descriptive `<title>`/`meta description`.

- [ ] **Step 2: Implement the editorial visual system**

Define embedded CSS variables for paper, ink, hazard orange, verified green, rule gray, serif, and monospace stacks. Build a wide hero, narrow reading column, evidence chips, full-width reliability schematic, responsive tables, status cards, print rules, visible focus styles, and a `prefers-reduced-motion` override.

- [ ] **Step 3: Draw the pipeline as native HTML/CSS**

Represent each gate as a semantic ordered stage with explicit pass and fail branches. Show Layer 1 around both static boundaries, Layer 2 around all-layer smoke quantization and live probes, and Layer 3 around full calibration plus runtime/evaluation acceptance.

- [ ] **Step 4: Add minimal progressive enhancement**

Use a short embedded script to mark the active table-of-contents link with `IntersectionObserver`. The page must remain fully readable with JavaScript disabled.

- [ ] **Step 5: Check content parity**

Confirm that all material Markdown claims, evidence labels, numbers, caveats, and reference URLs appear in the HTML. Decorative HTML copy may be shorter, but it must not change technical meaning.

### Task 4: Validate both deliverables

**Files:**
- Verify: `docs/automatic-quantization-pipeline-progress.md`
- Verify: `docs/automatic-quantization-pipeline-progress.html`

**Interfaces:**
- Consumes: Tasks 2 and 3 artifacts.
- Produces: verified deliverables with no placeholders, broken local links, or HTML structural errors.

- [ ] **Step 1: Parse and inspect HTML**

Run a Python standard-library `html.parser.HTMLParser` check that reports balanced headings, required landmarks, duplicate IDs, external asset references, and missing link text. Expected: all required landmarks present, zero duplicate IDs, and zero external CSS/JS/font/image dependencies.

- [ ] **Step 2: Validate repository-relative references**

Extract local Markdown links and HTML repository links, resolve them from `docs/`, and require every referenced path to exist. Expected: zero missing repository files.

- [ ] **Step 3: Scan for review comments and placeholders**

Run:

```powershell
rg -n -i "TODO:|TBD:|\{\{|\}\}|Alright the|No need to|Again, this|It's more like|\(Note that" docs/automatic-quantization-pipeline-progress.md docs/automatic-quantization-pipeline-progress.html docs/superpowers/specs/2026-07-13-automatic-quantization-progress-blog-design.md
```

Expected: no matches.

- [ ] **Step 4: Render desktop and mobile views**

Open the HTML locally, capture one desktop viewport near 1440×1000 and one mobile viewport near 390×844, and inspect the hero, navigation, pipeline, tables, reference list, focus styles, overflow, and print-oriented hierarchy.

- [ ] **Step 5: Run final diff checks**

Run:

```powershell
git -c safe.directory=D:/Work/llm-compressor diff --check
git -c safe.directory=D:/Work/llm-compressor status --short
```

Expected: `diff --check` is clean; status lists only the reviewed spec, this plan, and the two requested deliverables unless the user has added unrelated changes.
