# Concise Goal Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the three active-goal HTML pages as accurate, collaborator-facing updates readable in about three minutes each.

**Architecture:** Preserve each page's existing standalone HTML shell, visual theme, navigation, and footer, but replace the long article body with a small set of outcome-focused sections. Keep only decision-relevant evidence and link to repository reports for details.

**Tech Stack:** Semantic HTML5 and embedded CSS; no JavaScript or new dependencies.

## Global Constraints

- Target roughly 350–500 prose words per page, excluding tables and source labels.
- Write to busy collaborators; omit personal history, obsolete attempts, and run-by-run debugging.
- Preserve retained measurements exactly and distinguish results, failed gates, and future plans.
- Keep navigation, responsive styling, status labels, and working evidence links.
- Do not modify the unrelated untracked `artifacts/m3-cudagraph-rca-dryrun/` directory.

---

### Task 1: Rewrite Goal 1 around full-calibration evidence

**Files:**
- Modify: `docs/goals/goal-1-fast-parallel-quantization.html`

**Interfaces:**
- Consumes: full-run evidence in `BUGS_AND_FIXES.md`, especially full r4 and the corrected r5 follow-up.
- Produces: a concise Goal 1 field note linked from the unchanged program overview.

- [ ] **Step 1: Replace the hero summary**

State the outcome directly:

```html
<p class="deck">Distributed AWQ now reaches the 4–8 hour calibration target. The latest completed full run quantized in 7 hours, but a correctness gate rejected it; the corrected run remains the path to a usable checkpoint.</p>
```

Retain compact metadata for the 4–8 hour target, AWQ/GPTQ scope, and MiniMax-M3.

- [ ] **Step 2: Replace the article with four short sections**

Use these sections and facts:

1. **Objective and approach** — target a full AWQ/GPTQ process in 4–8 hours using the fork's distributed calibration path.
2. **Full-calibration result** — full r4 quantized for 7h00m; report its checkpoint/gate outcome, including that 1,423 of 21,888 routed-expert modules failed the dequant-vs-base gate because AWQ smoothing scales were inflated on dead norm channels. Do not present the checkpoint as usable.
3. **Current status** — summarize the dead-channel root-cause fix and state the latest corrected full-run status supported by repository evidence at edit time.
4. **Next steps** — complete the corrected run, pass checkpoint/scale/serving gates, then record end-to-end speedup and evaluate quality through Goal 2.

Remove smoke timing tiles, the one-GPU mistake, custom expert-scatter detour, prior-art narrative, and r2–r9 chronology.

- [ ] **Step 3: Keep a compact evidence list**

Link only:

```html
<a href="../../BUGS_AND_FIXES.md">BUGS_AND_FIXES.md</a>
<a href="../../M3_QUANT_SPEEDUP_PLAN.md">M3_QUANT_SPEEDUP_PLAN.md</a>
<a href="../../PROJECT_GOALS.md">PROJECT_GOALS.md</a>
```

- [ ] **Step 4: Verify the page**

Run:

```powershell
python -c "from pathlib import Path; from bs4 import BeautifulSoup; p=Path('docs/goals/goal-1-fast-parallel-quantization.html'); s=BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser'); assert s.title and s.find('nav') and s.find('main') and s.find('footer'); text=s.find('main').get_text(' ',strip=True); assert '7h00m' in text or '7 hours' in text; assert '71 min' not in text"
```

Expected: exit code 0. If BeautifulSoup is unavailable, use the standard-library HTML parser in Task 4.

- [ ] **Step 5: Commit Goal 1**

```powershell
git add -- docs/goals/goal-1-fast-parallel-quantization.html
git commit -m "docs: condense full-calibration goal update"
```

---

### Task 2: Reduce Goal 2 to results and reproducible harness essentials

**Files:**
- Modify: `docs/goals/goal-2-temporary-evaluation-pipeline.html`

**Interfaces:**
- Consumes: existing four-model result table and harness contract on the page.
- Produces: a concise Goal 2 result and setup summary linked from the unchanged program overview.

- [ ] **Step 1: Shorten the hero and purpose**

Explain in two short paragraphs that the harness measures paired model-to-model fidelity on identical prompts, samples, and decoding, and that scores are not public-leaderboard comparable.

- [ ] **Step 2: Keep one result table and one interpretation**

Retain the current four-way absolute-score table exactly:

```text
GPQA-Diamond: 80.7 / 61.7 / 81.7 / 79.3
MMLU-Pro:     73.3 / 69.0 / 76.3 / 76.7
GSM8K-CoT:    77.7 / 75.0 / 84.3 / 81.3
AIME25:       58.9 / 24.4 / 57.8 / 62.2
```

Order remains in-house GPTQ, cyankiwi AWQ, official MXFP8, BF16. Remove the duplicate BF16-delta table. Summarize that in-house GPTQ remains within a few points of BF16, MXFP8 is roughly baseline-equivalent, and the external AWQ comparator degrades materially.

- [ ] **Step 3: Replace detailed reproduction material with a compact harness setup**

Keep:

```text
lm-evaluation-harness 0.4.12, direct vLLM backend
Models: in-house GPTQ W4A8, cyankiwi AWQ W4A16, official MXFP8 W8A16, BF16
Tasks/subsets: GPQA 100, MMLU-Pro 100, GSM8K 100, AIME25 30
Seeds: 42, 1234, 4158
Generation: thinking enabled, temperature 1.0, top_p 0.95, max_gen_toks 16384
Serving: TP8 for quantized/MXFP8 models; TP16 across two nodes for BF16
```

Remove raw hashes, exact checkpoint paths, shell commands, obsolete r3 warning, completion chronology, and BF16 troubleshooting.

- [ ] **Step 4: Add limitations and next steps**

State that subset scores support internal model-to-model decisions only. List two next steps:

1. generate the merged four-way report;
2. apply the ready quality-evaluation pipeline to the next real full-calibration checkpoint.

- [ ] **Step 5: Verify and commit Goal 2**

Run:

```powershell
python -c "from pathlib import Path; t=Path('docs/goals/goal-2-temporary-evaluation-pipeline.html').read_text(encoding='utf-8'); assert t.count('<table>') == 1; assert '80.7%' in t and '62.2%' in t; assert 'lm-evaluation-harness' in t; assert 'real full-calibration checkpoint' in t"
```

Expected: exit code 0.

Then:

```powershell
git add -- docs/goals/goal-2-temporary-evaluation-pipeline.html
git commit -m "docs: focus evaluation update on results and setup"
```

---

### Task 3: Convert Goal 6 into a short future-work plan

**Files:**
- Modify: `docs/goals/goal-6-hopper-packed-nvfp4-w4a8.html`

**Interfaces:**
- Consumes: approved direction in `docs/superpowers/specs/2026-07-19-hopper-packed-nvfp4-w4a8-fallback-design.md`.
- Produces: a concise, explicitly unmeasured Goal 6 future-work page.

- [ ] **Step 1: Make planned status unmistakable**

Use:

```html
<p class="deck">Future work: test whether official NVFP4 checkpoints can keep four-bit weights packed on Hopper while converting fragments inside the GEMM for FP8 Tensor Core execution.</p>
```

Keep the planned/benchmark-gated status and state that no hardware measurements exist.

- [ ] **Step 2: Replace the body with four sections**

1. **Why this matters** — avoid persistent FP8 expansion while supporting official NVFP4 releases on H100/H200.
2. **Proposed direction** — extend Humming's existing packed E2M1/FP8 path; retain Marlin W4A16 as the fail-closed fallback.
3. **Decision gates** — proceed only if correctness passes, persistent memory remains close to packed W4A16, and at least one decision-relevant workload shows useful speedup without harming the primary bucket.
4. **Staged plan** — one-layer SM90 probe, dense proof of concept, serving qualification, then separate MoE production work only after all earlier gates pass.

Remove conversion equations, detailed group-16 mechanics, estimates, prototype inventory, analytical arithmetic, exhaustive thresholds, and contingency-kernel details.

- [ ] **Step 3: Keep only high-value sources**

Link the design spec, implementation plan, handoff, project goals, and upstream vLLM quantization documentation.

- [ ] **Step 4: Verify and commit Goal 6**

Run:

```powershell
python -c "from pathlib import Path; t=Path('docs/goals/goal-6-hopper-packed-nvfp4-w4a8.html').read_text(encoding='utf-8'); assert 'Future work' in t; assert 'no hardware measurements' in t.lower(); assert 'B8 = Q_E4M3' not in t; assert 'person-month' not in t"
```

Expected: exit code 0.

Then:

```powershell
git add -- docs/goals/goal-6-hopper-packed-nvfp4-w4a8.html
git commit -m "docs: shorten Hopper NVFP4 future-work plan"
```

---

### Task 4: Validate all three standalone pages

**Files:**
- Verify: `docs/goals/goal-1-fast-parallel-quantization.html`
- Verify: `docs/goals/goal-2-temporary-evaluation-pipeline.html`
- Verify: `docs/goals/goal-6-hopper-packed-nvfp4-w4a8.html`

**Interfaces:**
- Consumes: the three rewritten pages.
- Produces: structural, link, and length evidence for final review.

- [ ] **Step 1: Parse all pages with the standard library**

Run:

```powershell
python -c "from html.parser import HTMLParser; from pathlib import Path; files=list(Path('docs/goals').glob('*.html')); assert len(files)==3; [HTMLParser().feed(p.read_text(encoding='utf-8')) for p in files]; print('parsed', len(files), 'pages')"
```

Expected: `parsed 3 pages`.

- [ ] **Step 2: Check local links and approximate reading length**

Run:

```powershell
@'
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
import re

class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.in_main = False
        self.main_text = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.hrefs.append(values["href"])
        if tag == "main":
            self.in_main = True

    def handle_endtag(self, tag):
        if tag == "main":
            self.in_main = False

    def handle_data(self, data):
        if self.in_main:
            self.main_text.append(data)

for page in Path("docs/goals").glob("*.html"):
    parser = PageParser()
    parser.feed(page.read_text(encoding="utf-8"))
    for href in parser.hrefs:
        parsed = urlparse(href)
        if parsed.scheme or href.startswith("#"):
            continue
        target = (page.parent / parsed.path).resolve()
        assert target.exists(), f"{page}: missing {href}"
    words = re.findall(r"\b[\w–-]+\b", " ".join(parser.main_text))
    print(f"{page.name}: {len(words)} main words")
'@ | python -
```

Expected: three page names with word counts and no missing-link assertion. Manually confirm each count is consistent with a roughly three-minute read once tables are considered.

- [ ] **Step 3: Review the final diff**

Run:

```powershell
git diff HEAD~3 --check
git diff HEAD~3 --stat
git status --short
```

Expected: no whitespace errors; only the three goal pages changed by the implementation commits; the pre-existing untracked artifact directory remains untouched.
