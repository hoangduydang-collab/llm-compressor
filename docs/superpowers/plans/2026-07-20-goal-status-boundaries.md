# Goal Status Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mark Goals 1 and 2 done and Goal 3 active across the HTML research updates while clearly assigning remaining AWQ checkpoint work to Goal 3.

**Architecture:** Update only the program overview and the existing Goal 1/2 standalone pages. Preserve layout, navigation, evidence, and measurements; change status styling and prose at the goal boundaries.

**Tech Stack:** Semantic HTML5 and embedded CSS; standard-library Python for verification.

## Global Constraints

- Modify only `docs/automatic-quantization-pipeline-progress.html`, `docs/goals/goal-1-fast-parallel-quantization.html`, and `docs/goals/goal-2-temporary-evaluation-pipeline.html`.
- Keep `PROJECT_GOALS.md` byte-for-byte unchanged at Git object hash `090375ebde364cab8024bff44f99b743697c1aa0`.
- Do not create a Goal 3 page.
- Keep Goal 6 planned and benchmark-gated.
- Preserve existing navigation, responsive layout, result values, and evidence links.
- Do not modify the unrelated untracked `artifacts/m3-cudagraph-rca-dryrun/` directory.

---

### Task 1: Update the program overview

**Files:**
- Modify: `docs/automatic-quantization-pipeline-progress.html`

**Interfaces:**
- Consumes: the approved HTML-only status model.
- Produces: the index-level status shown to collaborators.

- [ ] **Step 1: Add a completed-status style**

Add beside the existing status classes:

```css
.status.done{color:#fff;background:var(--verified);}
```

- [ ] **Step 2: Update overview framing**

Replace the hero deck with wording equivalent to:

```html
<p class="deck">Six long-term goals with clear boundaries: parallel quantization and the evaluation pipeline are complete foundations; producing a correct in-house AWQ checkpoint is the active focus.</p>
```

Remove statements that the page exactly reproduces or mirrors current
`PROJECT_GOALS.md` statuses. Continue linking that file as the source of durable goal
definitions.

- [ ] **Step 3: Update the three goal cards**

Use:

```html
<span class="status done">Done</span>
```

for Goals 1 and 2. Describe Goal 1 as having met the 4–8 hour distributed-calibration
target and Goal 2 as a working fail-closed paired evaluation pipeline.

Use:

```html
<span class="status working">Work in progress</span>
```

for Goal 3. Its description must say the active work is producing a correct,
gate-passing in-house AWQ checkpoint; it must not say the work is tracked under Goal 1.

- [ ] **Step 4: Make Goal 3 the sole current focus**

Use:

```html
<h3>Current session focus: Goal 3</h3>
<p>Goals 1 and 2 are completed foundations. The active work is producing and qualifying a correct in-house AWQ checkpoint. Goals 4–6 remain future or benchmark-gated work.</p>
```

Update the “How to read these” copy so existing field notes may document completed or
planned goals, not only active goals.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
python -c "from pathlib import Path; t=Path('docs/automatic-quantization-pipeline-progress.html').read_text(encoding='utf-8'); assert t.count('status done') == 2; assert 'Current session focus: Goal 3' in t; assert 'After / with Goal 1' not in t; assert 'mirrors <code>PROJECT_GOALS.md</code>' not in t"
```

Expected: exit code 0.

Commit:

```powershell
git add -- docs/automatic-quantization-pipeline-progress.html
git commit -m "docs: focus goal overview on active AWQ work"
```

---

### Task 2: Close Goal 1 at the parallelization boundary

**Files:**
- Modify: `docs/goals/goal-1-fast-parallel-quantization.html`

**Interfaces:**
- Consumes: seven-hour full-calibration evidence already presented on the page.
- Produces: a completed Goal 1 note with remaining correctness work handed to Goal 3.

- [ ] **Step 1: Add and apply completed status**

Add:

```css
.status.done{color:#fff;background:var(--verified);}
```

Use:

```html
<span class="status done">Done</span>
```

in the hero.

- [ ] **Step 2: Rewrite the completion framing**

Use a deck equivalent to:

```html
<p class="deck">Done: distributed AWQ met the 4–8 hour calibration target with a seven-hour full run. The remaining checkpoint-correctness work belongs to Goal 3, not parallelization.</p>
```

Retain the r4 timing and failed correctness-gate facts.

- [ ] **Step 3: Replace open-goal wording with a Goal 3 handoff**

Replace “Current status” and “Acceptance path” with one concise scope-boundary section:

```html
<section id="handoff">
  <h2><span class="number">03 / Boundary</span>Checkpoint correctness moves to Goal 3</h2>
  <p>Goal 1 is complete because the distributed path demonstrated full calibration inside the target window. Producing a correct AWQ checkpoint, passing post-save and serving gates, and running quality evaluation are now Goal 3 work.</p>
</section>
```

Keep the root-cause fix as context only if it supports this handoff. Remove statements
that Goal 1 “remains open” or has an unfinished acceptance path.

- [ ] **Step 4: Correct the footer**

State that the seven-hour full run achieved the parallelization target, while its
checkpoint was rejected by a correctness gate and is not a usable AWQ artifact.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
python -c "from pathlib import Path; t=Path('docs/goals/goal-1-fast-parallel-quantization.html').read_text(encoding='utf-8'); assert 'status done' in t; assert 'Goal 1 is complete' in t; assert 'Goal 3 work' in t; assert 'goal remains open' not in t.lower(); assert 'target is not yet documented as achieved' not in t"
```

Expected: exit code 0.

Commit:

```powershell
git add -- docs/goals/goal-1-fast-parallel-quantization.html
git commit -m "docs: mark parallel quantization goal complete"
```

---

### Task 3: Close Goal 2 at the working-pipeline boundary

**Files:**
- Modify: `docs/goals/goal-2-temporary-evaluation-pipeline.html`

**Interfaces:**
- Consumes: completed four-model comparison and reproducible harness evidence.
- Produces: a completed Goal 2 note that treats future evaluations as pipeline usage.

- [ ] **Step 1: Add and apply completed status**

Add:

```css
.status.done{color:#fff;background:var(--verified);}
```

Use:

```html
<span class="status done">Done</span>
```

in the hero.

- [ ] **Step 2: State the completion criterion**

Update the deck and purpose section to say the deliverable is a working, reproducible,
fail-closed paired evaluation pipeline. The existing four-model results and harness
contract are its completion evidence.

- [ ] **Step 3: Replace future-work framing**

Replace the `Next` section with:

```html
<section id="available">
  <h2><span class="number">04 / Available</span>Ready for the next checkpoint</h2>
  <p>Goal 2 is complete: the pipeline can evaluate a qualified checkpoint against BF16 and existing quantized comparators under a shared contract.</p>
  <p>Generating additional merged reports and evaluating future in-house AWQ checkpoints are uses of this pipeline, not unfinished pipeline development.</p>
</section>
```

Keep the seeded-subset and public-leaderboard limitation.

- [ ] **Step 4: Verify and commit**

Run:

```powershell
python -c "from pathlib import Path; t=Path('docs/goals/goal-2-temporary-evaluation-pipeline.html').read_text(encoding='utf-8'); assert 'status done' in t; assert 'Goal 2 is complete' in t; assert 'uses of this pipeline' in t; assert '04 / Next' not in t; assert t.count('<table>') == 1"
```

Expected: exit code 0.

Commit:

```powershell
git add -- docs/goals/goal-2-temporary-evaluation-pipeline.html
git commit -m "docs: mark evaluation pipeline goal complete"
```

---

### Task 4: Verify HTML scope and consistency

**Files:**
- Verify: `docs/automatic-quantization-pipeline-progress.html`
- Verify: `docs/goals/goal-1-fast-parallel-quantization.html`
- Verify: `docs/goals/goal-2-temporary-evaluation-pipeline.html`
- Verify unchanged: `PROJECT_GOALS.md`

**Interfaces:**
- Consumes: all three status updates.
- Produces: final structural, link, and scope evidence.

- [ ] **Step 1: Parse pages and validate local links**

Run:

```powershell
@'
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = set()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)

pages = [
    Path("docs/automatic-quantization-pipeline-progress.html"),
    Path("docs/goals/goal-1-fast-parallel-quantization.html"),
    Path("docs/goals/goal-2-temporary-evaluation-pipeline.html"),
]
for page in pages:
    parser = PageParser()
    parser.feed(page.read_text(encoding="utf-8"))
    assert {"title", "main", "footer"} <= parser.tags
    for href in parser.hrefs:
        parsed = urlparse(href)
        if parsed.scheme or href.startswith("#"):
            continue
        assert (page.parent / parsed.path).resolve().exists(), f"{page}: missing {href}"
print("parsed and link-checked", len(pages), "pages")
'@ | python -
```

Expected: `parsed and link-checked 3 pages`.

- [ ] **Step 2: Verify the status contract and protected file**

Run:

```powershell
python -c "from pathlib import Path; o=Path('docs/automatic-quantization-pipeline-progress.html').read_text(encoding='utf-8'); g1=Path('docs/goals/goal-1-fast-parallel-quantization.html').read_text(encoding='utf-8'); g2=Path('docs/goals/goal-2-temporary-evaluation-pipeline.html').read_text(encoding='utf-8'); assert o.count('status done') == 2; assert 'Current session focus: Goal 3' in o; assert 'status done' in g1 and 'status done' in g2"
git hash-object -- PROJECT_GOALS.md
```

Expected hash: `090375ebde364cab8024bff44f99b743697c1aa0`.

- [ ] **Step 3: Review the implementation diff**

Run:

```powershell
git diff HEAD~3 --check
git diff HEAD~3 --stat
git status --short
```

Expected: no whitespace errors; implementation commits modify exactly the three HTML
files; the unrelated artifact directory remains untracked and untouched.
