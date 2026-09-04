# MiniMax-M3 Synthetic-versus-Real Spec-Dec Prompt Note Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone Markdown note that documents and correctly scopes the MiniMax-M3 EAGLE3 synthetic-prompt versus real-ShareGPT-prompt speculative-decoding result.

**Architecture:** Add one focused, collaborator-facing document under `docs/`; it summarizes the committed Wave 1 and Wave 2 evidence without changing the primary 1,400-line study. The document links back to the primary study and design packet, and names the NFS raw-evidence locations for reproducibility.

**Tech Stack:** Markdown, Git, existing MiniMax-M3 EAGLE3 result documents and aggregation scripts.

## Global Constraints

- Create only `docs/minimax-m3-specdec-synthetic-vs-real-prompts.md`; do not alter the primary report.
- Use the four top-level sections exactly: `Prompt sources`, `Experiment setup`, `Results`, and `Conclusions`.
- Treat `docs/m3-specdec-eagle3.md` as the authority for reported figures.
- State explicitly that Wave 1 and Wave 2 were separate windows, so their comparison is not a directly paired A/B.
- Do not claim local recomputation of raw counters; raw artifacts are on cluster NFS.
- Do not make claims about the collaborator's result without that experiment's configuration and raw measurements.

---

### Task 1: Write and validate the standalone experiment note

**Files:**
- Create: `docs/minimax-m3-specdec-synthetic-vs-real-prompts.md`
- Reference: `docs/m3-specdec-eagle3.md:40-50, 55-90, 125-135, 165-240`
- Reference: `M3_SPECDEC_EAGLE3_PLAN.md:50-70, 114-150, 155-214`
- Reference: `pipeline/slurm/run_specdec_eagle3_srun.sh`
- Reference: `pipeline/slurm/run_specdec_wave2_srun.sh`
- Reference: `pipeline/specdec_aggregate.py`
- Reference: `pipeline/specdec_wave2_aggregate.py`

**Interfaces:**
- Consumes: committed Wave 1 and Wave 2 measurements plus their documented launch and aggregation provenance.
- Produces: a linkable, self-contained interpretation note for collaborators.

- [ ] **Step 1: Create the document with source and setup sections**

Write the document title `# MiniMax-M3 EAGLE3 speculative decoding: synthetic vs real prompts`.

Under `## Prompt sources`, identify the comparison:

```markdown
| Arm | Prompt source | Construction | Input length | Output policy |
|---|---|---|---:|---|
| Wave 1 synthetic | AA-style input generator | Random synthetic tokens | 1k and 10k tokens | Natural stopping |
| Wave 2 real | aiperf `--public-dataset sharegpt` | Real ShareGPT first-turn user messages | mean ≈227 tokens | Natural stopping |
```

Explain that aiperf public-dataset loaders retain only the first conversation message, so the ShareGPT arm is real single-turn text rather than multi-turn chat. State that neither arm uses `ignore_eos` or a forced minimum output in this comparison.

Under `## Experiment setup`, state the common configuration:

```markdown
| Component | Configuration |
|---|---|
| Target | In-house MiniMax-M3 GPTQ W4AFP8 ABI overlay |
| Speculative drafter | `Inferact/MiniMax-M3-EAGLE3` at `44cafa5ace418d8b22e2958df0c6aa1f2476842c` |
| Serving | vLLM 0.24.0; Humming indexed 0.1.10; FP8 KV cache; prefix caching |
| Hardware/topology | One 8×H100 node, TP8/EP8 |
| Decoding comparison | k=0 control versus EAGLE3 k=3; temperature 0.6; natural stopping |
```

Name Wave 1 `20260727T061506Z` and Wave 2 `20260727T064934Z-wave2`, their implementation/result commits, launch scripts, and aggregation scripts. Add a bold cross-window caveat.

- [ ] **Step 2: Add the numerical results**

Under `## Results`, add a primary table with only directly comparable k=3 values:

```markdown
| Workload | Concurrency | Accepted length | Per-position acceptance | k=0 → k=3 tok/s/user | Decode speedup |
|---|---:|---:|---|---:|---:|
| Wave 1 synthetic (AA-style) | 1 | 2.450 | 0.70 / 0.46 / 0.29 | 137.5 → 236.0 | 1.72× |
| Wave 2 real (ShareGPT) | 1 | 2.473 | 0.690 / 0.459 / 0.324 | 137.9 → 249.8 | 1.81× |
| Wave 2 real (ShareGPT) | 10 | 2.503 | 0.700 / 0.479 / 0.324 | 77.7 → 128.8 | 1.66× |
```

Add a short supporting table for server throughput at concurrency 1:

```markdown
| Workload | k=0 → k=3 server tok/s | Speedup |
|---|---:|---:|
| Synthetic | 133.7 → 215.0 | 1.61× |
| ShareGPT | 132.7 → 222.3 | 1.68× |
```

Calculate and state the primary acceptance delta as `2.473 - 2.450 = +0.023` or approximately `+0.9%`. Include the report's controlled contrast that forced 8k output (`ignore_eos`) raised acceptance from `2.473` to `3.286` (`+33%`), separating output shape from prompt source.

- [ ] **Step 3: State conclusions and limits**

Under `## Conclusions`, make each interpretation explicit:

1. Synthetic random tokens and natural ShareGPT prompts had nearly identical accepted length in the documented conc-1 comparison; the primary report treats the +0.9% difference as noise.
2. Equal acceptance does not require equal throughput: the observed per-user speedups were 1.72× and 1.81×, respectively.
3. The earlier 3.20–3.35 acceptance estimate from hand-picked, greedy, forced-continuation prompts must not be used as a natural-real-prompt result; Wave 2 superseded the resulting 2.25× inference with the measured 1.81× ShareGPT result.
4. Within this study, output shape (+33%), content domain, and temperature (+4%) were stronger observed factors than prompt naturalness.

Add a `### Evidence and reproduction` subsection that links to the primary study and design packet, names the two NFS result directories, and gives both aggregation commands. Say the local checkout contains the documentation and scripts but not the raw counter-delta artifacts.

- [ ] **Step 4: Validate the document**

Run:

```powershell
git diff --check -- docs/minimax-m3-specdec-synthetic-vs-real-prompts.md
rg -n "^(## Prompt sources|## Experiment setup|## Results|## Conclusions)$" docs/minimax-m3-specdec-synthetic-vs-real-prompts.md
rg -n "2\.450|2\.473|1\.72×|1\.81×|cross-window|not a directly paired" docs/minimax-m3-specdec-synthetic-vs-real-prompts.md
```

Expected: no whitespace errors; all four required headings appear once; the key acceptance, throughput, and comparability terms are present.

- [ ] **Step 5: Commit the note**

Run:

```powershell
git add -- docs/minimax-m3-specdec-synthetic-vs-real-prompts.md
@'
docs(specdec): document synthetic versus real prompt result

Add a standalone, evidence-scoped note for the MiniMax-M3 EAGLE3 comparison of
AA-style synthetic prompts and natural ShareGPT prompts.
'@ | git commit -F -
```

Expected: one new documentation file committed without staging unrelated working-tree changes.
