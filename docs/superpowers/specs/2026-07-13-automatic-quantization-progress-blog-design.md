# Automatic Quantization Progress Blog — Design

## Purpose and audience

Create an internal engineering update for collaborators that explains the fork's long-term goal, reports the MiniMax-M3 lessons that motivate the architecture, distinguishes completed work from unresolved research, and makes the next collaboration points explicit.

The post is not a product announcement or a claim of universal model support. It is a research and engineering progress report. Repository evidence will be labeled as internal observation; external technical statements will cite primary or official sources.

## Deliverables

- A portable Markdown article with a Mermaid pipeline diagram and a readable text fallback.
- A self-contained HTML article that works without a build step or external assets.
- Matching content, claim boundaries, references, and status language across both versions.
- An HTML visual design based on the selected **Engineering field notes** direction.

## Narrative structure

1. **Opening thesis:** Day-zero quantization is a compatibility problem, not only a bit-width conversion problem.
2. **Goal:** Accept a newly released model, produce several target formats, and send candidates through a repeatable quality-evaluation pipeline.
3. **Problem 1 — compatibility across two boundaries:**
   - Pre-quantization: the quantizer must understand architecture, module mappings, fused layouts, and algorithm-specific transformations.
   - Post-quantization: the serving engine must interpret checkpoint metadata, tensor packing, model-specific components, and kernel layout correctly.
   - MiniMax-M3 case evidence: GPTQ serving became structurally healthy after checkpoint/config checks and serving fixes; AWQ W4AFP8 still produced degenerate output after full calibration, despite passing structural and serving checks. This is presented as evidence that algorithm-specific transforms such as AWQ smoothing require their own probes.
4. **Three-layer control system:**
   - Layer 1: pre- and post-quantization static gates.
   - Layer 2: diagnostic probes inside the run, with fail-fast termination.
   - Layer 3: full-calibration quantization followed by serving and evaluation gates.
5. **Problem 2 — multimodality:** explain why modality-specific calibration coverage matters, then explicitly defer implementation.
6. **Problem 3 — official-checkpoint fallback:** verify and carefully scope claims about NVFP4 checkpoints, Hopper support, Marlin, and W4A16 fallback. Separate established upstream behavior from the proposed runtime W4A8/W4AFP8 conversion research question.
7. **Near-term plan and collaboration requests:** harden the text-only pipeline first; seek kernel expertise for the fallback experiment; leave multimodality as a later workstream.

## Evidence and claim policy

Every substantive claim will use one of three labels or equivalent prose:

- **Observed in this fork:** supported by repository code, reports, or stored artifacts.
- **Supported upstream:** supported by official documentation, source code, issue/PR evidence, hardware documentation, or an original paper.
- **Hypothesis / proposed work:** not presented as an implemented capability.

The article will avoid exposing private absolute paths, node names, credentials, or unpublished collaborator identities. Repository-relative file references are acceptable for internal reproducibility.

The NVFP4 fallback section will be rewritten if verification does not support the initial formulation. In particular, the post will not equate weight storage format with arithmetic precision, nor claim that Marlin performs an NVFP4-to-W4A16 conversion unless official vLLM source or documentation establishes that exact path.

## Main visualization

The pipeline diagram will emphasize two compatibility boundaries and three layers of protection:

```text
New model
   │
   ▼
Architecture + format intake
   │
   ├── Pre-quant static gate ── fail → compatibility report
   ▼
Smoke calibration / quantization
   │     └── live diagnostic probes ── anomaly → stop early
   ▼
Candidate checkpoint
   │
   ├── Post-quant static gate ── fail → packing/config report
   ▼
Full calibration run
   ▼
Inference-engine load + serve smoke
   │
   ├── Runtime gate ── fail → loader/kernel diagnosis
   ▼
Evaluation suite + baseline comparison
   │
   └── pass → publish validated quantized artifact
```

Static checks do not imply semantic quality. The final evaluation gate remains authoritative for quality acceptance.

## HTML visual direction

- **Tone:** engineering field notes — editorial, rigorous, and visibly work-in-progress.
- **Palette:** warm paper background, near-black ink, oxidized orange for hazards, and restrained green for verified states.
- **Typography:** local serif and monospace stacks to avoid network dependencies; strong display hierarchy and compact technical annotations.
- **Layout:** wide editorial hero, narrow reading column, margin notes/status chips, full-width pipeline figure, evidence callouts, and a concise end-state roadmap.
- **Interaction:** lightweight table of contents, reduced-motion-safe reveal behavior, print styles, accessible contrast, semantic headings, and keyboard-visible links.
- **Differentiator:** the central pipeline appears like an annotated reliability schematic rather than a generic flowchart.

## Validation

- Confirm Markdown headings, links, Mermaid syntax, and relative repository references.
- Parse the HTML and verify that it is self-contained, responsive, printable, and free of external runtime dependencies.
- Render the HTML in a browser at desktop and mobile widths and visually inspect the hero, pipeline, tables, callouts, and references.
- Cross-check both artifacts for equivalent claims and citations.
- Re-run a placeholder and ambiguity scan before delivery.

## Out of scope

- Implementing the quantization pipeline itself.
- Resolving the MiniMax-M3 AWQ quality failure.
- Implementing multimodal calibration.
- Writing a custom NVFP4-to-W4A8/W4AFP8 runtime kernel.
- Publishing externally or sanitizing the article for a public audience.
