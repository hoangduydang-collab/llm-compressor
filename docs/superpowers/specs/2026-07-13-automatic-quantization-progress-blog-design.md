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
   - MiniMax-M3 case evidence is now presented as a sequence of discriminating gates rather than a simple GPTQ-pass/AWQ-fail comparison:
     - the original in-house GPTQ checkpoint failed a CPU-only serving-ABI gate with 228 runtime namespace/ignore mismatches;
     - a metadata-only portable overlay preserved tensor payloads, passed the gate, and subsequently produced coherent smoke generations;
     - repaired GPTQ and the external cyankiwi AWQ control both completed paired 2,047-token probes and five-task smoke runs without empty outputs or periodic loops;
     - the in-house AWQ W4AFP8 artifact remains unresolved, and attempted repair re-quantizations did not produce complete checkpoints;
     - representative AWQ diagnostics exposed additional harness/tracing failure modes, so they are evidence for fail-fast instrumentation, not yet a verdict on the production smoothing bug.
4. **Three-layer control system:**
   - Layer 1: pre- and post-quantization static gates. The pre-quantization planner is implemented for AWQ/GPTQ and MiniMax-M3 is its first regression profile, but the first real CLI run exposed a meta-device MoE offload incompatibility. A narrow fix and CPU regression test exist; cluster verification remains pending. The post-quantization ABI checker is proven for the documented MiniMax-M3 compressed-tensors profile, not model-agnostic.
   - Layer 2: representative-layer canaries and diagnostic probes inside the run, with fail-fast termination and durable per-layer evidence. Current AWQ harness results also demonstrate that a probe must validate its own execution coverage before drawing a model-quality conclusion.
   - Layer 3: guarded full-calibration quantization followed by serving smoke, teacher-forced distributional probes, generation-health checks, and paired evaluation gates.
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

Newly merged reports supersede the earlier shorthand that “GPTQ passed while AWQ failed.” The blog will distinguish the repaired in-house GPTQ checkpoint, the coherent external AWQ control, and the unresolved in-house AWQ W4AFP8 artifact. Small smoke-task scores are diagnostic only and will not be presented as statistically meaningful benchmark results.

## Main visualization

The pipeline diagram will emphasize two compatibility boundaries and three layers of protection:

```text
New model + target format + target runtime
   │
   ▼
Architecture + format intake
   │
   ├── Pre-quant static gate ── fail → compatibility report
   ▼
Representative-layer canary
   │     └── coverage + numerical probes ── anomaly → stop early
   ▼
Guarded full calibration / quantization
   │     └── per-layer diagnostics ── anomaly → stop + preserve evidence
   ▼
Candidate checkpoint + provenance
   │
   ├── Post-quant static gate ── fail → packing/config report
   ▼
Inference-engine load + serve smoke
   │
   ├── Runtime gate ── fail → loader/kernel diagnosis
   ▼
Teacher-forced probe + generation health + paired evaluation
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

## Repository evidence to cite

- `M3_PREQUANT_REAL_CLI_FAILURE_REPORT.md` for the first real pre-quant gate boundary, narrow fix, and pending cluster verification.
- `docs/quantization-static-serving-preflight-status-and-roadmap.md` and `M3_STATIC_ABI_GATE_REPORT.md` for the scope and 228-error GPTQ ABI failure.
- `M3_GPTQ_REPAIRED_ABI_PREFLIGHT_REPORT.md` for the tensor-preserving metadata overlay and three-model static preflight.
- `M3_3MODEL_GPTQ_AWQ_FINAL_REPORT.md` for the paired repaired-GPTQ/external-AWQ smoke result.
- `M3_AWQ_REQUANTIZATION_REPORT.md` and `M3_AWQ_REPRESENTATIVE_RERUN_REPORT.md` for incomplete AWQ repair attempts and probe self-validation lessons.
- `pipeline/prequant_compatibility.py`, `pipeline/m3_serve_abi.py`, `pipeline/m3_guarded_full.py`, and `pipeline/evalsuite/` for the implemented control surfaces.

## Out of scope

- Implementing the quantization pipeline itself.
- Resolving the MiniMax-M3 AWQ quality failure.
- Implementing multimodal calibration.
- Writing a custom NVFP4-to-W4A8/W4AFP8 runtime kernel.
- Publishing externally or sanitizing the article for a public audience.
