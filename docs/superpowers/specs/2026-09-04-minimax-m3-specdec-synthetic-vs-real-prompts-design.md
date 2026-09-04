# Design: MiniMax-M3 speculative-decoding prompt-source note

## Purpose

Create a standalone, collaborator-facing Markdown note documenting the in-house
comparison of MiniMax-M3 EAGLE3 speculative decoding on synthetic random-token
prompts and real ShareGPT prompts. The note records the evidence and its limits;
it does not attempt to explain or adjudicate a collaborator's differing result.

## Location and audience

Write `docs/minimax-m3-specdec-synthetic-vs-real-prompts.md`. The audience is
engineers evaluating how representative a speculative-decoding benchmark is of
production traffic.

## Content

The note has four required sections:

1. **Prompt sources** — explain that Wave 1 used the AA-style synthetic
   random-token generator with 1k and 10k input lengths, while Wave 2 Phase A
   used aiperf's public ShareGPT dataset (real single-turn user messages; mean
   input length about 227 tokens). State that neither source represents
   multi-turn conversation because the loader retains only the first turn.
2. **Experiment setup** — state the common model, drafter, topology, serving
   stack, k=3, temperature 0.6, and natural stopping configuration. Identify
   the separate Wave 1 and Wave 2 windows, their scripts and commits. Make
   clear that the comparison is cross-window, not a directly paired A/B.
3. **Results** — report acceptance length, per-position acceptance, and
   throughput for synthetic and real inputs. Include the real concurrency-10
   measurement as supporting evidence. Separately show the forced-continuation
   result to avoid conflating output shape with prompt source.
4. **Conclusions** — conclude that observed prompt naturalness changed
   acceptance by approximately 1% at concurrency 1 (2.45 to 2.473), within
   the report's noise interpretation. State that end-to-end speed nonetheless
   differed (1.72x vs 1.81x), and that output shape and content domain were
   larger observed factors.

## Evidence and reproducibility

Link to `docs/m3-specdec-eagle3.md` as the primary authority, and to
`M3_SPECDEC_EAGLE3_PLAN.md` for the pre-declared design. Cite:

- Wave 1 `20260727T061506Z`, commits `88d1997e` and `bc6344bf`;
- Wave 2 `20260727T064934Z-wave2`, commits `15b1aab1` and `48f2ea96`;
- `pipeline/slurm/run_specdec_eagle3_srun.sh` and
  `pipeline/slurm/run_specdec_wave2_srun.sh`;
- `pipeline/specdec_aggregate.py` and `pipeline/specdec_wave2_aggregate.py`.

Record the raw NFS artifact paths and aggregation commands, but do not claim
the local repository independently recomputes the raw counter deltas.

## Non-goals

- No comparison to collaborator results without their configuration and raw
  measurements.
- No deployment recommendation beyond the documented interpretation.
- No modification to the primary speculative-decoding report.

## Validation

Review the note for numerical consistency against the committed Wave 1/Wave 2
tables, explicit statement of cross-window comparability limits, valid local
links, and absence of unsupported causal claims.
