# MiniMax-M3 Sequential Trace Discriminator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trace-only two-root diagnostic that identifies where matched MiniMax decoder targets disappear before sequential partitioning.

**Architecture:** Extend the production tracer with an optional JSON-safe evidence sink, then invoke it through a MiniMax CLI using the production loader, patch, dataset, and collator. Persist complete evidence per root even when tracing raises.

**Tech Stack:** Python, PyTorch FX, llm-compressor sequential pipeline, pytest, JSON.

## Global Constraints

- Do not enter oneshot, calibration lifecycle, AWQ, quantization, or evaluation.
- Do not copy or reimplement the production tracing algorithm.
- Existing `trace_subgraphs` callers must remain behaviorally unchanged.
- Attempt both roots and persist partial evidence on failure.

---

### Task 1: Opt-in tracer evidence

**Files:**
- Modify: `src/llmcompressor/pipelines/sequential/helpers.py`
- Modify: `tests/llmcompressor/pipelines/test_sequential.py`

- [ ] Write failing tests for matched targets, target nodes, operation counts, partitions, subgraphs, graph code, and unchanged no-sink behavior.
- [ ] Run the focused tests and confirm the missing diagnostics argument is the failure.
- [ ] Populate the sink at target matching, raw graph creation, and partition boundaries.
- [ ] Run the focused and existing sequential tests.

### Task 2: Two-root MiniMax command

**Files:**
- Create: `pipeline/m3_trace_diagnostic.py`
- Create: `pipeline/tests/test_m3_trace_diagnostic.py`

- [ ] Write failing tests for signature-based sample filtering, root classification, atomic artifact output, trace exception persistence, and aggregate status.
- [ ] Run tests and confirm failure because the command is absent.
- [ ] Implement production loading, patching, exact collator construction, full/subtree tracing, import provenance, and artifacts.
- [ ] Run focused tests and compile/lint checks.

### Task 3: Executor handoff

**Files:**
- Modify: `M3_AWQ_REPRESENTATIVE_RERUN_REPORT.md`
- Modify: `MINIMAX_M3_HANDOFF.md`

- [ ] Document a detached, exclusive-node, trace-only executor command and smoke-first rule.
- [ ] Require complete aggregate/root reports, graphs, nodes, logs, return codes, and environment provenance in the returned commit.
- [ ] Run focused verification, inspect the diff, commit, and push.
