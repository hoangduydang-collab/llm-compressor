# Goal 6 · Packed NVFP4 W4A8 on Hopper — Field Note

> **Planned · benchmark-gated** · markdown twin of
> [`goal-6-hopper-packed-nvfp4-w4a8.html`](goal-6-hopper-packed-nvfp4-w4a8.html) — keep in sync.
> [← Program overview](../automatic-quantization-pipeline-progress.md)

**Contents:** [Why](#why) · [Sub-tasks](#sub-tasks) ·
[Decision gates](#decision-gates) · [Evidence](#evidence)

Future work: run vendor-released NVFP4 (4-bit) checkpoints efficiently on
current-generation (Hopper) GPUs, which lack native 4-bit compute — keep the
weights packed at 4 bits and compute on the GPUs' 8-bit units. Design and a CPU
prototype are committed; **there are no hardware results yet**, and all further
investment is gated on a benchmark proof.

## Why

Vendors increasingly ship NVFP4 checkpoints for the newest (Blackwell) GPUs.
Expanding those weights to 8 bits on Hopper works but gives back much of the
memory and bandwidth win. The question: can Hopper keep the 4-bit packing and
still run fast?

## Sub-tasks

<!-- EXTENSION POINT: append sub-tasks here AND in the HTML twin; IDs from PROJECT_GOALS.md goal 6. -->

- [ ] 6a · Dense proof of concept vs the existing compatibility paths — the gate for all further work

## Decision gates

The proof of concept must pass three bars before any production work begins:
**correctness** (matches the reference within tolerance), **memory** (no
persistent 8-bit copy of the weights), and **performance** (a useful win in at
least one serving scenario without slowing the primary one). Fail any → stop,
keep the evidence.

## Evidence

- Design: [`2026-07-19-hopper-packed-nvfp4-w4a8-fallback-design.md`](../superpowers/specs/2026-07-19-hopper-packed-nvfp4-w4a8-fallback-design.md)
- Handoff: [`HOPPER_NVFP4_W4A8_HANDOFF.md`](../../HOPPER_NVFP4_W4A8_HANDOFF.md) · Code: `pipeline/hopper_nvfp4_w4a8/`
- Contract: [`PROJECT_GOALS.md`](../../PROJECT_GOALS.md)
