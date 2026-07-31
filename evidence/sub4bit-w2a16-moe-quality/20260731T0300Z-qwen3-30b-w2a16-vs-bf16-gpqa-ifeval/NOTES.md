# Qwen3-30B-A3B-Instruct-2507: in-house W2A16 vs BF16 — paired quality A/B

**Date:** 2026-07-31 (Slurm job 13473 on gpu-h104; two earlier attempts 13471/13472
failed pre-eval, see "Launch failures" below)
**Candidate:** `artifacts/Qwen3-30B-A3B-Instruct-2507-autoround-W2A16-g128-ddp8/`
(AutoRound DDP 8×H100, iters=200, nsamples=512, int2 g128 sym, routers+lm_head BF16)
**Baseline:** BF16 `Qwen/Qwen3-30B-A3B-Instruct-2507`
**Harness:** benchmarks repo `quality.run_ab`, lm-eval 0.4.10, greedy temp 0 seed 0,
`--limit 100`, max_gen_toks 4096, chat completions, num_concurrent 16.
Both arms served from `serve-sub4` (vLLM 0.26.0 + PR#48918 port + humming main):
BF16 TP2 (GPUs 0–1, port 8410), W2A16 TP1 (GPU 2, port 8411), same node.
Fail-closed harness check: PASS (`harness-check.json`; tokenizer semantic identity,
task pins, gen params, topology, offline dataset caches).

**Comparability:** paired-subset comparison only (first-100 deterministic subsets,
greedy, 4096-token cap). Valid for model-to-model decisions; **NOT** comparable to
public leaderboard numbers. GPQA absolute scores in both arms are depressed by
budget censoring (even BF16 exhausts the 4096 cap on 52% of GPQA CoT items).

## Result: large quality regression — W2A16 @ iters=200 NOT usable as-is

| metric | BF16 | W2A16 | delta |
|---|---|---|---|
| gpqa_diamond_cot_zeroshot (exact_match, 0-shot) | 0.54 | 0.24 | **−0.30** |
| ifeval (prompt_level_strict_acc, 0-shot) | 0.84 | 0.45 | **−0.39** |
| GPQA exhausted@4096 rate | 0.52 | 0.81 | +0.29 |
| IFEval exhausted@4096 rate | 0.00 | 0.32 | +0.32 |
| IFEval completion-token spend ratio | 1.0 | 3.44× | |
| top-1 next-token agreement (20 decode paths) | — | 0.775 | |
| truncated-KL@20 mean / p95 | — | 0.49 / 2.24 | |

Failure mode (see samples jsonl in benchmarks results tree): under greedy decoding
the W2A16 model falls into verbatim repetition loops (e.g. "But the molecule is not
symmetric.\n\n" repeated until the cap) — hence the exhausted-rate explosion and
3.4× IFEval token spend. Primary-signal reading per repo convention: exhausted_rate
first (heavy censoring ⇒ score deltas partly reflect never-emitting-an-answer),
then token_spend_ratio. Instruction following halves even where output terminates.

The delta verdict machine says CAVEAT only because its flip-rate gate uses the 10
trivially-easy distribution QA probes (both arms 10/10) and no perf dimension was
measured. The general-suite deltas are the decision signal: **NO-SHIP at iters=200**.

## Interpretation / next steps (not yet run)

- iters=200 was the smoke-tier tuning budget; AutoRound's reference W2A16 recipes
  use iters≈1000 + larger nsamples. Rerun quant at iters=1000 (est. ~5×
  tuning wall-clock ⇒ ~8h on 8×H100) before judging W2A16 viability.
- Repetition-loop failure is a decode-time symptom of weight degradation, not a
  serving bug: the same stack served the model coherently in the short smoke, and
  BF16 through the identical serve-sub4 stack scores normally (0.84 IFEval).
- Distribution drift (top-1 agreement 77.5%, KL p95 2.24) is far outside healthy
  4-bit territory (>93%, ≪0.1) — consistent with genuine 2-bit quality loss, not
  a harness artifact.

## Launch failures (fixed, kept for the record)

1. Job 13471: `run_ab` crashed pre-eval — benchmarks repo's `run_ab.py` enables the
   reliability suite from the profile but never passes its built-in probe datasets
   to `run_quality`; orchestrator's fail-closed validator rejected. Fix: profiles
   set `RELIABILITY_METRICS=""` (reliability was extra; task asked for GPQA+IFEval).
2. Job 13472: general suite failed with `FileNotFoundError: lm_eval` — launch.sh
   had only the serve venv on PATH; the lm_eval CLI lives in the benchmarks venv.
   Fix: prepend `$BENCH_VENV/bin` to PATH after servers spawn + fail-closed
   `command -v lm_eval` gate. (Distribution suite from 13472 already matched 13473.)

## Files

- `launch.sh` — full serve+check+eval driver (srun/tmux, fail-closed gates)
- `harness_check.py`, `harness-check.json` — pre-eval contract check (PASS)
- `results/` — delta, general (both arms), distribution, usage_delta, task_flips,
  HTML report. Raw lm-eval sample jsonls remain in
  `/mnt/nfs/hoangduy/projects/benchmarks/results/qwen3-30b-a3b-2507-{w2a16,bf16}/`
- `eval-run-output.txt`, `serve-bf16.txt`, `serve-w2a16.txt`, `run_ab-output.txt`
