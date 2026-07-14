# Paired GPTQ/AWQ Quick Eval Rerun Report

Date: 2026-07-14  
Run root: `results/m3-quality/20260714T100300Z-m3-paired-gptq-awq-quick-rerun/`

## Scope

This rerun used the quick matrix with identical seeded samples for both models:

- Models: cyankiwi AWQ and in-house GPTQ ABI overlay
- Four arms: reasoning and broad shards for each model
- Samples per model: 430
  - GPQA Diamond: 100
  - IFEval: 100
  - AIME 2025: 30
  - MMLU-Pro: 100
  - GSM8K: 100
- Runtime: TP8, one node per arm, multiprocessing backend
- Time limit: 3 hours per arm

The previously passing smoke gate was reused, as requested; no new smoke arm
was included in this direct quick-eval launch.

## Outcome

The four arms did not all finish before the 3-hour Slurm limit. This is an
infrastructure/time-budget result, not a quality verdict.

| Arm | Durable task results | Outcome |
|---|---|---|
| cyankiwi AWQ reasoning | GPQA 100/100 | Timed out during the remaining reasoning workload |
| in-house GPTQ reasoning | GPQA 100/100 | Timed out during the remaining reasoning workload |
| cyankiwi AWQ broad | MMLU-Pro 100 + GSM8K 100 | Task aggregates were written; the arm later hit the time limit while handling the broad probe/restart path |
| in-house GPTQ broad | None | Timed out during model startup/loading before task aggregates were written |

Recorded task scores:

| Model | GPQA | MMLU-Pro | GSM8K |
|---|---:|---:|---:|
| cyankiwi AWQ | 24/100 | 76/100 | 97/100 |
| in-house GPTQ | 28/100 | not completed | not completed |

IFEval and AIME aggregates were not produced for either reasoning arm. The
partial scores above must not be used as the paired production quality gate.

## Root cause and artifacts

The 100-sample cap was applied correctly; the problem was that the full
16k-token generation budget made the 3-hour ceiling insufficient, especially
for reasoning prompts. Slurm terminated the affected steps at the time limit.

Raw logs and manifests are preserved under the run root, including:

- `production_launch_plan.json`
- `preflight/production_sample_manifest.json`
- `logs/production-*.out`
- `logs/production-*.err`
- per-arm `aggregate.json`, `eval_meta.json`, and `arm_manifest.json` where
  available

This run does not establish a GPTQ-vs-AWQ quality decision. A follow-up should
either use a longer time limit or reduce the generation/evaluation workload
with an explicitly documented deviation.
