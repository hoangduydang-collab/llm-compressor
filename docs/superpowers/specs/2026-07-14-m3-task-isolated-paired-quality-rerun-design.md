# MiniMax-M3 Paired Reasoning Quality Rerun Design

**Date:** 2026-07-14

**Revision:** r4.7, approved 2026-07-15

**Scope:** MiniMax-M3 paired GPTQ-versus-AWQ reasoning evaluation plus a
comparable BF16 companion baseline

**Status:** r4.7 evaluation, BF16 companion, and empty-output diagnostic approved

**Workflow state:** `READY_FOR_EXECUTOR`

This r4 design supersedes the earlier task-isolation, greedy GPQA, `sbatch`,
distributional-probe, and checkpoint-reuse designs in this file. Historical
artifacts and reports remain immutable, but none of their reasoning scores are
inputs to the r4 verdict.

## Decision and motivation

Determine whether the repaired in-house GPTQ checkpoint preserves reasoning
quality relative to the cyankiwi AWQ checkpoint. The earlier quick run cannot
answer that question reliably: its GPQA score used normalized continuation
likelihood, while current reasoning-model evaluations ask the model to generate
an answer after reasoning. A likelihood score is useful for conventional
quantization regression, but it is not comparable to the generated-answer
protocol behind modern MiniMax-M3 reasoning claims.

The rerun therefore keeps the paired 100-question budget but changes the four
reasoning tasks to task-native generated-answer protocols. It reuses the
repository's existing vLLM serving, manifests, checkpoint layout, health
records, paired statistics, and `srun` orchestration wherever their contracts
still apply. It does not create a second general evaluation framework.

## Goals

1. Evaluate GPQA Diamond, MMLU-Pro, GSM8K, and AIME 2025 with generated answers
   and task-appropriate extraction.
2. Use 100 unique paired questions per task and three paired sampling seeds per
   question. AIME 2025 is the only exception because its complete dataset has
   30 questions.
3. Make prompts, choice order, sampling, extraction, and scoring identical
   between AWQ and GPTQ.
4. Keep one model loaded while its node runs several tasks, avoiding repeated
   MiniMax-M3 startup cost.
5. Produce raw, auditable responses and failure diagnostics in the established
   EvalSuite artifact shape.
6. Give the executor copy-ready `srun` commands and no experiment-design work.

## Non-goals

- Reproducing a public full-dataset leaderboard number exactly.
- Claiming recovery relative to the official FP8 MiniMax-M3 checkpoint. The
  initial r4 verdict remains GPTQ versus AWQ; the separately returned BF16
  companion may support a later three-model comparison only after its complete
  evidence passes planner review.
- Rerunning IFEval. Its completed strict instruction-following result remains a
  separate, non-reasoning observation and is excluded from the new reasoning
  macro and gates.
- Rerunning either failed distributional probe.
- Reporting majority vote, pass@3, or best-of-three accuracy.
- Changing the model checkpoints, serving topology, tokenizer, chat template,
  or maximum output length.
- Adding automatic retries or letting the executor change prompts, seeds,
  sample IDs, thresholds, or task grouping at runtime.

## Chosen approach

Use the stock task-native lm-eval generated-answer tasks through the existing
reasoning runner interface. The runner uses the same vLLM backend as EvalSuite,
records responses in the existing per-task sample/checkpoint structure, and
feeds normalized binary outcomes into the existing paired comparison code.

This is preferable to changing only GPQA because all four capability tasks are
reasoning tasks, and preferable to copying an external harness wholesale
because serving, artifact validation, checkpointing, pairing, and reporting are
already implemented locally. Task prompts and extractors should be adapted from
the established upstream implementations named below rather than reauthored
from scratch.

## Task contracts

| Task | Questions | Attempts per question | Prompt and scoring contract |
| --- | ---: | ---: | --- |
| GPQA Diamond | 100 seeded from 198 | 3 | Pinned lm-eval `gpqa_diamond_cot_zeroshot`: zero-shot `generate_until`, four displayed A-D choices, and `exact_match,flexible-extract` scoring. |
| MMLU-Pro | 100 seeded and subject-stratified | 3 | Official five-shot chain-of-thought prompt and generated answer. Preserve ten A-J choices and use the official answer extractor. |
| GSM8K | 100 seeded | 3 | Existing official few-shot generated-solution task semantics with numeric final-answer extraction. |
| AIME 2025 | all 30 | 3 | Zero-shot generated reasoning with final boxed-integer extraction and exact integer scoring. |

The implementation must pin lm-eval 0.4.12 and record each resolved task name,
task version, output type, prompt configuration, filter/metric, and few-shot
count. Focused fixtures must cover representative valid answers, formatting
variants, and unparseable outputs. Local modifications are limited to adapting
lm-eval outputs to repository interfaces; no GPQA prompt or extractor is
reimplemented locally.

## BF16 companion baseline

The BF16 baseline is a separate run which reuses the r4 reasoning harness. It
must not modify, share a writable run root with, or relaunch the active GPTQ and
AWQ jobs. A committed BF16-only matrix supplies the executor with one fixed
model definition:

- checkpoint: `/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3`;
- two exclusive 8xH100 nodes per arm;
- tensor parallel size 8 and pipeline parallel size 2;
- Ray distributed executor;
- the existing `eval_minimax_m3_reasoning_r4.yaml` configuration;
- the same task aliases, two task shards, sampling seed, production limits,
  gates, and disabled distributional probe as the GPTQ/AWQ r4 matrix.

The smoke profile launches one two-node BF16 arm and evaluates two examples
each from GPQA Diamond, GSM8K, and AIME plus one example from every resolved
MMLU-Pro subject leaf, all under the three configured seeds. Its purpose is
only to validate the two-node Ray topology, BF16 model loading, vLLM
generation, per-filter checkpointing, three-seed execution, generation health,
and artifact completion. Smoke results are not quality evidence. A false smoke
gate stops the packet without a production launch or automatic retry.

Because this smoke is a setup check rather than a quality sample, it may carry
at most one isolated empty generation per model as an explicit warning. It
still requires every task/seed checkpoint, valid artifacts, the expected
distributed world size, and zero periodic loops. Smoke tolerance must never be
used to suppress or reinterpret full-run health data.

After a passing smoke gate, production launches two BF16 arms concurrently:

1. `gpqa`: GPQA Diamond, 100 questions and all three seeds;
2. `reasoning_suite`: MMLU-Pro 100, then GSM8K 100, then all 30 AIME 2025
   questions, with one model load retained across the three tasks.

Each production arm uses two nodes and has a 24-hour limit, so the maximum
concurrent allocation is four 8xH100 nodes. The launcher remains top-level
`srun` only. Task/seed checkpoints are atomic and resumable only within the
same BF16 run and unchanged contract; a retry still requires planner approval.

Comparability is fail-closed. Before BF16 production, the executor records the
active GPTQ/AWQ run root and proves that both runs have identical lm-eval
version, harness-contract SHA-256, tokenizer SHA-256, chat-template SHA-256,
production sample-manifest SHA-256, resolved task aliases, question counts,
few-shot settings, filter/metric keys, generation seeds, and generation
parameters. The only permitted differences are model identity, quantization
kind, and serving topology. A mismatch stops the run before production.

The BF16 evidence is returned independently. The planner may later compare its
sample/attempt UIDs against GPTQ and AWQ, but the executor does not merge roots,
interpret quality, or publish BF16 recovery claims.

### Sample identity and choice permutation

The main sample manifest is created once and shared by GPTQ and AWQ. The BF16
companion creates the same deterministic manifest in its independent run root
and must match the main manifest's SHA-256 before production. GPQA, GSM8K, and
MMLU-Pro use deterministic seed-42 selection; MMLU-Pro allocation is
proportional across resolved subject leaves with deterministic remainder
assignment. AIME uses all 30 questions.

Every question has a stable `sample_uid`. Every generated attempt is keyed by
`(sample_uid, generation_seed)`. GPQA uses the pinned lm-eval task's choice
preprocessing under fixed harness seed 42. Preflight must resolve the task twice
and prove that representative prompts and displayed-choice mappings are stable.
Runtime rows record the processed document and displayed choices, and merging
requires them to be identical across models and all three generation seeds.

## Shared generation contract

- Apply the official MiniMax-M3 tokenizer and chat template.
- Explicitly enable thinking; do not inherit an unset/default thinking mode.
- Set `temperature=1.0`, `top_p=0.95`, and `do_sample=true`.
- Use paired generation seeds `42`, `1234`, and `4158`.
- Allow at most 16,384 generated tokens per attempt.
- Generate exactly one response for each question and seed.
- Retain the complete response returned by lm-eval and never overwrite it with
  an extracted answer. Raw pre-postprocessing vLLM evidence is captured by the
  targeted replay because lm-eval's public result does not expose it.

The preflight must render a representative prompt for each task and verify the
resolved tokenizer, chat-template hash, thinking parameters, sampling values,
task formatter/extractor revision, sample-manifest hash, and model paths before
GPU launch. A mismatch is a stop-and-return condition.

## Metrics and statistical contract

The primary task metric is **mean pass@1**: the arithmetic mean of the binary
correctness of all individual attempts. Thus a 100-question task contributes
300 paired attempts per model, while AIME contributes 90. The three generations
are repeated measurements, not candidates for majority vote or pass@3.

For each task, report:

- each seed's pass@1 and the aggregate mean pass@1;
- GPTQ minus AWQ paired delta;
- paired bootstrap 95% confidence interval with 10,000 iterations;
- question-level GPTQ win, tie, and loss counts;
- parse failure, truncation, empty-response, and degeneration rates;
- output-token and reasoning-token distributions.

The bootstrap resampling unit is the question UID, not an individual attempt.
All three paired seed outcomes for a selected question stay together during a
bootstrap draw. Question-level win/tie/loss compares the mean correctness over
the three seeds. This avoids treating repeated generations as 300 independent
questions.

Existing score thresholds apply to aggregate task pass@1 and the macro over
the four reasoning tasks. Scientific validity and model health are separate:

- a missing expected UID/seed row, nonzero arm return, corrupt artifact, or
  contract mismatch is a hard validity failure;
- an empty or otherwise unusable response from a successfully completed model
  request is a complete attempt, receives correctness zero, and remains in the
  paired score;
- empty, parse-failure, truncation, and loop counts/rates are health advisories
  by model and task, not automatic invalidation of the completed study;
- the planner interprets systematic or asymmetric health failures alongside
  paired score deltas and confidence intervals.

The automated `quality_ok` verdict therefore uses score-recovery checks but not
a zero-count degeneration requirement. The machine-readable report preserves a
separate health advisory and never converts an empty answer into a retry or a
non-attempt. Because the experiment uses 100-question subsets, the confidence
intervals and raw deltas take priority over a binary gate near the threshold.
`gates.json` keeps `infrastructure_ok` as the scientific-validity signal,
computes `quality_ok` from score checks, and adds a non-gating
`health_advisory` with `has_findings` plus per-model/task counts and rates.

## Empty-output root-cause replay

The r4.5 smoke produced one processed empty GPTQ response for MMLU-Pro
economics doc 45 at seed 1234. The saved row proves that the request completed,
but lm-eval 0.4.12 retained only the post-processed string. It did not preserve
vLLM's raw text, output token IDs, finish reason, or stop reason, so the current
evidence cannot distinguish immediate EOS, task-stop behavior, thinking-marker
stripping, or a backend empty return.

Run one diagnostic replay on the in-house GPTQ checkpoint without modifying or
replacing any benchmark row. It loads the model once and executes two controls
from the exact saved rendered prompt and generation seed:

1. the observed smoke request with `max_gen_toks=256`;
2. the same request with the production `max_gen_toks=16384`.

Both controls preserve temperature 1.0, top-p 0.95, sampling enabled, seed
1234, tokenizer, chat template, thinking marker, and task stop sequences. The
diagnostic reuses the installed lm-eval vLLM adapter's tokenization, truncation,
sampling-parameter normalization, runtime preparation, and model lifecycle. A
narrow wrapper records vLLM `CompletionOutput` fields before applying the
pinned upstream thinking/stop postprocessor. It does not reimplement serving
or task scoring.

The replay artifact records raw text, token IDs, token count, finish reason,
stop reason, whether `</mm:think>` appeared, text after thinking removal, text
after task-stop removal, effective generation arguments, prompt SHA-256,
checkpoint identity, and environment versions. It classifies but does not
repair the result:

- zero raw tokens with EOS/stop evidence supports immediate termination;
- non-empty raw text becoming empty after postprocessing identifies the exact
  stripping stage;
- a length finish at 256 followed by a non-empty 16,384-token control supports
  a smoke-cap interaction;
- missing/malformed vLLM output or an engine error supports an infrastructure
  fault.

`min_tokens=1` is not part of either approved replay control. It may be tested
only in a later planner-authorized packet if immediate EOS is first confirmed.
Replay outputs never enter pass@1, replace the original attempt, authorize an
automatic retry, or change the paired/BF16 harness contract.

The executor-facing interface is one fixed command:

```bash
python -m pipeline.m3_empty_output_replay \
  --config pipeline/configs/eval_minimax_m3_reasoning_r4.yaml \
  --model /mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay \
  --samples results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl \
  --attempt-uid 8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878 \
  --out results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/diagnostics/empty-output-replay.json
```

The module validates the task, subtask, doc ID, seed, prompt hash, and original
generation arguments before GPU initialization. The two token caps are fixed
by the diagnostic contract rather than accepted as executor-selected options.

## Execution architecture

Use four independent top-level `srun` arms launched concurrently from a
detached `tmux` controller outside any existing Slurm allocation:

| Node arm | Ordered work |
| --- | --- |
| `cyankiwi_awq/gpqa` | GPQA |
| `inhouse_gptq/gpqa` | GPQA |
| `cyankiwi_awq/reasoning_suite` | MMLU-Pro, then GSM8K, then AIME 2025 |
| `inhouse_gptq/reasoning_suite` | MMLU-Pro, then GSM8K, then AIME 2025 |

Each arm requests one exclusive 8xH100 node and has an independent 24-hour
ceiling. The two suite arms load their model once and checkpoint after every
task and generation seed. GPQA receives separate nodes because its repeated
long reasoning generations would otherwise delay every other task.

The combined packet may use at most eight nodes:

1. Wave A starts paired production (four nodes), the exact GPTQ replay (one
   node), and BF16 smoke (two nodes) concurrently: seven nodes total.
2. Wave B starts BF16 production only after both the replay and BF16 smoke have
   ended. If paired production is still active, the two four-node production
   runs overlap at exactly eight nodes.
3. No other packet-owned allocation starts while eight nodes are active.

The cluster does not provide `sbatch`. The execution packet and durable planner
guidance must use top-level `srun` only. No worker may start a nested allocation,
and failure of one arm must not cancel siblings.

## Data flow and artifacts

```text
task-native upstream definitions + committed r4 config
                         |
                         v
preflight contract + one paired question manifest
                         |
                         v
four srun arms -> vLLM -> scored response per UID/seed
                         |
                         v
task extractor -> correctness + parse/health metadata
                         |
                         v
per-seed checkpoints -> model merge -> paired statistics
                         |
                         v
reasoning report + protocol-compliant executor evidence packet

saved failing prompt -> one-node raw vLLM replay -> diagnostic sidecar only
```

Every benchmark attempt row includes model label, task, source doc ID, sample
UID, generation seed, rendered prompt/generation arguments, processed response,
extracted answer, reference answer, correctness, and available health metadata.
The pinned lm-eval public interface returns only processed text, so raw vLLM
text, token IDs, finish reason, and stop reason are required in the diagnostic
replay sidecar rather than claimed for every production row. The checkpoint
marker records the expected and observed UID/seed pairs. Duplicate identical
rows collapse; conflicting duplicates fail closed.

The executor returns summaries and small artifacts in git. Raw logs and large
response files may remain in cluster storage, but the handoff must provide exact
paths, byte sizes, SHA-256 hashes, and bounded excerpts for failures.

## Resume and failure handling

- Do not import old GPQA, MMLU-Pro, GSM8K, or AIME checkpoints because their
  generation contracts differ from r4.
- Resume is allowed only within the same r4 run when the task contract,
  manifest, model, and generation settings hashes match exactly.
- Checkpoint after each generation seed so a scheduler interruption loses at
  most the active seed for the active task.
- A missing expected UID/seed row, duplicate conflict, nonzero arm return code,
  false completion marker, contract mismatch, malformed artifact, or
  asymmetric task definition blocks scientific validity. A present row with an
  empty processed response is instead scored incorrect and reported in health.
- Sampling/runtime failures are evidence, not silent skips. No automatic retry
  is authorized; the planner decides whether a revised packet is warranted.
- Aggregate completed siblings even after partial failure, clearly labelling
  the result incomplete and scientifically non-decisive.

## Implementation boundaries

The implementation should be a natural extension of the current pipeline:

1. Add an r4 reasoning configuration or explicitly version the current M3
   quality config; never silently change the historical config's meaning.
2. Extend the existing lm-eval runner for repeated generated-answer seeds while
   reusing the current vLLM lifecycle, sample UID utilities, static checkpoint
   layout, generation-health summaries, and comparison/reporting modules.
3. Extend paired statistics to group repeated seed outcomes by question during
   bootstrap and win/tie/loss calculation.
4. Update the task-isolated matrix and `srun` launcher to emit exactly the four
   arms above with a 24-hour ceiling and no distributional probe.
5. Add a fail-closed preflight contract and a copy-ready planner-to-executor
   packet. The executor must not reconstruct task commands.
6. Record the cluster's `srun`-only rule in durable planner guidance if it is
   not already present.
7. Separate scientific-validity, score-quality, and health-advisory outcomes;
   do not make a completed empty response an infrastructure failure.
8. Add a narrow exact-attempt replay that reuses the installed lm-eval/vLLM
   implementation and writes raw completion metadata to a diagnostic sidecar.

## Validation plan

Automated CPU tests must establish that:

- the manifest selects identical UIDs for both models, exactly 100 per task and
  all 30 AIME questions;
- MMLU-Pro selection is deterministic and subject-stratified;
- each question expands to exactly the three pinned generation seeds;
- GPQA's pinned lm-eval prompts and choice permutations are stable, paired, and
  auditable;
- upstream-derived formatters and extractors pass pinned fixtures;
- invalid and ambiguous final answers become explicit parse failures;
- aggregate pass@1 averages attempts without voting;
- bootstrap resamples question groups while retaining all three paired seeds;
- question-level win/tie/loss uses the three-seed mean;
- checkpoint/resume rejects any contract hash mismatch or incomplete UID/seed
  grid and collapses only identical duplicates;
- the launch plan contains exactly four one-node, eight-GPU `srun` arms with
  the specified task order and 24-hour ceiling;
- generated commands contain no `sbatch` or nested `srun` allocation;
- IFEval and distributional probes are absent from the r4 reasoning verdict.
- the BF16-only matrix emits one two-node TP8xPP2/Ray smoke arm and two
  two-node TP8xPP2/Ray production arms without launching GPTQ or AWQ;
- BF16 production is refused when any harness or production-manifest hash
  differs from the active GPTQ/AWQ run.
- the production score gate remains valid with one complete empty response,
  scores that response as incorrect, and emits a separate health advisory;
- replay postprocessing helpers identify whether raw text becomes empty at the
  thinking-marker or task-stop stage without importing vLLM in CPU tests;
- the diagnostic command selects exactly the saved model/task/doc/seed attempt,
  runs only the 256- and 16,384-token controls, and refuses contract drift;
- dry-run scheduling contains seven nodes in Wave A and never exceeds eight
  nodes when BF16 and paired production overlap.

Shell scripts must pass syntax checks, focused Python tests must pass, and the
dry-run output must show all resolved task, model, sampling, and resource values
before the executor is authorized to launch GPU work.

## Acceptance criteria

- GPTQ and AWQ are evaluated on the same question IDs, prompts, displayed
  choices, and three generation seeds.
- GPQA, MMLU-Pro, and GSM8K each produce 300 complete attempts per model; AIME
  produces 90 per model.
- All four tasks use generated-answer reasoning protocols with explicit
  thinking, not continuation likelihood.
- Results include auditable processed responses, raw replay evidence,
  parse/health diagnostics, per-seed pass@1, aggregate pass@1, paired deltas,
  grouped-bootstrap intervals, and question-level win/tie/loss.
- Execution uses no more than eight concurrent 8xH100 nodes, uses `srun` only,
  and preserves completed task/seed evidence when another arm fails.
- A completed empty model response remains an incorrect paired attempt and a
  health advisory; it does not invalidate or discard the other results.
- The exact GPTQ replay returns raw and postprocessed evidence for both approved
  token caps without changing any evaluation artifact or scientific contract.
- The BF16 smoke passes on TP8xPP2/Ray before its quick evaluation starts, and
  its returned GPQA/MMLU-Pro/GSM8K/AIME attempt grid matches the GPTQ/AWQ
  manifest and three-seed contract exactly.
- The report labels the result as a paired 100-question quantization study and
  does not present it as an exact reproduction of a public MiniMax-M3 score or
  as BF16 quality recovery.
- The executor returns in `RETURNED_FOR_ANALYSIS`; only the planner interprets
  the evidence and authorizes follow-up work.
