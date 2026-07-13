# MiniMax-M3 vLLM Quality Evaluation Design

**Date:** 2026-07-12

**Status:** Approved for implementation

## Objective

Build a reproducible, vLLM-first evaluation pipeline that compares MiniMax-M3
BF16 against three quantized checkpoints in less than five hours of wall-clock
time on the capable cluster. Model quality is the first deliverable. Serving
throughput, latency, TTFT, and memory benchmarking remain a separate follow-up
after the quality pipeline succeeds.

The first comparison matrix contains:

1. `MiniMaxAI/MiniMax-M3` BF16 at
   `/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3` as the canonical baseline.
2. The passing in-house GPTQ checkpoint at
   `/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123`.
3. `cyankiwi/MiniMax-M3-AWQ-INT4` from
   `/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4`.
4. `aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx` at
   `/mnt/nfs/hoangduy/hf_assets/aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx`
   as the initial additional community quantization.

All model paths are explicit defaults in a matrix manifest and can be
overridden at launch time. The in-house checkpoint default may be replaced by
the final repaired AWQ checkpoint when it becomes available without changing
the evaluation design.

## Why This Design

Recent frontier-model reports emphasize GPQA-Diamond, MMLU-Pro, current math,
coding, instruction following, and agentic benchmarks. MiniMax-M3 itself is
positioned primarily on coding, agentic, long-context, and multimodal quality.
However, reproducing vendor scores is not a suitable first quantization gate:
published evaluations often use very long completions, stochastic sampling,
multiple trials, private harnesses, or external tools. Those choices make a
four-checkpoint comparison slow and noisy.

The primary pipeline therefore measures *paired quantization fidelity*: every
checkpoint receives the same prompt rendering, exact examples, decoding
settings, and scoring implementation. It uses EleutherAI
`lm-evaluation-harness`, which is the established base of the Open LLM
Leaderboard, with its direct vLLM backend. The existing repository already
supports vLLM and SGLang model backends, per-task checkpointing, logged samples,
and paired post-hoc comparison. This design extends that code instead of
creating a second evaluator.

The pipeline does not claim to reproduce a vendor's headline score unless the
vendor's complete prompt, sampling, judging, and harness protocol is separately
implemented and named as such.

## Scope

### Included now

- Text-only MiniMax-M3 quality evaluation through the direct vLLM backend.
- A deterministic benchmark profile aimed at quantization comparison.
- Exact, shared sample manifests for every model.
- Modern downstream benchmarks plus a teacher-forced distributional probe.
- Quantization/checkpoint diagnostics.
- Paired statistical analysis and generation-health diagnostics.
- Parallel `srun` orchestration for a cluster with 8xH100 nodes.
- Resumable arms, incremental artifacts, validation, aggregation, and a single
  comparison report.
- Backend-neutral configuration boundaries so SGLang can be added later.

### Deferred

- SGLang execution validation. Existing backend support is preserved, but this
  milestone is accepted using vLLM.
- Serving throughput, latency, TTFT, ITL, memory, and concurrency sweeps.
- Full multimodal benchmarks such as MMMU-Pro and Video-MME.
- Full long-horizon agentic runs such as SWE-bench, Terminal-Bench, BrowseComp,
  and tau2. These are important production evaluations, but their harness and
  runtime variance make them a separate phase.
- Vendor-score reproduction profiles with stochastic multi-trial sampling.

## Evaluation Profiles

### Smoke profile

The smoke profile validates the complete path before expensive execution:

- Two examples from every configured harness task.
- Eight short teacher-forced probe sequences.
- One generation-health prompt designed to expose looping or length-cap
  behavior.
- All four model manifests are validated, but operators may select one model
  for the first load smoke.
- Expected wall time is dominated by model loading.

The smoke profile must prove that task names resolve, prompts render, samples
score, artifacts parse, and aggregation completes. It is not an accuracy gate.

### Production-quality profile

The first production-quality profile contains:

| Capability | Harness task | Sampling policy | Purpose |
|---|---|---:|---|
| Graduate science reasoning | GPQA-Diamond | Full public split | Current frontier reasoning and quantization sensitivity |
| Instruction following | IFEval | Full public split | Machine-verifiable instruction adherence |
| Competition math | AIME 2025 | Full public split | Long-form exact-answer reasoning |
| Broad difficult knowledge | MMLU-Pro | Fixed seeded stratified 2,000-example manifest | Representative broad-domain accuracy under the time SLA |
| Arithmetic/reasoning diagnostic | GSM8K | Full public split | Stable exact-answer regression diagnostic |

Exact harness task identifiers are resolved and frozen during cluster preflight
against the installed `lm-eval` revision. If a task has multiple upstream
aliases, the preflight writes the selected canonical identifier into the run
manifest. A missing task fails preflight; it is never silently skipped or
replaced.

All generation tasks use the MiniMax-M3 chat template and thinking mode. The
checkpoint-supported chat-template argument is resolved during preflight, and
the exact argument plus rendered-prompt hash are recorded. The primary paired
profile uses deterministic decoding and a fixed maximum generation length.
MiniMax's recommended stochastic settings (`temperature=1.0`, `top_p=0.95`) are
reserved for a future vendor-aligned profile because sampling noise would
obscure checkpoint-to-checkpoint quantization effects.

### Distributional fidelity probe

A calibration-disjoint, immutable prompt corpus is built once and identified by
content hash. It contains short, 8k-token, and 32k-token buckets. Each model is
run in teacher-forced mode with bounded prompt log-probabilities. The probe
records per-token information needed for:

- Mean negative log-likelihood.
- Perplexity and perplexity ratio to BF16.
- Bits-per-token increase.
- Mean, median, p95, and p99 observed-token log-probability drift.
- Top-1 token agreement.
- Top-5 and top-20 token-set overlap.
- BF16 top-token retention in the quantized top-k set.
- Drift by token-position and prompt-length bucket.

The report must not label a bounded top-k proxy as full-vocabulary KL
divergence. If a future backend exposes full logits safely, exact KL/Jensen-
Shannon metrics can be added under a distinct artifact schema version.

## Reproducibility Contract

Every production run writes a root `run_manifest.json` before GPU work begins.
It contains:

- Run ID, repository commit, dirty-worktree status, UTC timestamp, and launcher
  command.
- Model labels, resolved paths, checkpoint identity hashes, quantization
  metadata, and tokenizer/config/chat-template hashes.
- vLLM, lm-eval, Transformers, compressed-tensors, CUDA, driver, PyTorch, and
  Python versions.
- Backend configuration, tensor/expert parallel settings, maximum model length,
  KV-cache dtype, graph/eager mode, environment patches, and GPU type.
- Task identifiers, dataset revisions when available, exact sample-index
  manifest hash, few-shot settings, generation settings, seed, and metric keys.
- Expected arms and output paths.

The exact MMLU-Pro subset is selected once with a fixed seed and stratified by
subject. `lm-eval` receives explicit sample-index mappings rather than `limit`,
which otherwise evaluates a prefix. All checkpoints must use the same sample
manifest hash. Full-split tasks also write their resolved example identities so
pairing coverage can be audited.

Sample identity is `(task, subtask, document identity)`, not raw `doc_id`.
Group-task document IDs can repeat across subjects, so the current evaluator's
unnamespaced mapping must be corrected before MMLU-Pro comparisons are trusted.

## Execution Architecture

The capable cluster launches eight initial arms: two task shards for each of
four checkpoints. Each arm owns one 8xH100 node and loads one checkpoint with
TP=8. The two shards are balanced using smoke-derived timing estimates; the
static default separates generation-heavy tasks from multiple-choice and
distributional work. Up to seven remaining nodes are kept available for smoke,
retries, diagnostics, or finer sharding when timing evidence warrants it.

The launcher uses concurrent background `srun --exclusive` steps because
`sbatch` is unavailable in the target environment. It captures each step's
stdout, stderr, exit code, host, and start/end timestamps. One failed arm does
not terminate completed siblings. A rerun with the same run ID schedules only
missing or failed work.

Each arm writes per-task results immediately. Aggregation runs only after every
required artifact passes schema and provenance validation. The merger rejects:

- Mixed repository or harness revisions.
- Different model/tokenizer/chat-template identities under one model label.
- Different sample manifests or decoding settings.
- Duplicate samples with conflicting results.
- Missing required tasks, incomplete sample coverage, or unscored failures.

The wall-clock target is less than five hours from the start of GPU arms to a
validated report. Model download/staging is a preflight responsibility and is
not hidden inside the timed evaluation. Arm timing is recorded so the next
matrix can be rebalanced from evidence.

## Metrics and Analysis

### Downstream paired metrics

For every task and BF16-versus-quantized pair, report:

- Baseline and quantized score.
- Absolute score delta and BF16 score-recovery ratio.
- Paired sample coverage and missing-result counts.
- Flip ratio.
- Harmful flips: BF16 correct and quantized incorrect.
- Beneficial flips: BF16 incorrect and quantized correct.
- Conditional regression rate among BF16-correct samples.
- Conditional recovery rate among BF16-incorrect samples.
- Net harmful flips.
- Agreement and Cohen's kappa.
- Exact two-sided McNemar test for small discordant counts and a clearly labeled
  continuity-corrected asymptotic result for larger counts.
- Deterministic paired-bootstrap 95% confidence intervals for score delta, flip
  rate, and regression rate.
- Subject/category breakdown and worst subgroup.

Aggregate reporting includes task-macro and sample-micro views, but the macro
view is primary because a sampled broad benchmark must not overwhelm small
reasoning tasks. Also report the worst task delta and worst subgroup delta. No
arbitrary composite quality score is introduced.

### Perplexity and score-drift metrics

Teacher-forced results report BF16 and quantized NLL, perplexity, perplexity
ratio, relative perplexity increase, bits per token, and paired per-token drift
statistics. Continuous harness metrics retain their original values and use
paired deltas rather than being coerced into binary correctness.

### Generation-health metrics

For every generative task, report:

- Empty/missing response rate.
- Answer-extraction failure rate.
- Length-cap hit rate.
- Repeated n-gram and periodic-loop rate.
- Non-finite metric or log-probability count.
- Output-token and reasoning-token length distributions.
- Exact-output agreement with BF16 under deterministic decoding.
- Regression examples ranked by lost correctness and, when available, lost
  answer margin or log-probability.

Loop detection must cover the known MiniMax-M3 failure shape: a short periodic
token sequence repeated until the generation cap.

### Checkpoint and quantization diagnostics

Before scoring, each checkpoint produces a diagnostic artifact with:

- On-disk checkpoint size and compression ratio to BF16.
- Effective stored bits per original parameter.
- Quantized-parameter coverage and BF16 fallback percentage.
- Coverage broken down by dense attention, sparse attention/indexer, routed
  experts, shared experts, routers, norms, vision tower, and LM head.
- Quantization method, weight/activation types, group size, zero-point policy,
  ignore list, and provenance.
- Scale distribution summaries and zero/non-finite scale counts.
- Packed-code saturation estimates from a deterministic tensor sample when the
  format decoder is available.
- GPTQ/AWQ calibration reconstruction summaries when present in checkpoint run
  artifacts.

Unavailable third-party calibration metrics are recorded as `unavailable` with
a reason. They are never inferred or treated as zero.

## Gates

The report exposes individual gates, not a hidden composite. Initial defaults
are configurable and are recorded in the manifest:

- Every required task and expected sample is present and paired.
- No empty, non-finite, or periodic-loop generations.
- No task loses more than 2.0 absolute percentage points versus BF16.
- Task-macro score recovery is at least 98% of BF16.
- Overall conditional regression rate is no more than 5%.
- Distributional perplexity increase is no more than 10%.

Statistical evidence accompanies gates, but a wide confidence interval does not
turn a large observed degradation into a pass. The report includes both the
observed threshold decision and confidence interval.

These are evaluation defaults, not claims about universally acceptable
production quality. The first BF16/community matrix will calibrate whether the
thresholds need a documented revision.

## Artifacts

Each run produces:

```text
results/minimax-m3-quality/<run-id>/
  run_manifest.json
  sample_manifest.json
  preflight.json
  models/<model-label>/
    checkpoint_diagnostics.json
    shards/<shard-label>/
      arm_manifest.json
      aggregate.json
      samples/<task>.jsonl
      distributional_probe.jsonl
      generation_health.json
      stdout.log
      stderr.log
      return_code.txt
  merged/<model-label>/
    aggregate.json
    samples/<task>.jsonl
    distributional_probe.jsonl
    generation_health.json
  comparisons/<quantized-label>/
    compare.json
    report.md
  matrix.json
  report.md
```

`matrix.json` is the machine-readable source of truth. The root `report.md`
contains the four-model benchmark table, BF16-relative fidelity table,
distributional results, generation-health findings, checkpoint diagnostics,
gate results, runtime, failures, and links to pairwise reports.

## Error Handling and Resume Behavior

- Preflight errors stop before multi-node launch.
- Arm failures are isolated and recorded; completed arms remain reusable.
- Per-task writes are atomic, and reruns skip only validated completed tasks.
- The launcher returns nonzero if any required arm, aggregation check, or gate
  fails.
- Aggregation distinguishes evaluation failure from model-quality failure.
- No missing model/task is omitted from the final table.
- Logs and concise failure excerpts are returned through Git for analysis on the
  resource-limited cluster.

## Testing Strategy

CPU/unit tests cover:

- Configuration and matrix validation.
- Deterministic stratified sample-manifest generation.
- Namespaced group-task sample identities.
- Exact sample matching and coverage rejection.
- Flip, conditional regression/recovery, kappa, exact/asymptotic McNemar, and
  bootstrap interval calculations.
- Continuous metric handling.
- Loop, cap-hit, empty, invalid-answer, and non-finite detection.
- Distributional top-k fidelity calculations with missing-token cases.
- Checkpoint diagnostic schema and unavailable fields.
- Shard merge conflicts and provenance mismatch rejection.
- `srun` launcher dry-run topology, resume selection, and return-code handling.
- Markdown report rendering.

Cluster acceptance proceeds in this order:

1. Task-resolution and checkpoint preflight.
2. One-model smoke profile.
3. Four-model smoke and self-comparison; BF16 versus itself must have zero flips,
   zero score delta, kappa 1 where defined, and identical sample hashes.
4. Production matrix under the five-hour SLA.
5. Artifact validation and Git handoff for analysis.

## SGLang Follow-up Boundary

Backend-specific construction remains in `pipeline/lmeval_runner.py`. Sample
manifests, normalized artifacts, metrics, merging, gates, and reports do not
depend on vLLM. The SGLang follow-up adds and validates a launch profile using
the existing `eval.backend: sglang` support, then runs the same sample manifest
and compares backend-to-backend output separately from checkpoint-to-checkpoint
quality.

## Authoritative References

- MiniMax-M3 model card and serving recommendations:
  <https://huggingface.co/MiniMaxAI/MiniMax-M3>
- MiniMax-M3 official repository and reasoning modes:
  <https://github.com/MiniMax-AI/MiniMax-M3>
- EleutherAI lm-evaluation-harness:
  <https://github.com/EleutherAI/lm-evaluation-harness>
- lm-eval exact sample-index interface:
  <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md>
- GPTQ evaluation precedent:
  <https://arxiv.org/abs/2210.17323>
- Kimi K2.5 benchmark and repeated-sampling methodology:
  <https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/README.md>
- GLM-5.2 benchmark methodology and contemporary benchmark selection:
  <https://huggingface.co/zai-org/GLM-5.2/blob/main/README.md>
- vLLM prompt-log-probability interface:
  <https://docs.vllm.ai/en/stable/api/vllm/sampling_params/>

