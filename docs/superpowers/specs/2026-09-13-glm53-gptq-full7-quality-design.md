# GLM-5.3 native GPTQ W4AFP8 initial full7 quality evaluation

**Date:** 2026-09-13

**Status:** Approved for implementation

## Objective

Run one full-population seven-task quality arm for the completed native
GLM-5.3 GPTQ W4AFP8 checkpoint, then compare it offline with the corrected
historical in-house AWQ and PhalaCloud full7 artifacts.

This is an internal paired-regression instrument. It is not a leaderboard
submission and does not rerun either historical baseline.

## Decision summary

- Run the full seven-task suite immediately; do not add a quick-suite gate.
- Run only the new GPTQ candidate on GPUs.
- Compare with both corrected historical baselines.
- Treat a regression greater than 2 percentage points as a diagnostic flag,
  not an early-stop condition.
- Capture exact whole-suite token counters and per-sample outputs. Do not claim
  exact per-task or per-sample token usage.
- Give GPTQ a dedicated arm and model identity; never reuse the AWQ `ours`
  identity.

## Prior-art baseline

The design reproduces the first GLM-5.3 paired full7 evaluation rather than
inventing a new instrument:

- Internal runbook:
  `docs/glm53-w4afp8-rancher-evaluation.md`
- Corrected result report:
  `docs/status/2026-09-04-glm53-quant-and-quality.md`
- Rancher arm runner:
  `pipeline/k8s/glm53_quality_arm.sh`
- CPU staging and parity gates:
  `pipeline/k8s/stage-glm53-quality-eval.sh`
- Upstream harness:
  [lm-evaluation-harness v0.4.10](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.10)
- Serving engine:
  [SGLang v0.5.17](https://github.com/sgl-project/sglang/releases/tag/v0.5.17)

The historical comparison is valid only if the candidate uses the same task
populations, prompts, generation compatibility settings, loglikelihood path,
scorers, and serving topology.

## Fixed artifact identities

### Candidate

Completed native checkpoint:

```text
/mnt/cephfs/hoangduy/results/glm53-ep-gptq-w4afp8/full-ep8/20260912t183612z/output/304b8051cfb2b260b61ce0cbe330e02a98e73639-gptq-W4AFP8/20260912-184239/checkpoint
```

Required candidate identity:

```text
arm: gptq
served model: glm-5.3-w4afp8-gptq
quant source: offline-w4afp8-inhouse-ep-gptq
quant recipe: inhouse-ep-gptq-w4afp8
checkpoint format: native sglang-w4afp8
```

The native checkpoint must pass its existing manifest/config verification
before evaluation. No AWQ-style post-quantization conversion or MTP graft is
allowed in the evaluation path.

### Historical AWQ baseline

```text
run: full7-20260901t064327z
checkpoint: /mnt/cephfs/hoangduy/results/glm53-w4afp8-mtp/checkpoint
result: /mnt/cephfs/hoangduy/results/glm53-quality-paired/full7-20260901t064327z/results/glm-5.3-w4afp8-ours/sglang/quality/general.glm53-full7-20260901t064327z.json
```

This is the corrected AWQ run. The earlier AWQ result under
`full7-20260831t135418z` is invalid because a chat template was applied to
loglikelihood requests.

### Historical PhalaCloud baseline

```text
run: full7-20260831t135418z
checkpoint: /mnt/cephfs/.hf-cache/models--PhalaCloud--GLM-5.3-W4AFP8/snapshots/7e77d7b5592d748778459a0dac802e7fd407e593
result: /mnt/cephfs/hoangduy/results/glm53-quality-paired/full7-20260831t135418z/results/glm-5.3-w4afp8-phala/sglang/quality/general.glm53-full7-20260831t135418z.json
```

## Evaluation contract

### Tasks

Run full populations with no `--limit`:

1. `gsm8k`
2. `ifeval`
3. `gpqa_diamond_cot_zeroshot`
4. `mmlu`
5. `arc_challenge`
6. `hellaswag`
7. `truthfulqa_mc2`

Use lm-eval seed 0. Preserve task-native few-shot settings from the historical
full7 profile: GSM8K 5-shot, MMLU 5-shot, GPQA Diamond CoT 0-shot, and the
existing settings for the other tasks.

Headline metrics:

- GSM8K: exact match
- IFEval: strict prompt accuracy
- GPQA Diamond CoT: exact match
- MMLU: accuracy
- ARC Challenge: normalized accuracy
- HellaSwag: normalized accuracy
- TruthfulQA: MC2

### Historical generation compatibility

The historical full7 runs explicitly raised generative tasks to a 32,768-token
budget with thinking enabled. Current benchmark code correctly notes that an
lm-eval `--gen_kwargs max_gen_toks=32768` override replaces each task's native
budget rather than merely adding a ceiling.

For this run, that override is intentional and must be recorded as
`historical-full7-compatibility`:

```text
reasoning mode: reasoning / enable_thinking=true
temperature: 0 / greedy
max_gen_toks: 32768
num_concurrent: 16
```

This deliberate non-formal override is required for direct comparison with the
historical AWQ and PhalaCloud numbers. It is also why these scores must not be
presented as native-task or leaderboard absolutes.

### Loglikelihood path

MMLU, ARC Challenge, HellaSwag, and TruthfulQA MC2 use
`/v1/completions` with echo and logprobs.

The GLM chat template must not be applied to these prompts. The candidate
harness manifest must say that explicitly. A capability probe must confirm
response shape before the suite begins.

### Serving topology

Use the proven full7 deployment contract:

```text
image: lmsysorg/sglang:v0.5.17
topology: one pod, one container, 8x H100-80GB, TP=8
quantization: w4afp8
KV cache: fp8_e4m3
context length: 65536
memory fraction static: 0.75
chunked prefill: 2048
reasoning parser: glm45
tool-call parser: glm47
shared-expert fusion: disabled
speculative decoding: disabled
metrics exporter: enabled
```

The arm must override the runner's newer 164,800-token AA default. AA GPQA is
disabled and remains a separate evaluation.

## Components and changes

### Dedicated GPTQ profile

Add a benchmark profile derived from the corrected AWQ profile, changing only
the checkpoint/model identity and quantization provenance needed for GPTQ.
Keep task, scorer, tokenizer, endpoint, and serving declarations aligned with
the historical full7 contract.

The profile must enable per-sample output logging for every invocation.

### Arm renderer and pod template

Extend the existing renderer to accept `gptq` without changing `ours` or
`phala`. The generated pod must use:

- the candidate checkpoint above;
- a unique `gptq-full7-<UTC timestamp>` run tag;
- the dedicated GPTQ profile;
- a result root below
  `/mnt/cephfs/hoangduy/results/glm53-quality-paired/`;
- the current pinned llm-compressor commit.

### CPU-only staging

Reuse all existing staging gates and add candidate checks without weakening the
AWQ/Phala checks:

- pinned lm-eval, datasets, and NLTK closure;
- private offline HF dataset cache;
- IFEval official-population digest;
- profile render/dry-run;
- candidate tokenizer and chat-template parity with both baselines;
- candidate native checkpoint manifest/config validation;
- no mutable Hugging Face branch references.

All of these checks run before a GPU is claimed.

### Token-usage summary

Retain the raw SGLang metrics snapshots immediately before and after the general
suite. Build a normalized `token-usage.json` by subtracting matching counter
series. At minimum, retain available deltas for:

- prompt/input tokens;
- generated/output tokens;
- cached tokens;
- completed requests;
- aborted requests.

The summary must include source metric names, labels, before/after values,
deltas, scrape timestamps, and a completeness status. Missing counters or a
counter reset make token accounting `untrusted`; they do not silently become
zero.

Per-sample lm-eval outputs are retained separately through `--log_samples`.
They support error and flip analysis but are not evidence for exact API token
usage. No exact per-task or per-sample token totals will be reported.

### Three-way comparison

After the candidate finishes, read the candidate and two fixed historical
result JSONs and emit a three-way comparison artifact.

For each headline metric:

```text
delta_vs_awq = GPTQ - corrected historical AWQ
delta_vs_phala = GPTQ - historical PhalaCloud
flag if either delta < -0.02
```

Also record task sample counts and source artifact paths. Missing tasks, metric
aliases, or population mismatches make that comparison invalid rather than
being coerced into a score.

The suite always attempts all seven tasks. A quality regression changes the
reported conclusion but does not terminate the job.

## Data flow

```text
native GPTQ checkpoint
  -> CPU identity/parity/native-format gates
  -> dedicated SGLang GPTQ pod
  -> health, throughput, reasoning, and loglikelihood capability gates
  -> full7 lm-eval suite
  -> candidate results + per-sample outputs + raw metrics snapshots
  -> token-usage normalization
  -> offline join with fixed AWQ and PhalaCloud result JSONs
  -> three-way quality report
```

## Failure handling

- Refuse GPU launch if staging, identity, format, tokenizer, template, dataset,
  or scorer provenance fails.
- Refuse the suite if server health, aggregate throughput, reasoning parsing, or
  echo/logprobs capability fails.
- Preserve logs and partial artifacts on any suite failure, but do not publish
  a quality conclusion from incomplete results.
- Mark token usage untrusted if scrape pairing is incomplete or counters reset.
- Finish all tasks before evaluating the 2-point quality flags.
- Never fall back to the invalid historical AWQ artifact.

## Validation

Before launch:

1. Unit-test renderer acceptance and dedicated GPTQ identity.
2. Dry-run the GPTQ profile and assert the seven expected tasks.
3. Test that historical compatibility passes `max_gen_toks=32768` only to
   generative invocations.
4. Test that loglikelihood invocations never apply a chat template.
5. Test token-counter subtraction, missing counters, and reset detection.
6. Test three-way metric mapping and population mismatch refusal.
7. Run the CPU staging gates against the actual candidate checkpoint.

Inside the GPU pod:

1. Verify SGLang readiness.
2. Enforce the existing aggregate throughput floor of 250 tokens/s.
3. Probe reasoning-content separation.
4. Probe `/v1/completions` echo/logprobs shape.
5. Run the full suite only after every gate passes.

## Resource and workspace safety

- Do not modify collaborator checkpoints, caches, jobs, or workspaces.
- Write only to a new owned result root and new owned pod/job.
- Mount model artifacts read-only where practical.
- Immediately before launch, run the authoritative six-namespace Rancher GPU
  availability check.
- Show the selected free node and ask for approval before claiming 8 GPUs.
- Do not release, cancel, or signal any collaborator workload.

## Success criteria

Execution succeeds only when:

- all preflight and capability gates pass;
- all seven candidate task results are present at full expected populations;
- per-sample outputs are retained;
- raw before/after metrics and normalized token-usage status are retained;
- the three-way report references the exact candidate and historical artifacts.

The quality conclusion is:

- `no_large_regression` when neither historical comparison has a regression
  greater than 2 percentage points on any headline metric;
- `diagnostic_regression` otherwise, with every affected task and delta listed.

This quality label is a result, not a job-control gate.
