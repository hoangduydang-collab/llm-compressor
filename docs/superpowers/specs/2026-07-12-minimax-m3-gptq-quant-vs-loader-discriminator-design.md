# MiniMax-M3 GPTQ Quantization-vs-Loader Discriminator Design

## Goal and scope

Determine whether the in-house GPTQ quality failure originates in the
quantized checkpoint itself or in its export/loading/serving path, while
unblocking the existing quality evaluation. Model quality and quantization
fidelity remain the primary metrics. Serving performance and fresh
re-quantization remain out of scope until this discriminator identifies the
failing boundary.

The comparison keeps BF16, in-house GPTQ, and cyankiwi AWQ in scope. AutoRound
remains deferred. The capable cluster may run independent arms concurrently;
each arm must first pass a short smoke stage before any long evaluation.

## Existing evidence and working hypotheses

The latest benchmark smoke establishes two evaluator defects independently of
model quality: MMLU-Pro sample indices were generated from the unfiltered
12,032-row dataset although lm-eval validates each subject against its filtered
`eval_docs`, and nested generated responses were classified as non-generative
by the health analyzer.

The model evidence is mixed but useful. The in-house GPTQ checkpoint previously
answered two trivial canonical HTTP prompts correctly, and its layer-5 residual
norm was within 0.01 percent of reference. In the latest benchmark samples it
instead emitted multilingual token salad. The cyankiwi AWQ control emitted
coherent reasoning but reached the 256-token smoke cap. Therefore the old
binary `quality_ok` predicate and hidden-state norms alone are insufficient.

The primary hypotheses are:

1. Offline dequantized GPTQ distributions already diverge, implicating the
   quantized weights, scales, zero points, or calibration.
2. Offline dequantized distributions are sound but vLLM distributions diverge,
   implicating packing, export metadata, loader interpretation, or kernels.
3. Short prompts conceal drift that accumulates on realistic prompt lengths or
   through later layers; this explains the apparently contradictory smoke
   evidence without treating either observation as invalid.

Calibration data becomes the leading suspect only if hypothesis 1 is supported.
A fresh 7-to-15-hour quantization run is not justified before then.

## Evaluator correctness changes

Preflight will size every loaded leaf task using `len(task.eval_docs)`, the same
filtered document collection consumed by lm-eval's iterator. It will validate
every generated sample index before GPU launch. A failure must report the
canonical task, leaf task, actual size, and maximum selected index.

Sample normalization will unwrap only singleton textual response containers,
including the observed `['text']` shape, before generation-health analysis.
Structured loglikelihood responses such as `[[score, false]]` must remain
structured and non-generative. Health output will then report token counts,
length-cap hits, periodic loops, and repeated n-gram fractions for generated
tasks.

## Probe-first evidence flow

For probe-enabled smoke arms, the teacher-forced distributional probe runs
before benchmark tasks. Its JSONL and summary are retained even if a later task
fails. A probe failure fails that arm and prevents its benchmark stage; a
successful probe proceeds to the exact-sample smoke evaluation.

The probe uses the same immutable token corpus and reports, per token and in
aggregate, reference-token log probability, top-k overlap, argmax agreement,
rank displacement, and distributional divergence available from the returned
top-k support. Comparisons must include quantization-oriented summaries such as
argmax flip ratio, top-k disagreement, log-probability error quantiles, and
bucketed drift by prompt position. GPTQ and AWQ run concurrently on separate
single-node 8-GPU allocations. BF16 may run concurrently if its distributed
runtime initializes; its failure must not erase the quantized-arm evidence.

## Boundary localization and decision rules

The existing layer-boundary diagnostics are reused rather than expanded
speculatively. If GPTQ's teacher-forced distribution differs materially from
AWQ or BF16, a focused boundary run captures matched inputs at later as well as
early decoder layers and the LM head. Norm similarity is treated only as a
finite-value sanity check; tensor-level distance and output-distribution drift
drive localization.

The executor returns enough evidence for this agent to make the diagnosis:

- offline dequantized GPTQ diverges from BF16 before vLLM loading: investigate
  quantization recipe, scale/zero-point computation, and calibration corpus;
- offline dequantized GPTQ is close but loaded GPTQ diverges: investigate
  checkpoint packing, metadata translation, vLLM loader, and kernels;
- GPTQ is close on short tokens but diverges by position or later boundaries:
  inspect accumulated error and the first divergent layer/module;
- GPTQ and controls remain close in teacher-forced probes: investigate
  autoregressive sampling, chat rendering, KV-cache, and long-generation paths.

No costly recalibration begins without the first condition or equivalent direct
evidence.

## BF16 distributed-runtime diagnostic

BF16 remains a control, not a prerequisite for collecting GPTQ/AWQ evidence.
In parallel, the executor first requests a plain Ray placement group containing
16 one-GPU bundles and records placement-group state, `ray status`, and both
nodes' current Ray logs. A vLLM initialization attempt is bounded to ten
minutes. If plain placement succeeds but vLLM stalls, the result is classified
as vLLM-Ray integration failure rather than insufficient cluster resources.

## Verification and handoff contract

CPU tests follow red-green TDD for filtered leaf sizing, invalid-index errors,
singleton response normalization, preservation of loglikelihood structures,
and probe-first shell ordering/failure behavior. The local cluster may run only
short smoke checks.

The capable-cluster handoff specifies parallel `srun` commands, node/GPU counts,
timeouts, checkpoint paths, expected artifacts, and stop/go gates. The executor
must push raw stdout/stderr, manifests, exact package versions, all probe JSONL
and summaries, generation-health files, boundary records when requested, Ray
placement evidence, return codes, and a factual deviations log. Production
evaluation remains locked until all selected smoke arms have valid probe and
task evidence.
