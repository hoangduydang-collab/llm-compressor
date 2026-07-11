# MiniMax-M3 Paired Quality Diagnostic Design

## Objective

Identify the first serve-time boundary at which the portable full-calibration
MiniMax-M3 AWQ W4A8 checkpoint diverges from the working
`cyankiwi/MiniMax-M3-AWQ-INT4` reference. The experiment must distinguish an
infrastructure pass from a quality pass and return enough compact evidence for
analysis on a different cluster.

The immediate deliverable is diagnostic evidence, not a speculative loader
fix or another quantization run.

## Confirmed starting point

- Original AWQ and GPTQ checkpoints load but generate repetitive garbage.
- Serving the AWQ weights without runtime FP8 activation quantization remains
  garbage, so FP8 activation quantization is not sufficient to explain the
  failure.
- The routed projections used descriptive `gate_proj`, `down_proj`, and
  `up_proj` keys while the installed vLLM loader expected `w1`, `w2`, and `w3`.
- A payload-preserving re-export corrected those aliases and loaded
  successfully, but generation remained garbage. The alias mismatch was real
  but was not the complete quality root cause.
- The working reference and the portable checkpoint must be tested through the
  same vLLM installation and launch envelope before their runtime evidence can
  be compared.

## Scope

### In scope

1. Make MiniMax-M3 quality diagnostics available to both W4A8 checkpoints and
   the W4A16 cyankiwi control.
2. Report infrastructure health separately from output quality.
3. Run one sequential paired comparison on the same eight-GPU node.
4. Capture loader mappings and small loaded-parameter fingerprints for
   `lm_head`, shared experts, routed experts, and the attention/indexer boundary.
5. Measure shared-expert participation during a real prompt prefill.
6. Commit a compact, self-contained evidence bundle through Git.

### Out of scope

- CUDA-graph root-cause work. The quality comparison uses eager execution to
  avoid graph-capture and shared-stream race confounders.
- Re-quantization.
- Rewriting vLLM loader behavior before a failing boundary is identified.
- Committing checkpoints, core dumps, or multi-gigabyte raw logs to Git.
- Deleting original or portable checkpoints.

## Alternatives considered

### Immediate run with the current commit

This minimizes preparation, but the diagnostics are installed by
`serve_verify` only inside its W4A8 branch. A fresh reference-first run can
therefore omit the cyankiwi instrumentation. The current nonempty-output check
also reports repetitive garbage as successful. This option risks spending the
GPU allocation without producing decisive evidence.

### Exhaustive model-wide instrumentation

This could fingerprint every parameter and intermediate, but it increases log
volume, synchronization, runtime, and the chance that diagnostics perturb the
model. It is unnecessary before checking the highest-ranked boundaries.

### Selected: hardened, bounded paired comparison

Install dormant MiniMax-M3 diagnostics independently of the W4A8 kernel
patches, add an explicit quality assessment, and compare only the parameters
and intermediates needed to drive the existing hypothesis tree. This spends a
small amount of CPU-side work to make one GPU run substantially more decisive.

## Architecture

### 1. MiniMax-M3 diagnostic installation

`pipeline/serve_verify.py` will treat two kinds of site-package changes
separately:

- Required W4A8 execution patches remain guarded by
  `_is_w4a8_moe_scheme(...)`.
- Dormant loader audit, parameter fingerprint, and MoE forward probe
  installation occurs for every MiniMax-M3 checkpoint before vLLM workers are
  created. The environment variables still decide whether a diagnostic runs.

This makes the reference and candidate independent of prior persistent
site-package state and experiment ordering.

### 2. Output assessment

Infrastructure and quality have separate report fields:

- `loaded`: vLLM construction completed.
- `generation_completed`: generation returned a nonempty response.
- `quality_ok`: every required deterministic check passed.
- `ok`: infrastructure completion only, retained for compatibility if needed.

The quality assessment uses a small fixed prompt set rather than a single
nonempty string. It records each output and checks:

- an expected answer substring for simple factual prompts;
- excessive token or substring repetition;
- empty or non-finite/decoder-error outputs;
- aggregate `quality_ok` without overwriting the raw generations.

At minimum, the prompts cover a simple capital question and a simple arithmetic
question. Generation is greedy with a bounded token count. The exact prompts,
expected substrings, tokenizer/model name, and sampling arguments are stored in
the result manifest.

This is a smoke-quality discriminator, not a benchmark of model accuracy.

### 3. Parameter fingerprints

An environment-gated worker diagnostic emits compact records after model
construction. Each record contains:

- rank;
- checkpoint case name;
- canonical parameter name;
- dtype and shape;
- finite fraction for floating-point tensors;
- full-tensor norm for small BF16 control tensors where affordable;
- a deterministic bounded sample digest and sample statistics for large or
  packed tensors.

The diagnostic targets:

- `lm_head`;
- one early and one late shared-expert layer;
- corresponding routed-expert packed weights/scales;
- one early and one late attention q/k/v and MSA-indexer boundary.

It must not retain full tensor copies, gather full model parameters to one rank,
or print raw tensor contents. Cross-format W4A16/W4A8 routed weights are not
expected to have matching digests; fingerprints establish whether parameters
exist, are finite/nonzero, and are stable across ranks. BF16 components derived
from the same base model can be compared more directly.

### 4. Real-prefill MoE evidence

The existing MoE probe remains capture-safe and environment-gated. For a small
number of real-prefill calls it records input, shared-expert, routed, and final
MoE norms, plus whether the shared module exists. The paired run enables its
recompute mode only if the normal contribution metrics cannot distinguish a
dropped shared path.

Warmup, profiling, and CUDA-graph capture calls do not consume the real-prompt
probe budget. The paired quality run uses eager mode, further reducing this
ambiguity.

### 5. Paired experiment runner

A single runner executes two cases sequentially on the same clean eight-GPU
node:

1. `cyankiwi_reference`
2. `portable_awq_w4a8`

Both cases use the same:

- Git commit;
- Python environment and vLLM installation;
- GPU node and topology;
- TP/EP configuration;
- eager mode;
- model length, GPU utilization, tokenizer settings, and prompt set;
- environment snapshot and diagnostic flags.

The runner stops a case cleanly, verifies that its worker processes are gone,
and checks GPU availability before starting the next case. A reference quality
failure stops the matrix: candidate evidence is not interpretable against an
invalid control.

The remote agent may resolve runtime-only issues such as scheduler invocation,
paths, ports, and stale owned workers. Any deviation from the committed command
or envelope is recorded in the manifest rather than silently normalized.

## Evidence bundle

The paired runner creates a Git-sized directory containing:

- `run_manifest.json`: commit, host, GPU topology, package versions, checkpoint
  paths and index/config hashes, exact commands, environment knobs, and any
  deviations;
- one result JSON per case with infrastructure and quality fields;
- loader-audit summaries;
- parameter-fingerprint JSON Lines;
- MoE-probe extracts;
- raw generated outputs;
- bounded log excerpts around warnings/errors and diagnostic markers;
- `comparison.json` with the reference/candidate decision-tree outcome;
- checksums and absolute locations for large logs retained outside Git.

Secrets, full environment dumps, checkpoints, and unbounded raw logs are not
included. The result commit message identifies the diagnostic-code commit it
ran.

## Decision tree

1. **Reference fails to load or fails smoke quality:** classify the run as an
   invalid baseline and stop. Resolve environment/reference serving before
   inspecting candidate quality.
2. **Reference passes and candidate has missing, zero, non-finite, or unmatched
   `lm_head`:** isolate and repair the `lm_head` loader boundary.
3. **Reference passes and candidate lacks a shared module, shared weights, or a
   real shared-expert contribution:** isolate and repair shared-expert config,
   key mapping, loading, or exactly-once forward addition.
4. **Those boundaries are clean but candidate quality fails:** compare the
   q/k/v and MSA-indexer construction/loading path. Determine whether the BF16
   indexer is incorrectly fused into an INT4 projection.
5. **Attention/indexer is clean:** inspect W4A8 packed routed-expert loading and
   dequantized bounded samples against the source checkpoint.
6. **Only after loader/runtime boundaries are cleared:** design an experts-only
   W4A16 quantization experiment.

Each repair is a separate minimal hypothesis test followed by the same paired
quality smoke. Multiple loader changes are not bundled into one trial.

## Error handling and operational safety

- Diagnostic installation fails loudly when explicitly enabled but its target
  class or required hook is absent. Dormant diagnostics do not block ordinary
  serving.
- The runner distinguishes timeout, load failure, generation failure, quality
  failure, missing diagnostic evidence, and cleanup failure.
- Partial result directories carry an `inconclusive` status and are preserved.
- The runner never deletes checkpoints and never kills another user's
  processes.
- The portable 225 GB re-export is reused; no additional full checkpoint copy
  is part of this experiment.

## Testing and acceptance criteria

CPU tests cover:

- diagnostics are installed for MiniMax-M3 W4A16 and W4A8 paths;
- W4A8 execution patches remain W4A8-only;
- quality assessment accepts normal answers and rejects empty/repetitive
  garbage;
- injected audit/fingerprint wrappers execute against representative fake
  loaders, report matches, and restore original weight loaders;
- the paired runner dry-run emits two cases with identical comparison settings;
- result comparison stops on a failed reference and selects the correct next
  hypothesis for each evidence class;
- the evidence bundle contains required provenance without large artifacts or
  obvious secrets.

The GPU experiment is successful as a diagnostic run when:

- the cyankiwi reference loads and passes the smoke-quality checks;
- both cases return complete loader, fingerprint, generation, and probe
  records—or the candidate exposes a decisive earlier failure;
- `comparison.json` selects exactly one next investigation boundary;
- the compact evidence commit is sufficient to audit that classification on
  this cluster.

Quality resolution itself requires the candidate to pass the same paired smoke
after a separately reviewed minimal fix. It is not established by a successful
load or a nonempty response.

## Responsibilities and communication

The primary analysis agent owns experiment design, static code changes, CPU
verification, evidence interpretation, and selection of the next hypothesis.
The GPU-cluster agent owns execution, runtime-only adaptation, preservation of
large artifacts, and committing the compact evidence bundle. The GPU agent is
expected to use judgment when an issue is best understood in the live
environment, while avoiding speculative source changes that would invalidate
the paired comparison.

Git commits are the durable interface. Each handoff names its prerequisite
commit; each result commit names the code commit executed and records all
runtime deviations.

### Remote-agent handoff contract

The execution handoff is a committed, copyable runbook with four explicit
sections: objective, preflight, commands, and return checklist. It explains why
each diagnostic matters and identifies which settings are invariants versus
which runtime details the GPU agent may adapt. The receiving agent confirms the
code commit and preflight before consuming an eight-GPU allocation.

The GPU agent returns evidence rather than only a narrative conclusion. Before
declaring the handoff complete, the result commit must answer all of the
following:

1. **What ran?** Exact Git commit, commands, case order, start/end times,
   checkpoint paths plus config/index hashes, diagnostic flags, and every
   departure from the committed runbook.
2. **Where did it run?** Host, scheduler allocation/job identifiers, GPU model,
   GPU topology, driver/CUDA versions, Python environment, vLLM commit/version,
   Torch, compressed-tensors, FlashInfer, and relevant package versions.
3. **Was the baseline valid?** Reference load status, raw prompt outputs,
   per-prompt quality decisions, and any warnings or missing diagnostics.
4. **How did the candidate differ?** Candidate load/generation/quality results,
   loader matches and misses, parameter fingerprints, MoE contribution records,
   and a structured reference-versus-candidate comparison.
5. **What failed operationally?** Timeouts, OOMs, scheduler changes, orphan
   cleanup, instrumentation exceptions, missing records, retries, and whether a
   retry reused or changed the environment.
6. **Where is everything else?** Absolute paths, byte sizes, and SHA-256 hashes
   for full logs or other large artifacts kept outside Git, plus retention
   expectations if those paths are temporary.

The agent must preserve raw outputs even when its interpretation seems obvious.
It may add runtime analysis and propose the next hypothesis, but it does not
replace requested evidence with a summary or silently rerun under different
settings. If a required record cannot be collected, the result is marked
`inconclusive` with the reason and the remaining evidence is still committed.

The primary agent reviews the returned manifest mechanically before analyzing
the model result. Missing provenance, an invalid reference, changed comparison
variables, or absent required diagnostics causes a targeted evidence follow-up
rather than a speculative code fix.
