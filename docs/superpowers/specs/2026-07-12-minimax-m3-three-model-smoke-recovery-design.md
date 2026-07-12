# MiniMax-M3 Three-Model Smoke Recovery Design

## Scope

Resume quality evaluation with BF16, in-house GPTQ, and cyankiwi AWQ. Defer
aquaman AutoRound because faithful loading requires a pinned external mixed-bit
plugin, repository-specific key translation/dequantization, callable config
overrides, and equivalent integration in both lm-eval and the distributional
probe. Serving performance remains out of scope.

## Matrix and resources

The active matrix contains three models. AutoRound remains recorded as deferred
metadata with its checkpoint, revision, and reason. Smoke launches three arms
concurrently on four nodes: BF16 uses two nodes/16 H100, and each quantized arm
uses one node/8 H100. Production launches six arms on eight nodes after smoke.

## Reasoning semantics

MiniMax's chat template defaults to adaptive thinking and terminates reasoning
with `</mm:think>`. The evaluator leaves `enable_thinking` unset and supplies
only `think_end_token: </mm:think>` so generated-task metrics exclude reasoning
without violating lm-eval 0.4.12's restriction on thinking mode for
multiple-choice/loglikelihood tasks. Preflight rejects invalid combinations
before any GPU allocation.

## Failure handling

Smoke-gate validation is total: missing models, invalid artifacts, and zero or
missing probe timing produce explicit failed checks and reasons, never an
exception. A gate artifact is always written.

BF16 uses a separate two-node Ray topology preflight before model loading. It
records one file per rank with hostname, selected private IP, Ray/Python
versions, relevant Slurm variables, startup return code, `ray status`, and node
and GPU visibility. BF16 evaluation starts only if two alive nodes and 16 GPUs
are visible.

## Verification and handoff

CPU tests cover matrix topology, reasoning validation, zero-evidence gates,
and launcher contracts. The executor first runs the Ray-only check, then the
three concurrent smoke arms. Production remains locked behind
`ready_for_production: true`. All rank logs, smoke artifacts, versions, job IDs,
and deviations are returned through Git.
