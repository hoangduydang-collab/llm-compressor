# MiniMax-M3 Canonical Chat Quality Matrix Design

## Goal

Establish a valid MiniMax-M3 quality baseline and compare the portable W4A8
candidate against cyankiwi through both offline vLLM and canonical HTTP chat,
without involving CUDA graphs or raw-completion prompts.

## Experiment matrix

Run these four arms concurrently on separate eight-GPU nodes:

| Arm | Checkpoint | Interface |
| --- | --- | --- |
| `reference_offline_chat` | cyankiwi W4A16 | Offline `LLM.generate` after the official chat template |
| `candidate_offline_chat` | Portable W4A8 | Offline `LLM.generate` after the official chat template |
| `reference_http_chat` | cyankiwi W4A16 | OpenAI-compatible `/v1/chat/completions` |
| `candidate_http_chat` | Portable W4A8 | OpenAI-compatible `/v1/chat/completions` |

Every arm uses the same two user messages, greedy sampling, 64 output tokens,
`thinking_mode=disabled`, TP=8, expert parallelism, eager mode, block size 128,
FP8 KV cache, `max_model_len=2048`, GPU utilization 0.85, custom all-reduce
disabled, and shared-expert auxiliary streaming disabled. Loader audit, MoE
probe, and parameter fingerprint diagnostics are off so this measures the
normal quality path. The HTTP arms explicitly use eager mode so the separately
tracked CUDA-graph issue cannot contaminate this quality experiment.

## Components

### Canonical offline generation

`pipeline/serve_verify.py` obtains the tokenizer from the constructed vLLM
instance and applies its official chat template to each quality message with
`add_generation_prompt=True` and `thinking_mode="disabled"`. It sends each
rendered prompt independently to `LLM.generate` and records the rendered prompt,
generated token IDs, finish reason, stop reason, and raw decoded text.

Non-MiniMax models retain their existing raw configured prompt behavior.

### Per-arm runner

A new shell runner accepts exactly one matrix arm. It validates the selected
checkpoint, records the effective environment and command, runs either the
offline serve verifier or an eager HTTP server plus two chat requests, and
always builds compact evidence before exiting. It must not launch another arm
or modify a checkpoint.

### Parallel submission

A thin submission helper emits or submits four independent Slurm jobs with a
shared matrix ID and unique arm IDs, logs, ports, and result directories. A
dry-run mode validates commands without requiring GPUs, NFS assets, or Slurm.
Scheduler partition and node selection remain runtime decisions for the
executor and must be recorded as deviations.

### Evidence and classification

Each arm returns a manifest, checkpoint config/index hashes, software and GPU
provenance, exact command, raw outputs or HTTP response bodies, normalized
quality cases, termination metadata, notable logs, full-log hashes, retries,
and deviations. A CPU-only aggregator classifies:

- whether each reference interface passes;
- whether each candidate interface passes;
- whether offline and HTTP agree per checkpoint;
- whether the matrix is conclusive or missing an arm;
- the next quality boundary without making a CUDA-graph conclusion.

The candidate comparison is valid only against a passing reference using the
same interface. Infrastructure failure remains distinct from semantic quality
failure.

## Error handling

- Every arm has an independent result directory and may fail without stopping
  other arms.
- A server startup failure preserves the first traceback and returns an
  infrastructure verdict.
- A malformed or missing HTTP response is evidence, not an automatic retry.
- Any retry uses a new arm attempt directory and records changed variables.
- No arm enables diagnostics, CUDA graphs, re-quantization, checkpoint edits,
  or candidate repair.

## Verification

CPU tests cover official chat-template application, preservation of rendered
prompts and termination metadata, arm-to-command mapping, dry-run submission,
HTTP response classification, missing-arm behavior, and legacy non-MiniMax
generation. Shell scripts pass `bash -n`; Python files compile; all focused
tests and existing MiniMax-M3 quality tests must pass before handoff.

## Completion boundary

This implementation is ready for the executor when one pushed commit provides
the tested four-arm harness and an updated handoff with exact commands and
required Git return artifacts. The quality issue itself is resolved only after
runtime evidence shows the candidate passes canonical serving or after a
candidate-specific fault is identified and fixed. CUDA-graph RCA remains
deferred until then.
