# GLM-5.3 W4AFP8 — AA-LCR v1.1 public-methodology reproduction

**Date:** 2026-09-13  
**Status:** Design approved; written specification pending user review  
**Target:** in-house GLM-5.3 W4AFP8, SGLang TP=16 across `ca-gpu02` and
`ca-gpu03`

## 1. Goal and claim boundary

Run AA-LCR v1.1 against the existing in-house GLM-5.3 W4AFP8 endpoint using
every public contract Artificial Analysis (AA) discloses for Intelligence
Index v4.3:

- the official AA dataset and extracted documents;
- 100 questions, three repeats per question;
- the exact v1.1 candidate prompt and document ordering;
- AA's reasoning-model generation policy for GLM-5.3;
- the exact v1.1 judge system and user prompts; and
- GPT-5.6 Luna at medium reasoning as the equality checker.

The result is named **“AA-LCR v1.1 public-methodology reproduction.”** It is not
an official AA-produced score. AA does not publish its complete internal runner
or the exact OpenAI judge snapshot used for its leaderboard, and our endpoint
is the in-house quantized deployment rather than AA's provider endpoint.

Do not aggregate this result into an “Artificial Analysis Intelligence Index
v4.3” score.

## 2. Authoritative sources and immutable identity

AA's own artifacts are authoritative:

- Dataset: <https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR>
- Required revision:
  `9a77ef56b717057ade24ceab4d273712a0b4f19e`
- Revision title: `v1.1: correct 16 answer keys and document the judge system
  prompt (#12)`
- Leaderboard:
  <https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning>
- Index v4.3 announcement:
  <https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-3>
- Current methodology:
  <https://artificialanalysis.ai/methodology/intelligence-benchmarking/>

The runner downloads these two files at the required revision, never from
mutable `main`:

- `AA-LCR_Dataset.csv` — 100 v1.1 questions and corrected answers.
- `extracted_text/AA-LCR_extracted-text.zip` — expected LFS SHA-256
  `5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a`.

It computes and records SHA-256 for every downloaded input, including the CSV
and README, before use. Archive extraction rejects absolute paths, parent
traversal, links, duplicate members, and unexpected files.

The run fails before endpoint traffic unless the dataset has exactly:

- 100 unique question IDs, 1 through 100;
- 30 document sets;
- every referenced document present;
- document order matching the semicolon-separated
  `data_source_filenames`; and
- prompts whose `cl100k_base` counts reproduce the dataset's published
  `input_tokens`.

NVIDIA NeMo Skills is prior art for prompt construction and dataset loading,
but is not the execution dependency. Its current `aalcr` path follows mutable
dataset `main`, carries the older v1.0 judge prompt, and defaults to a different
judge. Using it unchanged would silently violate v1.1.

## 3. Candidate generation contract

For each question, concatenate documents in the exact CSV order:

```text
BEGIN INPUT DOCUMENTS

BEGIN DOCUMENT 1:
...
END DOCUMENT 1

...

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

...

END QUESTION
```

Generate three independent attempts per question, for 300 terminal candidate
records:

- candidate URL: the existing
  `glm-5-3-w4afp8-sglang` OpenAI-compatible chat endpoint;
- served model: `glm-5.3-w4afp8`;
- reasoning enabled;
- temperature `0.6`;
- top-p `1.0`;
- maximum output tokens `131072`;
- effective client concurrency `2`; and
- transport retries only, up to 30 attempts, matching AA's published error
  policy.

The endpoint's measured KV pool is 598,848 tokens. AA-LCR prompts range up to
about 115K `cl100k_base` tokens; two worst-case prompt-plus-output requests fit
with headroom, while three do not. Concurrency two is therefore the fail-safe
ceiling unless a fresh, recorded token audit proves a larger width safe.

Before generation, capture `/get_server_info`, `/v1/models`, the repository
commit, and the complete request policy. Capture server info again after
generation. A changed model path, tokenizer, context/KV configuration,
reasoning parser, quantization mode, TP topology, or speculative-decoding
configuration invalidates the run.

Use only `message.content` as the answer sent to the judge.
`message.reasoning_content` is retained separately for provenance and token
analysis but is never mixed into the final answer. Persist the raw response,
usage, finish reason, question ID, and repeat index before marking generation
complete.

## 4. Judge contract and API-key boundary

Judge each persisted final answer with OpenAI model `gpt-5.6-luna` and
`reasoning.effort=medium`, using the exact AA-LCR v1.1 system and user prompts
from the pinned README.

The judge receives only:

- the public question;
- the public official answer; and
- the candidate's final answer.

It does not receive the 100K source-document prompt or hidden reasoning.

The expected judge output is a JSON object with one `verdict` whose normalized
value is exactly `CORRECT` or `INCORRECT`. Transport, rate-limit, server, and
malformed-output failures may retry up to 30 times. A persistent failure makes
the run incomplete; it is never silently scored incorrect, excluded from the
denominator, or manually guessed.

The OpenAI key exists only as `OPENAI_API_KEY` in Kubernetes Secret
`hd-openai-aa-judge` in namespace `evaluation`. The Job manifest references:

```yaml
env:
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: hd-openai-aa-judge
        key: OPENAI_API_KEY
```

No config option accepts a literal key. Manifests, command lines, logs,
checkpoints, result documents, exceptions, and test fixtures contain only the
environment-variable and Secret names. Documentation provides a PowerShell
stdin workflow that prompts securely and applies the Secret without writing
the key to disk or shell history.

The judge client always passes
`base_url="https://api.openai.com/v1"` explicitly. An ambient
`OPENAI_BASE_URL` cannot redirect the key. Every preflight and judgment audit,
plus publication identity, records that exact endpoint.

The preflight verifies access to `gpt-5.6-luna` and executes one synthetic
judge canary. The recorded judge identity includes requested model, reasoning
effort, returned model identifier, endpoint, SDK version, and UTC time. Because
AA publishes no judge snapshot ID, any result records that exact limitation.
The returned model must be exactly `gpt-5.6-luna`; a mismatch is a retryable
judge protocol error and bounded exhaustion leaves the unit terminally
incomplete.

## 5. Components and data flow

Implement one focused Python module with four independently resumable commands:

1. `prepare` — download, hash, safely extract, validate, and construct prompts.
2. `generate` — call the candidate endpoint and checkpoint terminal responses.
3. `judge` — grade already-persisted final answers through OpenAI.
4. `summarize` — validate completeness and publish immutable exports.

A dedicated, hash-pinned evaluation environment is created by a CPU-only
staging Job. Do not mutate the existing GPQA environment or install
dependencies ad hoc into a running pod.

SQLite is the authoritative checkpoint because generation and judging update
independent state:

- `(question_id, repeat_index)` is the candidate primary key;
- each terminal generation record is immutable;
- each judgment is keyed by candidate identity plus a judge-contract hash;
- one successful preflight audit is stored in an immutable singleton table;
- changing dataset, prompts, generation policy, endpoint identity, or judge
  contract creates a new run fingerprint; and
- resume fills only missing transport work under the same fingerprint.

Candidate generation is pinned to `glm-5.3-w4afp8`. Both endpoint snapshots
record expected and observed served model, `/v1/models` must resolve that exact
identifier before generation and after generation, and every candidate request
uses the same identifier.

Published artifacts are:

- `run-manifest.json`;
- `candidates.jsonl`;
- `judgments.jsonl`;
- `summary.json`;
- `report.md`; and
- a file-digest manifest.

Publish exports atomically and without overwrite. The summary exists only when
all 300 candidates and all 300 judgments are terminal and valid. Headline
pass@1 is `CORRECT / 300`; also report question-macro accuracy, per-repeat
accuracy, category breakdown, output-token distribution, truncations, empty
answers, total judge retries (including preflight), and an explicit zero
terminal-judge-failure count.

`run --canary --limit 1 --repeats 1` is the only partial combined run. It
preflights the judge before candidate generation, persists one candidate, one
judgment, all audits, and prints a checkpoint status record. It intentionally
never summarizes or publishes a headline bundle. Combined runs without
`--canary` remain strict 100 × 3.

## 6. Failure and safety discipline

- Candidate and judge phases are separable. OpenAI failure never regenerates a
  candidate or spends additional GPU time.
- A returned candidate response is terminal, including empty, truncated, or
  context-rejected responses. Resume handles transport gaps only.
- `finish_reason=length` remains a scored candidate and is reported as
  truncated; it is not retried with a larger budget.
- No partial run receives a headline score.
- No secret value may appear in an exception or serialized HTTP request.
- Dataset/prompt/judge changes require a fresh run ID and fingerprint.
- Candidate generation uses the existing two-node serve and therefore requires
  an explicit GPU-workload approval immediately before launch, plus the
  authoritative Rancher availability/occupancy check.
- Creating or replacing the Kubernetes Secret and writing cluster/NFS
  artifacts also require explicit approval immediately before execution.

## 7. Verification plan

CPU tests cover:

- exact v1.1 system/user and candidate prompts, with pinned hashes;
- dataset cardinality, corrected-answer revision, document ordering, token
  counts, and safe extraction;
- request payloads and separation of final answer from reasoning;
- strict judge JSON parsing and retry classification;
- fingerprint sensitivity;
- immutable terminal records and resume behavior;
- atomic/no-clobber publication;
- summary recomputation from primitive records; and
- key redaction across configs, logs, exceptions, and artifacts.

Integration tests use fake candidate and judge HTTP servers to exercise
timeouts, rate limits, malformed JSON, duplicate responses, interruption,
resume, and complete publication without external traffic.

Live execution gates are:

1. CPU-only plan and dataset validation.
2. OpenAI model-access and synthetic judge canary.
3. One AA-LCR question × one repeat against the candidate endpoint.
4. Independent inspection of candidate, reasoning separation, judge prompt,
   token usage, and persisted provenance.
5. A fresh Rancher occupancy report and explicit user authorization.
6. Full 100 × 3 run at candidate concurrency two.
7. Offline/resumable judging and strict final validation.

No live call, Kubernetes write, or GPU request is part of implementation
itself; each is separately authorized at its execution gate.

Low-severity limitation: the current CLI re-prepares the pinned official
dataset for every phase, so `summarize` still requires dataset-network
availability before it opens an otherwise complete checkpoint.
