# GLM-5.3 W4AFP8 — public AA v4.2 evaluation set

**Date:** 2026-09-07  
**Status:** Evaluation scope specification  
**Models:** in-house GLM-5.3 W4AFP8 and PhalaCloud GLM-5.3 W4AFP8

## Motivation

The existing seven-task `lm-eval` suite is a useful paired instrument: both
W4AFP8 checkpoints see the same prompts, serving engine, tokenizer, and scoring
code. It is not a public GLM-5.3 leaderboard reproduction, however. In
particular, its generative tasks use an internal 32,768-token budget and its
GPQA result is one greedy sample rather than the Artificial Analysis (AA)
protocol.

The next evaluation set should produce task-level results that can be compared
with independently published GLM-5.3 results. AA Intelligence Index v4.2 is the
primary reference because AA publishes:

- the ten constituent evaluations and their weights;
- task populations, repeat counts, scoring rules, and major execution limits;
- public benchmark sources where release is permitted; and
- independently measured GLM-5.3 results.

“Comparable” in this document does **not** mean “uses the same dataset name.”
It means matching, as far as public artifacts permit, the dataset revision,
population, prompt, repeat count, model controls, tools, sandbox, judge, and
scoring rule. Any mismatch must remain visible in the result manifest.

This project initially targets the seven v4.2 tasks whose questions are public.
The remaining three are recorded but excluded because an official result cannot
be independently reproduced from public assets. AA itself states that 40% of
the v4.2 Index weight uses held-out data. See the
[v4.2 announcement](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-2)
and [current methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking/).

## Context-window contract

AA's general rule for a reasoning model is to request the maximum output allowed
by the model creator. Z.ai discloses a 128K maximum output for GLM-5.3, so the
AA-style request budget is 131,072 output tokens.

For a single request:

```text
required total context >= rendered input tokens + 131,072 output tokens
```

Consequences for the current deployment:

- A 131,072-token total window cannot reproduce that request for any non-empty
  prompt.
- The measured 164,800-token SGLang window leaves at most 33,728 tokens for the
  rendered prompt when the full output allowance is retained.
- Short static tasks can probably use the 164,800-token deployment, but their
  rendered prompts must be measured with the served GLM tokenizer first.
- AA-LCR supplies about 100K input tokens and therefore needs approximately
  231K tokens plus chat-template overhead for the full GLM output allowance.
  A 256K deployment is the practical minimum target.
- GDP.pdf has variable full-document inputs. Its LiteParse/OCR output must be
  tokenized task by task before choosing a serve window.
- Agentic tasks build histories over many turns. Their harness must compact or
  summarize before the next request exceeds the available input budget.

A run with a lower `max_new_tokens` may still be useful as a paired quality
measurement, but it must be labelled **AA-inspired**, not strictly
AA-output-policy comparable.

## Complete AA v4.2 scope

| Category | Task | Population × repeats | Public reproducibility | In this phase |
|---|---|---:|---|---|
| Agents | AA-Briefcase | 91 × 1 | Blocked: official scenarios are private; only Briefcase-Lite examples are public | No |
| Agents | GDPval-AA v2 | 220 × 1 | Conditional: public tasks and Stirrup; AA environment, judges, anchors, and sampling must be recreated | **Yes** |
| Agents | τ³-Banking | 97 × 5 | Conditional-high: public upstream suite and grader; AA simulator/judge configuration is external | **Yes** |
| Coding | Terminal-Bench v2.1 | 89 × 3 | High for the upstream task; conditional for AA environment parity | **Yes** |
| Coding | SciCode | 288 subproblems × 3 | High: public task and executable scorer; exact AA prompt/environment still must be pinned | **Yes** |
| General | AA-Omniscience | 6,000 × 1 | Blocked: only 600 questions are public | No |
| General | GDP.pdf | 100 × 5 | Conditional: public tasks; AA uses its own document preparation and judge contract | **Yes** |
| General | AA-LCR v1.1 | 100 × 3 | Conditional-high: public inputs and answers; external AA judge must match | **Yes** |
| Scientific reasoning | HLE, text-only | 2,158 × 1 | Conditional-high: public dataset and prompt; external AA equality checker must match | **Yes** |
| Scientific reasoning | CritPt | 70 × 5 | Blocked: exact grading requires the restricted official grading API and held-out solutions | No |

The selected set is therefore:

1. Fully public/upstream: Terminal-Bench v2.1, SciCode, and τ³-Banking.
2. Public questions with AA-specific infrastructure: HLE, AA-LCR v1.1,
   GDP.pdf, and GDPval-AA v2.

## Selected task contracts

### Terminal-Bench v2.1

| Item | Contract |
|---|---|
| Capability | Agentic coding and terminal use |
| AA population | 89 tasks, 3 repeats per task |
| Scoring | pass@1 averaged across repeats; a task passes only when every task test passes |
| Public source | [Terminal-Bench 2.1 repository](https://github.com/harbor-framework/terminal-bench-2-1) and [public leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.1) |
| AA execution | Terminus 2 agent in an E2B sandbox; maximum 250 episodes; two-hour timeout or the task-specific longer timeout |
| Model interface | Multi-turn text model with terminal actions |
| Context requirement | Not a fixed single prompt. A 130K/164.8K window can work only if the agent controls history growth. Full AA GLM output-policy parity additionally requires enough room for the 131,072-token output setting on every call. |
| Other requirements | Container images, task verifiers, terminal agent, sandbox capacity, deterministic environment pinning |
| Reproducibility assessment | The benchmark and verifier are public. An upstream-comparable score is achievable. AA comparability additionally requires matching Terminus 2, repeat count, episode limit, timeouts, and sandbox behavior. |

### SciCode

| Item | Contract |
|---|---|
| Capability | Scientific Python code generation |
| AA population | 288 test subproblems, 3 repeats |
| Scoring | Subproblem-level pass@1; generated step script must pass all unit tests |
| Public source | [SciCode project and dataset](https://scicode-bench.github.io/) |
| AA execution | Scientist-annotated background included in the prompt; isolated executor; 300-second execution timeout; dataset/scorer v1.0.1 |
| Model interface | Single-turn text generation followed by code extraction and execution |
| Context requirement | Expected to fit inside the 164,800-token serve with the full GLM output allowance because rendered inputs should be below 33,728 tokens; confirm by tokenizing every prompt before launch. A 130K total window cannot retain the full 131,072-token output request. |
| Other requirements | Isolated Python executor, exact dependency image, code-extraction logic, unit-test timeout enforcement |
| Reproducibility assessment | One of the strongest candidates for an independently reproducible result because the task and executable scorer are public. Pin the exact task commit, prompt construction, environment, repeats, and extraction behavior. |

### τ³-Banking

| Item | Contract |
|---|---|
| Capability | Knowledge retrieval, policy reasoning, and multi-step banking actions |
| AA population | 97 tasks, 5 repeats |
| Scoring | pass@1 against final backend database state |
| Public source | [Sierra tau2-bench](https://github.com/sierra-research/tau2-bench), upstream v1.0.1 dataset and grader |
| AA execution | About 700 policy documents / 195K corpus tokens; BM25 lexical search plus `grep` rather than full-corpus prompt stuffing; maximum 200 simulation steps |
| Model interface | Dual-control conversation with retrieval and banking tools |
| Context requirement | The 195K corpus does not need to fit in one request. Per-turn fit depends on retrieved evidence and history management. At 164,800 total context, retaining the full GLM output allowance leaves 33,728 prompt tokens. |
| Other requirements | Banking backend, tool schemas, `bm25_grep`, GPT-5.4 Mini at medium reasoning as both user simulator and natural-language assertion judge |
| Reproducibility assessment | Questions and upstream grader are public. The score is AA-comparable only if the upstream v1.0.1 state machine, retrieval mode, repeats, step limit, simulator, and assertion judge are matched. |

### HLE, text-only

| Item | Contract |
|---|---|
| Capability | Frontier academic reasoning |
| AA population | 2,158 text-only questions from the May 2025 HLE revision; 1 attempt each |
| Scoring | pass@1 using an equality-checker LLM, with the original HLE grading prompt |
| Public source | [cais/hle](https://huggingface.co/datasets/cais/hle) |
| AA execution | Text-only subset; multimodal questions are excluded |
| Model interface | Single-turn text response |
| Context requirement | Questions are not long-context inputs and should fit under 164,800 with the full output allowance; measure the longest rendered prompt. A 130K total window requires lowering the requested output budget. |
| Other requirements | **GPT-5.6 Luna (medium)** equality-checker judge and the published HLE checker prompt |
| Public comparator | AA publishes a GLM-5.3 (max) text-only HLE score. Do not compare against Z.ai's separate “HLE with tools” result. |
| Reproducibility assessment | Dataset and prompt are public. Strict score comparability depends on using the same judge model and checker prompt; substituting a judge creates a new scoring protocol. |

### AA-LCR v1.1

| Item | Contract |
|---|---|
| Capability | Retrieval and reasoning across multiple long documents |
| AA population | 100 questions, 3 repeats |
| Scoring | pass@1 with an equality-checker LLM |
| Public source | [ArtificialAnalysis/AA-LCR](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR) |
| AA execution | v1.1 system prompt, 16 corrected answer keys, about 100K input tokens per question measured with `cl100k_base`; v1.1 results are not comparable with v1.0 |
| Model interface | Single-turn long-context text response |
| Context requirement | AA states a 128K minimum, but strict GLM output-policy reproduction needs roughly 100K input + 131,072 output + template overhead. Target at least 256K total context. The current 164,800 serve leaves only about 64.8K output after a 100K input. |
| Other requirements | **GPT-5.6 Luna (medium)** equality-checker judge; v1.1 judge prompt and exact corrected dataset revision |
| Reproducibility assessment | Public and reproducible once enough context is available and the judge contract is matched. A capped 164.8K run is useful but is not strict AA output-policy parity. |

### GDP.pdf

| Item | Contract |
|---|---|
| Capability | Single-turn reasoning over long professional PDFs |
| AA population | 100 tasks across 10 domains, 5 repeats; 4,592 source pages and 1,275 grading criteria |
| Scoring | Headline All-pass rate: an attempt passes only if every criterion passes. Secondary task-macro Mean Pass averages criterion pass rates equally across tasks and repeats. |
| Public source | [surgeai/GDP.pdf](https://huggingface.co/datasets/surgeai/GDP.pdf) |
| AA execution | LiteParse text extraction plus OCR where needed; every extracted page is included. Image-capable models also receive page images, but text-only GLM-5.3 receives extracted text only. Single turn, no browsing or tools. |
| Model interface | One long text request and free-form answer |
| Context requirement | Unknown until AA-style extraction is reproduced and tokenized with the GLM tokenizer. Required context for strict output parity is `max extracted prompt + 131,072 + template overhead`; neither 130K nor 164.8K can be assumed sufficient. |
| Other requirements | LiteParse/OCR pipeline; **GPT-5.6 Luna Medium** judging every criterion; complete-verdict checks; errors and missing attempts score zero |
| Reproducibility assessment | The tasks are public, but AA and Surge results are not directly comparable: AA uses LiteParse/OCR and GPT-5.6 Luna Medium, while Surge uses provider PDF input and a different judge. Match the AA preparation and judge contracts before comparing with AA. |

### GDPval-AA v2

| Item | Contract |
|---|---|
| Capability | Agentic professional work producing files such as documents, spreadsheets, slides, and diagrams |
| AA population | 220 public gold tasks, 1 model rollout per task |
| Scoring | Blind pairwise grading and Bradley-Terry Elo, anchored to human expert deliverables at 1,000 |
| Public source | [OpenAI GDPval](https://huggingface.co/datasets/openai/gdpval) and AA's open-source [Stirrup](https://github.com/ArtificialAnalysis/Stirrup) agent harness |
| AA execution | E2B Linux sandbox; code execution, web fetch, Brave web search, finish/abandon tools, and view-image only for vision models; maximum 250 turns |
| Model interface | Multi-turn agent with file-system and web tools |
| Context requirement | Stirrup summarizes after a completed turn exceeds 70% of the context window and unwinds earlier history if necessary. A 130K or 164.8K model can run, but it will summarize earlier than AA's 1M-context GLM-5.3 endpoint and may therefore produce a different measurement. |
| Other requirements | AA-compatible sandbox image and repaired Office inputs; Brave Search API; web fetch; three-judge panel—GPT-5.5 medium, Gemini 3.1 Pro Preview high, and Claude Opus 4.8 high—plus comparison opponents and human reference deliverables |
| Reproducibility assessment | Task generation is publicly reproducible. The absolute AA Elo is not independently reproducible from the dataset alone because it depends on AA's patched inputs, judge sampling, opponent pool, and human anchors. A same-harness Ours-vs-Phala paired comparison remains valid but must not be presented as AA's Elo. |

## Interpretation

The seven selected tasks do not have one shared runnable “AA v4.2 harness.”
They comprise several independent upstream harnesses plus AA-specific execution
and grading contracts:

- SciCode is the cleanest conventional quality evaluation.
- HLE is straightforward inference but judge-dependent.
- Terminal-Bench and τ³-Banking are public agent benchmarks, not static prompt
  suites.
- AA-LCR directly tests the long-context behavior that the existing short
  `lm-eval` suite does not cover.
- GDP.pdf requires a document-token audit before deployment sizing.
- GDPval-AA v2 can reproduce task rollouts with Stirrup, but not AA's absolute
  Elo without its reference and comparison ecosystem.

Results should therefore be reported per task. They must not be aggregated into
an “AA Intelligence Index v4.2” score: three constituent datasets are excluded,
and several selected tasks cannot exactly reproduce AA's private execution or
grading state.
