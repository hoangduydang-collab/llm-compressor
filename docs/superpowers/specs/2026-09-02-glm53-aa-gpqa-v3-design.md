# GLM-5.3 W4AFP8 GPQA Diamond — NVIDIA AA v3 clone

**Date:** 2026-09-02

**Status:** Approved for implementation (harness locked 2026-09-02;
delivery switched to PyPI 26.3 the same day)

**Workflow state:** spec written; implementation plan not started until this
file is reviewed

## Objective

Score the two GLM-5.3 W4AFP8 quality arms on **GPQA Diamond** with a harness
that is citable as NVIDIA’s public clone of Artificial Analysis (AA)
Intelligence Index GPQA methodology, so the number can be compared to AA’s
published GLM-5.3 (max) GPQA (ballpark **~91.7%**). Arms:

| Arm | Profile | Checkpoint |
|---|---|---|
| In-house | `benchmarks/configs/glm/glm-5.3-w4afp8-ours.sh` | `/mnt/cephfs/hoangduy/results/glm53-w4afp8-mtp/checkpoint` |
| PhalaCloud | `benchmarks/configs/glm/glm-5.3-w4afp8-phala.sh` | `PhalaCloud/GLM-5.3-W4AFP8` snapshot already used by the quality arm |

This is **not** a new GPQA implementation. Prompt, choice layout, extract
regex, 5-repeat pass@1, and Diamond split all come from NVIDIA task
`gpqa_diamond_aa_v3` in **`nvidia-simple-evals==26.3`** (PyPI). We only glue
that task onto the existing SGLang quality-arm serve and set GLM **(max)**
decode.

## Why this design

AA’s live methodology
([Intelligence Benchmarking](https://artificialanalysis.ai/methodology/intelligence-benchmarking/))
pins GPQA Diamond at 198 items, 5 repeats, 4-choice instruction prompt,
multi-stage regex extract, pass@1. AA cites OpenAI `simple-evals` for the
**dataset**, and independently specifies the **prompt and regex**.

NVIDIA already packaged that contract as Eval Factory simple-evals task
`gpqa_diamond_aa_v3`. The same task ships in two forms: the NGC container
`nvcr.io/nvidia/eval-factory/simple-evals:26.03` and the public wheel
`nvidia-simple-evals==26.3`. We use the **wheel**. Verified locally
2026-09-02 by downloading `nvidia_simple_evals-26.3-py3-none-any.whl` (latest
on PyPI) and reading `core_evals/simple_evals/framework.yml`: the named task
is present with `n_samples: 5`, the AA prompt template, and the full extract
regex list. The PyPI README is stale (it only lists `gpqa_diamond_aa_v2`);
the wheel is the source of truth.

NVIDIA’s pip path is documented:
[text-gen eval](https://docs.nvidia.com/nemo/evaluator/latest/evaluation/run-evals/text-gen.html)
(`pip install nemo-evaluator nvidia-simple-evals`, then `nemo-evaluator run_eval`).

A homemade `quality/sampling` GPQA runner, or lm-eval
`gpqa_diamond_cot_zeroshot`, would diverge from that contract while looking
green. The standing rule is: do not rebuild a reputable harness that already
exists.

**Rejected alternatives (locked):**

| Option | Why not |
|---|---|
| EleutherAI lm-eval `gpqa_diamond_cot_zeroshot` / `gpqa_diamond_zeroshot` | Wrong prompt, extract, and repeats. Already the general-suite GPQA; not AA. |
| NeMo-Skills `ns_gpqa` + `eval/aai/mcq-4choices` | NVIDIA’s *quant-eval* golden (Model-Optimizer uses **16** repeats). They switched *off* `gpqa_diamond_aa_v3` to match Nemotron-3-Ultra, and noted those runs are not comparable to the simple-evals baseline. Wrong goal. |
| Custom avg@k runner in `benchmarks/quality/sampling` | Reimplements shuffle, prompt, and regex. Forbidden. |
| NEL Slurm + launcher-deployed vLLM | This cluster already serves these checkpoints with proven SGLang W4AFP8 flags (`glm53_quality_arm.sh`). Do not stand up a second engine. |
| NGC sidecar `simple-evals:26.03` | Same task as the 26.3 wheel. Needs `nvcr.io` login / `imagePullSecret`. Unnecessary once the wheel was confirmed to contain `gpqa_diamond_aa_v3`. |

## Harness identity (PyPI, not NGC)

Pin:

```
nvidia-simple-evals==26.3
```

(`nemo-evaluator` comes in as a dependency.) Install into a **dedicated
isolated venv**, never into `eval-sglang-0.5.17` (that venv is pinned to
`lm-eval[api,ifeval]==0.4.10` and `datasets==5.0.0` for IFEval’s population
digest). Suggested PVC path, following the existing persistent-venv pattern:

```
/mnt/cephfs/hoangduy/venvs/nvidia-simple-evals-26.3
```

No `--system-site-packages` against the SGLang image: this client only speaks
HTTP to localhost.

Staging **must** fail closed unless:

```
nemo-evaluator ls
```

lists `gpqa_diamond_aa_v3`. Do not trust the README.

No NGC API key. Gated GPQA still needs the existing HuggingFace secret
`hd-hf-token` (`HF_TOKEN`), which is unrelated.

## Architecture

Keep the existing Rancher quality-arm layout: **one pod, one container,
sequential arms**, SGLang on `127.0.0.1:30000`, no Service
(`pipeline/k8s/glm53-quality-arm.yaml.tmpl`). After serve + existing
capability gates, the **same** arm script calls `nemo-evaluator` from the
26.3 venv against localhost. Official analogue: NeMo Evaluator
[bring-your-own-endpoint](https://docs.nvidia.com/nemo/evaluator/nightly/deployment/bring-your-own-endpoint/index.html)
(`deployment.type=none`). We do not run `nemo-evaluator-launcher` Slurm or
let NEL deploy vLLM.

```
┌─ pod glm53-qual-<arm>-<tag> ─────────────────────────────────┐
│  lmsysorg/sglang:v0.5.17                                     │
│    glm53_quality_arm.sh                                      │
│      SGLang  127.0.0.1:30000                                 │
│      then:                                                   │
│      /mnt/cephfs/hoangduy/venvs/nvidia-simple-evals-26.3     │
│        nemo-evaluator run_eval                               │
│          --eval_type gpqa_diamond_aa_v3                      │
│          --model_url http://127.0.0.1:30000/v1/chat/completions │
└──────────────────────────────────────────────────────────────┘
         ours to completion, then phala. Offline delta optional.
```

General-suite `GENERAL_TASKS` must **not** grow this GPQA. lm-eval GPQA and
AA-v3 GPQA are different measurements; mixing them in one `kind=general`
artifact would mislabel the score.

## Inherited contract (do not reimplement)

Source of truth: `gpqa_diamond_aa_v3` defaults in
`nvidia-simple-evals==26.3` `framework.yml` (wheel inspected 2026-09-02).

| Piece | Value |
|---|---|
| Split | GPQA Diamond, 198 questions (`task: gpqa_diamond`) |
| Repeats | `n_samples: 5` |
| Metric | pass@1 over items × repeats |
| Prompt | AA 0-shot MCQ: last line `Answer: A/B/C/D`, then `{Question}` and `A)…D)` |
| Extract | Primary `(?i)[*_]{0,2}Answer…([A-Z])`, then boxed / “answer is …” fallbacks as in NVIDIA `custom_config.extraction` |
| Dataset lineage | OpenAI simple-evals GPQA Diamond (AA’s cited source) |
| Choice shuffle | Inherited from simple-evals; not reimplemented. Canary must log one rendered prompt and confirm permutation across repeats. |

Honesty label on every report: **NVIDIA `nvidia-simple-evals==26.3`
`gpqa_diamond_aa_v3` (AA methodology clone), not Artificial Analysis’s
private runner.** Decode and endpoint still differ from AA’s lab because we
override temperature / max tokens as below; those overrides are documented,
not hidden.

## Overrides we own (decode only)

NVIDIA’s *task* defaults are greedy (`temperature: 0.0`, `top_p: 1e-5`,
`max_new_tokens: 16384`, `request_timeout: 60`). Those are wrong for GLM
**(max)** thinking. Override only sampling / budget / thinking — never prompt
or regex.

| Knob | Value | Source |
|---|---|---|
| temperature | `0.6` | AA reasoning convention; `benchmarks/performance/aa/variants.py` `AA_TEMPERATURE_REASONING` |
| top_p | `1.0` | `AA_TOP_P` |
| max_new_tokens | `65536` | profile `MAX_OUTPUT_REASONING` |
| request timeout | `3600` s | thinking traces; 60 s would truncate and score noise |
| thinking | ON | profile `THINK_ON_EXTRA_BODY` / chat-completions extra body `chat_template_kwargs.enable_thinking: true` |
| `limit_samples` | unset on formal run | full 198 |

Formal run: 198 × 5 = 990 completions per arm.

Canary (required before either formal arm): `limit_samples=2` (or harness
equivalent `--first_n 2`) against the live endpoint. Must prove: thinking
toggle present, a completion longer than a few tokens, extract not
systematically `[invalid]`, and one shuffled prompt logged.

## What we write (glue)

Implementation lives next to the existing arm, not inside
`benchmarks/quality/general`. No new container in the pod template.

1. **Venv** — `stage-glm53-quality-eval.sh` creates/verifies
   `/mnt/cephfs/hoangduy/venvs/nvidia-simple-evals-26.3` with
   `nvidia-simple-evals==26.3` (isolated, not the lm-eval venv). Gate:
   `nemo-evaluator ls` contains `gpqa_diamond_aa_v3`.
2. **Launch sequencing** — after SGLang is healthy and existing capability
   gates pass, `glm53_quality_arm.sh` runs `nemo-evaluator run_eval
   --eval_type gpqa_diamond_aa_v3 --model_type chat --model_url
   http://127.0.0.1:30000/v1/chat/completions` with the decode overrides
   (YAML or `--overrides`). Wait for `/v1/models` (or the arm’s existing
   ready probe) before the first request.
3. **Staging datasets** — pull Diamond into `EVAL_HF_ROOT` (private cache,
   not `/mnt/cephfs/.hf-cache`). Arms keep `HF_HUB_OFFLINE=1`. Fail closed
   if `HF_TOKEN` is missing when this GPQA is in scope.
4. **Harness manifest** — write package version, wheel hash if cheap,
   task name, `n_samples`, temperature, top_p, max tokens, timeout,
   thinking-control hash, dataset revision if the harness reports one.
   New file next to the arm’s lm-eval `harness_manifest.json`; do not
   overwrite that identity.
5. **Artifacts** — NVIDIA’s `results.yml` / score JSON under the arm result
   dir, e.g. `…/aa-gpqa-v3/`. Do **not** ingest it as `kind=general`.
   Optional later: a tiny adapter into the benchmarks contract as a new
   kind; not required for the first comparable number.

If a future pin of `nvidia-simple-evals` drops `gpqa_diamond_aa_v3`,
**stop**. Do not reconstruct the task from regex snippets. Do not silently
fall back to `gpqa_diamond_aa_v2`.

## Out of scope

- Full AA Intelligence Index (HLE, SciCode, AA-LCR, …)
- Claiming the score *is* an Artificial Analysis listing
- Changing general-suite GPQA, IFEval, or the lm-eval pin
- NGC images / NGC API keys / extra sidecar containers
- NEL MLflow auto-export
- Parallel ours+phala (cluster still has one 8-GPU node for this job)
- Killing or retuning SGLang flags to “look more like AA’s lab”

## Risks (handle, do not redesign)

| Risk | Handling |
|---|---|
| Stale PyPI README omits `aa_v3` | Staging gate on `nemo-evaluator ls`, not on the README. |
| Mixing into lm-eval venv (`datasets` pin clash) | Isolated 26.3 venv; refuse if `ls` is missing because the wrong python was used. |
| Gated `Idavidrein/gpqa` + offline | Private cache + `hd-hf-token`; fail closed on miss. |
| Client starts before SGLang is ready | Existing arm ready probe; then eval. |
| 60 s default timeout | Override to 3600. |
| Mixing scores with lm-eval GPQA | Separate artifact path and report heading. |
| Shuffle not actually happening | Canary inspects two repeats of the same item. |

## Success criteria

1. Canary (2 items) completes with thinking ON and non-empty extracts.
2. Each formal arm finishes 198 × 5 against that arm’s checkpoint.
3. Report names the harness `gpqa_diamond_aa_v3` /
   `nvidia-simple-evals==26.3` and lists decode overrides.
4. Ours and PhalaCloud use the same package pin, prompt/extract (untouched),
   n_samples=5, and decode overrides — only weights differ.
5. No new GPQA prompt or regex code lands in this repo.
6. No NGC credential or extra container is required.

## References

- AA methodology (GPQA prompt + extract):
  https://artificialanalysis.ai/methodology/intelligence-benchmarking/
- NVIDIA simple-evals task `gpqa_diamond_aa_v3` (catalog / container twin):
  https://docs.nvidia.com/nemo/evaluator/0.2.5/evaluation/benchmarks/catalog/all/harnesses/simple_evals.html
- PyPI wheel (what we run):
  https://pypi.org/project/nvidia-simple-evals/26.3/
- NVIDIA pip install path:
  https://docs.nvidia.com/nemo/evaluator/latest/evaluation/run-evals/text-gen.html
- NEL bring-your-own-endpoint:
  https://docs.nvidia.com/nemo/evaluator/nightly/deployment/bring-your-own-endpoint/index.html
- Existing serve: `llm-compressor/pipeline/k8s/glm53_quality_arm.sh`,
  `glm53-quality-arm.yaml.tmpl`, `stage-glm53-quality-eval.sh`
- AA decode constants: `benchmarks/performance/aa/variants.py`
