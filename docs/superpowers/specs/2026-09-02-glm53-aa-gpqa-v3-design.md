# GLM-5.3 W4AFP8 GPQA Diamond — NVIDIA AA v3 clone

**Date:** 2026-09-02

**Status:** Approved for implementation (harness locked 2026-09-02)

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
regex, 5-repeat pass@1, and Diamond split all come from NVIDIA Eval Factory
task `gpqa_diamond_aa_v3`. We only glue that task onto the existing SGLang
quality-arm serve and set GLM **(max)** decode.

## Why this design

AA’s live methodology
([Intelligence Benchmarking](https://artificialanalysis.ai/methodology/intelligence-benchmarking/))
pins GPQA Diamond at 198 items, 5 repeats, 4-choice instruction prompt,
multi-stage regex extract, pass@1. AA cites OpenAI `simple-evals` for the
**dataset**, and independently specifies the **prompt and regex**.

NVIDIA already packaged that contract as Eval Factory simple-evals task
`gpqa_diamond_aa_v3` in container `nvcr.io/nvidia/eval-factory/simple-evals:26.03`
([task catalog](https://docs.nvidia.com/nemo/evaluator/0.2.5/evaluation/benchmarks/catalog/all/harnesses/simple_evals.html)).
The published defaults match AA’s prompt template and extract regex list, and
set `n_samples: 5`.

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

## Glossary — NGC image

**NGC** is NVIDIA GPU Cloud, NVIDIA’s container registry at `nvcr.io` (same
role as Docker Hub). An **NGC image** is an OCI image pulled from there.

The harness image is:

```
nvcr.io/nvidia/eval-factory/simple-evals:26.03
```

It is the eval **client**, not the model server. The quality-arm pod already
runs `lmsysorg/sglang:v0.5.17` for serving. The NGC image only talks to that
server over localhost. Pulling from `nvcr.io` normally needs an NGC API key
(`docker login nvcr.io` / a Kubernetes `imagePullSecret`). That credential is
a staging prerequisite, like `hd-hf-token` for gated GPQA.

## Architecture

Keep the existing Rancher quality-arm layout: **one pod, sequential arms**,
SGLang on `127.0.0.1:30000`, no Service, nothing another tenant can reach
(`pipeline/k8s/glm53-quality-arm.yaml.tmpl`).

Add a **sidecar container in the same pod** running the NGC simple-evals
image. Same network namespace → `http://127.0.0.1:30000/v1/chat/completions`.
Official analogue: NeMo Evaluator
[bring-your-own-endpoint](https://docs.nvidia.com/nemo/evaluator/nightly/deployment/bring-your-own-endpoint/index.html)
(`deployment.type=none`). We do not run `nemo-evaluator-launcher` Slurm or
let NEL deploy vLLM.

```
┌─ pod glm53-qual-<arm>-<tag> ─────────────────────────────────┐
│  container arm:     lmsysorg/sglang:v0.5.17                  │
│                     glm53_quality_arm.sh serve + gates        │
│                     127.0.0.1:30000                          │
│                              │                               │
│  container gpqa-aa: nvcr.io/nvidia/eval-factory/             │
│                     simple-evals:26.03                       │
│                     simple_evals --eval_name                 │
│                       gpqa_diamond_aa_v3                     │
└──────────────────────────────────────────────────────────────┘
         ours to completion, then phala. Offline delta optional.
```

The lm-eval 0.4.10 client venv (`eval-sglang-0.5.17`) is **untouched**. It
remains the general-suite scorer. NVIDIA simple-evals must not be pip-installed
into that venv.

General-suite `GENERAL_TASKS` must **not** grow this GPQA. lm-eval GPQA and
AA-v3 GPQA are different measurements; mixing them in one `kind=general`
artifact would mislabel the score.

## Inherited contract (do not reimplement)

Source of truth: NVIDIA task defaults for `gpqa_diamond_aa_v3` in
simple-evals:26.03, which copy AA’s published GPQA block.

| Piece | Value |
|---|---|
| Split | GPQA Diamond, 198 questions (`task: gpqa_diamond`) |
| Repeats | `n_samples: 5` |
| Metric | pass@1 over items × repeats |
| Prompt | AA 0-shot MCQ: last line `Answer: A/B/C/D`, then `{Question}` and `A)…D)` |
| Extract | Primary `(?i)[*_]{0,2}Answer…([A-Z])`, then boxed / “answer is …” fallbacks as in AA and NVIDIA `custom_config.extraction` |
| Dataset lineage | OpenAI simple-evals GPQA Diamond (AA’s cited source) |
| Choice shuffle | Inherited from simple-evals; not reimplemented. Canary must log one rendered prompt and confirm permutation across repeats. |

Honesty label on every report: **NVIDIA Eval Factory `gpqa_diamond_aa_v3` (AA
methodology clone), not Artificial Analysis’s private runner.** Decode and
endpoint still differ from AA’s lab if we override temperature / max tokens
as below; those overrides are documented, not hidden.

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
| thinking | ON | profile `THINK_ON_EXTRA_BODY` / adapter `chat_template_kwargs.enable_thinking: true` |
| `limit_samples` | unset on formal run | full 198 |

Formal run: 198 × 5 = 990 completions per arm.

Canary (required before either formal arm): `limit_samples=2` (or harness
equivalent `--first_n 2`) against the live endpoint. Must prove: thinking
toggle present, a completion longer than a few tokens, extract not
systematically `[invalid]`, and one shuffled prompt logged.

## What we write (glue)

Implementation lives next to the existing arm, not inside
`benchmarks/quality/general`.

1. **Pod template** — second container in `glm53-quality-arm.yaml.tmpl`:
   image `nvcr.io/nvidia/eval-factory/simple-evals:26.03`,
   `imagePullSecret` for NGC, same cephfs mounts as the arm (private HF cache
   + result dir). No extra GPU request on the sidecar.
2. **Launch sequencing** — after SGLang is healthy and existing capability
   gates pass, run `simple_evals` against `http://127.0.0.1:30000/v1/chat/completions`
   with `--eval_name gpqa_diamond_aa_v3` and the decode overrides. Wait for
   `/v1/models` (or the arm’s existing ready probe) before the first request.
3. **Staging** — extend `stage-glm53-quality-eval.sh` to pull Diamond into
   `EVAL_HF_ROOT` (private cache, not `/mnt/cephfs/.hf-cache`). Arms keep
   `HF_HUB_OFFLINE=1`. Fail closed if `HF_TOKEN` is missing when GPQA is in
   scope (same pattern as gated lm-eval GPQA).
4. **Harness manifest** — write image digest, task name, `n_samples`,
   temperature, top_p, max tokens, timeout, thinking-control hash, dataset
   revision if the harness reports one, next to the arm’s existing
   `harness_manifest.json` (new file is fine; do not overwrite lm-eval
   identity).
5. **Artifacts** — NVIDIA’s score JSON / `results.yml` under the arm result
   dir, e.g. `…/aa-gpqa-v3/`. Do **not** ingest it as `kind=general`. Optional
   later: a tiny adapter into the benchmarks contract as a new kind; not
   required for the first comparable number.

Fallback if NGC pull is impossible: a **separate** venv (not
`eval-sglang-0.5.17`) pinned to a `nvidia-simple-evals` release that **lists**
`gpqa_diamond_aa_v3` (some 26.3 PyPI listings omit it). If no such pin exists,
**stop**. Do not reconstruct aa_v3 from regex snippets.

## Out of scope

- Full AA Intelligence Index (HLE, SciCode, AA-LCR, …)
- Claiming the score *is* an Artificial Analysis listing
- Changing general-suite GPQA, IFEval, or the lm-eval pin
- NEL MLflow auto-export
- Parallel ours+phala (cluster still has one 8-GPU node for this job)
- Killing or retuning SGLang flags to “look more like AA’s lab”

## Risks (handle, do not redesign)

| Risk | Handling |
|---|---|
| NGC pull blocked on the node | Staging gate. Fallback only if PyPI pin exposes `gpqa_diamond_aa_v3`. Else refuse. |
| Gated `Idavidrein/gpqa` + offline | Private cache + `hd-hf-token`; fail closed on miss. |
| Sidecar starts before SGLang is ready | Existing arm ready probe; sidecar waits. |
| 60 s default timeout | Override to 3600. |
| Mixing scores with lm-eval GPQA | Separate artifact path and report heading. |
| Shuffle not actually happening | Canary inspects two repeats of the same item. |

## Success criteria

1. Canary (2 items) completes with thinking ON and non-empty extracts.
2. Each formal arm finishes 198 × 5 against that arm’s checkpoint.
3. Report names the harness `gpqa_diamond_aa_v3` / simple-evals:26.03 and
   lists decode overrides.
4. Ours and PhalaCloud use the same task image digest, prompt/extract (untouched),
   n_samples=5, and decode overrides — only weights differ.
5. No new GPQA prompt or regex code lands in this repo.

## References

- AA methodology (GPQA prompt + extract):
  https://artificialanalysis.ai/methodology/intelligence-benchmarking/
- NVIDIA simple-evals task `gpqa_diamond_aa_v3`:
  https://docs.nvidia.com/nemo/evaluator/0.2.5/evaluation/benchmarks/catalog/all/harnesses/simple_evals.html
- NGC container:
  https://catalog.ngc.nvidia.com/orgs/nvidia/eval-factory/containers/simple-evals
- NEL bring-your-own-endpoint:
  https://docs.nvidia.com/nemo/evaluator/nightly/deployment/bring-your-own-endpoint/index.html
- Existing serve: `llm-compressor/pipeline/k8s/glm53_quality_arm.sh`,
  `glm53-quality-arm.yaml.tmpl`, `stage-glm53-quality-eval.sh`
- AA decode constants: `benchmarks/performance/aa/variants.py`
