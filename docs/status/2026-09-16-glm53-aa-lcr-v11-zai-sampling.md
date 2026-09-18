# Sep 16 – Sep 18

## Duy

### What this note is

**AA-faithful** GLM-5.3 W4AFP8 scores on AA-LCR v1.1: Z.ai's recommended
sampling (temperature 1.0 / top_p 0.95) at AA's max-output policy (131,072),
i.e. AA's own rule applied on both axes. Covers four runs:

1. the headline in-house **AWQ** result;
2. a raised-cap ablation that established the cap buys nothing;
3. the mislabelled 0.6 / 1.0 generic-sampling baseline; and
4. an in-house **GPTQ** arm on a byte-identical serve — a controlled
   quantization A/B (§GPTQ vs AWQ).

Supersedes the provenance labels in
[`2026-09-14-glm53-aa-lcr-v11.md`](2026-09-14-glm53-aa-lcr-v11.md) (see its
correction block). Public-methodology reproduction, not an official AA
leaderboard score.

### Key result

**AWQ pass@1: 79.00% (237 / 300).** AA's published GLM-5.3 figure is **80%** — a
**1.0 pp** gap, down from 8.7 pp on the mislabelled 0.6 baseline. **GPTQ scores
78.33% (235/300)**, statistically indistinguishable (paired McNemar p ≈ 0.87).

| Arm / config | pass@1 | Truncated | >131k | Gen tokens | Untrunc. acc |
|---|---:|---:|---:|---:|---:|
| **AWQ, 1.0/0.95 @131,072** | **237/300 = 79.00%** | 2 | 0 | 1,908,829 | 237/298 |
| **GPTQ, 1.0/0.95 @131,072** | **235/300 = 78.33%** | 1 | 0 | **1,560,828** | 235/299 |
| AWQ, 1.0/0.95 @262,144 | 233/300 = 77.67% | 2 | 2 | 2,060,660 | 233/298 |
| AWQ, 0.6/1.0 @131,072 | 214/300 = 71.33% | 32 | 0 | 5,078,829 | 213/268 |

Sampling moved the score by **7.67 pp**. Quantization method moved it by
**0.67 pp at p ≈ 0.87**. The knob that matters is sampling, not the quantizer.

Headline AWQ run: id `glm53-w4afp8-aa-lcr-v11-full-t1p95-r2`, fingerprint
`b027edecfa9eebb0d90b38a9b7fe078faae75d7ad66ea5c40c50c6f00d2c7508`, code
revision `5f090bd636753795a4d36d0a4733f200d1f95d4f`, 2 h 34 m wall clock.
GPTQ arm: id `glm53-gptq-aa-lcr-v11-full-t1p95-r1`, fingerprint
`20984ca6b7881bd739fce82e44904ed46589e71c6b39de8fb042d7e7c88a7ca7`, code
revision `5807665c25e75eb05481377388857de3b5b25a66`, 2 h 09 m wall clock.
Both on serve TP=16 / EAGLE 3-1-4 / fp8 KV / pool 598,848. Zero candidate
errors and zero judge failures on all four runs (GPTQ had 1 judge retry).

Question-macro accuracy is also 79.00%. Per-repeat: 76 / 79 / 82.

| Document category | 1.0/0.95 @131k | 0.6/1.0 @131k |
|---|---:|---:|
| Industry_Reports | 22/24 = 91.7% | 83.3% |
| Academia | 13/15 = 86.7% | 66.7% |
| Company_Documents | 152/189 = 80.4% | 71.4% |
| Marketing | 14/18 = 77.8% | 83.3% |
| Government_Consultations | 22/33 = 66.7% | 60.6% |
| Legal | 11/18 = 61.1% | 61.1% |
| Survey_Reports | 3/3 = 100% | 100% |

Legal is the standing weak spot — flat at 61.1% across both sampling configs, so
it is not a sampling artifact. Marketing moved down but n=18.

### GPTQ vs AWQ — a controlled A/B, and no detectable quality difference

The GPTQ arm ran on the *same endpoint* after the collaborator swapped the
checkpoint. Every serving variable was identical — Service, `tp_size` 16, 2
nodes, EAGLE 3-1-4, `fp8_e4m3` KV, `mem_fraction_static` 0.80,
`max_total_num_tokens` 598,848, SGLang 0.5.17 — as were sampling, cap,
concurrency, gate, dataset revision and judge contract. **Only the checkpoint
differed**, which makes this a genuinely controlled comparison rather than two
runs put side by side.

Because both arms cover the same 300 units, the arms are compared **paired**
rather than by each headline's binomial stderr:

| | |
|---|---:|
| Both correct | 218 |
| GPTQ correct, AWQ wrong | 17 |
| AWQ correct, GPTQ wrong | 19 |
| Neither | 46 |
| Discordant pairs | **36** |
| McNemar exact, two-sided | **p ≈ 0.87** |

**There is no detectable quality difference between the two quantizations on
this benchmark.** Note the shape: 36 of 300 units (12%) flip between arms while
the *net* difference is 2 units. Unseeded sampling churns far more items than
the quantizer does — which is also why a single unpaired run cannot resolve a
difference this size, and why future arms should be compared paired.

Use the paired test, not the headline delta, for any future arm comparison on
this benchmark.

### GPTQ spends fewer output tokens than AWQ, and the gap grows with generation length

On AA-LCR the GPTQ checkpoint generated **18.2% fewer** completion tokens than
AWQ for the same 300 units and the same prompts, and finished **25 minutes
sooner** at identical effective concurrency:

| | GPTQ | AWQ | Δ |
|---|---:|---:|---:|
| Total completion tokens | **1,560,828** | 1,908,829 | **−18.2%** |
| Mean completion tokens | 5,203 | 6,362 | −18.2% |
| Median completion tokens | 2,088 | 2,205 | −5.3% |
| Truncated at 131,072 | 1 | 2 | — |
| Reasoning share of completion | 97.9% | 98.3% | — |
| Prompt cache hit rate | 99.70% | 99.70% | — |
| Effective concurrency | 2.00 | 2.00 | — |
| p50 request latency | 23.9 s | 27.6 s | −13% |
| Wall clock | **2 h 09 m** | 2 h 34 m | −16% |

This is consistent with, and much larger than, the same effect measured on the
full7 suite in
[`2026-09-14-glm53-ep-gptq-w4afp8-and-full7.md`](2026-09-14-glm53-ep-gptq-w4afp8-and-full7.md)
(§Whole-suite tokens), where identical prompts and request counts gave:

| Arm | Generated tokens | vs AWQ |
|---|---:|---:|
| GPTQ | 5,112,329 | **−3.2%** |
| In-house AWQ | 5,281,637 | — |
| Phala | 5,490,247 | +4.0% |

**The effect scales with generation length.** full7 averages ~47 generated
tokens per request (5.11 M over 108,963 requests); AA-LCR averages ~5,200 — two
orders of magnitude longer — and the GPTQ advantage grows from 3.2% to 18.2%.
Short-generation and loglikelihood-scored tasks barely expose it; long-form
reasoning exposes it strongly. That is the regime where it matters commercially,
since output tokens dominate serving cost and latency.

Interpretation and caveats:

- **It is a verbosity difference, not a truncation artifact.** Both arms
  truncated ~1–2 of 300, and the median moved only −5.3% while the mean moved
  −18.2%, so the saving comes from a thinner long tail rather than uniformly
  shorter answers.
- **It costs nothing in quality here** (p ≈ 0.87), so on this benchmark GPTQ is
  the better operational choice: same score, 18% fewer output tokens, 16% less
  wall clock.
- **n=300, single unseeded draw per arm.** The direction now agrees across two
  independent instruments (full7 and AA-LCR), which is what makes it worth
  recording; the magnitude on any one benchmark should not be quoted as a
  constant.
- **Reasoning tokens are 98% of all completion tokens on both arms**, so this is
  a difference in *thinking* length, not answer length. `reasoning_effort` is
  unset on both serves (GLM-5.3 default `max`), so the comparison is fair, but
  that knob is the obvious lever if output cost needs reducing further.

### The raised cap bought nothing — settled

The 262,144 run was authorized because the 0.6 baseline truncated 32/300 at the
131,072 cap with empty answers. It did not help:

- Only **2 of 300** completions exceeded 131,072. Both ran to the full 262,144,
  truncated, and scored INCORRECT. Both also generated past Z.ai's own disclosed
  128K output maximum, i.e. outside what the creator claims to support.
- The same sampling at **131,072 scored higher** (79.00% vs 77.67%) on **7% fewer
  generated tokens**, with the identical 2/300 truncation count.
- The two runs truncated *different* items (262k: `q16 r0`, `q26 r1`; 131k:
  `q49 r1`, `q81 r0`), which is expected unseeded and confirms the 1.3 pp
  difference between them is sampling noise (±2.4 pp binomial stderr at n=300),
  not a cap effect.

The cap was never the binding constraint. Completion percentiles at Z.ai
sampling: **p50 2,214, p90 15,649, p99 49,895** — 99% of attempts finish under
50k tokens against a 131,072 cap.

### What actually moved the score

Sampling, and specifically the elimination of low-temperature runaway
generation. Where the completion-length mass sits:

| Completion length | 0.6/1.0 @131k | 1.0/0.95 @262k | 1.0/0.95 @131k |
|---|---:|---:|---:|
| ≤ 8k | 240 | 235 | 237 |
| 8k – 64k | 28 | 63 | 61 |
| 64k – 131,072 | **32** (all capped) | 0 | **2** (capped) |
| > 131,072 | 0 | 2 (both capped) | 0 |

At temperature 0.6 with an untruncated tail (top_p 1.0), 32 traces ran away into
the 64k–131k band and every one hit the wall with an empty answer. At 1.0 / 0.95
that band holds 2 and 0 respectively, and total generation is **62% lower**
(5.08M → 1.91M) at the same cap. The mass did not vanish — it moved from the
64k+ tail into the 8k–64k band (28 → 61), i.e. traces that used to ramble to the
cap now terminate in the tens of thousands.

Note that non-truncated accuracy barely moved (79.48% → 79.53%). On items that
finish either way the sampling change is not an improvement; the entire headline
gain is about not running away. **A long tail on this benchmark is a sampling
problem, not a budget problem.**

### Provenance — why the earlier labels were wrong

AA publishes a generic default sampling config (temperature 0.6 / top_p 1.0) and
**overrides it with the model creator's recommended config whenever the lab
publishes one**. Z.ai recommends 1.0 / 0.95 for GLM-5.3, so the lab-override
branch is AA-faithful and 0.6 / 1.0 is the ablation. The AA GPQA arm moved onto
this rule in `189e3ac7`; the AA-LCR runner was left on AA's generic default until
`dc617c55`, which is why the Sep-14 bundle calls 0.6 / 1.0 "AA's published
reasoning defaults" and the 262k bundle calls itself an ablation.

The 131,072 cap is likewise **not an AA-published figure**. AA requests the
maximum output the model creator allows; Z.ai discloses a 128K output maximum, so
the AA-policy budget is 131,072.

`summary.json` now states both derivations, so a bundle no longer leaves a reader
to infer them:

```json
"sampling_provenance":   "Z.ai (GLM-5.3 creator) recommended config, the lab-override branch of AA's sampling rule",
"max_tokens_provenance": "AA's max-output policy applied to Z.ai's disclosed 128K output maximum for GLM-5.3 (not an AA-published figure)"
```

The r2 bundle is the first to publish as
`"AA-LCR v1.1 public-methodology reproduction"` with a single limitation line.

### Caveats

- **Unseeded.** The request body carries only model, messages, max_tokens,
  temperature, top_p and chat_template_kwargs — no seed, on any run. Each run is
  an independent draw; ±2.4 pp stderr at n=300, p≈0.79. A seed would not buy
  determinism anyway under batch-varying kernels plus EAGLE.
- **Not official AA.** In-house endpoint, our runner, our reconstruction of AA's
  rule. The 1.0 pp gap to AA's 80% is within one stderr, so "matches AA" is the
  honest reading, not "still behind".
- **W4AFP8, not the BF16 reference.** This is the quantized checkpoint; no BF16
  AA-LCR run exists to separate quantization loss from reproduction error.

### Runner changes behind these runs

`5d41244e` → `dc617c55` → `5f090bd6`:

- `--candidate-max-tokens` added (default 131,072, pinned into the run
  contract). Editing the constant instead would have let a raised-cap run claim
  to be a public-methodology reproduction, since
  `uses_public_methodology_sampling` compares against that constant.
- `--candidate-concurrency` relaxed from a pinned 2 to a 1–2 **admission
  ceiling**, plus `_AdmissionGate`: holds at one in-flight attempt while any
  attempt has run past `--candidate-long-attempt-seconds` (default 900). AA-LCR
  prompts are 76,820–114,611 tokens, so two worst-case 262,144-cap requests need
  ~753,510 KV tokens against a 598,848-token pool; two 131,072-cap requests fit.
  SGLang 0.5.17 handles pool exhaustion with `retract_all`, which releases
  **every** request in the batch, so the gate exists to keep that from arising.
  It held at 1 exactly twice in the 262k run — matching its two long attempts —
  and zero times in the 131k run, which never went near the cap.
- Fail-closed budget check in `build_run_contract`: rejects a cap whose
  worst-case single trace cannot fit the serve's `max_total_num_tokens`, before
  any endpoint traffic. Prompt lengths there are cl100k counts; measured against
  the serve, GLM tokenizes ~1% higher (95,407 actual vs 94,494 cl100k), so treat
  it as a floor rather than a certificate. **Consequence for future runs: a
  131,072-cap AA-LCR run needs a pool ≥ ~247k tokens.** The two-node TP=16 serve
  has 598,848; a single-node TP=8 serve (~164,800) will be refused.
- Sampling defaults moved to Z.ai's recommended 1.0 / 0.95, with
  `AA_GENERIC_TEMPERATURE` / `AA_GENERIC_TOP_P` retained for the ablation branch.
- `-t1` pair repointed to fresh `…-r2` run ids: the r1 attempt died ~9 min in
  (16 answers, 284 retryable transport errors, 0 judgments) when the serve
  disappeared, and both code revision and endpoint identity have moved since, so
  its checkpoint is fingerprint-incompatible. Dead r1 checkpoint left in place.

### Artifacts

| | Path |
|---|---|
| Headline bundle (AWQ) | `/mnt/cephfs/hoangduy/results/glm53-aa-lcr-v11/glm53-w4afp8-aa-lcr-v11-full-t1p95-r2/` |
| GPTQ arm | `…/glm53-gptq-aa-lcr-v11-full-t1p95-r1/` |
| Cap ablation | `…/glm53-w4afp8-aa-lcr-v11-full-t1p95-2x-r1/` |
| Generic-sampling ablation | `…/glm53-w4afp8-aa-lcr-v11-full-r1/` |
| Checkpoints | `/mnt/cephfs/hoangduy/aa-lcr-v11-work/<run id>/run.sqlite` |
| Runbook | [`docs/runbooks/aa-lcr-v11.md`](../runbooks/aa-lcr-v11.md) |

Served checkpoints, from each bundle's
`run-manifest.json` → `run_contract.endpoint_deployment_identity.model_path`:

| Arm | Checkpoint |
|---|---|
| AWQ | `/mnt/cephfs/hoangduy/results/glm53-w4afp8-mtp/checkpoint` |
| GPTQ | `/mnt/cephfs/hoangduy/results/glm53-ep-gptq-w4afp8/full-ep8/20260912t183612z/output/304b8051cfb2b260b61ce0cbe330e02a98e73639-gptq-W4AFP8/20260912-184239/checkpoint` |

**Reporting hazard:** the GPTQ serve reuses `--served-model-name glm-5.3-w4afp8`
and `--quantization w4afp8`, so `summary.json` records
`candidate_model: "glm-5.3-w4afp8"` for *both* arms — the runner's served-model
identity check cannot distinguish them, and a fresh run would not catch a
wrong-checkpoint mix-up (only a resume would, via the fingerprint). The arms are
identified by run id and by `model_path` above. The GPTQ manifests therefore
assert `EXPECT_MODEL_PATH_SUBSTRING=gptq-W4AFP8` against `/get_server_info` and
exit 12 before generation if it does not match.

Bundle files each: `summary.json`, `report.md`, `candidates.jsonl`,
`judgments.jsonl`, `run-manifest.json`, `files.sha256`.

### Operational note worth fixing

Both full Jobs' stdout carried only the admission-gate line. The manifests use
`exec > >(tee -a "$LOG") 2>&1`, and the container can exit before that process
substitution flushes, so the completion JSON was lost from both `kubectl logs`
**and** the CephFS runlog, on a Job that exited 0. Progress and results had to be
read from `run.sqlite` and the published bundle instead. Fix with a `wait` on the
tee PID before the script exits if the runlog is ever meant to be authoritative.

Related: a log-tail monitor is the wrong instrument for these Jobs for the same
reason — poll the Job's conditions instead, so silence means "still running"
rather than "buffered or crashed".
