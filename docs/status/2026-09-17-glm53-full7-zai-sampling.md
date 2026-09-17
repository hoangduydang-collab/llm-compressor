# Sep 17

## Duy

### What this note is

The three GLM-5.3 W4AFP8 quality arms — in-house AWQ, native EP-GPTQ, and
PhalaCloud — rerun on the cheap full7 instrument under **Z.ai's recommended
sampling (temperature 1.0 / top_p 0.95)**, with GPQA excluded and a 32,768
output cap. Run tag `full7t1p95-20260917t1855z`.

It answers one question: does putting this suite on the model creator's
recommended decoding change the verdict? **It does not.** The sampling change
moves every task inside its own stderr, and it structurally cannot move four of
the six tasks at all.

This also closes a provenance defect. Until now the general suite had **no
decoding channel**: the profiles declared `REASONING_TEMP=0`, that variable only
ever drove the reliability/sampling/long-context runners, and the greedy
behaviour of the published full7 numbers came from gsm8k's own yaml
(`do_sample: false, temperature: 0.0`) by *inheritance*. The profile stated a
temperature that described nothing about the requests actually sent.

Not an AA protocol and not a public-methodology number. Paired candidate-vs-peer
only. For the AA-faithful GLM-5.3 figure under this same sampling see
[the AA-LCR v1.1 note](2026-09-16-glm53-aa-lcr-v11-zai-sampling.md) — 79.00%
at a 131,072 cap. Do not mix the two instruments.

### Contract

| Item | Value |
|---|---|
| Run tag / RUN_ID | `full7t1p95-20260917t1855z` / `glm53-full7t1p95-20260917t1855z` |
| Arms | `ours` (in-house AWQ), `gptq` (native EP-GPTQ), `phala` (PhalaCloud) |
| Tasks | `gsm8k ifeval mmlu arc_challenge hellaswag truthfulqa_mc2` — full populations, seed 0 |
| GPQA | **excluded** (concurrently occupied on another node); see the token-count note below |
| Decode | **temperature 1.0 / top_p 0.95**, `GENERAL_MAX_GEN_TOKS=32768`, thinking ON, concurrency 16 |
| Loglikelihood | chat template **off** (`GENERAL_CHAT_TEMPLATE_ON_LOGLIKELIHOOD=0`) |
| Serve | SGLang 0.5.17, TP8, w4afp8, FP8 KV, **CTX=65536**, mem_frac 0.75, chunked prefill 2048, no speculative decoding, shared-expert fusion off |
| Node | one 8×H100 (`ca-gpu07`), all three arms sequentially |
| Eval code | `benchmarks` `9c6b6162b19df5573109cb3012175495713b2533` |
| Arm runner | `llm-compressor` `93796710505c24dc881346899d2b64a837bfdfd3` |
| Run mode | `diagnostic`, `formal: false`, `gating_eligible: false`, population scope `full` |

`CTX=65536` is pinned explicitly rather than taking the arm default of 164800
(which is sized for AA GPQA), because 65536 is what the three historical full7
runs used.

### Wall clock

Times UTC. Every arm exited 0.

| Arm | Start | Finish | Wall | Weight load → healthy | Suite |
|---|---|---|---|---|---|
| `ours` | 2026-09-16 18:56:14 | 21:31:58 | **2 h 35 m 44 s** | 2,260 s (~37 m) | 1 h 57 m 27 s |
| `gptq` | 21:32:01 | 2026-09-17 00:18:35 | **2 h 46 m 34 s** | 2,610 s (~43 m) | 2 h 01 m 27 s |
| `phala` | 00:18:38 | 02:55:14 | **2 h 36 m 36 s** | 2,230 s (~37 m) | 1 h 57 m 53 s |
| **Campaign** | **18:56:14** | **02:55:14** | **7 h 59 m 00 s** | | |

All three arms were applied to the cluster **up front**, pinned to the same node
and each requesting all 8 GPUs, so Kubernetes serialized them: the gaps between
arms are **3 seconds**. That is the reason to queue rather than launch by hand —
`ca-gpu07` was claimed by another namespace's pod within minutes of the last arm
releasing it, and any manual gap would have lost the node mid-campaign.

Dropping GPQA is most of why an arm is ~2 h 40 m here against the Sep-13 GPTQ
arm's 4 h 23 m.

### Gates

All fail-closed gates passed on every arm:
`ifeval_preflight`, `preflight`, `serve_healthy`, `throughput`, `ll_shape`,
`general_suite` = 0, plus `gptq_source_format=0` on the GPTQ arm.
Decode throughput probed **512 tok/s** aggregate on all three (floor 250).
`think_marker_leaked: False` everywhere — the `glm45` parser stripped the CoT, so
the scorer never saw reasoning text as an answer. Zero aborted requests.

### Scores

| Task | ours | gptq | phala | greedy ours/gptq/phala |
|---|---:|---:|---:|---|
| GSM8K exact (strict) | 97.35 ±0.44 | **97.57** ±0.42 | 96.82 ±0.48 | 97.65 / 97.27 / 97.19 |
| IFEval strict prompt | **90.76** ±1.25 | 88.91 ±1.35 | 90.20 ±1.28 | 89.65 / 90.39 / 90.76 |
| MMLU acc | 86.63 ±0.28 | **86.84** ±0.27 | 86.81 ±0.27 | 86.67 / 86.84 / 86.66 |
| ARC Challenge acc_norm | 69.37 ±1.35 | 68.86 ±1.35 | **70.31** ±1.34 | 68.77 / 68.94 / 69.80 |
| HellaSwag acc_norm | 89.20 ±0.31 | 88.97 ±0.31 | **89.35** ±0.31 | 89.37 / 89.00 / 89.29 |
| TruthfulQA MC2 | **62.88** ±1.46 | 61.77 ±1.45 | 62.54 ±1.46 | 62.99 / 61.92 / 62.50 |

Greedy references: `ours` `full7-20260901t064327z`, `phala`
`full7-20260831t135418z`, `gptq` `gptq-full7-20260913t1448z`.

### The sampling change bought nothing

Delta of each arm against **its own** greedy run, in percentage points:

| Task | ours | gptq | phala |
|---|---:|---:|---:|
| gsm8k | −0.30 | +0.30 | −0.37 |
| ifeval | +1.11 | −1.48 | −0.56 |
| mmlu | −0.04 | −0.00 | +0.15 |
| arc_challenge | +0.60 | −0.08 | +0.51 |
| hellaswag | −0.17 | −0.03 | +0.06 |
| truthfulqa_mc2 | −0.11 | −0.15 | +0.04 |

Every move is inside that task's stderr (ifeval ±1.3, arc ±1.35). The two
largest, ifeval `ours` +1.11 and `gptq` −1.48, point in *opposite* directions,
which is what noise looks like rather than an effect.

**The ceiling is structural, not bad luck.** Only **gsm8k and ifeval are
generative**. MMLU, ARC, HellaSwag and TruthfulQA are `output_type:
multiple_choice`, scored as teacher-forced loglikelihood, and lm-eval ignores
`gen_kwargs` on that path — the published payload says so itself
(`generation.applies_to: "generative path only"`). So **4 of 6 tasks cannot
respond to sampling by construction**, and a "full7 at Z.ai sampling" headline
is really a two-task experiment.

Their residual drift is therefore *not* a sampling effect: ≤0.15 pp on five of
the six MC cells, with one outlier at `ours` ARC +0.60 pp — 7 items out of
1,172. That is FP8-KV and batching changing logprob reduction order on near
ties, the ordinary non-determinism of this serve, not a decoding change.

**Contrast AA-LCR, where the identical sampling change was worth +7.67 pp**
(71.33% → 79.00%). There it worked by killing traces that ran away into the
64k–131k band and hit the cap with an empty answer. full7 generates short, so
there is no runaway tail to fix. **Do not spend a node re-running full7 to
change sampling; the instrument cannot resolve it.** Ask sampling questions on a
long-generation benchmark.

### Arm vs arm — the comparison this run actually licenses

All three arms decoded identically (`effective_gen_kwargs` is byte-identical in
all three payloads) and saw identical prompts, so the arm-to-arm read is valid
even though each arm's absolute generative scores are single stochastic draws.

GPTQ vs in-house AWQ: **+0.22** gsm8k, **+0.21** mmlu, **−1.85** ifeval,
**−1.11** truthfulqa, **−0.51** arc, **−0.23** hellaswag.

Nothing crosses the spec's 2 pp diagnostic threshold, and the largest (ifeval
−1.85) is not resolved at n=541 with ±1.35 stderr. **This table understates the
GPTQ case**: its one unambiguous win in the greedy campaign was GPQA Diamond CoT
(67.68% vs 61.11% AWQ, 55.56% Phala, **+6.57 / +12.12**), and GPQA is excluded
here. A three-way verdict on GPTQ should not be drawn from this run alone.

### Token spend

| Arm | Prompt | Generated | Cached | Requests | Aborted |
|---|---:|---:|---:|---:|---:|
| ours | 12,297,842 | 2,600,315 | 2,688 | 108,765 | 0 |
| gptq | 12,297,842 | **2,809,756** | 2,432 | 108,765 | 0 |
| phala | 12,297,842 | 2,466,975 | 2,496 | 108,765 | 0 |

Prompt tokens and request counts are **identical to the token** across all three
arms. That is the pairing evidence: same population, same prompts, same request
shape — only the checkpoint and the sampled continuations differ.

Against the greedy campaign, requests fell 108,963 → 108,765, a difference of
**exactly 198** — GPQA Diamond's population. A clean confirmation that task
selection is the only population change.

**One unresolved observation.** The generated-token ordering *reversed*. Greedy:
GPTQ 5,112,329 < AWQ 5,281,637 < Phala 5,490,247 (GPTQ leanest). Here: Phala <
AWQ < GPTQ, with GPTQ **+8.1%** over AWQ and **+13.9%** over Phala (GPTQ
loudest). **This is not attributable**, because two variables changed at once:
the sampling *and* the removal of GPQA, which was both GPTQ's strongest task and
a long-CoT one. Cross-run totals are not comparable (5.1–5.5 M vs 2.5–2.8 M)
for the same reason. Isolating it needs a greedy 6-task control, which is now a
one-variable run: `GENERAL_TEMPERATURE=0` on the same tag.

### Harness changes behind this run

Two, both in support of the run rather than incidental.

**1. `GENERAL_TEMPERATURE` / `GENERAL_TOP_P`** (`benchmarks` `9c6b616`). The
general suite gained its first decoding channel. `top_p` is plumbed alongside
temperature rather than after it because a creator's recommended config is a
*pair*; sending half of one evaluates a configuration nobody specified. The
existing request-mutation channel (`THINK_*_EXTRA_BODY` → `extra_gen_kwargs`)
could not carry it: `validate_think_body` is a strict allowlist of reasoning
toggles, and widening it would let arbitrary keys onto the `lm_eval` argv, which
is what that schema exists to prevent. Both knobs are opt-in and absent by
default, so a profile declaring neither produces byte-identical argv; a
present-but-invalid value is **refused**, not laundered to `None`; and `0` stays
`0` rather than collapsing to "absent", so a greedy control run states
temperature 0 in its record instead of silently inheriting task pins.

Confirmed in the pinned scorer, not assumed: lm-eval 0.4.10's
`LocalChatCompletion._create_payload` does `gen_kwargs.pop("do_sample", False)`
— it **discards** `do_sample`. gsm8k's pinned `do_sample: false` therefore never
reaches SGLang and cannot defeat `temperature: 1.0`. `temperature` is popped and
sent; `top_p` and `chat_template_kwargs` reach the engine through `**gen_kwargs`.

**2. `tokenizer.json` parity is now semantic, not a digest**
(`llm-compressor` `9379671`). The template-parity gate blocked the GPTQ arm, and
the block was a false positive. That checkpoint's `tokenizer.json` carries a
truncation block — `{'direction': 'Right', 'max_length': 2048, 'strategy':
'LongestFirst', 'stride': 0}` — where ours and PhalaCloud's have `null`. The
2048 is a fingerprint of the GPTQ calibration run (`ultrachat_200k`, 256 ×
2048): the tokenizer was serialized with calibration's truncation state still
enabled.

This mattered rather than being cosmetic, because the loglikelihood path
tokenizes **client-side**: a tokenizer that really truncated at 2048 would
silently cut MMLU's 5-shot prompts on one arm only and quietly invalidate the
comparison. Measured instead of assumed — `transformers`'
`PreTrainedTokenizerFast` resets backend truncation per call
(`set_truncation_and_padding` → `no_truncation()` when truncation is not
requested), so `backend_tokenizer.truncation` loads as `None` for **both**, and a
6,401-token string encodes to **identical ids** through both `encode()` and
`__call__()`. Everything that decides tokenization is byte-equal across all
three arms: vocab (154,820), merges (321,649), normalizer, pre_tokenizer,
post_processor, decoder, and all 36 added_tokens. It also explains why the
Sep-13 GPTQ arm scored sanely.

So `tokenizer.json` now gets the treatment `tokenizer_config.json` already had,
for the reason that file's own comment gives — blocking on a raw digest fails
comparisons that are in fact sound. `truncation`/`padding` are inert; everything
else still blocks; `chat_template.jinja` remains a raw-digest blocker because it
*is* the prompt text.

Both changes are covered by tests that execute the logic rather than grep for
it: 13 cases for the sampling channel, 5 for the parity classifier (a
truncation-only delta passes, while vocab, normalizer and chat-template deltas
each still fail the gate).

### Two operational notes worth keeping

**The staged tree must be normalised to LF.** The `profile_line_endings` gate
failed the first two staging attempts. `core.autocrlf=true` on the Windows
checkout means git stores LF but writes CRLF to the worktree, and `git archive`
did not help. bash keeps the CR *inside every value*, so a CRLF profile yields
`GENERAL_TOP_P=$'0.95\r'`. 47 shell scripts needed `sed -i 's/\r$//'` after
extraction. The gate earned its keep.

**A CPU-only pod still needs the GPU toleration.** The cephfs PVC is
region-locked to the GPU nodes (`ca-van3`), and the untainted CPU nodes fail the
PersistentVolume node affinity — so a preflight pod that requests zero GPUs
sits `Pending` forever without
`tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]`.
Same reason the AA-LCR Jobs carry it.

### Artifacts

| | Path |
|---|---|
| Results root | `/mnt/cephfs/hoangduy/results/glm53-quality-paired/full7t1p95-20260917t1855z/` |
| Per-arm client dirs | `client-{ours,gptq,phala}/` — `client.log`, `gates.txt`, `serve.log`, `harness_manifest.json`, `metrics-{before,after}.txt` |
| Result JSON | `results/glm-5.3-w4afp8-{ours,gptq,phala}/sglang/quality/general.glm53-full7t1p95-20260917t1855z.json` |
| Staging gates | `/mnt/cephfs/hoangduy/results/glm53-quality-paired/stage-gates.txt`, `template-parity.json`, `tok-digests.txt` |
| Durable run logs | `/mnt/cephfs/hoangduy/runlogs/glm53-qual-{ours,gptq,phala}-full7t1p95-20260917t1855z.log` |

### What to do next

1. **Do not rerun full7 for sampling reasons.** The conclusion is settled.
2. **Re-add GPQA** if a three-way GPTQ verdict is wanted — it is the only task
   that separated the arms by more than 2 pp. Needs `HF_TOKEN` in the arm's
   secret *and* the `Idavidrein/gpqa` licence accepted on that account.
3. **A greedy 6-task control** (`GENERAL_TEMPERATURE=0`, same tag) would make the
   generated-token reversal a one-variable question. ~8 h on one node.
4. **AA-LCR for `phala` and `gptq`** is still outstanding; only the in-house arm
   has a figure. It needs 16 GPUs (TP=16) — one worst-case trace is ~245,683 KV
   tokens against a TP=8 pool of 164,800, so concurrency 1 alone does not make
   it fit on one node.
