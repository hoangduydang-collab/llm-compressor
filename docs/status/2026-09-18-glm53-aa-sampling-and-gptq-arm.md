# Sep 14 - Sep 18

## Duy

### What I worked on

- **Found and fixed a wrong sampling config in our AA reproductions.** We were
  using AA's *generic* default (temperature 0.6 / top_p 1.0) where AA's own rule
  says to use the model creator's recommended config (Z.ai: 1.0 / 0.95). Rerun
  under the correct branch, **both** AA benchmarks jumped ~10 points and landed
  within ~1 point of AA's published GLM-5.3 figures.
- **The third arm now has numbers.** Native expert-parallel GPTQ → W4AFP8
  finished, qualified, and has been benchmarked. Last week's tables were
  two-arm (ours vs PhalaCloud); GPTQ was still a quantization job.
- **Three benchmarks this week** instead of one: AA-LCR v1.1, AA GPQA Diamond,
  and the cheap full7 suite — the last one reran three-way. AA-LCR runs on
  **Zhou Yu's two nodes** (TP=16), which is what makes its 131k output cap
  possible at all; its GPTQ arm is in flight.
- Weekly detail lives in three notes, merged here:
  [AA-LCR](2026-09-16-glm53-aa-lcr-v11-zai-sampling.md),
  [EP GPTQ + first full7](2026-09-14-glm53-ep-gptq-w4afp8-and-full7.md),
  [full7 rerun](2026-09-17-glm53-full7-zai-sampling.md).

### Key result 1, the sampling config was costing us ~10 points on both AA benchmarks

**What was wrong.** AA publishes a generic default of temperature 0.6 / top_p
1.0 **and overrides it with the model creator's recommended config whenever the
lab publishes one.** Z.ai recommends 1.0 / 0.95 for GLM-5.3, so for this model
the lab-override branch is the AA-faithful setting and **0.6 / 1.0 is the
ablation.** We had it backwards on both benchmarks. Last week's GPQA note and the
first AA-LCR note both call 0.6 / 1.0 "the public methodology"; that label is
wrong, though the measurements themselves stand.

**Rerun under the correct branch.**

| Benchmark | Was (0.6 / 1.0) | Now (1.0 / 0.95) | AA published | Gap to AA |
|---|---:|---:|---:|---:|
| **GPQA Diamond**, ours | 81.82% | **91.52%** ±0.87 | ~91.7% | 9.9 pp → **0.2 pp** |
| **GPQA Diamond**, PhalaCloud | 79.39% | **91.11%** ±0.85 | ~91.7% | 12.3 pp → **0.6 pp** |
| **AA-LCR v1.1**, ours | 71.33% | **79.00%** | 80% | 8.7 pp → **1.0 pp** |

GPQA is the formal AA protocol: 198 Diamond items × 5 repeats = 990 completions
per arm, every response HTTP 200. AA-LCR is 100 questions × 3 repeats = 300.
Both are **public-methodology reproductions, not official AA runs** — in-house
endpoint, our runner, our reconstruction of AA's rule.

AA-LCR runs against **Zhou Yu's two nodes** as a standing TP=16 serve
(`gpu02`+`gpu03`, pool 598,848), because a 131,072-cap trace does not fit a
single-node pool. The GPTQ arm is on it now; PhalaCloud is the one cell still
missing. GPQA, by contrast, is one node per arm.

**Same mechanism on both benchmarks, and it is not a budget problem.** Low
temperature with an untruncated tail (top_p 1.0) makes a minority of traces run
away until they hit the output cap and return an empty answer. Correct sampling
removes the runaway band; it does not make the model smarter on items that
finish either way.

| GPQA, per arm | ours 0.6/1.0 | ours 1.0/0.95 | phala 0.6/1.0 | phala 1.0/0.95 |
|---|---:|---:|---:|---:|
| Hit the 131,072 cap | 120 / 990 (12.1%) | **4 / 990 (0.4%)** | 155 / 990 (15.7%) | **14 / 992 (1.4%)** |
| Avg completion tokens | 26,463 | **14,757** | 30,745 | **16,332** |
| Wall clock, one 8×H100 | 37.1 h | **11.2 h** | 46.2 h | **14.3 h** |

Last week's note guessed that "truncation is probably most of the gap to AA's
91.7%" and computed a 87.9% ceiling if every length-finish scored zero. That
hypothesis is now confirmed, with the cause identified: the truncation was a
**sampling artifact**, not too small a budget. On AA-LCR the same pattern —
32/300 cap hits at 0.6/1.0, 2/300 at 1.0/0.95, with non-truncated accuracy
essentially unchanged (79.48% → 79.53%). The entire headline gain on both
benchmarks is about not running away.

**The correct config is also ~3x cheaper.** GPQA went from 83 h of node time for
the pair to ~25 h, and AA-LCR's generated tokens fell 62% (5.08M → 1.91M) at the
same cap. We were paying triple to score ten points lower.

**Raising the cap is settled as the wrong lever.** A 262,144 AA-LCR rerun was
authorized because of those 32 truncations. It converted **zero** items — only 2
of 300 completions exceeded 131,072, both ran to the full cap, both scored
incorrect — and it scored *lower* than 131,072 (77.67% vs 79.00%) on 7% more
tokens. At correct sampling, 99% of AA-LCR attempts finish under 50k tokens
against a 131,072 cap. **Do not raise the cap.**

### Key result 2, the GPTQ arm

**The checkpoint.** Native expert-parallel GPTQ → W4AFP8, served as written with
no AWQ `to_sglang` conversion. Full EP8 job **15 h 04 m**, exit 0, 371 GiB, 8
main shards + MTP. Representative qualification passed first: EP4 and EP8 produce
**identical greedy outputs, 100% top-1 over 24 positions**, with EP8 peak memory
at **51.8% of EP4** — the memory claim that was still unproven last week.

Against in-house AWQ (~20.5 h quantize + ~3.5 h conversion/repatch/graft) this
path skips `to_sglang`, indexer repatch and standalone MTP graft entirely. The
extra ~4 h over the W4A16 predecessor is native compression + shard write +
CT-layout restore + hashing, not a slower GPTQ walk.

**Where GPTQ stands on quality.** Two instruments, and they disagree in an
informative way.

*Cheap full7 suite, greedy, with GPQA (Sep 13):*

| Task | GPTQ | in-house AWQ | Phala | vs AWQ |
|---|---:|---:|---:|---:|
| GSM8K | 97.27% | 97.65% | 97.19% | −0.38 |
| IFEval | 90.39% | 89.65% | 90.76% | +0.74 |
| **GPQA Diamond CoT** (flexible) | **67.68%** | 61.11% | 55.56% | **+6.57** |
| MMLU | 86.84% | 86.67% | 86.66% | +0.17 |
| ARC Challenge | 68.94% | 68.77% | 69.80% | +0.17 |
| HellaSwag | 89.00% | 89.37% | 89.29% | −0.37 |
| TruthfulQA MC2 | 61.92% | 62.99% | 62.50% | −1.07 |

GPQA is the only task that separated the arms by more than the 2 pp diagnostic
threshold, and GPTQ won it by **+6.57 over AWQ and +12.12 over Phala**. GPTQ also
generated 3.2% fewer tokens than AWQ and 6.9% fewer than Phala.

*The AA GPQA arm for GPTQ is still running* (started Sep 16, holding gpu04). That
is the number that matters for the three-way verdict, since AA GPQA is where the
arms actually separate and where we have a published reference.

**Do not read a GPTQ verdict off the full7 rerun.** It excluded GPQA (the node
was occupied), which removes the only task that has ever distinguished the arms.
GPTQ vs AWQ there is +0.22 GSM8K, +0.21 MMLU, −1.85 IFEval, −1.11 TruthfulQA —
nothing crossing 2 pp, and IFEval unresolved at n=541 (±1.35).

### The full7 rerun moved nothing, and that is the useful finding

All three arms, 6 tasks (GPQA excluded), 32,768 cap, CTX=65536, one node each,
**7 h 59 m** total. Every gate passed on every arm; all exited 0.

Every per-arm delta against its own greedy run sits inside that task's stderr —
largest are IFEval ours +1.11 and GPTQ −1.48, pointing in *opposite* directions,
which is what noise looks like.

**The ceiling is structural.** Only GSM8K and IFEval are generative. MMLU, ARC,
HellaSwag and TruthfulQA are `output_type: multiple_choice`, scored as
teacher-forced loglikelihood, and lm-eval ignores `gen_kwargs` on that path — so
**4 of 6 tasks cannot respond to sampling at all.** "full7 at Z.ai sampling" is
really a two-task experiment, and full7 generates short, so there is no runaway
tail for the fix to remove. **Conclusion: do not spend a node re-running full7
for sampling reasons.** Ask sampling questions on long-generation benchmarks.

**It did close a provenance defect.** The general suite had **no decoding channel
at all**: profiles declared `REASONING_TEMP=0`, which only ever drove the
reliability/sampling/long-context runners, and the greedy behaviour of every
published full7 number came from gsm8k's own yaml (`do_sample: false`) by
*inheritance*. The profile stated a temperature that described nothing about the
requests sent. `GENERAL_TEMPERATURE` / `GENERAL_TOP_P` now exist, are opt-in,
refuse invalid values rather than laundering them, and are recorded in the
published payload.

### Two gates that earned their keep

- **`tokenizer.json` parity blocked the GPTQ arm on a false positive.** The GPTQ
  checkpoint's tokenizer carries a `truncation: max_length 2048` block — a
  fingerprint of its 256×2048 calibration — where the other two have null. This
  was worth checking rather than waiving: the loglikelihood path tokenizes
  **client-side**, so a tokenizer that really truncated at 2048 would have
  silently cut MMLU's 5-shot prompts on one arm only. Measured: `transformers`
  resets backend truncation per call, so a 6,401-token string encodes to
  **identical ids** on both, and vocab/merges/processors/added-tokens are
  byte-equal across all three. Parity is now semantic for that file; anything
  touching vocabulary still blocks.
- **A fail-closed budget check refuses an AA-LCR run that cannot fit.** A
  131,072-cap AA-LCR trace needs a KV pool ≥ ~247k tokens (largest prompt
  114,611 + cap). The two-node TP=16 serve has 598,848; a single-node TP=8 serve
  (~164,800) is refused before any endpoint traffic.

### What is blocking me: GPU resources

Still the binding constraint, and tighter than last week.

- **The single-node pool is full.** 6 of 72 GPUs free, **zero fully-free nodes**,
  largest schedulable pod 4 GPUs. `gpu07` was claimed by another namespace within
  minutes of our last full7 arm releasing it. This is what constrains full7 and
  the AA GPQA arms, which are one-node-per-arm.
- **AA-LCR is not capacity-blocked** — it runs on Zhou Yu's two nodes
  (`gpu02`+`gpu03`) as a standing TP=16 serve, which is the only place a
  131,072-cap trace fits (pool 598,848 vs the ~247k a worst-case trace needs; a
  single-node TP=8 pool of ~164,800 is refused by the budget check). Its
  constraint is **serial, not scarce**: the serve hosts one checkpoint at a time,
  so each arm means re-pointing it and reloading. ~2 h 34 m per arm plus load.
- **Queue the whole campaign up front.** Applying all three full7 arms at once,
  pinned to one node and each requesting all 8 GPUs, made Kubernetes serialize
  them with **3-second** gaps and left no window for another tenant. Worth doing
  by default on a contended cluster.
- Quantization still needs a predictable 8-GPU allocation of its own, not one
  shared with eval work.

### Plan for next week

**Finish the AA set at the correct config.**

- **GPTQ's AA GPQA arm** — in flight on `gpu04`; it is the number the three-way
  verdict rests on.
- **GPTQ's AA-LCR arm** — in flight on Zhou Yu's two nodes as of Sep 18
  (`hd-aa-lcr-v11-full-gptq`, canary already Complete). Due ~1 h out.
- **AA-LCR for PhalaCloud** — the last missing cell. Re-point the two-node serve
  at the PhalaCloud snapshot once the GPTQ arm publishes.
- **HLE text-only** (2,158 questions) — carried over, not started. Still needs
  the AA equality-checker judge wired up, the one dependency outside our cluster.

**Cheap and worth doing.**

- **Re-add GPQA to full7** for the three-way — it is the only task that ever
  separated the arms by more than 2 pp. Needs `HF_TOKEN` in the arm secret *and*
  the `Idavidrein/gpqa` licence accepted on that account.
- **A greedy 6-task full7 control** (`GENERAL_TEMPERATURE=0`, same tag) would
  make one loose end a one-variable question: the generated-token ordering
  reversed between the greedy and sampled runs (GPTQ leanest before, loudest
  after), but sampling and the GPQA removal changed together, so it is currently
  unattributed.

**Still open on the quantization side.** Direct native export (removes the second
whole-checkpoint write), expert activation-scale alignment (the MoE path wants
static per-group scales; our preset is dynamic per-token), and the fact that no
BF16 GLM-5.3 fits an 8×H100 node — so a defect shared by all three arms stays
invisible on every benchmark above.
