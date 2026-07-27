# M3 EAGLE3 speculative decoding on our W4AFP8 arm — design packet

**Status:** WAVE 1 **EXECUTED** 2026-07-27 (user-signed: "agree with the experiment
you suggested" + "just try the AA-style sweep first"). Window
`m3-specdec-eagle3/20260727T061506Z` — 4 arms, all rc=0, all gates passed.
Results: `docs/m3-specdec-eagle3.md`. **Verdict: k=3 gives 1.72–1.75× output speed
at conc 1 (not the 2.5–3.5× hoped for) and also raises conc-10 throughput
1.64–1.75×; adopt k=3 for the latency tier, but conc 32/64 is unmeasured so do not
enable globally.** Wave 2 (natural prompts, conc 32/64, suite reasoning path) is
proposed at the end of the results doc and not yet signed off.
Owner: full-stack agent per `FULL_STACK_AGENT_PROTOCOL.md`.

## Decision question

Does EAGLE3 speculative decoding raise **conc-1 output speed** on our
serve-ready arm enough to adopt for latency-tier serving, and what does it cost
at load?

Prior (estimated from the 20260726T132617Z window, not measured): **1.5–2.0×** at
conc 1. Our conc-1 step is latency-bound — 7.30 ms measured against a ~0.67 ms
weight-read bound (≈24.8B active params at 4 bit over 8×H100) — and a 4-token
step costs only 1.21× a 1-token step, so verification is nearly free; the limiter
is drafter cost plus acceptance. The user's target of 2.5–3.5× needs mean accepted
length ≥3.3 at k=3, which no published EAGLE3/MTP result supports.

## What we draft with (searched first, per the prime directive)

- **MTP is unavailable, not unsupported.** `config.json` declares
  `num_mtp_modules: 7` / `num_nextn_predict_layers: 1`, and our vLLM registers
  `MiniMaxM3MTP` (`vllm/models/minimax_m3/{nvidia,amd}/mtp.py`), but the released
  weights contain **zero** MTP tensors — 0 of 23,416 (BF16), 0 of 45,838
  (vendor MXFP8), layers 0–59 only, no `eh_proj`/`enorm`/`shared_head`. HF
  discussion #5 confirms; no release announced.
- **Adopted drafter:** `Inferact/MiniMax-M3-EAGLE3`, rev
  `44cafa5ace418d8b22e2958df0c6aa1f2476842c`, 6.53 GB, on disk at
  `/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3`. Config is
  `LlamaForCausalLMEagle3` (1 layer, hidden 6144, MHA 64 heads, full 200,064
  draft vocab) → resolves through `registry.py:611` in our own vLLM 0.24.0.
- **Target support is already in our stack:** `MiniMaxM3SparseForCausalLM` and
  `MiniMaxM3SparseForConditionalGeneration` both declare `SupportsEagle3`
  (`vllm/models/minimax_m3/nvidia/model.py:934,982`), and our `SpeculativeConfig`
  accepts `method: eagle3` plus the `attention_backend` field.
- **Third-party validation of the same drafter:** the vLLM M3 recipe pins
  `num_speculative_tokens: 3` with `attention_backend: FLASH_ATTN`, and a
  production MXFP8 deployment (user-supplied manifest, 2026-07-13) runs that
  config through c=16 and 30-min c=32 soaks with 0 aborts / 0 restarts / 0 Xid.
  Both are B200/H200-class or MXFP8 — **neither covers H100 + our W4AFP8 +
  Humming**, which is exactly the gap this experiment closes.

## Design — wave 1 (AA-style sweep only)

Target, kernel, topology and serve flags are **fixed and identical** to the
`20260726T132617Z` window's `gptq-hum-idx-0110` arm (in-house GPTQ W4AFP8 ABI
overlay, Humming indexed 0.1.10, TP8/EP8, `MAX_MODEL_LEN=131072`,
`LLMC_M3_CAPTURE_SYNC=sync`). The **only** per-arm difference is
`--speculative-config`:

| arm | `SPEC_K` | port | node |
|---|---|---|---|
| `k0-control` | 0 (no spec config) | 8020 | 1 × 8×H100 |
| `k1` | 1 | 8021 | 1 |
| `k3` | 3 (vLLM-recipe / production value) | 8022 | 1 |
| `k5` | 5 | 8023 | 1 |

Workload: **AA-style sweep, 1k + 10k input × conc 1 / 10** — natural output, no
`ignore_eos`. Deliberate: forced continuation past the natural stop would inflate
acceptance (repetitive text is easy to draft), so the pinned-8k reasoning shape is
*not* the right first measurement. The suite-native reasoning path is wave 2, only
if wave 1 shows a real gain.

Cost: 4 nodes in parallel, ~20 min boot + ~15 min sweep ≈ **45 min wall**, well
inside the 10 idle compute nodes visible at launch.

## Gates (fail-closed, before any number is believed)

1. Serve preflight + **Humming attestation `valid: true`** on every arm (same
   gate as the window; no CUTLASS fallback).
2. **Spec-dec actually active** on `k>0` arms: `serve.log` must show
   `num_speculative_tokens=k` and the `SpecDecoding metrics` logger must report
   drafted > 0. A "no gain" result from silently-disabled spec-dec is the one
   failure mode that would produce a wrong conclusion.
3. **Greedy-equivalence probe** (`pipeline/specdec_greedy_probe.py`): 8 fixed
   prompts, `temperature=0`, 256 tokens, thinking on, run on every arm. Rejection
   sampling is distribution-preserving, so `k>0` output should match `k0` — a
   large early divergence indicates a broken multi-query verify path (M3's sparse
   attention feeds k query positions per sequence, which our MSA/cudagraph path
   has only ever run at 1). Gate: ≥6/8 prompts identical for the first 32 tokens;
   below that the arm is reported as suspect, not as a speedup.
4. AA runner rc=0, all cells `status=ok`; no Xid, no worker restart.

## Metrics returned

Per arm, per cell: output speed (`output_token_throughput_per_user`), TTFT,
`aggregate_output_tps`, natural OSL. Per arm, per phase: mean acceptance length,
per-position acceptance vector, drafted/accepted totals (from `serve.log`
`SpecDecoding metrics` lines and `/metrics` deltas around each phase). Plus the
greedy-probe match table.

OSL caveat carried from the window: AA aggregate throughput is OSL-confounded
across arms — but here all arms serve the *same* checkpoint at the same
temperature, so OSL should be comparable and any large shift is itself a finding.

## Decision rule (set now, not after seeing the data)

- **Adopt for a latency tier** if the best `k` gives conc-1 output speed ≥ **+40%**
  over `k0` AND gate 3 passes AND the conc-10 aggregate regression is ≤ **10%**.
- **Report-only** if the gain is real but conc-10 regresses more than 10% → then
  spec-dec is a load-gated mode, not a default.
- **Reject** if gains are < 20% or gate 3 fails.

The 2.5–3.5× question is answered either way: wave 1 measures the actual
multiplier at conc 1.

## Wave 2 design (user-signed 2026-07-27)

User chose: ShareGPT as the natural-prompt source; concurrency 16/32/64 at full
request counts; temp-0 cells at **both** conc 1 and 10 (not just conc 1); and "run
all phases in parallel".

**Parallelism correction that was applied:** the phases get parallel *hardware*,
not a shared server. Running a conc-64 pinned-output load beside a conc-1 latency
cell on one endpoint would confound both, so each phase gets its own pair of
serves — 6 nodes, wall time set by the longest phase rather than the sum.

Only `k` needs a serve of its own; temperature and prompt set are per-request
client parameters. k=1 and k=5 are dropped (wave 1 measured both as dominated).

| phase | serves | prompts | output | temp | conc | answers |
|---|---|---|---|---|---|---|
| `natural` | 8030 / 8031 | ShareGPT (94,145 real conversations) | natural, cap 2048 | 0.6 then 0 | 1, 10 | the production multiplier; how much of wave 1's acceptance gap was temperature vs prompt naturalness |
| `load` | 8032 / 8033 | synthetic 1k | pinned 8k (`ignore_eos`) | 0.6 | 16, 32, 64 | the concurrency where spec-dec stops paying — gates a global default |
| `lowconc` | 8034 / 8035 | synthetic 1k | pinned 8k | 0.6 | 1, 4 | like-for-like against the two-axis report's tables |

Why temp 0 is a bound and not the headline: rejection sampling accepts a draft
whenever it survives the target's distribution, and at temp 0 the target is an
argmax that agrees with the drafter's argmax far more often. Production sampling
is 0.6, so the 0.6 cells are the number to quote and the 0 cells bound it.
Note the pinned-output phases carry two opposing biases — synthetic prompts
depress acceptance, `ignore_eos` continuation past the natural stop inflates it —
so the `natural` phase is the one that describes real traffic.

ShareGPT is staged in-workspace at
`artifacts/aiperf-datasets/.cache/aiperf/datasets/` (gitignored; the path is
CWD-relative by aiperf's design) so arms run with `HF_HUB_OFFLINE=1`.

Acceptance is captured per cell from `/metrics` counter deltas, not just the log
cadence, so each cell has its own acceptance number rather than a run average.

Gates and decision rule carry over from wave 1, plus: **enable k=3 by default only
up to the highest concurrency where aggregate output tok/s is ≥ control**; above
that it must be load-gated.

## Phase D design (user-signed 2026-07-27): does prompt length change the answer?

User question: "would the result change for the natural prompts arm if we use
longer prompts (>1k), still natural?" — then "reuse the agentic suite since it is
naturally generated (verify)".

**Verification result: the agentic suite is NOT naturally generated, so it cannot
answer this.** Its `warm`/`cold` modes build prompts from aiperf's *synthetic*
token generators (`--shared-system-prompt-length`, `--user-context-prompt-length`,
`--synthetic-input-tokens-mean`) — token counts, no real text — and
`run_perf_agentic.sh:119` pins output with `ignore_eos: true` +
`max_tokens = min_tokens = AG_OUTPUT_TOKENS`, the exact shape measured here as
inflating acceptance +33%. Its `replay` mode also forces `ignore_eos`, and
`mooncake` mode has no trace on disk (`env.sh:92` is a `/PATH/TO` placeholder) and
Mooncake traces carry lengths/hashes rather than text.

**Nor is any aiperf public dataset multi-turn**: every public loader keeps only the
first message (ShareGPT `conversations[0]`, SpecBench `turns[0]`, HF-conversation
"Extracts the first message ... producing single-turn Conversations").

**Adopted instrument (prime directive — reputable existing resource):**
`nvidia/SPEED-Bench` (paper arXiv:2604.09557, blog, and NVIDIA's `specdec_bench`
framework), registered in our aiperf 0.8.0 already. It is built for this exact
measurement — "evaluate speculative decoding across diverse semantic domains and
realistic serving regimes ... acceptance-rate characteristics and end-to-end
throughput" — and its *throughput* split is fixed-ISL buckets crossed with entropy
tiers, so length and content vary independently:

| axis | values |
|---|---|
| ISL bucket | 1k, 8k, 32k (verified under the M3 tokenizer: 1018/8100/32408 mean) |
| entropy tier | `low_entropy` (code, sorting — copy-heavy best case), `high_entropy` (creative writing — worst case) |
| concurrency | 1 (all six cells), 10 (1k and 8k only, to bound KV and wall clock) |

Two release facts required a staging step (`pipeline/stage_speedbench.py`) rather
than `--public-dataset speed_bench_*`:

1. **~45% of the public parquet is masked** ("FULL BENCHMARK DATA SHOULD BE FETCHED
   FROM THE SOURCE USING SPECDEC_BENCH") and aiperf's `SpeedBenchLoader` does not
   filter those rows. A masked row is short and repetitive, so it would break the
   ISL bucket *and* inflate acceptance. The controller gates fail-closed on zero
   surviving placeholders plus per-file sha256.
2. **The `mixed` tier is 100% masked** (512/512) in every throughput split, so
   phase D reports `low_entropy` and `high_entropy` only.

Clean prompts available per cell after filtering: 419/478 (1k), 343/465 (8k),
338/486 (32k) — against the 40–100 each cell consumes.

Output is **natural** (`max_tokens` 2048, no `ignore_eos`), temp 0.6, and
`--random-seed 42` is fixed so both arms draw identical prompts in identical order.

**Harness comparability:** these numbers are **not** directly comparable to
published SPEED-Bench scores. We run the public release's *clean subset* rather
than the full data fetched via `specdec_bench`, the `mixed` tier is absent, and the
serving stack is our own W4AFP8 + Humming. The cells are valid for
control-vs-k3 decisions in-window, which is what phase D is for.

Prior expectation to test: the length axis is already flat on synthetic prompts
(1.72× at 1k, 1.75× at 10k in wave 1) and ITL is flat at 7.25–7.30 ms from 227 to
10,000 tokens of context, so per-token speedup should be length-invariant. The one
untested mechanism is copyable long context, which is what `low_entropy` at 8k/32k
probes and what synthetic random tokens structurally cannot show.

Launchers: `pipeline/slurm/run_specdec_phaseD_srun.sh` +
`pipeline/slurm/specdec_phaseD_arm.sh` (ports 8040/8041). Window
`20260727T073533Z-phaseD`.

## Raw evidence

`/mnt/nfs/hoangduy/results/m3-specdec-eagle3/<RUN_TS>/` — per arm: `serve.log`,
`backend-attestation.json`, `spec-metrics.log`, `metrics-{pre,post}-aa.txt`,
`greedy-probe.json`, `aa-sweep.log`, `aa-results.path`.
Launcher: `pipeline/slurm/run_specdec_eagle3_srun.sh` (controller, `srun` from
detached tmux) + `pipeline/slurm/specdec_eagle3_arm.sh` (per-arm).
