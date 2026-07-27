# M3 EAGLE3 speculative decoding — wave 1 results (AA-style sweep)

Window `m3-specdec-eagle3/20260727T061506Z`, 4 arms, all rc=0, every gate passed.
Design + decision rule: `M3_SPECDEC_EAGLE3_PLAN.md`.

**Answer to the question that prompted this ("can spec-dec give 2.5–3.5× at conc 1?"):
no — measured 1.72–1.75× at concurrency 1 with the recipe's k=3, on AA-style
synthetic prompts. The best real-prompt estimate is ≈2.25×, derived below from
measured acceptance, and still short of 2.5×.**

## What was held fixed

Target, kernel, topology and every serve flag are identical to the
`20260726T132617Z` window's `gptq-hum-idx-0110` arm: in-house GPTQ W4AFP8 ABI
overlay, Humming indexed 0.1.10, TP8/EP8 on one 8×H100 node,
`MAX_MODEL_LEN=131072`, `kv_cache_dtype=fp8`, prefix caching on,
`LLMC_M3_CAPTURE_SYNC=sync`. The only per-arm difference is
`--speculative-config`. The control is re-measured **in this window**, and it
reproduces the earlier one to 0.2% (137.5 / 136.9 here vs 137.3 / 137.1 on 07-26),
so every ratio below is a same-window ratio.

Drafter: `Inferact/MiniMax-M3-EAGLE3` @ `44cafa5ace418d8b22e2958df0c6aa1f2476842c`
(6.53 GB, `LlamaForCausalLMEagle3`, 1 layer, hidden 6144, full 200,064 draft
vocab), loaded by our own vLLM 0.24.0 with no code change. MTP was not an option:
`config.json` declares `num_mtp_modules: 7` but neither the BF16 nor the vendor
MXFP8 release ships a single MTP tensor.

## Output speed — tok/s (AA p50), × = vs the in-window control

| cell | control | k=1 | k=3 | k=5 |
|---|---|---|---|---|
| 1k × conc 1 | 137.5 | 194.7 (1.42×) | **236.0 (1.72×)** | 228.8 (1.66×) |
| 10k × conc 1 | 136.9 | 200.9 (1.47×) | **239.0 (1.75×)** | 231.5 (1.69×) |
| 1k × conc 10 | 94.2 | 121.7 (1.29×) | 137.2 (1.46×) | 139.1 (1.48×) |
| 10k × conc 10 | 67.3 | 83.3 (1.24×) | **108.3 (1.61×)** | 92.2 (1.37×) |

## Aggregate output tok/s — spec-dec is not a latency-for-throughput trade here

| cell | control | k=1 | k=3 | k=5 |
|---|---|---|---|---|
| 1k × conc 1 | 133.7 | 185.9 | 215.0 | 216.4 |
| 10k × conc 1 | 118.5 | 159.5 | 186.9 | 176.9 |
| 1k × conc 10 | 570.0 | 679.8 | **1000.1 (1.75×)** | 994.7 |
| 10k × conc 10 | 391.8 | 453.5 | **641.3 (1.64×)** | 513.4 |

Natural OSL is comparable across arms (410–625 tokens; 582 vs 608 in the
1k×10 cell), so these aggregates are not the OSL-confounded kind that the
two-axis report warns about. **Concurrency 10 gains throughput as well as
latency** — at that load 8×H100 is nowhere near compute-saturated for this MoE
(the same node does 3,267 tok/s at conc 64), so the extra verified tokens ride in
spare capacity.

TTFT is unaffected at conc 1 (118 → 131 ms at k=3) and roughly flat at conc 10
(2,094 → 2,173 ms at 10k; the 1k×10 cell reads 461 → 293 ms, which is scheduling
noise, not a mechanism).

## Acceptance — and why the workload changes the answer

| arm | mean accepted length | avg draft acceptance | per-position |
|---|---|---|---|
| k=1 | 1.69 | 69.1% | 0.69 |
| k=3 | 2.45 | 48.2% | 0.70, 0.46, 0.29 |
| k=5 | 2.55 | 31.0% | 0.64, 0.40, 0.25, 0.16, 0.10 |

Divide accepted length by measured speedup to get what a step costs:

| arm | accepted length ÷ speedup | step cost vs plain decode |
|---|---|---|
| k=1 | 1.69 / 1.42 | 1.19× |
| k=3 | 2.45 / 1.72 | 1.42× |
| k=5 | 2.55 / 1.66 | 1.54× |

Each extra draft token costs a consistent ~0.11–0.12× of a decode step, which is
why **k=3 is the optimum and k=5 is a net loss**: it buys +0.10 accepted length
(2.45 → 2.55) for +0.12× step cost, and its per-position rates have collapsed to
0.10 by the fifth token. Nothing deeper than k=3 is worth serving.

**The prompt distribution matters more than k.** On the greedy probe's real
prompts (English questions, temp 0) the same k=3 arm accepted
**0.862 / 0.740 / 0.600 → mean length 3.20–3.35**, against 2.45 on AA's synthetic
random-token prompts at temp 0.6. Holding the measured 1.42× step cost and
substituting real-prompt acceptance gives **3.2 ÷ 1.42 ≈ 2.25×** at conc 1. That
is an inference from two measured quantities, not a measurement — confirming it on
natural prompts is the first item of wave 2.

## Gates

- Humming attestation `valid: true` on all four arms; no fallback to CUTLASS.
- Spec-dec provably active on every k>0 arm (`speculative_config` in the engine
  banner, `SpecDecodingLogging` reporting drafted > 0).
- Greedy equivalence vs control at temp 0: k=1 **7/8**, k=3 **7/8**, k=5 **6/8**
  prompts identical for ≥120 characters (one k=1 output identical for its full
  651 characters). Divergence appears after a few hundred characters, which is the
  floating-point signature of different batch shapes; a broken multi-query verify
  path would diverge within the first tokens on every prompt. M3's sparse
  attention is wired for spec-dec rather than bypassed — the drafter shares the
  target's `topk_indices_buffer`.
- AA runner rc=0, all 16 cells `status=ok`, no Xid, no worker restart.

## Verdict against the packet's decision rule

Rule was: adopt for a latency tier if conc-1 ≥ +40%, greedy passes, and conc-10
regression ≤ 10%. k=3 delivers **+72–75%** at conc 1, passes the greedy gate, and
*improves* conc-10 aggregate by 64–75%. **→ ADOPT k=3 for the latency tier.**

Scope limit, stated because the data does not cover it: this sweep went no higher
than **concurrency 10**. Spec-dec normally turns into a throughput loss once the
batch saturates compute, and our own high-load numbers live at conc 32–64. Do not
enable it globally on that basis — the conc 32/64 measurement is wave 2.

## Wave 2 (proposed, not run)

1. Acceptance and speedup on **natural prompts** (agentic warm shape or a replayed
   trace) — tests the ≈2.25× inference and is the number that describes production.
2. **conc 32 / 64** at k=3 — finds the load where spec-dec stops paying, which
   sets the gating threshold.
3. The suite-native reasoning path at k=3, for a like-for-like number against the
   two-axis report's pinned-output tables.

## Raw evidence

`/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T061506Z/` — per arm
`serve.log`, `spec-boot.log`, `spec-metrics.log`, `backend-attestation.json`,
`greedy-probe.json`, `metrics-{pre,post}-aa.txt`, `aa-sweep.log`; window-level
`AGGREGATE.md`, `aggregate.json`, `arm-provenance.txt`, `actual-commit.txt`.
AA cells: `benchmarks/results/minimax-m3-specdec-{k0-control,k1,k3,k5}/self-hosted/perf/aa-sweep/20260727T061506Z/`.

Infrastructure note: `gpu-h97`, `gpu-h98` and `gpu-h101` were reported `idle` by
slurm while another user's out-of-band DeepSeek-V4 run held all 8 GPUs on each
(~32 GiB free). Two arms were refused by our serve preflight and one died mid-boot
when the foreign job took memory after preflight had passed; all three were
re-run on verified-clean nodes via `pipeline/slurm/relaunch_specdec_eagle3_arms.sh`.
No arm reported numbers from a contended node.
