# M3 perf — the two-axis view (window `m3-two-axis-perf/20260726T132617Z`)

Per `M3_TWO_AXIS_PERF_PLAN.md` (user-signed). One controller, ten arms, all
rc=0, no fallbacks (every serve booted at `max_model_len=131072`, including the
2-node BF16 and MXFP8). Humming attestations valid on both Humming arms.
Serving stack for every arm: vLLM 0.24.0 fork, capture-sync fix on, 1 node
8×H100 TP8/EP8 (BF16: 2 nodes TP16/ray). Suite `PERF_STRICT=1`.

Table formats (identical everywhere):

- **Primary path** (suite-native reasoning, 1k input / 8k pinned output):
  rows = concurrency 1/4/16/64, cells = **TPOT p50 ms (system output tok/s)**.
- **AA-style sweep** (natural output): rows = input×conc incl. 100k, cells =
  **per-user decode p50 tok/s (TTFT p50 ms)**.

Metric names (aiperf 0.8.0 fields, so nothing is guessed):

| reported as | aiperf field | definition |
|---|---|---|
| TPOT p50 (ms) | `inter_token_latency` | (request latency − TTFT) ÷ (output tokens − 1) |
| per-user decode (tok/s) | `output_token_throughput_per_user` | exactly 1 ÷ ITL — one user's decode rate. aiperf/InferenceX plot this axis as "interactivity"; at conc 1 it is 1000 ÷ TPOT |
| system output tok/s | `output_token_throughput` | all output tokens ÷ benchmark wall time (prefill is in the denominator, so it is not a pure decode rate) |
| per-GPU output tok/s | derived | system output tok/s ÷ GPUs (8 quant, 16 BF16) |
| TTFT (ms) | `time_to_first_token` | first streamed token (= first reasoning token) |
| requests/s | `request_throughput` | completed requests ÷ wall time; on the pinned-8k shape = system output tok/s ÷ 8000 |

Earlier revisions of this doc said "aggregate tok/s" for system output throughput
and "interactivity tok/s" for per-user decode — same numbers, non-standard names.

Highlight convention (same as the HTML report): **bold** = best in the row, and
every cell within 1% of it is bolded too (that is the noise floor, so
co-leaders show as co-leaders); every data row carries at least one bold cell.
† marks cells that are not valid comparisons (cyankiwi's runaway generations) —
excluded from the ranking.

## Axis 1 — Kernel (model fixed: in-house GPTQ W4AFP8)

### Primary path

Sources: CUTLASS anchor `20260725T122256Z`; Humming arms `20260726T033158Z`.
(Same shapes/stack as today's window; cross-window TPOT noise floor is ±0.2%,
verified again today: the rerun idx-0110 arm reproduced 7.30/8.82/12.23/19.40
vs 7.29/8.82/12.22/19.43.)

| conc | CUTLASS | Hum idx 0.1.10 | Hum grp 0.1.10 | Hum idx 0.1.11pk | Hum grp 0.1.11pk |
|---|---|---|---|---|---|
| 1 | 9.73 (102) | **7.29 (137)** | 8.48 (117) | 7.59 (131) | 8.58 (116) |
| 4 | 11.91 (335) | **8.82 (452)** | 9.91 (402) | **8.90 (448)** | 9.89 (403) |
| 16 | 15.28 (1042) | **12.22 (1302)** | 13.28 (1198) | **12.23 (1300)** | 13.04 (1220) |
| 64 | 22.28 (2849) | 19.43 (3262) | 20.64 (3046) | **19.21 (3299)** | 20.11 (3128) |

### AA-style sweep (this window, all five kernel arms, uniform 131072 ctx)

| input×conc | CUTLASS | Hum idx 0.1.10 | Hum grp 0.1.10 | Hum idx 0.1.11pk | Hum grp 0.1.11pk |
|---|---|---|---|---|---|
| 1k × 1 | 102.2 (114) | **137.3 (114)** | 117.8 (125) | 132.3 (131) | 116.1 (124) |
| 1k × 10 | 71.9 (403) | 92.2 (423) | 84.9 (447) | **93.3 (432)** | 88.0 (427) |
| 10k × 1 | 102.0 (532) | **137.1 (554)** | 117.5 (557) | 131.9 (554) | 115.9 (561) |
| 10k × 10 | 56.6 (1705) | **75.8 (2685)** | 65.4 (2709) | 64.3 (1868) | 60.4 (1790) |
| 100k × 1 | 100.3 (5193) | **130.9 (5391)** | 115.5 (5473) | 127.8 (5512) | 113.7 (5456) |
| 100k × 10 | 15.7 (24529) | 13.8 (18110) | **17.7 (24389)** | 14.4 (18130) | 16.1 (29715) |

Readings:

- **Kernel ordering is stable across every input length at low/mid load:**
  Humming indexed > grouped > CUTLASS, 0.1.10 indexed fastest (matches the
  suite).
- **Long-context decode holds up:** conc-1 per-user decode barely moves from
  1k to 100k input (137.3 → 130.9 idx-0110; 102.2 → 100.3 CUTLASS). 100k
  conc-1 TTFT ≈ 5.2–5.5 s on all W4AFP8 arms.
- 100k × 10 is a saturation cell (≈1M live prompt tokens): TTFT ~18–30 s,
  per-user decode 14–18 tok/s, ordering noisy — capacity data, not a kernel
  A/B cell.
- Packed-K repeats yesterday's story: −3–4% conc-1 decode on indexed, ~flat on
  grouped, mixed at conc-10.

## Axis 2 — Quantization (kernel fixed: Humming indexed 0.1.10 where format admits)

All five arms in **this** window — one stack, one day, one config. Kernel per
arm: in-house GPTQ + in-house AWQ r7 → Humming idx 0.1.10 (both are W4AFP8
group-128); cyankiwi → Marlin (it is W4A16 group-32, **cannot** run Humming);
MXFP8 and BF16 → their native paths. BF16 runs TP16 on **2 nodes (16 GPUs)** —
per-GPU numbers divide by 16, the quant arms by 8.

### Primary path (reasoning)

| conc | GPTQ·Hum | AWQ-r7·Hum | cyankiwi·Marlin | MXFP8 | BF16 (16 GPU) |
|---|---|---|---|---|---|
| 1 | **7.30 (137)** | **7.30 (136)** | 7.96 (118) | 9.31 (107) | 12.40 (81) |
| 4 | **8.82 (451)** | **8.80 (453)** | 9.33 (415) | 11.40 (349) | 16.23 (246) |
| 16 | **12.23 (1300)** | **12.18 (1303)** | 12.35 (1268) | 16.42 (968) | 22.48 (711) |
| 64 | **19.40 (3267)** | **19.43 (3262)** | 21.10 (2923) | 27.52 (2177) | 35.16 (1700) |

Per-GPU output throughput at conc-64: GPTQ/AWQ 408, cyankiwi 365, MXFP8 272, BF16 106
tok/s/GPU → in-house W4AFP8 = **3.84× BF16 per GPU**.

### Serve-ready vs BF16 baseline (per-GPU efficiency)

Serve-ready = in-house GPTQ W4AFP8 on Humming indexed 0.1.10 — the one arm with
a shipping quality verdict (recovery 97.4–101.1% on all seven tasks). Both
columns come from this window. Per-GPU = the cell divided by that arm's GPU
count (BF16 16, quant 8).

| metric | BF16 baseline (16×H100, 2 nodes) | serve-ready W4AFP8 (8×H100, 1 node) | advantage |
|---|---|---|---|
| GPUs · nodes to serve | 16 · 2 | **8 · 1** | half the fleet, no cross-node hop in the serving path |
| weights on disk | 796 GB | **225 GB** | **3.5× smaller** — why it fits one node's HBM |
| reasoning conc 1 · output tok/s (single stream, system = per-user decode) | 81 (5.0/GPU) | **137 (17.1/GPU)** | 1.7× per user · **3.4× per GPU** |
| reasoning conc 4 · system output tok/s | 246 (15.4/GPU) | **451 (56.4/GPU)** | 1.8× total · **3.7× per GPU** |
| reasoning conc 16 · system output tok/s | 711 (44.4/GPU) | **1300 (162.5/GPU)** | 1.8× total · **3.7× per GPU** |
| reasoning conc 64 · system output tok/s | 1700 (106/GPU) | **3267 (408/GPU)** | 1.9× total · **3.8× per GPU** |
| GPU-hours per 1M output tokens (conc 64) | 2.61 | **0.68** | **3.8× cheaper** per token at full load |
| agentic warm 16 · per-user decode tok/s | 41.4 | **72.2** | **1.7× per user**, on half the GPUs |
| agentic warm 16 · system output tok/s | 490 (30.6/GPU) | **727 (90.9/GPU)** | 1.5× total · **3.0× per GPU** — prefill-bound, BF16's best regime |
| agentic warm 16 · p95 TTFT ms | **783** | 893 | BF16 wins by 12% (16 GPUs prefill faster); both inside the 1 s SLO |
| 100k-token prompt conc 1 · per-user decode tok/s | 78.4 | **130.9** | **1.7× per user** on half the GPUs; TTFT also 8% better (5.39 s vs 5.88 s) |

Decode-bound work improves 3.4–3.8× per GPU, prefill-bound agentic work 3.0×. The other ship-capable
options are worse buys: vendor MXFP8 is quality-clean but 272 tok/s/GPU at
conc-64 (2.6× BF16, vs our 3.8×); in-house AWQ r7 is speed-identical to GPTQ
(within 1% in every cell) but has no quality verdict yet; cyankiwi is
quality-disqualified.

### Agentic (warm / cold) — per-user decode tok/s (TTFT p50/p95 ms)

| conc | GPTQ·Hum | AWQ-r7·Hum | cyankiwi·Marlin | MXFP8 | BF16 |
|---|---|---|---|---|---|
| warm 1 | **138.9 (177/209)** | **138.2 (196/225)** | 127.5 (177/212) | 108.1 (180/206) | 80.8 (190/226) |
| warm 4 | 106.5 (315/487) | **107.8 (302/433)** | 101.0 (304/423) | 84.0 (309/439) | 60.3 (299/420) |
| warm 16 | 72.2 (586/893) | **73.2 (609/898)** | 69.0 (734/1085) | 55.6 (704/1062) | 41.4 (511/783) |
| warm 32 | **49.1 (761/1314)** | 47.3 (688/1231) | 48.1 (1004/1836) | 38.8 (979/1647) | 27.3 (521/808) |
| cold 1 | **139.5 (565/597)** | **139.1 (575/595)** | 127.8 (806/840) | 108.6 (881/893) | 81.5 (628/645) |
| cold 4 | **86.2 (912/1928)** | **85.4 (877/1921)** | 73.4 (1312/2881) | 62.3 (1438/3163) | 51.2 (921/2110) |
| cold 16 | 29.3 (2109/5490) | **29.6 (1782/5560)** | 21.6 (2678/8243) | 12.4 (4937/7329) | 14.7 (2978/4567) |
| cold 32 | **10.0 (3215/6731)** | **10.0 (3203/7030)** | 6.9 (5356/10011) | 5.5 (4710/12167) | 7.2 (2755/8536) |

1 s p95 TTFT SLO: warm — held by the in-house arms through conc 16 (cyankiwi and
MXFP8 miss at 16), only BF16 (16 GPUs) still holds at 32; cold — missed by every
arm past conc 1. The HTML report marks each over-SLO cell with an amber bar.

### Throughput view — system output tok/s (per GPU)

| workload · conc | GPTQ·Hum | AWQ-r7·Hum | cyankiwi·Marlin | MXFP8 | BF16 (16 GPU) |
|---|---|---|---|---|---|
| reasoning 1 | **137 (17.1)** | **136 (17.0)** | 118 (14.7) | 107 (13.4) | 81 (5.0) |
| reasoning 4 | **451 (56.4)** | **453 (56.6)** | 415 (51.9) | 349 (43.7) | 246 (15.4) |
| reasoning 16 | **1300 (162.5)** | **1303 (162.8)** | 1268 (158.6) | 968 (121.0) | 711 (44.4) |
| reasoning 64 | **3267 (408.4)** | **3262 (407.8)** | 2923 (365.4) | 2177 (272.1) | 1700 (106.2) |
| agentic warm 1 | **112 (14.0)** | 109 (13.7) | 104 (13.0) | 91 (11.3) | 70 (4.4) |
| agentic warm 4 | 292 (36.6) | **299 (37.4)** | 285 (35.6) | 247 (30.8) | 189 (11.8) |
| agentic warm 16 | **727 (90.9)** | **727 (90.9)** | 662 (82.8) | 576 (72.0) | 490 (30.6) |
| agentic warm 32 | **1036 (129.5)** | **1036 (129.6)** | 931 (116.4) | 825 (103.2) | 708 (44.3) |

Cold-regime system throughput (same shape, cache defeated), GPTQ/AWQ/cyankiwi/
MXFP8/BF16: conc 1 → 81/80/66/58/56; conc 4 → 183/183/138/124/133; conc 16 →
236/235/164/120/162; conc 32 → 219/219/147/126/179 tok/s.

Agentic throughput is prefill-bound (≈100 output tokens per turn on a 7.3k-token
prefix), which is why it lands at 1036 tok/s @32 against 3267 in the pinned
reasoning shape — and why BF16's 16 GPUs close the per-GPU gap there to 2.9×
(44.3 vs 129.5) instead of 3.8×.

AA cells also carry `aggregate_output_tps` and `request_throughput_rps`, but
natural output lengths ran 307–696 tokens across arms (11k–16k for cyankiwi), so
AA throughput is OSL-confounded — use the pinned-output reasoning path for
throughput comparisons.

### AA-style sweep

| input×conc | GPTQ·Hum | AWQ-r7·Hum | cyankiwi·Marlin† | MXFP8 | BF16 |
|---|---|---|---|---|---|
| 1k × 1 | **137.3 (114)** | **137.0 (117)** | 126.3 (110) | 107.7 (118) | 80.9 (129) |
| 1k × 10 | **92.2 (423)** | 90.6 (428) | 87.4 (656) | 71.1 (638) | 50.5 (491) |
| 10k × 1 | **137.1 (554)** | **137.1 (549)** | 124.6† (807) | 107.6 (836) | 80.6 (583) |
| 10k × 10 | **75.8 (2685)** | 70.4 (2200) | 85.9† (4023) | 55.8 (4386) | 44.3 (2350) |
| 100k × 1 | 130.9 (5391) | **133.4 (5410)** | 116.0† (7703) | 104.9 (8283) | 78.4 (5883) |
| 100k × 10 | 13.8 (18110) | 17.4 (22852) | 56.5† (33162) | 15.4 (37886) | **22.5 (31815)** |

### Readings

- **The headline of the axis: quant method does not matter for speed — format
  does.** GPTQ and AWQ r7 on the same kernel are within noise in *every* cell
  (reasoning TPOT 7.30 vs 7.30, 19.40 vs 19.43; agentic and AA within ~1%). The
  perf ladder is format/kernel: W4AFP8·Humming > W4A16·Marlin > MXFP8 > BF16.
- **BF16 rerun on today's stack confirms the old baseline**: conc-1 80.7 tok/s
  vs 81 in the 07-22 window — the "rerun might recover ≤7%" caveat resolves to
  ~0.
- **The old "MXFP8/cyankiwi collapse at conc-4" did not reproduce** on today's
  stack/config: both arms complete every concurrency (MXFP8 conc-64 agg 2177).
  Treat the collapse as a property of the 07-22 window, now historical.
- † **cyankiwi validity flags (its arm only):**
  - AA long-input cells are runaway generations: at 10k/100k input it decodes
    to the 16,384-token cap (avg OSL 16,375–16,383 vs 340–700 for every other
    arm) — its "wins" at 10k×10/100k×10 are a different (much longer-decode)
    workload, not comparable cells. The quality pathology is perf-visible.
  - Suite reasoning stops ~2.6% short of the pinned 8k decode at conc-64
    (119/640 requests, avg OSL 7795) despite `ignore_eos` — small favorable
    bias to its aggregate numbers. All other arms: 5–11 mismatched requests
    total across all 12 cells.

## Cross-window reproducibility (this window vs earlier ones)

- Suite reasoning TPOT, idx-0110: 7.30/8.82/12.23/19.40 today vs
  7.29/8.82/12.22/19.43 (033158Z) vs 7.29/8.82/12.24/19.42 (122256Z) — ±0.2%.
- AA per-user decode, idx-0110: 137.3/92.2/137.1 today vs 137.4/91.7/137.2
  (040130Z, different ctx config) — ±0.5%.
- CUTLASS AA conc-1 decode 102.2 vs suite anchor 102.5.

## Raw

- Controller artifacts: `/mnt/nfs/hoangduy/results/m3-two-axis-perf/20260726T132617Z/`
- Suite: `benchmarks/results/{minimax-m3-inhouse-gptq-hum-idx-0110,minimax-m3-inhouse-awq-r7-hum-idx-0110,minimax-m3-awq-cyankiwi,minimax-m3-mxfp8,minimax-m3-bf16}/vllm/perf/{reasoning,agentic}/20260726T132617Z/`
- AA: `benchmarks/results/<profile>/self-hosted/perf/aa-sweep/20260726T132617Z/`
  (profiles above plus `minimax-m3-inhouse-gptq-{cutlass,hum-grp-0110,hum-idx-0111,hum-grp-0111}`)
- AWQ r7 checkpoint: `m3-ddp-awq-full-r7-gatealpha/.../checkpoint-vllm-w123`;
  GPTQ: `artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay`.
