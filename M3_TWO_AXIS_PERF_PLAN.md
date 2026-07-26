# M3 two-axis performance report — data-completion plan (design packet)

**Status:** EXECUTED 2026-07-26 (user-signed). Window
`m3-two-axis-perf/20260726T132617Z` — all ten arms rc=0, no fallbacks
(every serve incl. BF16/MXFP8 booted at 131072). Results:
`docs/m3-two-axis-perf.md`; narrative: `M3_OFFICIAL_PERF_RESULTS.html`.
**Date:** 2026-07-26. Owner: full-stack agent per `FULL_STACK_AGENT_PROTOCOL.md`.

## Objective

Restructure the perf story into two axes and collect the missing cells so each
axis is a single, internally-uniform window:

- **Axis 1 — Kernel** (model fixed: in-house GPTQ
  `artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay`):
  CUTLASS (stream-off → stream-on history) → Humming indexed/grouped 0.1.10 →
  indexed/grouped 0.1.11 packed-K.
- **Axis 2 — Quantization** (kernel fixed: Humming **indexed 0.1.10** wherever
  the checkpoint format admits it): in-house GPTQ, in-house AWQ r7, cyankiwi,
  BF16, MXFP8.

Each axis reports **both** standard paths with one consistent table format:

- **Primary path** (suite-native, pinned OSL): reasoning = 1k input / 8k forced
  output (`ignore_eos`), conc **1 / 4 / 16 / 64** (`R_INPUT_TOKENS=1000`,
  `R_OUTPUT_TOKENS=8000`, `CONC_REASONING="1 4 16 64"` — verified in
  `benchmarks/env.sh`). Suite also carries non-reasoning (1k→500) and agentic
  (conc 1/4/16/32, warm+cold).
- **AA-style sweep** (natural output): extended matrix
  **1k / 10k / 100k input × conc 1 / 10** (the runner natively supports 100k:
  `INPUT_TOKENS={"1k":1000,"10k":10000,"100k":100000}`, request counts
  8/8/4 at conc-1, 10 at conc-10, AA floors 1000/1500/2000 recorded not
  forced).

## Format-compatibility facts (verified from checkpoint configs)

| arm | format | Humming idx-0110 applicable? |
|---|---|---|
| in-house GPTQ (w123 ABI overlay) | W4AFP8, group-128, actorder static | yes (current arms) |
| in-house AWQ r7 (gate-alpha) | W4AFP8, group-128, no actorder | **yes** |
| cyankiwi `MiniMax-M3-AWQ-INT4` | **W4A16, group-32, no act quant** | **no — Marlin W4A16 only** |
| BF16 `MiniMaxAI/MiniMax-M3` | unquantized | no — native BF16 path |
| MXFP8 `MiniMaxAI/MiniMax-M3-MXFP8` | MXFP8 | no — its own FP8 path |

So "rerun on the best kernel" concretely means: AWQ r7 moves onto Humming
indexed 0.1.10; cyankiwi/BF16/MXFP8 are rerun **contemporaneously on today's
serving stack** (post shared-stream fix, vLLM 0.24.0) on their only available
paths, so no axis-2 cell mixes serving stacks. The 07-22 five-arm window is kept
in the report as historical/secondary only.

## Run matrix (new cluster work)

One node 8×H100 per arm, TP8/EP8, `srun` from detached tmux controllers,
`PERF_STRICT=1`, Humming attestation fail-closed on Humming arms.
**Serving `MAX_MODEL_LEN=131072`** for every new arm (needed by 100k-input AA
cells; model max positions 1,048,576). Axis-1 primary tables stay on the
existing uniform 40960 windows; every *new* window is uniform at 131072.

### Wave A — axis 2, combined arms (primary suite + extended AA per serve)

| arm | ckpt | kernel path | new data |
|---|---|---|---|
| gptq·hum-idx-0110 | w123 ABI overlay | Humming indexed 0.1.10 | suite + AA (anchors both axes) |
| awq-r7·hum-idx-0110 | AWQ r7 gate-alpha | Humming indexed 0.1.10 | suite + AA |
| cyankiwi·marlin | cyankiwi AWQ-INT4 | Marlin W4A16 | suite + AA |
| bf16 | MiniMax-M3 | native | suite + AA |
| mxfp8 | MiniMax-M3-MXFP8 | native FP8 | suite + AA |

### Wave B — axis 1, AA-only arms (extended AA per serve; primary already exists)

| arm | kernel path | new data |
|---|---|---|
| gptq·cutlass | CUTLASS W4A8 MoE | AA (was never run on CUTLASS) |
| gptq·hum-grp-0110 | grouped 0.1.10 | AA extended (100k cells; uniform 131072 window) |
| gptq·hum-idx-0111 | indexed 0.1.11 packed-K | AA extended |
| gptq·hum-grp-0111 | grouped 0.1.11 packed-K | AA extended |

Wave A's gptq·hum-idx-0110 AA result is shared with axis 1 (same arm), so the
AA table gets all five kernel arms from the same window/config.

### Cost / wall-clock

9 serves total. Wave A ≈ 3–3.5 h/arm (serve ≈ 20 min + suite ≈ 2 h + AA ≈
30–45 min with 100k cells); Wave B ≈ 1–1.5 h/arm. ~7 idle compute + 2 idle
debug nodes visible now → both waves largely parallel, **~4–5 h wall** total.

### Contingencies (pre-agreed, no re-negotiation)

- BF16 or MXFP8 fails to serve at 131072 (KV pressure): retry at 40960 with the
  100k AA cells marked n/a for that arm; suite unaffected.
- AWQ-on-Humming attestation failure: fail closed, report — do **not** silently
  fall back to CUTLASS (never drop a broken arm; root-cause it).
- 100k×conc-10 cell is included (runner defines it, 10 requests); it is the
  riskiest cell (≈1M live prompt tokens) — cells run in matrix order with 100k
  last, per-cell artifacts survive a late failure and can be hand-parsed.

### Pass/fail & "done"

- Every arm: serve preflight + (Humming arms) attestation `"valid": true`;
  suite rc=0 under `PERF_STRICT=1`; AA runner rc=0 with all cells `status=ok`
  (100k cells may report floors-not-cleared — expected, recorded).
- Done = both axes' primary + AA tables fully populated from uniform windows,
  report restructured (md + HTML) into the two-axis layout.

## AA-methodology fidelity audit (vs artificialanalysis.ai/methodology/performance-benchmarking, fetched 2026-07-26)

Faithful in our runner (`performance/aa/run_aa_sweep.py`):

- **Axes**: input 1k/10k/100k × single (conc-1) / parallel-10 — exactly AA's.
- **Output length**: natural generation, never forced (`ignore_eos`/min-tokens
  deliberately absent); AA answer-token floors 1000/1500/2000 recorded and
  *checked*, not imposed — matches AA's "at least N answer tokens".
- **Sampling**: temperature 0.6 (AA's reasoning-model setting), top_p left at
  default 1, thinking enabled via `chat_template_kwargs`.
- **Metrics**: TTFT = first streamed token (= first reasoning token, AA's
  definition); output speed = per-user tokens/s after first token
  (`output_token_throughput_per_user`), AA's "Output Speed".
- **Prompt hygiene**: unique synthetic prompts per request, prefix pool 0 —
  no prefix-cache advantage.

Known deviations (already documented in the runner; keep them in every report):

1. **Prompt content**: synthetic random-token prompts vs AA's dynamically
   generated real long-form content + task variety (their v2.2.0). Matters for
   speculative decoding; our serving doesn't use spec-dec, so low impact.
2. **Token units**: we count with the model's own tokenizer; AA counts with
   tiktoken `o200k_base`. Cross-provider absolute numbers are therefore not
   unit-identical; our A/B deltas are unaffected.
3. **Sampling regime**: one batch per cell (8/10/4 requests) vs AA's repeated
   runs (8×/day, 72-h median; 100k weekly, 14-day median). Ours is a snapshot.
4. **Missing metric**: AA also reports "time to first *answer* token" (post-
   thinking); we don't split reasoning vs answer chunks (nor AA's last-80%%-of-
   answer-chunks fallback — unneeded, vLLM streams reasoning tokens).
5. **Vantage**: localhost client vs AA's GCP us-central1 over WAN — we measure
   the engine, not the network (intentional).
6. Non-reasoning models would need temperature 0 per AA; M3 runs are reasoning
   so 0.6 is the correct setting here.
7. **Practical M3 caveat** (measured, not design): natural outputs on synthetic
   prompts ran 340–740 tokens — AA floors never cleared, so *aggregate* cell
   throughput is OSL-confounded; per-user decode speed and TTFT are the
   comparable columns. Verdict: the sweep is honestly "AA-style", not
   AA-comparable — exactly what its own docstring claims.

## Report restructure (md + HTML) — after data lands

- **Axis 1 — Kernel** section: evolution narrative (stream-off → stream-on →
  Humming 0.1.10 → 0.1.11 packed-K), primary table (rows = conc 1/4/16/64,
  cols = kernel arms, cell = TPOT p50 ms + agg tok/s), AA table (rows =
  input×conc incl 100k, cols = kernel arms, cell = per-user decode tok/s +
  TTFT p50 ms).
- **Axis 2 — Quantization** section: same two table shapes, cols = model arms
  (kernel noted per arm: Humming idx-0110 / Marlin / native), new uniform
  window as primary results; 07-22 five-arm tables demoted to a clearly-labeled
  historical subsection.
- Table format identical across axes and across md/HTML.
