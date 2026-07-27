# M3 EAGLE3 speculative decoding on our W4AFP8 arm — design packet

**Status:** WAVE 1 LAUNCHED 2026-07-27 (user-signed: "agree with the experiment
you suggested" + "just try the AA-style sweep first"). Owner: full-stack agent
per `FULL_STACK_AGENT_PROTOCOL.md`.

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

## Raw evidence

`/mnt/nfs/hoangduy/results/m3-specdec-eagle3/<RUN_TS>/` — per arm: `serve.log`,
`backend-attestation.json`, `spec-metrics.log`, `metrics-{pre,post}-aa.txt`,
`greedy-probe.json`, `aa-sweep.log`, `aa-results.path`.
Launcher: `pipeline/slurm/run_specdec_eagle3_srun.sh` (controller, `srun` from
detached tmux) + `pipeline/slurm/specdec_eagle3_arm.sh` (per-arm).
