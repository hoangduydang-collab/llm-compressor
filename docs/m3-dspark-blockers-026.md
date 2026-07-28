# DSpark on MiniMax-M3: blocked by an upstream Model-Runner-V2 incompatibility

**Status 2026-07-28: axis-1 measurement not possible.** Root-caused offline from source.
Not a config error on our side, and not a drafter-quality finding — DSpark never got far
enough to produce an acceptance number.

Window: `/mnt/nfs/hoangduy/results/m3-specdec-dspark/20260728T075431Z-k-sweep`
(gpu-h123, job 13422, cancelled after the k=0 control).

## What passed

The smoke serve `D-k8-a` cleared every wiring gate. This matters: the dominant risk for
this drafter was vLLM silently falling back to M3's 3-layer EAGLE3 default aux layers,
which is what cost the in-house GLM-5.2 DSpark study a window (acceptance 1.27 vs ~3.9).

| gate | result |
|---|---|
| aux layers resolved **from config** = `(2, 13, 24, 36, 47, 58)` | pass |
| drafter resident, 29.47 GiB (band 27.5–32.5) | pass |
| `'method': 'dspark'`, `num_speculative_tokens: 8` | pass |
| Humming attested, carrying `VLLM_VERSION_PROVISIONAL` | pass |
| acceptance floor (`ACC_MIN=2.2`) | **never reported — engine died first** |

17 of 30 requests completed (89 s of clean generation) before the crash, so thousands of
decode steps succeeded. This is a rare batch shape, not a systematic incompatibility.

## The blocker

All 8 workers simultaneously:

```
vllm/models/minimax_m3/common/indexer.py:321
    assert num_decode_tokens == num_decodes * decode_query_len
AssertionError
```

The causal chain, each link read from the installed source:

1. `method="dspark"` **hard-forces Model Runner V2** in `vllm/config/vllm.py`, with an
   explicit comment: *"DSpark is implemented only by the V2 GPU model runner … If V2 is
   unsupported for the rest of the config, `_validate_v2_model_runner` raises rather than
   silently falling back to V1 (which can't run dspark)."* So V2 is not optional here.
2. M3's architecture `MiniMaxM3SparseForConditionalGeneration` is **absent** from
   `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` in *both* 0.24.0 and 0.26.0. Therefore
   **every EAGLE3 number we have ever published ran on Model Runner V1.** DSpark is the
   first thing to put M3 on V2.
3. M3's MSA indexer was written for V1's batch shapes. vLLM's own M3 day-0 blog says the
   day-0 work "updates the MSA decode indexer, top-k selection, and sparse GQA decode
   kernels to support **a uniform `decode_query_len`**, … flattening speculative
   verification tokens in request-major order" — i.e. the EAGLE3 shape, fixed k, one
   query block per request.
4. The assert is a **real invariant**, not a stale check: `nd = num_decode_tokens` and the
   kernel is called as `minimax_m3_index_decode(iq[:nd], …, d.block_table, d.seq_lens,
   d.decode_query_len, …)`. It guarantees `iq[:nd]` holds exactly
   `num_decodes × decode_query_len` query rows. **Relaxing it would read past `iq`.**
   Any fix belongs in the split/bookkeeping, upstream — not in a local assert removal.

### What is NOT yet established

The exact violating batch shape. The obvious candidate is full-CG request padding:
`query_start_loc_np[num_reqs + 1 :] = num_tokens` gives padded slots `query_len == 0`,
and `split_decodes_and_prefills(require_uniform=True)` then takes its padded-uniform
early return, yielding `num_decodes = num_reqs_padded` against real-token
`num_decode_tokens`. Note lines 319 and 321 disagree about this on their face — 319
explicitly tolerates zero-length slots, 321 cannot.

**But the arithmetic does not close:** for FULL cudagraphs, `prepare_attn` passes
`num_tokens = input_batch.num_tokens_after_padding`, and for a uniform-decode capture
`num_tokens_padded == num_reqs_padded × decode_query_len`, so the clean padded case comes
out *consistent*. Something else is different about the failing batch. Do not patch the
attention path on the padding theory alone.

## Next step: instrument, don't guess

1. Patch the assert to dump `query_lens_cpu`, `num_reqs`, `num_reqs_after_padding`,
   `num_tokens`, `num_actual_tokens`, `cg_mode`, `decode_query_len` and re-raise.
   One 25-minute serve converts this from a hypothesis into exact evidence.
2. Test `cudagraph_mode=PIECEWISE` as a workaround — upstream's own annotation in
   `v1/worker/gpu/cudagraph_utils.py` reads
   `num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)`.
   Pass via `EXTRA_VLLM_ARGS` as `--compilation-config {"cudagraph_mode":"PIECEWISE"}`
   (no spaces, survives word-splitting). Costs FULL-CG decode perf but would let
   acceptance be measured.

## Consequence for the study design

DSpark-vs-EAGLE3 now crosses **two** boundaries, not one: 0.24.0→0.26.0 *and* MRV1→MRV2.
A speed comparison would need EAGLE3 re-measured under V2 (forceable via
`VLLM_USE_V2_MODEL_RUNNER=1`) *and* a 0.26.0 whose k=0 baseline does not IMA — see the
status block in `m3-serve-venv-026.md`.

**Acceptance is the cheap decisive quantity.** It is a property of the drafter, not the
kernel path, and DSpark's published low-entropy acceptance (3.19) already sits *below* our
measured EAGLE3 3.86. An acceptance-only probe can therefore settle whether DSpark is
worth any further node time, without ever establishing kernel-path parity. Spend the node
there first.

## Upstream-reportable (user-gated — outward-facing, do not file without say-so)

- `minimax_m3/common/indexer.py`: uniform-decode invariant violated under Model Runner V2;
  lines 319 and 321 disagree about zero-length padded request slots.
- Reproducer: MiniMax-M3 + `--speculative-config {"method":"dspark",...}`, TP8,
  `FULL_AND_PIECEWISE`, 8k prompts, conc 1 — fails within ~20 requests.
