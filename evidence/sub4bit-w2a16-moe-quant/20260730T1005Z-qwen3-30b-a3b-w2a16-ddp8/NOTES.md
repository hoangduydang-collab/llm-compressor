# In-house sub-4-bit MoE: DDP AutoRound W2A16 quant + Humming serving — PASS

Dates: quant 2026-07-30 (job 13467, gpu-h104, 8×H100, 1h40m wall);
serve smoke 2026-07-31 (job 13469, gpu-h104, 1×H100).

## What was proven

The COMPLETE in-house loop for sub-4-bit MoE: our own pipeline quantized
Qwen3-30B-A3B-Instruct-2507 to **W2A16 g128 symmetric compressed-tensors
pack-quantized** with **distributed (DDP) AutoRound on 8 GPUs**, and the
resulting checkpoint serves coherently on the vLLM+Humming path.

- Artifact: `artifacts/Qwen3-30B-A3B-Instruct-2507-autoround-W2A16-g128-ddp8/`
  (8.6 GB from 57 GB BF16; single safetensors; routers + lm_head BF16;
  all 388 Linears per block quantized incl. 128 routed experts).
- Quant gate (`launch.sh`): format=pack-quantized num_bits=2 group=128
  sym=True → QUANT_RESULT: PASS (`full-run-output.txt`).
- Serve smoke (serves/qwen3-30b-a3b-w2a16-humming-smoke/launch-inhouse.sh,
  venv `serve-sub4`): backend chain `HummingLinearKernel` →
  `CompressedTensorsWNA16MarlinMoEMethod` → `'HUMMING' WNA16 MoE backend`
  (indexed gemm); probes "42" (3 completion tokens), "Paris", fluent MoE
  definition → SMOKE_RESULT: PASS (`serve-inhouse-output.txt`).

## Stack (quant side)

venv `quant-sub4` = clone of `quant` with:
- llm-compressor upstream main `0.12.1.dev92+g8cec0acc` (non-editable,
  replaces the fork's editable install)
- auto-round **0.14.1** (REQUIRED: 0.13.1's DDP never syncs gradients — see
  bug 1 below)
- compressed-tensors nightly **0.17.2a20260729** (REQUIRED: LLMC main's save
  path calls `compress_model(skip_compressed=...)`, absent in 0707 nightly)
- torch 2.11.0 / transformers 5.14.1 (unchanged from `quant`)
- one local patch to site-packages `llmcompressor/datasets/utils.py`
  (bug 2 below; same fix committed to our fork as ddd4f9f9)

Run: `torchrun --nproc_per_node=8 quant_w2a16_ddp.py`, 512 pile-10k samples
(64/rank), iters=200, seqlen 2048, batch_size 8/rank, torch_compile off,
offline HF cache (HF_HOME=/mnt/nfs/hoangduy/hf_assets/xet, HF_HUB_OFFLINE=1).
~1.9 min/block × 48 blocks; DDP = data-parallel tuning (same iters each rank,
grads all-reduce-AVG each step; sign(avg)==sign(sum) so SignSGD-safe).

## Two bugs found before spending the full run

1. **auto-round 0.13.1 DDP is a silent no-op**: `setup_ddp_if_needed_` built
   the DDP wrapper, rebound only a local variable, returned None; tuning
   forwarded the raw block. Empirically verified (2-rank gloo test: grads
   differ). Fixed upstream in 0.14.1: returns `(block, sync_fn)`;
   `sync_gradients()` runs before every `optimizer.step()`.
2. **LLMC dist data partition applied to a local variable only** (upstream
   main bug, present in our fork too): `_make_sampler` partitioned a local
   dataset copy, but the DataLoader wrapped the FULL dataset, so the sampler's
   shard-relative indices selected the same first-partition rows on every rank
   — every rank calibrated on identical data (smoke 1: bit-identical per-rank
   losses; smoke 2 after fix: distinct losses, disjoint shards). Fix: shard in
   `format_calibration_data` before the DataLoader (fork commit ddd4f9f9 with
   mocked-dist regression test). RETROACTIVE IMPLICATION: prior distributed
   GPTQ/AWQ runs that loaded identical datasets on all ranks (our
   pipeline/calibration.py `split[:N]` pattern) effectively calibrated on
   N/world_size unique samples.

## What this does NOT show

- No quality claim beyond smoke coherence (paired-subset eval vs the 4-bit
  arm still pending; iters=200 is modest for 2-bit).
- No performance claim (shared node, first-run JIT, no benchmark).
- DDP scaling caveat: wall-clock per block is constant in world size at fixed
  per-rank batch; ranks multiply effective batch/data, not solve speed.
- Per-rank best-iteration snapshots use local loss (no cross-rank vote);
  rank-0's weights are what get saved.

## Files

- `quant_w2a16_ddp.py` / `launch.sh` — parameterized quant script + launcher
  (env: MODEL, SAVE_DIR, NPROC, NSAMPLES, ITERS)
- `full-run-output.txt` — 8-rank 30B run (gate PASS at end)
- `smoke-run-output.txt` — smoke 1 (0.6B, 2 ranks): CT save API failure +
  identical-loss evidence of bug 2
- `smoke2-run-output.txt` — smoke 2 after fixes: distinct per-rank losses,
  save + gate PASS
- `serve-inhouse-output.txt` — serving smoke PASS on serve-sub4
