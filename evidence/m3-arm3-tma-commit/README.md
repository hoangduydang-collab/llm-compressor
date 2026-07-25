# Evidence: Humming TMA store commit-group defect (grouped_contiguous arm 3)

Raw evidence for the fourth Humming defect found while root-causing the
grouped arm's early-EOS pathology in perf window `20260725T122256Z`
(1,594 OSL-mismatch warnings vs ~10 on the CUTLASS/indexed arms;
169/640 early-terminated requests at reasoning conc-64).

**Defect.** `humming/include/humming/utils/ptx/tma.cuh` defines
`tma_commit_store_group()` (`cp.async.bulk.commit_group`) but the codebase
never calls it. PTX `cp.async.bulk.wait_group N` only waits on *committed*
bulk async-groups, so every `tma_wait_store_group` in the kernel is a no-op.
Consequences:

1. The math warps "wait" for the TMA C store then release the producer
   (`consumer.arrive`). The epilogue `reduce` smem buffer lives in a union
   with the producer's stage buffers (`utils/storage.cuh`), so the next
   block's A/B/scale TMA loads overwrite the smem the in-flight C store is
   still reading → intermittent whole-tile garbage (~20–25% of launches in
   the small-M BK=256 buckets that production decode batches hit).
2. Stream-K slice 0 releases its tile lock before its store completes →
   bitwise nondeterminism at BM=8/16.

**Fix.** `pipeline/slurm/patch_humming_tma_store_commit.py` commits each
issued store/reduce into a bulk async-group right after issuance
(CUTLASS's mandatory `tma_store_arrive()` / `tma_store_wait<0>()` pattern).
A second PTX-required fix, `pipeline/slurm/patch_humming_tma_store_fence.py`
(`fence.proxy.async.shared::cta` before TMA stores), was applied first and
measured to NOT be the observed corruption on its own.

## Files

- `prefix-tile-forensics-v1.json` — pre-fix repro
  (`pipeline/m3_humming_grouped_tile_forensics.py`, w13/512 routing,
  BM=32/BK=256): **10/48 launches** with corrupted tiles. The per-row ratio
  test in the tile dumps shows bad tiles are *partially* correct — row
  segments at ratio exactly 1.0000 to the reference adjacent to huge
  pipeline-data garbage — i.e. the store's smem source was overwritten
  mid-read. Run: `m3-arm3-tile-forensics/20260725T161441Z`.
- `fence-only-tile-forensics.json` — with ONLY the fence patch applied:
  **11/48 launches** still bad, proving the missing fence was not the
  observed corruption. Run: `m3-arm3-fence-verify/20260725T161926Z`.
- `postfix-tile-forensics.json` — with the commit-group patch applied:
  **0/96 launches** bad. Run: `m3-arm3-commit-verify/20260725T162957Z`.
- `postfix-scale-probe.json` — full two-geometry bucket sweep
  (`pipeline/m3_humming_grouped_scale_probe.py`) after the fix: every
  bucket `[ok]`, `full==exact` everywhere, and determinism restored at the
  previously nondeterministic w13/w2 BM=8/16 stream-K buckets.
- `postfix-verify-stdout.txt` — the verification run's stdout, including
  the fail-closed `--check` of all four declared patch SHAs before any GPU
  work.

Full runs (launchers, rc files, srun logs) remain under
`/mnt/nfs/hoangduy/results/m3-arm3-{tile-forensics,fence-verify,commit-verify}/`.
Sibling defects: `evidence/m3-arm3-grouped-bounds/` (last-expert shape_m
bound). None of the four Humming defects are fixed in upstream 0.1.11.
