# AWQ calibration cost is ~linear in tokens (RETRACTION); and we are using the documented worst-case I/O pattern

2026-08-28. Two findings, both from measurements taken today. The first changes the
production plan for GLM-5.2 (and GLM-5.3). The second explains the storage
behaviour we have been fighting all week and identifies what is actually ours to
fix.

## RETRACTION (same day, later): finding 1 below was WRONG

**The claim "token scaling is decisively sublinear" does not survive measurement.**
Two compounding errors produced it:

1. **The 256-sample run's log was rotated by the kubelet.** Its grid search logged
   134 distinct modules; the 512-sample run logs **257**, matching the
   `Smoothing: /257` progress bar. Module count is a property of the model, not the
   sample count, so 256 must also be 257 -- meaning ~48% of its grid search was cut
   off and the "2m00s" figure below covers only the tail.
2. **The argument defending that figure was also wrong.** It reads "804 lines / 6
   ranks = 134 = one rank's share of ~771 modules". In fact every rank logs all 257
   modules (1542 = 257 x 6). The suspicious coincidence -- the log beginning at
   exactly the first grid-search line -- was noticed and then reasoned past.

Both runs were then re-run with `kubectl logs -f` capturing from process start.
Both show 1542 lines / 257 distinct modules, so neither is truncated:

| | 256 x 2048 | 512 x 2048 | ratio |
|---|---|---|---|
| layer-3 calibrating pass | ~36 s (42-43 batches) | ~66 s (85-86 batches) | 1.83x |
| layer-3 propagating pass | ~3 s | ~5 s | 1.7x |
| **layer-3 grid search** | **220 s** | **407 s** | **1.85x** |
| per-layer total | ~259 s | ~478 s | 1.85x |
| whole 4-layer pipeline | 412 s | 683 s | 1.66x |
| load + dispatch (NVMe subset) | ~21-44 s | ~21-39 s | flat |

So **doubling calibration tokens costs ~1.85x**, i.e. cost is essentially linear
(exponent ~0.89), and the mechanism claimed below -- "the grid search is
independent of token count" -- is false. AWQ's grid search evaluates candidate
smoothing scales against the cached calibration activations, so it scales with
tokens like the forwards do.

Why the original comparison misled: the smoke's 7.4 min for layer 3 at 32x512 was
not a usable baseline. It ran on cephfs rather than NVMe, with 8 ranks rather than
6, and paid per-module disk-offload onload/offload overhead. Treating it as
comparable to a subset run made a storage difference look like a scaling law.

**Corrected projections** (per-layer compute measured, cephfs stream ~120 s/layer):

| config | walk | + load ~53 min, save ~35-80 min | with prefetch + load fix |
|---|---|---|---|
| 256 x 2048 | 8.2 h | **~10.0 h** | **~6.9 h** |
| 512 x 2048 | 13.0 h | ~14.8 h | ~11 h |

Both still fit inside a day, so the recommendation to run 256 x 2048 stands -- but
because it is affordable, NOT because extra tokens are free. Do not read the
sections below as saying calibration data is cheap.

One incidental confirmation of the contention finding: the same 24.17 GiB subset
build took 2m53s, then 5m01s while the smoke's 1457 GB save was running, then
**32 s** once the save had finished. Same work, same code, 9x spread from
filesystem contention alone.

---

## 1. Token scaling is decisively sublinear (SUPERSEDED -- see retraction above)

**Run:** `quant-glm52-awq-20260828t132157z`, exit 0. AWQ, W4AFP8, 6 GPUs,
`--evidence-only`, `--subset-layers 4`. 256 samples x 2048 tokens = **524,288
calibration tokens** against one targeted layer (3, the first MoE layer),
`moe_calibrate_all_experts: true`, reading a 24.17 GiB depth-truncated subset from
node-local NVMe.

**Measured:**

| span | wall |
|---|---|
| subset build (24.17 GiB, 9 source shards, cephfs -> NVMe) | 2m53s |
| pipeline start (13:24:32) -> grid search start (13:30:09) | 5m37s |
| ...of which `Dispatching model` | 41s |
| layer-3 AWQ grid search (13:30:09 -> 13:32:09) | **2m00s** |
| pipeline total (13:24:32 -> 13:32:13) | **7m41s** |
| job total incl. subset build and gates | ~10m26s |

The grid-search span is the complete one, not a fragment of a rotated log: 804
`AWQ grid search for model.layers.3.*` lines over 6 ranks = 134 distinct modules
per rank, which is one rank's share of that layer's ~771 quantized modules.

Quality was healthy: `Error reduction statistics: avg=0.9029, median=0.9087,
min=0.6985, max=0.9708`.

**The comparison.** The AWQ smoke calibrated the same layer 3 at 32 samples x 512
tokens = 16,384 tokens and took **7.4 min** (smoothing 4:22). The probe used **32x
the calibration tokens** and layer 3 cost roughly **6 min** -- slightly *less*.

Linear scaling would have predicted 32 x 7.4 min = **3.9 hours** for that one
layer. We measured about 6 minutes.

**Confound, stated plainly:** the probe changed two things at once -- token count
AND storage (local NVMe vs cephfs), plus 6 ranks vs 8. So this run does not
cleanly isolate the token exponent. It does not need to. The smoke's layer-3 time
contained at most ~150 s of cephfs streaming (18.4 GiB at the measured 125-153
MB/s), so attributing *every* removable second to storage still leaves 32x the
tokens costing about the same wall clock. The gap between 3.9 h and 6 min is three
orders of magnitude too large for the confound to rescue linearity.

**Mechanism** (well-supported by the decomposition, not independently isolated):
AWQ cost splits into two terms with different scaling.
- The activation-statistics forward pass is **linear in tokens**.
- The per-module smoothing-scale **grid search is independent of token count** --
  it is a fixed grid per module, and on a 256-expert MoE layer there are ~771
  modules.

On this architecture the token-independent term dominates, which is why extra
calibration data is nearly free.

**Consequence for the production run.** Per-layer cost measured ~6 min from NVMe.
A production run must read each layer from cephfs (the 1403 GiB model does not fit
on the 642 GB local device), adding the ~2 min/layer stream:

    walk   78 layers x ~8 min        ~= 10.4 h
    load                             ~=  1 h
    save   1457 GB, measured today   ~=  4 h   (degrading; see below)
                                        ------
                                        ~15-16 h

**Under a day at 256 samples x 2048 tokens** -- which is 8x the canonical AWQ
recipe (128 x 512 = 65,536 tokens), not a reduction of it.

This **reverses** the earlier recommendation to cut samples to 96 to reach ~21 h.
That projection extrapolated the smoke's per-layer time linearly in tokens, which
double-counted: it inherited a cephfs-bound per-layer cost *and* multiplied it by
the token ratio. Calibration data volume is not the lever. The load and the save
are.

## 2. We are reading a 1.4 TB model over a network filesystem via mmap

This is the "are we doing something wrong" answer, and it is yes.

**Facts about our mount**, read from inside a running pod rather than from docs:

    10.132.4.63:6789:/ /mnt/cephfs ceph rw,noatime,name=inference-test,acl,
                                        mds_namespace=cephfs-inference-test
    /sys/class/bdi/ceph-11: read_ahead_kb=8192

So: **no `rsize`, no `wsize`, no `rasize`** are set. `sc-file-ceph-ca` carries
`mountOptions: ["noatime"]` and nothing else; readahead is the kernel default 8
MiB. Published CephFS tuning guidance is that raising `rasize`/`rsize` for
sequential streaming workloads can roughly double throughput.

**We cannot change it.** `kubectl auth can-i` returns **no** for
`create storageclass`, `patch storageclass`, and `patch pv`. And the PVC
`model-cache-shared` is bound through four PVs across the `evaluation`, `kernels`
and `training` namespaces, so it is shared infrastructure -- a mount-option change
is an admin request with collaborator impact, not a unilateral tweak.

**The bigger issue is the access pattern, which IS ours.** safetensors loads via
`mmap`, which turns a model read into many small scattered page faults. On a
network filesystem every fault that misses readahead costs a round trip, and our
mon RTT is 28 ms. The literature is unambiguous that this combination is the worst
case, reporting **30-50x** slowdowns, and `fastsafetensors` exists specifically to
replace mmap with aggregated `pread`/GDS transfers (4.8-7.5x on local storage).
Our own numbers sit squarely in that band: **31 MB/s** cephfs single-stream vs
**1632 MB/s** local NVMe single-stream is 53x.

Note what mmap does *not* explain: the sequential walk already achieves 125-153
MB/s per layer, which is at the cephfs 8-reader aggregate ceiling (135-260 MB/s).
So read concurrency is not the gap -- we already run 8 ranks and saturate. The
mmap penalty shows up in single-stream and per-tensor overhead (~60 ms/tensor over
58,794 tensors in the load phase), not in the aggregate walk rate.

**What we already do right:** `dtype: bfloat16` is set explicitly in the configs,
so the widely-cited `torch_dtype="auto"` fix (a reported 18 min -> 2 min, from
avoiding an fp32 upcast) does not apply to us.

**`fastsafetensors` is not a drop-in.** Published integration is with vLLM, not
transformers, and the paper measured local storage only. Adopting it in
llm-compressor's loading path is real work, and the honest ranking below puts
cheaper fixes first.

## Ranked levers, after today

1. **The load reads ~all 59,585 tensors and discards ~99% of them** (`mapping.convert`
   runs before the disk-index skip check). ~53 min/run, entirely ours to fix, and
   it is also the phase where the mmap penalty is worst.
2. **The save writes 1457 GB when ~10 GB changed** -- ~145x amplification, ~4 h,
   and today it *degraded* from 240 s to 600+ s per 50 GB shard for reasons still
   unexplained.
3. **Ask the storage admins** for `rasize`/`rsize` on a CA-region class, or for a
   flash class in `ca-van3`. Cheap to ask, ~2x if granted, not ours to do.
4. **Replace mmap in the loading path** (fastsafetensors or a `pread` reader).
   Largest theoretical win, largest effort, and partly redundant with (1).
5. **Calibration data volume** -- *not* a lever, per finding 1. Do not spend
   quality to buy time that is not there.

Node-local NVMe (measured: `/dev/nvme1n1p2`, ext4, rotational=0, 642 GB free,
1632 MB/s @1 reader flat to 1723 MB/s @16, 921 MB/s write) does not help a
production run: the model does not fit, and writes are a wash against cephfs's 833
MB/s burst. What it buys is ~60x cheaper experiments, which is how finding 1 was
obtained at all -- and it caught two bugs in ~1 minute each that would have cost
~63 min per attempt against the full model.
