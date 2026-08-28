# AWQ calibration cost is NOT linear in tokens; and we are using the documented worst-case I/O pattern

2026-08-28. Two findings, both from measurements taken today. The first changes the
production plan for GLM-5.2 (and GLM-5.3). The second explains the storage
behaviour we have been fighting all week and identifies what is actually ours to
fix.

## 1. Token scaling is decisively sublinear

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
