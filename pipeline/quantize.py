"""Quantize stage: load -> oneshot -> sanity-generate -> save compressed.

Produces a vLLM-servable ``pack-quantized`` compressed-tensors checkpoint in
``<run_dir>/checkpoint``.
"""

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from pipeline import metrics, versioning
from pipeline.calibration import (
    CalibrationPartition,
    build_calibration_dataset_with_partition,
    calibration_partition_manifest,
)
from pipeline.config import PipelineConfig
from pipeline.distributed import DistributedContext
from pipeline.provenance import log_model_provenance
from pipeline.recipe import build_recipe, describe_recipe
from pipeline.vl_artifacts import ensure_vl_processor_artifacts

# Schemes whose weights are INT-packed and need explicit pack-quantized on save
# for vLLM to pick the right loader/kernel.
_PACK_QUANTIZED_SCHEMES = {"W4AFP8", "W4A8", "W4A16", "W4A16_ASYM"}


def _process_read_bytes() -> int:
    """Cumulative bytes this process has read (``/proc/self/io``), 0 if unavailable."""
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            if line.startswith("read_bytes:"):
                return int(line.split(":", 1)[1])
    except OSError:
        pass
    return 0


@contextmanager
def _save_heartbeat(ckpt: Path, interval: float = 60.0):
    """Log liveness/progress lines while ``save_pretrained`` runs.

    The disk-offload save has a long silent phase (r11, 2026-07-18): rank 0
    reads every offloaded weight back through the disk-cache index before the
    first shard is written, with zero output for 1h+ — indistinguishable from
    a hang in the logs. This thread prints, every ``interval`` seconds, the
    shard count/bytes written so far plus the process's cumulative read bytes
    so the read-back phase itself is visibly progressing.
    """
    stop = threading.Event()
    start = time.monotonic()
    read0 = _process_read_bytes()

    def _beat() -> None:
        last_written = 0
        while not stop.wait(interval):
            written = 0
            shards = 0
            try:
                for f in ckpt.glob("*.safetensors"):
                    shards += 1
                    written += f.stat().st_size
            except OSError:
                pass  # shard replaced mid-scan; next tick recounts
            read_gb = (_process_read_bytes() - read0) / 1e9
            rate_mb = (written - last_written) / interval / 1e6
            print(
                f"[pipeline] save-heartbeat +{time.monotonic() - start:.0f}s: "
                f"{shards} shards / {written / 1e9:.1f} GB written "
                f"({rate_mb:.0f} MB/s), {read_gb:.1f} GB read back",
                flush=True,
            )
            last_written = written

    thread = threading.Thread(target=_beat, name="save-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


@contextmanager
def _tied_weights_meta_buffer_compat(model):
    """Backport transformers' get_parameter_or_buffer fix for offloaded saves.

    With disk offload the state dict holds meta tensors, and transformers
    5.12.1's ``remove_tied_weights_from_state_dict`` resolves each meta entry
    via ``model.get_parameter(name)`` — which raises AttributeError for
    registered buffers (M3's router ``e_score_correction_bias``), killing the
    save before the first shard (smoke r11, 2026-07-18; ranks then hang in the
    save-wait collective until the PG watchdog fires). Upstream main already
    fixed the branch to call ``model.get_parameter_or_buffer(name)``; shadow
    the instance method with that helper for the duration of the save. No-op
    once the installed transformers carries the fix.
    """
    import inspect

    from transformers.modeling_utils import remove_tied_weights_from_state_dict

    fixed_upstream = "get_parameter_or_buffer" in inspect.getsource(
        remove_tied_weights_from_state_dict
    )
    if fixed_upstream or not hasattr(model, "get_parameter_or_buffer"):
        yield
        return

    # bypass the instance shadow: get_parameter_or_buffer itself calls
    # self.get_parameter, which would recurse into the shim
    cls_get_parameter = type(model).get_parameter

    def _get_parameter_or_buffer(name: str):
        try:
            return cls_get_parameter(model, name)
        except AttributeError:
            return model.get_buffer(name)

    model.get_parameter = _get_parameter_or_buffer
    try:
        yield
    finally:
        model.__dict__.pop("get_parameter", None)


@contextmanager
def _deferred_weight_conversion_compat(model):
    """Backport transformers main's per-shard weight-format revert for offloaded saves.

    transformers 5.12.1 reverts weight-name conversions on the whole state dict
    before the shard loop (modeling_utils.py:3511). With disk offload the state
    dict holds meta tensors, so after that revert every offloaded entry carries
    its checkpoint-format name (e.g. M3's ``language_model.model.*.block_sparse_moe``)
    while ``load_offloaded_parameter`` resolves names against the runtime module
    tree (``model.language_model.*.mlp``) — the first offloaded tensor raises and
    the save dies seconds into "Writing model shards" (smoke r12, 2026-07-18).
    ``WeightConverter`` entries are worse: reverting on meta chunks/concats into
    brand-new meta tensors that nothing can materialize from disk.

    Upstream main fixed this by skipping the early revert when offloaded and
    reverting each shard *after* its tensors are loaded back (modeling_utils.py
    ~3649-3676 on main). This shim backports that behavior without copying the
    400-line save method:

    - ``revert_weight_conversion(model, state_dict)`` becomes a passthrough when
      the dict contains meta tensors (records ``deferred=True``);
    - ``safe_save_file(shard, ...)`` then applies the real revert to each fully
      materialized shard right before writing.

    The index that ``save_pretrained`` writes still maps runtime names, so when
    ``deferred`` is set the caller must run :func:`rebuild_safetensors_index`
    afterwards. Yields a dict whose ``"deferred"`` key reports whether the
    passthrough triggered. Self-disables once the installed transformers does
    per-shard reverts.
    """
    import inspect

    from transformers import modeling_utils as mu

    state = {"deferred": False}
    fixed_upstream = "revert_weight_conversion(model_to_save, shard_state_dict)" in (
        inspect.getsource(mu.PreTrainedModel.save_pretrained)
    )
    if fixed_upstream:
        yield state
        return

    orig_revert = mu.revert_weight_conversion
    orig_save_file = mu.safe_save_file

    def _revert_or_defer(model_to_save, state_dict):
        if model_to_save is model and any(
            t.device.type == "meta" for t in state_dict.values()
        ):
            state["deferred"] = True
            return state_dict
        return orig_revert(model_to_save, state_dict)

    def _save_file_reverted(tensors, filename, metadata=None):
        if state["deferred"]:
            # every tensor in the shard is materialized by now; renames and
            # converter chunk/concat ops operate on real data as upstream does
            tensors = orig_revert(model, tensors)
        return orig_save_file(tensors, filename, metadata=metadata)

    mu.revert_weight_conversion = _revert_or_defer
    mu.safe_save_file = _save_file_reverted
    try:
        yield state
    finally:
        mu.revert_weight_conversion = orig_revert
        mu.safe_save_file = orig_save_file


def rebuild_safetensors_index(ckpt: Path) -> int:
    """Rewrite ``model.safetensors.index.json`` from the actual shard headers.

    Needed after a save under :func:`_deferred_weight_conversion_compat`: the
    per-shard revert renames/splits tensors after ``save_pretrained`` computed
    its weight map, so the written index maps runtime names that no longer
    exist in the shards. Non-index metadata (e.g. ``total_parameters``) is
    preserved; ``total_size`` is recomputed from the headers.

    Returns the number of tensors indexed (0 when the checkpoint is unsharded
    and has no index file).
    """
    index_path = ckpt / "model.safetensors.index.json"
    if not index_path.exists():
        return 0

    import struct

    weight_map: dict[str, str] = {}
    total_size = 0
    for shard in sorted(ckpt.glob("model-*.safetensors")):
        with open(shard, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            weight_map[name] = shard.name
            start, end = info["data_offsets"]
            total_size += end - start

    index = json.loads(index_path.read_text())
    index.setdefault("metadata", {})["total_size"] = total_size
    index["weight_map"] = weight_map
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return len(weight_map)


_PREWARM_CHUNK_BYTES = 32 * 1024 * 1024


def _prewarm_read_file(path: Path) -> int:
    """Sequentially read ``path`` (through symlinks) to pull it into page cache."""
    read = 0
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(_PREWARM_CHUNK_BYTES):
                read += len(chunk)
    except OSError:
        pass  # deleted/broken entry; the save path decides what actually matters
    return read


def prewarm_offload_page_cache(
    offload_dir: Path, max_threads: int = 16
) -> threading.Thread | None:
    """Prefetch every disk-offload file into the OS page cache, in background.

    The offloaded ``save_pretrained`` gather is a single thread doing on-demand
    per-tensor NFS reads (transformers modeling_utils TODO: safetensors holds
    the GIL, so it cannot parallelize) — observed 2h+ on r11 (2026-07-18).
    Parallel sequential prefetch runs at aggregate NFS streaming speed instead,
    and the model (~900 GB) fits in the node's page cache (2 TB RAM), so the
    serial gather then reads from RAM. Read-only: no correctness risk.

    Returns the started controller thread (daemon), or None when disabled via
    ``M3_SAVE_PREWARM=0`` or when ``offload_dir`` has no files.
    """
    from concurrent.futures import ThreadPoolExecutor

    if os.environ.get("M3_SAVE_PREWARM", "1") == "0":
        return None
    try:
        # resolve symlinks (disk cache links unmodified tensors to base shards)
        files = sorted(
            {p.resolve() for p in Path(offload_dir).iterdir() if not p.is_dir()}
        )
    except OSError:
        files = []
    if not files:
        return None

    def _run() -> None:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            total = sum(pool.map(_prewarm_read_file, files))
        elapsed = max(time.monotonic() - started, 1e-6)
        print(
            f"[pipeline] save-prewarm done: {len(files)} files, "
            f"{total / 1e9:.1f} GB in {elapsed:.0f}s "
            f"({total / 1e9 / elapsed:.2f} GB/s)",
            flush=True,
        )

    print(
        f"[pipeline] save-prewarm: prefetching {len(files)} offload files "
        f"into page cache with {max_threads} threads (M3_SAVE_PREWARM=0 disables)",
        flush=True,
    )
    controller = threading.Thread(target=_run, name="save-prewarm", daemon=True)
    controller.start()
    return controller


@contextmanager
def _minimax_meta_rank_config_compat(pretrained_model_cls):
    """Keep compressed-tensors' meta-rank tie setting out of M3 model kwargs.

    compressed-tensors injects ``tie_word_embeddings=False`` on non-source
    ranks so Transformers does not compare meta tensors while tying weights.
    Transformers 5.12 forwards that keyword into MiniMax's sparse model
    constructor, which does not accept it. Install this wrapper *under* the
    compressed-tensors loader so the setting is retained on the composite
    config but removed before model construction.
    """
    original_descriptor = vars(pretrained_model_cls)["from_pretrained"]
    original = pretrained_model_cls.from_pretrained.__func__

    @classmethod
    @wraps(original)
    def from_pretrained(cls, *args, **kwargs):
        config = kwargs.get("config")
        if (
            "tie_word_embeddings" in kwargs
            and getattr(config, "model_type", None) == "minimax_m3_vl"
        ):
            kwargs = dict(kwargs)
            tie_word_embeddings = kwargs.pop("tie_word_embeddings")
            config.tie_word_embeddings = tie_word_embeddings
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                text_config.tie_word_embeddings = tie_word_embeddings
        return original(cls, *args, **kwargs)

    pretrained_model_cls.from_pretrained = from_pretrained
    try:
        yield
    finally:
        pretrained_model_cls.from_pretrained = original_descriptor


def _log_backbone_dtype(model) -> None:
    """Log resolved backbone dtype so fp32 linearize regressions are visible in logs."""
    text_cfg = getattr(model.config, "text_config", None)
    text_dtype = getattr(text_cfg, "dtype", None) if text_cfg is not None else None
    sample_param_dtype = None
    for name, module in model.named_modules():
        if ".mlp.experts." in name and name.endswith(".down_proj"):
            param = getattr(module, "weight", None)
            if param is not None:
                sample_param_dtype = param.dtype
                break
    print(
        "[pipeline] backbone dtype: "
        f"text_config.dtype={text_dtype} "
        f"sample_expert_weight.dtype={sample_param_dtype}"
    )


def _load_model_and_tokenizer(cfg: PipelineConfig):
    import transformers
    from transformers import AutoTokenizer, PreTrainedModel

    from llmcompressor.utils import load_context
    from pipeline.minimax_m3_config import apply_minimax_m3_config

    m = cfg.model
    model_cls = getattr(transformers, m.auto_class)
    from_pretrained_kwargs: dict = {"trust_remote_code": m.trust_remote_code}
    if m.dtype and m.dtype != "auto":
        from_pretrained_kwargs["dtype"] = m.dtype
    if m.device_map is not None:
        from_pretrained_kwargs["device_map"] = m.device_map
    if m.offload_folder is not None:
        from_pretrained_kwargs["offload_folder"] = m.offload_folder
    if m.max_memory is not None:
        # YAML may give strings like "500e9"; coerce to float.
        from_pretrained_kwargs["max_memory"] = {
            k: float(v) for k, v in m.max_memory.items()
        }

    from_pretrained_kwargs = apply_minimax_m3_config(
        m.id, from_pretrained_kwargs, trust_remote_code=m.trust_remote_code
    )

    # load_context() patches from_pretrained so fused MoE experts load in a
    # linearized, quantizable layout (and handles offloaded loading).
    # This compatibility wrapper must be entered before load_context so it sits
    # below compressed-tensors' non-source/meta-rank from_pretrained wrapper.
    with _minimax_meta_rank_config_compat(PreTrainedModel), load_context(model_cls):
        model = model_cls.from_pretrained(m.id, **from_pretrained_kwargs)
    _log_backbone_dtype(model)
    tokenizer = AutoTokenizer.from_pretrained(
        m.id, trust_remote_code=m.trust_remote_code
    )
    return model, tokenizer


def _persist_ignore_to_config(ckpt: Path, ignore: list[str]) -> None:
    """Ensure the recipe's ignore patterns survive into the saved config.

    llm-compressor prunes ignore patterns that didn't match a *quantized* module
    from the serialized ``quantization_config.ignore``. That silently drops
    entries for modules it (correctly) left unquantized -- e.g. the MoE router
    ``mlp.gate`` (and, for VL MoE, the vision tower / MSA indexer). Downstream
    loaders (vLLM) then treat those Linears as quantized and either fail to load
    or, worse, mis-load them -> broken routing -> garbage output. We re-add the
    intended ignore patterns so the on-disk config reflects what was actually
    skipped.
    """
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        return
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    qc = data.get("quantization_config")
    if not qc:
        return
    saved = list(qc.get("ignore", []))
    added = [p for p in ignore if p not in saved]
    if added:
        qc["ignore"] = saved + added
        with cfg_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"[pipeline] persisted ignore patterns to config: {added}")


def _sample_generation(model, tokenizer, prompt: str) -> str:
    from compressed_tensors.offload import dispatch_model

    dispatch_model(model)
    sample = tokenizer(prompt, return_tensors="pt")
    sample = {k: v.to(model.device) for k, v in sample.items()}
    output = model.generate(**sample, max_new_tokens=64)
    return tokenizer.decode(output[0])


def _evidence_paths(run_dir: Path, dist_ctx: DistributedContext) -> dict[str, Path]:
    return {
        "metrics": dist_ctx.rank_path(run_dir / "quant_metrics.jsonl"),
        "provenance": dist_ctx.rank_path(run_dir / "model_provenance.json"),
        "partition": dist_ctx.rank_path(run_dir / "calibration_partition.json"),
    }


def _persist_calibration_partition(
    run_dir: Path,
    dataset,
    partition: CalibrationPartition,
    dist_ctx: DistributedContext,
) -> Path:
    path = _evidence_paths(run_dir, dist_ctx)["partition"]
    manifest = calibration_partition_manifest(dataset, partition)
    manifest["distributed"] = dist_ctx.snapshot()
    path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def install_distributed_disk_update_offload_patch() -> bool:
    """Make ``DistributedDiskCache.update_offload`` distributed-safe.

    Upstream compressed-tensors (0.17.2a20260707, and main as of 2026-07-18)
    overrides ``offload``/``__delitem__`` with source-rank gating but inherits
    ``DiskCache.update_offload`` unchanged, so during AWQ smoothing every rank
    concurrently ``os.unlink``s and rewrites the SAME shared index file --
    a race that killed smoke r10 with ``FileNotFoundError`` on the shared
    ``ct_disk_cache_*.safetensors`` (and silently corrupts when it doesn't
    crash). All ranks compute identical smoothed data (activation stats are
    synchronized), so gate the write to the source rank and barrier, mirroring
    ``DistributedDiskCache.offload``. The file path is unchanged by the
    rewrite, so non-source index entries stay valid.

    :return: True if the patch was installed, False if upstream already
        defines a distributed ``update_offload`` (drop this patch then).
    """
    import torch.distributed as dist
    from compressed_tensors.distributed import is_source_process
    from compressed_tensors.offload.cache.disk import DiskCache
    from compressed_tensors.offload.cache.dist_disk import DistributedDiskCache

    if "update_offload" in vars(DistributedDiskCache):
        return False

    def update_offload(self, offloaded, data):
        if is_source_process():
            DiskCache.update_offload(self, offloaded, data)
        if dist.is_available() and dist.is_initialized():
            # writers-before-readers: no rank may onload until the write lands
            dist.barrier()

    DistributedDiskCache.update_offload = update_offload
    return True


# VMA headroom reserved for everything that is NOT a shared-weights segment:
# CUDA contexts, glibc/allocator arenas, loaded libraries, and the per-layer
# calibration activation cache (each >128KiB cpu tensor is its own mmap).
_VMA_GUARD_SLACK = 24_000

# The offloaded module tree holds MORE tensors than the checkpoint index:
# load_context linearizes fused MoE expert tensors into per-expert quantizable
# Linears. Empirical anchor: M3's 23,416-entry index produced 63,122 shm
# segments in run 20260717T064357Z-m3-ddp-awq-full-r1 (2.70x). Round up to 3x
# so the estimate stays conservative for this model family.
_VMA_LINEARIZATION_FACTOR = 3.0


def estimate_shared_offload_segments(
    index_json: Path,
    cpu_budget_bytes: float,
    expansion_factor: float = _VMA_LINEARIZATION_FACTOR,
) -> tuple[int, int, int]:
    """Estimate the per-process VMA demand of distributed shared-CPU offload.

    ``DistributedCPUCache`` creates one ``/dev/shm`` segment per offloaded
    tensor and every rank mmaps all of them, so the plan's VMA demand is the
    number of module-tree tensors that fit in the cpu budget (accelerate fills
    devices sequentially, so the fitting fraction approximates the count).
    ``expansion_factor`` scales the checkpoint-index tensor count up to the
    post-linearization module tree (see ``_VMA_LINEARIZATION_FACTOR``).

    :return: (planned shm segments, total checkpoint tensors, total bytes)
    """
    data = json.loads(index_json.read_text())
    n_tensors = len(data["weight_map"])
    total_bytes = int(data["metadata"]["total_size"])
    n_offloaded = int(n_tensors * expansion_factor)
    if cpu_budget_bytes >= total_bytes:
        planned = n_offloaded
    else:
        planned = int(n_offloaded * (cpu_budget_bytes / total_bytes))
    return planned, n_tensors, total_bytes


def _offloaded_save_health(save_pretrained_source: str) -> str:
    """Classify the installed save_pretrained for offloaded sharded saves.

    - ``"shimmed"``: pre-5.14, no per-shard revert — our save shims
      (`_tied_weights_meta_buffer_compat`, `_deferred_weight_conversion_compat`)
      own the path (proven by smoke r13).
    - ``"healthy"``: per-shard revert present and the sharded weight-map
      bookkeeping is sane.
    - ``"broken"``: per-shard revert present but the weight-map update is the
      generator-of-dicts form shipped in 5.14.0/5.14.1 (and upstream main as
      of 2026-07-18): ``weight_map.update({k: basename} for k in ...)`` feeds
      ``dict.update`` 1-element dicts → ValueError, masked by the broad
      except as the "unlucky sharding" RuntimeError — every sharded offloaded
      original-format save crashes at the end of shard 1.
    """
    if "revert_weight_conversion(model_to_save, shard_state_dict)" not in (
        save_pretrained_source
    ):
        return "shimmed"
    if "} for k in shard_state_dict.keys()" in save_pretrained_source:
        return "broken"
    return "healthy"


def assert_transformers_offloaded_save_healthy() -> None:
    """Fail-closed gate: refuse to start a run whose offloaded save is
    known-broken, instead of crashing hours later at the end of shard 1.

    transformers 5.14.x needs a one-line venv hotfix to its sharded
    weight-map update (see BUGS_AND_FIXES.md, "Transformers 5.14.1 upgrade");
    this gate catches a venv rebuild that silently dropped the hotfix.
    """
    import inspect

    from transformers import modeling_utils as mu

    health = _offloaded_save_health(
        inspect.getsource(mu.PreTrainedModel.save_pretrained)
    )
    if health == "broken":
        raise RuntimeError(
            "transformers save_pretrained carries the per-shard weight-format "
            "revert, but its sharded weight_map update is the known-broken "
            "generator-of-dicts form — every sharded offloaded save crashes "
            "after shard 1. Re-apply the one-line hotfix to "
            f"{mu.__file__} (see BUGS_AND_FIXES.md 'Transformers 5.14.1 "
            "upgrade') or install transformers<=5.12.1 (save shims cover it)."
        )
    print(f"[pipeline] offloaded-save gate OK: transformers path is {health}")


def assert_smooth_fold_consistency(
    ckpt: Path,
    base: Path,
    layers: list[int],
    threshold: float = 0.02,
    scale_mean_range: tuple[float, float] = (0.05, 20.0),
) -> None:
    """Fail-closed post-save gate: a smoothing fold applied to balance layers
    (router / shared experts) must be matched by the inverse fold in the norm,
    AND the fold itself must be bounded.

    The magnitude bound exists because a fold can be perfectly
    self-consistent yet still fatal: full-run r4 (2026-07-19) hit the AWQ
    dead-channel scale degeneracy, folding a uniform ~x128 into layers
    8/10-13 — norm weights landed at ~-0.992 where bf16 cannot resolve
    per-channel gains. Healthy folds have norm-implied scale means of
    0.7-0.9; anything outside ``scale_mean_range`` is a degenerate fold.
    Layers that were not smoothed audit as scale == 1 and pass trivially,
    so the gate is safe to run over every layer in partial-layer smokes.

    The full-calibration AWQ run r2 (2026-07-18) saved a checkpoint whose
    routers carried the per-channel smoothing multiply while the offset norm
    kept base values — ``CalibrationOffsetNorm.restore`` wrote via raw
    ``.data`` assignment, which the disk OffloadCache never persisted. The
    static serving-ABI gate cannot see numerics; this gate re-derives the
    norm-implied scale from the saved checkpoint and requires it to explain
    the router and shared-expert weight changes.

    Reference magnitudes (relative L2, layers 3/31/59): consistent AWQ fold
    (r9) ≈ 3e-3; no smoothing (GPTQ) ≈ 2e-3; lost norm fold (r2) 0.09–0.27.

    Skips (with a printed reason) when the checkpoint does not expose
    M3-style norm/router tensor names.
    """
    from pipeline.m3_checkpoint_scale_audit import audit_checkpoint

    try:
        report = audit_checkpoint(Path(base), Path(ckpt), layers)
    except (ValueError, KeyError, FileNotFoundError) as err:
        print(f"[pipeline] smooth-fold gate skipped (names not resolvable): {err}")
        return

    failures = []
    lo, hi = scale_mean_range
    for layer, record in report["layers"].items():
        for component in ("router_compensation", "shared_gate_up_compensation"):
            err = record[component]["relative_l2_error"]
            # `not <=` so NaN (e.g. non-finite prediction) fails closed.
            if not (err <= threshold):
                failures.append(f"layer {layer} {component}: rel_l2={err:.3e}")
            scale_stats = record[component]["scale"]
            scale_mean = scale_stats["mean"]
            if scale_stats["finite_fraction"] < 1.0:
                failures.append(
                    f"layer {layer} {component}: non-finite norm-implied scale"
                )
            elif scale_mean is None or not (lo <= scale_mean <= hi):
                failures.append(
                    f"layer {layer} {component}: degenerate fold, "
                    f"scale mean={scale_mean} outside [{lo}, {hi}] "
                    "(healthy ~0.7-0.9; dead-channel degeneracy ~128)"
                )
    if failures:
        raise RuntimeError(
            "smooth-fold consistency gate FAILED — balance-layer weights moved "
            "in ways the saved norm cannot explain (lost or partial smoothing "
            "fold; checkpoint numerics are inconsistent):\n  "
            + "\n  ".join(failures)
            + f"\n  threshold={threshold} (consistent fold ≈ 3e-3; see "
            "BUGS_AND_FIXES.md 'AWQ smoothing fold lost under disk offload')"
        )
    print(
        "[pipeline] smooth-fold gate OK: norm-implied scale explains balance "
        f"layers on layers {sorted(int(k) for k in report['layers'])}"
    )


def assert_quant_checkpoint_verified(ckpt: Path, base: Path | None = None) -> None:
    """Fail-closed post-save gate: structural + sampled tensor verification.

    Runs ``pipeline.verify_quant_checkpoint.verify(ckpt, check_tensors=True,
    dequant_base=base)``: structure (ignore shapes, swiglu scalars, expert
    layout), sampled finiteness of ``weight_scale`` / ``weight_packed``, and —
    when ``base`` is given — value-level dequant-vs-base agreement (fitted
    per-column smoothing scale, W4-error residual). The full-calibration AWQ
    run r3 (2026-07-19) saved uninitialized weight scales for ~7/8 of
    quantized modules — the distributed qparam broadcast filled a temporary
    onload the disk OffloadCache never persisted — and every earlier gate
    (ABI, smooth-fold, save-health) passed because none of them read the
    quantized tensors. Non-owner corruption hits ~87.5% of modules, so a
    20-module sample cannot miss it; the dequant check additionally catches
    any transform the saved tensors cannot explain (garbage packed values,
    wrong scales that happen to be finite, lost per-column multiplies).
    """
    from pipeline.verify_quant_checkpoint import verify

    dequant_base = base if base is not None and Path(base).is_dir() else None
    if base is not None and dequant_base is None:
        print(f"[pipeline] dequant check skipped (base not a local dir): {base}")
    rc = verify(Path(ckpt), check_tensors=True, dequant_base=dequant_base)
    if rc != 0:
        raise RuntimeError(
            "quant checkpoint verification gate FAILED (see [FAIL] lines "
            "above) — do not serve or evaluate this checkpoint; see "
            "BUGS_AND_FIXES.md 'distributed qparam broadcast lost under "
            "disk offload'"
        )
    print("[pipeline] quant-verify gate OK: structure + sampled tensors healthy")


def assert_vma_budget_for_shared_offload(
    cfg: PipelineConfig,
    dist_ctx: DistributedContext,
    *,
    _max_map_count: int | None = None,
    _shm_total_bytes: float | None = None,
) -> None:
    """Fail-closed gate: refuse a distributed shared-CPU offload plan whose
    per-tensor shm segment count would approach ``vm.max_map_count``.

    Run 20260717T064357Z-m3-ddp-awq-full-r1 died at 63,122 shm segments
    against the default 65,530 VMA cap (rank-0 ENOMEM before calibration,
    3h NCCL broadcast timeout, no checkpoint). Weights-side sibling of the
    pin_memory VMA incident in BUGS_AND_FIXES.md. Cap ``model.max_memory.cpu``
    so weights overflow to disk offload, or raise the sysctl, to pass.
    Set M3_SKIP_VMA_GUARD=1 to bypass explicitly.
    """
    if not dist_ctx.enabled:
        return
    if os.environ.get("M3_SKIP_VMA_GUARD") == "1":
        print("[pipeline] M3_SKIP_VMA_GUARD=1: skipping VMA budget gate")
        return
    # getattr: contract tests drive run_quantize with minimal cfg stand-ins
    if getattr(cfg.model, "device_map", None) != "auto_offload":
        return
    index_json = Path(cfg.model.id) / "model.safetensors.index.json"
    if not index_json.is_file():
        print(
            "[pipeline] VMA gate: no local safetensors index at "
            f"{index_json}; cannot estimate segment count, skipping"
        )
        return

    # Mirror compressed-tensors load.py: an explicit max_memory.cpu wins;
    # otherwise auto_offload budgets the whole of /dev/shm in distributed mode.
    max_memory = cfg.model.max_memory or {}
    if "cpu" in max_memory:
        budget = float(max_memory["cpu"])
    elif _shm_total_bytes is not None:
        budget = _shm_total_bytes
    else:
        budget = float(shutil.disk_usage("/dev/shm").total)

    if _max_map_count is not None:
        limit = _max_map_count
    else:
        limit = int(Path("/proc/sys/vm/max_map_count").read_text())

    planned, n_tensors, total_bytes = estimate_shared_offload_segments(
        index_json, budget
    )
    if planned + _VMA_GUARD_SLACK > limit:
        raise RuntimeError(
            "Distributed shared-CPU offload plan exceeds the VMA budget: "
            f"~{planned} shm segments (of {n_tensors} checkpoint tensors, "
            f"{total_bytes / 1e9:.0f} GB) + {_VMA_GUARD_SLACK} slack > "
            f"vm.max_map_count={limit}. Every rank mmaps every shared "
            "segment, so this plan fails with ENOMEM mid-load and wastes the "
            "allocation (see BUGS_AND_FIXES.md, DDP weights-side VMA "
            "exhaustion, 2026-07-17). Remedies: cap model.max_memory.cpu "
            "(e.g. 32e9) so weights overflow to disk offload, or have a node "
            "admin raise vm.max_map_count (>=1048576). "
            "Set M3_SKIP_VMA_GUARD=1 to bypass."
        )
    print(
        f"[pipeline] VMA gate OK: ~{planned} planned shm segments + "
        f"{_VMA_GUARD_SLACK} slack <= vm.max_map_count={limit}"
    )


def run_quantize(
    cfg: PipelineConfig,
    run_dir: Path,
    dist_ctx: DistributedContext | None = None,
    *,
    save_checkpoint: bool = True,
) -> Path:
    """Execute the quantize stage. Returns the checkpoint directory."""
    from llmcompressor import oneshot
    from pipeline.minimax_m3_config import (
        ensure_minimax_m3_vllm_serve_config,
        patch_minimax_m3_for_text_calibration,
        register_minimax_m3_awq_mappings,
    )

    dist_ctx = dist_ctx or DistributedContext()
    evidence_paths = _evidence_paths(run_dir, dist_ctx)

    assert_transformers_offloaded_save_healthy()
    assert_vma_budget_for_shared_offload(cfg, dist_ctx)
    if dist_ctx.enabled and install_distributed_disk_update_offload_patch():
        print(
            "[pipeline] patched DistributedDiskCache.update_offload "
            "(source-rank write + barrier; see BUGS_AND_FIXES.md)"
        )
    model, tokenizer = _load_model_and_tokenizer(cfg)
    # Capture load/environment provenance BEFORE calibration: where the loaded
    # modeling code comes from (installed transformers vs trust_remote_code) and
    # whether sequential_targets match any module. A zero match count is the
    # direct cause of the sequential-trace collapse (single subgraph -> no
    # calibration -> un-smoothed weights). Written next to the checkpoint.
    log_model_provenance(
        model,
        cfg.calibration.sequential_targets,
        out_path=evidence_paths["provenance"],
    )
    if patch_minimax_m3_for_text_calibration(model):
        print(
            "[pipeline] patched MiniMax-M3 get_placeholder_mask "
            "for text-only calibration"
        )
        register_minimax_m3_awq_mappings()
        print("[pipeline] registered MiniMax-M3 AWQ mappings")
        if os.environ.get("M3_AWQ_GATE_ALPHA_FOLD", "0").lower() in {"1", "true", "yes"}:
            # r7 gate-alpha fold: the gate->down mapping is only function-
            # preserving with per-expert alpha/limit co-scaling attached
            # (pipeline/m3_gate_alpha_fold.py). linearize_moe is idempotent —
            # oneshot would run it anyway; running it here lets us attach the
            # consumers before calibration starts. Fail closed if nothing was
            # prepared.
            from llmcompressor.modeling.moe.linearize import linearize_moe
            from pipeline.m3_gate_alpha_fold import (
                assert_gate_alpha_fold_ready,
                attach_minimax_m3_gate_alpha_fold,
            )

            linearize_moe(model)
            prepared = attach_minimax_m3_gate_alpha_fold(model)
            assert_gate_alpha_fold_ready(model, prepared)
            print(
                f"[pipeline] gate-alpha fold prepared on {prepared} experts "
                "(per-expert, per-channel scales; alpha/limit co-scaling active)"
            )

    ds, partition = build_calibration_dataset_with_partition(cfg.calibration, tokenizer)
    _persist_calibration_partition(run_dir, ds, partition, dist_ctx)
    recipe = build_recipe(cfg.quantization)

    oneshot_kwargs: dict = dict(
        model=model,
        # Text-only calibration: pass the loaded tokenizer so oneshot does not
        # AutoProcessor.from_pretrained (M3 needs trust_remote_code for that).
        processor=tokenizer,
        trust_remote_code_model=cfg.model.trust_remote_code,
        dataset=ds,
        recipe=recipe,
        max_seq_length=cfg.calibration.max_seq_length,
        num_calibration_samples=(
            len(ds) if dist_ctx.enabled else cfg.calibration.num_samples
        ),
        moe_calibrate_all_experts=cfg.calibration.moe_calibrate_all_experts,
    )
    if dist_ctx.enabled:
        # Dataset shuffling and global-to-rank partitioning already happened in
        # pipeline.calibration. Keep llm-compressor from shuffling each shard.
        oneshot_kwargs["shuffle_calibration_samples"] = False
    if cfg.calibration.sequential_targets:
        oneshot_kwargs["sequential_targets"] = cfg.calibration.sequential_targets
    if cfg.calibration.pipeline:
        oneshot_kwargs["pipeline"] = cfg.calibration.pipeline

    # Capture llm-compressor's internal METRIC-level logs (GPTQ error/time, etc.)
    # to a per-run JSONL alongside the checkpoint.
    metrics_path = evidence_paths["metrics"]
    with metrics.capture_quant_metrics(metrics_path):
        oneshot(**oneshot_kwargs)

    ckpt = versioning.checkpoint_dir(run_dir)
    if save_checkpoint:
        # compressed-tensors distributed saving is collective: every rank calls
        # model.save_pretrained, then only rank zero writes shared side artifacts.
        save_kwargs: dict = {"save_compressed": True}
        if cfg.quantization.scheme in _PACK_QUANTIZED_SCHEMES:
            save_kwargs["quantization_format"] = "pack-quantized"
        if dist_ctx.is_source or not dist_ctx.enabled:
            if cfg.model.offload_folder is not None:
                # overlaps with the gather: prefetched files hit page cache
                prewarm_offload_page_cache(Path(cfg.model.offload_folder))
            print(
                f"[pipeline] saving checkpoint to {ckpt} — with disk offload the "
                "first shard can take 1h+ (offloaded weights read back first); "
                "heartbeat below every 60s",
                flush=True,
            )
            with (
                _tied_weights_meta_buffer_compat(model),
                _deferred_weight_conversion_compat(model) as deferral,
                _save_heartbeat(ckpt),
            ):
                model.save_pretrained(str(ckpt), **save_kwargs)
            if deferral["deferred"]:
                n_indexed = rebuild_safetensors_index(ckpt)
                print(
                    "[pipeline] offloaded save deferred weight-format revert to "
                    f"per-shard; rebuilt safetensors index ({n_indexed} tensors)",
                    flush=True,
                )
        else:
            # Non-source ranks hold meta tensors and mostly wait in collectives;
            # a heartbeat there would be 8x duplicate noise.
            with (
                _tied_weights_meta_buffer_compat(model),
                _deferred_weight_conversion_compat(model),
            ):
                model.save_pretrained(str(ckpt), **save_kwargs)
        dist_ctx.barrier()

        if dist_ctx.is_source:
            tokenizer.save_pretrained(str(ckpt))

            # vLLM VL load needs image-processor configs; tokenizer.save_pretrained
            # alone does not write preprocessor_config.json.
            if cfg.model.auto_class == "AutoModelForImageTextToText":
                added = ensure_vl_processor_artifacts(
                    ckpt,
                    cfg.model.id,
                    trust_remote_code=cfg.model.trust_remote_code,
                )
                if added:
                    print(f"[pipeline] saved VL processor artifacts: {added}")

                cfg_patches = ensure_minimax_m3_vllm_serve_config(ckpt, cfg.model.id)
                if cfg_patches:
                    print(
                        f"[pipeline] patched saved config for vLLM serve: {cfg_patches}"
                    )

            # Preserve intended ignore patterns for downstream loaders.
            _persist_ignore_to_config(ckpt, cfg.quantization.ignore)
            versioning.write_recipe(run_dir, describe_recipe(cfg.quantization))
            print(f"[pipeline] saved checkpoint to {ckpt}")
        dist_ctx.barrier()

        # After the barrier so a gate failure on the source rank cannot strand
        # other ranks in a collective (the r11/r12 zombie choreography).
        if dist_ctx.is_source or not dist_ctx.enabled:
            # Default to every M3 MoE layer: r4's degenerate folds sat on
            # layers 8/10-13, which a 3/31/59 spot-check cannot see. Layers
            # that were not smoothed audit as scale == 1 and pass trivially.
            gate_layers_env = os.environ.get("M3_DIAGNOSTIC_LAYERS", "")
            if gate_layers_env.strip():
                gate_layers = [
                    int(part) for part in gate_layers_env.split(",") if part.strip()
                ]
            else:
                gate_layers = list(range(3, 60))
            assert_smooth_fold_consistency(ckpt, Path(cfg.model.id), gate_layers)
            assert_quant_checkpoint_verified(ckpt, Path(cfg.model.id))
    else:
        # A partial-layer smoke is evidence only. The completion marker appears
        # only after every rank finishes calibration and reaches this barrier.
        dist_ctx.barrier()
        if dist_ctx.is_source:
            (run_dir / "smoke_complete.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "checkpoint_saved": False,
                        "distributed": dist_ctx.snapshot(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        dist_ctx.barrier()

    # Distributed generation is not part of calibration and can require a
    # different dispatch topology. Keep the existing local check only.
    if cfg.quantization.sample_generation and save_checkpoint and not dist_ctx.enabled:
        print("\n========== SAMPLE GENERATION ==========")
        try:
            print(_sample_generation(model, tokenizer, cfg.serve.prompt))
        except Exception as exc:  # generation issues should not lose the checkpoint
            print(f"[warn] sample generation failed: {exc}")
        print("=======================================\n")

    # Summarize the captured internal metrics into metadata.json.
    summary = metrics.summarize_quant_metrics(metrics_path)
    if dist_ctx.is_source:
        versioning.update_metadata(run_dir, {"quant_metrics": summary})
    print(f"[pipeline] quant metrics: {summary}")

    return ckpt
