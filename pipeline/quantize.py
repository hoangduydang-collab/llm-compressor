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

    SINCE 2026-08-28 the patterns are RESOLVED AGAINST THE SAVED TENSORS rather
    than written verbatim, because writing them verbatim was itself an r8-class
    defect. A recipe pattern says "the int4 modifier must not touch this"; the
    config's ignore list says "no loader should treat this as quantized". Those
    are different statements, and ``re:.*self_attn[.].*`` is correct as the first
    and fatal as the second -- the FP8 modifier owns those same modules by
    explicit target, and loaders check ignore BEFORE targets.

    Measured on the real GLM-5.2 AWQ checkpoint (routerfix, 20260828-150142)
    before this fix: 16 of 784 quantized modules were shadowed, and those 16 were
    the ENTIRE FP8 leg -- all 10 MLA projections, all 3 shared experts, all 3
    dense MLPs. Not one would have served quantized. Same mechanism that made M3
    r8 emit garbage at exit code 0 (2026-07-24); see pipeline/serve_ignore.py.
    """
    from pipeline.serve_ignore import checkpoint_modules, resolve_ignore_patterns

    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        return
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    qc = data.get("quantization_config")
    if not qc:
        return
    saved = list(qc.get("ignore", []))
    wanted = [p for p in ignore if p not in saved]
    if not wanted:
        return

    try:
        modules, quantized = checkpoint_modules(ckpt)
    except (FileNotFoundError, KeyError, ValueError) as err:
        # Fail loud rather than fall back to the verbatim write: the verbatim
        # write is the bug, and a checkpoint whose ignore list was never checked
        # against its tensors must not look like one that was.
        raise RuntimeError(
            "cannot persist ignore patterns safely: the saved checkpoint's "
            f"tensor names are not readable ({err}). Writing recipe patterns "
            "unchecked is what shadowed the entire FP8 leg on M3 r8."
        ) from err

    added, report = resolve_ignore_patterns(wanted, modules, quantized)
    qc["ignore"] = saved + [entry for entry in added if entry not in saved]
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    concrete = [entry for entry in added if not entry.startswith("re:")]
    print(
        f"[pipeline] persisted ignore to config: {len(added)} entries "
        f"({len(added) - len(concrete)} patterns, {len(concrete)} concrete)"
    )
    for pattern, record in report.items():
        if record.get("overflow"):
            print(
                f"[pipeline] WARNING: ignore pattern {pattern!r} shadows "
                f"{record['shadowed_count']} quantized modules and expands to "
                f"{record['replaced_with_count']} concrete entries, over the cap. "
                "DROPPED. This checkpoint is not serve-ready: the catch-all "
                "config group will claim modules that carry no quant tensors. "
                "Expected on partial-scope smokes, which are not served."
            )
        else:
            print(
                f"[pipeline] ignore pattern {pattern!r} would have shadowed "
                f"{record['shadowed_count']} QUANTIZED modules "
                f"(e.g. {record['shadowed'][:3]}); replaced with "
                f"{record['replaced_with_count']} concrete unquantized entries"
            )


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


def _stamp_mixed_precision_formats(model) -> dict[str, int]:
    """Per-scheme compression formats for mixed int4+FP8 checkpoints (r8).

    ``infer_model_format`` respects ``scheme.format`` when set: stamp 4-bit
    int weight groups as ``pack-quantized`` (what vLLM's W4A8 CUTLASS path
    loads) and 8-bit float weight groups as ``float-quantized`` (plain fp8
    ``weight`` + per-channel ``weight_scale``). Returns per-format module
    counts for logging. Idempotent; schemes without weights are skipped.
    """
    from compressed_tensors.config import CompressionFormat
    from compressed_tensors.quantization.utils import is_module_quantized

    counts: dict[str, int] = {}
    for module in model.modules():
        if not is_module_quantized(module):
            continue
        weights = getattr(module.quantization_scheme, "weights", None)
        if weights is None:
            continue
        if weights.num_bits == 4 and str(weights.type) == "int":
            fmt = CompressionFormat.pack_quantized
        elif weights.num_bits == 8 and str(weights.type) == "float":
            fmt = CompressionFormat.float_quantized
        else:
            continue
        module.quantization_scheme.format = fmt.value
        counts[fmt.value] = counts.get(fmt.value, 0) + 1
    return counts


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


# Sampling knobs whose non-default presence makes transformers' strict
# validation demand do_sample=True. Values are the transformers defaults; a
# config left at the default does not trip validation.
_SAMPLING_DEFAULTS = {
    "top_p": 1.0,
    "top_k": 50,
    "typical_p": 1.0,
    "min_p": None,
    "temperature": 1.0,
}


def repair_generation_config(model) -> list[str]:
    """Make ``model.generation_config`` survive transformers strict validation.

    WHY THIS EXISTS. GLM-5.2 ships a ``generation_config.json`` written by
    transformers 5.12.0 that sets ``top_p: 0.95`` with no ``do_sample``. 5.12's
    validation tolerated it; 5.14's ``save_pretrained`` calls
    ``validate(strict=True)`` and raises::

        ValueError: GenerationConfig is invalid:
        - `top_p`: `do_sample` is not set to `True`. However, `top_p` is set to
          `0.95` -- this flag is only used in sample-based generation modes.

    It fires at the very END of the save, so both GLM-5.2 smoke arms burned
    ~5-6 h of 4-GPU calibration each and wrote no checkpoint. Pinning to 5.14.1
    to match the M3 environment is what surfaced it -- the same shape as the
    sharded-save hotfix in envs/.

    WHY do_sample=True RATHER THAN DROPPING top_p. Both silence the error, but
    unsetting ``top_p`` would change the checkpoint's default decoding relative
    to the official model, silently, for everyone who loads it downstream.
    ``temperature: 1.0`` + ``top_p: 0.95`` is unambiguously a sampling recipe,
    so enabling sampling is the semantics-preserving repair; dropping the
    parameter is not.

    Fail-closed: if the config still will not validate, raise rather than let
    the save die hours later, and rather than shipping a config we silently
    mangled.

    :param model: model about to be saved.
    :return: human-readable descriptions of the repairs made (empty if none).
    """
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return []

    try:
        generation_config.validate(strict=True)
        return []
    except Exception:
        pass  # needs repair; the specific complaint is handled below

    changes: list[str] = []
    if not getattr(generation_config, "do_sample", False):
        active = sorted(
            name
            for name, default in _SAMPLING_DEFAULTS.items()
            if getattr(generation_config, name, default) not in (default, None)
        )
        if active:
            generation_config.do_sample = True
            changes.append(
                "do_sample: False -> True (config sets sampling params: "
                f"{', '.join(active)})"
            )

    try:
        generation_config.validate(strict=True)
    except Exception as exc:
        raise RuntimeError(
            "model.generation_config cannot be made to pass transformers' "
            f"strict validation. Repairs applied: {changes or 'none'}. "
            f"Remaining problem: {exc}. Refusing to start rather than failing "
            "at the end of a multi-hour save."
        ) from exc

    return changes


def assert_checkpoint_save_preflight(model, run_dir: Path) -> None:
    """Exercise the real save path before paying for calibration.

    WHY. The checkpoint save is a single all-or-nothing operation at the end of
    a multi-hour run, and everything inside it -- config serialization,
    generation-config validation, directory creation, free space -- is only
    discovered then. GLM-5.2's ``top_p``-without-``do_sample`` cost two 4-GPU
    runs about 11 GPU-hours between them and produced no artifact, because the
    validator that rejected it runs at the END of ``save_pretrained``.

    So do the cheap parts of the save for real, now, against the actual output
    filesystem: serialize both configs and write a probe file. That covers the
    config classes of failure, plus a read-only mount, a missing parent, or a
    full volume. It takes milliseconds and turns "6 hours then nothing" into
    "fails before the GPUs warm up".

    What it deliberately does NOT cover: shard writing throughput, the
    offloaded weight-format revert, and tied-weight handling. Those need real
    tensors. ``assert_transformers_offloaded_save_healthy`` gates the known
    revert bug separately; the rest is genuinely only testable by saving.

    :param model: loaded model whose configs will be serialized.
    :param run_dir: run directory; the probe is written under it.
    :raises RuntimeError: if any part of the probe fails.
    """
    import shutil
    import tempfile

    probe_parent = Path(run_dir)
    try:
        probe_parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            f"save preflight: cannot create run dir {probe_parent}: {exc}"
        ) from exc

    probe = Path(tempfile.mkdtemp(prefix=".save-preflight-", dir=probe_parent))
    try:
        # The exact calls transformers makes inside save_pretrained, in order.
        for name in ("config", "generation_config"):
            obj = getattr(model, name, None)
            if obj is None:
                continue
            try:
                obj.save_pretrained(str(probe))
            except Exception as exc:
                raise RuntimeError(
                    f"save preflight: model.{name}.save_pretrained failed, which "
                    "would abort the real save AFTER calibration. Fix this "
                    f"before spending GPU time. Underlying error: {exc}"
                ) from exc

        # A real write, so a read-only mount or full volume surfaces here.
        try:
            (probe / "probe.bin").write_bytes(b"\0" * (1 << 20))
        except Exception as exc:
            raise RuntimeError(
                f"save preflight: cannot write to {probe_parent}: {exc}"
            ) from exc

        written = sorted(p.name for p in probe.iterdir())
        print(
            f"[pipeline] save preflight OK: serialized {written} to "
            f"{probe_parent}",
            flush=True,
        )
    finally:
        shutil.rmtree(probe, ignore_errors=True)


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


def resolve_norm_gain_offset(model) -> float | None:
    """The architecture's norm gain form, or None if it cannot be established.

    The smooth-fold gate re-derives the norm-implied scale from saved tensors,
    which requires knowing whether the norm applies ``output * weight`` (offset
    0.0) or ``output * (1 + weight)`` (offset 1.0). Guessing is not safe in
    either direction: with the wrong form a perfectly consistent fold reports a
    large relative L2 error, so the gate fails a healthy run at the very end,
    after all the calibration has been paid for.

    Authority is the two registries in llmcompressor/preflight/quantization.py,
    which are assertions that someone READ the class's forward. Anything not in
    either registry, or a model mixing both forms, returns None so the caller
    skips rather than inventing an answer.
    """
    from llmcompressor.preflight.quantization import (
        KNOWN_OFFSET_NORM_CLASSES,
        KNOWN_ORDINARY_NORM_CLASSES,
    )

    seen: set[str] = set()
    for module in model.modules():
        name = type(module).__name__
        if name in KNOWN_OFFSET_NORM_CLASSES:
            seen.add("offset")
        elif name in KNOWN_ORDINARY_NORM_CLASSES:
            seen.add("ordinary")

    if seen == {"offset"}:
        return 1.0
    if seen == {"ordinary"}:
        return 0.0
    return None


def assert_smooth_fold_consistency(
    ckpt: Path,
    base: Path,
    layers: list[int],
    threshold: float = 0.02,
    scale_mean_range: tuple[float, float] = (0.05, 20.0),
    *,
    norm_gain_offset: float | None = None,
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

    if norm_gain_offset is None:
        print(
            "[pipeline] smooth-fold gate skipped: the norm gain form could not "
            "be established (no norm class in KNOWN_OFFSET_NORM_CLASSES or "
            "KNOWN_ORDINARY_NORM_CLASSES, or the model mixes both). Auditing "
            "with the wrong form fails a healthy fold, so this skips instead."
        )
        return

    try:
        report = audit_checkpoint(
            Path(base), Path(ckpt), layers, norm_gain_offset=norm_gain_offset
        )
    except (ValueError, KeyError, FileNotFoundError) as err:
        print(f"[pipeline] smooth-fold gate skipped (names not resolvable): {err}")
        return

    failures = []
    attention_checked = 0
    attention_absent = 0
    lo, hi = scale_mean_range
    for layer, record in report["layers"].items():
        if "moe_side" in record:
            print(f"[pipeline] smooth-fold gate: layer {layer} {record['moe_side']}")
        for component in ("router_compensation", "shared_gate_up_compensation"):
            if component not in record:
                continue
            err = record[component]["relative_l2_error"]
            comp_threshold = threshold
            if record[component].get("fp8_dequantized"):
                # r8 mixed recipes FP8-quantize the shared experts: the
                # dequantized witness carries ~3.6% rel RMS rounding noise
                # on top of any fold, so widen the band (a lost fold still
                # shows >0.09 — see the r2 reference magnitudes above).
                comp_threshold = max(threshold, 0.08)
            # `not <=` so NaN (e.g. non-finite prediction) fails closed.
            if not (err <= comp_threshold):
                failures.append(
                    f"layer {layer} {component}: rel_l2={err:.3e} "
                    f"(threshold {comp_threshold})"
                )
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

        # The attention-side norms, unchecked by this gate until 2026-08-28.
        # Auditing them is what makes the DSA indexer's compensation verifiable at
        # all; with only post_attention_layernorm covered, a lost or partial fold
        # anywhere in the attention block was invisible.
        for norm_component, group in (record.get("attention_fold") or {}).items():
            if group.get("status") != "checked":
                continue
            for consumer, entry in group["consumers"].items():
                status = entry.get("status")
                if status in ("missing_from_candidate", "missing_from_base"):
                    failures.append(
                        f"layer {layer} {norm_component}/{consumer}: {status} "
                        "(a weight the base has is not in the saved checkpoint, "
                        "or vice versa)"
                    )
                    continue
                if status != "checked":
                    attention_absent += 1
                    continue
                attention_checked += 1
                err = entry["relative_l2_error"]
                comp_threshold = max(threshold, 0.08) if entry.get(
                    "fp8_dequantized"
                ) else threshold
                if not (err <= comp_threshold):
                    failures.append(
                        f"layer {layer} {norm_component}/{consumer}: "
                        f"rel_l2={err:.3e} (threshold {comp_threshold})"
                    )
                scale_stats = entry["scale"]
                if scale_stats["finite_fraction"] < 1.0:
                    failures.append(
                        f"layer {layer} {norm_component}/{consumer}: "
                        "non-finite norm-implied scale"
                    )
                elif scale_stats["mean"] is None or not (
                    lo <= scale_stats["mean"] <= hi
                ):
                    failures.append(
                        f"layer {layer} {norm_component}/{consumer}: degenerate "
                        f"fold, scale mean={scale_stats['mean']} outside [{lo}, {hi}]"
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
    # Report the attention-side coverage explicitly. A count of 0 checked is not
    # a failure -- some architectures expose none of these names -- but it MUST be
    # visible, because "the gate passed" reading as "the attention block was
    # audited" is exactly the confusion that let the indexer go unexamined.
    print(
        f"[pipeline] smooth-fold gate: attention-side consumers checked="
        f"{attention_checked}, absent={attention_absent}"
        + (
            "  (0 checked -- this architecture exposes none of the audited "
            "attention names, so the attention block is NOT covered)"
            if attention_checked == 0
            else ""
        )
    )


def assert_quant_checkpoint_verified(
    ckpt: Path,
    base: Path | None = None,
    fp8_dynamic_targets: list[str] | None = None,
) -> None:
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
    # Pick the keep-bf16 expectation set from the model rather than taking the
    # M3 default: on a GLM checkpoint the M3 preset fails 5x for modules GLM
    # does not have (vision_tower, multi_modal_projector, patch_merge,
    # block_sparse_moe, indexer), which is what made the router-fix validation
    # run exit 1 with an otherwise clean report.
    from pipeline.verify_quant_checkpoint import _IGNORE_PRESETS

    arch = ""
    try:
        import json as _json
        arch = " ".join(
            _json.loads((Path(ckpt) / "config.json").read_text(encoding="utf-8"))
            .get("architectures", []) or []
        )
    except Exception:
        pass
    preset = "glm52" if "Glm" in arch else "m3"
    print(f"[pipeline] quant-verify keep-bf16 preset: {preset} (arch={arch!r})")
    # DERIVED FROM THE RECIPE, not a flag. The verifier lists the DSA indexer under
    # must-stay-bf16, which was right while the indexer was BF16 by policy. A recipe
    # that deliberately FP8-quantizes indexer.wq_b / wk -- matching zai-org's own
    # FP8 release and PhalaCloud's W4AFP8, both of which do -- must not be failed by
    # a gate encoding the older stance. Reading it off the targets means the two
    # cannot disagree, where a flag would go stale the first time someone copied a
    # config.
    allow_fp8: set[str] = set()
    if any("indexer" in t for t in (fp8_dynamic_targets or [])):
        allow_fp8.add("msa_indexer")
        print("[pipeline] quant-verify: recipe FP8-targets the DSA indexer, so "
              "msa_indexer is allowed to be fp8-quantized")
    rc = verify(
        Path(ckpt),
        check_tensors=True,
        dequant_base=dequant_base,
        expect_ignore=_IGNORE_PRESETS[preset],
        allow_fp8_components=allow_fp8,
    )
    if rc != 0:
        raise RuntimeError(
            "quant checkpoint verification gate FAILED (see [FAIL] lines "
            "above) — do not serve or evaluate this checkpoint; see "
            "BUGS_AND_FIXES.md 'distributed qparam broadcast lost under "
            "disk offload'"
        )
    print("[pipeline] quant-verify gate OK: structure + sampled tensors healthy")


def _resolve_weight_index(model_id: str) -> Path | None:
    """Locate a local ``model.safetensors.index.json`` for ``model_id``.

    ``model.id`` is usually a HUB REPO ID (``zai-org/GLM-5.2``), not a directory.
    Treating it as a path makes ``Path(model_id)/"model.safetensors.index.json"``
    a relative path that never exists, so any check built on it silently skips
    for every cache-resolved model — which is exactly how the VMA gate went inert
    on the first GLM-5.2 run, printing "no local safetensors index at
    zai-org/GLM-5.2/model.safetensors.index.json".

    Order: an explicit local directory first (so a path-style ``model.id`` still
    wins), then the HF cache. Never downloads: a gate must not depend on network.
    Returns ``None`` when nothing local is found.
    """
    filename = "model.safetensors.index.json"

    direct = Path(model_id) / filename
    if direct.is_file():
        return direct

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    try:
        cached = try_to_load_from_cache(repo_id=model_id, filename=filename)
    except Exception:
        # A malformed repo id, or a cache layout we do not understand, must not
        # crash the run before quantization has even started.
        return None

    # try_to_load_from_cache returns a str on hit, or the _CACHED_NO_EXIST
    # sentinel object when the file is known to be absent upstream.
    if isinstance(cached, str) and Path(cached).is_file():
        return Path(cached)
    return None


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
    index_json = _resolve_weight_index(cfg.model.id)
    if index_json is None:
        print(
            "[pipeline] VMA gate: could not resolve a local "
            "model.safetensors.index.json for "
            f"{cfg.model.id!r} (tried the path directly and the HF cache); "
            "cannot estimate segment count, skipping"
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

    # Repair the generation config HERE, immediately after load, not at save
    # time. transformers validates it strictly at the very end of
    # save_pretrained, so an unrepairable config costs the entire calibration
    # before it surfaces: GLM-5.2's shipped top_p-without-do_sample cost both
    # smoke arms ~5-6 h of 4-GPU time and produced no checkpoint. Running it
    # here makes that failure mode cost seconds.
    for change in repair_generation_config(model):
        print(f"[pipeline] generation_config repaired: {change}", flush=True)

    # Then prove the save path actually works, on the real output filesystem,
    # while failing still costs seconds. Runs on every rank: mkdtemp gives each
    # a unique probe dir so they cannot race, and each rank's view of the shared
    # PVC is worth checking independently.
    assert_checkpoint_save_preflight(model, run_dir)

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
    if getattr(cfg.calibration, "sequential_weight_prefetch", False):
        depth = int(getattr(cfg.calibration, "sequential_weight_prefetch_depth", 1) or 1)
        oneshot_kwargs["sequential_weight_prefetch"] = True
        oneshot_kwargs["sequential_weight_prefetch_depth"] = depth
        print(
            "[pipeline] sequential_weight_prefetch=True (depth "
            f"{depth}): the next layer's weight files are advised WILLNEED while "
            "the current layer computes, and files no later layer needs are "
            "advised DONTNEED. Both halves matter -- the release half is what "
            "keeps page cache from pinning the memory cgroup at its limit, where "
            "100% of reclaim becomes direct reclaim (measured 23% full stall).",
            flush=True,
        )

    if getattr(cfg.calibration, "stop_after_last_target", False):
        oneshot_kwargs["stop_after_last_target"] = True
        print(
            "[pipeline] stop_after_last_target=True (SMOKE ONLY): the sequential "
            "walk will stop after the last subgraph with a compression target",
            flush=True,
        )

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
            if cfg.quantization.fp8_dynamic_targets:
                # Mixed int4+FP8 checkpoint (r8): a global quantization_format
                # stamps EVERY config group — the r8 v2 smoke packed the FP8
                # attention/shared/dense weights through the int4 compressor
                # (weight_packed) which vLLM's FP8 path cannot load. But pure
                # inference is wrong the other way (the W4AFP8 group infers
                # 'int-quantized'). Stamp per-scheme formats instead; the
                # model-level format then flattens to 'mixed-precision' and
                # vLLM resolves formats per config group.
                counts = _stamp_mixed_precision_formats(model)
                print(f"[pipeline] per-scheme compression formats: {counts}")
            else:
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
                # Derived from the model, not M3's sparse range. The old
                # constant range(3, 60) left GLM-5.2/5.3 layers 60-77 (18 of
                # 75 MoE layers) unaudited, so a fold lost only in that tail --
                # the r2/r3/r7 failure mode -- would pass silently.
                depth = getattr(getattr(model, "config", None), "num_hidden_layers", 60)
                first_dense = getattr(
                    getattr(model, "config", None), "first_k_dense_replace", 3
                )
                gate_layers = list(range(int(first_dense), int(depth)))
                # Add the DSA indexer layers. Starting at first_k_dense_replace
                # covers every MoE layer, which is right for the router and shared
                # experts -- but GLM's indexer_types marks layers 0,1,2 as "full"
                # (own indexer) and layer 3 as "shared" (no indexer at all), so the
                # attention-side audit reported `absent=3` and checked no indexer on
                # a run whose whole point was an indexer change. An audit that
                # cannot see the component under test is not an audit.
                indexer_types = getattr(
                    getattr(model, "config", None), "indexer_types", None
                ) or []
                indexer_layers = [
                    i for i, kind in enumerate(indexer_types) if kind == "full"
                ]
                added = sorted(set(indexer_layers) - set(gate_layers))
                if added:
                    gate_layers = sorted(set(gate_layers) | set(indexer_layers))
                    print(
                        f"[pipeline] smooth-fold gate: added indexer layers {added} "
                        "so the attention-side audit is not vacuous"
                    )
            norm_gain_offset = resolve_norm_gain_offset(model)
            print(
                "[pipeline] norm gain form: "
                + (
                    "unresolved"
                    if norm_gain_offset is None
                    else f"output * ({norm_gain_offset:g} + weight)"
                )
            )
            assert_smooth_fold_consistency(
                ckpt,
                Path(cfg.model.id),
                gate_layers,
                norm_gain_offset=norm_gain_offset,
            )
            assert_quant_checkpoint_verified(
                ckpt,
                Path(cfg.model.id),
                fp8_dynamic_targets=list(cfg.quantization.fp8_dynamic_targets or []),
            )
            # Storage-vs-scheme consistency: no ignore entry may hide a module
            # that IS quantized. Loaders check ignore before targets, so a
            # shadowed module serves as unquantized -- its quantized bytes cast
            # into unscaled parameters, garbage output, exit code 0 (M3 r8 ABI
            # smoke, 2026-07-24). Runs LAST because it reads the final config,
            # after _persist_ignore_to_config and any serve-config patching.
            # Enforced unconditionally, including on partial-scope smokes. An
            # earlier revision exempted them, reasoning that their
            # layer-restriction pattern covers too many unquantized modules to
            # enumerate. That exemption was unnecessary AND weakening: when
            # resolution overflows, _persist_ignore_to_config DROPS the pattern
            # and warns, so no shadowing survives and this gate passes anyway.
            # The only way it fires is a pattern that genuinely hides a quantized
            # module -- which is never acceptable, smoke or not.
            from pipeline.serve_ignore import assert_no_ignore_shadowing

            assert_no_ignore_shadowing(ckpt)
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
