"""Offloaded-save weight-format revert: transformers 5.12.1 reverts name
conversions on the whole state dict before the shard loop, so with disk
offload every meta entry carries its checkpoint-format name while
``load_offloaded_parameter`` resolves names against the runtime module tree —
the first offloaded tensor raises and the save dies (smoke r12, 2026-07-18).
Upstream main reverts per shard after materialization;
``_deferred_weight_conversion_compat`` backports that.
"""

import inspect
import json

import pytest
import torch
from safetensors import safe_open
from transformers import PretrainedConfig, PreTrainedModel
from transformers.core_model_loading import Concatenate, WeightConverter, WeightRenaming

from pipeline.quantize import (
    _deferred_weight_conversion_compat,
    rebuild_safetensors_index,
)

# same self-disable condition the shim uses: once the installed transformers
# reverts per shard (>=5.14), the early-revert crash no longer exists and the
# shim must be a passthrough
_FIXED_UPSTREAM = "revert_weight_conversion(model_to_save, shard_state_dict)" in (
    inspect.getsource(PreTrainedModel.save_pretrained)
)


class _TinyConfig(PretrainedConfig):
    model_type = "tiny-deferred-revert-test"


class _TinyModel(PreTrainedModel):
    config_class = _TinyConfig

    def __init__(self, config):
        super().__init__(config)
        # runtime name differs from checkpoint name via WeightRenaming
        self.renamed_dense = torch.nn.Linear(4, 4, bias=False)
        # runtime-fused tensor that revert SPLITS back into checkpoint tensors
        # (mirrors M3's dense-layer mlp.gate_up_proj WeightConverter)
        self.mlp = torch.nn.Module()
        self.mlp.gate_up_proj = torch.nn.Linear(4, 8, bias=False)


def _make_model():
    torch.manual_seed(0)
    model = _TinyModel(_TinyConfig())
    # load-direction transforms (checkpoint -> runtime); save reverses them
    model._weight_conversions = [
        WeightRenaming(source_patterns=r"^dense\.", target_patterns="renamed_dense."),
        WeightConverter(
            source_patterns=["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            target_patterns="mlp.gate_up_proj.weight",
            operations=[Concatenate(dim=0)],
        ),
    ]
    return model


_CHECKPOINT_NAMES = {"dense.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight"}


def _saved_tensors(ckpt) -> dict[str, torch.Tensor]:
    out = {}
    for f in ckpt.glob("*.safetensors"):
        with safe_open(str(f), framework="pt", device="cpu") as fh:
            for name in fh.keys():
                out[name] = fh.get_tensor(name)
    return out


def _offload_to_disk(model, offload_dir):
    """Disk-offload every module the way compressed-tensors' to_accelerate does:
    meta parameters + AlignDevicesHook whose weights_map is keyed by RUNTIME
    names, plus an hf_device_map so save_pretrained takes the offloaded path."""
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset
    from safetensors.torch import save_file

    index = {}
    for name, tensor in model.state_dict().items():
        f = offload_dir / f"{name}.safetensors"
        save_file({name: tensor}, str(f))
        index[name] = {
            "safetensors_file": str(f),
            "weight_name": name,
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
    loader = OffloadedWeightsLoader(index=index, save_folder=str(offload_dir))

    for mod_name, module in model.named_modules():
        if not any(True for _ in module.parameters(recurse=False)):
            continue
        hook = AlignDevicesHook(
            execution_device=torch.device("cpu"),
            offload=True,
            weights_map=PrefixedDataset(prefix=f"{mod_name}.", dataset=loader),
            offload_buffers=True,
        )
        add_hook_to_module(module, hook)  # init_hook metas the parameters
    assert all(p.device.type == "meta" for p in model.parameters())
    model.hf_device_map = {"": "cpu", "mlp": "disk"}


def test_baseline_save_reverts_names(tmp_path):
    """Sanity: a plain (non-offloaded) save writes checkpoint-format names."""
    model = _make_model()
    model.save_pretrained(str(tmp_path))
    assert set(_saved_tensors(tmp_path)) == _CHECKPOINT_NAMES


def test_offloaded_save_crashes_without_shim_and_matches_baseline_with_it(tmp_path):
    baseline_dir = tmp_path / "baseline"
    model = _make_model()
    model.save_pretrained(str(baseline_dir))
    baseline = _saved_tensors(baseline_dir)

    offload_dir = tmp_path / "offload"
    offload_dir.mkdir()
    _offload_to_disk(model, offload_dir)

    # the r12 failure mode, pinned: early revert renames/splits meta entries,
    # then load_offloaded_parameter cannot resolve them on the runtime tree.
    # Once upstream reverts per shard, the crash disappears and the shim must
    # report itself inert instead.
    if not _FIXED_UPSTREAM:
        with pytest.raises(Exception):
            model.save_pretrained(str(tmp_path / "broken"))

    ckpt = tmp_path / "shimmed"
    with _deferred_weight_conversion_compat(model) as deferral:
        model.save_pretrained(str(ckpt))
    assert deferral["deferred"] is (not _FIXED_UPSTREAM)

    saved = _saved_tensors(ckpt)
    assert set(saved) == _CHECKPOINT_NAMES
    for name in _CHECKPOINT_NAMES:
        assert torch.equal(saved[name], baseline[name]), name


def test_offloaded_sharded_save_index_rebuilt(tmp_path):
    model = _make_model()
    offload_dir = tmp_path / "offload"
    offload_dir.mkdir()
    _offload_to_disk(model, offload_dir)

    ckpt = tmp_path / "sharded"
    with _deferred_weight_conversion_compat(model) as deferral:
        # tiny shard size (bytes) forces one tensor per shard -> index written
        model.save_pretrained(str(ckpt), max_shard_size=100)
    assert deferral["deferred"] is (not _FIXED_UPSTREAM)

    # under fixed upstream the rebuild is an idempotent no-op on a correct index
    n = rebuild_safetensors_index(ckpt)
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    saved = _saved_tensors(ckpt)
    assert set(saved) == _CHECKPOINT_NAMES
    assert set(index["weight_map"]) == _CHECKPOINT_NAMES
    assert n == len(index["weight_map"])
    # every mapped file exists and actually holds the tensor it claims
    for name, shard in index["weight_map"].items():
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as fh:
            assert name in fh.keys()
    assert index["metadata"]["total_size"] == sum(
        t.numel() * t.element_size() for t in saved.values()
    )


def test_shim_noop_without_meta_tensors(tmp_path):
    """A regular save under the shim must not defer nor change behavior."""
    model = _make_model()
    with _deferred_weight_conversion_compat(model) as deferral:
        model.save_pretrained(str(tmp_path))
    assert deferral["deferred"] is False
    assert set(_saved_tensors(tmp_path)) == _CHECKPOINT_NAMES


def test_rebuild_index_returns_zero_without_index_file(tmp_path):
    assert rebuild_safetensors_index(tmp_path) == 0
