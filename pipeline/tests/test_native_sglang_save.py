"""Exercise actual CT compression, collective saving and native first writes."""

import json
from pathlib import Path

import pytest
import torch
from compressed_tensors.compressors.naive_quantized import FloatQuantizationCompressor
from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
from compressed_tensors.offload import disable_onloading, offload_module
from compressed_tensors.quantization import QuantizationStatus, preset_name_to_scheme
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.utils import get_direct_state_dict
from safetensors.torch import load_file
from transformers import PretrainedConfig, PreTrainedModel

from llmcompressor.observers import Observer
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)
from pipeline.native_sglang_save import (
    assert_native_sglang_preflight,
    native_sglang_save,
    verify_native_sglang_checkpoint,
)
from pipeline.to_sglang_w4afp8 import default_unpacker


class TinyConfig(PretrainedConfig):
    model_type = "glm_moe_dsa"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = 1
        self.num_nextn_predict_layers = 1  # source advertises a head it did not load
        self.tie_word_embeddings = False


class TinyGlm(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config=None):
        super().__init__(config or TinyConfig())
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.o_proj = torch.nn.Linear(128, 128, bias=False)
        layer.self_attn.indexer = torch.nn.Module()
        layer.self_attn.indexer.wk = torch.nn.Linear(128, 64, bias=False)
        layer.self_attn.indexer.weights_proj = torch.nn.Linear(128, 2, bias=False)
        layer.mlp = torch.nn.Module()
        expert = torch.nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(expert, name, torch.nn.Linear(128, 128, bias=False))
        layer.mlp.experts = torch.nn.ModuleList([expert])
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([layer])
        self.lm_head = torch.nn.Linear(128, 8, bias=False)
        self.bfloat16()

    def forward(self, x):
        return self.model.layers[0].self_attn.o_proj(x)


def make_model():
    torch.manual_seed(531)
    model = TinyGlm()
    expected = {}
    for name, module in model.named_modules():
        if (
            not isinstance(module, torch.nn.Linear)
            or name == "lm_head"
            or name.endswith("weights_proj")
        ):
            continue
        expert = ".experts." in name
        scheme = preset_name_to_scheme("W4AFP8" if expert else "FP8_BLOCK", [name])
        initialize_module_for_quantization(module, scheme)
        observer = Observer.load_from_registry(
            "minmax", base_name="weight", args=scheme.weights
        )
        qparams = observer(module.weight).get_qparams()
        module.weight_scale.data.copy_(qparams["scale"])
        module.weight_zero_point.data.copy_(qparams["zero_point"])
        module.quantization_status = QuantizationStatus.FROZEN
        compressor = (
            PackedQuantizationCompressor if expert else FloatQuantizationCompressor
        )
        state = {
            k: v for k, v in get_direct_state_dict(module).items() if v is not None
        }
        expected[name] = {
            k: v.clone() for k, v in compressor.compress(state, scheme).items()
        }
    modify_save_pretrained(model)
    return model, expected


def assert_native(path, expected):
    tensors = {}
    shards = sorted(Path(path).glob("*.safetensors"))
    assert shards
    for shard in shards:
        tensors.update(load_file(shard))
    assert not any(
        k.endswith(("weight_packed", "weight_scale", "weight_shape", "weight_g_idx"))
        for k in tensors
    )
    for name, ct in expected.items():
        weight = tensors[f"{name}.weight"]
        scale = tensors[f"{name}.weight_scale_inv"]
        if "weight_packed" in ct:
            # Independent decode of bytes as signed nibbles, compared with CT's
            # own unpacker. Avoid using the implementation's native unpacker.
            raw = weight.view(torch.uint8).to(torch.int16)
            values = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(128, 128)
            values = torch.where(values >= 8, values - 16, values).to(torch.int8)
            assert torch.equal(
                values, default_unpacker(ct["weight_packed"], torch.Size([128, 128]))
            )
            assert torch.equal(scale, ct["weight_scale"])
            parent, _, projection = name.rpartition(".")
            suffix = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[projection]
            assert torch.equal(
                tensors[f"{parent}.{suffix}.input_scale"],
                torch.ones(1, dtype=torch.bfloat16),
            )
        else:
            assert torch.equal(weight.view(torch.uint8), ct["weight"].view(torch.uint8))
            assert scale.dtype == torch.float32
            assert torch.equal(scale, ct["weight_scale"].float())
    config = json.loads((Path(path) / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "w4afp8"
    assert config["quantization_config"]["moe_input_scale_policy"] == "fixed_unit"
    assert config["num_nextn_predict_layers"] == 0
    assert verify_native_sglang_checkpoint(path)["ok"]
    index_path = Path(path) / "model.safetensors.index.json"
    if index_path.exists():
        assert set(json.loads(index_path.read_text())["weight_map"]) == set(tensors)


def assert_restored(model, expected, *, forward=True):
    assert model.config.num_nextn_predict_layers == 1
    assert not hasattr(model.config, "quantization_config")
    for name, ct in expected.items():
        module = model.get_submodule(name)
        state = {
            k: v for k, v in get_direct_state_dict(module).items() if v is not None
        }
        assert set(state) == set(ct)
        for key, want in ct.items():
            got = state[key]
            assert got.dtype == want.dtype
            if got.dtype == torch.float8_e4m3fn:
                got, want = got.view(torch.uint8), want.view(torch.uint8)
            assert torch.equal(got, want)
        assert module.quantization_status == QuantizationStatus.COMPRESSED
    assert not hasattr(model.model.layers[0].mlp.experts[0], "w1")
    assert hasattr(model, "ct_decompress_hook")
    if forward:
        output = model(torch.randn(1, 128).bfloat16())
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("disk", [False, True])
def test_actual_ct_save_first_write_and_restore(tmp_path, monkeypatch, disk):
    import transformers.modeling_utils as mu

    from pipeline.quantize import (
        _deferred_weight_conversion_compat,
        _tied_weights_meta_buffer_compat,
    )

    model, expected = make_model()
    if disk:
        (tmp_path / "offload").mkdir()
        for module in model.modules():
            offload_module(module, "cpu", "disk", offload_dir=str(tmp_path / "offload"))
    original_writer = mu.safe_save_file
    written = []

    def check_first_write(tensors, filename, **kwargs):
        assert not any(
            k.endswith(("weight_packed", "weight_scale", "weight_shape"))
            for k in tensors
        )
        written.append(filename)
        return original_writer(tensors, filename, **kwargs)

    monkeypatch.setattr(mu, "safe_save_file", check_first_write)
    with native_sglang_save(model) as info, _tied_weights_meta_buffer_compat(
        model
    ), _deferred_weight_conversion_compat(model):
        assert info["mtp_present"] is False
        model.save_pretrained(tmp_path / "native", max_shard_size="30KB")
    assert written
    assert_native(tmp_path / "native", expected)
    assert_restored(model, expected)


def test_manifest_records_serialized_dtype_for_offloaded_buffer(tmp_path, monkeypatch):
    import transformers.modeling_utils as mu

    from pipeline.quantize import (
        _deferred_weight_conversion_compat,
        _tied_weights_meta_buffer_compat,
    )

    model, expected = make_model()
    model.model.layers[0].mlp.gate = torch.nn.Module()
    model.model.layers[0].mlp.gate.register_buffer(
        "e_score_correction_bias", torch.arange(8, dtype=torch.float32)
    )
    original_writer = mu.safe_save_file

    def emulate_offloaded_write(tensors, filename, **kwargs):
        tensors = dict(tensors)
        key = "model.layers.0.mlp.gate.e_score_correction_bias"
        if key in tensors:
            tensors[key] = tensors[key].to(torch.bfloat16)
        return original_writer(tensors, filename, **kwargs)

    monkeypatch.setattr(mu, "safe_save_file", emulate_offloaded_write)
    with native_sglang_save(model), _tied_weights_meta_buffer_compat(
        model
    ), _deferred_weight_conversion_compat(model):
        model.save_pretrained(tmp_path / "native", max_shard_size="30KB")

    manifest = json.loads(
        (tmp_path / "native" / "native_sglang_manifest.json").read_text()
    )
    key = "model.layers.0.mlp.gate.e_score_correction_bias"
    assert manifest["tensors"][key]["dtype"] == "torch.bfloat16"
    assert_native(tmp_path / "native", expected)


def test_writer_failure_restores_context_and_ct_state(tmp_path, monkeypatch):
    import transformers.modeling_utils as mu

    from llmcompressor.transformers.compression import compressed_tensors_utils as api

    model, expected = make_model()
    factory = api.ModelCompressor

    def fail(*args, **kwargs):
        raise OSError("test writer failure")

    monkeypatch.setattr(mu, "safe_save_file", fail)
    with pytest.raises(OSError, match="test writer failure"):
        with native_sglang_save(model):
            model.save_pretrained(tmp_path / "failed")
    assert api.ModelCompressor is factory
    assert_restored(model, expected)
    with native_sglang_save(model):
        pass


@pytest.mark.parametrize(
    "damage, message",
    [
        ("fp8_channel", "128x128"),
        ("int8", "INT4 group 128"),
        ("group_actorder", "actorder"),
        ("g_idx", "weight_g_idx"),
        ("missing_fp8", "missing required"),
        ("quantized_indexer", "unquantized"),
        ("model_type", "glm_moe_dsa"),
        ("static_activation", "activation"),
        ("mtp", "MTP"),
    ],
)
def test_preflight_rejects_unsupported_contract(damage, message):
    model, _ = make_model()
    attn = model.model.layers[0].self_attn
    expert = model.model.layers[0].mlp.experts[0].gate_proj
    if damage == "fp8_channel":
        attn.o_proj.quantization_scheme.weights.strategy = "channel"
    elif damage == "int8":
        expert.quantization_scheme.weights.num_bits = 8
    elif damage == "group_actorder":
        expert.quantization_scheme.weights.actorder = "group"
    elif damage == "g_idx":
        expert.register_buffer("weight_g_idx", torch.arange(128))
    elif damage == "missing_fp8":
        delattr(attn.indexer.wk, "quantization_scheme")
    elif damage == "quantized_indexer":
        attn.indexer.weights_proj.quantization_scheme = attn.o_proj.quantization_scheme
    elif damage == "model_type":
        model.config.model_type = "llama"
    elif damage == "static_activation":
        expert.quantization_scheme.input_activations.dynamic = False
    else:
        model.config.num_hidden_layers = 0
    with pytest.raises(ValueError, match=message):
        assert_native_sglang_preflight(model)


def test_preflight_accepts_resolved_schemes_before_initialization():
    model = TinyGlm()
    schemes = {}
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and name != "lm_head"
            and not name.endswith("weights_proj")
        ):
            schemes[name] = preset_name_to_scheme(
                "W4AFP8" if ".experts." in name else "FP8_BLOCK", [name]
            )
    assert assert_native_sglang_preflight(model, schemes) == {
        "expert": 3,
        "fp8": 2,
        "unquantized": 2,
    }


@pytest.mark.parametrize("damage", ["weight", "scale", "inventory", "config", "hash"])
def test_manifest_verifier_rejects_corruption(tmp_path, damage):
    from safetensors.torch import save_file

    model, _ = make_model()
    path = tmp_path / "native"
    with native_sglang_save(model):
        model.save_pretrained(path)
    if damage == "config":
        config_path = path / "config.json"
        config = json.loads(config_path.read_text())
        config["num_nextn_predict_layers"] = 1
        config_path.write_text(json.dumps(config))
    elif damage == "hash":
        manifest_path = path / "native_sglang_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"].clear()
        manifest_path.write_text(json.dumps(manifest))
    else:
        shard = path / "model.safetensors"
        tensors = load_file(shard)
        name = "model.layers.0.self_attn.o_proj"
        if damage == "weight":
            tensors[f"{name}.weight"].view(torch.uint8)[0, 0] ^= 1
        elif damage == "scale":
            scale = tensors[f"{name}.weight_scale_inv"]
            scale[0, 0] = torch.nextafter(scale[0, 0], torch.tensor(float("inf")))
        else:
            tensors.pop(f"{name}.weight")
        save_file(tensors, shard)
    with pytest.raises(ValueError):
        verify_native_sglang_checkpoint(path)


def _distributed_save_worker(rank, root, disk):
    import datetime

    import torch.distributed as dist

    from pipeline.quantize import (
        _deferred_weight_conversion_compat,
        _tied_weights_meta_buffer_compat,
    )

    torch.set_num_threads(2)
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        model, expected = make_model()
        for module in model.modules():
            offload_module(
                module,
                "cpu",
                "disk" if disk else "cpu",
                **({"offload_dir": str(root / "offload")} if disk else {}),
            )
        with native_sglang_save(model), _tied_weights_meta_buffer_compat(
            model
        ), _deferred_weight_conversion_compat(model):
            model.save_pretrained(root / "native", max_shard_size="30KB")
        # Source-only file verification must not touch any collective caches.
        if rank == 0:
            assert_native(root / "native", expected)
        dist.barrier()
        assert_restored(model, expected, forward=False)
        # CT's model decompression hook explicitly does not support distributed
        # decompression. Independently deleting shared DiskCache entries during
        # forward can race another rank's onload. Verify the restored shared
        # state first, then materialize private CPU dictionaries without deleting
        # cache files before exercising the forward on every rank.
        from compressed_tensors.offload import remove_dispatch

        remove_dispatch(model, onload_tensors=True)
        dist.barrier()
        assert_restored(model, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("disk", [False, True])
def test_two_process_collective_native_save(tmp_path, disk):
    import torch.multiprocessing as mp

    (tmp_path / "offload").mkdir()
    context = mp.spawn(
        _distributed_save_worker, args=(str(tmp_path), disk), nprocs=2, join=False
    )
    try:
        while not context.join(timeout=30):
            pass
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join()


def test_repeated_export_preserves_already_compressed_fp8(tmp_path):
    model, expected = make_model()
    for name in ("first", "second"):
        with native_sglang_save(model):
            model.save_pretrained(tmp_path / name)
        assert_native(tmp_path / name, expected)
    assert_restored(model, expected)


def test_cannot_write_uncompressed_or_replace_native_state(tmp_path):
    model, _ = make_model()
    original_save = model.save_pretrained
    with native_sglang_save(model):
        with pytest.raises(ValueError, match="save_compressed=True"):
            model.save_pretrained(tmp_path / "invalid", save_compressed=False)
        with pytest.raises(ValueError, match="unmodified model state"):
            model.save_pretrained(tmp_path / "invalid", state_dict={})
    assert model.save_pretrained is original_save
    assert not (tmp_path / "invalid").exists()


def test_actual_transformers_glm_inventory_after_linearization():
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    from llmcompressor.modeling.moe.linearize import linearize_moe

    config = GlmMoeDsaConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        n_routed_experts=2,
        num_experts=2,
        num_experts_per_tok=1,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_n_heads=1,
        index_head_dim=128,
        max_position_embeddings=32,
    )
    model = GlmMoeDsaForCausalLM(config)
    with pytest.raises(ValueError, match="missing required|fused"):
        assert_native_sglang_preflight(model)
    linearize_moe(model)
    schemes = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name == "lm_head" or name.endswith("weights_proj"):
            continue
        schemes[name] = preset_name_to_scheme(
            "W4AFP8" if ".experts." in name else "FP8_BLOCK", [name]
        )
    counts = assert_native_sglang_preflight(model, schemes)
    assert counts["expert"] == 6
    assert counts["fp8"] == 20


def _distributed_invalid_scale_worker(rank, root, expert):
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        model, _ = make_model()
        name = (
            "model.layers.0.mlp.experts.0.up_proj"
            if expert
            else "model.layers.0.self_attn.indexer.wk"
        )
        model.get_submodule(name).weight_scale.data.fill_(float("nan"))
        for module in model.modules():
            offload_module(module, "cpu", "disk", offload_dir=str(root / "offload"))
        with pytest.raises(
            ValueError, match="weight scales must be finite and positive"
        ):
            with native_sglang_save(model):
                model.save_pretrained(root / "invalid")
        assert not (root / "invalid").exists()
        assert model.config.num_nextn_predict_layers == 1
        with disable_onloading():
            for module in model.modules():
                if getattr(module, "quantization_scheme", None) is not None:
                    assert module.quantization_status == QuantizationStatus.FROZEN
                    assert not hasattr(module, "weight_scale_inv")
                    assert module.weight.dtype == torch.bfloat16
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("expert", [False, True])
def test_two_process_invalid_scale_fails_collectively_before_mutation(tmp_path, expert):
    import time

    import torch.multiprocessing as mp

    (tmp_path / "offload").mkdir()
    context = mp.spawn(
        _distributed_invalid_scale_worker,
        args=(str(tmp_path), expert),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=10):
            if time.monotonic() > deadline:
                pytest.fail("native scale failure stranded a collective rank")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join()


@pytest.mark.parametrize("missing_indexer_target", [False, True])
@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_pipeline_preflights_before_oneshot_and_selects_native_save(
    tmp_path, monkeypatch, missing_indexer_target, method
):
    from types import SimpleNamespace

    import llmcompressor
    import pipeline.native_sglang_save as native
    import pipeline.quantize as entrypoint
    from pipeline.config import PipelineConfig

    # Calibration numbers are produced with the real installed observer above;
    # this test isolates the entrypoint order/routing from the GPTQ algorithm.
    model, expected = make_model()
    cfg = PipelineConfig()
    cfg.model.id = str(tmp_path)
    cfg.quantization.checkpoint_format = "sglang-w4afp8"
    cfg.quantization.method = method
    cfg.quantization.gptq_expert_parallel = method == "gptq"
    cfg.quantization.fp8_weights_before_gptq = method == "gptq"
    cfg.quantization.fp8_scheme = "FP8_BLOCK"
    cfg.quantization.ignore = ["lm_head", "re:.*self_attn.*"]
    cfg.quantization.fp8_dynamic_targets = [
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.indexer.wk",
    ]
    if missing_indexer_target:
        cfg.quantization.fp8_dynamic_targets.pop()
    cfg.quantization.sample_generation = False
    events = []
    tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    monkeypatch.setattr(
        entrypoint, "_load_model_and_tokenizer", lambda cfg: (model, tokenizer)
    )
    monkeypatch.setattr(entrypoint, "log_model_provenance", lambda *a, **kw: None)
    monkeypatch.setattr(
        entrypoint,
        "build_calibration_dataset_with_partition",
        lambda *a, **kw: ([{"input_ids": [1, 2]}], None),
    )
    monkeypatch.setattr(
        entrypoint, "_persist_calibration_partition", lambda *a, **kw: None
    )
    original_preflight = native.assert_native_sglang_preflight
    original_verify = native.verify_native_sglang_checkpoint

    def preflight(model, schemes=None):
        if schemes is not None:
            events.append("preflight")
        return original_preflight(model, schemes)

    def oneshot(**kwargs):
        assert events == ["preflight"]
        assert kwargs["model"] is model
        events.append("oneshot")

    def verify(path):
        events.append("native_verify")
        return original_verify(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("native output must not enter CT checkpoint gates")

    monkeypatch.setattr(native, "assert_native_sglang_preflight", preflight)
    monkeypatch.setattr(native, "verify_native_sglang_checkpoint", verify)
    monkeypatch.setattr(llmcompressor, "oneshot", oneshot)
    for name in (
        "_persist_ignore_to_config",
        "assert_quant_checkpoint_verified",
        "assert_smooth_fold_consistency",
    ):
        monkeypatch.setattr(entrypoint, name, forbidden)
    if missing_indexer_target:
        with pytest.raises(ValueError, match="missing required FP8_BLOCK"):
            entrypoint._run_quantize(cfg, tmp_path)
        assert events == ["preflight"]
    else:
        checkpoint = entrypoint._run_quantize(cfg, tmp_path)
        assert events == ["preflight", "oneshot", "native_verify"]
        assert_native(checkpoint, expected)
        assert_restored(model, expected)


def test_real_awq_lifecycle_writes_native_payload_and_restores_state(
    tmp_path, monkeypatch
):
    import transformers.modeling_utils as mu

    from llmcompressor import oneshot
    from llmcompressor.core import active_session
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from tests.llmcompressor.modeling.moe.test_expert_parallel_equivalence import (
        build_model,
    )
    from tests.llmcompressor.modifiers.gptq.test_expert_parallel_oneshot import (
        _calibration,
        _processor,
    )

    model = build_model(
        num_hidden_layers=3,
        first_k_dense_replace=1,
        n_experts=2,
        config_overrides={
            "hidden_size": 128,
            "intermediate_size": 128,
            "moe_intermediate_size": 128,
            "kv_lora_rank": 128,
            "q_lora_rank": 128,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 64,
            "v_head_dim": 64,
        },
    )
    model.config.use_cache = False
    model.config._attn_implementation = "eager"
    expert_targets = [
        r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
    ]
    recipe = [
        AWQModifier(duo_scaling=False, n_grid=3),
        QuantizationModifier(targets=expert_targets, scheme="W4AFP8"),
        QuantizationModifier(
            targets=["Linear"],
            scheme="FP8_BLOCK",
            ignore=expert_targets
            + [
                "lm_head",
                r"re:.*mlp\.gate$",
                r"re:.*indexer\.(weights_proj|k_norm)$",
            ],
        ),
    ]
    expert_name = "model.layers.1.mlp.experts.0.gate_proj"
    fold_name = "model.layers.1.post_attention_layernorm.weight"
    before_awq = model.get_submodule(expert_name).weight.detach().clone()
    try:
        oneshot(
            model=model,
            processor=_processor(),
            recipe=recipe,
            dataset=_calibration(0),
            num_calibration_samples=2,
            sequential_targets=["GlmMoeDsaDecoderLayer"],
            pipeline="sequential",
            moe_calibrate_all_experts=True,
        )
        assert not torch.equal(before_awq, model.get_submodule(expert_name).weight)
        assert_native_sglang_preflight(model)

        expert = model.get_submodule(expert_name)
        expert_state = {
            key: value.detach().clone()
            for key, value in get_direct_state_dict(expert).items()
            if value is not None
        }
        expected_ct = PackedQuantizationCompressor.compress(
            expert_state, expert.quantization_scheme
        )
        fp8_name = "model.layers.1.self_attn.o_proj"
        fp8 = model.get_submodule(fp8_name)
        fp8_state = {
            key: value.detach().clone()
            for key, value in get_direct_state_dict(fp8).items()
            if value is not None
        }
        expected_fp8 = FloatQuantizationCompressor.compress(
            fp8_state, fp8.quantization_scheme
        )
        expected_fold = model.state_dict()[fold_name].detach().clone()
        original_writer = mu.safe_save_file
        first_write_checked = False

        def check_native_first_write(tensors, filename, **kwargs):
            nonlocal first_write_checked
            assert not any(key.endswith(".weight_packed") for key in tensors)
            first_write_checked = True
            return original_writer(tensors, filename, **kwargs)

        monkeypatch.setattr(mu, "safe_save_file", check_native_first_write)
        output = tmp_path / "native-awq"
        with native_sglang_save(model):
            model.save_pretrained(output, max_shard_size="1MB")
        assert first_write_checked
        assert verify_native_sglang_checkpoint(output)["ok"]

        tensors = {}
        for shard in output.glob("*.safetensors"):
            tensors.update(load_file(shard))
        shape = torch.Size(expected_ct["weight_shape"].tolist())
        unpacked = default_unpacker(expected_ct["weight_packed"], shape)
        raw = tensors[f"{expert_name}.weight"].view(torch.uint8).to(torch.int16)
        values = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(shape)
        values = torch.where(values >= 8, values - 16, values).to(torch.int8)
        assert torch.equal(values, unpacked)
        assert torch.equal(
            tensors[f"{expert_name}.weight_scale_inv"], expected_ct["weight_scale"]
        )
        assert torch.equal(
            tensors["model.layers.1.mlp.experts.0.w1.input_scale"],
            torch.ones(1, dtype=torch.bfloat16),
        )
        assert torch.equal(
            tensors[f"{fp8_name}.weight"].view(torch.uint8),
            expected_fp8["weight"].view(torch.uint8),
        )
        assert tensors[f"{fp8_name}.weight_scale_inv"].dtype == torch.float32
        assert torch.equal(
            tensors[f"{fp8_name}.weight_scale_inv"],
            expected_fp8["weight_scale"].float(),
        )
        assert torch.equal(tensors[fold_name], expected_fold)

        for module, expected in ((expert, expected_ct), (fp8, expected_fp8)):
            restored = {
                key: value
                for key, value in get_direct_state_dict(module).items()
                if value is not None
            }
            assert restored.keys() == expected.keys()
            for key, want in expected.items():
                actual = restored[key]
                assert actual.dtype == want.dtype
                if actual.dtype == torch.float8_e4m3fn:
                    actual, want = actual.view(torch.uint8), want.view(torch.uint8)
                assert torch.equal(actual, want)
            assert module.quantization_status == QuantizationStatus.COMPRESSED
        assert torch.equal(model.state_dict()[fold_name], expected_fold)
    finally:
        active_session().reset()
