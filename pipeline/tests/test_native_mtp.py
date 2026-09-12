"""CPU contract tests for complete source-driven native draft assembly."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import pipeline.native_mtp as mtp
from pipeline.native_sglang_save import _tensor_hash, verify_native_sglang_checkpoint


def _config():
    return {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
        "hidden_size": 128,
        "moe_intermediate_size": 128,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_attention_heads": 2,
        "q_lora_rank": 128,
        "kv_lora_rank": 128,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 64,
        "v_head_dim": 64,
        "index_head_dim": 32,
        "index_n_heads": 2,
        "vocab_size": 16,
        "tie_word_embeddings": False,
    }


def _json(path, data):
    path.write_text(json.dumps(data))


def _source(path, config=None):
    path.mkdir()
    config = config or _config()
    torch.manual_seed(53)
    tensors = {
        name: torch.randn(spec["shape"]).to(
            torch.float32 if spec["dtype"] == "F32" else torch.bfloat16
        )
        for name, spec in mtp.source_inventory(config).items()
    }
    tensors.update(
        {
            f"model.layers.{i}.input_layernorm.weight": torch.ones(128).bfloat16()
            for i in range(config["num_hidden_layers"])
        }
    )
    save_file(tensors, path / "source.safetensors")
    _json(path / "config.json", config)
    _json(
        path / mtp.INDEX,
        {"weight_map": {name: "source.safetensors" for name in tensors}},
    )
    return tensors


def _main(path, *, indexed=False, dtype=torch.bfloat16):
    path.mkdir()
    config = _config() | {"num_nextn_predict_layers": 0}
    config["quantization_config"] = {
        "quant_method": "w4afp8",
        "group_size": 128,
        "weight_block_size": [128, 128],
        "moe_input_scale_policy": "fixed_unit",
        "moe_activation_scheme": "static",
        "linear_activation_scheme": "dynamic",
        "ignored_layers": [],
    }
    tensors = {
        f"model.layers.{i}.input_layernorm.weight": torch.ones(128, dtype=dtype)
        for i in range(2)
    }
    tensors["model.layers.0.mlp.experts.0.gate_proj.weight"] = torch.ones(
        (128, 64), dtype=torch.int8
    )
    tensors["model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"] = torch.ones(
        (128, 1), dtype=dtype
    )
    filename = "model-00001-of-00001.safetensors" if indexed else "model.safetensors"
    save_file(tensors, path / filename)
    if indexed:
        _json(
            path / mtp.INDEX,
            {
                "metadata": {"total_size": 1},
                "weight_map": {name: filename for name in tensors},
            },
        )
    _json(path / "config.json", config)
    _json(
        path / mtp.MANIFEST,
        {
            "version": 1,
            "format": "sglang-w4afp8",
            "mtp_present": False,
            "moe_input_scale_policy": "fixed_unit",
            "tensors": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in tensors.items()
            },
            "sha256": {name: _tensor_hash(value) for name, value in tensors.items()},
        },
    )
    return path / filename


def _snapshot(path):
    return {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("main_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_complete_assembly_reuses_kernels_preserves_main_and_hashes_copies(
    tmp_path, indexed, main_dtype
):
    source, target = tmp_path / "source", tmp_path / "target"
    tensors = _source(source)
    main_file = _main(target, indexed=indexed, dtype=main_dtype)
    main_hash = hashlib.sha256(main_file.read_bytes()).hexdigest()
    plan = mtp.preflight_native_mtp(source, _config())
    assert len(plan.weight_map) == 29
    assert len(mtp.native_inventory(plan.config)) == 51
    assert plan.provenance()["source_tensors"] == 29
    evidence = mtp.assemble_native_mtp(target, plan, max_shard_bytes=100_000)
    assert evidence["expert_quantization"] == "rtn"
    assert evidence["tensors"] == 51
    assert hashlib.sha256(main_file.read_bytes()).hexdigest() == main_hash
    assert verify_native_sglang_checkpoint(target)["ok"]
    additions = {}
    for filename in evidence["shards"]:
        additions.update(load_file(target / filename))
    manifest = json.loads((target / mtp.MANIFEST).read_text())
    assert set(additions) <= manifest["sha256"].keys()
    for name, value in additions.items():
        assert _tensor_hash(value) == manifest["sha256"][name]
    for name in plan.weight_map:
        kind = mtp.classify(name, plan.layer)
        module = name.removesuffix(".weight")
        if kind == "expert":
            values, scale = mtp.quantize_int4_group_rtn(tensors[name])
            assert torch.equal(additions[name], mtp.pack_nibbles_int8(values))
            assert torch.equal(additions[module + ".weight_scale_inv"], scale)
            parent, _, projection = module.rpartition(".")
            suffix = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[projection]
            assert torch.equal(
                additions[f"{parent}.{suffix}.input_scale"], torch.ones(1).bfloat16()
            )
            assert module + ".input_scale" not in additions
        elif kind == "fp8":
            values, scale = mtp.quantize_block_fp8(tensors[name])
            assert torch.equal(
                additions[name].view(torch.uint8), values.view(torch.uint8)
            )
            assert torch.equal(additions[module + ".weight_scale_inv"], scale)
        else:
            assert torch.equal(additions[name], tensors[name])
            assert additions[name].dtype == tensors[name].dtype
    # kv_a has 192 output rows, exercising the actual source's padded FP8 case.
    assert additions["model.layers.2.self_attn.kv_a_proj_with_mqa.weight"].shape == (
        192,
        128,
    )
    assert (
        additions["model.layers.2.mlp.gate.e_score_correction_bias"].dtype
        == torch.float32
    )
    with pytest.raises(ValueError, match="absent MTP"):
        mtp.assemble_native_mtp(target, plan)


def test_full_public_geometry_matches_verified_evidence():
    evidence = json.loads(
        (
            Path(__file__).parents[2]
            / "docs/evidence/2026-09-12-native-mtp-source-contract.json"
        ).read_text()
    )
    config = _config() | {
        "num_hidden_layers": 78,
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_attention_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "v_head_dim": 256,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "vocab_size": 154880,
    }
    inventory = mtp.source_inventory(config)
    assert len(inventory) == evidence["mtp_tensor_count"] == 791
    assert (
        len(mtp.native_inventory(config))
        == evidence["expected_native_mtp_tensors"]
        == 2337
    )
    for name, spec in evidence["mtp_nonexpert_and_expert0_metadata"].items():
        assert inventory[name] == spec


@pytest.mark.parametrize(
    "damage",
    [
        "missing_expert",
        "missing_copy",
        "extra_expert",
        "wrong_shape",
        "wrong_dtype",
        "depth",
        "architecture",
        "source_quantized",
        "no_draft",
    ],
)
def test_preflight_rejects_incomplete_mismatched_source(tmp_path, damage):
    source = tmp_path / "source"
    tensors = _source(source)
    config = _config()
    if damage in {"depth", "architecture"}:
        config["num_hidden_layers" if damage == "depth" else "model_type"] = (
            3 if damage == "depth" else "deepseek_v3"
        )
    elif damage in {"source_quantized", "no_draft"}:
        source_config = _config()
        source_config[
            "quantization_config"
            if damage == "source_quantized"
            else "num_nextn_predict_layers"
        ] = {"quant_method": "fp8"} if damage == "source_quantized" else 0
        _json(source / "config.json", source_config)
    else:
        name = "model.layers.2.mlp.experts.0.gate_proj.weight"
        if damage == "missing_expert":
            tensors.pop(name)
        elif damage == "missing_copy":
            tensors.pop("model.layers.2.shared_head.norm.weight")
        elif damage == "extra_expert":
            tensors[name.replace("experts.0", "experts.2")] = tensors[name].clone()
        elif damage == "wrong_shape":
            tensors[name] = torch.zeros((128, 64)).bfloat16()
        else:
            tensors[name] = tensors[name].float()
        save_file(tensors, source / "source.safetensors")
        _json(
            source / mtp.INDEX,
            {"weight_map": {name: "source.safetensors" for name in tensors}},
        )
    with pytest.raises(ValueError, match="MTP"):
        mtp.preflight_native_mtp(source, config)


def test_depth_one_rejected_for_legacy_runtime():
    with pytest.raises(ValueError, match="legacy NextN"):
        mtp.source_inventory(_config() | {"num_hidden_layers": 1})


def test_hub_source_uses_loaded_commit_and_keeps_snapshot_symlinks(
    tmp_path, monkeypatch
):
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    _source(snapshot)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for name in ("config.json", mtp.INDEX, "source.safetensors"):
        (snapshot / name).rename(blobs / name)
        (snapshot / name).symlink_to(blobs / name)
    calls = []

    def cached(*, repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        return str(snapshot / filename)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    config = _config() | {"_commit_hash": "a" * 40}
    plan = mtp.preflight_native_mtp("owner/source", config)
    assert plan.source == snapshot
    assert calls == [("owner/source", mtp.INDEX, "a" * 40)]
    assert plan.revision == "a" * 40
    with pytest.raises(ValueError, match="_commit_hash"):
        mtp.preflight_native_mtp("owner/source", _config())


@pytest.mark.parametrize("changed", ["config.json", mtp.INDEX, "source.safetensors"])
def test_source_changed_across_calibration_is_rejected_without_destination_changes(
    tmp_path, changed
):
    source, target = tmp_path / "source", tmp_path / "target"
    _source(source)
    _main(target)
    plan = mtp.preflight_native_mtp(source, _config())
    previous = _snapshot(target)
    # Replacing a file with identical bytes still changes its retained identity.
    replacement = source / "replacement"
    replacement.write_bytes((source / changed).read_bytes())
    replacement.replace(source / changed)
    with pytest.raises(ValueError, match="source changed"):
        mtp.assemble_native_mtp(target, plan)
    assert _snapshot(target) == previous


@pytest.mark.parametrize(
    "failure",
    [
        "quantize",
        "stage_corruption",
        "shard_collision",
        "publish_metadata",
        "during_read",
    ],
)
@pytest.mark.parametrize("indexed", [False, True])
def test_failures_restore_original_artifact(tmp_path, monkeypatch, failure, indexed):
    source, target = tmp_path / "source", tmp_path / "target"
    _source(source)
    _main(target, indexed=indexed)
    plan = mtp.preflight_native_mtp(source, _config())
    if failure == "shard_collision":
        (target / "model-mtp-00001.safetensors").write_bytes(b"unrelated file")
    original = _snapshot(target)
    quantize = mtp.quantize_int4_group_rtn
    if failure in {"quantize", "during_read"}:

        def broken(weight):
            if failure == "quantize":
                raise RuntimeError("injected quantization error")
            os.utime(source / "source.safetensors", ns=(1, 1))
            return quantize(weight)

        monkeypatch.setattr(mtp, "quantize_int4_group_rtn", broken)
    elif failure == "stage_corruption":
        original_verify = mtp._verify_additions

        def corrupted(stage, weight_map, expected, hashes):
            name = next(iter(weight_map.values()))
            with (stage / name).open("r+b") as handle:
                handle.seek(-1, 2)
                byte = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([byte[0] ^ 1]))
            return original_verify(stage, weight_map, expected, hashes)

        monkeypatch.setattr(mtp, "_verify_additions", corrupted)
    elif failure == "publish_metadata":
        replace = os.replace

        def broken_replace(src, dst):
            if Path(src).name == mtp.MANIFEST:
                raise OSError("injected metadata publication error")
            return replace(src, dst)

        monkeypatch.setattr(mtp.os, "replace", broken_replace)
    with pytest.raises((ValueError, RuntimeError, FileExistsError, OSError)):
        mtp.assemble_native_mtp(target, plan, max_shard_bytes=100_000)
    assert _snapshot(target) == original
    assert not list(target.glob(".native-mtp-stage-*"))
    assert verify_native_sglang_checkpoint(target)["ok"]


def test_interruption_marker_invalidates_checkpoint(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    _source(source)
    _main(target)
    plan = mtp.preflight_native_mtp(source, _config())
    replace = os.replace

    def interrupted(src, dst):
        if Path(src).name == mtp.MANIFEST:
            raise KeyboardInterrupt("interrupted during metadata publish")
        return replace(src, dst)

    monkeypatch.setattr(mtp.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        mtp.assemble_native_mtp(target, plan)
    assert (target / mtp.MARKER).exists()
    with pytest.raises(ValueError, match="incomplete native MTP"):
        verify_native_sglang_checkpoint(target)
    with pytest.raises(ValueError, match="incomplete native MTP"):
        mtp.assemble_native_mtp(target, plan)


@pytest.mark.parametrize(
    "damage",
    [
        "remove_copied_hash",
        "remove_tensor",
        "fake_policy",
        "fake_source",
        "missing_flag",
        "bad_layer",
        "copied_bytes",
        "shard_provenance",
    ],
)
def test_final_verifier_rejects_corrupt_or_incomplete_mtp(tmp_path, damage):
    source, target = tmp_path / "source", tmp_path / "target"
    _source(source)
    _main(target)
    evidence = mtp.assemble_native_mtp(
        target, mtp.preflight_native_mtp(source, _config())
    )
    manifest = json.loads((target / mtp.MANIFEST).read_text())
    name = "model.layers.2.shared_head.norm.weight"
    if damage == "remove_copied_hash":
        del manifest["sha256"][name]
    elif damage == "remove_tensor":
        # Even altering both index and manifest cannot redefine the closed inventory.
        del manifest["tensors"][name]
        del manifest["sha256"][name]
        index = json.loads((target / mtp.INDEX).read_text())
        del index["weight_map"][name]
        _json(target / mtp.INDEX, index)
    elif damage == "fake_policy":
        manifest["mtp"]["expert_quantization"] = "gptq"
    elif damage == "fake_source":
        manifest["mtp"]["source"] = {}
    elif damage == "missing_flag":
        del manifest["mtp_present"]
    elif damage == "bad_layer":
        manifest["mtp"]["layer"] = 1
    elif damage == "shard_provenance":
        manifest["mtp"]["shards"] = ["made-up.safetensors"]
    else:
        filename = evidence["shards"][0]
        tensors = load_file(target / filename)
        tensors[name].add_(1)
        save_file(tensors, target / filename)
    _json(target / mtp.MANIFEST, manifest)
    with pytest.raises(ValueError, match="MTP|hashes|tensor bytes"):
        verify_native_sglang_checkpoint(target)


def test_quantization_scale_underflow_fails_and_rolls_back(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    tensors = _source(source)
    tensors["model.layers.2.mlp.experts.0.gate_proj.weight"].fill_(2.0**-133)
    save_file(tensors, source / "source.safetensors")
    _main(target)
    original = _snapshot(target)
    with pytest.raises(ValueError, match="nonfinite/nonpositive scale"):
        mtp.assemble_native_mtp(target, mtp.preflight_native_mtp(source, _config()))
    assert _snapshot(target) == original


def test_mtp_policy_config_and_recipe_provenance():
    from pipeline.config import (
        ModelConfig,
        PipelineConfig,
        QuantizationConfig,
        load_config,
    )
    from pipeline.recipe import describe_recipe

    assert QuantizationConfig().mtp_policy == "absent"
    for policy in ("unknown", "source-rtn"):
        cfg = PipelineConfig(
            model=ModelConfig(id="source"),
            quantization=QuantizationConfig(mtp_policy=policy),
        )
        with pytest.raises(ValueError, match="mtp_policy|requires native"):
            cfg.validate()
    root = Path(__file__).parents[1] / "configs"
    for name in (
        "glm53_distributed_w4afp8_awq_full.yaml",
        "glm53_ep_gptq_w4afp8_full.yaml",
        "glm53_ep_gptq_w4afp8_representative.yaml",
    ):
        cfg = load_config(root / name)
        policy = "absent" if "representative" in name else "source-rtn"
        assert cfg.quantization.mtp_policy == policy
        assert describe_recipe(cfg.quantization)["mtp_policy"] == policy


@pytest.mark.parametrize(
    "mode", ["source", "non_source", "bad_source", "assembly_error"]
)
def test_pipeline_mtp_preflight_and_source_only_finalization_order(
    tmp_path, monkeypatch, mode
):
    from types import SimpleNamespace

    import llmcompressor
    import pipeline.native_sglang_save as native
    import pipeline.quantize as entrypoint
    from pipeline.config import PipelineConfig
    from pipeline.distributed import DistributedContext
    from pipeline.tests.test_native_sglang_save import make_model

    model, _ = make_model()
    cfg = PipelineConfig()
    cfg.model.id = str(tmp_path)
    cfg.quantization.checkpoint_format = "sglang-w4afp8"
    cfg.quantization.mtp_policy = "source-rtn"
    cfg.quantization.method = "awq"
    cfg.quantization.fp8_scheme = "FP8_BLOCK"
    cfg.quantization.ignore = ["lm_head", "re:.*self_attn.*"]
    cfg.quantization.fp8_dynamic_targets = [
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.indexer.wk",
    ]
    cfg.quantization.sample_generation = False
    events = []
    rank = 1 if mode == "non_source" else 0
    dist_ctx = DistributedContext(enabled=True, rank=rank, world_size=2)
    monkeypatch.setattr(dist_ctx, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(
        entrypoint, "assert_vma_budget_for_shared_offload", lambda *a: None
    )
    monkeypatch.setattr(
        entrypoint, "install_distributed_disk_update_offload_patch", lambda: False
    )
    tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    monkeypatch.setattr(
        entrypoint, "_load_model_and_tokenizer", lambda cfg: (model, tokenizer)
    )
    monkeypatch.setattr(entrypoint, "log_model_provenance", lambda *a, **kw: None)
    monkeypatch.setattr(
        entrypoint,
        "build_calibration_dataset_with_partition",
        lambda *a: ([{"input_ids": [1, 2]}], None),
    )
    monkeypatch.setattr(entrypoint, "_persist_calibration_partition", lambda *a: None)
    plan = SimpleNamespace(provenance=lambda: {"validated": True})

    def preflight(source, config):
        events.append("mtp_preflight")
        assert source == cfg.model.id and config is model.config
        if mode == "bad_source":
            raise ValueError("injected incomplete source")
        return plan

    def oneshot(**kwargs):
        assert events == ["mtp_preflight"]
        events.append("oneshot")

    def assemble(path, source_plan):
        assert events == ["mtp_preflight", "oneshot", "barrier", "barrier"]
        assert dist_ctx.is_source and source_plan is plan
        events.append("assemble")
        if mode == "assembly_error":
            raise ValueError("injected assembly failure")
        return {"ok": True}

    def verify(path):
        events.append("verify")
        return {"ok": True}

    monkeypatch.setattr(mtp, "preflight_native_mtp", preflight)
    monkeypatch.setattr(mtp, "assemble_native_mtp", assemble)
    monkeypatch.setattr(llmcompressor, "oneshot", oneshot)
    monkeypatch.setattr(native, "verify_native_sglang_checkpoint", verify)
    if mode == "bad_source":
        with pytest.raises(ValueError, match="incomplete source"):
            entrypoint._run_quantize(cfg, tmp_path, dist_ctx)
        assert events == ["mtp_preflight"]
    elif mode == "assembly_error":
        with pytest.raises(ValueError, match="assembly failure"):
            entrypoint._run_quantize(cfg, tmp_path, dist_ctx)
        assert events == ["mtp_preflight", "oneshot", "barrier", "barrier", "assemble"]
    else:
        entrypoint._run_quantize(cfg, tmp_path, dist_ctx)
        expected = ["mtp_preflight", "oneshot", "barrier", "barrier"]
        assert events == expected + (["assemble", "verify"] if rank == 0 else [])


@pytest.mark.parametrize(
    "damage",
    [
        "header_limit",
        "payload_truncated",
        "index_header_disagree",
        "unsafe_filename",
        "duplicate_json",
    ],
)
def test_preflight_rejects_malformed_source_metadata(tmp_path, damage):
    source = tmp_path / "source"
    _source(source)
    shard = source / "source.safetensors"
    if damage == "header_limit":
        shard.write_bytes((mtp.MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
    elif damage == "payload_truncated":
        with shard.open("r+b") as handle:
            handle.truncate(shard.stat().st_size - 128)
    elif damage in {"unsafe_filename", "index_header_disagree"}:
        index = json.loads((source / mtp.INDEX).read_text())
        name = "model.layers.2.eh_proj.weight"
        index["weight_map"][name] = (
            "../outside.safetensors"
            if damage == "unsafe_filename"
            else "other.safetensors"
        )
        (source / "other.safetensors").write_bytes(shard.read_bytes())
        _json(source / mtp.INDEX, index)
    else:
        (source / "config.json").write_text(
            '{"model_type":"glm_moe_dsa","model_type":"other"}'
        )
    with pytest.raises(ValueError):
        mtp.preflight_native_mtp(source, _config())


@pytest.mark.parametrize(
    "filename", [mtp.MARKER, "model-mtp-00001.safetensors", ".native-mtp-stage-old"]
)
def test_destination_preflight_rejects_old_assembly_files(tmp_path, filename):
    mtp.assert_native_mtp_destination(tmp_path)
    (tmp_path / filename).touch()
    with pytest.raises((ValueError, FileExistsError)):
        mtp.assert_native_mtp_destination(tmp_path)


def test_assembly_never_reads_main_tensor_payloads(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import safetensors

    source, target = tmp_path / "source", tmp_path / "target"
    _source(source)
    main_file = _main(target)
    plan = mtp.preflight_native_mtp(source, _config())
    original_open = safetensors.safe_open

    @contextmanager
    def checked_open(path, *args, **kwargs):
        with original_open(path, *args, **kwargs) as handle:
            if Path(path) == main_file:

                class HeaderOnly:
                    def keys(self):
                        return handle.keys()

                    def get_tensor(self, name):
                        pytest.fail("assembly reread a main tensor payload")

                yield HeaderOnly()
            else:
                yield handle

    monkeypatch.setattr(safetensors, "safe_open", checked_open)
    mtp.assemble_native_mtp(target, plan)


def test_real_glm_native_writer_then_source_assembly(tmp_path):
    from compressed_tensors.quantization import (
        QuantizationStatus,
        preset_name_to_scheme,
    )
    from compressed_tensors.quantization.lifecycle.initialize import (
        initialize_module_for_quantization,
    )

    from llmcompressor.observers import Observer
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )
    from pipeline.native_sglang_save import _EXPERT, _FP8, native_sglang_save
    from tests.llmcompressor.modeling.moe.test_expert_parallel_equivalence import (
        build_model,
    )

    source, target = tmp_path / "source", tmp_path / "target"
    _source(source)
    model = build_model(
        num_hidden_layers=2,
        first_k_dense_replace=1,
        n_experts=2,
        config_overrides=_config()
        | {"intermediate_size": 128, "num_experts_per_tok": 1},
    ).bfloat16()
    plan = mtp.preflight_native_mtp(source, model.config)
    for name, module in model.named_modules():
        if not (_EXPERT.fullmatch(name) or _FP8.fullmatch(name)):
            continue
        scheme = preset_name_to_scheme(
            "W4AFP8" if _EXPERT.fullmatch(name) else "FP8_BLOCK", [name]
        )
        initialize_module_for_quantization(module, scheme)
        observer = Observer.load_from_registry(
            "minmax", base_name="weight", args=scheme.weights
        )
        qparams = observer(module.weight).get_qparams()
        module.weight_scale.data.copy_(qparams["scale"])
        module.weight_zero_point.data.copy_(qparams["zero_point"])
        module.quantization_status = QuantizationStatus.FROZEN
    modify_save_pretrained(model)
    with native_sglang_save(model):
        model.save_pretrained(target)
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.glob("*.safetensors")
    }
    evidence = mtp.assemble_native_mtp(target, plan)
    assert evidence["tensors"] == 51
    assert verify_native_sglang_checkpoint(target)["ok"]
    assert all(
        hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
        for name, digest in before.items()
    )
