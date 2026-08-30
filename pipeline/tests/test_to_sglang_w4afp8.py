"""End-to-end tests for the compressed-tensors -> SGLang w4afp8 conversion.

Built on a synthetic two-layer checkpoint with real geometry (hidden 6144 scaled
down, group 128, an int4 expert triplet, an fp8 attention projection, a folded
shared-expert pair, and passthrough norms/router) so the plumbing is exercised
without touching the 394 GB artifact.

The tests that matter most are the ones asserting the FOLD is handled: a dropped
fold produces a checkpoint that loads and serves plausible-looking noise, and no
structural check can see it.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from pipeline.sglang_w4afp8_kernels import unpack_nibbles_int8  # noqa: E402
from pipeline.to_sglang_w4afp8 import (  # noqa: E402
    build_config,
    convert,
    recover_fold_scale,
)

HIDDEN = 256
INTER = 128
GROUP = 128
FP8_TARGETS = [
    r"re:.*model[.]layers[.][0-9]+[.]self_attn[.](o_proj)$",
    r"re:.*model[.]layers[.][0-9]+[.]mlp[.]shared_experts[.](gate_proj|up_proj|down_proj)$",
]


def _unpack_int32(packed: torch.Tensor, shape) -> torch.Tensor:
    """Inverse of :func:`_pack_int32`, paired with it so these tests exercise
    the CONVERSION PLUMBING without needing compressed-tensors installed.

    The real converter delegates to compressed-tensors' unpack_from_int32
    because the on-disk nibble convention belongs to whichever version wrote
    the checkpoint; that pairing is verified separately, on real tensors, by
    `python -m pipeline.to_sglang_w4afp8 --conformance-only`. These tests
    therefore cover names, shapes, folds, sharding and the index -- not the
    int32 storage convention.
    """
    rows = packed.shape[0]
    words = packed.to(torch.int64)
    nibbles = torch.stack([(words >> (4 * i)) & 0xF for i in range(8)], dim=-1)
    flat = nibbles.reshape(rows, -1)[:, : shape[1]]
    return torch.where(flat > 7, flat - 16, flat).to(torch.int8)


def _pack_int32(values: torch.Tensor) -> torch.Tensor:
    """Minimal stand-in for compressed-tensors' pack_to_int32 (8 nibbles/word)."""
    rows, cols = values.shape
    assert cols % 8 == 0
    v = (values.to(torch.int32) & 0xF).reshape(rows, cols // 8, 8)
    out = torch.zeros(rows, cols // 8, dtype=torch.int32)
    for i in range(8):
        out |= v[:, :, i] << (4 * i)
    return out


@pytest.fixture
def synthetic(tmp_path):
    """A base BF16 model and a compressed-tensors W4AFP8 'checkpoint' of it."""
    torch.manual_seed(7)
    base_dir = tmp_path / "base"
    ckpt_dir = tmp_path / "run" / "checkpoint"
    base_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)

    base: dict[str, torch.Tensor] = {}
    ckpt: dict[str, torch.Tensor] = {}

    for layer in (0, 1):
        pref = f"model.layers.{layer}"

        # --- norms: input_layernorm unfolded, post_attention folded by s ---
        base[f"{pref}.input_layernorm.weight"] = torch.rand(HIDDEN).bfloat16() + 0.5
        ckpt[f"{pref}.input_layernorm.weight"] = base[
            f"{pref}.input_layernorm.weight"
        ].clone()

        norm = torch.rand(HIDDEN).bfloat16() + 0.5
        fold = (torch.rand(HIDDEN) * 0.6 + 0.7).bfloat16()  # s in [0.7, 1.3]
        base[f"{pref}.post_attention_layernorm.weight"] = norm
        ckpt[f"{pref}.post_attention_layernorm.weight"] = (
            norm.float() / fold.float()
        ).bfloat16()

        # --- fp8 attention o_proj: unfolded ---
        w = torch.randn(HIDDEN, HIDDEN).bfloat16()
        base[f"{pref}.self_attn.o_proj.weight"] = w
        scale = w.float().abs().amax(dim=1, keepdim=True) / 448.0
        ckpt[f"{pref}.self_attn.o_proj.weight"] = (
            (w.float() / scale).to(torch.float8_e4m3fn)
        )
        ckpt[f"{pref}.self_attn.o_proj.weight_scale"] = scale.bfloat16()

        # --- fp8 shared experts: gate/up FOLDED, down unfolded ---
        for proj, folded in (("gate_proj", True), ("up_proj", True),
                             ("down_proj", False)):
            raw = torch.randn(INTER, HIDDEN).bfloat16()
            base[f"{pref}.mlp.shared_experts.{proj}.weight"] = raw
            effective = raw.float() * (fold.float() if folded else 1.0)
            s = effective.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 448.0
            ckpt[f"{pref}.mlp.shared_experts.{proj}.weight"] = (
                (effective / s).to(torch.float8_e4m3fn)
            )
            ckpt[f"{pref}.mlp.shared_experts.{proj}.weight_scale"] = s.bfloat16()

        # --- router: passthrough ---
        base[f"{pref}.mlp.gate.weight"] = torch.randn(8, HIDDEN).bfloat16()
        ckpt[f"{pref}.mlp.gate.weight"] = base[f"{pref}.mlp.gate.weight"].clone()

        # --- DSA indexer: BF16 in the source, because the AWQ recipe ignores
        # it, but the engine builds wk/wq_b with a quant_config and so demands
        # FP8. weights_proj is built WITHOUT one and must stay BF16. ---
        for name, shape in (("wk", (64, HIDDEN)),
                            ("wq_b", (64, HIDDEN)),
                            ("weights_proj", (4, HIDDEN))):
            key = f"{pref}.self_attn.indexer.{name}.weight"
            base[key] = torch.randn(*shape).bfloat16()
            ckpt[key] = base[key].clone()

        # --- int4 experts ---
        for expert in (0, 1):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                name = f"{pref}.mlp.experts.{expert}.{proj}"
                q = torch.randint(-7, 8, (INTER, HIDDEN), dtype=torch.int8)
                ckpt[f"{name}.weight_packed"] = _pack_int32(q)
                ckpt[f"{name}.weight_scale"] = (
                    torch.rand(INTER, HIDDEN // GROUP) + 0.1
                ).bfloat16()
                ckpt[f"{name}.weight_shape"] = torch.tensor(
                    [INTER, HIDDEN], dtype=torch.int64
                )

    save_file(base, str(base_dir / "model.safetensors"), metadata={"format": "pt"})
    save_file(ckpt, str(ckpt_dir / "model.safetensors"), metadata={"format": "pt"})

    (ckpt_dir / "config.json").write_text(
        json.dumps({
            "architectures": ["GlmMoeDsaForCausalLM"],
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "ignore": ["lm_head", "re:.*mlp[.]gate$"],
            },
        }),
        encoding="utf-8",
    )
    (ckpt_dir.parent / "recipe.json").write_text(
        json.dumps({"fp8_dynamic_targets": FP8_TARGETS}), encoding="utf-8"
    )
    return base_dir, ckpt_dir, tmp_path / "out"


def test_conversion_succeeds_and_emits_the_expected_names(synthetic):
    base_dir, ckpt_dir, out = synthetic
    assert convert(ckpt_dir, base_dir, out, shard_bytes=10**7, unpacker=_unpack_int32) == 0

    index = json.loads((out / "model.safetensors.index.json").read_text())
    keys = set(index["weight_map"])

    # experts: renamed and repacked, weight_shape dropped
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in keys
    assert "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv" in keys
    assert "model.layers.0.mlp.experts.0.gate_proj.weight_packed" not in keys
    assert "model.layers.0.mlp.experts.0.gate_proj.weight_shape" not in keys

    # fp8 rest: block scale replaces the per-channel one
    assert "model.layers.0.self_attn.o_proj.weight_scale_inv" in keys
    assert "model.layers.0.self_attn.o_proj.weight_scale" not in keys

    # passthrough untouched
    assert "model.layers.0.mlp.gate.weight" in keys
    assert "model.layers.0.input_layernorm.weight" in keys


def test_expert_weights_survive_the_repack_bit_exactly(synthetic):
    """Repacking is a pure re-encoding: every int4 value must come back."""
    from safetensors import safe_open

    base_dir, ckpt_dir, out = synthetic
    assert convert(ckpt_dir, base_dir, out, shard_bytes=10**7, unpacker=_unpack_int32) == 0

    name = "model.layers.0.mlp.experts.0.gate_proj"
    index = json.loads((out / "model.safetensors.index.json").read_text())
    with safe_open(str(out / index["weight_map"][f"{name}.weight"]),
                   framework="pt") as handle:
        packed = handle.get_tensor(f"{name}.weight")
        scale_out = handle.get_tensor(f"{name}.weight_scale_inv")
    with safe_open(str(ckpt_dir / "model.safetensors"), framework="pt") as handle:
        scale_in = handle.get_tensor(f"{name}.weight_scale")

    assert packed.dtype == torch.int8
    assert packed.shape == (INTER, HIDDEN // 2)
    assert unpack_nibbles_int8(packed).shape == (INTER, HIDDEN)
    # The scale is renamed, never recomputed.
    assert torch.equal(scale_out, scale_in)


def test_block_scale_shape_matches_the_loader_formula(synthetic):
    from safetensors import safe_open

    base_dir, ckpt_dir, out = synthetic
    assert convert(ckpt_dir, base_dir, out, shard_bytes=10**7, unpacker=_unpack_int32) == 0

    name = "model.layers.0.self_attn.o_proj"
    index = json.loads((out / "model.safetensors.index.json").read_text())
    with safe_open(str(out / index["weight_map"][f"{name}.weight"]),
                   framework="pt") as handle:
        weight = handle.get_tensor(f"{name}.weight")
        scale = handle.get_tensor(f"{name}.weight_scale_inv")
    assert weight.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert scale.shape == (HIDDEN // 128, HIDDEN // 128)


def test_the_fold_is_reapplied_to_shared_expert_gate_and_up(synthetic):
    """The core correctness property. Rebuilding shared_experts.gate_proj from
    the raw base weight without re-applying s would leave the output wrong by
    exactly s -- loadable, plausible, and silent."""
    from safetensors import safe_open

    base_dir, ckpt_dir, out = synthetic
    assert convert(ckpt_dir, base_dir, out, shard_bytes=10**7, unpacker=_unpack_int32) == 0

    index = json.loads((out / "model.safetensors.index.json").read_text())
    for proj in ("gate_proj", "up_proj", "down_proj"):
        name = f"model.layers.0.mlp.shared_experts.{proj}"
        with safe_open(str(out / index["weight_map"][f"{name}.weight"]),
                       framework="pt") as handle:
            q = handle.get_tensor(f"{name}.weight").float()
            s = handle.get_tensor(f"{name}.weight_scale_inv")
        rebuilt = q * s.repeat_interleave(128, -2).repeat_interleave(128, -1)
        with safe_open(str(ckpt_dir / "model.safetensors"), framework="pt") as h:
            on_disk = (h.get_tensor(f"{name}.weight").float()
                       * h.get_tensor(f"{name}.weight_scale").float())
        resid = ((rebuilt - on_disk).norm() / on_disk.norm()).item()
        # Two independent e4m3 roundings ~= 0.037. A dropped fold on gate/up
        # would land near |1-s| ~ 0.15 for s in [0.7, 1.3].
        assert resid < 0.06, f"{proj}: {resid}"


def test_dropped_fold_would_be_caught(synthetic, monkeypatch, capsys):
    """Prove the cross-check has teeth by disabling the fold and requiring a
    non-zero exit -- otherwise the guard is decoration.

    Asserts the REASON as well as the exit code: a test satisfied by any
    non-zero exit would also pass if the converter crashed for an unrelated
    reason, which is how a guard quietly stops guarding.
    """
    import pipeline.to_sglang_w4afp8 as mod

    base_dir, ckpt_dir, out = synthetic
    monkeypatch.setattr(
        mod, "recover_fold_scale", lambda b, c: torch.ones_like(b.float())
    )
    assert mod.convert(ckpt_dir, base_dir, out, shard_bytes=10**7, unpacker=_unpack_int32) == 1
    captured = capsys.readouterr().out
    assert "rebuilt non-expert weights disagree" in captured, captured
    # And the disagreement must be of fold magnitude (s in [0.7, 1.3]), not a
    # marginal overshoot of the bound.
    line = next(l for l in captured.splitlines() if "rebuilt-vs-ondisk" in l)
    worst = float(line.split("max=")[1].split()[0])
    assert worst > 0.08, line


def test_recover_fold_scale_inverts_the_norm_division():
    norm = torch.rand(64).bfloat16() + 0.5
    fold = (torch.rand(64) * 0.6 + 0.7).bfloat16()
    divided = (norm.float() / fold.float()).bfloat16()
    recovered = recover_fold_scale(norm, divided)
    # BF16 has 8 mantissa bits, so ~0.4% relative spacing on each of two values.
    assert torch.allclose(recovered, fold.float(), rtol=0.02)


def test_recover_fold_scale_survives_a_zero_gain():
    norm = torch.tensor([1.0, 0.0, 2.0]).bfloat16()
    ckpt = torch.tensor([0.5, 0.0, 1.0]).bfloat16()
    recovered = recover_fold_scale(norm, ckpt)
    assert torch.isfinite(recovered).all()
    assert recovered[1].item() == 1.0  # pinned, not nan


_IGNORE_SRC = {
    "architectures": ["X"],
    "quantization_config": {
        "ignore": ["lm_head", "re:.*mlp[.]gate$", "model.layers.0.mlp.gate"],
    },
}


def test_build_config_drops_regex_ignores_when_it_cannot_resolve_them():
    """is_layer_skipped does prefix matching, so an unresolved `re:` pattern
    would match nothing and silently un-ignore a module. With no module list to
    resolve against, dropping it is the least-bad option -- emitting it would
    look like coverage while providing none."""
    out = build_config(_IGNORE_SRC)
    assert out["quantization_config"]["quant_method"] == "w4afp8"
    assert out["quantization_config"]["group_size"] == 128
    assert out["quantization_config"]["ignored_layers"] == [
        "lm_head", "model.layers.0.mlp.gate"
    ]
    assert out["architectures"] == ["X"]


def test_build_config_expands_regex_ignores_against_real_module_names():
    """The reason dropping is not good enough.

    Whether the source ALSO lists every regex match literally is a property of
    the llm-compressor version that wrote the checkpoint. Here layer 1's gate is
    matched only by the pattern, so dropping it would leave a BF16 Linear the
    loader does not know to skip.
    """
    out = build_config(_IGNORE_SRC, module_names=[
        "model.layers.0.mlp.gate",
        "model.layers.1.mlp.gate",
        "model.layers.1.self_attn.o_proj",
        "model.layers.1.mlp.experts.0.gate_proj",
    ])
    ignored = out["quantization_config"]["ignored_layers"]
    assert "model.layers.1.mlp.gate" in ignored
    # Literals stay, and stay first; no duplicate for the layer-0 gate that both
    # the literal list and the pattern name.
    assert ignored[:2] == ["lm_head", "model.layers.0.mlp.gate"]
    assert ignored.count("model.layers.0.mlp.gate") == 1
    # The pattern must not drag in modules it does not match.
    assert "model.layers.1.self_attn.o_proj" not in ignored
    assert "model.layers.1.mlp.experts.0.gate_proj" not in ignored


def test_build_config_expansion_only_matches_modules_that_exist():
    """A pattern naming a layer the checkpoint does not carry (layer 78, the MTP
    block, until it is grafted) must resolve to nothing rather than be copied
    through as an unusable pattern."""
    out = build_config(
        {"quantization_config": {"ignore": ["re:.*layers[.]78[.].*"]}},
        module_names=["model.layers.0.mlp.gate"],
    )
    assert out["quantization_config"]["ignored_layers"] == []


def test_layer_subset_and_dry_run(synthetic):
    base_dir, ckpt_dir, out = synthetic
    assert convert(ckpt_dir, base_dir, out, dry_run=True, unpacker=_unpack_int32) == 0
    assert not out.exists() or not any(out.iterdir())

    assert convert(ckpt_dir, base_dir, out, layers=[1], shard_bytes=10**7, unpacker=_unpack_int32) == 0
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert not any(".layers.0." in k for k in index["weight_map"])
    assert any(".layers.1." in k for k in index["weight_map"])


def test_index_total_size_matches_bytes_on_disk(synthetic):
    base_dir, ckpt_dir, out = synthetic
    assert convert(ckpt_dir, base_dir, out, shard_bytes=10**6, unpacker=_unpack_int32) == 0
    index = json.loads((out / "model.safetensors.index.json").read_text())

    from safetensors import safe_open

    actual = 0
    for shard in sorted({v for v in index["weight_map"].values()}):
        with safe_open(str(out / shard), framework="pt") as handle:
            for key in handle.keys():
                t = handle.get_tensor(key)
                actual += t.numel() * t.element_size()
    assert index["metadata"]["total_size"] == actual


def test_shards_are_named_with_the_final_count(synthetic):
    base_dir, ckpt_dir, out = synthetic
    assert convert(ckpt_dir, base_dir, out, shard_bytes=10**6, unpacker=_unpack_int32) == 0
    shards = sorted(p.name for p in out.glob("*.safetensors"))
    assert shards, "no shards written"
    assert not any("SHARDS" in name for name in shards)
    total = len(shards)
    for name in shards:
        assert name.endswith(f"of-{total:05d}.safetensors"), name


def test_indexer_wk_and_wq_b_are_fp8_even_though_the_recipe_ignored_them(synthetic):
    """The bug that cost a 2.5 h conversion re-run.

    SGLang's W4AFp8Config.from_config never passes ignored_layers to the
    constructor, so self.ignored_layers is [] for every checkpoint and
    is_layer_skipped always returns False. Every LinearBase gets
    Fp8LinearMethod. dsa_indexer.py builds wk and wq_b with quant_config and
    weights_proj without one, so the first two need weight_scale_inv and the
    third must not have it -- regardless of what any config field says.

    PhalaCloud/GLM-5.3-W4AFP8 does exactly this: its config declares
    modules_to_not_convert including "indexer", and wk/wq_b carry
    weight_scale_inv anyway.
    """
    base_dir, ckpt_dir, out = synthetic
    assert convert(ckpt_dir, base_dir, out, shard_bytes=10**7,
                   unpacker=_unpack_int32) == 0
    keys = set(json.loads(
        (out / "model.safetensors.index.json").read_text())["weight_map"])

    for layer in (0, 1):
        pref = f"model.layers.{layer}.self_attn.indexer"
        for name in ("wk", "wq_b"):
            assert f"{pref}.{name}.weight" in keys
            assert f"{pref}.{name}.weight_scale_inv" in keys, (
                f"{name} has no scale; Fp8LinearMethod would fail to load it")
        # Built without a quant_config, so a scale here would be wrong.
        assert f"{pref}.weights_proj.weight" in keys
        assert f"{pref}.weights_proj.weight_scale_inv" not in keys


def test_indexer_weights_are_actually_e4m3_not_copied_bf16(synthetic):
    """Presence of a scale is not enough -- the weight must be re-encoded."""
    from safetensors import safe_open

    base_dir, ckpt_dir, out = synthetic
    assert convert(ckpt_dir, base_dir, out, shard_bytes=10**7,
                   unpacker=_unpack_int32) == 0
    wmap = json.loads(
        (out / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.layers.0.self_attn.indexer.wk.weight"
    with safe_open(str(out / wmap[key]), framework="pt") as handle:
        assert handle.get_slice(key).get_dtype() == "F8_E4M3"
    plain = "model.layers.0.self_attn.indexer.weights_proj.weight"
    with safe_open(str(out / wmap[plain]), framework="pt") as handle:
        assert handle.get_slice(plain).get_dtype() == "BF16"


def test_expert_input_scale_uses_the_shard_ids_sglang_looks_for(synthetic):
    """gate_proj -> w1, up_proj -> w3, down_proj -> w2.

    make_expert_input_scale_params_mapping builds checkpoint names as
    experts.{i}.{w1,w2,w3}.input_scale, NOT gate_proj/up_proj/down_proj. The
    old names matched nothing, so the tensors were dead weight -- harmless only
    because the parameters are pre-filled with ones. The vendor release uses
    w1/w2/w3 for input_scale while keeping gate_proj/up_proj/down_proj for the
    weights; an upstream inconsistency we have to match.
    """
    base_dir, ckpt_dir, out = synthetic
    assert convert(ckpt_dir, base_dir, out, shard_bytes=10**7,
                   unpacker=_unpack_int32) == 0
    keys = set(json.loads(
        (out / "model.safetensors.index.json").read_text())["weight_map"])

    pref = "model.layers.0.mlp.experts.0"
    assert f"{pref}.w1.input_scale" in keys
    assert f"{pref}.w3.input_scale" in keys
    assert f"{pref}.w2.input_scale" in keys
    # The names that never matched must be gone.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert f"{pref}.{proj}.input_scale" not in keys
        # ...while the WEIGHTS keep the proj names.
        assert f"{pref}.{proj}.weight" in keys


def test_conversion_fails_closed_if_indexer_modules_stop_matching(synthetic,
                                                                  monkeypatch):
    """A silent miss here produces an artifact that cannot load, so the
    converter must refuse rather than fall through to the copy path."""
    import pipeline.to_sglang_w4afp8 as mod

    base_dir, ckpt_dir, out = synthetic
    # Patch the SELECTOR, not _ENGINE_FP8_SUFFIXES. Patching the constant would
    # also empty the set of modules the check expects to find, so the two would
    # agree at zero and the guard could never fire -- the check has to notice
    # that modules exist while nothing selected them.
    monkeypatch.setattr(mod.Plan, "is_engine_fp8", lambda self, module: False)
    assert mod.convert(ckpt_dir, base_dir, out, shard_bytes=10**7,
                       unpacker=_unpack_int32) == 2
