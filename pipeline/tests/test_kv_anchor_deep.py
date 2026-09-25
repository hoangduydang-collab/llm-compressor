from __future__ import annotations

import json

import pytest
import torch

from pipeline import kv_anchor_deep as kd
from pipeline import kv_anchor_probe as kp
from pipeline.sglang_w4afp8_kernels import pack_nibbles_int8, quantize_block_fp8


def _tiny_ckpt(tmp_path):
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    torch.manual_seed(0)
    base = GlmMoeDsaConfig(indexer_types=["full", "shared"] * 39)   # exercise a shared-indexer layer
    d = tmp_path / "tiny"
    GlmMoeDsaForCausalLM(kp.tiny_config(base)).to(torch.bfloat16).save_pretrained(d)
    return d


def test_dequantize_sglang_w4afp8_roundtrip():
    g = torch.Generator().manual_seed(0)
    q = torch.randint(-8, 8, (256, 512), generator=g)
    scale = torch.rand(256, 4, generator=g).to(torch.bfloat16) + 0.1
    w = torch.randn(256, 384, generator=g)
    wq, winv = quantize_block_fp8(w)
    raw = {
        "model.layers.3.mlp.experts.0.gate_proj.weight": pack_nibbles_int8(q),
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv": scale,
        "model.layers.3.mlp.experts.0.w1.input_scale": torch.ones(1),
        "model.layers.3.self_attn.o_proj.weight": wq,
        "model.layers.3.self_attn.o_proj.weight_scale_inv": winv,
        "model.layers.3.input_layernorm.weight": torch.ones(512, dtype=torch.bfloat16),
    }
    out, fp8 = kd.dequantize_sglang_w4afp8(raw)
    assert set(out) == {"model.layers.3.mlp.experts.0.gate_proj.weight", "model.layers.3.self_attn.o_proj.weight",
                        "model.layers.3.input_layernorm.weight"}
    want = (q.float() * scale.float().repeat_interleave(128, dim=1)).to(torch.bfloat16)
    assert torch.equal(out["model.layers.3.mlp.experts.0.gate_proj.weight"], want)
    o = out["model.layers.3.self_attn.o_proj.weight"].float()
    assert ((o - w).pow(2).sum() / w.pow(2).sum()).item() < 2e-3          # fp8 rounding only
    assert fp8 == {"model.layers.3.self_attn.o_proj"}
    with pytest.raises(ValueError):
        kd.dequantize_sglang_w4afp8({"a.weight": torch.zeros(2, 2), "a.weight_scale_inv": torch.ones(1)})


def test_layer_config_slices_per_layer_lists(tmp_path):
    d = tmp_path / "ck"
    d.mkdir()
    cfg = {"num_hidden_layers": 78, "num_nextn_predict_layers": 1, "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
           "indexer_types": ["full", "shared"] * 39, "layer_types": ["x"] * 79, "quantization_config": {"q": 1}}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"model.layers.5.a.weight": "s0"}}))
    c = kd.LayerSource(str(d), str(tmp_path / "st")).layer_config(5)
    assert c["num_hidden_layers"] == 1 and c["mlp_layer_types"] == ["sparse"] and c["indexer_types"] == ["shared"]
    assert c["layer_types"] == ["x"] and "quantization_config" not in c and c["num_nextn_predict_layers"] == 0


def test_range_reads_match_safetensors(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file

    d = tmp_path / "ck"
    d.mkdir()
    g = torch.Generator().manual_seed(3)
    t = {"model.layers.0.a.weight": torch.randn(300, 7, generator=g).to(torch.bfloat16),
         "model.layers.1.b.weight": torch.randint(-128, 127, (1000,), generator=g).to(torch.int8),
         "model.layers.1.c.weight": torch.randn(64, 64, generator=g).to(torch.float8_e4m3fn),
         "model.embed_tokens.weight": torch.randn(10, 4, generator=g)}
    save_file(t, str(d / "s0.safetensors"))
    (d / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "s0.safetensors" for k in t}}))
    src = kd.LayerSource(str(d), str(tmp_path / "st"))
    got = src.read(list(t))
    with safe_open(str(d / "s0.safetensors"), framework="pt") as h:
        for k in t:
            assert got[k].dtype == t[k].dtype and torch.equal(got[k].view(torch.uint8), h.get_tensor(k).view(torch.uint8))
    src.prefetch(1)
    assert set(src.raw_layer(1)) == {"model.layers.1.b.weight", "model.layers.1.c.weight"}
    assert kd.read_span(d / "s0.safetensors", 8, 5000, threads=3, chunk=512) == (d / "s0.safetensors").read_bytes()[8:5000]


def test_preflight_stream_matches_full_forward(tmp_path):
    assert kd.preflight_stream(None, torch.device("cpu"), tmp_path) < 1e-6


def test_load_layer_fails_closed_on_missing_weights(tmp_path, monkeypatch):
    src = kd.LayerSource(str(_tiny_ckpt(tmp_path)), str(tmp_path / "st"))
    raw = src.raw_layer
    monkeypatch.setattr(src, "raw_layer", lambda L: {k: v for k, v in raw(L).items() if "kv_b_proj" not in k})
    with pytest.raises(RuntimeError, match="did not load cleanly"):
        kd.load_layer(src, 0, torch.device("cpu"))


def test_act_fp8_hooks_change_outputs_only_when_enabled(tmp_path):
    src = kd.LayerSource(str(_tiny_ckpt(tmp_path)), str(tmp_path / "st"))
    h = torch.randn(1, 64, 256, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    layer, rot, n0 = kd.load_layer(src, 3, torch.device("cpu"))
    src.dequant = False
    build = src.build
    src.build = lambda L: (build(L)[0], {"model.layers.0.self_attn.kv_a_proj_with_mqa"})
    layer8, rot8, n8 = kd.load_layer(src, 3, torch.device("cpu"), act_fp8=True)
    assert n0 == 0 and n8 == 2                               # one FP8 linear + the expert bank
    a = kd.run_layer(layer, rot, h, torch.zeros(1, 64, 32, dtype=torch.int32))[0]
    b = kd.run_layer(layer8, rot8, h, torch.zeros(1, 64, 32, dtype=torch.int32))[0]
    assert not torch.equal(a, b) and ((a - b).float().pow(2).sum() / a.float().pow(2).sum()).item() < 0.05


def test_stream_end_to_end_and_resume(tmp_path):
    ck = _tiny_ckpt(tmp_path)
    src = kd.LayerSource(str(ck), str(tmp_path / "st"))
    g = torch.Generator().manual_seed(2)
    mk = lambda: torch.randint(0, 1000, (256,), generator=g)
    windows = {"train": [mk() for _ in range(3)], "eval": [mk() for _ in range(2)], "gen": [mk()]}
    kw = dict(cbs=(8, 16), kmeans_iters=2, lexico_every=2, lexico_atoms=32, lexico_s=4, lexico_fit_tokens=512,
              lexico_iters=2, qvg_c=8, state_every=2)
    out = tmp_path / "out"
    res = kd.stream(src, windows, torch.device("cpu"), out, **kw)
    assert sorted(res) == [0, 1, 2, 3]
    for L, r in res.items():
        assert {"eval", "gen"} <= set(r) and r["lexico"] == (L in (0, 2, 3))
        arms = r["eval"]["arms"]
        assert "qvg8-int2" in arms and "cb16-ch-int2" in arms and arms["direct-ch-int4"]["attn_rel_mse"] >= 0
    # resume from the state saved after layer 1: layers 2-3 must be recomputed identically
    first = {L: res[L]["eval"]["arms"]["direct-ch-int4"]["attn_rel_mse"] for L in (2, 3)}
    for L in (2, 3):
        (out / "layers" / f"{L:03d}.json").unlink()
    again = kd.stream(kd.LayerSource(str(ck), str(tmp_path / "st2")), windows, torch.device("cpu"), out, **kw)
    assert sorted(again) == [2, 3]
    for L in (2, 3):
        assert again[L]["eval"]["arms"]["direct-ch-int4"]["attn_rel_mse"] == pytest.approx(first[L], rel=1e-6)

    # diagnostics: sink-exempt variants of the key arms, attention mass on token 0
    d = kd.stream(kd.LayerSource(str(ck), str(tmp_path / "st3")), windows, torch.device("cpu"), tmp_path / "o3",
                  layers=2, diag=True, dump_layers={1}, **kw)
    e = d[1]["eval"]
    dm = torch.load(tmp_path / "o3" / "dump" / "001.pt")
    assert not (tmp_path / "o3" / "dump" / "000.pt").exists() and len(dm["X"]) == len(windows["eval"])
    d_lat = dm["X"][0].shape[1]
    assert dm["X"][0].shape[0] == 256 and dm["cb16"].shape == (16, d_lat) and dm["key_mass"][0].shape == (256,)
    assert {"cb16-ch-int2|keep1", "cb16-ch-int2|keep4", "gmean-ch-int2|keep4"} <= set(e["arms"])
    assert e["arms"]["cb16-ch-int2|keep1"]["token0_latent_rel_mse"] == 0.0
    assert 0.0 < e["attn_mass_on_token0"] < 1.0 and e["token0_norm_over_median"] > 0
    assert {"cb16z-ch-int2|keep1", "cb16z-tok-int4", "direct-tok-int3", "cb16w1-tok-int2", "cb16w2-ch-int2",
            "cb16zs-tok-int4", "direct-tok-int2|tiny", "cb16z-tok-int3|tiny"} <= set(e["arms"])
    assert 0.0 <= e["tiny_exact_fraction"] <= 1.0
    assert e["arms"]["direct-tok-int2|tiny"]["bits"] >= e["arms"]["direct-tok-int2"]["bits"]
    assert 0.0 <= e["small_token_fraction"] <= 1.0 and 0.0 <= e["attn_mass_on_small_tokens"] <= 1.0
    assert 0.0 <= e["zero_anchor_fraction"] <= 1.0 and e["token0_prenorm_rms_over_median"] > 0
    assert e["token0_prenorm_rms_over_sqrt_eps"] > 0 and d[1]["kv_norm_gain_abs"]["max"] > 0
    # sink_keep: token 0 exact in every arm, and no |keep variants
    s = kd.stream(kd.LayerSource(str(ck), str(tmp_path / "st4")), windows, torch.device("cpu"), tmp_path / "o4",
                  layers=1, sink_keep=2, diag=True, **kw)[0]["eval"]["arms"]
    assert all(v["token0_latent_rel_mse"] == 0.0 for v in s.values()) and not any("|keep" in n for n in s)
