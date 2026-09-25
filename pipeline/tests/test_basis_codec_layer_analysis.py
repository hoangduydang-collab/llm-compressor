from __future__ import annotations

import json

import pytest
import torch

from pipeline import basis_codec_layer_analysis as bc


def _rel(xq, x):
    return ((xq - x).pow(2).sum() / x.pow(2).sum()).item()


def test_int4_on_gaussian_matches_theory():
    # Optimal 16-level uniform quantizer on N(0,1) has MSE 0.01154 (Max 1960);
    # per-group scaling with a clip search lands close to it.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, 128, generator=g)
    e4 = _rel(bc.quant_groups(x, 4), x)
    e3 = _rel(bc.quant_groups(x, 3), x)
    assert 0.009 < e4 < 0.016
    assert 3.0 < e3 / e4 < 5.0


def test_values_on_grid_are_exact():
    q = torch.arange(-8, 8, dtype=torch.float32).repeat(8)[:128]
    x = (q * 0.5).unsqueeze(0)
    assert bc.quant_groups(x, 4).equal(x)


def test_zero_bits_and_sign_quantizer():
    x = torch.randn(10, 128)
    assert bc.quant_groups(x, 0).abs().sum() == 0
    xq = bc.quant_groups(x, 1)
    assert torch.equal(torch.sign(xq), torch.sign(x))


def test_quant_dirs_groups_within_each_row():
    # a huge value in row 0 must not coarsen row 1's scale
    y = torch.randn(2, 200) * torch.tensor([[1000.0], [1.0]])
    yq = bc.quant_dirs(y, 4)
    assert yq.shape == y.shape
    assert _rel(yq[1:], y[1:]) < 0.02


def test_allocate_bits():
    b = bc.allocate_bits(torch.tensor([16.0, 1.0, 1.0, 1.0]), 12)
    assert sum(b) == 12 and b[0] == max(b) and b[0] - b[1] == 2
    assert bc.allocate_bits(torch.ones(8), 24) == [3] * 8


def test_dense_pca_gains_only_with_structure(monkeypatch):
    monkeypatch.setattr(bc, "DENSE_DIMS", (16,))
    g = torch.Generator().manual_seed(0)
    E, m, n = 2, 64, 256
    iid = torch.randn(E, m, n, generator=g).to(torch.bfloat16)
    # every 16-block lives (mostly) in a shared 2-D subspace
    basis = torch.linalg.qr(torch.randn(16, 16, generator=g))[0][:, :2]
    coef = torch.randn(E, m, n // 16, 2, generator=g)
    low = ((coef @ basis.T).reshape(E, m, n) + 0.01 * torch.randn(E, m, n, generator=g)).to(torch.bfloat16)
    r_iid = bc.part_dense(iid, torch.device("cpu"))["d16"]
    r_low = bc.part_dense(low, torch.device("cpu"))["d16"]
    assert r_iid["bound_bits_saved_amgm"] < 0.05
    assert r_low["bound_bits_saved_amgm"] > 2.0
    assert r_low["pca@3.0"]["rel_mse"] < 0.25 * r_low["ident@3.0"]["rel_mse"]
    assert r_iid["pca@3.0"]["rel_mse"] == pytest.approx(r_iid["ident@3.0"]["rel_mse"], rel=0.15)


def test_omp_recovers_exact_sparse_combinations():
    # orthonormal atoms: greedy OMP recovery is guaranteed (random overcomplete
    # dictionaries are too coherent for exact 2-sparse recovery)
    g = torch.Generator().manual_seed(0)
    D = torch.linalg.qr(torch.randn(32, 32, generator=g))[0][:, :24].T.contiguous()
    idx = torch.stack([torch.randperm(24, generator=g)[:2] for _ in range(500)])
    coef = torch.randn(500, 2, generator=g).abs() + 0.5
    X = bc.reconstruct(D, idx, coef)
    i2, c2 = bc.omp(X, D, 2)
    assert _rel(bc.reconstruct(D, i2, c2), X) < 1e-8


def test_dictionary_fit_beats_random_init():
    g = torch.Generator().manual_seed(0)
    Dt = torch.randn(32, 8, generator=g)
    Dt = Dt / Dt.norm(dim=1, keepdim=True)
    idx = torch.stack([torch.randperm(32, generator=g)[:2] for _ in range(4000)])
    X = bc.reconstruct(Dt, idx, torch.randn(4000, 2, generator=g))
    D = bc.fit_dictionary(X, 32, 2, 10, torch.Generator().manual_seed(1))
    i2, c2 = bc.omp(X, D, 2)
    assert _rel(bc.reconstruct(D, i2, c2), X) < 0.05
    with pytest.raises(ValueError):
        bc.fit_dictionary(X[:10], 32, 2, 1, g)


def test_eigen_experts_two_generators():
    g = torch.Generator().manual_seed(0)
    A, B = torch.randn(2, 32, 64, generator=g)
    mix = torch.randn(8, 2, generator=g)
    W = (mix[:, 0, None, None] * A + mix[:, 1, None, None] * B).to(torch.bfloat16)
    r = bc.part_experts(W, torch.device("cpu"), chunk=512)
    assert r["raw"]["K2"] < 1e-3 and r["raw"]["K1"] > 0.01


def _fake_ckpt(tmp, E=4, h=256, inter=128, drop=None, quantized=False):
    from safetensors.torch import save_file

    tmp.mkdir(parents=True, exist_ok=True)
    cfg = {"model_type": "glm_moe_dsa", "n_routed_experts": E, "first_k_dense_replace": 1}
    if quantized:
        cfg["quantization_config"] = {}
    (tmp / "config.json").write_text(json.dumps(cfg))
    g = torch.Generator().manual_seed(0)
    shards, wmap = [{}, {}], {}
    for e in range(E):
        for p, shape in (("gate", (inter, h)), ("up", (inter, h)), ("down", (h, inter))):
            k = f"model.layers.1.mlp.experts.{e}.{p}_proj.weight"
            if k == drop:
                continue
            shards[e % 2][k] = torch.randn(shape, generator=g).to(torch.bfloat16)
            wmap[k] = f"model-{e % 2:05d}.safetensors"
    shards[0]["model.layers.1.mlp.gate.weight"] = torch.zeros(E, h, dtype=torch.bfloat16)
    wmap["model.layers.1.mlp.gate.weight"] = "model-00000.safetensors"
    for i, s in enumerate(shards):
        save_file(s, str(tmp / f"model-{i:05d}.safetensors"))
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}))
    return tmp


def test_end_to_end_on_fake_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "SPARSE_CONFIGS", ((8, 16, 2, 4),))
    monkeypatch.setattr(bc, "DENSE_DIMS", (8, 16))
    ck = _fake_ckpt(tmp_path / "ck")
    out = tmp_path / "out"
    bc.main(["--ckpt", str(ck), "--out", str(out), "--device", "cpu", "--eval-row-stride", "8",
             "--n-fit", "2048", "--mod-iters", "2", "--control-experts", "2"])
    res = json.loads((out / "results.json").read_text())
    assert res["meta"]["layer"] == 1 and res["meta"]["experts"] == 4
    assert res["meta"]["shapes"] == {"gate": [128, 256], "up": [128, 256], "down": [256, 128]}
    for p in bc.PROJS:
        r = res["results"][p]
        assert set(r) == {"target", "dense", "sparse", "experts"}
        assert r["sparse"]["d8_K16_s2_c4"]["bpw"] == pytest.approx(2 * (4 + 4) / 8 + 2 / 8 * 16 / 128
                                                                   + 16 * 8 * 16 / (4 * 128 * 256))
    assert "INT4-g128" in (out / "summary.md").read_text()


@pytest.mark.parametrize("kw", [{"drop": "model.layers.1.mlp.experts.2.up_proj.weight"}, {"quantized": True}])
def test_loader_fails_closed(tmp_path, kw):
    ck = _fake_ckpt(tmp_path, **kw)
    with pytest.raises(SystemExit):
        bc.load_layer(str(ck), None, None, 2)


def test_loader_rejects_dense_layer(tmp_path):
    with pytest.raises(SystemExit, match="dense"):
        bc.load_layer(str(_fake_ckpt(tmp_path)), 0, None, 2)
