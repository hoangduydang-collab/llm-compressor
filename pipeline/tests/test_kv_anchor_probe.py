from __future__ import annotations

import math

import pytest
import torch

from pipeline import kv_anchor_probe as kv


def _rel(a, b):
    return ((a - b).pow(2).sum() / b.pow(2).sum()).item()


def test_chunk_predictors_are_exact_on_matching_structure():
    g = torch.Generator().manual_seed(0)
    T, d, N = 64, 128, 16
    const = torch.randn(T // N, 1, d, generator=g).expand(-1, N, -1).reshape(T, d)
    for s in ("first", "mean"):
        P, cost = kv.predict(const, s, N, {})
        assert _rel(P, const) < 1e-3                      # only fp8 anchor error left
        assert cost == pytest.approx((8 * d + 16) / (N * d))
    dirs = torch.randn(T // N, 1, d, generator=g)
    rank1 = (torch.randn(T // N, N, 1, generator=g) * dirs).reshape(T, d)
    P, cost = kv.predict(rank1, "rank1", N, {})
    assert _rel(P, rank1) < 1e-3 and cost > (8 * d + 16) / (N * d)
    with pytest.raises(ValueError):
        kv.predict(torch.randn(60, d), "mean", N, {})


def test_quant_residual_groupings_keep_shape_and_isolate_outliers():
    g = torch.Generator().manual_seed(1)
    R = torch.randn(256, 128, generator=g)
    R[:, 0] *= 1000                                        # one outlier channel
    tok = kv.quant_residual(R, "tok", 4)
    ch = kv.quant_residual(R, "ch", 4)
    assert tok.shape == ch.shape == R.shape
    # per-channel groups isolate the outlier channel; per-token groups do not
    assert _rel(ch[:, 1:], R[:, 1:]) < 0.1 * _rel(tok[:, 1:], R[:, 1:])


def test_kmeans_finds_clusters_and_codebook_arm_costs_index_bits():
    g = torch.Generator().manual_seed(2)
    centers = torch.randn(8, 128, generator=g) * 10
    X = centers[torch.randint(0, 8, (4000,), generator=g)] + 0.01 * torch.randn(4000, 128, generator=g)
    C = kv.kmeans(X, 8, 10, torch.Generator().manual_seed(0))
    assert _rel(C[kv.nearest(X, C)], X) < 1e-3
    P, cost = kv.predict(X, "cb8", None, {"cb8": C})
    assert cost == pytest.approx(3 / 128)
    with pytest.raises(ValueError):
        kv.kmeans(X[:4], 8, 1, g)


def test_arm_bit_accounting():
    X = torch.randn(128, 128)
    ctx = {"mu": X.mean(0, keepdim=True)}
    assert kv.apply_arm(X, ("fp8_unscaled", None, None, None), ctx)[1] == 8.0
    assert kv.apply_arm(X, ("direct", None, "tok", 3), ctx)[1] == pytest.approx(3 + 16 / 128)
    assert kv.apply_arm(X, ("mean", 16, "ch", 2), ctx)[1] == pytest.approx(2 + 16 / 128 + (8 * 128 + 16) / (16 * 128))
    names = {kv.arm_name(a) for a in kv.all_arms()}
    assert "mean-N16-ch-int2" in names and "gmean-tok-int4" in names and "fp8_tok" in names
    attn = kv.attn_arms(kv.all_arms())
    assert all(a[1] in (None, *kv.ATTN_CHUNKS) for a in attn) and len(attn) < len(kv.all_arms())


def test_lag_correlation_of_ar1_process():
    g = torch.Generator().manual_seed(3)
    rho, T, d = 0.9, 20000, 128
    x = torch.zeros(T, d)
    noise = torch.randn(T, d, generator=g)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + math.sqrt(1 - rho ** 2) * noise[t]
    acc = {}
    kv.lag_stats(x, torch.zeros(1, d), acc)
    for L in (1, 4, 16):
        c = acc[L][0] / math.sqrt(acc[L][1] * acc[L][2])
        assert c == pytest.approx(rho ** L, abs=0.02)


def test_chunk_clustering_contiguous_vs_random():
    S, k = 1024, 64
    contiguous = torch.stack([torch.arange(max(0, s - k + 1), max(0, s - k + 1) + k) for s in range(S)])
    g = torch.Generator().manual_seed(4)
    random = torch.stack([torch.randperm(max(s + 1, k), generator=g)[:k] for s in range(S)])
    c = kv.chunk_clustering(contiguous, chunks=(16,), stride=64)["N16"]
    r = kv.chunk_clustering(random, chunks=(16,), stride=64)["N16"]
    assert c["chunks_per_selected_token"] < 0.1 < r["chunks_per_selected_token"]
    assert c["vs_random_selection"] < 0.3 and r["vs_random_selection"] == pytest.approx(1.0, abs=0.1)


def test_preflight_end_to_end_on_tiny_model():
    res = kv.run_preflight(None, torch.device("cpu"))
    assert set(res["layers"]) == {0, 1, 2, 3}
    r0 = res["layers"][0]
    assert r0["noop_attn_rel"] < 1e-6
    assert r0["arms"]["direct-tok-int4"]["attn_rel_mse"] is not None
    assert r0["arms"]["mean-N8-tok-int4"]["attn_rel_mse"] is None       # 8 is not an attention chunk here
    assert res["dsa_layer0"]["validation_overlap"] >= 0.99


def test_tiny_config_survives_real_token_ids():
    # the real GLM-5.3 config's pad/eos ids are far beyond the tiny vocab
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    real = GlmMoeDsaConfig(pad_token_id=154820, eos_token_id=[154820, 154827], bos_token_id=154822)
    cfg = kv.tiny_config(real)
    assert cfg.pad_token_id < cfg.vocab_size
    GlmMoeDsaForCausalLM(cfg)


def test_aa_lcr_sets_and_windows(tmp_path):
    base = tmp_path / "extracted_text" / "lcr"
    for i in range(30):
        d = base / f"cat{i % 3}" / f"set{i:02d}"
        d.mkdir(parents=True)
        (d / "a.txt").write_text(f"doc {i} " * 50)
    sets = kv.aa_lcr_sets(tmp_path)
    assert len(sets) == 30
    tok = lambda text, add_special_tokens=False: {"input_ids": list(range(len(text.split())))}
    train, evals, names = kv.make_windows(tok, sets, 64, 16, torch.device("cpu"))
    assert len(train) == len(evals) == 15 and all(w.shape == (64,) for w in train + evals)
    (base / "cat0" / "set00" / "a.txt").unlink()
    with pytest.raises(SystemExit):
        kv.aa_lcr_sets(tmp_path)
