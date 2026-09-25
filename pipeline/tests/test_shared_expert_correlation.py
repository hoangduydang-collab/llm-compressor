from __future__ import annotations

import json

import pytest
import torch

from pipeline import shared_expert_correlation as sec

H, I, E = 96, 32, 4


def _shared(g):
    return {"gate": torch.randn(I, H, generator=g), "up": torch.randn(I, H, generator=g),
            "down": torch.randn(H, I, generator=g)}


def _permuted_copies(S, g, noise=0.02):
    """Routed expert e = shared with neurons shuffled and rescaled, plus noise."""
    R = {p: [] for p in sec.PROJS}
    for _ in range(E):
        perm = torch.randperm(I, generator=g)
        scale = torch.randn(I, generator=g).sign() * (0.5 + torch.rand(I, generator=g))
        R["gate"].append(S["gate"][perm] * scale[:, None] + noise * torch.randn(I, H, generator=g))
        R["up"].append(S["up"][perm] * scale[:, None] + noise * torch.randn(I, H, generator=g))
        R["down"].append(S["down"][:, perm] * scale[None, :] + noise * torch.randn(H, I, generator=g))
    return {p: torch.stack(v).to(torch.bfloat16) for p, v in R.items()}


def _run(R, S, monkeypatch):
    monkeypatch.setattr(sec, "SUBSPACE_KS", (I,))
    return sec.analyze(R, {p: v.to(torch.bfloat16) for p, v in S.items()}, torch.device("cpu"),
                       list(range(E)), list(range(E)))


def test_detects_shuffled_rescaled_copies(monkeypatch):
    g = torch.Generator().manual_seed(0)
    S = _shared(g)
    r = _run(_permuted_copies(S, g), S, monkeypatch)
    assert r["A_neuron_max_abs_cos"]["vs_shared"]["p50"] > 0.95
    assert r["C_delta_residual_energy"]["vs_shared"]["max"] < 0.01
    # routed rows lie in the shared expert's I-dim row space
    assert r["B_subspace_energy"]["gate"][f"k{I}"]["routed_in_shared"] > 0.99


def test_unrelated_experts_look_like_noise(monkeypatch):
    g = torch.Generator().manual_seed(1)
    S = _shared(g)
    R = {p: torch.randn((E, *S[p].shape), generator=g).to(torch.bfloat16) for p in sec.PROJS}
    r = _run(R, S, monkeypatch)
    assert r["A_neuron_max_abs_cos"]["vs_shared"]["p50"] < 0.25
    assert r["C_delta_residual_energy"]["vs_shared"]["p50"] > 0.8
    b = r["B_subspace_energy"]["gate"][f"k{I}"]
    assert b["routed_in_shared"] == pytest.approx(b["isotropic"], abs=0.1)


def test_delta_residual_counts_unmatched_neurons():
    g = torch.Generator().manual_seed(2)
    base = torch.randn(4, 16, generator=g)
    Ne = torch.cat([base, torch.randn(4, 16, generator=g)])     # 8 neurons, only 4 base
    r = sec.delta_residual(Ne, base)
    assert r == pytest.approx(Ne[4:].pow(2).sum().item() / Ne.pow(2).sum().item(), rel=1e-4)


def _fake_ckpt(tmp, with_shared=True):
    from safetensors.torch import save_file

    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "config.json").write_text(json.dumps({"n_routed_experts": E, "first_k_dense_replace": 1}))
    g = torch.Generator().manual_seed(0)
    t = {}
    names = [f"experts.{e}" for e in range(E)] + (["shared_experts"] if with_shared else [])
    for n in names:
        for p, shape in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I))):
            t[f"model.layers.1.mlp.{n}.{p}_proj.weight"] = torch.randn(shape, generator=g).to(torch.bfloat16)
    save_file(t, str(tmp / "model-00000.safetensors"))
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00000.safetensors" for k in t}}))
    return tmp


def test_end_to_end_and_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(sec, "SUBSPACE_KS", (16, I))
    ck = _fake_ckpt(tmp_path / "ck")
    out = tmp_path / "out"
    sec.main(["--ckpt", str(ck), "--out", str(out), "--device", "cpu",
              "--subspace-experts", "2", "--match-experts", "2", "--control-experts", "3"])
    res = json.loads((out / "shared_expert_results.json").read_text())
    assert res["meta"]["shared_shapes"]["down"] == [H, I]
    assert set(res["results"]) >= {"A_neuron_max_abs_cos", "B_subspace_energy", "C_delta_residual_energy"}
    assert "vs shared" in (out / "shared_expert_summary.md").read_text()
    with pytest.raises(SystemExit):
        sec.load_shared(str(_fake_ckpt(tmp_path / "noshared", with_shared=False)), 1)
