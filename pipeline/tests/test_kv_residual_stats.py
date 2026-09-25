from __future__ import annotations

import torch

from pipeline import kv_residual_stats as rs


def test_residual_stats_on_clustered_latents(tmp_path):
    g = torch.Generator().manual_seed(0)
    C = torch.randn(8, 256, generator=g) * 4
    X = C[torch.randint(0, 8, (512,), generator=g)] + 0.1 * torch.randn(512, 256, generator=g)
    X[0] *= 0.0                                                   # sink: picks the zero centroid
    km = torch.full((512,), 0.5 / 511)
    km[0] = 0.5
    torch.save({"X": [X.half()], "key_mass": [km], "cb8": C.half()}, tmp_path / "009.pt")
    rs.main([str(tmp_path), "--cb", "cb8"])
    s = rs.layer_stats(torch.load(tmp_path / "009.pt"), "cb8")
    assert s["residual_over_token_rest"]["p50"] < 0.05             # tokens sit on their centroid
    assert s["zero_anchor_fraction"] == 1 / 512 and abs(s["attn_mass_on_zero_anchor_tokens"] - 0.5) < 1e-6
    # a single 20x spike in one 128-token channel group dominates that group's crest factor
    V = torch.ones(128, 128)
    V[5, 7] = 20.0
    spike = 20 / ((127 + 400) / 128) ** 0.5
    assert abs(rs.crest(V, "ch")[7, 0] - spike) < 1e-4 and abs(rs.crest(V, "tok")[5, 0] - spike) < 1e-4
    assert rs.crest(V, "ch")[6, 0] == 1.0
    assert (tmp_path / "residual_stats_cb8.json").exists()
