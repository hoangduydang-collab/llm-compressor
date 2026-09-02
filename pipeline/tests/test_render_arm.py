"""CPU tests for glm53 quality-arm pod rendering."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "pipeline" / "k8s"))
import render_arm  # noqa: E402


def test_render_arm_persists_aa_gpqa(tmp_path: Path):
    out = tmp_path / "arm.yaml"
    rc = render_arm.main([
        "--arm", "ours",
        "--model", "/mnt/cephfs/hoangduy/results/glm53-w4afp8-mtp/checkpoint",
        "--run-tag", "t",
        "--ref", "deadbeef",
        "--out", str(out),
        "--aa-gpqa", "1",
        "--reasoning", "reasoning",
    ])
    assert rc == 0, out.read_text(encoding="utf-8")[:500]
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    env = {e["name"]: e.get("value") for e in doc["spec"]["containers"][0]["env"]
           if "value" in e}
    assert env["AA_GPQA"] == "1"
    assert "@@AA_GPQA@@" not in out.read_text(encoding="utf-8")
    assert "@@ARM@@" not in out.read_text(encoding="utf-8")
