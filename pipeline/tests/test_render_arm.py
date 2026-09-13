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
        "--aa-gpqa-only", "1",
        "--reasoning", "reasoning",
    ])
    assert rc == 0, out.read_text(encoding="utf-8")[:500]
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    env = {e["name"]: e.get("value") for e in doc["spec"]["containers"][0]["env"]
           if "value" in e}
    assert env["AA_GPQA"] == "1"
    assert env["AA_GPQA_ONLY"] == "1"
    assert env.get("HOLD_AFTER") in (None, "")
    assert "@@AA_GPQA@@" not in out.read_text(encoding="utf-8")


def test_render_arm_hold_after_persists(tmp_path: Path):
    out = tmp_path / "hold.yaml"
    rc = render_arm.main([
        "--arm", "ours",
        "--model", "/mnt/cephfs/hoangduy/results/glm53-w4afp8-mtp/checkpoint",
        "--run-tag", "t",
        "--ref", "deadbeef",
        "--out", str(out),
        "--aa-gpqa", "1",
        "--aa-gpqa-only", "1",
        "--hold-after", "1",
        "--reasoning", "reasoning",
    ])
    assert rc == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    env = {e["name"]: e.get("value") for e in doc["spec"]["containers"][0]["env"]
           if "value" in e}
    assert env["HOLD_AFTER"] == "1"
    assert "sleep infinity" in doc["spec"]["containers"][0]["args"][0]
    assert "@@ARM@@" not in out.read_text(encoding="utf-8")


def test_render_gptq_persists_native_context_tokenizer_revision_and_profile(tmp_path: Path):
    out = tmp_path / "gptq.yaml"
    revision = "a" * 64
    rc = render_arm.main([
        "--arm", "gptq",
        "--model", "/mnt/cephfs/hoangduy/results/glm53-ep-gptq-w4afp8/checkpoint",
        "--run-tag", "t",
        "--ref", "deadbeef",
        "--out", str(out),
        "--context-length", "65536",
        "--served-tokenizer-revision", revision,
    ])
    assert rc == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    env = {e["name"]: e.get("value") for e in doc["spec"]["containers"][0]["env"]
           if "value" in e}
    assert env["CTX"] == "65536"
    assert env["SERVED_TOKENIZER_REVISION"] == revision
    assert env["PROFILE"] == "configs/glm/glm-5.3-w4afp8-gptq.sh"
