from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline" / "k8s"))
try:
    import render_glm53_ep_gptq_chain as render
except ImportError:
    render = None


def test_rendered_chain_is_fail_closed_and_releases_on_success(tmp_path: Path):
    assert render is not None, "chain renderer is missing"
    out = tmp_path / "chain.yaml"
    assert render.main([
        "--run-tag", "20260910t120000z",
        "--ref", "a" * 40,
        "--out", str(out),
    ]) == 0

    text = out.read_text(encoding="utf-8")
    assert "@@" not in text
    doc = yaml.safe_load(text)
    assert doc["kind"] == "Job"
    assert doc["metadata"]["name"] == "glm53-ep-gptq-20260910t120000z"
    assert doc["spec"]["backoffLimit"] == 0
    assert doc["spec"]["template"]["spec"]["restartPolicy"] == "Never"
    assert "nodeSelector" not in doc["spec"]["template"]["spec"]

    container = doc["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == (
        "docker.io/lmsysorg/sglang@"
        "sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155"
    )
    resources = container["resources"]
    assert resources["requests"]["nvidia.com/gpu"] == 8
    assert resources["limits"]["nvidia.com/gpu"] == 8
    script = container["args"][0]
    markers = [
        "test_expert_parallel_nccl.py",
        'run_lane "representative-ep4"',
        'run_lane "representative-ep8"',
        'run_lane "representative-ddp4"',
        'run_lane "full-ep8"',
    ]
    positions = [script.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "fail_and_hold" in script
    assert "sleep 86400" in script
    assert script.count("sleep 86400") == 1
    assert "CHAIN_STATUS=success" in script
    success_tail = script[script.index("CHAIN_STATUS=success"):]
    assert "sleep 86400" not in success_tail
    assert 'write_status "$CHAIN_STATUS" full-ep8 0 "$FULL_CKPT"' in success_tail
    assert (
        "models--zai-org--GLM-5.3-BF16/snapshots/"
        "304b8051cfb2b260b61ce0cbe330e02a98e73639"
    ) in script
    assert "/mnt/cephfs/hoangduy/results/glm53-ep-gptq/20260910t120000z" in script


def test_renderer_persists_optional_node_pin(tmp_path: Path):
    assert render is not None, "chain renderer is missing"
    out = tmp_path / "chain.yaml"
    assert render.main([
        "--run-tag", "t",
        "--ref", "b" * 40,
        "--node", "gpu06",
        "--out", str(out),
    ]) == 0

    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["spec"]["template"]["spec"]["nodeSelector"] == {
        "kubernetes.io/hostname": "gpu06"
    }
