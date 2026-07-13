"""CPU dry-run tests for MiniMax-M3 shared-expert repair arms."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.m3_routed_diagnostics import VLLM_SHARED_EXPERT_IGNORE
from pipeline.m3_shared_expert_repair import EXPECTED_ARMS

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "pipeline/slurm/test_m3_shared_expert_repair_arm.sh"


def _checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minimax_m3_vl",
                "quantization_config": {
                    "ignore": ["re:.*mlp[.]shared_experts[.].*"],
                    "config_groups": {
                        "group_0": {
                            "input_activations": {"num_bits": 8, "type": "float"}
                        }
                    },
                },
            }
        )
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}})
    )
    (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_repair_arms_prepare_alias_overlay_in_dry_run():
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        candidate = root / "candidate"
        _checkpoint(candidate)
        source_hash = _sha(candidate / "config.json")
        for arm in EXPECTED_ARMS:
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(REPO_ROOT),
                    "DRY_RUN": "1",
                    "MATRIX_ID": "repair-test",
                    "ARM": arm,
                    "CANDIDATE_CKPT": str(candidate),
                    "RESULTS_ROOT": str(root / "full"),
                    "EVIDENCE_ROOT": str(root / "evidence"),
                }
            )
            completed = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr + completed.stdout
            run_dir = root / "full/repair-test" / arm
            evidence_dir = root / "evidence/repair-test" / arm
            overlay = json.loads((run_dir / "checkpoint/config.json").read_text())
            ignore = overlay["quantization_config"]["ignore"]
            assert ignore.count(VLLM_SHARED_EXPERT_IGNORE) == 1
            activation = overlay["quantization_config"]["config_groups"]["group_0"][
                "input_activations"
            ]
            assert (activation is None) == (arm == "repaired_w4a16_offline")
            assert _sha(candidate / "config.json") == source_hash
            manifest = json.loads((evidence_dir / "arm_manifest.json").read_text())
            assert manifest["source_config_sha256"] == source_hash
            assert manifest["overlay_config_sha256"] == _sha(
                run_dir / "checkpoint/config.json"
            )
            assert manifest["config_overlay"]["shared_expert_ignore"] == (
                VLLM_SHARED_EXPERT_IGNORE
            )
            expected_interface = "http" if arm.endswith("_http") else "offline"
            assert manifest["interface"] == expected_interface


def test_unknown_repair_arm_fails_before_writing_evidence():
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        completed = subprocess.run(
            ["bash", str(RUNNER)],
            cwd=REPO_ROOT,
            env={**os.environ, "DRY_RUN": "1", "ARM": "unknown"},
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 2
        assert "unknown ARM" in completed.stderr
