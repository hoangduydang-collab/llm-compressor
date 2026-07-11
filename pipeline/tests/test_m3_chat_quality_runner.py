"""CPU dry-run tests for the one-arm MiniMax-M3 chat quality runner."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.m3_chat_quality import EXPECTED_ARMS

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "pipeline/slurm/test_m3_chat_quality_arm.sh"


def _checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "minimax_m3_vl"}))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))


def test_all_chat_quality_arms_dry_run_with_fixed_envelope():
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        reference = root / "reference"
        candidate = root / "candidate"
        _checkpoint(reference)
        _checkpoint(candidate)
        for arm in EXPECTED_ARMS:
            full_root = root / "full"
            evidence_root = root / "evidence"
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(REPO_ROOT),
                    "DRY_RUN": "1",
                    "MATRIX_ID": "matrix-test",
                    "ARM": arm,
                    "REFERENCE_CKPT": str(reference),
                    "CANDIDATE_CKPT": str(candidate),
                    "RESULTS_ROOT": str(full_root),
                    "EVIDENCE_ROOT": str(evidence_root),
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
            arm_dir = evidence_root / "matrix-test" / arm
            manifest = json.loads((arm_dir / "arm_manifest.json").read_text())
            role = "reference" if arm.startswith("reference_") else "candidate"
            interface = "offline" if arm.endswith("offline_chat") else "http"
            assert manifest["checkpoint_role"] == role
            assert manifest["interface"] == interface
            assert manifest["quality_envelope"]["enforce_eager"] is True
            assert manifest["quality_envelope"]["thinking_mode"] == "disabled"
            assert manifest["diagnostics"] == {
                "M3_LOAD_AUDIT": "0",
                "M3_MOE_PROBE": "0",
                "M3_PARAM_FINGERPRINT": "0",
            }
            expected = reference if role == "reference" else candidate
            assert Path(manifest["checkpoint"]) == expected.resolve()
            assert not (full_root / "matrix-test" / arm / "serve.log").exists()


def test_unknown_chat_quality_arm_fails_before_writing_evidence():
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "MATRIX_ID": "matrix-test",
                "ARM": "not-an-arm",
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
        assert completed.returncode == 2
        assert "unknown ARM" in completed.stderr
        assert not (root / "evidence").exists()


def test_live_preflight_failure_still_bundles_arm_evidence():
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        reference = root / "reference"
        _checkpoint(reference)
        evidence_root = root / "evidence"
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(REPO_ROOT),
                "MATRIX_ID": "matrix-failed-preflight",
                "ARM": "reference_offline_chat",
                "REFERENCE_CKPT": str(reference),
                "RESULTS_ROOT": str(root / "full"),
                "EVIDENCE_ROOT": str(evidence_root),
                "ENV_FILE": str(root / "missing-env.sh"),
                "VENV_ACTIVATE": str(root / "missing-activate"),
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

        assert completed.returncode == 2
        arm_dir = evidence_root / "matrix-failed-preflight/reference_offline_chat"
        assert (arm_dir / "arm_manifest.json").is_file()
        report = json.loads((arm_dir / "arm_report.json").read_text())
        assert report["infrastructure_ok"] is False
        assert (arm_dir / "return_code.txt").read_text().strip() == "2"
