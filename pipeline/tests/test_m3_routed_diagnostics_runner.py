"""CPU dry-run tests for routed-expert diagnostic arms."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.m3_routed_diagnostics import EXPECTED_ARMS, prepare_checkpoint_overlay

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "pipeline/slurm/test_m3_routed_diagnostics_arm.sh"


def _checkpoint(path: Path, *, with_activations: bool) -> None:
    path.mkdir(parents=True)
    activation = {"num_bits": 8, "type": "float"} if with_activations else None
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minimax_m3_vl",
                "quantization_config": {
                    "config_groups": {
                        "group_0": {"input_activations": activation}
                    }
                },
            }
        )
    )
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")


def test_prepare_w4a16_overlay_does_not_mutate_source_checkpoint():
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        source = root / "source"
        overlay = root / "overlay"
        _checkpoint(source, with_activations=True)
        original = (source / "config.json").read_text()

        prepare_checkpoint_overlay(source, overlay, disable_activations=True)

        assert (source / "config.json").read_text() == original
        config = json.loads((overlay / "config.json").read_text())
        group = config["quantization_config"]["config_groups"]["group_0"]
        assert group["input_activations"] is None
        assert (overlay / "model-00001-of-00001.safetensors").is_symlink()


def test_all_diagnostic_arms_dry_run_with_fixed_envelope():
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        reference = root / "reference"
        candidate = root / "candidate"
        _checkpoint(reference, with_activations=False)
        _checkpoint(candidate, with_activations=True)
        for arm in EXPECTED_ARMS:
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(REPO_ROOT),
                    "DRY_RUN": "1",
                    "MATRIX_ID": "diag-test",
                    "ARM": arm,
                    "REFERENCE_CKPT": str(reference),
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
            manifest = json.loads(
                (root / "evidence/diag-test" / arm / "arm_manifest.json").read_text()
            )
            assert manifest["quality_envelope"]["prompt_mode"] == "chat_template"
            assert manifest["diagnostics"]["M3_PARAM_FINGERPRINT"] == "1"
            assert manifest["diagnostics"]["M3_MOE_PROBE_RECOMPUTE"] == "1"
            expected_overlay = (
                "input_activations=null" if arm == "candidate_w4a16" else None
            )
            assert manifest["config_overlay"] == expected_overlay
