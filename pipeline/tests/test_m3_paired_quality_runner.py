"""CPU-only dry-run tests for the paired MiniMax-M3 quality runner."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "pipeline/slurm/test_m3_paired_quality.sh"


def _write_checkpoint(path: Path, *, w4a8: bool) -> None:
    path.mkdir(parents=True)
    activation = {"num_bits": 8} if w4a8 else None
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minimax_m3_vl",
                "quantization_config": {
                    "config_groups": {
                        "group_0": {
                            "weights": {"num_bits": 4},
                            "input_activations": activation,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}}),
        encoding="utf-8",
    )


def test_paired_runner_dry_run_records_identical_eager_envelopes():
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        reference = root / "reference"
        candidate = root / "candidate"
        results = root / "runs"
        evidence = root / "evidence"
        _write_checkpoint(reference, w4a8=False)
        _write_checkpoint(candidate, w4a8=True)
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(REPO_ROOT),
                "DRY_RUN": "1",
                "RUN_ID": "test-run",
                "REFERENCE_CKPT": os.path.relpath(reference, REPO_ROOT),
                "CANDIDATE_CKPT": os.path.relpath(candidate, REPO_ROOT),
                "RESULTS_ROOT": str(results),
                "EVIDENCE_ROOT": str(evidence),
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

        assert completed.returncode == 0, completed.stderr
        manifest = json.loads(
            (evidence / "test-run/run_manifest.json").read_text(encoding="utf-8")
        )
        assert [case["name"] for case in manifest["cases"]] == [
            "cyankiwi_reference",
            "portable_awq_w4a8",
        ]
        assert manifest["case_order"] == [
            "cyankiwi_reference",
            "portable_awq_w4a8",
        ]
        assert manifest["started_at"].endswith("+00:00")
        assert manifest["finished_at"].endswith("+00:00")
        assert all(Path(case["checkpoint"]).is_absolute() for case in manifest["cases"])
        reference_command, candidate_command = [
            case["command"] for case in manifest["cases"]
        ]
        assert reference_command[reference_command.index(
            "serve.enforce_eager=true"
        )] == candidate_command[candidate_command.index(
            "serve.enforce_eager=true"
        )]
        assert manifest["comparison_envelope"]["tensor_parallel_size"] == 8
        assert manifest["comparison_envelope"]["enable_expert_parallel"] is True
        assert manifest["diagnostics"] == {
            "M3_LOAD_AUDIT": "1",
            "M3_MOE_PROBE": "1",
            "M3_PARAM_FINGERPRINT": "1",
        }
        assert not (results / "test-run/cyankiwi_reference/serve.log").exists()
