"""CPU dry-run tests for MiniMax-M3 layer-boundary arms."""

import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.m3_layer_boundary_diagnostics import ARM_SPECS, EXPECTED_ARMS
from pipeline.m3_routed_diagnostics import (
    VLLM_ROUTER_IGNORE,
    VLLM_SHARED_EXPERT_IGNORE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "pipeline/slurm/test_m3_layer_boundary_arm.sh"


def _checkpoint(path: Path, *, activations: bool) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minimax_m3_vl",
                "quantization_config": {
                    "ignore": ["re:.*mlp[.]shared_experts[.].*"],
                    "config_groups": {
                        "group_0": {
                            "input_activations": (
                                {"num_bits": 8, "type": "float"}
                                if activations
                                else None
                            )
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


def test_every_boundary_arm_dry_runs_with_its_single_variable_envelope():
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        reference = root / "reference"
        candidate = root / "candidate"
        _checkpoint(reference, activations=False)
        _checkpoint(candidate, activations=True)
        for arm in EXPECTED_ARMS:
            completed = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(REPO_ROOT),
                    "DRY_RUN": "1",
                    "MATRIX_ID": "boundary-test",
                    "ARM": arm,
                    "REFERENCE_CKPT": str(reference),
                    "CANDIDATE_CKPT": str(candidate),
                    "RESULTS_ROOT": str(root / "raw"),
                    "EVIDENCE_ROOT": str(root / "evidence"),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr + completed.stdout
            manifest = json.loads(
                (root / "evidence/boundary-test" / arm / "arm_manifest.json").read_text()
            )
            spec = ARM_SPECS[arm]
            assert manifest["interface"] == spec.interface
            assert manifest["checkpoint_role"] == spec.checkpoint_role
            assert manifest["quality_envelope"]["enable_expert_parallel"] == spec.enable_ep
            assert manifest["quality_envelope"]["kv_cache_dtype"] == spec.kv_cache_dtype
            assert manifest["diagnostics"]["M3_LAYER_BOUNDARY"] == (
                "1" if spec.interface == "offline" else "0"
            )
            config = json.loads((root / "raw/boundary-test" / arm / "checkpoint/config.json").read_text())
            ignore = config["quantization_config"]["ignore"]
            if spec.checkpoint_role == "candidate":
                assert VLLM_SHARED_EXPERT_IGNORE in ignore
            assert (VLLM_ROUTER_IGNORE in ignore) == spec.router_alias
