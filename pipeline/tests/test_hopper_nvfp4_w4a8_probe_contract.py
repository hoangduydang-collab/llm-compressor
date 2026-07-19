from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE = REPO_ROOT / "pipeline/hopper_nvfp4_w4a8/gpu_probe.py"
LAUNCHER = REPO_ROOT / "pipeline/slurm/run_hopper_nvfp4_w4a8_probe_srun.sh"


def test_probe_is_bounded_deterministic_and_checks_exact_contract():
    source = PROBE.read_text(encoding="utf-8")

    for token in (
        "SHAPE_N = 128",
        "SHAPE_K = 128",
        "M_VALUES = (1, 8, 32)",
        "manual_seed",
        "range(16)",
        "float8e4m3",
        "float4e2m1",
        "weight_scale_group_size=16",
        'weight_scale_type="group_tensor"',
        "use_f16_accum=False",
        'mma_type="wgmma"',
        "k16_isolation",
        "fragment_scale_isolation",
        "exact_emulation",
        "bf16_nvfp4",
        "persistent_ratio",
        "deterministic",
        "kernel_filename",
        "cuobjdump",
        "WGMMA.MMA_ASYNC",
        "dtype=torch.int32",
        "view(SHAPE_N, SHAPE_K // 8, 8)",
        "HummingLayerMethod.transform_humming_layer",
    ):
        assert token in source
    assert "from_pretrained" not in source
    assert "snapshot_download" not in source


def test_launcher_uses_one_bounded_top_level_srun_and_preserves_evidence():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert source.count("srun ") == 1
    for token in (
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:1",
        "RUN_ID=",
        "RESULT_ROOT=",
        "rev-parse HEAD",
        "patch_humming_nvfp4_w4a8.py",
        "--check",
        "gpu_probe.py",
        "probe.json",
        "srun.returncode",
        "torch.cuda.get_device_capability",
        "manual_seed",
        "sha256sum",
    ):
        assert token in source
    assert "sbatch" not in source
    assert "huggingface" not in source.lower()
    assert "lm_eval" not in source
    assert "vllm serve" not in source


def test_installer_uses_a_dedicated_humming_cache_without_recursive_deletion():
    installer = REPO_ROOT / "pipeline/slurm/install_humming_nvfp4_w4a8.sh"
    source = installer.read_text(encoding="utf-8")

    assert "HUMMING_CACHE_DIR" in source
    assert "nvfp4-w4a8-v1" in source
    assert "rm -rf" not in source
    assert "find " not in source


def test_launcher_validates_structured_result_instead_of_trusting_exit_only():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'payload["passed"] is True' in source
    assert 'payload["device_capability"] == [9, 0]' in source
    assert 'payload["sass"]["fp8_wgmma_found"] is True' in source
    assert 'payload["memory"]["persistent_ratio"] <= 1.10' in source
    assert 'payload["layer_transform"]["passed"] is True' in source
    assert 'payload["fragment_scale_isolation"]["passed"] is True' in source
