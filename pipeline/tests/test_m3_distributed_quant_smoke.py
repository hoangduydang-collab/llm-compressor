import os
import subprocess
from pathlib import Path

from pipeline.config import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "pipeline/configs/minimax_m3_distributed_smoke.yaml"
LAUNCHER = ROOT / "pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh"
BASH = next(
    (
        path
        for path in (
            Path("C:/Program Files/Git/bin/bash.exe"),
            Path("C:/Program Files/Git/usr/bin/bash.exe"),
        )
        if path.is_file()
    ),
    Path("bash"),
)
LAYER_EXCLUSION = (
    "re:.*language_model[.]layers[.]"
    "(?!(?:3|31|59)(?:[.]|$))[0-9]+(?:[.]|$).*"
)


def test_smoke_config_reuses_production_recipe_with_three_layers_only():
    cfg = load_config(CONFIG)

    assert cfg.model.id == "MiniMaxAI/MiniMax-M3"
    assert cfg.model.device_map == "auto_offload"
    assert cfg.quantization.method == "gptq"
    assert cfg.quantization.scheme == "W4AFP8"
    assert LAYER_EXCLUSION in cfg.quantization.ignore
    assert cfg.calibration.num_samples == 8
    assert cfg.calibration.max_seq_length == 512
    assert cfg.calibration.sequential_targets == ["MiniMaxM3VLDecoderLayer"]
    assert cfg.quantization.sample_generation is False
    assert cfg.serve.enabled is False
    assert cfg.eval.enabled is False


def test_dry_run_uses_two_sequential_srun_torchrun_smokes(tmp_path):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "RUN_ID": "pytest-distributed-smoke",
        "RESULT_ROOT": str(tmp_path / "results"),
        "LOG_ROOT": str(tmp_path / "logs"),
    }

    output = subprocess.check_output(
        [str(BASH), str(LAUNCHER)], cwd=ROOT, env=env, text=True
    )

    assert output.count("srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8") == 2
    assert output.count("torchrun --nproc_per_node=8 -m pipeline.run") == 2
    assert output.count("--evidence-only") == 2
    assert output.index("quantization.method=gptq") < output.index(
        "quantization.method=awq"
    )
    assert "sbatch" not in output


def test_launcher_owns_top_level_srun_and_rejects_nested_slurm():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "SLURM_JOB_ID" in text
    assert "--worker" in text
    assert "nvidia-smi" in text
    assert "/proc/meminfo" in text
    assert "/usr/bin/time -v" in text
    assert "sbatch" not in text
    assert "pipeline.m3_awq_representative" not in text
    assert "M3_AWQ_HOOK_TRACE" not in text
    assert "--run-probe" not in text
