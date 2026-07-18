import os
import subprocess
from pathlib import Path

from pipeline.config import load_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "pipeline/configs/minimax_m3_distributed_smoke.yaml"
LAUNCHER = ROOT / "pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh"
CONVERSION_MAPPINGS = ROOT / "src/llmcompressor/modeling/moe/conversion_mappings.py"
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
    "re:.*language_model[.]layers[.](?!(?:3|31|59)(?:[.]|$))[0-9]+(?:[.]|$).*"
)


def test_smoke_config_reuses_production_recipe_with_three_layers_only():
    cfg = load_config(CONFIG)

    assert cfg.model.id == "MiniMaxAI/MiniMax-M3"
    assert cfg.model.device_map == "auto_offload"
    # 32e9 mirrors production disk-offload (VMA fix, BUGS_AND_FIXES.md
    # 2026-07-17): weights overflow to DistributedDiskCache instead of
    # per-tensor shm segments that exceed vm.max_map_count.
    assert cfg.model.max_memory == {"cpu": 32_000_000_000}
    assert cfg.quantization.method == "gptq"
    assert cfg.quantization.scheme == "W4AFP8"
    assert LAYER_EXCLUSION in cfg.quantization.ignore
    assert cfg.calibration.num_samples == 8
    assert cfg.calibration.max_seq_length == 512
    assert cfg.calibration.sequential_targets == ["MiniMaxM3VLDecoderLayer"]
    assert cfg.quantization.sample_generation is False
    assert cfg.serve.enabled is False
    assert cfg.eval.enabled is False


def test_dry_run_uses_two_full_cpu_srun_torchrun_smokes(tmp_path):
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
    # r7 lesson: without --cpus-per-task, Slurm 21.08 binds each step task to a
    # single physical core despite the exclusive 192-CPU allocation.
    assert output.count("--cpus-per-task=192") == 2
    assert output.count("torchrun --nproc_per_node=8 -m pipeline.run") == 2
    assert output.count("--evidence-only") == 2
    assert output.index("quantization.method=gptq") < output.index(
        "quantization.method=awq"
    )
    assert "sbatch" not in output


def test_launcher_runs_methods_in_parallel_on_separate_nodes_by_default():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'PARALLEL_METHODS="${PARALLEL_METHODS:-1}"' in text
    assert 'CPUS_PER_TASK="${CPUS_PER_TASK:-192}"' in text
    # each method's srun is backgrounded and awaited so both arms hold their
    # own exclusive node concurrently
    assert 'method_pids[$method]=$!' in text
    assert 'wait "${method_pids[$method]}"' in text


def test_launcher_owns_top_level_srun_and_rejects_nested_slurm():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "SLURM_JOB_ID" in text
    assert "--worker" in text
    assert "nvidia-smi" in text
    assert "/proc/meminfo" in text
    assert "MemAvailable" in text
    assert "torch.cuda.device_count()" in text
    assert "df -B1 /dev/shm" in text
    assert "/usr/bin/time -v" in text
    assert "driver_version" in text
    assert "torch_cuda_build" in text
    assert "sbatch" not in text
    assert "pipeline.m3_awq_representative" not in text
    assert "M3_AWQ_HOOK_TRACE" not in text
    assert "--run-probe" not in text


def test_launcher_uses_configurable_post_linearization_memory_floors():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert (
        'MIN_MEM_AVAILABLE_BYTES="${MIN_MEM_AVAILABLE_BYTES:-1200000000000}"'
        in text
    )
    # r6 lesson: the distributed CPU offload keeps one full model copy in
    # /dev/shm, so the gate defaults to the checkpoint's exact size (+5%), not
    # a fixed IPC floor. The old 128 GB default let AWQ launch on a node with
    # only 213 GB free and die mid-load.
    assert 'MIN_SHM_AVAILABLE_BYTES="${MIN_SHM_AVAILABLE_BYTES:-auto}"' in text
    assert "128000000000" not in text
    assert "900000000000" not in text
    assert "model.safetensors.index.json" in text
    assert '"metadata"]["total_size"]' in text
    assert "total * 105 // 100" in text
    assert "shm_available < min_shm_bytes" in text


def test_launcher_reclaims_orphaned_torch_shm_before_capacity_gate():
    """r6 lesson: ranks hard-killed mid-run leak /dev/shm/torch_* files (852 GB
    on gpu-h101 after the r5 GPTQ crash), starving the next arm's model load.
    On an exclusively held node, any $USER-owned torch_* segment not mapped by
    a live process is leakage and must be reclaimed before the capacity gate.
    """
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "/dev/shm/torch_" in text
    assert "/proc/[0-9]*/maps" in text
    assert '[[ -e "$stale_file" && -O "$stale_file" ]]' in text
    assert "stale_shm_files_removed=" in text
    assert "min_shm_available_bytes_required=" in text


def test_minimax_m3_linearizes_experts_while_loading():
    text = CONVERSION_MAPPINGS.read_text(encoding="utf-8")

    assert "MiniMaxM3VLExperts" in text
    assert '"minimax_m3_vl": MiniMaxM3VLExperts' in text
    assert '"minimax_m3_vl": (' in text
    assert r"mlp\.experts\.(\d+)\.w1\." in text
    assert r"mlp\.experts\.(\d+)\.w2\." in text
    assert r"mlp\.experts\.(\d+)\.w3\." in text
    assert r"mlp.experts.\1.gate_proj." in text
    assert r"mlp.experts.\1.down_proj." in text
    assert r"mlp.experts.\1.up_proj." in text
