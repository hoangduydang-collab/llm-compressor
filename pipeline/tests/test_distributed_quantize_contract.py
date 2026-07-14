import json
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from pipeline.calibration import CalibrationPartition
from pipeline.distributed import DistributedContext

ROOT = Path(__file__).resolve().parents[2]


def test_quantize_module_import_does_not_require_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import pipeline.quantize"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_partition_manifest_uses_rank_path(monkeypatch, tmp_path):
    from pipeline.quantize import _persist_calibration_partition

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        current_device=lambda: 3,
        get_device_name=lambda index: "NVIDIA H100 80GB HBM3",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    ctx = DistributedContext(enabled=True, rank=3, world_size=8, local_rank=3)
    partition = CalibrationPartition(8, 3, 8, 3, 4)
    dataset = [{"input_ids": [31, 32]}]

    path = _persist_calibration_partition(tmp_path, dataset, partition, ctx)

    assert path == tmp_path / "calibration_partition.rank-3.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["rank"] == 3
    assert data["local_num_samples"] == 1
    assert data["distributed"]["cuda_current_device"] == 3


def test_single_process_manifest_preserves_legacy_name(tmp_path):
    from pipeline.quantize import _persist_calibration_partition

    ctx = DistributedContext()
    partition = CalibrationPartition(2, 0, 1, 0, 2)
    dataset = [{"input_ids": [1]}, {"input_ids": [2]}]

    path = _persist_calibration_partition(tmp_path, dataset, partition, ctx)

    assert path == tmp_path / "calibration_partition.json"


def test_evidence_paths_are_rank_local_when_distributed(tmp_path):
    from pipeline.quantize import _evidence_paths

    ctx = DistributedContext(enabled=True, rank=2, world_size=8, local_rank=2)

    paths = _evidence_paths(tmp_path, ctx)

    assert paths == {
        "metrics": tmp_path / "quant_metrics.rank-2.jsonl",
        "provenance": tmp_path / "model_provenance.rank-2.json",
        "partition": tmp_path / "calibration_partition.rank-2.json",
    }


def _quantize_config():
    return SimpleNamespace(
        model=SimpleNamespace(
            trust_remote_code=True,
            auto_class="AutoModelForImageTextToText",
            id="MiniMaxAI/MiniMax-M3",
        ),
        calibration=SimpleNamespace(
            sequential_targets=["MiniMaxM3VLDecoderLayer"],
            max_seq_length=512,
            num_samples=8,
            moe_calibrate_all_experts=True,
            pipeline="sequential",
        ),
        quantization=SimpleNamespace(
            scheme="W4AFP8",
            ignore=["lm_head"],
            sample_generation=False,
        ),
        serve=SimpleNamespace(prompt="hello"),
    )


class _NeverSaveModel:
    def save_pretrained(self, *args, **kwargs):
        raise AssertionError("evidence-only smoke saved a model checkpoint")


class _NeverSaveTokenizer:
    def save_pretrained(self, *args, **kwargs):
        raise AssertionError("evidence-only smoke saved tokenizer artifacts")


def test_single_process_evidence_only_preserves_legacy_sampler_kwargs(
    monkeypatch, tmp_path
):
    import pipeline.quantize as quantize

    calls = []
    fake_llmcompressor = types.ModuleType("llmcompressor")
    fake_llmcompressor.oneshot = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "llmcompressor", fake_llmcompressor)

    fake_m3 = types.ModuleType("pipeline.minimax_m3_config")
    fake_m3.patch_minimax_m3_for_text_calibration = lambda model: False
    fake_m3.register_minimax_m3_awq_mappings = lambda: None
    fake_m3.ensure_minimax_m3_vllm_serve_config = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "pipeline.minimax_m3_config", fake_m3)

    monkeypatch.setattr(
        quantize,
        "_load_model_and_tokenizer",
        lambda cfg: (_NeverSaveModel(), _NeverSaveTokenizer()),
    )
    monkeypatch.setattr(quantize, "log_model_provenance", lambda *a, **k: None)
    partition = CalibrationPartition(2, 0, 1, 0, 2)
    dataset = [{"input_ids": [1]}, {"input_ids": [2]}]
    monkeypatch.setattr(
        quantize,
        "build_calibration_dataset_with_partition",
        lambda cal, tokenizer: (dataset, partition),
    )
    monkeypatch.setattr(quantize, "build_recipe", lambda cfg: ["native-recipe"])
    monkeypatch.setattr(
        quantize.metrics, "capture_quant_metrics", lambda path: nullcontext()
    )
    monkeypatch.setattr(
        quantize.metrics, "summarize_quant_metrics", lambda path: {"count": 1}
    )

    result = quantize.run_quantize(
        _quantize_config(), tmp_path, DistributedContext(), save_checkpoint=False
    )

    assert result == tmp_path / "checkpoint"
    assert not (tmp_path / "checkpoint").exists()
    assert json.loads((tmp_path / "smoke_complete.json").read_text())["status"] == (
        "complete"
    )
    assert calls[0]["num_calibration_samples"] == 8
    assert "shuffle_calibration_samples" not in calls[0]


def test_distributed_evidence_only_uses_rank_local_sampler_kwargs(
    monkeypatch, tmp_path
):
    import pipeline.quantize as quantize

    calls = []
    fake_llmcompressor = types.ModuleType("llmcompressor")
    fake_llmcompressor.oneshot = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "llmcompressor", fake_llmcompressor)
    fake_m3 = types.ModuleType("pipeline.minimax_m3_config")
    fake_m3.patch_minimax_m3_for_text_calibration = lambda model: False
    fake_m3.register_minimax_m3_awq_mappings = lambda: None
    fake_m3.ensure_minimax_m3_vllm_serve_config = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "pipeline.minimax_m3_config", fake_m3)
    monkeypatch.setattr(
        quantize,
        "_load_model_and_tokenizer",
        lambda cfg: (_NeverSaveModel(), _NeverSaveTokenizer()),
    )
    monkeypatch.setattr(quantize, "log_model_provenance", lambda *a, **k: None)
    partition = CalibrationPartition(8, 3, 8, 3, 4)
    dataset = [{"input_ids": [31, 32]}]
    monkeypatch.setattr(
        quantize,
        "build_calibration_dataset_with_partition",
        lambda cal, tokenizer: (dataset, partition),
    )
    monkeypatch.setattr(quantize, "build_recipe", lambda cfg: ["native-recipe"])
    monkeypatch.setattr(
        quantize.metrics, "capture_quant_metrics", lambda path: nullcontext()
    )
    monkeypatch.setattr(
        quantize.metrics, "summarize_quant_metrics", lambda path: {"count": 1}
    )
    ctx = DistributedContext(enabled=True, rank=3, world_size=8, local_rank=3)
    monkeypatch.setattr(ctx, "barrier", lambda: None)
    monkeypatch.setattr(
        ctx,
        "snapshot",
        lambda: {
            "enabled": True,
            "rank": 3,
            "world_size": 8,
            "local_rank": 3,
            "cuda_current_device": 3,
        },
    )

    quantize.run_quantize(_quantize_config(), tmp_path, ctx, save_checkpoint=False)

    assert calls[0]["num_calibration_samples"] == 1
    assert calls[0]["shuffle_calibration_samples"] is False


def test_run_cli_accepts_evidence_only(monkeypatch):
    from pipeline import run

    captured = {}

    def capture_args(args, dist_ctx):
        captured["evidence_only"] = args.evidence_only
        return 0

    monkeypatch.setattr(
        run.DistributedContext, "from_environment", lambda: DistributedContext()
    )
    monkeypatch.setattr(run, "_run", capture_args)

    assert run.main(["--config", "unused.yaml", "--evidence-only"]) == 0
    assert captured == {"evidence_only": True}
