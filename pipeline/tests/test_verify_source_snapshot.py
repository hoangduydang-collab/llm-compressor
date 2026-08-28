"""Tests for the source-artifact identity check.

This guards a mistake already made once: 64 GB of the GLM-5.3 **FP8** release was
downloaded before anyone noticed it was the wrong artifact. The naming is inverted
from GLM-5.2 -- for 5.3 the UNSUFFIXED repo (`zai-org/GLM-5.3`) is the FP8 one and
`-BF16` is unquantized, where for 5.2 the unsuffixed repo was BF16 -- so
"looks right" is not a check.

It matters most in the unattended path: a job submitted before its weights exist
blocks until the download completes, and a complete-but-wrong download must abort
rather than start a ten-hour run on a source AWQ cannot consume.
"""

import json
from pathlib import Path

import pytest

from pipeline.verify_source_snapshot import check, main

BF16 = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "num_hidden_layers": 78,
    "first_k_dense_replace": 3,
}
FP8 = dict(
    BF16,
    quantization_config={
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    },
)


def _snapshot(path: Path, config: dict, shards: int = 0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for i in range(shards):
        (path / f"model-{i:05d}-of-{shards:05d}.safetensors").write_bytes(b"")
    return path


def test_bf16_release_passes(tmp_path: Path):
    snap = _snapshot(tmp_path / "bf16", BF16)
    assert check(
        snap,
        require_unquantized=True,
        expect_layers=78,
        expect_arch="GlmMoeDsaForCausalLM",
    ) == []


def test_fp8_release_is_rejected(tmp_path: Path):
    """The actual mistake. It must fail, and the message must say why."""
    snap = _snapshot(tmp_path / "fp8", FP8)
    problems = check(snap, require_unquantized=True)
    assert len(problems) == 1
    assert "quantization_config" in problems[0]
    assert "fp8" in problems[0]


def test_quantized_release_passes_when_not_required_unquantized(tmp_path: Path):
    """The flag is opt-in; a GPTQ-from-FP8 experiment would legitimately want it."""
    snap = _snapshot(tmp_path / "fp8", FP8)
    assert check(snap, expect_layers=78) == []


def test_wrong_depth_is_caught(tmp_path: Path):
    snap = _snapshot(tmp_path / "small", dict(BF16, num_hidden_layers=4))
    problems = check(snap, expect_layers=78)
    assert problems and "num_hidden_layers=4" in problems[0]


def test_wrong_architecture_is_caught(tmp_path: Path):
    snap = _snapshot(tmp_path / "other", dict(BF16, architectures=["Glm4MoeForCausalLM"]))
    problems = check(snap, expect_arch="GlmMoeDsaForCausalLM")
    assert problems and "architectures=" in problems[0]


def test_shard_count_is_checked_when_asked(tmp_path: Path):
    """141 shards means the FP8 release even if config.json were somehow clean."""
    snap = _snapshot(tmp_path / "half", BF16, shards=141)
    problems = check(snap, expect_shards=282)
    assert problems and "141 safetensors shards" in problems[0]
    assert check(_snapshot(tmp_path / "full", BF16, shards=282), expect_shards=282) == []


def test_missing_config_is_a_problem_not_a_crash(tmp_path: Path):
    """A partially-downloaded snapshot must report, not raise -- it is checked from
    inside a wait loop that would otherwise die on the first poll."""
    empty = tmp_path / "empty"
    empty.mkdir()
    problems = check(empty, require_unquantized=True)
    assert problems and "no config.json" in problems[0]


def test_all_problems_are_reported_together(tmp_path: Path):
    """One run should show every reason, so a fix is not iterative."""
    snap = _snapshot(tmp_path / "bad", dict(FP8, num_hidden_layers=4,
                                            architectures=["Nope"]))
    problems = check(
        snap,
        require_unquantized=True,
        expect_layers=78,
        expect_arch="GlmMoeDsaForCausalLM",
    )
    assert len(problems) == 3


@pytest.mark.parametrize("config,expected_rc", [(BF16, 0), (FP8, 1)])
def test_cli_exit_codes(tmp_path: Path, config, expected_rc):
    """The container script branches on the exit code, so it is the contract."""
    snap = _snapshot(tmp_path / "cli", config)
    rc = main([str(snap), "--require-unquantized", "--expect-layers", "78",
               "--expect-arch", "GlmMoeDsaForCausalLM"])
    assert rc == expected_rc
