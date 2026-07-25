"""Static orchestration contract for Humming performance arms."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARM = REPO_ROOT / "pipeline/slurm/perf_eval_arm.sh"


def test_humming_attestation_runs_after_readiness_and_before_benchmark():
    source = ARM.read_text(encoding="utf-8")

    local_start = source.index('if [ "$MODE" = local ]; then')
    remote_start = source.index("\nelse\n", local_start)
    local_block = source[local_start:remote_start]
    attestation = local_block.index("pipeline.m3_humming_w4a8 attest")
    readiness = local_block.index('[ "$ready" = 0 ]')
    benchmark = source.index("performance/scripts/preflight.sh")

    assert "M3_W4A8_BACKEND" in local_block
    assert readiness < attestation
    assert local_start + attestation < benchmark
    assert '--out "$C/backend-attestation.json"' in local_block
    assert source.count("pipeline.m3_humming_w4a8 attest") == 1
    assert "sbatch" not in source.lower()
