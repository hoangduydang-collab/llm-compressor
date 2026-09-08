"""Executable two-GPU gate; an unavailable GPU backend is explicitly skipped.

Run this file on the executor before real-width qualification. It uses spawn,
rank-local CUDA devices, a finite NCCL timeout, real GPTQ and collective export.
No CPU/Gloo result counts as passing this gate.
"""

import pytest
import torch
import torch.distributed as dist

from tests.llmcompressor.modifiers.gptq.test_expert_parallel_distributed import _launch
from tests.llmcompressor.modifiers.gptq.test_expert_parallel_oneshot import (
    _lifecycle_worker,
)

pytestmark = pytest.mark.skipif(
    not dist.is_available()
    or not dist.is_nccl_available()
    or torch.cuda.device_count() < 2,
    reason="requires two CUDA GPUs and NCCL; execute on the GPU executor",
)


def _nccl_worker(rank, workdir, case):
    assert dist.get_world_size() == 2
    assert dist.get_backend() == "nccl"
    assert torch.cuda.current_device() == rank
    _lifecycle_worker(
        rank,
        workdir,
        dynamic=case == "mixed-disk",
        compare_baseline=case == "ddp-ep-parity",
        disk_offload=case != "ddp-ep-parity",
    )


@pytest.mark.parametrize("case", ["ddp-ep-parity", "weight-only-disk", "mixed-disk"])
def test_two_gpu_nccl_expert_gptq_collective_save_reload(tmp_path, case):
    _launch(_nccl_worker, tmp_path, case, backend="nccl")
