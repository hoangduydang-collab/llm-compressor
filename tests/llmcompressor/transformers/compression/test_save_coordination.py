"""coordinate_collective_save: a source-rank save failure must surface on all
ranks within seconds instead of letting non-source ranks march into the next
collective the crashed source never joins (smoke r11 + r12, 2026-07-18: 3h+ of
silent GPU spin, traceback lost to SIGKILL)."""

import datetime
import socket

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from llmcompressor.transformers.compression.compressed_tensors_utils import (
    coordinate_collective_save,
)


def test_single_process_success_passthrough():
    with coordinate_collective_save() as outcome:
        pass
    assert outcome["error"] is None


def test_single_process_reraises_recorded_error():
    with pytest.raises(ValueError, match="boom"):
        with coordinate_collective_save() as outcome:
            outcome["error"] = ValueError("boom")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_rank(rank: int, world_size: int, port: int, source_fails: bool):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        if source_fails:
            if rank == 0:
                # source records its failure; coordinate re-raises the original
                try:
                    with coordinate_collective_save(
                        timeout=datetime.timedelta(seconds=60)
                    ) as outcome:
                        outcome["error"] = ValueError("disk save exploded")
                except ValueError as e:
                    assert "disk save exploded" in str(e)
                else:
                    raise AssertionError("source rank did not re-raise")
            else:
                # non-source learns of the failure and aborts loudly
                try:
                    with coordinate_collective_save(
                        timeout=datetime.timedelta(seconds=60)
                    ):
                        pass
                except RuntimeError as e:
                    assert "source-rank save_pretrained failed" in str(e)
                    assert "disk save exploded" in str(e)
                else:
                    raise AssertionError("non-source rank did not raise")
        else:
            with coordinate_collective_save(
                timeout=datetime.timedelta(seconds=60)
            ) as outcome:
                pass
            assert outcome["error"] is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("source_fails", [False, True])
def test_two_rank_gloo_coordination(source_fails):
    port = _free_port()
    mp.spawn(
        _run_rank,
        args=(2, port, source_fails),
        nprocs=2,
        join=True,
    )
