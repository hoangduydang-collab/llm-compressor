"""Save-phase heartbeat: the disk-offload save is silent for 1h+ while rank 0
reads offloaded weights back (r11, 2026-07-18); the heartbeat makes that phase
visibly alive in torchrun.out."""

import time

from pipeline.quantize import _process_read_bytes, _save_heartbeat


def test_heartbeat_emits_progress_lines_and_stops(tmp_path, capsys):
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "model-00001-of-00002.safetensors").write_bytes(b"x" * 2048)

    with _save_heartbeat(ckpt, interval=0.05):
        time.sleep(0.2)
        (ckpt / "model-00002-of-00002.safetensors").write_bytes(b"y" * 4096)
        time.sleep(0.2)
    # after exit the thread is stopped: no further lines
    baseline = capsys.readouterr().out
    time.sleep(0.15)
    assert capsys.readouterr().out == ""

    lines = [l for l in baseline.splitlines() if "save-heartbeat" in l]
    assert len(lines) >= 2
    # counts both shards once the second lands, reports written bytes and
    # cumulative read-back volume
    assert any("2 shards" in l for l in lines)
    assert all("GB written" in l and "GB read back" in l for l in lines)


def test_heartbeat_survives_missing_checkpoint_dir(tmp_path, capsys):
    # save_pretrained creates the dir itself; the first ticks may race it
    with _save_heartbeat(tmp_path / "not-yet-created", interval=0.05):
        time.sleep(0.15)
    lines = capsys.readouterr().out.splitlines()
    assert any("0 shards" in l for l in lines)


def test_process_read_bytes_is_nonnegative_int():
    value = _process_read_bytes()
    assert isinstance(value, int)
    assert value >= 0
