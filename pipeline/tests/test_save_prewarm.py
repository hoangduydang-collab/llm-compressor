"""Save-phase page-cache prewarm: the offloaded save_pretrained gather is a
single thread doing on-demand NFS reads (r11, 2026-07-18); parallel sequential
prefetch pulls the offload files into page cache ahead of it."""

import os

from pipeline.quantize import _prewarm_read_file, prewarm_offload_page_cache


def test_prewarm_reads_files_and_symlink_targets(tmp_path, capsys):
    offload = tmp_path / "offload"
    offload.mkdir()
    (offload / "ct_disk_cache_a.safetensors").write_bytes(b"a" * 4096)
    target = tmp_path / "base-shard.safetensors"
    target.write_bytes(b"b" * 8192)
    (offload / "ct_disk_cache_b.safetensors").symlink_to(target)
    # broken symlink must not raise
    (offload / "ct_disk_cache_gone.safetensors").symlink_to(tmp_path / "missing")

    thread = prewarm_offload_page_cache(offload, max_threads=2)

    assert thread is not None
    thread.join(timeout=10)
    assert not thread.is_alive()
    out = capsys.readouterr().out
    assert "save-prewarm: prefetching 3 offload files" in out
    assert "save-prewarm done: 3 files, 0.0 GB" in out


def test_prewarm_disabled_by_env(tmp_path, monkeypatch):
    (tmp_path / "f.safetensors").write_bytes(b"x")
    monkeypatch.setenv("M3_SAVE_PREWARM", "0")

    assert prewarm_offload_page_cache(tmp_path) is None


def test_prewarm_noop_on_empty_or_missing_dir(tmp_path):
    assert prewarm_offload_page_cache(tmp_path / "does-not-exist") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert prewarm_offload_page_cache(empty) is None


def test_prewarm_read_file_counts_bytes_and_tolerates_missing(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"z" * 12345)
    assert _prewarm_read_file(f) == 12345
    assert _prewarm_read_file(tmp_path / "missing.bin") == 0
