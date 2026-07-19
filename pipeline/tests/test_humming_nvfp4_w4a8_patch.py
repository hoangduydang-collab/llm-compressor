# ruff: noqa: E501 -- fixture lines intentionally mirror exact upstream anchors.
from __future__ import annotations

import dataclasses
import hashlib
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.hopper_nvfp4_w4a8 import humming_patch
from pipeline.hopper_nvfp4_w4a8.humming_patch import (
    BACKUP_SUFFIX,
    MARKER,
    PATCH_TARGETS,
    PatchError,
    patch_humming_tree,
)
from pipeline.slurm import patch_humming_nvfp4_w4a8 as patch_cli

_SOURCES = {
    "humming/config/config.py": """
class LayerConfig:
    _cpp_extra_names: ClassVar[tuple[str, ...]] = (
        "is_channel_weight_scale",
        "is_block_weight_scale",
        "is_group_weight_scale",
        "is_tensor_weight_scale",
        "has_input_scale",
    )

    def __post_init__(self):
        self.is_group_weight_scale = self.weight_scale_type in [
            WeightScaleType.GROUP,
            WeightScaleType.GROUP_TENSOR,
        ]
""".lstrip(),
    "humming/kernel/humming.py": """
    def check_scale(self):
        if self.input_scale_group_size > 0:
            assert self.input_scale_group_size >= 256 // self.a_dtype.num_bits
        if self.weight_scale_group_size > 0:
            assert self.weight_scale_group_size >= 256 // self.a_dtype.num_bits
        if self.weight_scale_group_size_n > 1:
            assert self.weight_scale_group_size_n >= 64
""".lstrip(),
    "humming/layer.py": """
        if "TENSOR" in str(meta.weight_scale_type):
            global_scale = tensors.get("global_scale", None)
        else:
            global_scale = None

        if meta.use_fused_e8m0_scale:
            assert weight_scale is not None
""".lstrip(),
    "humming/include/humming/memory/s2r_loader/loader_bs.cuh": """
  static constexpr bool kIsBlock = LayerConfig::kIsBlockWeightScale;
  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;
  static constexpr uint32_t kGroupSize = kIsChannel ? BlockShape::K : LayerConfig::kWeightScaleGroupSize;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kNumSubBlocks = WarpShape::N / 16;
  static constexpr uint32_t kNumScalesPerSubBlock = !kUseFusedE8m0Scale && (kIsChannel || (ElementA::kBits != 16 && !kUseWgmma)) ? 4 : 2;
  static constexpr uint32_t kNumScales = kNumSubBlocks * kNumScalesPerSubBlock;
  static constexpr uint32_t kNumBytesPerThread = kNumScales * ElementBS::kBits / 8;

  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    if constexpr (kIsBlock) {
      load_block(smem_ptr, regs_ptr, iter_id);
    } else {
      load_group_or_channel(smem_ptr, regs_ptr, iter_id);
    }
  }

  CUDA_INLINE
  void load_group_or_channel(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    uint32_t warp_id = threadIdx.x / 32;

    uint32_t n_warp_id = warp_id % N_WARPS / kNumWarpsPerMiniBlock;
    constexpr uint32_t warp_load_delta = (16 / kNumScalesPerSubBlock);
    uint32_t s_sh_rd = (kLoadItersPerGroup * warp_load_delta * kNumWarpsPerMiniBlock) * n_warp_id;

    if constexpr (kUseFusedE8m0Scale) {
      s_sh_rd += (threadIdx.x % 32) / 4 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    } else if constexpr (kUseWgmma && kIsChannel) {
      s_sh_rd += (threadIdx.x % 32) / 8 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    } else if constexpr (kUseWgmma && ElementA::kBits != 16) {
      s_sh_rd += (threadIdx.x % 32) / 4 * kNumWarpsPerMiniBlock + warp_id % kNumWarpsPerMiniBlock;
    }

    if constexpr (kGroupSize < BlockShape::K) {
      uint32_t k_index = (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K + iter_id * kPartMmaShapeK;
      uint32_t group_index = k_index / kGroupSize;
      s_sh_rd += group_index * kSmemStrideLoadType;
    };

    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);
    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);

    PRAGMA_UNROLL
    for (uint32_t j = 0; j < kLoadItersPerGroup; j++) {
      uint32_t smem_idx = warp_load_delta * j + s_sh_rd;
      reg_ptr_load[j] = smem_ptr_load[smem_idx];
    }
  };
""".lstrip(),
    "humming/include/humming/arith/mainloop_arith.cuh": """
class MainloopArithmetic : F16Conversion<ElementC> {
private:
  static constexpr bool kUseWgmma = MmaOpClass::kMmaType == MmaType::WGMMA;
  static constexpr bool kIsF16Accum = MmaOpClass::kCTypeBits == 16;
  static constexpr bool kHasZeroPoint = LayerConfig::kHasZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;
  static constexpr uint32_t kNumSubBlocksN = WarpShape::N / 16;
  static constexpr uint32_t kNumBSPerSubBlock = !kUseFusedE8m0Scale && !kUseWgmma && ElementA::kBits < 16 ? 4 : 2;

  CUDA_INLINE
  void may_apply_bs_and_zp_on_b(uint32_t *regs_b, uint32_t j, uint32_t buffer_id) {
    may_process_bs_before_apply_on_b(j, buffer_id);

    if constexpr (ElementA::kBits == 16 && kExpOffset.x) {
      uint32_t exp_val = kExpOffset.x;
    }
  };

  CUDA_INLINE
  void may_apply_as_and_bs_on_wgmma_c(uint32_t *regs_c_ptr, uint32_t m, uint32_t k, uint32_t iter_id) {
    if constexpr (ElementA::kBits == 16) return;
    if constexpr (!kIsGroupInputScale && !kIsGroupWeightScale && !kIsBlockWeightScale) return;
    if constexpr (!kUseWgmma) return;
    if constexpr (kUseFusedE8m0Scale) return;
""".lstrip(),
    "humming/include/humming/mma/wgmma.cuh": """
struct WgmmaFixture {
  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
      bool scale_d = true;
      constexpr bool kApplyScaleOnC = !kUseFusedE8m0Scale && ElementA::kBits != 16 &&
          (LayerConfig::kInputScaleGroupSize > 0 || LayerConfig::kWeightScaleGroupSize > 0);
      if constexpr (!kUseFusedE8m0Scale && ElementA::kBits != 16 && LayerConfig::kInputScaleGroupSize > 0) {
        scale_d = (iter_id * kPartMmaShapeK) % LayerConfig::kInputScaleGroupSize > 0;
      }
      if constexpr (!kUseFusedE8m0Scale && ElementA::kBits != 16 && LayerConfig::kWeightScaleGroupSize > 0) {
        scale_d = scale_d && (iter_id * kPartMmaShapeK) % LayerConfig::kWeightScaleGroupSize > 0;
      }
  };

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    uint32_t index = 0;
    constexpr bool kIsGroupWeightScale = LayerConfig::kIsGroupWeightScale;
    constexpr bool kIsBlockWeightScale = LayerConfig::kIsBlockWeightScale;

    if constexpr (ElementA::kBits < 16 && !kUseFusedE8m0Scale && (kIsGroupWeightScale || kIsBlockWeightScale)) {
      index = 1;
    }

    return regs_c_as_ptr<T>(index);
  };
""".lstrip(),
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _make_tree(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "site-packages"
    fixture_targets = []
    for target in PATCH_TARGETS:
        source = _SOURCES[target.relative_path]
        path = root / target.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.encode())
        fixture_targets.append(
            dataclasses.replace(target, pristine_sha256=_sha256(source))
        )
    monkeypatch.setattr(humming_patch, "PATCH_TARGETS", tuple(fixture_targets))
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        target.relative_path: (root / target.relative_path).read_bytes()
        for target in humming_patch.PATCH_TARGETS
    }


def test_default_targets_pin_all_six_humming_sources():
    assert {target.relative_path for target in PATCH_TARGETS} == set(_SOURCES)
    assert all(len(target.pristine_sha256) == 64 for target in PATCH_TARGETS)


def test_patch_changes_all_targets_and_is_idempotent(tmp_path, monkeypatch):
    root = _make_tree(tmp_path, monkeypatch)

    first = patch_humming_tree(root, apply=True)
    after_first = _snapshot(root)
    second = patch_humming_tree(root, apply=True)

    assert first.status == "patched"
    assert first.wrote is True
    assert second.status == "patched"
    assert second.wrote is False
    assert _snapshot(root) == after_first
    assert all(MARKER.encode() in data for data in after_first.values())

    for target in humming_patch.PATCH_TARGETS:
        backup = root / f"{target.relative_path}{BACKUP_SUFFIX}"
        assert backup.read_bytes() == _SOURCES[target.relative_path].encode()


def test_patch_accepts_crlf_checkout_and_preserves_original_backup(
    tmp_path, monkeypatch
):
    root = _make_tree(tmp_path, monkeypatch)
    target = humming_patch.PATCH_TARGETS[0]
    path = root / target.relative_path
    crlf_source = path.read_bytes().replace(b"\n", b"\r\n")
    path.write_bytes(crlf_source)

    report = patch_humming_tree(root, apply=True)

    assert report.status == "patched"
    backup = Path(f"{path}{BACKUP_SUFFIX}")
    assert backup.read_bytes() == crlf_source


def test_atomic_replacement_preserves_source_mode(tmp_path, monkeypatch):
    root = _make_tree(tmp_path, monkeypatch)
    target = humming_patch.PATCH_TARGETS[0]
    path = root / target.relative_path
    path.chmod(0o644)
    before_mode = stat.S_IMODE(path.stat().st_mode)

    patch_humming_tree(root, apply=True)

    assert stat.S_IMODE(path.stat().st_mode) == before_mode


def test_atomic_backup_refuses_to_truncate_an_existing_path(tmp_path):
    source = tmp_path / "source.py"
    backup = tmp_path / "source.py.backup"
    source.write_bytes(b"pristine")
    backup.write_bytes(b"do-not-overwrite")

    with pytest.raises(FileExistsError):
        humming_patch._atomic_backup(source, backup, source.read_bytes())

    assert backup.read_bytes() == b"do-not-overwrite"


def test_validation_failure_is_transactional(tmp_path, monkeypatch):
    root = _make_tree(tmp_path, monkeypatch)
    broken = root / humming_patch.PATCH_TARGETS[-1].relative_path
    broken.write_text("unknown source", encoding="utf-8")
    before = _snapshot(root)

    with pytest.raises(PatchError, match="unknown source hash"):
        patch_humming_tree(root, apply=True)

    assert _snapshot(root) == before
    assert not list(root.rglob(f"*{BACKUP_SUFFIX}"))


def test_missing_anchor_is_rejected_before_any_write(tmp_path, monkeypatch):
    root = _make_tree(tmp_path, monkeypatch)
    target = humming_patch.PATCH_TARGETS[0]
    path = root / target.relative_path
    bad_source = path.read_text().replace('        "has_input_scale",\n', "")
    path.write_bytes(bad_source.encode())
    replacement = dataclasses.replace(target, pristine_sha256=_sha256(bad_source))
    monkeypatch.setattr(
        humming_patch,
        "PATCH_TARGETS",
        (replacement,) + humming_patch.PATCH_TARGETS[1:],
    )
    before = _snapshot(root)

    with pytest.raises(PatchError, match="anchor"):
        patch_humming_tree(root, apply=True)

    assert _snapshot(root) == before


def test_atomic_write_failure_restores_sources_and_removes_new_backups(
    tmp_path, monkeypatch
):
    root = _make_tree(tmp_path, monkeypatch)
    before = _snapshot(root)
    real_atomic_write = humming_patch._atomic_write
    failed = False

    def fail_first_patched_write(path, data):
        nonlocal failed
        if not failed and MARKER.encode() in data:
            failed = True
            raise OSError("injected replacement failure")
        real_atomic_write(path, data)

    monkeypatch.setattr(humming_patch, "_atomic_write", fail_first_patched_write)

    with pytest.raises(OSError, match="injected replacement failure"):
        patch_humming_tree(root, apply=True)

    assert _snapshot(root) == before
    for target in humming_patch.PATCH_TARGETS:
        assert not Path(f"{root / target.relative_path}{BACKUP_SUFFIX}").exists()


def test_check_requires_a_completely_patched_tree(tmp_path, monkeypatch):
    root = _make_tree(tmp_path, monkeypatch)
    assert patch_humming_tree(root, apply=False).status == "pristine"

    patch_humming_tree(root, apply=True)
    assert patch_humming_tree(root, apply=False).status == "patched"

    first_target = humming_patch.PATCH_TARGETS[0]
    (root / first_target.relative_path).write_bytes(
        _SOURCES[first_target.relative_path].encode()
    )
    with pytest.raises(PatchError, match="partially patched"):
        patch_humming_tree(root, apply=False)


def test_apply_resumes_exact_partial_tree_from_valid_backups(tmp_path, monkeypatch):
    root = _make_tree(tmp_path, monkeypatch)
    patch_humming_tree(root, apply=True)
    restored = humming_patch.PATCH_TARGETS[:2]
    for target in restored:
        path = root / target.relative_path
        backup = Path(f"{path}{BACKUP_SUFFIX}")
        path.write_bytes(backup.read_bytes())

    with pytest.raises(PatchError, match="partially patched"):
        patch_humming_tree(root, apply=False)

    report = patch_humming_tree(root, apply=True)

    assert report.status == "patched"
    assert report.wrote is True
    assert patch_humming_tree(root, apply=False).status == "patched"


def test_apply_resumes_after_backups_created_before_first_replace(
    tmp_path, monkeypatch
):
    root = _make_tree(tmp_path, monkeypatch)
    for target in humming_patch.PATCH_TARGETS:
        path = root / target.relative_path
        Path(f"{path}{BACKUP_SUFFIX}").write_bytes(path.read_bytes())

    with pytest.raises(PatchError, match="unexpected backups"):
        patch_humming_tree(root, apply=False)

    report = patch_humming_tree(root, apply=True)

    assert report.status == "patched"
    assert report.wrote is True


def test_cli_check_and_json_report(tmp_path, monkeypatch):
    root = _make_tree(tmp_path, monkeypatch)
    report_path = tmp_path / "patch-report.json"

    assert patch_cli.main(["--root", str(root), "--check"]) == 1
    assert patch_cli.main(["--root", str(root), "--json", str(report_path)]) == 0
    assert report_path.exists()
    assert patch_cli.main(["--root", str(root), "--check"]) == 0


def test_cli_can_be_invoked_by_file_path():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "pipeline/slurm/patch_humming_nvfp4_w4a8.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: patch_humming_nvfp4_w4a8.py" in result.stdout


def test_policy_contract_is_present_in_patched_sources(tmp_path, monkeypatch):
    root = _make_tree(tmp_path, monkeypatch)
    patch_humming_tree(root, apply=True)

    config = (root / "humming/config/config.py").read_text()
    kernel = (root / "humming/kernel/humming.py").read_text()
    layer = (root / "humming/layer.py").read_text()
    loader = (
        root / "humming/include/humming/memory/s2r_loader/loader_bs.cuh"
    ).read_text()
    arith = (root / "humming/include/humming/arith/mainloop_arith.cuh").read_text()
    wgmma = (root / "humming/include/humming/mma/wgmma.cuh").read_text()

    for predicate in (
        "MmaType.WGMMA",
        "dtypes.float8e4m3",
        "dtypes.float4e2m1",
        "weight_scale_group_size == 16",
        "WeightScaleType.GROUP_TENSOR",
        "not self.has_zero_point",
    ):
        assert predicate in config
    assert '"use_nvfp4_w4a8_g16"' in config

    assert "allow_nvfp4_w4a8_g16" in kernel
    assert "not self.use_f16_accum" in kernel
    assert "or allow_nvfp4_w4a8_g16" in kernel

    assert "kUseNvfp4W4a8G16 ? 4" in loader
    assert "group_index + 1" in loader
    assert "subblock * 4 + 2" in loader
    assert "kUseNvfp4W4a8G16 ? 4" in arith
    assert "j * 4 + scale_byte" in arith
    assert "kScaleByteForBRegister" in arith
    assert "{0, 1, 2, 3}" in arith
    assert "0.125f" in arith
    assert "num42float4" in arith
    assert "if constexpr (kUseNvfp4W4a8G16) return;" in arith

    assert wgmma.count("!kUseNvfp4W4a8G16") >= 4
    assert "final_regs_c_as_ptr" in wgmma
    assert layer.count("global_scale * 8.0") == 1
