# ruff: noqa: E501 -- exact upstream C++ anchors intentionally retain source lines.
"""Fail-closed source overlay for Humming 0.1.10 NVFP4 W4A8 support."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

PINNED_VERSION = "0.1.10"
PINNED_TAG = "v0.1.10"
PINNED_COMMIT = "4351af3a8fcdce1a8dee50104ba49566af2427fb"
MARKER = "LLMC_NVFP4_W4A8_G16_V1"
BACKUP_SUFFIX = ".llmc-orig-v0.1.10"


class PatchError(RuntimeError):
    """Raised when the installed source cannot be patched deterministically."""


Transform = Callable[[str], str]


@dataclasses.dataclass(frozen=True)
class PatchTarget:
    relative_path: str
    pristine_sha256: str
    transform: Transform


@dataclasses.dataclass(frozen=True)
class PatchFileReport:
    relative_path: str
    before_sha256: str
    after_sha256: str
    status: str

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PatchReport:
    package_version: str
    root: str
    pinned_tag: str
    pinned_commit: str
    status: str
    wrote: bool
    files: tuple[PatchFileReport, ...]

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["files"] = [item.to_dict() for item in self.files]
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def _patch_config(text: str) -> str:
    text = _replace_once(
        text,
        '        "has_input_scale",\n    )',
        '        "has_input_scale",\n        "use_nvfp4_w4a8_g16",\n    )',
        "config cpp-extra anchor",
    )
    anchor = """        self.is_group_weight_scale = self.weight_scale_type in [
            WeightScaleType.GROUP,
            WeightScaleType.GROUP_TENSOR,
        ]"""
    replacement = (
        anchor
        + """

        # LLMC_NVFP4_W4A8_G16_V1: exact packed NVFP4 -> Hopper FP8 policy.
        self.use_nvfp4_w4a8_g16 = (
            self.mma_type == MmaType.WGMMA
            and self.a_dtype == dtypes.float8e4m3
            and self.b_dtype == dtypes.float4e2m1
            and self.bs_dtype == dtypes.float8e4m3
            and self.input_scale_group_size == 0
            and self.weight_scale_group_size == 16
            and self.weight_scale_type == WeightScaleType.GROUP_TENSOR
            and not self.has_zero_point
        )"""
    )
    return _replace_once(text, anchor, replacement, "config policy anchor")


def _patch_kernel(text: str) -> str:
    anchor = """        if self.weight_scale_group_size > 0:
            assert self.weight_scale_group_size >= 256 // self.a_dtype.num_bits"""
    replacement = """        if self.weight_scale_group_size > 0:
            # LLMC_NVFP4_W4A8_G16_V1: the specialized WGMMA path consumes
            # distinct K16 scales on B and requires FP32 accumulation.
            allow_nvfp4_w4a8_g16 = (
                self.use_nvfp4_w4a8_g16 and not self.use_f16_accum
            )
            assert (
                self.weight_scale_group_size >= 256 // self.a_dtype.num_bits
                or allow_nvfp4_w4a8_g16
            )"""
    return _replace_once(text, anchor, replacement, "kernel scale-check anchor")


def _patch_layer(text: str) -> str:
    anchor = """        else:
            global_scale = None

        if meta.use_fused_e8m0_scale:"""
    replacement = """        else:
            global_scale = None

        # LLMC_NVFP4_W4A8_G16_V1: B-register scaling uses scale / 8, so
        # compensate exactly once without mutating the checkpoint tensor.
        if meta.use_nvfp4_w4a8_g16:
            assert global_scale is not None
            global_scale = global_scale * 8.0

        if meta.use_fused_e8m0_scale:"""
    return _replace_once(text, anchor, replacement, "layer global-scale anchor")


def _patch_loader_bs(text: str) -> str:
    text = _replace_once(
        text,
        "  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;",
        "  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;\n"
        "  // LLMC_NVFP4_W4A8_G16_V1\n"
        "  static constexpr bool kUseNvfp4W4a8G16 = LayerConfig::kUseNvfp4W4a8G16;",
        "loader policy anchor",
    )
    old_scales = (
        "  static constexpr uint32_t kNumScalesPerSubBlock = "
        "!kUseFusedE8m0Scale && (kIsChannel || (ElementA::kBits != 16 && "
        "!kUseWgmma)) ? 4 : 2;"
    )
    new_scales = (
        "  static constexpr uint32_t kNumScalesPerSubBlock = "
        "kUseNvfp4W4a8G16 ? 4 : (!kUseFusedE8m0Scale && "
        "(kIsChannel || (ElementA::kBits != 16 && !kUseWgmma)) ? 4 : 2);"
    )
    text = _replace_once(text, old_scales, new_scales, "loader scale-count anchor")

    load_anchor = """  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    if constexpr (kIsBlock) {
      load_block(smem_ptr, regs_ptr, iter_id);
    } else {
      load_group_or_channel(smem_ptr, regs_ptr, iter_id);
    }
  }"""
    load_replacement = """  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    if constexpr (kUseNvfp4W4a8G16) {
      load_nvfp4_w4a8_g16(smem_ptr, regs_ptr, iter_id);
    } else if constexpr (kIsBlock) {
      load_block(smem_ptr, regs_ptr, iter_id);
    } else {
      load_group_or_channel(smem_ptr, regs_ptr, iter_id);
    }
  }

  CUDA_INLINE
  void load_nvfp4_w4a8_g16(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    static_assert(kUseWgmma);
    static_assert(kIsGroup);
    static_assert(kGroupSize == 16);
    static_assert(kPartMmaShapeK == 32);
    static_assert(ElementA::kBits == 8);
    static_assert(ElementBS::kBits == 8);

    // One ordinary WGMMA scale row supplies two bytes per N16 subblock.
    // Gather the same N positions from both adjacent K16 rows, then interleave
    // each N16 fragment as [two K0:16 scales][two K16:32 scales].
    constexpr uint32_t kScalesPerK16SubBlock = 2;
    constexpr uint32_t kBytesPerK16 =
        kNumSubBlocks * kScalesPerK16SubBlock * ElementBS::kBits / 8;
    constexpr uint32_t kRowsPerMiniBlock = 128 / kScalesPerK16SubBlock;
    constexpr uint32_t kWarpsPerMiniBlock = CEIL_DIV(kRowsPerMiniBlock, WarpShape::N);
    constexpr uint32_t kMaxBytesPerLoad = ElementBS::kBits / kWarpsPerMiniBlock;
    constexpr uint32_t kBytesPerLoad = MIN(kBytesPerK16, kMaxBytesPerLoad);
    using Nvfp4LoadType = typename LoadTypeChooser<kBytesPerLoad>::Type;
    static_assert(sizeof(Nvfp4LoadType) == kBytesPerK16);

    constexpr uint32_t kWarpLoadDelta = 16 / kScalesPerK16SubBlock;
    constexpr uint32_t kSmemStrideNvfp4 =
        kSmemStride * 16 / sizeof(Nvfp4LoadType);
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t n_warp_id = warp_id % N_WARPS / kWarpsPerMiniBlock;
    uint32_t s_sh_rd = kWarpLoadDelta * kWarpsPerMiniBlock * n_warp_id;
    s_sh_rd += (threadIdx.x % 32) / 4 * kWarpsPerMiniBlock;
    s_sh_rd += warp_id % kWarpsPerMiniBlock;

    uint32_t k_index =
        (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K
        + iter_id * kPartMmaShapeK;
    uint32_t group_index = k_index / kGroupSize;
    s_sh_rd += group_index * kSmemStrideNvfp4;

    const Nvfp4LoadType *smem_ptr_load =
        reinterpret_cast<const Nvfp4LoadType *>(smem_ptr);
    uint32_t next_group_index = group_index + 1;
    Nvfp4LoadType row0 = smem_ptr_load[s_sh_rd];
    Nvfp4LoadType row1 = smem_ptr_load[
        s_sh_rd + (next_group_index - group_index) * kSmemStrideNvfp4];
    const uint8_t *row0_bytes = reinterpret_cast<const uint8_t *>(&row0);
    const uint8_t *row1_bytes = reinterpret_cast<const uint8_t *>(&row1);
    uint8_t *result_bytes = reinterpret_cast<uint8_t *>(regs_ptr);
    PRAGMA_UNROLL
    for (uint32_t subblock = 0; subblock < kNumSubBlocks; subblock++) {
      result_bytes[subblock * 4] = row0_bytes[subblock * 2];
      result_bytes[subblock * 4 + 1] = row0_bytes[subblock * 2 + 1];
      result_bytes[subblock * 4 + 2] = row1_bytes[subblock * 2];
      result_bytes[subblock * 4 + 3] = row1_bytes[subblock * 2 + 1];
    }
  }"""
    return _replace_once(text, load_anchor, load_replacement, "loader dispatch anchor")


def _patch_mainloop_arith(text: str) -> str:
    text = _replace_once(
        text,
        "class MainloopArithmetic : F16Conversion<ElementC> {",
        "class MainloopArithmetic : F16Conversion<ElementC>, F8Conversion<ElementA> {",
        "mainloop conversion-base anchor",
    )
    text = _replace_once(
        text,
        "  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;",
        "  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;\n"
        "  // LLMC_NVFP4_W4A8_G16_V1\n"
        "  static constexpr bool kUseNvfp4W4a8G16 = LayerConfig::kUseNvfp4W4a8G16;",
        "mainloop policy anchor",
    )
    old_count = (
        "  static constexpr uint32_t kNumBSPerSubBlock = "
        "!kUseFusedE8m0Scale && !kUseWgmma && ElementA::kBits < 16 ? 4 : 2;"
    )
    new_count = (
        "  static constexpr uint32_t kNumBSPerSubBlock = "
        "kUseNvfp4W4a8G16 ? 4 : (!kUseFusedE8m0Scale && !kUseWgmma && "
        "ElementA::kBits < 16 ? 4 : 2);"
    )
    text = _replace_once(text, old_count, new_count, "mainloop scale-count anchor")

    apply_anchor = """  void may_apply_bs_and_zp_on_b(uint32_t *regs_b, uint32_t j, uint32_t buffer_id) {
    may_process_bs_before_apply_on_b(j, buffer_id);"""
    apply_replacement = """  void may_apply_bs_and_zp_on_b(uint32_t *regs_b, uint32_t j, uint32_t buffer_id) {
    if constexpr (kUseNvfp4W4a8G16) {
      static_assert(kUseWgmma);
      static_assert(!kIsF16Accum);
      static_assert(std::is_same<ElementA, Float8E4M3>::value);
      static_assert(std::is_same<ElementB, Float4E2M1>::value);
      static_assert(std::is_same<ElementBS, Float8E4M3>::value);
      static_assert(kWeightScaleGroupSize == 16);
      static_assert(!kHasZeroPoint);
      static_assert(sizeof(typename MmaOpClass::BRegisters) / sizeof(uint32_t) == 4);

      using F8 = typename F8Conversion<ElementA>::scalar_t;
      using F8x4 = typename F8Conversion<ElementA>::scalar_t4;
      F8x4 *b_vals = reinterpret_cast<F8x4 *>(regs_b);
      F8 *bs_vals = reinterpret_cast<F8 *>(bs[buffer_id]);

      // dequant_b1248 reverses registers only within source-word pairs:
      // output {0,1} comes from K0:16 and {2,3} from K16:32. The specialized
      // loader has already arranged its bytes in that exact B-register order.
      constexpr uint32_t kScaleByteForBRegister[4] = {0, 1, 2, 3};

      PRAGMA_UNROLL
      for (uint32_t register_id = 0; register_id < 4; register_id++) {
        float4 values = F8Conversion<ElementA>::num42float4(b_vals[register_id]);
        uint32_t scale_byte = kScaleByteForBRegister[register_id];
        float scale = static_cast<float>(bs_vals[j * 4 + scale_byte]) * 0.125f;
        F8 *packed = reinterpret_cast<F8 *>(&b_vals[register_id]);
        packed[0] = F8(values.x * scale);
        packed[1] = F8(values.y * scale);
        packed[2] = F8(values.z * scale);
        packed[3] = F8(values.w * scale);
      }
      return;
    }

    may_process_bs_before_apply_on_b(j, buffer_id);"""
    text = _replace_once(
        text, apply_anchor, apply_replacement, "mainloop B-scale anchor"
    )

    c_anchor = """  void may_apply_as_and_bs_on_wgmma_c(uint32_t *regs_c_ptr, uint32_t m, uint32_t k, uint32_t iter_id) {
    if constexpr (ElementA::kBits == 16) return;"""
    c_replacement = """  void may_apply_as_and_bs_on_wgmma_c(uint32_t *regs_c_ptr, uint32_t m, uint32_t k, uint32_t iter_id) {
    if constexpr (kUseNvfp4W4a8G16) return;
    if constexpr (ElementA::kBits == 16) return;"""
    return _replace_once(text, c_anchor, c_replacement, "mainloop C-scale anchor")


def _patch_wgmma(text: str) -> str:
    text = _replace_once(
        text,
        "  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;",
        "  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;\n"
        "  // LLMC_NVFP4_W4A8_G16_V1\n"
        "  static constexpr bool kUseNvfp4W4a8G16 = LayerConfig::kUseNvfp4W4a8G16;",
        "wgmma policy anchor",
    )
    text = _replace_once(
        text,
        "      constexpr bool kApplyScaleOnC = !kUseFusedE8m0Scale && ElementA::kBits != 16 &&",
        "      constexpr bool kApplyScaleOnC = !kUseNvfp4W4a8G16 && !kUseFusedE8m0Scale && ElementA::kBits != 16 &&",
        "wgmma accumulator-scale anchor",
    )
    text = _replace_once(
        text,
        "      if constexpr (!kUseFusedE8m0Scale && ElementA::kBits != 16 && LayerConfig::kInputScaleGroupSize > 0) {",
        "      if constexpr (!kUseNvfp4W4a8G16 && !kUseFusedE8m0Scale && ElementA::kBits != 16 && LayerConfig::kInputScaleGroupSize > 0) {",
        "wgmma input-boundary anchor",
    )
    text = _replace_once(
        text,
        "      if constexpr (!kUseFusedE8m0Scale && ElementA::kBits != 16 && LayerConfig::kWeightScaleGroupSize > 0) {",
        "      if constexpr (!kUseNvfp4W4a8G16 && !kUseFusedE8m0Scale && ElementA::kBits != 16 && LayerConfig::kWeightScaleGroupSize > 0) {",
        "wgmma weight-boundary anchor",
    )
    text = _replace_once(
        text,
        "    if constexpr (ElementA::kBits < 16 && !kUseFusedE8m0Scale && (kIsGroupWeightScale || kIsBlockWeightScale)) {",
        "    if constexpr (ElementA::kBits < 16 && !kUseNvfp4W4a8G16 && !kUseFusedE8m0Scale && (kIsGroupWeightScale || kIsBlockWeightScale)) {",
        "wgmma final-accumulator anchor",
    )
    return text


PATCH_TARGETS = (
    PatchTarget(
        "humming/config/config.py",
        "20406dc0ce4fcb957a035c4f04fc0f1c41746519be495ce39bcb2ff270d327e4",
        _patch_config,
    ),
    PatchTarget(
        "humming/kernel/humming.py",
        "4f51361a17de489366fd849fa305ebc6d055b117531352c5963985f19ff09804",
        _patch_kernel,
    ),
    PatchTarget(
        "humming/layer.py",
        "31a6d7c0394683b9c1236fd93dc275e19a289b66488a57ca8b531fe30f29c8dd",
        _patch_layer,
    ),
    PatchTarget(
        "humming/include/humming/memory/s2r_loader/loader_bs.cuh",
        "4f3d64befa294c90a23e05f3cde28474ee46cae4ac9f36ac605dd8aec76480ae",
        _patch_loader_bs,
    ),
    PatchTarget(
        "humming/include/humming/arith/mainloop_arith.cuh",
        "1ca6831856827cebb1402bc167a4468a1002862ebe0f498b81017539618a8a5d",
        _patch_mainloop_arith,
    ),
    PatchTarget(
        "humming/include/humming/mma/wgmma.cuh",
        "77f89c87c8404818e1596592f4423d3c5609d1fcbad17a2f2fd373d92fe1383a",
        _patch_wgmma,
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_newlines(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _source_sha256(data: bytes) -> str:
    return _sha256(_normalize_newlines(data))


def _transform_bytes(target: PatchTarget, pristine: bytes) -> bytes:
    try:
        text = _normalize_newlines(pristine).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchError(f"{target.relative_path}: source is not UTF-8") from exc
    patched = target.transform(text)
    if MARKER not in patched:
        raise PatchError(f"{target.relative_path}: transform omitted marker {MARKER}")
    return patched.encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    source_stat = path.stat()
    source_mode = stat.S_IMODE(source_stat.st_mode)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.chmod(temporary, source_mode)
        if hasattr(os, "chown"):
            os.chown(temporary, source_stat.st_uid, source_stat.st_gid)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_backup(source: Path, backup: Path, data: bytes) -> None:
    """Durably publish a backup without ever truncating an existing path."""

    source_stat = source.stat()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=backup.parent, prefix=f".{backup.name}.", delete=False
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.chmod(temporary, stat.S_IMODE(source_stat.st_mode))
        if hasattr(os, "chown"):
            os.chown(temporary, source_stat.st_uid, source_stat.st_gid)
        # Hard-link publication is atomic and fails if the backup already
        # exists; unlike rename it cannot overwrite another transaction.
        os.link(temporary, backup)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _resolve_installed_humming() -> tuple[Path, str]:
    try:
        version = importlib.metadata.version("humming-kernels")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PatchError("humming-kernels is not installed") from exc
    if version != PINNED_VERSION:
        raise PatchError(
            f"humming-kernels=={PINNED_VERSION} is required, found {version}"
        )
    try:
        import humming
    except ImportError as exc:
        raise PatchError(
            "humming-kernels distribution is present but import failed"
        ) from exc
    return Path(humming.__file__).resolve().parent.parent, version


def patch_humming_tree(root: Path, *, apply: bool) -> PatchReport:
    """Validate and optionally apply the complete six-file Humming overlay."""

    root = Path(root).resolve()
    prepared: list[tuple[PatchTarget, Path, bytes, bytes, str]] = []
    states: set[str] = set()
    pristine_paths_with_backups: list[str] = []

    for target in PATCH_TARGETS:
        path = root / target.relative_path
        backup = Path(f"{path}{BACKUP_SUFFIX}")
        if not path.is_file():
            raise PatchError(f"missing patch target: {target.relative_path}")
        current = path.read_bytes()
        current_hash = _source_sha256(current)

        if current_hash == target.pristine_sha256:
            if backup.exists():
                if (
                    not backup.is_file()
                    or _source_sha256(backup.read_bytes()) != current_hash
                ):
                    raise PatchError(
                        f"{target.relative_path}: pristine source has invalid backup"
                    )
                pristine_paths_with_backups.append(target.relative_path)
            patched = _transform_bytes(target, current)
            state = "pristine"
        elif MARKER.encode() in current:
            if not backup.is_file():
                raise PatchError(
                    f"{target.relative_path}: patched marker exists without backup"
                )
            pristine = backup.read_bytes()
            backup_hash = _source_sha256(pristine)
            if backup_hash != target.pristine_sha256:
                raise PatchError(
                    f"{target.relative_path}: backup has unknown source hash {backup_hash}"
                )
            patched = _transform_bytes(target, pristine)
            if _normalize_newlines(current) != patched:
                raise PatchError(
                    f"{target.relative_path}: marker exists but patched bytes are unknown"
                )
            state = "patched"
        else:
            raise PatchError(
                f"{target.relative_path}: unknown source hash {current_hash}; "
                f"expected {target.pristine_sha256}"
            )

        states.add(state)
        prepared.append((target, path, current, patched, state))

    if len(states) != 1:
        if not apply:
            raise PatchError(f"partially patched Humming tree: states={sorted(states)}")
        missing_or_invalid_backups = []
        for target, path, _, _, _ in prepared:
            backup = Path(f"{path}{BACKUP_SUFFIX}")
            if (
                not backup.is_file()
                or _source_sha256(backup.read_bytes()) != target.pristine_sha256
            ):
                missing_or_invalid_backups.append(target.relative_path)
        if missing_or_invalid_backups:
            raise PatchError(
                "partially patched Humming tree cannot resume without exact backups: "
                + ", ".join(missing_or_invalid_backups)
            )
        for _, path, _, patched, file_state in prepared:
            if file_state == "pristine":
                _atomic_write(path, patched)
        states = {"patched"}
        prepared = [
            (target, path, current, patched, "patched")
            for target, path, current, patched, _ in prepared
        ]
        resumed_partial = True
    else:
        resumed_partial = False

    state = next(iter(states))
    if state == "pristine" and pristine_paths_with_backups:
        if not apply or len(pristine_paths_with_backups) != len(PATCH_TARGETS):
            raise PatchError(
                "pristine Humming tree has unexpected backups: "
                + ", ".join(pristine_paths_with_backups)
            )
        # A process may terminate after creating every exact backup but before
        # replacing the first source. Complete that deterministic transaction.
        for _, path, _, patched, _ in prepared:
            _atomic_write(path, patched)
        state = "patched"
        resumed_partial = True
    wrote = resumed_partial
    if state == "pristine" and apply:
        created_backups: list[Path] = []
        try:
            for target, path, current, _, _ in prepared:
                backup = Path(f"{path}{BACKUP_SUFFIX}")
                _atomic_backup(path, backup, current)
                created_backups.append(backup)

            for _, path, _, patched, _ in prepared:
                _atomic_write(path, patched)
        except FileExistsError as exc:
            for backup in created_backups:
                backup.unlink()
            raise PatchError(
                f"{target.relative_path}: refusing to overwrite existing backup"
            ) from exc
        except Exception as apply_error:
            restore_errors: list[str] = []
            for _, path, current, _, _ in prepared:
                try:
                    _atomic_write(path, current)
                except Exception as restore_error:
                    restore_errors.append(f"{path}: {restore_error}")
            if restore_errors:
                raise PatchError(
                    "overlay apply failed and source restoration was incomplete; "
                    "backups retained: " + "; ".join(restore_errors)
                ) from apply_error
            for backup in created_backups:
                backup.unlink()
            raise
        wrote = True
        state = "patched"

    files = tuple(
        PatchFileReport(
            relative_path=target.relative_path,
            before_sha256=_sha256(current),
            after_sha256=_sha256(patched if state == "patched" else current),
            status=state,
        )
        for target, _, current, patched, _ in prepared
    )
    return PatchReport(
        package_version=PINNED_VERSION,
        root=str(root),
        pinned_tag=PINNED_TAG,
        pinned_commit=PINNED_COMMIT,
        status=state,
        wrote=wrote,
        files=files,
    )


def patch_installed_humming(*, apply: bool) -> PatchReport:
    root, version = _resolve_installed_humming()
    report = patch_humming_tree(root, apply=apply)
    if report.package_version != version:
        raise PatchError("internal package-version mismatch")
    return report
