"""CPU-only tests for native MiniMax-M3 Humming W4A8 qualification."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from pipeline import m3_humming_w4a8
from pipeline.m3_checkpoint_diagnostics import classify_module
from pipeline.m3_humming_w4a8 import (
    RuntimeFacts,
    _distribution_integrity,
    classify_backend_log,
    evaluate_preflight,
    main,
)


def valid_config() -> dict[str, object]:
    return {
        "architectures": ["MiniMaxM3ForCausalLM"],
        "model_type": "minimax_m3",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 128,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": True,
                        "strategy": "token",
                        "group_size": None,
                        "dynamic": True,
                    },
                }
            },
            "ignore": [
                "re:.*block_sparse_moe[.]shared_experts[.].*",
                "re:.*block_sparse_moe[.]gate$",
                "lm_head",
            ],
        },
    }


def valid_abi_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "valid": True,
        "format": "pack-quantized",
        "inventory": {"quantized_modules": 64},
        "components": {"routed_experts": {"quantized": 64}},
        "errors": [],
    }


def valid_runtime() -> RuntimeFacts:
    return RuntimeFacts(
        vllm_version="0.24.0",
        humming_version="0.1.10",
        device_capability=(9, 0),
        humming_source_integrity="record-matched",
        humming_source_mismatches=(),
        nvfp4_overlay_detected=False,
        humming_cache_dir="/tmp/cache-m3-gptq-w4a8-v1",
        f16_accum="0",
        moe_gemm_type="indexed",
        normal_patch_status="patched",
        humming_patch_status="patched",
    )


def _set_path(root: dict[str, object], path: tuple[str, ...], value: object) -> None:
    current = root
    for key in path[:-1]:
        current = current[key]  # type: ignore[assignment,index]
    current[path[-1]] = value


def test_routed_expert_component_key_is_stable():
    module = "language_model.model.layers.3.block_sparse_moe.experts.0.w1"

    assert classify_module(module) == "routed_experts"


def test_accepts_exact_native_humming_w4a8_contract():
    report = evaluate_preflight(valid_config(), valid_abi_report(), valid_runtime())

    assert report["valid"] is True
    assert report["backend"] == "humming"
    assert report["reason_codes"] == []
    assert report["details"]["weight_group_size"] == 128
    assert report["details"]["activation_strategy"] == "token"
    assert report["details"]["gemm_type"] == "indexed"


def test_accepts_qualified_0111_packedk_side_install():
    runtime = replace(valid_runtime(), humming_version="0.1.11")

    report = evaluate_preflight(valid_config(), valid_abi_report(), runtime)

    assert report["valid"] is True
    assert report["reason_codes"] == []


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"vllm_version": "0.23.0"}, "VLLM_VERSION_MISMATCH"),
        ({"humming_version": "0.1.9"}, "HUMMING_VERSION_MISMATCH"),
        ({"humming_version": "0.1.12"}, "HUMMING_VERSION_MISMATCH"),
        ({"device_capability": (8, 0)}, "SM_NOT_90"),
        (
            {
                "humming_source_integrity": "mismatch",
                "humming_source_mismatches": ("humming/layer.py",),
            },
            "HUMMING_SOURCE_INTEGRITY",
        ),
        ({"nvfp4_overlay_detected": True}, "NVFP4_OVERLAY_PRESENT"),
        (
            {"humming_cache_dir": "/tmp/cache-nvfp4-w4a8-v1"},
            "HUMMING_CACHE_NAMESPACE",
        ),
        ({"f16_accum": "1"}, "F16_ACCUM_ENABLED"),
        # grouped_masked exists in Humming's enum but is the batched-experts
        # path, which we do not serve; anything unrecognised must fail closed
        # rather than inherit vLLM's silent fallback to indexed.
        ({"moe_gemm_type": "grouped_masked"}, "MOE_GEMM_TYPE_UNSUPPORTED"),
        ({"moe_gemm_type": "nonsense"}, "MOE_GEMM_TYPE_UNSUPPORTED"),
        ({"normal_patch_status": "missing"}, "NORMAL_PATCH_INCOMPLETE"),
        ({"humming_patch_status": "missing"}, "HUMMING_PATCH_INCOMPLETE"),
    ],
)
def test_rejects_runtime_contract_mismatch(changes, reason):
    runtime = replace(valid_runtime(), **changes)

    report = evaluate_preflight(valid_config(), valid_abi_report(), runtime)

    assert report["valid"] is False
    assert report["backend"] == "humming"
    assert report["reason_codes"] == [reason]


def test_reports_nvfp4_overlay_before_generic_source_mismatch():
    runtime = replace(
        valid_runtime(),
        humming_source_integrity="mismatch",
        humming_source_mismatches=("humming/layer.py",),
        nvfp4_overlay_detected=True,
    )

    report = evaluate_preflight(valid_config(), valid_abi_report(), runtime)

    assert report["reason_codes"] == ["NVFP4_OVERLAY_PRESENT"]


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (
            ("quantization_config", "quant_method"),
            "gptq",
            "QUANT_METHOD_MISMATCH",
        ),
        (
            ("quantization_config", "format"),
            "dense",
            "FORMAT_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "weights",
                "num_bits",
            ),
            8,
            "WEIGHT_BITS_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "weights",
                "type",
            ),
            "float",
            "WEIGHT_TYPE_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "weights",
                "symmetric",
            ),
            False,
            "WEIGHT_SYMMETRY_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "weights",
                "strategy",
            ),
            "channel",
            "WEIGHT_STRATEGY_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "weights",
                "group_size",
            ),
            64,
            "WEIGHT_GROUP_SIZE_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "weights",
                "dynamic",
            ),
            True,
            "WEIGHT_DYNAMIC_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "input_activations",
                "num_bits",
            ),
            16,
            "ACTIVATION_BITS_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "input_activations",
                "type",
            ),
            "int",
            "ACTIVATION_TYPE_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "input_activations",
                "symmetric",
            ),
            False,
            "ACTIVATION_SYMMETRY_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "input_activations",
                "strategy",
            ),
            "group",
            "ACTIVATION_STRATEGY_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "input_activations",
                "group_size",
            ),
            128,
            "ACTIVATION_GROUP_SIZE_MISMATCH",
        ),
        (
            (
                "quantization_config",
                "config_groups",
                "group_0",
                "input_activations",
                "dynamic",
            ),
            False,
            "ACTIVATION_DYNAMIC_MISMATCH",
        ),
        (
            ("quantization_config", "ignore"),
            [],
            "IGNORE_METADATA_MISSING",
        ),
    ],
)
def test_rejects_checkpoint_contract_mismatch(path, value, reason):
    config = valid_config()
    _set_path(config, path, value)

    report = evaluate_preflight(config, valid_abi_report(), valid_runtime())

    assert report["valid"] is False
    assert report["reason_codes"] == [reason]


def test_rejects_missing_activation_group_size_field():
    config = valid_config()
    group = config["quantization_config"]["config_groups"]["group_0"]  # type: ignore[index]
    del group["input_activations"]["group_size"]  # type: ignore[index]

    report = evaluate_preflight(config, valid_abi_report(), valid_runtime())

    assert report["reason_codes"] == ["ACTIVATION_GROUP_SIZE_MISMATCH"]


def test_rejects_missing_or_ambiguous_linear_group():
    config = valid_config()
    groups = config["quantization_config"]["config_groups"]  # type: ignore[index]
    groups["group_0"]["targets"] = ["Conv2d"]  # type: ignore[index]

    missing = evaluate_preflight(config, valid_abi_report(), valid_runtime())

    config = valid_config()
    groups = config["quantization_config"]["config_groups"]  # type: ignore[index]
    groups["group_1"] = copy.deepcopy(groups["group_0"])  # type: ignore[index]
    ambiguous = evaluate_preflight(config, valid_abi_report(), valid_runtime())

    assert missing["reason_codes"] == ["LINEAR_GROUP_CARDINALITY"]
    assert ambiguous["reason_codes"] == ["LINEAR_GROUP_CARDINALITY"]


def test_rejects_invalid_abi_or_missing_routed_experts():
    abi = valid_abi_report()
    abi["valid"] = False
    invalid = evaluate_preflight(valid_config(), abi, valid_runtime())

    abi = valid_abi_report()
    abi["components"]["routed_experts"]["quantized"] = 0  # type: ignore[index]
    missing = evaluate_preflight(valid_config(), abi, valid_runtime())

    assert invalid["reason_codes"] == ["ABI_INVALID"]
    assert missing["reason_codes"] == ["ROUTED_EXPERTS_MISSING"]


def valid_preflight() -> dict[str, object]:
    return {
        "schema_version": 1,
        "valid": True,
        "backend": "humming",
        "reason_codes": [],
        "details": {},
    }


@pytest.mark.parametrize(
    "quantization_line",
    [
        "INFO non-default args: {'quantization': 'humming'}",
        'INFO engine config: {"quantization": "humming"}',
        "INFO quantization=humming",
    ],
)
def test_attests_indexed_humming_backend(quantization_line):
    text = "\n".join(
        [
            quantization_line,
            "INFO Using indexed gemm for humming moe",
            "INFO Humming cache hit for gemm config",
        ]
    )

    report = classify_backend_log(text, valid_preflight())

    assert report["valid"] is True
    assert report["backend"] == "humming"
    assert report["gemm_type"] == "indexed"
    assert report["reason_codes"] == []
    assert report["details"]["compile_cache_lines"] == [
        "INFO Humming cache hit for gemm config"
    ]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "INFO Using indexed gemm for humming moe",
            "QUANTIZATION_MARKER_MISSING",
        ),
        (
            "INFO quantization=humming",
            "HUMMING_GEMM_MARKER_MISSING",
        ),
        (
            # Indexed was requested but grouped ran: a real backend, wrong arm.
            "INFO quantization=humming\n"
            "INFO Using grouped_contiguous gemm for humming moe",
            "UNEXPECTED_GEMM_TYPE_SELECTED",
        ),
        (
            "INFO quantization=humming\n"
            "INFO Using indexed gemm for humming moe\n"
            "INFO Using CUTLASS W4A8 MoE backend",
            "CUTLASS_FALLBACK_DETECTED",
        ),
        (
            "INFO quantization=humming\n"
            "INFO Using indexed gemm for humming moe\n"
            "INFO Using Marlin kernel",
            "MARLIN_FALLBACK_DETECTED",
        ),
        (
            "INFO quantization=humming\n"
            "INFO Using indexed gemm for humming moe\n"
            "INFO UnquantizedFusedMoEMethod",
            "UNQUANTIZED_FALLBACK_DETECTED",
        ),
        (
            "INFO quantization=humming\n"
            "INFO Using indexed gemm for humming moe\n"
            "INFO Using grouped_contiguous gemm for humming moe",
            "CONTRADICTORY_GEMM_MARKERS",
        ),
    ],
)
def test_rejects_ambiguous_or_fallback_server_log(text, reason):
    report = classify_backend_log(text, valid_preflight())

    assert report["valid"] is False
    assert report["reason_codes"] == [reason]


def grouped_preflight() -> dict[str, object]:
    preflight = valid_preflight()
    preflight["details"] = {"gemm_type": "grouped_contiguous"}
    return preflight


def test_attests_grouped_humming_backend():
    """Arm 3: grouped_contiguous is attested when it is the requested strategy."""
    text = "\n".join(
        [
            "INFO quantization=humming",
            "INFO Using grouped_contiguous gemm for humming moe",
        ]
    )

    report = classify_backend_log(text, grouped_preflight())

    assert report["valid"] is True
    assert report["gemm_type"] == "grouped_contiguous"
    assert report["reason_codes"] == []
    assert report["details"]["grouped_marker"] is True
    assert report["details"]["indexed_marker"] is False
    assert report["details"]["expected_gemm_type"] == "grouped_contiguous"


def test_grouped_request_that_silently_ran_indexed_is_rejected():
    """vLLM falls back to indexed for values it does not recognise.

    That fallback is the dangerous failure mode for arm 3: the server comes up,
    serves correctly, and produces numbers attributed to the wrong kernel. The
    requested strategy's marker must be present, not merely *some* Humming one.
    """
    text = "\n".join(
        [
            "INFO quantization=humming",
            "INFO Using indexed gemm for humming moe",
        ]
    )

    report = classify_backend_log(text, grouped_preflight())

    assert report["valid"] is False
    assert report["reason_codes"] == ["UNEXPECTED_GEMM_TYPE_SELECTED"]
    assert report["gemm_type"] is None


def test_grouped_request_with_no_gemm_marker_at_all_is_rejected():
    report = classify_backend_log("INFO quantization=humming", grouped_preflight())

    assert report["valid"] is False
    assert report["reason_codes"] == ["HUMMING_GEMM_MARKER_MISSING"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", "indexed"),
        ("indexed", "indexed"),
        ("grouped", "grouped_contiguous"),
        ("grouped_contiguous", "grouped_contiguous"),
        ("GROUPED", "grouped_contiguous"),
        (" indexed ", "indexed"),
        ("grouped_masked", None),
        ("dense", None),
    ],
)
def test_normalize_gemm_type_matches_vllm_alias_mapping(raw, expected):
    assert m3_humming_w4a8.normalize_gemm_type(raw) == expected


def test_rejects_invalid_preflight_before_log_classification():
    preflight = valid_preflight()
    preflight["valid"] = False

    report = classify_backend_log(
        "INFO quantization=humming\nINFO Using indexed gemm for humming moe",
        preflight,
    )

    assert report["reason_codes"] == ["PREFLIGHT_INVALID"]


def test_distribution_integrity_rejects_unhashed_humming_file(
    monkeypatch,
    tmp_path,
):
    package = tmp_path / "humming"
    package.mkdir()
    hashed = package / "layer.py"
    unhashed = package / "local_override.py"
    hashed.write_text("pristine\n", encoding="utf-8")
    unhashed.write_text("local mutation\n", encoding="utf-8")
    digest = hashlib.sha256(hashed.read_bytes()).digest()
    encoded = (
        __import__("base64")
        .urlsafe_b64encode(digest)
        .rstrip(b"=")
        .decode("ascii")
    )

    class FakeDistribution:
        files = [
            SimpleNamespace(
                parts=PurePosixPath("humming/layer.py").parts,
                hash=SimpleNamespace(mode="sha256", value=encoded),
                path="humming/layer.py",
            ),
            SimpleNamespace(
                parts=PurePosixPath("humming/local_override.py").parts,
                hash=None,
                path="humming/local_override.py",
            ),
        ]

        def locate_file(self, file):
            return tmp_path / file.path

    monkeypatch.setattr(
        "pipeline.m3_humming_w4a8.importlib.metadata.distribution",
        lambda name: FakeDistribution(),
    )

    (
        status,
        mismatches,
        overlay,
        unhashed_bytecode,
        declared,
    ) = _distribution_integrity()

    assert status == "mismatch"
    assert mismatches == ("humming/local_override.py",)
    assert overlay is False
    assert unhashed_bytecode == 0


def test_distribution_integrity_tolerates_unhashed_cached_bytecode(
    monkeypatch,
    tmp_path,
):
    """humming-kernels 0.1.10 lists unhashed __pycache__ entries in RECORD."""

    package = tmp_path / "humming"
    (package / "__pycache__").mkdir(parents=True)
    hashed = package / "layer.py"
    hashed.write_text("pristine\n", encoding="utf-8")
    present_pyc = package / "__pycache__" / "layer.cpython-312.pyc"
    present_pyc.write_bytes(b"\x00compiled")
    digest = hashlib.sha256(hashed.read_bytes()).digest()
    encoded = (
        __import__("base64")
        .urlsafe_b64encode(digest)
        .rstrip(b"=")
        .decode("ascii")
    )

    class FakeDistribution:
        files = [
            SimpleNamespace(
                parts=PurePosixPath("humming/layer.py").parts,
                hash=SimpleNamespace(mode="sha256", value=encoded),
                path="humming/layer.py",
            ),
            SimpleNamespace(
                parts=PurePosixPath(
                    "humming/__pycache__/layer.cpython-312.pyc"
                ).parts,
                hash=None,
                path="humming/__pycache__/layer.cpython-312.pyc",
            ),
            SimpleNamespace(
                parts=PurePosixPath(
                    "humming/__pycache__/dtypes.cpython-312.pyc"
                ).parts,
                hash=None,
                path="humming/__pycache__/dtypes.cpython-312.pyc",
            ),
        ]

        def locate_file(self, file):
            return tmp_path / file.path

    monkeypatch.setattr(
        "pipeline.m3_humming_w4a8.importlib.metadata.distribution",
        lambda name: FakeDistribution(),
    )

    (
        status,
        mismatches,
        overlay,
        unhashed_bytecode,
        declared,
    ) = _distribution_integrity()

    assert status == "record-matched"
    assert mismatches == ()
    assert overlay is False
    assert unhashed_bytecode == 2


def test_overlay_marker_in_unhashed_bytecode_is_still_detected(
    monkeypatch,
    tmp_path,
):
    package = tmp_path / "humming"
    (package / "__pycache__").mkdir(parents=True)
    hashed = package / "layer.py"
    hashed.write_text("pristine\n", encoding="utf-8")
    tainted = package / "__pycache__" / "layer.cpython-312.pyc"
    tainted.write_bytes(b"\x00LLMC_NVFP4_W4A8_G16_V1")
    digest = hashlib.sha256(hashed.read_bytes()).digest()
    encoded = (
        __import__("base64")
        .urlsafe_b64encode(digest)
        .rstrip(b"=")
        .decode("ascii")
    )

    class FakeDistribution:
        files = [
            SimpleNamespace(
                parts=PurePosixPath("humming/layer.py").parts,
                hash=SimpleNamespace(mode="sha256", value=encoded),
                path="humming/layer.py",
            ),
            SimpleNamespace(
                parts=PurePosixPath(
                    "humming/__pycache__/layer.cpython-312.pyc"
                ).parts,
                hash=None,
                path="humming/__pycache__/layer.cpython-312.pyc",
            ),
        ]

        def locate_file(self, file):
            return tmp_path / file.path

    monkeypatch.setattr(
        "pipeline.m3_humming_w4a8.importlib.metadata.distribution",
        lambda name: FakeDistribution(),
    )

    (
        status,
        mismatches,
        overlay,
        unhashed_bytecode,
        declared,
    ) = _distribution_integrity()

    assert overlay is True
    assert status == "record-matched"
    assert unhashed_bytecode == 1


def _single_file_distribution(tmp_path, relative, payload, record_hash):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)

    class FakeDistribution:
        files = [
            SimpleNamespace(
                parts=PurePosixPath(relative).parts,
                hash=SimpleNamespace(mode="sha256", value=record_hash),
                path=relative,
            ),
        ]

        def locate_file(self, file):
            return tmp_path / file.path

    return FakeDistribution()


def _b64(payload: bytes) -> str:
    import base64

    return (
        base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
        .rstrip(b"=")
        .decode("ascii")
    )


def test_distribution_integrity_reports_declared_patch(monkeypatch, tmp_path):
    """A reviewed patch is surfaced, not silently accepted as pristine."""

    relative = "humming/schema/compressed_tensors.py"
    patched = b'assert self.format in ["pack-quantized"]\n'
    pristine = b"assert self.format in []\n"
    distribution = _single_file_distribution(
        tmp_path, relative, patched, _b64(pristine)
    )
    monkeypatch.setattr(
        "pipeline.m3_humming_w4a8.importlib.metadata.distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        "pipeline.m3_humming_w4a8.DECLARED_PATCH_SHA256",
        {relative: hashlib.sha256(patched).hexdigest()},
    )

    status, mismatches, overlay, _, declared = _distribution_integrity()

    assert status == "record-matched-with-declared-patch"
    assert mismatches == ()
    assert declared == (relative,)
    assert overlay is False


def test_declared_path_with_unexpected_content_is_still_a_mismatch(
    monkeypatch,
    tmp_path,
):
    relative = "humming/schema/compressed_tensors.py"
    tampered = b"import os; os.system('curl evil')\n"
    pristine = b"assert self.format in []\n"
    expected_patch = b'assert self.format in ["pack-quantized"]\n'
    distribution = _single_file_distribution(
        tmp_path, relative, tampered, _b64(pristine)
    )
    monkeypatch.setattr(
        "pipeline.m3_humming_w4a8.importlib.metadata.distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        "pipeline.m3_humming_w4a8.DECLARED_PATCH_SHA256",
        {relative: hashlib.sha256(expected_patch).hexdigest()},
    )

    status, mismatches, _, _, declared = _distribution_integrity()

    assert status == "mismatch"
    assert mismatches == (relative,)
    assert declared == ()


def test_preflight_accepts_declared_patch_integrity():
    runtime = replace(
        valid_runtime(),
        humming_source_integrity="record-matched-with-declared-patch",
        humming_declared_patches=("humming/schema/compressed_tensors.py",),
    )

    report = evaluate_preflight(valid_config(), valid_abi_report(), runtime)

    assert report["valid"] is True
    assert report["details"]["humming_declared_patches"] == (
        "humming/schema/compressed_tensors.py",
    )


def test_preflight_rejects_undeclared_patch_path():
    runtime = replace(
        valid_runtime(),
        humming_source_integrity="record-matched-with-declared-patch",
        humming_declared_patches=("humming/kernel/humming.cuh",),
    )

    report = evaluate_preflight(valid_config(), valid_abi_report(), runtime)

    assert report["valid"] is False
    assert report["reason_codes"] == ["HUMMING_UNDECLARED_PATCH"]


def test_preflight_cli_writes_valid_report(monkeypatch, tmp_path):
    out = tmp_path / "preflight.json"
    monkeypatch.setattr(
        "pipeline.m3_humming_w4a8.preflight_checkpoint",
        lambda checkpoint: valid_preflight(),
    )

    rc = main(
        [
            "preflight",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["valid"] is True
    assert out.read_text(encoding="utf-8").endswith("\n")


def test_attest_cli_returns_contract_failure_and_writes_report(tmp_path):
    preflight = tmp_path / "preflight.json"
    log = tmp_path / "serve.log"
    out = tmp_path / "attestation.json"
    preflight.write_text(json.dumps(valid_preflight()), encoding="utf-8")
    log.write_text("INFO quantization=humming\n", encoding="utf-8")

    rc = main(
        [
            "attest",
            "--preflight",
            str(preflight),
            "--log",
            str(log),
            "--out",
            str(out),
        ]
    )

    assert rc == 1
    assert json.loads(out.read_text(encoding="utf-8"))["reason_codes"] == [
        "HUMMING_GEMM_MARKER_MISSING"
    ]


def test_attest_cli_returns_discovery_error_for_missing_input(tmp_path):
    rc = main(
        [
            "attest",
            "--preflight",
            str(tmp_path / "missing-preflight.json"),
            "--log",
            str(tmp_path / "missing-serve.log"),
            "--out",
            str(tmp_path / "attestation.json"),
        ]
    )

    assert rc == 2
