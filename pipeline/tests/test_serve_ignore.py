"""An ignore entry must never hide a module the checkpoint actually quantized.

This is the r8-class defect, in the words of the incident that found it:

    the serialized quantization_config still carries [...] the GPTQ recipe's broad
    quant-layout ignore regexes (re:.*self_attn[.].*, ...). vLLM checks ignore
    FIRST, so every FP8 module served as "unquantized": raw fp8 bits were cast
    into bf16 params without their scales -> garbage output.

Exit code 0, coherent-looking tokens, every offline gate green -- because no gate
compared the ignore list against the tensors.

Measured on the real GLM-5.2 AWQ checkpoint (routerfix, 20260828-150142) before
the fix: 16 of 784 quantized modules shadowed, and those 16 were the ENTIRE FP8
leg (all 10 MLA projections across layers 0 and 3, all 3 shared experts of layer
3, all 3 dense MLPs of layer 0). Not one would have served quantized.
``test_reproduces_the_measured_glm_shadowing`` encodes that case so it stays fixed.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pipeline.serve_ignore import (
    assert_no_ignore_shadowing,
    audit_checkpoint_ignore,
    checkpoint_modules,
    match_name,
    resolve_ignore_patterns,
    shadowed_by,
)


def _checkpoint(
    path: Path,
    *,
    quantized: dict[str, str],
    plain: list[str],
    ignore: list[str],
    groups: dict | None = None,
) -> Path:
    """Write a checkpoint whose tensors and config can disagree.

    ``quantized`` maps module -> marker suffix family ("fp8" writes weight +
    weight_scale, "int4" writes weight_packed + weight_scale), ``plain`` lists
    modules with only a bf16 weight.
    """
    path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for module, kind in quantized.items():
        if kind == "int4":
            tensors[f"{module}.weight_packed"] = torch.zeros(2, 2, dtype=torch.int32)
        else:
            tensors[f"{module}.weight"] = torch.zeros(2, 2)
        tensors[f"{module}.weight_scale"] = torch.ones(2)
    for module in plain:
        tensors[f"{module}.weight"] = torch.zeros(2, 2)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, path / shard)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )
    (path / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "ignore": ignore,
                    "config_groups": groups or {},
                }
            }
        )
    )
    return path


# --- the matcher must be the loader's -----------------------------------------


def test_match_name_is_prefix_regex():
    """vLLM's check_equal_or_regex_match and compressed_tensors' match_name both
    use re.match. A checker with different semantics than the loader is worse
    than no checker."""
    assert match_name("model.layers.0.self_attn.q_a_proj", "re:.*self_attn[.].*")
    assert not match_name("model.layers.0.mlp.gate", "re:.*self_attn[.].*")
    assert match_name("lm_head", "lm_head")
    assert not match_name("model.lm_head", "lm_head")


# --- reading the artifact -----------------------------------------------------


def test_checkpoint_modules_finds_quantized_by_marker(tmp_path: Path):
    ckpt = _checkpoint(
        tmp_path / "c",
        quantized={"a.q_proj": "fp8", "a.experts.0.gate_proj": "int4"},
        plain=["a.indexer.wk"],
        ignore=[],
    )
    modules, quantized = checkpoint_modules(ckpt)
    assert quantized == {"a.q_proj", "a.experts.0.gate_proj"}
    assert "a.indexer.wk" in modules and "a.indexer.wk" not in quantized
    # a weight_packed module has no plain .weight, and must still be a module
    assert "a.experts.0.gate_proj" in modules


def test_single_shard_checkpoint_without_an_index(tmp_path: Path):
    path = tmp_path / "c"
    path.mkdir()
    save_file(
        {"a.q_proj.weight": torch.zeros(2, 2), "a.q_proj.weight_scale": torch.ones(2)},
        path / "model.safetensors",
    )
    (path / "config.json").write_text(json.dumps({"quantization_config": {"ignore": []}}))
    _, quantized = checkpoint_modules(path)
    assert quantized == {"a.q_proj"}


def test_missing_tensors_raises_rather_than_reporting_clean(tmp_path: Path):
    """No index and no shard must not audit as 'nothing shadowed'."""
    path = tmp_path / "empty"
    path.mkdir()
    with pytest.raises(FileNotFoundError):
        checkpoint_modules(path)


# --- the defect ---------------------------------------------------------------


def test_shadowed_by_names_the_quantized_victims():
    quantized = {
        "model.layers.0.self_attn.q_a_proj",
        "model.layers.0.mlp.experts.0.gate_proj",
    }
    assert shadowed_by("re:.*self_attn[.].*", quantized) == [
        "model.layers.0.self_attn.q_a_proj"
    ]
    assert shadowed_by("re:.*mlp[.]gate$", quantized) == []


def test_reproduces_the_measured_glm_shadowing(tmp_path: Path):
    """The real case, at the real shape: the recipe's four broad patterns against
    a checkpoint whose FP8 leg is layers 0/3 attention + layer 3 shared experts +
    layer 0 dense MLP, with layer-3 experts at int4."""
    fp8 = [
        f"model.layers.{layer}.self_attn.{proj}"
        for layer in (0, 3)
        for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
    ] + [
        f"model.layers.3.mlp.shared_experts.{proj}"
        for proj in ("gate_proj", "up_proj", "down_proj")
    ] + [
        f"model.layers.0.mlp.{proj}" for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    int4 = [f"model.layers.3.mlp.experts.{e}.gate_proj" for e in range(4)]
    plain = [
        f"model.layers.{layer}.self_attn.indexer.{proj}"
        for layer in (0, 1, 2)
        for proj in ("wq_b", "wk", "weights_proj")
    ] + ["model.layers.3.mlp.gate", "lm_head"]
    ckpt = _checkpoint(
        tmp_path / "glm",
        quantized={**{m: "fp8" for m in fp8}, **{m: "int4" for m in int4}},
        plain=plain,
        ignore=[
            "lm_head",
            "re:.*mlp[.]gate$",
            "re:.*mlp[.]shared_experts[.].*",
            "re:.*self_attn[.].*",
            "re:.*layers[.][0-2][.].*",
        ],
    )
    report = audit_checkpoint_ignore(ckpt)
    assert not report["ok"]
    # every FP8 module is shadowed; no int4 module is
    assert set(report["shadowed_modules"]) == set(fp8)
    assert not any(m in report["shadowed_modules"] for m in int4)
    assert "re:.*self_attn[.].*" in report["shadowing_patterns"]


def test_a_consistent_checkpoint_audits_clean(tmp_path: Path):
    ckpt = _checkpoint(
        tmp_path / "ok",
        quantized={"model.layers.0.self_attn.q_a_proj": "fp8"},
        plain=["model.layers.0.self_attn.indexer.wk", "lm_head"],
        ignore=["lm_head", "model.layers.0.self_attn.indexer.wk"],
    )
    assert audit_checkpoint_ignore(ckpt)["ok"]
    assert_no_ignore_shadowing(ckpt)  # must not raise


def test_gate_message_names_the_pattern_and_a_victim(tmp_path: Path):
    """A gate that says only 'inconsistent' sends the reader back to a 59k-module
    tree."""
    ckpt = _checkpoint(
        tmp_path / "bad",
        quantized={"model.layers.0.self_attn.q_a_proj": "fp8"},
        plain=[],
        ignore=["re:.*self_attn[.].*"],
    )
    with pytest.raises(RuntimeError) as excinfo:
        assert_no_ignore_shadowing(ckpt)
    message = str(excinfo.value)
    assert "re:.*self_attn[.].*" in message
    assert "model.layers.0.self_attn.q_a_proj" in message


# --- the fix: resolve patterns against the tensors ----------------------------


def test_harmless_pattern_passes_through_unchanged():
    modules = {"a.mlp.gate", "a.q_proj"}
    quantized = {"a.q_proj"}
    entries, report = resolve_ignore_patterns(
        ["re:.*mlp[.]gate$"], modules, quantized
    )
    assert entries == ["re:.*mlp[.]gate$"]
    assert report == {}


def test_shadowing_pattern_is_replaced_by_concrete_unquantized_modules():
    """The coverage a `targets: ["Linear"]` catch-all group needs is preserved --
    the pattern is not merely dropped."""
    modules = {
        "model.layers.0.self_attn.q_a_proj",       # quantized
        "model.layers.0.self_attn.indexer.wk",     # not
        "model.layers.1.self_attn.q_a_proj",       # not
    }
    quantized = {"model.layers.0.self_attn.q_a_proj"}
    entries, report = resolve_ignore_patterns(
        ["re:.*self_attn[.].*"], modules, quantized
    )
    assert "re:.*self_attn[.].*" not in entries
    assert set(entries) == {
        "model.layers.0.self_attn.indexer.wk",
        "model.layers.1.self_attn.q_a_proj",
    }
    assert report["re:.*self_attn[.].*"]["shadowed"] == [
        "model.layers.0.self_attn.q_a_proj"
    ]


def test_resolution_output_actually_audits_clean(tmp_path: Path):
    """End to end: resolve, write, audit. A fix that still leaves the config
    shadowing would pass the two unit tests above and fail the only thing that
    matters."""
    quantized_modules = {"model.layers.0.self_attn.q_a_proj"}
    plain = ["model.layers.0.self_attn.indexer.wk", "model.layers.1.self_attn.q_a_proj"]
    modules = quantized_modules | set(plain)
    entries, _ = resolve_ignore_patterns(
        ["re:.*self_attn[.].*"], modules, quantized_modules
    )
    ckpt = _checkpoint(
        tmp_path / "fixed",
        quantized={m: "fp8" for m in quantized_modules},
        plain=plain,
        ignore=entries,
    )
    assert audit_checkpoint_ignore(ckpt)["ok"]


def test_overflow_drops_the_pattern_and_records_it():
    """A partial-scope smoke's layer-restriction pattern covers tens of thousands
    of unquantized modules. Writing them all into config.json is not useful, so
    the pattern is dropped -- but the caller must be able to tell that happened,
    because the result is not serve-ready."""
    quantized = {"m.0.a"}
    modules = {"m.0.a"} | {f"m.{i}.b" for i in range(50)}
    entries, report = resolve_ignore_patterns(
        ["re:m[.].*"], modules, quantized, max_concrete=10
    )
    assert entries == []
    assert report["re:m[.].*"]["overflow"] is True
    assert report["re:m[.].*"]["shadowed_count"] == 1


def test_exact_module_ignore_that_shadows_is_also_caught():
    """Not only regexes. A concrete entry naming a quantized module is the same
    defect, and would be the shape of a hand-edited config."""
    entries, report = resolve_ignore_patterns(
        ["a.q_proj"], {"a.q_proj"}, {"a.q_proj"}
    )
    assert entries == []
    assert report["a.q_proj"]["shadowed"] == ["a.q_proj"]


def test_int4_modules_are_protected_too():
    """weight_packed is a quant marker as much as weight_scale is; a pattern
    hiding an int4 expert is the same failure."""
    entries, report = resolve_ignore_patterns(
        ["re:.*experts[.].*"],
        {"a.experts.0.gate_proj", "a.mlp.gate"},
        {"a.experts.0.gate_proj"},
    )
    assert entries == []
    assert report["re:.*experts[.].*"]["shadowed_count"] == 1
