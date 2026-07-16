"""CPU tests for the MiniMax-M3 quality evaluation matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pipeline.m3_quality_eval import (
    GateThresholds,
    build_exact_sample_manifest,
    build_launch_plan,
    build_profile_sample_manifests,
    evaluate_gates,
    load_matrix,
    project_probe_overhead,
    render_matrix_report,
    resolve_task_aliases,
    validate_and_merge,
    validate_reasoning_config,
    validate_sample_indices,
    validate_smoke_gate,
)
from pipeline.m3_quality_preflight import (
    _metric_filter_keys,
    _representative_task_view,
    _task_config_value,
    build_reasoning_harness_contract,
    inspect_reasoning_task_records,
)

MATRIX = Path("pipeline/configs/minimax_m3_quality_matrix.yaml")
TASK_ISOLATED_MATRIX = Path(
    "pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml"
)

REASONING_R4_MATRIX = Path(
    "pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml"
)

BF16_REASONING_R4_MATRIX = Path(
    "pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml"
)


def test_bf16_reasoning_r4_matrix_uses_tp16_pp1_ray():
    model = load_matrix(BF16_REASONING_R4_MATRIX).models[0]

    assert model.nodes == 2
    assert model.tensor_parallel_size == 16
    assert model.pipeline_parallel_size == 1
    assert model.distributed_executor_backend == "ray"


def test_default_matrix_has_three_active_models_and_autoround_deferred():
    spec = load_matrix(MATRIX)

    assert [model.label for model in spec.models] == [
        "bf16",
        "inhouse_gptq",
        "cyankiwi_awq",
    ]
    assert [model.label for model in spec.deferred_models] == ["aquaman_autoround"]
    assert "OneCompression" in spec.deferred_models[0].reason
    assert [shard.name for shard in spec.shards] == ["reasoning", "broad"]
    assert len(spec.expected_arms) == 6
    assert spec.models[0].nodes == 2
    assert spec.models[0].tensor_parallel_size == 8
    assert spec.models[0].pipeline_parallel_size == 2
    assert all(model.nodes == 1 for model in spec.models[1:])
    assert all(model.tensor_parallel_size == 8 for model in spec.models[1:])
    assert all(model.pipeline_parallel_size == 1 for model in spec.models[1:])
    assert spec.smoke_node_count == 4
    assert spec.production_node_count == 8
    assert spec.probe.total_tokens == 49_152
    assert spec.probe.max_overhead_seconds == 1_800


def test_task_isolated_matrix_groups_tasks_into_six_model_arms(tmp_path):
    spec = load_matrix(TASK_ISOLATED_MATRIX)

    assert [shard.name for shard in spec.shards] == [
        "reasoning",
        "broad_math",
        "distributional_probe",
    ]
    assert [shard.tasks for shard in spec.shards] == [
        ("gpqa_diamond", "ifeval"),
        ("mmlu_pro", "gsm8k", "aime_2025"),
        (),
    ]
    assert [shard.distributional_probe for shard in spec.shards] == [
        False,
        False,
        True,
    ]
    assert spec.sampling["production_samples_per_task"] == 100
    assert spec.scheduling.max_parallel_arms == 6
    assert spec.scheduling.arm_time_limit == "16:00:00"

    gate = tmp_path / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": True}))
    plan = build_launch_plan(spec, profile="production", smoke_gate=gate)
    assert len(plan["arms"]) == 6
    assert plan["total_nodes"] == 6
    assert plan["max_parallel_arms"] == 6
    assert plan["max_concurrent_nodes"] == 6
    assert plan["arm_time_limit"] == "16:00:00"


def test_reasoning_r4_matrix_has_four_arms_and_no_probe(tmp_path):
    spec = load_matrix(REASONING_R4_MATRIX)

    assert [(shard.name, shard.tasks) for shard in spec.shards] == [
        ("gpqa", ("gpqa_diamond",)),
        ("reasoning_suite", ("mmlu_pro", "gsm8k", "aime_2025")),
    ]
    assert spec.task_aliases["gpqa_diamond"] == ("gpqa_diamond_cot_zeroshot",)
    assert spec.probe.enabled is False
    assert spec.sampling["production_samples_per_task"] == 100
    assert spec.scheduling.max_parallel_arms == 4
    assert spec.scheduling.arm_time_limit == "24:00:00"

    gate = tmp_path / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": True}))
    plan = build_launch_plan(spec, profile="production", smoke_gate=gate)
    assert len(plan["arms"]) == 4
    assert plan["total_nodes"] == 4
    assert plan["max_parallel_arms"] == 4
    assert plan["max_concurrent_nodes"] == 4
    assert plan["arm_time_limit"] == "24:00:00"
    assert all(arm["distributional_probe"] is False for arm in plan["arms"])
    assert spec.gates.max_perplexity_increase is None


def test_reasoning_r4_smoke_omits_disabled_probe_budget():
    spec = load_matrix(REASONING_R4_MATRIX)
    report = _passing_smoke_report(spec)
    for evidence in report["models"].values():
        evidence["tasks_scored"] = 4
        evidence.pop("probe")

    result = validate_smoke_gate(spec, report)

    assert result["ready_for_production"] is True
    assert all(
        "probe_budget" not in model["checks"] for model in result["models"].values()
    )


def _write_matrix_variant(tmp_path, **updates):
    raw = yaml.safe_load(
        Path("pipeline/configs/minimax_m3_paired_gptq_awq_quick.yaml").read_text()
    )
    raw.update(updates)
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_matrix_rejects_shard_without_tasks_or_probe(tmp_path):
    path = _write_matrix_variant(
        tmp_path,
        shards=[{"name": "empty", "tasks": [], "distributional_probe": False}],
    )

    with pytest.raises(ValueError, match="must contain tasks or a probe"):
        load_matrix(path)


@pytest.mark.parametrize(
    ("scheduling", "message"),
    [
        ({"max_parallel_arms": 0, "arm_time_limit": "08:00:00"}, "positive"),
        ({"max_parallel_arms": 6, "arm_time_limit": "8 hours"}, "HH:MM:SS"),
    ],
)
def test_matrix_rejects_invalid_scheduling(tmp_path, scheduling, message):
    path = _write_matrix_variant(tmp_path, scheduling=scheduling)

    with pytest.raises(ValueError, match=message):
        load_matrix(path)


def test_probe_projection_enforces_overhead_budget():
    fast = project_probe_overhead(
        smoke_tokens=2048,
        smoke_elapsed_seconds=30.0,
        production_tokens=49_152,
        budget_seconds=1800,
    )
    slow = project_probe_overhead(
        smoke_tokens=2048,
        smoke_elapsed_seconds=120.0,
        production_tokens=49_152,
        budget_seconds=1800,
    )

    assert fast["projected_seconds"] == pytest.approx(720.0)
    assert fast["within_budget"] is True
    assert slow["projected_seconds"] == pytest.approx(2880.0)
    assert slow["within_budget"] is False


def test_task_alias_resolution_is_explicit_and_fails_missing():
    aliases = {
        "gpqa_diamond": ("gpqa_diamond", "leaderboard_gpqa"),
        "ifeval": ("ifeval", "leaderboard_ifeval"),
    }
    assert resolve_task_aliases(aliases, {"leaderboard_gpqa", "ifeval"}) == {
        "gpqa_diamond": "leaderboard_gpqa",
        "ifeval": "ifeval",
    }
    with pytest.raises(ValueError, match="gpqa_diamond"):
        resolve_task_aliases(aliases, {"ifeval"})


def test_reasoning_config_uses_adaptive_minimax_mode_and_strip_token():
    validate_reasoning_config(
        {
            "eval": {
                "enable_thinking": None,
                "think_end_token": "</mm:think>",
                "tasks": [{"name": "gpqa"}, {"name": "aime25"}],
            }
        }
    )


def test_reasoning_config_rejects_enable_thinking_before_gpu_load():
    with pytest.raises(ValueError, match="adaptive mode"):
        validate_reasoning_config(
            {
                "eval": {
                    "enable_thinking": True,
                    "think_end_token": "</mm:think>",
                    "tasks": [{"name": "gpqa"}],
                }
            }
        )


def test_r4_reasoning_config_requires_explicit_thinking_seeds_and_end_token():
    valid = {
        "eval": {
            "enable_thinking": True,
            "think_end_token": "</mm:think>",
            "generation_seeds": [42, 1234, 4158],
        }
    }
    validate_reasoning_config(valid)

    for field, value, message in (
        ("enable_thinking", False, "enable_thinking"),
        ("generation_seeds", [42], "generation_seeds"),
        ("think_end_token", "</think>", "think_end_token"),
    ):
        invalid = {"eval": dict(valid["eval"])}
        invalid["eval"][field] = value
        with pytest.raises(ValueError, match=message):
            validate_reasoning_config(invalid)


def _valid_reasoning_contract_inputs():
    task_details = {
        "gpqa_diamond": (
            "gpqa_diamond_cot_zeroshot",
            "2.2",
            0,
            "exact_match,flexible-extract",
        ),
        "mmlu_pro": ("mmlu_pro", "3.1", 5, "exact_match,custom-extract"),
        "gsm8k": ("gsm8k_cot", "3.0", 8, "exact_match,strict-match"),
        "aime_2025": ("aime25", "1.0", 0, "exact_match,none"),
    }
    records = {}
    for canonical, (installed, version, shots, metric) in task_details.items():
        records[canonical] = {
            "canonical_name": canonical,
            "installed_name": installed,
            "output_type": "generate_until",
            "task_version": version,
            "num_fewshot": shots,
            "metric": metric,
            "available_metric_filters": [metric],
            "representative_prompt_sha256": f"prompt-{canonical}",
            "repeat_prompt_sha256": f"prompt-{canonical}",
            "displayed_choices": ["A", "B", "C", "D"]
            if canonical == "gpqa_diamond"
            else None,
            "repeat_displayed_choices": ["A", "B", "C", "D"]
            if canonical == "gpqa_diamond"
            else None,
            "correct_displayed_label": "B" if canonical == "gpqa_diamond" else None,
            "repeat_correct_displayed_label": "B"
            if canonical == "gpqa_diamond"
            else None,
        }
    return {
        "revision": "0.4.12",
        "task_records": records,
        "generation_seeds": [42, 1234, 4158],
        "gen_kwargs": {
            "temperature": 1.0,
            "top_p": 0.95,
            "do_sample": True,
            "max_gen_toks": 16384,
        },
    }


def test_reasoning_contract_pins_paper_grade_generation_tasks():
    contract = build_reasoning_harness_contract(**_valid_reasoning_contract_inputs())

    assert contract["valid"] is True
    assert contract["lm_eval_version"] == "0.4.12"
    assert contract["tasks"]["gpqa_diamond"]["task_version"] == "2.2"
    assert contract["tasks"]["gpqa_diamond"]["output_type"] == "generate_until"
    assert contract["tasks"]["gsm8k"]["num_fewshot"] == 8
    assert contract["generation"]["generation_seeds"] == [42, 1234, 4158]


def test_representative_prompt_passes_callable_chat_renderer_to_lm_eval():
    class FakeTask:
        eval_docs = [{"question": "Q", "answer": "B"}]

        def set_fewshot_seed(self, seed):
            assert seed == 42

        def fewshot_context(
            self,
            doc,
            num_fewshot,
            *,
            apply_chat_template,
            fewshot_as_multiturn,
            chat_template,
            system_instruction,
        ):
            assert apply_chat_template is True
            assert fewshot_as_multiturn is True
            assert callable(chat_template)
            return chat_template(
                [{"role": "user", "content": doc["question"]}],
                add_generation_prompt=True,
            )

        def doc_to_choice(self, doc):
            return ["A", "B", "C", "D"]

        def doc_to_target(self, doc):
            return doc["answer"]

    class FakeManager:
        def load(self, names):
            return {"tasks": {names[0]: FakeTask()}}

    class FakeTokenizer:
        chat_template = "a Jinja template string is not the renderer"
        name_or_path = "fake"

        def apply_chat_template(self, messages, *, add_generation_prompt=False):
            suffix = "<assistant>" if add_generation_prompt else ""
            return messages[0]["content"] + suffix

    record = _representative_task_view(
        FakeManager(),
        FakeTokenizer(),
        installed_name="gpqa_diamond_cot_zeroshot",
        num_fewshot=0,
    )

    assert record["displayed_choices"] == ["A", "B", "C", "D"]
    assert record["correct_displayed_label"] == "B"


def test_representative_generated_task_uses_processed_gpqa_choices():
    class FakeTask:
        config = {"doc_to_choice": None}
        eval_docs = [
            {
                "Question": "Q",
                "choices": ["first", "second", "third", "fourth"],
                "answer": "(C)",
            }
        ]

        def set_fewshot_seed(self, seed):
            pass

        def fewshot_context(self, doc, num_fewshot, **kwargs):
            return "rendered GPQA prompt"

        def doc_to_choice(self, doc):
            raise TypeError("doc_to_choice was called but not set in config")

        def doc_to_target(self, doc):
            return doc["answer"]

    class FakeManager:
        def load(self, names):
            return {"tasks": {names[0]: FakeTask()}}

    class FakeTokenizer:
        name_or_path = "fake"

        def apply_chat_template(self, messages, **kwargs):
            return "rendered"

    record = _representative_task_view(
        FakeManager(),
        FakeTokenizer(),
        installed_name="gpqa_diamond_cot_zeroshot",
        num_fewshot=0,
    )

    assert record["displayed_choices"] == [
        "first",
        "second",
        "third",
        "fourth",
    ]
    assert record["correct_displayed_label"] == "(C)"


def test_reasoning_inspection_handles_task_group_and_audits_all_leaves():
    class FakeTask:
        OUTPUT_TYPE = "generate_until"
        VERSION = "3.1"

        def __init__(self, task_name, subject):
            self.task_name = task_name
            self.config = {
                "dataset_path": "TIGER-Lab/MMLU-Pro",
                "dataset_name": subject,
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "custom-extract"}],
            }
            self.eval_docs = [{"question": subject, "answer": "A"}]

        def set_fewshot_seed(self, seed):
            pass

        def fewshot_context(self, doc, num_fewshot, **kwargs):
            return f"{self.task_name}:{doc['question']}:{num_fewshot}"

        def doc_to_target(self, doc):
            return doc["answer"]

    class FakeManager:
        def load(self, names):
            assert names == ["mmlu_pro"]
            return {
                "tasks": {
                    "mmlu_pro_math": FakeTask("mmlu_pro_math", "math"),
                    "mmlu_pro_biology": FakeTask("mmlu_pro_biology", "biology"),
                },
                "groups": {"mmlu_pro": object()},
                "group_map": {"mmlu_pro": ["mmlu_pro_math", "mmlu_pro_biology"]},
            }

    class FakeTokenizer:
        name_or_path = "fake"

        def apply_chat_template(self, messages, **kwargs):
            return "rendered"

    records = inspect_reasoning_task_records(
        FakeManager(),
        FakeTokenizer(),
        resolved={"mmlu_pro": "mmlu_pro"},
        configured_tasks=[
            {
                "name": "mmlu_pro",
                "metric": "exact_match,custom-extract",
                "num_fewshot": 5,
            }
        ],
        leaf_sizes={"mmlu_pro": {"mmlu_pro_biology": 10, "mmlu_pro_math": 20}},
    )

    record = records["mmlu_pro"]
    assert record["leaf_tasks"] == ["mmlu_pro_biology", "mmlu_pro_math"]
    assert record["representative_leaf"] == "mmlu_pro_biology"
    assert record["output_type"] == "generate_until"
    assert record["task_version"] == "3.1"
    assert record["available_metric_filters"] == ["exact_match,custom-extract"]
    assert record["available_samples"] == 30


def test_task_config_dict_subclass_reads_dataclass_fields():
    class FakeTaskConfig(dict):
        dataset_path = "Idavidrein/gpqa"
        metric_list = [{"metric": "exact_match"}]
        filter_list = [
            {"name": "strict-match"},
            {"name": "flexible-extract"},
        ]

    class FakeTask:
        config = FakeTaskConfig()

    task = FakeTask()

    assert _task_config_value(task, "dataset_path") == "Idavidrein/gpqa"
    assert _metric_filter_keys(task) == [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda values: values.update(revision="0.4.11"), "revision"),
        (
            lambda values: values["task_records"]["gpqa_diamond"].update(
                output_type="multiple_choice"
            ),
            "output_type",
        ),
        (lambda values: values.update(generation_seeds=[42]), "generation_seeds"),
        (
            lambda values: values["task_records"]["gpqa_diamond"].update(
                repeat_prompt_sha256="different"
            ),
            "prompt",
        ),
    ],
)
def test_reasoning_contract_rejects_harness_drift(mutation, message):
    values = _valid_reasoning_contract_inputs()
    mutation(values)

    with pytest.raises(ValueError, match=message):
        build_reasoning_harness_contract(**values)


def test_build_exact_sample_manifest_writes_stratified_hash(tmp_path):
    output = tmp_path / "sample_manifest.json"
    manifest = build_exact_sample_manifest(
        task_name="mmlu_pro",
        leaf_sizes={"mmlu_pro_math": 100, "mmlu_pro_history": 50},
        total=30,
        seed=42,
        output=output,
        harness_revision="lm-eval-test",
    )

    assert output.is_file()
    assert manifest["sha256"]
    assert sum(len(indices) for indices in manifest["tasks"]["mmlu_pro"].values()) == 30
    assert json.loads(output.read_text()) == manifest


def _passing_smoke_report(spec):
    return {
        "schema_version": 1,
        "profile": "smoke",
        "sample_manifest_sha256": "samples",
        "models": {
            model.label: {
                "infrastructure_ok": True,
                "tasks_scored": 5,
                "sample_manifest_sha256": "samples",
                "empty_count": 0,
                "periodic_loop_count": 0,
                "artifacts_valid": True,
                "distributed_world_size": model.tensor_parallel_size
                * model.pipeline_parallel_size,
                "probe": {"tokens": 2048, "elapsed_seconds": 30.0},
            }
            for model in spec.models
        },
    }


def test_smoke_gate_requires_every_model_and_projects_probe_budget():
    spec = load_matrix(MATRIX)
    result = validate_smoke_gate(spec, _passing_smoke_report(spec))

    assert result["ready_for_production"] is True
    assert result["models"]["bf16"]["probe_projection"]["within_budget"] is True
    assert result["models"]["bf16"]["distributed_world_size"] == 16


def test_smoke_gate_reports_zero_probe_evidence_without_raising():
    spec = load_matrix(MATRIX)
    report = _passing_smoke_report(spec)
    report["models"]["inhouse_gptq"]["probe"] = {
        "tokens": 0,
        "elapsed_seconds": 0,
    }

    result = validate_smoke_gate(spec, report)

    projection = result["models"]["inhouse_gptq"]["probe_projection"]
    assert result["ready_for_production"] is False
    assert projection["within_budget"] is False
    assert projection["reason"] == "missing positive smoke probe timing"


def test_smoke_gate_rejects_missing_or_looping_model():
    spec = load_matrix(MATRIX)
    report = _passing_smoke_report(spec)
    report["models"].pop("cyankiwi_awq")
    report["models"]["inhouse_gptq"]["periodic_loop_count"] = 1

    result = validate_smoke_gate(spec, report)

    assert result["ready_for_production"] is False
    assert "cyankiwi_awq" in result["missing_models"]
    assert result["models"]["inhouse_gptq"]["passed"] is False


def test_reasoning_r4_smoke_allows_one_empty_but_production_gate_stays_strict():
    spec = load_matrix(
        Path("pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml")
    )
    report = _passing_smoke_report(spec)
    for evidence in report["models"].values():
        evidence["tasks_scored"] = sum(len(shard.tasks) for shard in spec.shards)
    report["models"]["inhouse_gptq"]["empty_count"] = 1

    result = validate_smoke_gate(spec, report)

    assert result["ready_for_production"] is True
    gptq = result["models"]["inhouse_gptq"]
    assert gptq["checks"]["empty_outputs"] is True
    assert gptq["empty_output_count"] == 1
    assert gptq["max_smoke_empty_outputs"] == 1
    assert gptq["warnings"] == ["smoke observed 1 empty generation(s)"]
    assert spec.gates.max_degeneration_failures == 0

    report["models"]["inhouse_gptq"]["empty_count"] = 2
    result = validate_smoke_gate(spec, report)
    assert result["ready_for_production"] is False


def _write_arm(
    root: Path,
    *,
    model: str,
    shard: str,
    sample_sha: str,
    task: str | None,
    sample_uid: str | None,
    complete: bool = True,
) -> None:
    arm = root / "models" / model / "shards" / shard
    (arm / "samples").mkdir(parents=True)
    (arm / "arm_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run",
                "model_label": model,
                "shard": shard,
                "sample_manifest_sha256": sample_sha,
                "eval_config_sha256": "eval",
                "tokenizer_sha256": "tokenizer",
                "chat_template_sha256": "chat",
                "git_commit": "commit",
            }
        ),
        encoding="utf-8",
    )
    aggregate = {task: {"acc,none": 1.0}} if task is not None else {}
    (arm / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
    if task is not None:
        (arm / "samples" / f"{task}.jsonl").write_text(
            json.dumps(
                {
                    "sample_uid": sample_uid,
                    "task": task,
                    "subtask": task,
                    "correct": 1,
                    "metric_value": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        (arm / "distributional_probe.jsonl").write_text(
            json.dumps(
                {
                    "corpus_sha256": "corpus",
                    "prompt_id": "prompt",
                    "length_bucket": "short",
                    "prompt_token_count": 2,
                    "position": 1,
                    "observed_token_id": 7,
                    "observed_logprob": -1.0,
                    "top_logprobs": [{"token_id": 7, "logprob": -1.0, "rank": 1}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (arm / "return_code.txt").write_text("0\n", encoding="utf-8")
    (arm / "arm_complete.json").write_text(
        json.dumps({"complete": complete}), encoding="utf-8"
    )


def _write_run_manifest(
    root: Path, *, sample_sha="samples", expected_arms=None
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run",
                "git_commit": "commit",
                "sample_manifest_sha256": sample_sha,
                "eval_config_sha256": "eval",
                "tokenizer_sha256": "tokenizer",
                "chat_template_sha256": "chat",
                "expected_arms": expected_arms
                or [
                    {"model_label": "bf16", "shard": "reasoning"},
                    {"model_label": "quant", "shard": "reasoning"},
                ],
                "models": ["bf16", "quant"],
                "baseline_label": "bf16",
            }
        ),
        encoding="utf-8",
    )


def _write_scientific_run_contract(root: Path, **overrides) -> None:
    contract = {
        "schema_version": 1,
        "lm_eval_version": "0.4.12",
        "harness_contract_sha256": "harness",
        "sample_manifest_sha256": "samples",
        "eval_config_sha256": "eval",
        "tokenizer_sha256": "tokenizer",
        "chat_template_sha256": "chat",
        "rendered_prompt_sha256": "prompt",
        "generation_seeds": [42, 1234, 4158],
        "expected_question_counts": {
            "gpqa_diamond_cot_zeroshot": 100,
            "mmlu_pro": 100,
            "gsm8k_cot": 100,
            "aime25": 30,
        },
        "resolved_tasks": {
            "gpqa_diamond": "gpqa_diamond_cot_zeroshot",
            "mmlu_pro": "mmlu_pro",
            "gsm8k": "gsm8k_cot",
            "aime_2025": "aime25",
        },
    }
    contract.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_manifest.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )


def test_run_contract_gate_accepts_comparable_independent_roots(tmp_path):
    from pipeline import m3_quality_eval as quality

    reference = tmp_path / "gptq-awq"
    candidate = tmp_path / "bf16"
    _write_scientific_run_contract(reference)
    _write_scientific_run_contract(candidate)

    report = quality.compare_run_contracts(reference, candidate)

    assert report["valid"] is True
    assert report["mismatches"] == {}
    assert report["matched_fields"] == list(quality.RUN_CONTRACT_FIELDS)


def test_run_contract_gate_reports_mismatch_and_cli_fails(tmp_path):
    from pipeline import m3_quality_eval as quality

    reference = tmp_path / "gptq-awq"
    candidate = tmp_path / "bf16"
    report_path = tmp_path / "contract_gate.json"
    _write_scientific_run_contract(reference)
    _write_scientific_run_contract(candidate, generation_seeds=[42])

    rc = quality.main(
        [
            "contract-gate",
            "--reference-root",
            str(reference),
            "--candidate-root",
            str(candidate),
            "--out",
            str(report_path),
        ]
    )

    assert rc == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert report["mismatches"] == {
        "generation_seeds": {
            "reference": [42, 1234, 4158],
            "candidate": [42],
        }
    }


def test_merge_rejects_sample_manifest_mismatch(tmp_path):
    _write_run_manifest(tmp_path)
    _write_arm(
        tmp_path,
        model="bf16",
        shard="reasoning",
        sample_sha="samples",
        task="gpqa_diamond",
        sample_uid="a",
    )
    _write_arm(
        tmp_path,
        model="quant",
        shard="reasoning",
        sample_sha="different",
        task="gpqa_diamond",
        sample_uid="a",
    )

    with pytest.raises(ValueError, match="sample manifest"):
        validate_and_merge(tmp_path)


def test_merge_builds_pairwise_self_consistent_comparison(tmp_path):
    _write_run_manifest(tmp_path)
    for model in ("bf16", "quant"):
        _write_arm(
            tmp_path,
            model=model,
            shard="reasoning",
            sample_sha="samples",
            task="gpqa_diamond",
            sample_uid="a",
        )

    result = validate_and_merge(tmp_path)

    comparison = result["comparisons"]["quant"]
    assert comparison["tasks"]["gpqa_diamond"]["flip_rate"] == 0.0
    assert comparison["tasks"]["gpqa_diamond"]["delta"] == 0.0
    assert (tmp_path / "matrix.json").is_file()
    assert (tmp_path / "merged" / "quant" / "samples" / "gpqa_diamond.jsonl").is_file()


def _write_repeated_arm(root: Path, model: str, *, missing_last=False) -> None:
    arm = root / "models" / model / "shards" / "gpqa"
    (arm / "samples").mkdir(parents=True)
    manifest = json.loads((root / "run_manifest.json").read_text())
    (arm / "arm_manifest.json").write_text(
        json.dumps(
            {
                **{
                    field: manifest[field]
                    for field in (
                        "run_id",
                        "git_commit",
                        "sample_manifest_sha256",
                        "eval_config_sha256",
                        "tokenizer_sha256",
                        "chat_template_sha256",
                        "harness_contract_sha256",
                    )
                },
                "model_label": model,
                "shard": "gpqa",
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "sample_uid": question,
            "attempt_uid": f"{question}:{seed}",
            "generation_seed": seed,
            "correct": 1,
        }
        for question in ("q1", "q2")
        for seed in (42, 1234, 4158)
    ]
    if missing_last:
        rows.pop()
    (arm / "samples" / "gpqa_diamond_cot_zeroshot.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (arm / "aggregate.json").write_text(
        json.dumps({"gpqa_diamond_cot_zeroshot": {"exact_match": 1.0}}),
        encoding="utf-8",
    )
    health = arm / "generation_health"
    health.mkdir()
    (health / "gpqa_diamond_cot_zeroshot.json").write_text(
        json.dumps(
            {
                "missing_count": 0,
                "empty_count": 0,
                "answer_extraction_failure_count": 0,
                "length_cap_hit_count": 0,
                "periodic_loop_count": 0,
                "nonfinite_metric_count": 0,
                "reasoning_failure_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (arm / "return_code.txt").write_text("0\n", encoding="utf-8")
    (arm / "arm_complete.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )


def _prepare_repeated_run(root: Path) -> None:
    _write_run_manifest(
        root,
        expected_arms=[
            {"model_label": model, "shard": "gpqa"} for model in ("bf16", "quant")
        ],
    )
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        harness_contract_sha256="harness",
        generation_seeds=[42, 1234, 4158],
        expected_question_counts={"gpqa_diamond_cot_zeroshot": 2},
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_merge_preserves_three_attempts_per_question(tmp_path):
    _prepare_repeated_run(tmp_path)
    for model in ("bf16", "quant"):
        _write_repeated_arm(tmp_path, model)

    result = validate_and_merge(tmp_path)
    sample_file = tmp_path / "merged/quant/samples/gpqa_diamond_cot_zeroshot.jsonl"
    rows = [json.loads(line) for line in sample_file.read_text().splitlines()]

    assert result["infrastructure_ok"] is True
    assert len(rows) == 6
    assert len({row["sample_uid"] for row in rows}) == 2
    assert len({row["attempt_uid"] for row in rows}) == 6


def test_merge_rejects_incomplete_repeated_seed_grid(tmp_path):
    _prepare_repeated_run(tmp_path)
    _write_repeated_arm(tmp_path, "bf16")
    _write_repeated_arm(tmp_path, "quant", missing_last=True)

    with pytest.raises(ValueError, match="seed grid"):
        validate_and_merge(tmp_path)


def test_merge_rejects_false_completion_marker(tmp_path):
    _write_run_manifest(tmp_path)
    _write_arm(
        tmp_path,
        model="bf16",
        shard="reasoning",
        sample_sha="samples",
        task="gpqa_diamond",
        sample_uid="a",
    )
    _write_arm(
        tmp_path,
        model="quant",
        shard="reasoning",
        sample_sha="samples",
        task="gpqa_diamond",
        sample_uid="a",
        complete=False,
    )

    result = validate_and_merge(tmp_path)

    assert result["infrastructure_ok"] is False
    assert result["failures"] == [{"arm": "quant/reasoning", "arm_complete": False}]


def test_merge_accepts_one_probe_only_arm_per_model(tmp_path):
    expected = [
        {"model_label": model, "shard": shard}
        for model in ("bf16", "quant")
        for shard in ("gpqa_diamond", "distributional_probe")
    ]
    _write_run_manifest(tmp_path, expected_arms=expected)
    for model in ("bf16", "quant"):
        _write_arm(
            tmp_path,
            model=model,
            shard="gpqa_diamond",
            sample_sha="samples",
            task="gpqa_diamond",
            sample_uid="a",
        )
        _write_arm(
            tmp_path,
            model=model,
            shard="distributional_probe",
            sample_sha="samples",
            task=None,
            sample_uid=None,
        )

    result = validate_and_merge(tmp_path)

    assert result["infrastructure_ok"] is True
    assert "gpqa_diamond" in result["comparisons"]["quant"]["tasks"]
    assert result["comparisons"]["quant"]["distributional"]["perplexity_ratio"] == 1.0
    for model in ("bf16", "quant"):
        assert (tmp_path / "merged" / model / "distributional_probe.jsonl").is_file()


def test_quality_failure_is_distinct_from_infrastructure_failure():
    matrix = {
        "infrastructure_ok": True,
        "comparisons": {
            "quant": {
                "tasks": {
                    "gpqa_diamond": {
                        "delta": -0.03,
                        "score_recovery_ratio": 0.96,
                        "regressions_a_correct_b_wrong": 3,
                        "both_correct": 97,
                    }
                },
                "distributional": {"perplexity_ratio": 1.02},
                "generation_health": {"degeneration_failures": 0},
            }
        },
    }

    gates = evaluate_gates(
        matrix,
        GateThresholds(
            max_task_drop=0.02,
            min_macro_recovery=0.98,
            max_conditional_regression=0.05,
            max_perplexity_increase=0.10,
            max_degeneration_failures=0,
        ),
    )

    assert gates["infrastructure_ok"] is True
    assert gates["quality_ok"] is False
    assert gates["models"]["quant"]["max_task_drop"]["passed"] is False


def test_r4_complete_empty_response_is_health_advisory_not_quality_failure():
    matrix = {
        "infrastructure_ok": True,
        "comparisons": {
            "quant": {
                "tasks": {
                    "gpqa": {
                        "n_paired": 300,
                        "delta": 0.0,
                        "score_recovery_ratio": 1.0,
                        "regressions_a_correct_b_wrong": 0,
                        "both_correct": 100,
                    }
                },
                "generation_health": {
                    "baseline": {
                        "tasks": {
                            "gpqa": {
                                "samples": 300,
                                "empty_count": 0,
                                "empty_rate": 0.0,
                            }
                        },
                        "degeneration_failures": 0,
                    },
                    "candidate": {
                        "tasks": {
                            "gpqa": {
                                "samples": 300,
                                "empty_count": 1,
                                "empty_rate": 1 / 300,
                            }
                        },
                        "degeneration_failures": 1,
                    },
                    "degeneration_failures": 1,
                },
            }
        },
    }

    gates = evaluate_gates(matrix, GateThresholds(0.02, 0.98, 0.05, None, 0))

    assert gates["infrastructure_ok"] is True
    assert gates["quality_ok"] is True
    assert "degeneration_failures" not in gates["models"]["quant"]
    advisory = gates["health_advisory"]
    assert advisory["has_findings"] is True
    assert advisory["models"]["quant"]["combined_degeneration_failures"] == 1
    assert advisory["models"]["quant"]["candidate"]["tasks"]["gpqa"][
        "empty_count"
    ] == 1


def test_bf16_only_matrix_preserves_merged_health_in_advisory(tmp_path):
    spec = load_matrix(BF16_REASONING_R4_MATRIX)
    _write_run_manifest(
        tmp_path,
        expected_arms=[
            {"model_label": model, "shard": shard}
            for model, shard in spec.expected_arms
        ],
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["models"] = [model.label for model in spec.models]
    manifest["baseline_label"] = spec.baseline_label
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    for shard in spec.shards:
        task = shard.tasks[0]
        _write_arm(
            tmp_path,
            model="bf16",
            shard=shard.name,
            sample_sha="samples",
            task=task,
            sample_uid=f"{shard.name}-sample",
        )
        health_dir = (
            tmp_path
            / "models"
            / "bf16"
            / "shards"
            / shard.name
            / "generation_health"
        )
        health_dir.mkdir()
        health_dir.joinpath(f"{task}.json").write_text(
            json.dumps(
                {
                    "samples": 100,
                    "empty_count": 1 if task == "gpqa_diamond" else 0,
                    "reasoning_failure_count": (
                        1 if task == "gpqa_diamond" else 0
                    ),
                    "nonfinite_metric_count": 0,
                }
            ),
            encoding="utf-8",
        )

    matrix = validate_and_merge(tmp_path)
    gates = evaluate_gates(matrix, spec.gates)

    assert matrix["comparisons"] == {}
    assert matrix["generation_health"]["bf16"]["tasks"]["gpqa_diamond"][
        "empty_count"
    ] == 1
    advisory = gates["health_advisory"]
    assert advisory["has_findings"] is True
    assert advisory["models"]["bf16"]["has_findings"] is True
    assert advisory["models"]["bf16"]["baseline"]["tasks"]["gpqa_diamond"][
        "reasoning_failure_count"
    ] == 1


def test_matrix_report_surfaces_quantization_metrics_and_gates():
    matrix = {
        "baseline_label": "bf16",
        "comparisons": {
            "gptq": {
                "tasks": {
                    "gpqa": {
                        "delta": -0.01,
                        "flip_rate": 0.03,
                        "conditional_regression_rate": 0.02,
                        "score_recovery_ratio": 0.99,
                    }
                },
                "distributional": {"perplexity_ratio": 1.04},
                "generation_health": {"degeneration_failures": 1},
            }
        },
    }
    gates = {
        "quality_ok": True,
        "models": {"gptq": {"quality_ok": True}},
        "health_advisory": {"has_findings": True},
    }

    report = render_matrix_report(matrix, gates)

    assert "MiniMax-M3 Quality Matrix" in report
    assert "gptq" in report
    assert "flip rate" in report.lower()
    assert "conditional regression" in report.lower()
    assert "perplexity ratio" in report.lower()
    assert "Health advisory: findings present" in report
    assert "PASS" in report


def test_launch_plan_parallelizes_three_smoke_then_six_production_arms(tmp_path):
    spec = load_matrix(MATRIX)
    smoke = build_launch_plan(spec, profile="smoke")
    assert len(smoke["arms"]) == 3
    assert sum(arm["nodes"] for arm in smoke["arms"]) == 4
    assert all(len(arm["tasks"]) == 5 for arm in smoke["arms"])

    gate = tmp_path / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": True}))
    production = build_launch_plan(spec, profile="production", smoke_gate=gate)
    assert len(production["arms"]) == 6
    assert sum(arm["nodes"] for arm in production["arms"]) == 8
    assert production["max_parallel_arms"] == 6


def test_production_launch_plan_refuses_failed_smoke_gate(tmp_path):
    spec = load_matrix(MATRIX)
    gate = tmp_path / "smoke_gate.json"
    gate.write_text(json.dumps({"ready_for_production": False}))
    with pytest.raises(ValueError, match="smoke gate"):
        build_launch_plan(spec, profile="production", smoke_gate=gate)


def test_profile_manifests_make_tiny_smoke_and_mmlu_only_production(tmp_path):
    leaf_sizes = {
        "gpqa_diamond": {"gpqa_diamond": 100},
        "mmlu_pro": {"mmlu_math": 1000, "mmlu_history": 499, "mmlu_law": 1},
    }
    outputs = build_profile_sample_manifests(
        resolved_tasks={"gpqa_diamond": "gpqa_diamond", "mmlu_pro": "mmlu_pro"},
        leaf_sizes=leaf_sizes,
        mmlu_task="mmlu_pro",
        mmlu_total=1200,
        seed=42,
        output_dir=tmp_path,
        harness_revision="test",
    )
    smoke = json.loads(outputs["smoke"].read_text())
    production = json.loads(outputs["production"].read_text())
    assert sum(len(v) for v in smoke["tasks"]["gpqa_diamond"].values()) == 2
    mmlu_smoke = smoke["tasks"]["mmlu_pro"]
    assert sum(len(v) for v in mmlu_smoke.values()) == len(mmlu_smoke)
    assert all(len(indices) == 1 for indices in mmlu_smoke.values())
    assert set(production["tasks"]) == {"mmlu_pro"}
    assert sum(len(v) for v in production["tasks"]["mmlu_pro"].values()) == 1200
    assert smoke["sha256"] and production["sha256"]


def test_preflight_inspects_loaded_leaf_evaluation_splits():
    from types import SimpleNamespace

    from pipeline.m3_quality_preflight import inspect_leaf_sizes

    task_a = SimpleNamespace(
        config=SimpleNamespace(test_split="test"),
        dataset={"test": range(70)},
        eval_docs=range(7),
    )
    task_b = SimpleNamespace(
        config=SimpleNamespace(validation_split="validation"),
        dataset={"validation": range(30)},
        eval_docs=range(3),
    )
    manager = SimpleNamespace(
        load=lambda names: {"tasks": {"leaf_a": task_a, "leaf_b": task_b}}
    )
    assert inspect_leaf_sizes(manager, "group") == {"leaf_a": 7, "leaf_b": 3}


def test_tokenizer_contract_requires_every_served_model_to_match_reference():
    from pipeline.m3_quality_preflight import compare_tokenizer_contracts

    reference = {
        "tokenizer_sha256": "tokenizer",
        "chat_template_sha256": "chat",
        "rendered_prompt_sha256": "prompt",
    }
    report = compare_tokenizer_contracts(
        reference,
        {
            "awq": dict(reference),
            "gptq": {**reference, "chat_template_sha256": "wrong"},
        },
    )

    assert report["valid"] is False
    assert report["models"]["awq"]["matches_reference"] is True
    assert report["models"]["gptq"]["mismatches"] == ["chat_template_sha256"]


def test_sample_index_validation_reports_resolved_leaf_bounds():
    tasks = {"mmlu_pro": {"mmlu_pro_biology": [0, 716, 6382]}}
    sizes = {"mmlu_pro": {"mmlu_pro_biology": 717}}

    with pytest.raises(
        ValueError,
        match=(
            r"task=mmlu_pro leaf=mmlu_pro_biology size=717 "
            r"max_selected_index=6382"
        ),
    ):
        validate_sample_indices(tasks, sizes)


def test_static_serving_abi_gate_collects_all_model_reports_before_failure(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from pipeline import m3_quality_preflight as preflight

    models = [
        SimpleNamespace(label="broken", path=tmp_path / "broken"),
        SimpleNamespace(label="control", path=tmp_path / "control"),
    ]
    reports = {
        "broken": {"valid": False, "errors": [{"code": "namespace_miss"}]},
        "control": {"valid": True, "errors": []},
    }
    monkeypatch.setattr(
        preflight,
        "inspect_checkpoint_serving_abi",
        lambda path: reports[path.name],
    )

    failures = preflight.inspect_all_serving_abis(models, tmp_path / "reports")

    assert [failure["label"] for failure in failures] == ["broken"]
    assert (tmp_path / "reports" / "broken.json").is_file()
    assert (tmp_path / "reports" / "control.json").is_file()
    with pytest.raises(ValueError, match="broken.*namespace_miss"):
        preflight.require_all_serving_abis(failures)


def test_static_serving_abi_gate_rejects_before_runtime():
    from pipeline.m3_quality_preflight import require_valid_serving_abi

    report = {
        "valid": False,
        "errors": [
            {
                "code": "plain_runtime_module_not_ignored",
                "module": (
                    "language_model.model.layers.3.block_sparse_moe."
                    "shared_experts.gate_up_proj"
                ),
            }
        ],
    }

    with pytest.raises(ValueError, match="static serving ABI.*shared_experts"):
        require_valid_serving_abi("inhouse_awq", report)


def test_minimax_mmlu_metric_matches_installed_generation_task():
    import yaml

    config = yaml.safe_load(
        Path("pipeline/configs/eval_minimax_m3_quality.yaml").read_text()
    )
    mmlu = next(task for task in config["eval"]["tasks"] if task["name"] == "mmlu_pro")
    assert mmlu["metric"] == "exact_match,custom-extract"
