# Execution packet: MiniMax-M3 grouped paired quality rerun

- Protocol version: 1
- State: `READY_FOR_EXECUTOR`
- Packet revision: 2026-07-14-r3
- Planner owner: Codex planner
- Intended executor: cluster executor
- Base Git commit: `eb1a025e`
- Decision question: Does repaired in-house GPTQ preserve enough quality versus
  cyankiwi AWQ to justify later performance evaluation?

This packet supersedes r2. The executor cluster supports `srun`, not `sbatch`.
Run six independent one-node allocations from detached `tmux`; do not translate
this packet into an array or a parent allocation.

## Exact work and reuse

| Arm | Reused checkpoint | Remaining work |
| --- | --- | --- |
| GPTQ `reasoning` | GPQA 100/100 (0.28) | IFEval |
| GPTQ `broad_math` | none | MMLU-Pro, GSM8K, AIME 2025 |
| GPTQ `distributional_probe` | none | 8,192-token probe |
| AWQ `reasoning` | GPQA 100/100 (0.24) | IFEval |
| AWQ `broad_math` | MMLU-Pro 100/100 (0.76), GSM8K 100/100 (0.97) | AIME 2025 |
| AWQ `distributional_probe` | none | 8,192-token probe |

GPTQ MMLU reached 96/100 before timeout but did not checkpoint, so it must rerun.
No production IFEval, AIME, or probe completed. Each `srun` has an independent
`16:00:00` ceiling. Do not retry or start performance work.

## Pull, environment, and workspace

```bash
set -euo pipefail
cd /mnt/nfs/hoangduy/projects/llm-compressor
git fetch origin
git checkout duy-branch
git pull --ff-only origin duy-branch
git merge-base --is-ancestor eb1a025e HEAD
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
test -z "${SLURM_JOB_ID:-}"
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
git ls-files --others --exclude-standard | awk '!/^(results|artifacts)\//' | tee /tmp/m3-r3-workspace-blockers
test ! -s /tmp/m3-r3-workspace-blockers
```

Stop if any check fails. Existing untracked paths under `results/` and
`artifacts/` are record-and-proceed conditions under the repo protocol.

## Fresh run, preflight, and smoke reuse

```bash
set -euo pipefail
MATRIX=pipeline/configs/minimax_m3_paired_gptq_awq_task_isolated_quick.yaml
OLD_ROOT=results/m3-quality/20260714T100300Z-m3-paired-gptq-awq-quick-rerun
SMOKE_ROOT=results/m3-quality/20260714T064000Z-m3-paired-gptq-awq-quick
SMOKE_GATE="$SMOKE_ROOT/smoke_gate.json"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-paired-gptq-awq-grouped-r3"
RUN_ROOT="results/m3-quality/$RUN_ID"
test -f "$SMOKE_GATE"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
date -u +%FT%TZ >"$RUN_ROOT/controller_start_utc.txt"

python -m pipeline.m3_quality_preflight \
  --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/preflight.log"
test "${PIPESTATUS[0]}" -eq 0
cmp "$OLD_ROOT/preflight/production_sample_manifest.json" \
    "$RUN_ROOT/preflight/production_sample_manifest.json"
python - "$SMOKE_ROOT/run_manifest.json" "$RUN_ROOT/run_manifest.json" \
  pipeline/configs/minimax_m3_paired_gptq_awq_quick.yaml "$MATRIX" \
  "$SMOKE_GATE" <<'PYSMOKE' | tee "$RUN_ROOT/smoke_reuse_check.json"
import json, sys, yaml
from pathlib import Path

old_run = json.loads(Path(sys.argv[1]).read_text())
new_run = json.loads(Path(sys.argv[2]).read_text())
old_matrix = yaml.safe_load(Path(sys.argv[3]).read_text())
new_matrix = yaml.safe_load(Path(sys.argv[4]).read_text())
gate = json.loads(Path(sys.argv[5]).read_text())
fields = ("sample_manifest_sha256", "eval_config_sha256", "tokenizer_sha256", "chat_template_sha256")
checks = {field: old_run.get(field) == new_run.get(field) for field in fields}
def contract(matrix):
    keys = ("label", "path", "kind", "nodes", "tensor_parallel_size",
            "pipeline_parallel_size", "distributed_executor_backend")
    return [{key: model.get(key, 1 if key == "pipeline_parallel_size" else None)
             for key in keys} for model in matrix["models"]]
checks["model_and_serving_contract"] = contract(old_matrix) == contract(new_matrix)
checks["prior_gate_ready"] = gate.get("ready_for_production") is True
result = {"reusable": all(checks.values()), "checks": checks}
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["reusable"] else 1)
PYSMOKE
test "${PIPESTATUS[0]}" -eq 0
```

The exact manifest comparison is mandatory. Stop rather than importing prior
tasks if it fails.

## Fail-closed harness contract

This is a deterministic lm-eval paired-quality harness, not a reproduction of
MiniMax's undisclosed full internal evaluation recipe. It uses the official base
tokenizer/chat template, adaptive thinking, standard lm-eval task definitions,
and greedy decoding so AWQ and GPTQ see identical requests. Because this quick
run uses a seeded 100-item subset (and all 30 AIME items), its absolute scores
must not be presented as directly comparable to full public leaderboard scores.
The official model card recommends `temperature=1.0, top_p=0.95` for general
inference, but does not publish an equivalent recipe for these five benchmark
scores: https://huggingface.co/MiniMaxAI/MiniMax-M3

```bash
set -euo pipefail
python - "$RUN_ROOT" <<'PYHARNESS' | tee "$RUN_ROOT/harness_contract_check.json"
import json, sys, yaml
from pathlib import Path

root = Path(sys.argv[1])
cfg = yaml.safe_load((root / "preflight/resolved_eval_config.yaml").read_text())
run = json.loads((root / "run_manifest.json").read_text())
tokenizers = json.loads((root / "preflight/tokenizer_contract.json").read_text())
ev, serve = cfg["eval"], cfg["serve"]
expected_tasks = {
    "gpqa_diamond_zeroshot": ("acc_norm,none", 0),
    "ifeval": ("prompt_level_strict_acc,none", 0),
    "aime25": ("exact_match,none", 0),
    "mmlu_pro": ("exact_match,custom-extract", 5),
    "gsm8k": ("exact_match,strict-match", 5),
}
actual_tasks = {task["name"]: (task["metric"], task["num_fewshot"])
                for task in ev["tasks"] if task.get("limit") is None}
checks = {
    "lm_eval_0_4_12": run.get("lm_eval_version") == "0.4.12",
    "served_tokenizers_match_official_source": tokenizers.get("valid") is True,
    "resolved_tasks": actual_tasks == expected_tasks,
    "chat_template": ev.get("apply_chat_template") is True,
    "fewshot_multiturn": ev.get("fewshot_as_multiturn") is True,
    "adaptive_thinking": ev.get("enable_thinking") is None,
    "think_end_token": ev.get("think_end_token") == "</mm:think>",
    "greedy": ev.get("gen_kwargs", {}).get("temperature") == 0.0
              and ev.get("gen_kwargs", {}).get("do_sample") is False,
    "generation_ceiling": ev.get("gen_kwargs", {}).get("max_gen_toks") == 16384,
    "vllm_backend": ev.get("backend") == "vllm",
    "tp8_ep": serve.get("tensor_parallel_size") == 8
              and serve.get("enable_expert_parallel") is True,
    "serving_shape": serve.get("block_size") == 128
                     and serve.get("kv_cache_dtype") == "fp8"
                     and serve.get("max_model_len") == 65536,
}
result = {
    "valid": all(checks.values()), "checks": checks,
    "comparison_scope": "paired directional quick evaluation",
    "direct_public_score_comparability": False,
    "reason": "seeded 100-item subsets (AIME has 30) and no published identical MiniMax recipe",
}
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["valid"] else 1)
PYHARNESS
test "${PIPESTATUS[0]}" -eq 0
```

## Validate and import four completed task checkpoints

```bash
set -euo pipefail
python - "$OLD_ROOT" "$RUN_ROOT" <<'PYREUSE' | tee "$RUN_ROOT/reused_task_checkpoints.json"
import hashlib, json, math, sys
from pathlib import Path
from pipeline.evalsuite.health import summarize_generation_health
from pipeline.evalsuite.sampling import stable_sample_uid

old, new = map(Path, sys.argv[1:])
mappings = [
    ("inhouse_gptq", "reasoning", "reasoning", "gpqa_diamond_zeroshot", "acc_norm,none", 100, 0.28),
    ("cyankiwi_awq", "reasoning", "reasoning", "gpqa_diamond_zeroshot", "acc_norm,none", 100, 0.24),
    ("cyankiwi_awq", "broad", "broad_math", "mmlu_pro", "exact_match,custom-extract", 100, 0.76),
    ("cyankiwi_awq", "broad", "broad_math", "gsm8k", "exact_match,strict-match", 100, 0.97),
]
old_sample_manifest = old / "preflight" / "production_sample_manifest.json"
new_sample_manifest = new / "preflight" / "production_sample_manifest.json"
new_eval_config = new / "preflight" / "resolved_eval_config.yaml"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
if old_sample_manifest.read_bytes() != new_sample_manifest.read_bytes():
    raise SystemExit("old/new production manifests differ")
selection = json.loads(new_sample_manifest.read_text())["tasks"]
sample_sha = sha(new_sample_manifest)
config_sha = sha(new_eval_config)
old_run = json.loads((old / "run_manifest.json").read_text())
new_run = json.loads((new / "run_manifest.json").read_text())
records = []
targets = {}
for model, old_shard, new_shard, task, metric, expected, expected_score in mappings:
    source = old / "models" / model / "shards" / old_shard
    arm_manifest_path = source / "arm_manifest.json"
    aggregate_path = source / "aggregate.json"
    sample_path = source / "samples" / f"{task}.jsonl"
    if not arm_manifest_path.is_file() or not aggregate_path.is_file() or not sample_path.is_file():
        raise SystemExit(f"missing reusable checkpoint for {model}/{task}")
    arm_manifest = json.loads(arm_manifest_path.read_text())
    required_manifest = {
        "run_id": old_run["run_id"], "git_commit": old_run["git_commit"],
        "model_label": model, "shard": old_shard,
        "sample_manifest_sha256": sample_sha, "eval_config_sha256": config_sha,
        "tokenizer_sha256": new_run["tokenizer_sha256"],
        "chat_template_sha256": new_run["chat_template_sha256"],
    }
    for key, value in required_manifest.items():
        if arm_manifest.get(key) != value:
            raise SystemExit(f"source provenance mismatch {model}/{task}/{key}")
    aggregate = json.loads(aggregate_path.read_text())
    metrics = aggregate.get(task)
    if not isinstance(metrics, dict) or not metrics:
        raise SystemExit(f"missing aggregate task {model}/{task}")
    score = metrics.get(metric)
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)) \
            or abs(float(score) - expected_score) >= 1e-9:
        raise SystemExit(f"unexpected {metric} for {model}/{task}: {score}")
    allowed = {(subtask, int(doc_id)) for subtask, ids in selection[task].items()
               for doc_id in ids}
    unique = {}
    for line in sample_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = row.get("sample_uid")
        if not uid:
            raise SystemExit(f"missing sample_uid for {model}/{task}")
        subtask, doc_id = row.get("subtask"), row.get("doc_id")
        if row.get("task") != task or (subtask, int(doc_id)) not in allowed:
            raise SystemExit(f"sample outside manifest for {model}/{task}/{uid}")
        if uid != stable_sample_uid(task, subtask, doc_id):
            raise SystemExit(f"invalid stable UID for {model}/{task}/{uid}")
        if uid in unique and unique[uid] != row:
            raise SystemExit(f"conflicting duplicate {model}/{task}/{uid}")
        unique[uid] = row
    if len(unique) != expected:
        raise SystemExit(f"expected {expected} unique rows for {model}/{task}, got {len(unique)}")
    target = new / "models" / model / "shards" / new_shard
    (target / "samples").mkdir(parents=True, exist_ok=True)
    (target / "generation_health").mkdir(parents=True, exist_ok=True)
    target_aggregate = targets.setdefault(target, {})
    target_aggregate[task] = aggregate[task]
    rows = [unique[key] for key in sorted(unique)]
    (target / "samples" / f"{task}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    (target / "generation_health" / f"{task}.json").write_text(
        json.dumps(summarize_generation_health(rows), indent=2)
    )
    records.append({"model": model, "task": task, "rows": len(rows), "source": str(source)})
for target, aggregate in targets.items():
    (target / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
print(json.dumps({"validated": True, "imports": records}, indent=2))
PYREUSE
test "${PIPESTATUS[0]}" -eq 0
```

This deliberately deduplicates the old GSM8K evidence from 200 stored rows to
100 identical stable UIDs. Any conflict or missing artifact is a stop condition.

## Dry run, then persistent launch

```bash
set -euo pipefail
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$MATRIX" --run-root "$RUN_ROOT" \
  --smoke-gate "$SMOKE_GATE" --dry-run | tee "$RUN_ROOT/srun_dry_run.log"
test "$(grep -c '^srun ' "$RUN_ROOT/srun_dry_run.log")" -eq 6
test "$(grep -c -- '--time 16:00:00' "$RUN_ROOT/srun_dry_run.log")" -eq 6
! grep -q sbatch "$RUN_ROOT/srun_dry_run.log"

SESSION="m3-quality-$RUN_ID"
tmux new-session -d -s "$SESSION" \
  "cd '$PWD' && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && export PYTHONPATH='$PWD/src:$PWD' && bash pipeline/slurm/run_m3_quality_eval_srun.sh --profile production --matrix '$MATRIX' --run-root '$RUN_ROOT' --smoke-gate '$SMOKE_GATE' >'$RUN_ROOT/controller.log' 2>&1; printf '%s\n' \$? >'$RUN_ROOT/controller.rc'"
printf '%s\n' "$SESSION" >"$RUN_ROOT/tmux_session.txt"
```

Do not run the controller inside another allocation. Do not cancel healthy arms
when a sibling fails.

## Monitor, aggregate, and return

```bash
set -euo pipefail
tmux capture-pane -pt "$SESSION" -S -200
squeue -u "$USER" -o '%.18i %.9T %.20N %.10M %.10l'
tail -n 100 "$RUN_ROOT/controller.log"
```

Wait for `controller.rc`. Then capture exact scheduler accounting from the IDs
recorded by the six arm manifests. If fewer manifests exist, preserve the
controller logs and report the missing allocations rather than inventing IDs:

```bash
set -euo pipefail
python - "$RUN_ROOT" <<'PYJOBS' >"$RUN_ROOT/slurm_job_ids.txt"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ids = sorted({json.loads(path.read_text()).get("slurm_job_id")
              for path in root.glob("models/*/shards/*/arm_manifest.json")})
for job_id in ids:
    if job_id:
        print(job_id)
PYJOBS
while read -r job_id; do
  sacct -j "$job_id" \
    --format=JobIDRaw,JobName%40,State,ExitCode,NodeList,AllocTRES,Submit,Start,End,Elapsed,Timelimit
  scontrol show job "$job_id" -dd
done <"$RUN_ROOT/slurm_job_ids.txt" | tee "$RUN_ROOT/slurm_accounting_final.txt"
```

Aggregate even after partial failure:

```bash
set -uo pipefail
set +e
python -m pipeline.m3_quality_eval aggregate --matrix "$MATRIX" --root "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/aggregate.log"
AGGREGATE_RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$AGGREGATE_RC" >"$RUN_ROOT/aggregate.return_code.txt"
```

Return the full small evidence tree: preflight/manifests, reuse validation,
tokenizer and harness-contract checks, launch plan, dry-run, controller log/rc,
six arm manifests and logs, aggregates,
deduplicated samples, generation health, probes, scheduler evidence, matrix,
gates, report, commands, versions, timings, deviations, and missing artifacts.
For retained large files record absolute path, byte size, and SHA-256. Commit and
push on `duy-branch`, mark this packet `RETURNED_FOR_ANALYSIS`, and stop. No retry,
quality interpretation, model adoption, performance run, or publication is
authorized.
