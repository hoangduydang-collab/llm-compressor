# MiniMax-M3 GPTQ Discriminator Result Report

## Run

Run root:

`results/m3-quality/20260712-115317-m3-gptq-discriminator`

The compact evidence bundle is approximately 31 MB and contains no model
checkpoint payloads.

Models:

- GPTQ: `artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123`
- AWQ control: `/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4`
- BF16 control: `/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3`

## Completed evidence

CPU validation passed:

```text
54 passed
```

Fresh preflight passed with the corrected task-specific sample sizing and
manifest validation.

GPTQ and AWQ both completed the teacher-forced distributional probe:

- same corpus hash: `f1e6e4a3c7323bf0d43cd0a670adce667b8e1e0cdc7982879298ef41afdb0704`
- 2,047 tokens each
- GPTQ elapsed: 7.39 seconds
- AWQ elapsed: 8.61 seconds

Probe files:

- `models/inhouse_gptq/shards/smoke/distributional_probe.jsonl`
- `models/inhouse_gptq/shards/smoke/distributional_probe.summary.json`
- `models/cyankiwi_awq/shards/smoke/distributional_probe.jsonl`
- `models/cyankiwi_awq/shards/smoke/distributional_probe.summary.json`

The 16-bundle Ray placement diagnostic passed:

- 2 active nodes
- 16 visible GPUs
- placement group ready
- 66.94 seconds elapsed

Evidence is under `ray_placement/`.

## Benchmark result

Both GPTQ and AWQ loaded and ran all five configured task stages far enough to
write partial aggregates and generation-health/sample artifacts. Both then
failed on the same evaluator metric contract:

```text
KeyError: "metric 'acc,none' not in lm-eval results for task 'mmlu_pro';
available: ['sample_len', 'exact_match,custom-extract']"
```

This is an evaluation configuration defect, not a model-quality conclusion.
The partial benchmark outputs must not be treated as complete scores.

Relevant logs:

- `logs/gptq-smoke.out`
- `logs/gptq-smoke.err`
- `logs/awq-smoke.out`
- `logs/awq-smoke.err`

## BF16 result

The bounded BF16 diagnostic was relaunched as Slurm job `12803` after wiring
the validated placement gate into the arm runner's expected
`ray_preflight/gate.json` path. It then failed before model loading in the
distributional probe:

```text
Value error, World size (16) is larger than the number of available GPUs (8)
in this node. If this is intentional and you are using:
- ray, set '--distributed-executor-backend ray'.
```

The BF16 benchmark runner received the Ray backend, but the separate
distributional probe did not and attempted TP=16 on one 8-GPU process. This is
a runner wiring defect, not evidence against BF16 or Ray capacity. Full BF16
logs and Ray runtime artifacts are under:

- `logs/bf16-smoke.out`
- `logs/bf16-smoke.err`
- `models/bf16/shards/smoke/ray_runtime/`

The final return code was `bf16=1`.

## Environment and deviations

- Ray was installed in the shared quant venv because it was absent:
  `ray==2.56.0`, `msgpack==1.2.1`.
- The Ray topology helper was repaired to use routable node IPs, explicit head
  addressing, detached startup, and an observed two-node/16-GPU gate.
- An initial hand-written parallel shell wrapper had incorrect `&` precedence;
  it launched GPTQ but not AWQ/Ray correctly. The commands were relaunched as
  separate explicit `srun` processes. No checkpoint or experiment setting was
  changed.
- The two long-running jobs `12777` and `12778` visible during this run are
  unrelated AWQ re-quantization jobs and were not touched.

## Decision

Do not launch production or start fresh re-quantization from this evidence.
The probe artifacts are suitable for the primary agent's GPTQ-vs-AWQ
distribution comparison. The next code fixes should correct the `mmlu_pro`
metric mapping, pass the distributed executor backend into the BF16
distributional probe, and make the BF16 gate path explicit instead of relying
on a hardcoded `ray_preflight` directory.
