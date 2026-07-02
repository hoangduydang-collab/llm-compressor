# Quantization Pipeline (M1 + M2)

A config-driven, one-command pipeline that turns any HF (MoE) checkpoint into a
vLLM-servable **W4AFP8 / W4A8** artifact on Hopper, with a serve-handoff check,
an accuracy gate, and versioned outputs.

```
pipeline/
  config.py          # YAML schema (dataclasses) + loader
  recipe.py          # (method, scheme) -> llm-compressor modifiers
  calibration.py     # build the calibration dataset
  quantize.py        # load -> oneshot -> sanity-gen -> save pack-quantized
  serve_verify.py    # boot vLLM on the artifact, confirm load + sane output
  eval_gate.py       # lm-eval (Wikitext PPL + MMLU) vs baseline -> pass/fail
  evalsuite/         # full static + agentic eval + quant-vs-original compare
  versioning.py      # timestamped artifact dirs + reproducible metadata
  run.py             # CLI tying the stages together
  sweep.py           # method x format comparison matrix (M2 Phase 3)
  m3_enablement.py   # MiniMax-M3 MoE linearization probe (Stage 3)
  configs/           # ready-to-run configs per stage
```

## Install (on the 8xH100 cluster)

A single environment handles quantize + serve + eval (vLLM 0.24.0 stable serves
W4AFP8 MoE correctly — see the MoE-gate note below). Using `uv` + a project venv:

```bash
source /mnt/nfs/hoangduy/env.sh                       # sets $UV, caches, WORK_ROOT
"$UV" venv --python 3.12 /mnt/nfs/hoangduy/venvs/quant
source /mnt/nfs/hoangduy/venvs/quant/bin/activate     # AFTER env.sh so it wins on PATH
"$UV" pip install -e .
"$UV" pip install -r pipeline/requirements.txt        # vllm + lm-eval
```

Per-session afterwards: `source env.sh` then `source venvs/quant/bin/activate`
(order matters), and `export HOME=/mnt/nfs/hoangduy` (the cluster HOME is not
writable; `pipeline._env` also redirects FlashInfer/HOME caches at runtime).
GPU runs go through Slurm (see `pipeline/slurm/`).

## One command

```bash
python -m pipeline.run --config pipeline/configs/qwen3_30b_a3b.yaml
```

Runs all stages: **quantize -> serve-verify -> eval-gate**, writing everything
to `artifacts/<run_slug>/<timestamp>/`:

```
checkpoint/        # vLLM-loadable compressed-tensors model
config.yaml        # exact resolved config for this run
recipe.json        # method/scheme/ignore summary
metadata.json      # git SHA, package versions, GPU SM, timing
serve_report.json  # vLLM load + sample output
eval_report.json   # gate metrics + pass/fail
```

Run a single stage with `--stage quantize|serve|eval`, reuse a checkpoint with
`--checkpoint <dir>`, and override any field with `--set a.b.c=value`.

## Evaluation suite (static + agentic + quant-vs-original)

The `pipeline/evalsuite/` package runs the full static lm-eval suite with
per-sample logging, optional tau2 agentic eval (reuses
`benchmarks/llm-perf-benchmarks/performance/calibration/run_calibration.sh`),
and a post-hoc quantized-vs-original comparison with flip-rate, Cohen's kappa,
and McNemar statistics.

When `eval.log_samples: true` (default), `pipeline.run --stage eval` writes:

```
artifacts/<run_slug>/<ts>/evalsuite/
  aggregate.json           # task-level metrics
  samples/<task>.jsonl     # per-doc correctness (for flip-rate)
  agentic_samples.jsonl    # if agentic.enabled
  eval_meta.json
```

### Standalone usage

```bash
# 1. Evaluate ORIGINAL model
python -m pipeline.evalsuite.cli run \
  --config pipeline/configs/eval_full.yaml \
  --model <original-hf-id-or-path> \
  --out evals/original

# 2. Evaluate QUANTIZED checkpoint (same config)
python -m pipeline.evalsuite.cli run \
  --config pipeline/configs/eval_full.yaml \
  --model artifacts/<slug>/<ts>/checkpoint \
  --out evals/quant

# 3. Post-hoc comparison (no GPU)
python -m pipeline.evalsuite.cli compare \
  --a evals/original --b evals/quant --out evals/compare \
  --config pipeline/configs/eval_full.yaml
# -> evals/compare/compare.json + report.md
```

### SGLang backend (native w4afp8 checkpoints)

Checkpoints with `quant_method: w4afp8` in `config.json` (e.g.
`PhalaCloud/GLM-5.2-W4AFP8`) are SGLang-native and cannot load in vLLM.
Use `eval.backend: sglang` and install SGLang (>= 0.5.13.post1 for GLM-5.2).

```bash
python -m pipeline.evalsuite.cli run \
  --config pipeline/configs/eval_glm52_w4afp8_sglang_h100.yaml \
  --out evals/glm52-w4afp8-phala
```

On **8× H100 80GB**, do not use Phala's 1M `context_length` / large KV pool for
eval. Cap KV with `serve.max_model_len` (→ SGLang `context_length`),
`serve.gpu_memory_utilization` (→ `mem_fraction_static`), and
`serve.sglang_kwargs.max_total_tokens`. Static lm-eval tasks fit in 2k context;
`max_total_tokens=8192` and `max_running_requests=1` are enough. If CUDA graph
capture still OOMs, use `eval_glm52_w4afp8_sglang_safe.yaml` (graphs off).
On **8× H200**, use `eval_glm52_w4afp8_sglang_h200.yaml` for Phala-like perf.

Or override an existing config:

```bash
python -m pipeline.evalsuite.cli run \
  --config pipeline/configs/eval_full.yaml \
  --model /path/to/GLM-5.2-W4AFP8 \
  --out evals/glm52-w4afp8-phala \
  --set eval.backend=sglang \
  --set model.trust_remote_code=true \
  --set serve.tensor_parallel_size=8 \
  --set serve.sglang_kwargs.quantization=w4afp8 \
  --set serve.sglang_kwargs.disable_shared_experts_fusion=true
```

**sglang-eval venv** (do not mix with `pip install -e .` — it upgrades torch and
breaks FlashInfer):

```bash
source /mnt/nfs/hoangduy/env.sh
"$UV" venv --python 3.12 /mnt/nfs/hoangduy/venvs/sglang-eval
source /mnt/nfs/hoangduy/venvs/sglang-eval/bin/activate
"$UV" pip install "sglang==0.5.13.post1" --prerelease=allow
"$UV" pip install "lm-eval>=0.4.5" pyyaml
export PYTHONPATH=/mnt/nfs/hoangduy/projects/llm-compressor
```

### Agentic (optional)

Enable in YAML (`agentic.enabled: true`) and configure:

- `tau2_dir` — cloned [tau2-bench](https://github.com/sierra-research/tau2-bench) with `uv sync`
- `user_base`, `user_model`, `user_key_file` — separate user-simulator endpoint
- `agent_base` / serve the model under test at an OpenAI-compatible URL

If user-sim is not configured, agentic self-skips.

### Test procedure

1. **Unit tests:** `pytest pipeline/tests/test_compare.py`
2. **Smoke:** `--set eval.tasks.0.limit=8` on a small model; self-compare must show 0 flips
3. **Agentic smoke:** `domain: mock`, `num_tasks: 1` against a live endpoint
4. **E2E:** BF16 vs W4AFP8 checkpoint on full static suite; review `report.md`

### Deferred: serving performance

Throughput / TTFT / concurrency sweeps are **not** in this pipeline stage. Use the
aiperf suite in `benchmarks/llm-perf-benchmarks/` (`scripts/run_all.sh`) against
the same served endpoint when needed. `report.md` includes a pointer stub.

## Method x scheme matrix

`quantization.method` x `quantization.scheme`:

| method | what it does |
| --- | --- |
| `gptq` | GPTQ INT4 weight quant (error compensation) |
| `awq` | AWQ scales + QuantizationModifier |
| `smoothquant+gptq` | SmoothQuant transform, then GPTQ |
| `smoothquant+awq` | SmoothQuant transform, then AWQ |
| `autoround` | AutoRound learned rounding (tier-2) |
| `spinquant+{gptq,awq}` | SpinQuant rotation, then weight quant (tier-2) |

`scheme`: `W4AFP8` (INT4 g128 weights + dynamic per-token FP8 activations) or
`W4A8` (INT4 g128 weights + dynamic INT8 activations). Both Hopper-native; saved
as `pack-quantized` for vLLM.

## Staged rollout (smoke test -> real run)

1. **Stage 0 - plumbing smoke** (`configs/qwen1_5_moe_smoke.yaml`): tiny MoE,
   32 samples, confirms the whole CLI path saves a pack-quantized artifact.
   Re-run with `--set quantization.scheme=W4A8` to confirm parameterization.
2. **Stage 1 - small MoE** (`configs/qwen3_30b_a3b.yaml`): Qwen3-30B-A3B, the 4
   M1 cells `{gptq,awq} x {W4AFP8,W4A8}` via `--set`.
3. **Stage 2 - serve + gate**: produce a BF16 baseline once
   (`--make-baseline artifacts/qwen3-30b-a3b-baseline.json`), point
   `eval.baseline` at it, and the gate becomes pass/fail. **Completes M1 DoD.**
4. **Stage 3 - M3 enablement** (`m3_enablement.py`): probe whether M3's experts
   auto-linearize; if not, author a `LinearExperts2D` subclass under
   `src/llmcompressor/modeling/`. Serve stock M3 in parallel (TP=8, EP,
   `--block-size 128`) to surface load issues early.
5. **Stage 4 - M3 candidates** (`configs/minimax_m3.yaml`): quantize the M3
   backbone (ignore vision + MSA indexer + gate + lm_head) to W4A8 and W4AFP8;
   serve on 8xH100 with FP8 KV + parsers.
6. **Stage 5 - sweep + ship** (`sweep.py`): run the method x format matrix on the
   collaborator benchmark, compare vs the MX baseline, pick the winner, stress
   test. **Completes M2 DoD.**

```bash
python -m pipeline.sweep --config pipeline/configs/minimax_m3.yaml \
    --methods gptq awq smoothquant+gptq smoothquant+awq \
    --schemes W4A8 W4AFP8
# -> sweep_results.{json,csv}
```

## MiniMax-M3 enablement (critical path)

`src/llmcompressor/modeling/` has no MiniMax module. Generic linearization only
covers experts that implement the transformers v5 `use_experts_implementation`
protocol or have a registered `LinearExperts2D`. Run the probe first:

```bash
python -m pipeline.m3_enablement --config pipeline/configs/minimax_m3.yaml
```

If it reports `needs_custom_module: true`, port a linear-experts module from the
GLM-4 / MiniMax-M2 definitions following
`docs/developer-tutorials/add-moe-support.md`.

## Notes

- The `eval.baseline` JSON is `{task: {metric: value}}`; produce it with
  `--make-baseline`. Higher-is-better metrics gate on `recovery_threshold`;
  perplexity gates on `max_ppl_increase`.
- `serve_verify.py` uses vLLM's offline `LLM`. Tool/reasoning parsers in the M3
  config's `serve.extra_args` apply to the HTTP `vllm serve` command, not the
  offline check.
- For models that exceed CPU RAM, set `model.device_map: auto_offload` +
  `model.offload_folder` + `model.max_memory`.

## Serve constraint: W4A8 MoE expert width must be a multiple of 256

The vLLM CUTLASS W4A8 grouped-GEMM MoE kernel requires each routed expert's
`moe_intermediate_size` to be divisible by **256**. `serve_verify.py` runs a
cheap preflight on the checkpoint config and fails fast (before loading vLLM)
with an actionable message when this is violated. Examples:

- Qwen1.5-MoE-A2.7B: `moe_intermediate_size=1408` (= 5.5 x 256) -> incompatible.
- Qwen3-30B-A3B: `moe_intermediate_size=768` (= 3 x 256) -> compatible.

Note this is on the **per-partition** width. With expert parallelism
(`serve.enable_expert_parallel: true`) each rank holds whole experts, so the
full width is used; with plain tensor parallelism the width is divided by TP,
which can break the multiple-of-256 requirement (e.g. 768 with TP=2 -> 384).

If a model you must serve fails this check:

1. **Pad** the expert intermediate dim up to the next multiple of 256
   (zero-pad gate/up_proj output rows + down_proj input cols; numerically
   lossless). Requires a small repack step at quantize/save time.
2. **Switch scheme** for that model to a kernel that has no such constraint
   (e.g. `W4A16` MoE via Marlin, or `FP8`) at the cost of FP8 activations /
   memory.
3. Sharding (TP/EP) cannot rescue a fundamentally non-256 width - only padding
   or a scheme change can.

Check any model's geometry before a run:

```bash
python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('<model_id>', trust_remote_code=True); print(getattr(c,'moe_intermediate_size', getattr(getattr(c,'text_config',c),'moe_intermediate_size',None)))"
```

## MoE gate must stay in the saved config's `ignore` (auto-handled)

The MoE router (`mlp.gate`) is correctly left **unquantized** during
quantization, but llm-compressor prunes ignore patterns that didn't match a
*quantized* module from the serialized `quantization_config.ignore`. That
silently drops the gate (and, for VL MoE, `vision_tower` / MSA `indexer`) from
the on-disk config. vLLM then treats those unquantized Linears as quantized and
either fails to load or mis-loads them -> **broken routing -> garbage output**
(empty / `\r\n` repetitions), even though the checkpoint weights are correct.

`quantize.py` fixes this via `_persist_ignore_to_config()`: after save, it
re-adds every recipe `ignore` pattern into the checkpoint's `config.json`. To
repair a checkpoint produced before this fix:

```bash
python - <<'PY'
import json, glob
for cfgp in glob.glob('artifacts/*/*/checkpoint/config.json'):
    d=json.load(open(cfgp)); qc=d.get('quantization_config',{}); ig=list(qc.get('ignore',[]))
    if 're:.*mlp.gate$' not in ig:
        qc['ignore']=ig+['re:.*mlp.gate$']; json.dump(d, open(cfgp,'w'), indent=2); print('patched', cfgp)
PY
```

Sanity check any checkpoint: its `config.json` `quantization_config.ignore`
should list `lm_head` and the MoE gate (plus vision/indexer for VL MoE).

## Deferred: SGLang serving (NOT in current scope)

Current scope is **quantization + evaluation on vLLM only**. SGLang serving is
deferred. Captured here so it is not re-investigated from scratch.

Key finding (verified Jun 2026): **no conversion stage and no quantize-stage
change is needed.** The compressed-tensors `pack-quantized` checkpoint is
engine-agnostic; SGLang reads the HF `config.json` `quantization_config` and
does the CUTLASS int8 repack in-memory at load, just like vLLM. There is no
"SGLang format" to convert to.

Caveats to handle when this is picked up:

- **W4AFP8 MoE support in SGLang is an open, unmerged PR**
  ([sgl-project/sglang#21741](https://github.com/sgl-project/sglang/pull/21741),
  `CompressedTensorsW4AFP8MoE`). Not in a release yet -> run that branch or wait
  for merge before expecting stock SGLang to load our W4AFP8 MoE artifacts.
- **`config.json` scheme hint:** that PR notes SGLang may otherwise fall back to
  `CompressedTensorsWNA16MoE`; a small post-save patch to `quantization_config`
  may be required to route to the W4AFP8 MoE scheme.
- **group_size must be 128** (CUTLASS W4A8 kernel limit) - already our default.
- **W4A8 (INT4 + INT8 act) on NVIDIA is effectively vLLM-only today**; SGLang
  lists `W4A8 linear` as TBD and the W4A8_DYNAMIC MoE entry is Ascend/ModelSlim,
  not the NVIDIA compressed-tensors path. Standardize on **W4AFP8** for anything
  that must later run on both engines.

Implementation sketch when un-deferred (contained; does not touch quantize or
the checkpoint format):

- add `serve.backend: vllm|sglang` to the config; `serve_verify.py` branches to
  `sglang.Engine(model_path=..., tp_size=..., mem_fraction_static=...)` (note
  the different arg names vs vLLM).
- `eval_gate.py` already routes via `eval.backend`; lm-eval supports
  `--model sglang`, so only the per-backend `model_args` string differs
  (`tp_size`/`dp_size`/`mem_fraction_static`).
- re-run the accuracy gate on both backends and compare (watch the TP8
  regression reported in #21741).
