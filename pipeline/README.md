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
  versioning.py      # timestamped artifact dirs + reproducible metadata
  run.py             # CLI tying the stages together
  sweep.py           # method x format comparison matrix (M2 Phase 3)
  m3_enablement.py   # MiniMax-M3 MoE linearization probe (Stage 3)
  configs/           # ready-to-run configs per stage
```

## Install (on the 8xH100 cluster)

```bash
cd /mnt/nfs/hoangduy/projects/llm-compressor
pip install -e .
pip install -r pipeline/requirements.txt   # vllm + lm-eval
```

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
