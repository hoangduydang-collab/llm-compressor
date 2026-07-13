# MiniMax-M3 AWQ Representative-Layer Rerun

## Verdict

The six-arm diagnostic completed, but produced no usable AWQ mapping evidence.
The result is an `incomplete` infrastructure/calibration-path failure, not a
quality pass or a quality failure.

Run:

```text
20260713T043659Z-m3-awq-representative
```

Controller:

```text
started: 2026-07-13T04:37:04Z
finished: 2026-07-13T05:04:20Z
controller rc: 1
```

All six arms returned `rc=1`:

| Arm | Status | Resolved | Completed | Skipped | Unprocessed |
|---|---|---:|---:|---:|---:|
| `offsetfix-layer8` | infrastructure failure | 129 | 0 | 0 | 129 |
| `offsetfix-layer31` | infrastructure failure | 129 | 0 | 0 | 129 |
| `offsetfix-layer59` | infrastructure failure | 129 | 0 | 0 | 129 |
| `nosmooth-layer8` | infrastructure failure | 128 | 0 | 0 | 128 |
| `nosmooth-layer31` | infrastructure failure | 128 | 0 | 0 | 128 |
| `nosmooth-layer59` | infrastructure failure | 128 | 0 | 0 | 128 |

The compact matrix confirms:

```json
{
  "pass": 0,
  "quality_failure": 0,
  "infrastructure_failure": 6,
  "missing": 0,
  "verdict": "incomplete"
}
```

## Failure

Every arm reached AWQ initialization and lifecycle finalization, but completed
zero mapping grid searches:

```text
AWQ produced no completed mapping metrics; skipped_mappings=0
RuntimeError: AWQ completed zero mapping grid searches:
resolved=128/129 skipped=0 unprocessed=128/129
```

The runner raised this error from
`pipeline/m3_awq_representative.py:693` after reading each arm's
`lifecycle.json`. This is the same empty-completion class as the previous
representative attempt, although the old `ZeroDivisionError` is fixed: the AWQ
metric logger now returns safely and the harness reports the lifecycle defect
explicitly.

Representative `lifecycle.json` artifacts were written for all six arms. They
show nonzero resolved mappings but zero completed and skipped mappings, leaving
all resolved mappings unprocessed. No `arm.json` quality evidence was produced.

## Infrastructure status

The tmux controller survived and waited for all six arms. The arms were
launched with `srun --exclusive --nodes=1`; at startup they occupied six
distinct nodes:

```text
offsetfix-layer8   gpu-h123
offsetfix-layer31  gpu-h113
offsetfix-layer59  gpu-h114
nosmooth-layer8    gpu-h115
nosmooth-layer31   gpu-h116
nosmooth-layer59   gpu-h117
```

There was no node-collision OOM, nested-Slurm rejection, tool interruption, or
Slurm timeout. The failure is therefore inside the representative AWQ
calibration/lifecycle path, not the detached launcher or allocation topology.

## Durable evidence

Logs:

```text
/mnt/nfs/hoangduy/logs/m3-awq-representative/20260713T043659Z-m3-awq-representative/
```

Results:

```text
/mnt/nfs/hoangduy/results/m3-awq-representative/20260713T043659Z-m3-awq-representative/
```

The controller log, six arm logs, `matrix.json`, and `report.md` are retained
at those paths. Their SHA-256 values are recorded below:

| Artifact | SHA-256 |
|---|---|
| `controller.log` | `1c77b87ac53ed3ecd2525efdcdd1f2e968f0b6eaf851315eb462b7e31b2c4978` |
| `offsetfix-layer8.log` | `e156b08d67a062742a2f12b7495e2cd7bf47bd3e23846849d48ea981b5aae69d` |
| `offsetfix-layer31.log` | `75a39907ce08f98cd51c8d7bed664beafde9a4e05dcd8eb9e6bc5f4bbe278c9e` |
| `offsetfix-layer59.log` | `a78297c8e7804c516cd27009a6c2a547c89208b197ee5d410ac6132d2820f5b2` |
| `nosmooth-layer8.log` | `5e5d0ecfa50b99a4700c847fcb01a00d3c25214bc1519abaa59af95955bee94d` |
| `nosmooth-layer31.log` | `40bfc21c4bf376def008b3a2222d2bf76c3267dc433dbd5a0e2b6da82da2746c` |
| `nosmooth-layer59.log` | `d6e3a47e865dbe32b299143e4b9218e45a40ea9121fb8c94fa1373e7b196cbb9` |
| `matrix.json` | `5d079339019c99828c3eee3de66045354215c9bb58e7178ccb06bd0a7926aefa` |
| `report.md` | `40b409e503bf351b4e064b571cb8494cab4afceca69d48b72484c5c6eb17ed7d` |

## Next action

Do not start full AWQ re-quantization. The representative harness still needs
to explain why calibration reaches resolved mappings but never executes a
mapping grid search. The next planner should inspect:

- `pipeline/m3_awq_representative.py` around lifecycle capture and the
  `completed_mapping_count` guard;
- `src/llmcompressor/modifiers/transform/awq/base.py` around mapping execution,
  skipped metrics, and `_log_error_metrics`;
- the per-arm `lifecycle.json` and logs above.

The unrelated BF16 production-evaluation commits were fetched from
`origin/duy-branch` but were not pulled into the worktree because an existing
uncommitted quality-matrix change would have been overwritten. They are not
part of this report.

## Planner analysis and instrumentation (2026-07-13)

### The bug is not yet addressed

Commits `2c052ff5` and `41b22714` only made the audit *expose* the empty
metrics and stopped the old `ZeroDivisionError`. The root cause —
`_smooth_activation_stats` never populating, so `_apply_smoothing` is a no-op —
is untouched.

### `completed=0` is MiniMax-M3-specific, not a shared-machinery bug

I reproduced the full real calibration path on **CPU** with a tiny real
`Qwen3MoeForCausalLM` driven through `oneshot` → `linearize_moe` →
`moe_calibration_context` → `AWQModifier`, instrumented with this harness's exact
resolved/completed/skipped accounting (no GPU, no MiniMax weights; DataLoader
input so no tokenizer needed):

| Configuration (mirrors) | resolved | completed |
|---|---:|---:|
| class-name sequential target (production) | 10 | 10 |
| single-layer isolation, first layer | 5 | 5 |
| single-layer isolation, middle layer | 5 | 5 |
| single-layer isolation, **last** layer | 5 | 5 |

Every configuration computes scales. The one-epoch lag (a layer's stats land on
the *next* `_apply_smoothing` call) always resolves. **So the generic
AWQ + linearize + all-experts + single-layer FX-sequential machinery is sound;
the failure is specific to MiniMax-M3** — a VL model (`minimax_m3_vl`,
`MiniMaxM3SparseForConditionalGeneration`) with a custom sparse MoE, offset
`MiniMaxM3VLRMSNorm`, and an FX trace fragile enough to need
`patch_minimax_m3_for_text_calibration`.

Leading hypothesis: under FX-traced sequential calibration the per-expert
`Linear`s are **inlined into the subgraph** rather than executed as `call_module`
nodes, so AWQ's activation-cache forward hook never fires and
`_smooth_activation_stats` stays empty → zero smoothing applied → garbage.

### Instrumentation added (this commit)

`AuditedAWQModifier` now records a **native** per-balance-layer forward-fire
counter (independent of AWQ's HooksMixin hooks), keyed by `smooth_name`, plus a
per-epoch timeline of `len(_smooth_activation_stats)`. Results are written into
each arm's existing `lifecycle.json` under a new `diagnostics` block. The
instrumentation is fail-safe (never changes an arm's verdict) and on by default;
disable with `M3_AWQ_HOOK_TRACE=0`.

Validated on the CPU surrogate: the counter reads `0` when stats are empty and
`>0` (with `never_fired=0`) exactly when smoothing completes — a faithful,
independent signal.

### Executor: run one instrumented arm (no GPU allocation for quality; standard calibration only)

```bash
python -m pipeline.m3_awq_representative arm \
  --layer 8 --variant offsetfix \
  --config pipeline/configs/minimax_m3_full_calib.yaml \
  --output-dir results/m3-awq-representative/diag-layer8-offsetfix
```

The arm still exits non-zero at the `completed=0` guard, but writes
`lifecycle.json` first. Read its `diagnostics` block:

- **`total_balance_forward_events == 0` / `balance_layers_never_fired_count`
  equals the resolved count** → the expert `Linear` forwards never executed
  during calibration. Confirms the FX-inlining / routing-bypass hypothesis. Fix
  direction: mark the MiniMax expert `Linear`s (or the linearized expert module)
  as **leaf modules** for the sequential tracer, or ensure the linearized expert
  forward dispatches through the per-expert `Linear.__call__`.
- **`total_balance_forward_events > 0` but stats stay empty** (see
  `smooth_activation_stats_timeline`) → the modules run but AWQ's activation-cache
  closure does not accumulate. Implicates the closure / loss-mask / all-experts
  path in `AWQModifier._setup_activation_cache_hooks`, not tracing.

Do not allocate GPUs for quality eval or start full re-quantization for this
diagnostic — it is a single standard calibration arm.

## Instrumented executor result (2026-07-13)

The requested single `offsetfix-layer8` diagnostic was queued with `srun` in
detached tmux and moved to the `debug` partition when compute resources were
busy:

```text
Slurm job: 12854
Partition/node: debug/gpu-h125
Result: /mnt/nfs/hoangduy/results/m3-awq-representative/diag-layer8-offsetfix
Log: /mnt/nfs/hoangduy/logs/m3-awq-representative/20260713T065500Z-m3-awq-diag-layer8-offsetfix.log
```

The arm completed with `return_code=1` after writing `lifecycle.json`. Its
diagnostics are decisive:

```text
resolved_mapping_count=129
completed_mapping_count=0
skipped_mapping_count=0
unprocessed_mappings=129
total_balance_forward_events=0
balance_layers_never_fired_count=129
smooth_activation_stats_timeline=[{epoch: 0, smooth_activation_stats_len: 0,
                                   total_balance_forward_events: 0}]
```

This confirms that none of the 129 MiniMax balance-layer targets executed
during the calibration pass. AWQ therefore collected no activation statistics,
performed no smoothing, and launched no grid searches. The result is an
infrastructure/calibration-path failure, not a quality verdict.

The diagnostic did not encounter a resource, launcher, CUDA, or model-loading
failure. The evidence supports the FX/sequential-tracing hypothesis: the
linearized MiniMax expert linears are not traversed as executable module
forwards, so the AWQ activation hooks never fire. Do not begin full
re-quantization until the tracing/dispatch path is corrected and this
instrumented arm reports nonzero balance forward events.

## Planner analysis of the MoE forward source + structural probe (2026-07-13)

I read the remote source the executor supplied in
`M3_MINIMAX_MOE_FORWARD_ANALYSIS.md` and traced the full local dispatch path.
It resolves the *original* mystery but deepens the *post-linearization* one:

**What the remote source explains.** The unmodified `MiniMaxM3VLExperts.forward`
runs `F.linear` against slices of stacked `gate_up_proj`/`down_proj` parameters
and never calls per-expert `nn.Linear` modules. So on the *raw* checkpoint there
are no per-expert modules to hook. That is expected and is exactly why
`linearize_moe` exists.

**Why that is not the whole story.** After `linearize_moe`, the executor's own
meta-model check confirms `layers[8].mlp.experts` is `LinearExperts2D`, whose
`forward` loops `for expert_index in range(num_experts): expert = self[i];
expert(...)` — i.e. it *does* dispatch through per-expert `ExpertMLPWithGate`
modules (and their `gate_proj`/`up_proj`/`down_proj` `Linear`s). The 129 resolved
mappings confirm those per-expert `Linear`s exist. So the linearized container
*should* fire the AWQ hooks.

**I exhaustively ruled out the obvious suspects, locally:**

- `linearize_moe` detection: `MiniMaxM3VLExperts` matches `FusedExpertsProtocol`,
  so `get_non_linearized_moes` finds it and `set_submodule` swaps in
  `LinearExperts2D` (confirmed by 129 resolved + the meta check).
- Order of operations: `linearize_moe` runs in `oneshot` (line ~230) *before*
  `session.initialize` / AWQ hook registration, and inside
  `moe_calibration_context` (so `_CALIBRATE_ALL_EXPERTS=True` during both trace
  and calibration). Hooks land on the linearized children.
- Sequential target match: the arm sets `sequential_targets=[<instance path>]`;
  I confirmed `match_named_modules` matches that plain path to exactly the layer,
  so layer 8 is a **leaf** — not traced into, run as opaque Python.
- Subgraph execution: `subgraph.forward(model, **inputs)` binds `self=model`, so
  `call_module` nodes resolve against the **live** model objects AWQ hooked.
- CPU repro parity: a real linearized Qwen3-MoE through this exact
  path — class-name target *and* single-layer isolation — computes scales
  (`completed>0`). The generic machinery is sound.

Every static path predicts the linearized experts fire, yet all 129 balance
targets show zero forward events. The divergence must therefore be
**runtime-structural** and MiniMax-specific: either the sparse block is not
entered for this layer, the container does not loop its children, an expert
bypasses its own `Linear`s, or AWQ hooked a stale/duplicate object rather than
the one that executes.

### Structural probe added (this commit)

`AuditedAWQModifier` now installs a native, fail-safe probe that walks the live
target decoder's `mlp -> mlp.experts -> mlp.experts[0] -> mlp.experts[0].gate_proj`
chain (plus `mlp.gate`, `mlp.shared_experts`). It records each object's runtime
type, a per-level forward-fire counter, `num_experts`, and
`resolved_object_is_live` — whether the executing `gate_proj` is the *same object*
AWQ resolved into `self._resolved_mappings`. Written into `lifecycle.json` under
`diagnostics.structure_probe`. Validated on a CPU surrogate: counts propagate at
every level and `resolved_object_is_live` correctly tracks object identity.
On/off with `M3_AWQ_HOOK_TRACE` (default on); never changes the arm verdict.

### Executor: rerun the same single instrumented arm (no GPU quality eval)

```bash
python -m pipeline.m3_awq_representative arm \
  --layer 8 --variant offsetfix \
  --config pipeline/configs/minimax_m3_full_calib.yaml \
  --output-dir results/m3-awq-representative/diag2-layer8-offsetfix
```

Same standard calibration arm as before (it still exits non-zero at the
`completed=0` guard after writing `lifecycle.json`). Read
`diagnostics.structure_probe.fire_counts` — the first level with count `0`
pinpoints the fix:

| First zero level | Root cause | Fix direction |
|---|---|---|
| `mlp` = 0 | sparse block not entered (dense path / dead branch / layer not executed) | check layer selection + traced subgraph contents |
| `mlp.experts` = 0 (mlp>0) | block runs, dispatches around the linearized experts object | MiniMax `mlp.forward` calls a path other than `self.experts(...)` |
| `mlp.experts.0` = 0 (experts>0) | container does not loop children | MiniMax-specific `LinearExperts2D` subclass forward |
| `gate_proj` = 0 (experts.0>0) | expert bypasses its own `Linear`s | expert forward wiring |
| `gate_proj` > 0 but balance count 0, or `resolved_object_is_live` false | AWQ hooked a stale/duplicate object | re-resolve mappings against live modules post-trace |

Do not allocate GPUs for quality eval or begin re-quantization — this is a
single standard calibration arm.

## Structural probe result (2026-07-13)

The requested single-arm rerun was launched in detached tmux with one
exclusive GPU:

```text
Slurm job: 12860
Node: gpu-h123
Run: 20260713T075500Z-m3-awq-structural-layer8
Arm: offsetfix-layer8
Controller finished: 2026-07-13T08:21:33+00:00
Controller rc: 1
Arm rc: 1
```

The arm reached the lifecycle guard and wrote its evidence. It resolved 129
AWQ mappings but completed zero grid searches:

```text
resolved_mapping_count=129
completed_mapping_count=0
skipped_mapping_count=0
unprocessed_mappings=129
total_balance_forward_events=0
smooth_activation_stats_len=0
```

The structural dispatch probe recorded:

```json
{
  "fire_counts": {
    "mlp": 0,
    "mlp.shared_experts": 0,
    "mlp.gate": 0,
    "mlp.experts": 0,
    "mlp.experts.0": 0,
    "mlp.experts.0.gate_proj": 0
  },
  "types": {
    "mlp": "MiniMaxM3VLSparseMoeBlock",
    "mlp.shared_experts": "MiniMaxM3VLDenseMLP",
    "mlp.gate": "MiniMaxM3VLTopKRouter",
    "mlp.experts": "LinearExperts2D",
    "mlp.experts.0": "ExpertMLPWithGate",
    "mlp.experts.0.gate_proj": "Linear"
  },
  "num_experts": 129,
  "resolved_object_is_live": true
}
```

This is more specific than the previous hook-only result. The first zero is
the enclosing `mlp` itself: layer 8's sparse-MoE block is never entered during
the calibration pass. Because the live object identity check is true, this
rules out stale AWQ mappings. Because every deeper level is also zero, it also
rules out `LinearExperts2D` child dispatch and expert `Linear` wiring.

The remaining investigation target is the layer execution/sequential
calibration path before MoE dispatch: inspect the traced subgraph and the
MiniMax text-calibration patch to determine why the selected decoder layer is
not being called. This run is an infrastructure/calibration-path failure, not
a quantized-quality verdict.

## Planner analysis of the structural probe + layer-level probe (2026-07-13)

The probe is decisive at the level it measured: `mlp` is the first zero, and
`resolved_object_is_live: true`. That eliminates the entire "AWQ hooked the
wrong object" family — AWQ resolved the exact live `gate_proj` that would
execute — and eliminates every dispatch bug *inside* the linearized experts.
The MoE block for layer 8 is simply never entered.

**But `mlp = 0` on its own is not yet consistent with either tracing outcome**,
which is why I traced the partition logic in `pipelines/sequential/helpers.py`:

- If the sequential target **matched**, `get_sequential_ancestors` makes layer 8
  a *leaf*: not traced into, executed as an opaque `call_module` against the live
  model. Its `mlp` would then fire (this is exactly my passing CPU repro).
- If the target **did not match** (empty target set), *every* module is a leaf,
  so the trace stops at the top-level `self.model(...)` call and the whole model
  runs eagerly — `mlp` would fire too.

Neither predicts `mlp = 0`. The missing piece is one level up: **is layer 8's
decoder `forward` executed at all during the calibration pass?** If it is, the
fault is a sparse/dense dispatch or an inlined-MoE issue *inside* the layer; if
it is not, the fault is in partitioning/tracing (the layer is absent from the
executed subgraphs, or its subgraph never runs).

### Extended probe (this commit)

I widened the structural probe upward from the decoder layer:
`language_model -> layers -> decoder -> {input_layernorm, self_attn,
post_attention_layernorm, mlp} -> experts -> experts[0] -> gate_proj`. It also
now emits `diagnostics.num_sequential_epochs` (the number of executed
subgraphs — `1` means the sequential target produced no partition boundary and
the whole model was traced as a single graph; `>1` means the layer partitioned
as expected). Same fail-safe, no-verdict-impact, `M3_AWQ_HOOK_TRACE` toggle.

### Executor: rerun the same single arm (no GPU quality eval)

```bash
python -m pipeline.m3_awq_representative arm \
  --layer 8 --variant offsetfix \
  --config pipeline/configs/minimax_m3_full_calib.yaml \
  --output-dir results/m3-awq-representative/diag3-layer8-offsetfix
```

Read `diagnostics.num_sequential_epochs` and
`diagnostics.structure_probe.fire_counts`. The combination is fully decisive:

| Signal | Meaning | Fix locus |
|---|---|---|
| `num_sequential_epochs == 1` | target never partitioned; whole model = one traced graph | why `sequential_targets=[<instance path>]` fails to match/partition on the real VL model |
| `decoder == 0` | layer 8's `forward` never executes in any subgraph | subgraph partition / trace omits the layer |
| `decoder > 0` but `mlp == 0` | layer runs but skips its MoE block | sparse/dense dispatch, or MoE inlined into the trace (module `__call__` bypassed) |
| `self_attn > 0`, `mlp == 0` | attention runs, MoE path specifically bypassed | MoE-block dispatch inside the layer forward |

If it is cheap, also paste `diagnostics.smooth_activation_stats_timeline` from
the **existing** `20260713T075500Z` `lifecycle.json` — its length already
equals `num_sequential_epochs` and may answer the partition question with no
rerun. Do not allocate GPUs for quality eval or begin re-quantization.

## Root-cause reframing: the arm deviated from production tracing (2026-07-13)

The zero-cost check settled it: `num_sequential_epochs = 1`. The whole model
was traced as a single subgraph, so the decoder layers were inlined and no
expert forward could fire. Crucially, this is **not the production calibration
path** — it is an artifact of how the arm was targeting layers:

- **Production** (`pipeline/configs/minimax_m3*.yaml`) uses
  `sequential_targets: ["MiniMaxM3VLDecoderLayer"]` — a **class name**, so every
  decoder layer is a sequential target/leaf and executes eagerly.
- **The arm** overrode this to a single decoder **instance path**
  (`run_arm`: `sequential_targets = ["model.language_model.layers.8"]`), which
  on the real VL model produced no partition boundary (one subgraph) and inlined
  the layers.

So `completed=0` here is most likely a **harness artifact**, not direct proof of
the production AWQ garbage bug. The diagnostic must mirror the real quantization
run as closely as possible; single-layer isolation belongs in the quantization
**ignore list**, not in the sequential-tracing target.

### Fix (this commit): production-faithful tracing, ignore-list isolation

`prepare_arm_config` / `run_arm` no longer override `sequential_targets`. The
arm now keeps the config's class-name target (`["MiniMaxM3VLDecoderLayer"]`)
so tracing/partitioning match production, and isolates layer 8 purely via the
existing `layer_exclusion_pattern` ignore entry (AWQ mappings are already scoped
to the layer). The legacy single-instance-path override is preserved behind
`M3_AWQ_SINGLE_LAYER_TRACE=1` for controlled A/B comparison.

### Executor: rerun the same single arm with production tracing (no GPU quality eval)

```bash
python -m pipeline.m3_awq_representative arm \
  --layer 8 --variant offsetfix \
  --config pipeline/configs/minimax_m3_full_calib.yaml \
  --output-dir results/m3-awq-representative/diag4-classname-layer8-offsetfix
```

Interpretation:

- **`completed > 0`** (expect `num_sequential_epochs` ≈ number of decoder
  layers, and `structure_probe.fire_counts.mlp > 0`) → confirms the
  instance-path override was the artifact **and** that the production
  calibration/smoothing path is sound. The original garbage bug then lives
  elsewhere (offset-norm / config mismatch, scale application, or W4AFP8), and
  the investigation should redirect there.
- **`completed == 0` again** with class-name tracing → the failure is real on
  the production path; the structure probe then localizes it.

Note this run does a full sequential pass over all decoder layers (only layer 8
is grid-searched), so it is closer to production cost than the isolated arm.

### Highest-value parallel check (no rerun, likely already in logs)

Independently of the harness, confirm whether the **original production AWQ
quantization run** (the one that produced the garbage model) itself completed
grid searches / wrote real smoothing scales. If it did, the smoothing path was
fine and the garbage cause is elsewhere; if it did not, that is the same
failure mode and the direct target. This is the fact that ties the diagnostic
back to the original bug.

Durable evidence:

```text
Result:
/mnt/nfs/hoangduy/results/m3-awq-representative/20260713T075500Z-m3-awq-structural-layer8/offsetfix-layer8/lifecycle.json

Arm log:
/mnt/nfs/hoangduy/logs/m3-awq-representative/20260713T075500Z-m3-awq-structural-layer8/offsetfix-layer8.log

Controller log:
/mnt/nfs/hoangduy/logs/m3-awq-representative/20260713T075500Z-m3-awq-structural-layer8/controller.log
```

## Zero-cost partition check (2026-07-13)

Before launching another arm, the existing lifecycle artifact was inspected as
requested. Its complete sequential timeline contains exactly one entry:

```json
{
  "smooth_activation_stats_timeline": [
    {
      "epoch": 0,
      "smooth_activation_stats_len": 0,
      "total_balance_forward_events": 0
    }
  ]
}
```

Therefore the existing run implies:

```text
inferred num_sequential_epochs = 1
```

This is consistent with the sequential target producing no partition boundary.
Combined with `mlp=0`, it moves the leading suspicion toward the layer
execution/partition path rather than any expert dispatch implementation.

The zero-cost artifact cannot fully distinguish these two remaining cases,
because the original probe did not instrument the enclosing decoder or
language-model layers:

1. layer 8 was omitted from the executed subgraph; or
2. layer 8 executed but its decoder forward skipped the sparse-MoE block.

No rerun or GPU allocation was performed for this check. The next diagnostic,
if needed, is the planner's upward probe recording
`language_model -> layers -> decoder -> {input_layernorm, self_attn,
post_attention_layernorm, mlp}` fire counts and the explicit sequential-epoch
count.

## Diag3 and production-faithful diag4 results (2026-07-13)

Both requested single-arm runs completed. The second run used the planner's
production-tracing change from commit `97ff419e`, which keeps the configured
class-name sequential target (`MiniMaxM3VLDecoderLayer`) instead of forcing the
single decoder instance path.

| Run | Job | Node | `num_sequential_epochs` | Completed | RC |
|---|---:|---|---:|---:|---:|
| `diag3-layer8-offsetfix` | 12861 | `gpu-h123` | 1 | 0 | 1 |
| `diag4-classname-layer8-offsetfix` | 12862 | `gpu-h117` | 1 | 0 | 1 |

The upward probe in `diag4` recorded zero at every level:

```json
{
  "language_model": 0,
  "layers": 0,
  "decoder": 0,
  "decoder.input_layernorm": 0,
  "decoder.self_attn": 0,
  "decoder.post_attention_layernorm": 0,
  "mlp": 0,
  "mlp.shared_experts": 0,
  "mlp.gate": 0,
  "mlp.experts": 0,
  "mlp.experts.0": 0,
  "mlp.experts.0.gate_proj": 0
}
```

Both runs also had:

```text
resolved_mapping_count=129
completed_mapping_count=0
total_balance_forward_events=0
smooth_activation_stats_len=0
```

Consequently, the class-name tracing change did not create a partition
boundary in the real arm: the whole model still produced one sequential
subgraph, and no live language-model or decoder module fired. The proposed
fix therefore did not resolve the runtime behavior; the partition/tracing
mechanism still needs investigation before this can be treated as a
production-path AWQ result.

The targeted CPU regression suite also exposed one stale expectation after the
intentional behavior change:

```text
46 passed, 1 failed
FAILED pipeline/tests/test_m3_awq_representative.py::
test_prepare_arm_config_isolates_one_layer_without_mutating_source
```

That test still expects the old instance-path sequential target, while the
implementation now intentionally preserves the production class-name target.
No checkpoint or quality verdict was produced.

Durable evidence:

```text
Diag3 lifecycle:
/mnt/nfs/hoangduy/results/m3-awq-representative/diag3-layer8-offsetfix/offsetfix-layer8/lifecycle.json

Diag4 lifecycle:
/mnt/nfs/hoangduy/results/m3-awq-representative/diag4-classname-layer8-offsetfix/offsetfix-layer8/lifecycle.json

Diag3 logs:
/mnt/nfs/hoangduy/logs/m3-awq-representative/20260713T083700Z-m3-awq-diag3-layer8/

Diag4 logs:
/mnt/nfs/hoangduy/logs/m3-awq-representative/20260713T084500Z-m3-awq-diag4-classname-layer8/
```

## Planner root cause: the VL multimodal wrapper is not traceable (CPU, no GPU, no weights) — 2026-07-13

The diag3/diag4 "1 subgraph, `completed=0`, all-zero probe" outcome is **not**
a targeting-syntax problem and **not** an AWQ/MoE bug. It is a **sequential
tracing failure specific to the MiniMax-M3-VL multimodal wrapper**. All three
findings below were reproduced locally on CPU with tiny random-weight models
(transformers 5.12, in-repo `llmcompressor`); no cluster job, no GPU, no real
checkpoint.

### 1. The generic AWQ + `linearize_moe` + sequential path is sound

A tiny real `Qwen3MoeForCausalLM` taken through the exact
`oneshot -> linearize_moe -> moe_calibration_context -> AWQModifier` path, with
expert-level AWQ mappings mirroring the MiniMax MoE mapping shape:

```text
sequential_targets=["Qwen3MoeDecoderLayer"]  -> 5 subgraphs, experts fire
                                                (2408 balance events), completed=20
sequential_targets=[r"re:.*layers\.1$"]      -> 2 subgraphs, isolates layer 1,
                                                completed=5
```

So **both** class-name and single-instance-path targeting partition correctly
and drive expert forwards + grid searches to completion. The machinery works.
(The first sequential epoch always shows `completed=0` because the head
subgraph precedes the first target — an expected artifact, not the bug.)

### 2. The class name is correct; the VL wrapper's forward is the blocker

`MiniMaxM3VLDecoderLayer` exists (modeling line 632) and `match_named_modules`
matches it (repro: 3/3 tiny decoder layers matched). But tracing a tiny
`MiniMaxM3VLModel` (the vision+text wrapper) as root with a **text-only**
sample **fails inside the multimodal front-matter**:

```text
File ".../modeling_minimax_m3_vl.py", line 1356, in get_placeholder_mask
    inputs_embeds[special_image_mask].numel() == image_features.numel(),
AttributeError: 'NoneType' object has no attribute 'numel'
```

Mechanism: the sequential tracer must symbolically execute everything *above*
the decoder loop. In the VL wrapper that includes `get_image_features` (the
vision tower), `get_placeholder_mask`, `masked_scatter`, and the
`if pixel_values is not None` / `if image_features is not None` branches. The
autowrapper rewrites the image/video branches into `torch.fx.wrap` functions
that return **non-None Proxies**, so `image_features is not None` evaluates
true during tracing and the image-token-count validation runs on symbolic
values. The trace never reaches the decoder layers -> no target nodes ->
degenerate single subgraph -> decoders never calibrated (`completed=0`,
all-zero probe including `language_model`). Locally this surfaces as a hard
`TraceError`; in the production arm, with different inputs/config, the same
region instead degenerated silently to 1 subgraph.

### 3. Tracing the language-model subtree as root partitions cleanly

Same tiny model, root = `model.language_model` (`MiniMaxM3VLTextModel`),
text-only sample:

```text
root= MiniMaxM3VLTextModel
matched decoder layers: 3
num_subgraphs= 4    (expected 4 = 3 layers + head)   PASS
```

### The `get_placeholder_mask` crash is real but ALREADY PATCHED in production

`pipeline/quantize.py:125` calls `patch_minimax_m3_for_text_calibration(model)`
(`pipeline/minimax_m3_config.py:257`), which coerces the non-Tensor
`image_features`/`video_features` proxies to `None` before `get_placeholder_mask`
runs — exactly neutralizing the `.numel()` crash above. So the raw crash is not
what production hits.

### CRITICAL NEGATIVE RESULT: the architecture alone traces fine — I cannot reproduce the production collapse offline

With that same patch applied, I traced progressively more production-faithful
tiny models, all text-only sample, all on CPU:

| Root model | Attention | Patch | `num_subgraphs` (expected) |
|---|---|---|---|
| `MiniMaxM3VLModel` (VL wrapper) | full | no | **RAISES** (`.numel()` on None) |
| `MiniMaxM3VLModel` | full | yes | 4 (4) ✓ |
| `MiniMaxM3VLModel` | sparse (Lightning Indexer) | yes | 4 (4) ✓ |
| `MiniMaxM3SparseForConditionalGeneration` (real root; path `model.language_model.layers.N`) | sparse | yes | 4 (4) ✓ |

The last row matches production's structure (root wrapper → `self.model` VL →
`language_model.layers`) and **still partitions correctly**. So neither the
multimodal wrapper (once patched), the sparse Lightning-Indexer attention, nor
the extra wrapper nesting reproduces the production `1 subgraph / completed=0`.
`infer_sequential_targets` also passes `["MiniMaxM3VLDecoderLayer"]` through
unchanged (ruled out).

**Conclusion: the collapse is not architectural. It is a production
load/environment factor my offline reconstruction lacks.** The dominant suspect
is `trust_remote_code: true` (config: `id: MiniMaxAI/MiniMax-M3`,
`auto_class: AutoModelForImageTextToText`): the checkpoint may load its own
remote modeling `.py`, in which the decoder layer class is **not** literally
named `MiniMaxM3VLDecoderLayer` (or lives in a `transformers_modules.*` module).
Then `match_named_modules(model, ["MiniMaxM3VLDecoderLayer"])` matches **zero**
modules → no target nodes → single degenerate subgraph → decoders never
calibrated → `completed=0`, `smooth_activation_stats_len=0` → **an effectively
un-smoothed 4-bit model, i.e. the original garbage-at-eval symptom.** This also
explains why external `cyankiwi_awq` passed: a target that actually matches the
loaded class.

### Decisive GPU-free checks for the executor (single short probe, no rerun/quant)

Load the model exactly as the arm does (`trust_remote_code=True`,
`AutoModelForImageTextToText`) and print, before any calibration:

```python
dl = [m for _, m in model.named_modules()
      if type(m).__name__.endswith("DecoderLayer")]
print("decoder class:", type(dl[0]).__name__, "| module:", type(dl[0]).__module__)
from compressed_tensors.utils.match import match_named_modules
print("match count for MiniMaxM3VLDecoderLayer:",
      len(list(match_named_modules(model, ["MiniMaxM3VLDecoderLayer"]))))
```

- **match count == 0** → confirmed root cause. Fix = set `sequential_targets`
  to the *actual* loaded class name (what `type(dl[0]).__name__` prints), or
  target by instance-path regex `re:.*language_model[.]layers[.]\d+$` (proven to
  partition in the generic repro). No architecture change needed.
- **match count == 61 (all decoders)** but trace still yields 1 subgraph →
  grep the trace log for `Expected .* subgraphs, but only traced` (helpers.py:162)
  and capture the traced graph's `call_module` targets; the remote-code forward is
  then defeating FX/autowrap and needs a `tracing_ignore` / traceable wrapper.

Also confirm (from the ORIGINAL production log) whether it wrote real smoothing
scales / completed grid searches — expected zero if the trace degenerated.

### Fix direction (independent of which check fires)

AWQ modifies only the language model's Linear weights (smoothing scales fold
into decoder-layer weights); the vision tower and projector are never quantized.
So quantizing the text subtree yields bit-identical quantized weights. Preferred
fix: keep the full model loaded but make `sequential_targets` match the real
decoder class (or the `language_model.layers.N` instance path). Physical
detach/reattach of the text model also works but risks `lm_head`↔`embed_tokens`
tying and config/save plumbing, so it is the fallback, not the first choice.

Local repro scripts (planner scratchpad, not committed): `repro_awq_moe_stats.py`
(generic-path proof: classname & instance-path both complete) and
`repro_minimax_trace.py` (VL trace matrix above).

### Wired-in provenance logging (self-diagnosing on the real box)

Because the collapse is an environment/load factor, the pipeline now records it
directly instead of leaving it to guesswork. `pipeline/provenance.py`
(`log_model_provenance`) runs in both the production quantize path
(`pipeline/quantize.py`, before oneshot) and the representative arm
(`pipeline/m3_awq_representative.py`, after target resolution). It writes
`model_provenance.json` next to the run outputs and prints a summary capturing:

- **Model-code origin**: loaded class name/module/file and `is_remote_code`
  (`transformers_modules.*` ⇒ trust_remote_code loaded the checkpoint's own
  modeling code), plus `config.auto_map`.
- **Decoder-layer classes**: the actual class name(s) present and per-class
  counts (installed vs remote), so a name that isn't `MiniMaxM3VLDecoderLayer`
  is visible immediately.
- **Target match**: `match_count` for each configured `sequential_targets`
  entry, flagged `<-- ZERO MATCH (collapse cause)` when zero.
- **Full environment**: python/platform/CUDA + a complete installed-package
  snapshot (`environment.installed_packages`, pip-freeze equivalent) for exact
  local-vs-cluster diffing.

`LOCAL_ENV_PROVENANCE.json` (repo root) is the planner's local baseline
(transformers 5.12.1, torch 2.12.1+cpu, 127 packages, `match_count=3`,
`is_remote_code=false`). The executor diffs the cluster's `model_provenance.json`
against it: a decoder class name ≠ `MiniMaxM3VLDecoderLayer` and/or
`match_count=0` confirms the root cause; a transformers/llmcompressor version
skew would show up in the package diff.
