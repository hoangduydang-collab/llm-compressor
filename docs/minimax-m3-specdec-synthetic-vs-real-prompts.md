# MiniMax-M3 EAGLE3 speculative decoding: synthetic vs real prompts

## Prompt sources

This note documents the MiniMax-M3 EAGLE3 study's comparison between its
AA-style synthetic-prompt sweep (Wave 1) and its natural ShareGPT-prompt arm
(Wave 2 Phase A). It does not compare this study with a collaborator's result:
that requires the collaborator's serving configuration and raw measurements.

| Arm | Prompt source | How the prompt is made | Input length | Output policy |
|---|---|---|---:|---|
| Wave 1 synthetic | AA-style input generator | Random synthetic tokens, rather than natural-language text | 1k and 10k tokens | Natural stopping |
| Wave 2 real | aiperf `--public-dataset sharegpt` | Pre-staged ShareGPT conversations; aiperf keeps the first user message | Mean ≈227 tokens | Natural stopping |

The synthetic arm ran `performance.aa.run_aa_sweep` with `--inputs 1k,10k`;
its prompt content was deliberately random tokens. The ShareGPT arm read the
pre-staged `ShareGPT_V3_unfiltered_cleaned_split.json` dataset offline and sent
requests through aiperf's chat endpoint. Public-dataset loaders retain only the
first message, so this is real single-turn text, not a multi-turn conversation.

Neither arm used `ignore_eos` or a forced minimum output in this comparison.
That matters: later in the study, forcing an 8k continuation raised acceptance
substantially and must not be confused with a prompt-source effect.

## Experiment setup

| Component | Configuration |
|---|---|
| Target | In-house MiniMax-M3 GPTQ W4AFP8 ABI overlay |
| Speculative drafter | `Inferact/MiniMax-M3-EAGLE3` at `44cafa5ace418d8b22e2958df0c6aa1f2476842c` |
| Serving | vLLM 0.24.0; Humming indexed 0.1.10; FP8 KV cache; prefix caching |
| Hardware/topology | One 8×H100 node, TP8/EP8 |
| Decoding comparison | k=0 control versus EAGLE3 k=3, temperature 0.6, natural stopping |

| Measurement | Window | Arms | Implementation and result commits |
|---|---|---|---|
| Wave 1 synthetic sweep | `20260727T061506Z` | k=0/1/3/5; 1k/10k × conc-1/10 | `88d1997e`, `bc6344bf` |
| Wave 2 Phase A natural prompts | `20260727T064934Z-wave2` | k=0/3; ShareGPT at conc-1/10 | `15b1aab1`, `48f2ea96` |

The wave controllers were
[`pipeline/slurm/run_specdec_eagle3_srun.sh`](../pipeline/slurm/run_specdec_eagle3_srun.sh)
and
[`pipeline/slurm/run_specdec_wave2_srun.sh`](../pipeline/slurm/run_specdec_wave2_srun.sh).
They held the target, kernel, topology, and serve flags fixed within each wave;
the speculative configuration was the serve-level variable.

**Comparability limit:** this is a cross-window comparison, not a directly
paired A/B. Wave 1 reported an arm-level acceptance mean from periodic
`SpecDecodingLogging` lines, whereas Wave 2 used per-cell Prometheus counter
deltas. The primary report deliberately uses the figures as a practical
decomposition of prompt naturalness, but they do not support a precise
statistical claim below the study's noise floor.

### Reproduction procedure

There are two distinct reproduction levels:

1. **Re-aggregate the historical artifacts** to reproduce the reported tables.
   This is the recommended path; it does not reserve GPUs or serve a model.
2. **Run a new comparable workload** to collect a fresh comparison. The former
   Polaris cluster is retired, so this requires porting the legacy launchers to
   the current scheduler, filesystem, and GPU topology. A fresh run measures
   the same design, not bit-for-bit identical throughput.

#### Current archive location (verified 2026-09-04)

The former cluster's result artifacts were migrated to a private archive on
`138.252.188.36`. The archive is under an owner-specific path and is not
directly accessible to collaborators. Request access or an exported copy from
the experiment owner; do not assume the former `/mnt/nfs/hoangduy` path, or
any owner-specific archive path, is available.

```bash
export ARCHIVE_HOST=138.252.188.36
# Obtain ARCHIVE_HOME from the experiment owner after access is approved.
```

The private archive contains both comparison windows and their expected primary
evidence:

```text
$SPECDEC_ROOT/20260727T061506Z/
  aggregate.json, actual-commit.txt, arm-k3/{serve.log,spec-metrics.log,
  aa-sweep.log,backend-attestation.json}

$SPECDEC_ROOT/20260727T064934Z-wave2/
  aggregate.json, actual-commit.txt, arm-natural-k3/serve.log,
  arm-natural-k3/metrics/natural-t06-c1-{pre,post}.txt,
  arm-natural-k3/natural/t06/conc_1/profile_export_aiperf.json
```

#### Re-aggregate the historical artifacts

After the experiment owner provides a permitted archive mount or copy, set
`ARCHIVE_HOME` to its root. The aggregators use only the Python standard
library, so the retired quant/perf virtual environments are not required for
this step:

```bash
export REPO="$ARCHIVE_HOME/projects/llm-compressor"
export WAVE1_ROOT="$ARCHIVE_HOME/results/m3-specdec-eagle3/20260727T061506Z"
export WAVE2_ROOT="$ARCHIVE_HOME/results/m3-specdec-eagle3/20260727T064934Z-wave2"
export PY=python3
```

Use a separate writable directory for regenerated JSON; the input windows are
historical evidence and should be treated as read-only:

```bash
export OUT="$ARCHIVE_HOME/results/reproductions/m3-specdec-prompt-source"
mkdir -p "$OUT"

"$PY" "$REPO/pipeline/specdec_aggregate.py" \
  --root "$WAVE1_ROOT" \
  --out-json "$OUT/wave1-aggregate.json"

"$PY" "$REPO/pipeline/specdec_wave2_aggregate.py" \
  --root "$WAVE2_ROOT" \
  --out-json "$OUT/wave2-aggregate.json"
```

The archived re-aggregation was verified on 2026-09-04:

- Wave 1 regenerates the k=3 acceptance evidence (2.45; per-position
  0.70/0.46/0.29) and greedy-equivalence table. It cannot regenerate the AA
  throughput cells because the `aa_sweep_summary.json` it references remained
  in the retired external `benchmarks/results/` tree. Use the preserved
  `$WAVE1_ROOT/aggregate.json` for the original 1k, conc-1 137.5 → 236.0
  tok/s/user row until that summary is restored.
- Wave 2 regenerates the complete natural ShareGPT table, including the
  temp-0.6, conc-1 2.473 accepted length and 137.9 → 249.8 tok/s/user
  result. Its per-cell metric snapshots and aiperf profile export migrated.

#### Run a new comparable workload

The legacy controllers are preserved for their experiment contract, but cannot
run unchanged on the current private archive host: they hard-code the retired
`/mnt/nfs/hoangduy` paths and submit exclusive Slurm jobs on the former
cluster. Start with a fresh isolated checkout and record the runtime revision
from each historical window's `actual-commit.txt`. The report commits
(`88d1997e`, `bc6344bf`, `15b1aab1`, and `48f2ea96`) identify implementation
and documentation milestones, but the recorded `actual-commit.txt` is
authoritative for a historically aligned run.

Before scheduling a new run, port the controllers
[`run_specdec_eagle3_srun.sh`](../pipeline/slurm/run_specdec_eagle3_srun.sh)
and
[`run_specdec_wave2_srun.sh`](../pipeline/slurm/run_specdec_wave2_srun.sh) to
the current environment. Preserve the following invariants:

| Requirement | Historical contract | Migration status verified 2026-09-04 |
|---|---|---|
| Target checkpoint | `artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay` | Not found at the equivalent archive path; restore or substitute with a documented, verified equivalent |
| EAGLE3 drafter | `hf_assets/Inferact/MiniMax-M3-EAGLE3`; architecture must be `LlamaForCausalLMEagle3` | Not found at the equivalent archive path; restore exact revision |
| Humming side-install | `venvs/humming-0.1.10-site`, with `ct_input_format`, `grouped_expert_bounds`, `tma_store_fence`, and `tma_store_commit` checks passing | Side-install directory is present; patch checks were not re-run |
| Python environments | `venvs/quant` and `venvs/perf`; the latter must provide aiperf 0.8.x | Neither historical virtual environment was found; rebuild from pinned requirements before a new run |
| Real-prompt dataset | `artifacts/aiperf-datasets/.cache/aiperf/datasets/ShareGPT_V3_unfiltered_cleaned_split.json`, pre-staged because the run is offline | Not found at the equivalent archive path; re-stage the exact dataset or record its replacement and hash |
| GPU allocation | Wave 1 used four exclusive 8×H100 nodes; Wave 2 used six | Must be provisioned on the current platform; reserve an equivalent uncontended topology, or document the intentional deviation |

The controllers enforce the drafter architecture, aiperf version, Humming
patches, backend attestation, and speculation activation. Check GPU occupancy
and obtain approval for the exclusive allocation before starting a ported
controller.

Wave 1 starts four serves: k=0, 1, 3, and 5. Each executes AA-style 1k/10k
inputs at concurrency 1 and 10. Wave 2 starts six serves: the k=0/k=3 pair
for each of `natural`, `load`, and `lowconc`. Only the `natural` pair answers
this note's real-prompt question: it sends ShareGPT requests at temperature
0.6 and 0, with request counts 40 at concurrency 1 and 100 at concurrency 10.

Before interpreting a re-run, retain these controller outputs:

```text
Wave 1: arm-*/{serve.log,spec-boot.log,spec-metrics.log,backend-attestation.json,
        greedy-probe.json,aa-sweep.log,metrics-pre-aa.txt,metrics-post-aa.txt}
Wave 2: arm-natural-k*/{serve.log,spec-boot.log,backend-attestation.json,
        natural/t06/conc_{1,10}/,metrics/natural-t06-c{1,10}-{pre,post}.txt}
Both:   actual-commit.txt, arm-provenance.txt, *-srun.log, controller-done.txt
```

Re-aggregate each new window with the two commands above, substituting the new
window paths. Compare each k=3 arm with its k=0 control from the same window;
do not use cross-window throughput deltas as a paired effect.

## Results

### Primary synthetic-versus-real observation

| Workload | Concurrency | Accepted length | Per-position acceptance | k=0 → k=3 tok/s/user | Decode speedup |
|---|---:|---:|---|---:|---:|
| Wave 1 synthetic, AA-style sweep | arm-wide | 2.450 | 0.70 / 0.46 / 0.29 | 137.5 → 236.0 (1k, conc-1) | 1.72× |
| Wave 2 real, ShareGPT | 1 | 2.473 | 0.690 / 0.459 / 0.324 | 137.9 → 249.8 | 1.81× |
| Wave 2 real, ShareGPT | 10 | 2.503 | 0.700 / 0.479 / 0.324 | 77.7 → 128.8 | 1.66× |

At concurrency 1, ShareGPT's accepted length was `2.473 - 2.450 = +0.023`
(about +0.9%) above the synthetic sweep's mean. The primary study records that
as **+1%, noise**, not as evidence that natural-language prompts draft better
than synthetic prompts.

The server-throughput figures point in the same direction but should likewise
not be treated as paired:

| Workload | k=0 → k=3 server tok/s | Speedup |
|---|---:|---:|
| Wave 1 synthetic, 1k input at conc-1 | 133.7 → 215.0 | 1.61× |
| Wave 2 real, ShareGPT at conc-1 | 132.7 → 222.3 | 1.68× |

### Separating output shape from prompt source

The tempting contrary observation was a greedy probe on eight hand-picked
English prompts, which measured 3.20–3.35 accepted tokens. It was not a
natural-real-prompt experiment: it used temperature 0 and forced continuation.
The report initially used it to infer a 2.25× real-traffic speedup, then
superseded that inference with the direct ShareGPT measurement (1.81×).

The controlled contrast is:

| Change from natural ShareGPT baseline at conc-1 | Accepted length | Effect |
|---|---:|---:|
| Natural ShareGPT, temp 0.6 | 2.473 | Baseline |
| Greedy sampling, temp 0 | 2.575 | +4% |
| Force an 8k output with `ignore_eos` | 3.286 | +33% |

The +33% increase comes from forcing the model to continue its own output past
the natural stopping point. It is an output-shape effect, not evidence that
real prompts themselves improve drafting.

### Evidence and reproduction

The primary authority is
[`docs/m3-specdec-eagle3.md`](m3-specdec-eagle3.md), especially its Wave 1
results, Wave 2 Phase A table, and factor decomposition. The pre-declared
methodology is in
[`M3_SPECDEC_EAGLE3_PLAN.md`](../M3_SPECDEC_EAGLE3_PLAN.md).

The former cluster is retired. Raw evidence is retained in a private archive
on `138.252.188.36`, not in this local checkout. Collaborators need an
owner-provided mount or export; this note intentionally does not publish the
owner-specific storage path.

```text
results/m3-specdec-eagle3/20260727T061506Z/
results/m3-specdec-eagle3/20260727T064934Z-wave2/
```

Use the re-aggregation commands in
[Re-aggregate the historical artifacts](#re-aggregate-the-historical-artifacts)
to write new JSON outside these preserved windows.

This local repository contains the documentation, launchers, and aggregators.
The migrated archive contains the raw aiperf artifacts and metric-counter
snapshots needed to recompute the acceptance values.

## Conclusions

1. The remembered finding is supported: synthetic random-token prompts and
   natural ShareGPT prompts had essentially the same documented EAGLE3
   acceptance length—2.450 versus 2.473 at the relevant conc-1 comparison.
   The primary report treats the approximately +1% difference as noise.
2. Similar acceptance does not imply identical end-to-end performance. The
   observed decode speedup was 1.72× in the synthetic 1k conc-1 cell and 1.81×
   for real ShareGPT at conc-1. Because the measurements are from separate
   windows and aggregate acceptance differently, this difference is
   directional context, not a causal prompt-source effect.
3. Do not cite the earlier 2.25× estimate based on the hand-picked greedy,
   forced-continuation probe. Wave 2 measured natural ShareGPT traffic directly
   and established 1.81× as the applicable conc-1 result.
4. In this study, output shape was the largest identified influence on
   acceptance (+33%), temperature was smaller (+4%), and prompt naturalness
   was negligible (+1%, noise). Later SPEED-Bench measurements also found
   content domain to be much more consequential than prompt length.
