# Sep 16

## Duy

### What this note is

GLM-5.3 W4AFP8 on AA-LCR v1.1 under **Z.ai's recommended sampling**
(temperature 1.0 / top_p 0.95) — the lab-override branch of AA's own sampling
rule, and therefore the AA-faithful setting for this model. Run also carried a
doubled output cap (262,144), which turned out to be unnecessary.

Supersedes the provenance labels in
[`2026-09-14-glm53-aa-lcr-v11.md`](2026-09-14-glm53-aa-lcr-v11.md) (see its
correction block). Still a public-methodology reproduction, not an official AA
leaderboard score.

### Key result

**Pass@1: 77.67% (233 / 300)**, up from **71.33% (214 / 300)** at temperature
0.6 / top_p 1.0. AA's published GLM-5.3 (max) v1.1 figure is **80%**.

| | 0.6 / 1.0 @ 131,072 | 1.0 / 0.95 @ 262,144 |
|---|---:|---:|
| Pass@1 | 214/300 = 71.33% | **233/300 = 77.67%** |
| Question-macro accuracy | 71.33% | 77.67% |
| Per-repeat | 72 / 74 / 68 | 75 / 80 / 78 |
| Truncated (`finish_reason=length`) | **32** | **2** |
| Non-truncated accuracy | 213/268 = 79.48% | 233/298 = 78.19% |
| Total generated tokens | 5,078,829 | **2,060,660** |
| Wall clock | ~2 Jobs, resumed | **2 h 52 m** |
| Candidate errors / judge failures | 0 / 0 | 0 / 0 |

Run id `glm53-w4afp8-aa-lcr-v11-full-t1p95-2x-r1`, fingerprint
`52199a49b50ac8e529abe59337dba44016252e9cb2569c31d9f4e88b458c3f67`, code
revision `5d41244e67c08ca12046d332d790795d020298b2`.

### The doubled cap contributed nothing

Only **2 of 300** completions exceeded 131,072: `q16 r0` and `q26 r1`. Both ran
to the full 262,144, truncated, and scored INCORRECT. The extra 131k of budget
converted **zero** items and spent ~524k tokens doing it. Both also generated
past Z.ai's own disclosed 128K output maximum, i.e. outside what the creator
claims the model supports.

The gain is entirely a sampling effect. Where the completion-length mass sits:

| Completion length | 0.6 / 1.0 | 1.0 / 0.95 |
|---|---:|---:|
| ≤ 8k | 240 | 235 |
| 8k – 32k | 28 | 61 |
| 32k – 64k | 0 | 2 |
| 64k – 131,072 | **32** (all capped) | **0** |
| > 131,072 | 0 | 2 |

At temperature 0.6 with an untruncated tail (top_p 1.0), 32 traces ran away into
the 64k–131k band and every one hit the wall with an empty answer. At 1.0 / 0.95
that band is empty. Total generation fell 59% *despite* a 2× higher cap.

Arithmetic of the gain: the 1.0/0.95 run has 30 more naturally-finishing items
(298 vs 268); they scored at roughly the population rate, netting +19 correct.
Those 30 all finished **under** 131,072, so they would have finished at the
original cap too.

Note that non-truncated accuracy went slightly *down* (79.48% → 78.19%). On
items that finish either way, the sampling change is not an improvement. The
entire headline movement is about not running away.

### Consequence

**A long tail on this benchmark is a sampling problem, not a budget problem.**
Prefer 131,072 and do not raise the cap again without new evidence. The next run
worth doing is 1.0 / 0.95 at 131,072 — AA's policy on both axes — which this
data predicts lands at ~233/300 ≈ 77.7%, since only the two out-of-spec traces
behave differently and both are wrong either way. `hd-aa-lcr-v11-full-t1.yaml`
already encodes it.

### Runner changes behind this run

`aa_lcr_v11.py` at `5d41244e`, then corrected the same day:

- `--candidate-max-tokens` added (default stays 131,072, pinned into the run
  contract). Editing the constant instead would have let a raised-cap run claim
  to be a public-methodology reproduction, since
  `uses_public_methodology_sampling` compares against that constant.
- `--candidate-concurrency` relaxed from a pinned 2 to a 1–2 **admission
  ceiling**, plus `_AdmissionGate`: it holds at one in-flight attempt while any
  attempt has run past `--candidate-long-attempt-seconds` (default 900). AA-LCR
  prompts are 76,820–114,611 tokens, so two worst-case 262,144-cap requests need
  ~753,510 KV tokens against a 598,848-token pool; two 131,072-cap requests fit.
  SGLang 0.5.17 handles pool exhaustion with `retract_all`, which releases
  **every** request in the batch, so the gate exists to keep that from arising.
  It held at 1 exactly **twice** in this run — matching the two long attempts —
  and never fired across the other 298.
- Fail-closed budget check in `build_run_contract`: rejects a cap whose
  worst-case single trace cannot fit the serve's `max_total_num_tokens`, before
  any endpoint traffic. Prompt lengths there are cl100k counts; measured against
  the serve, GLM tokenizes ~1% higher (95,407 actual vs 94,494 cl100k), so treat
  it as a floor rather than a certificate.
- Sampling defaults moved to Z.ai's recommended 1.0 / 0.95 with
  `AA_GENERIC_TEMPERATURE` / `AA_GENERIC_TOP_P` retained for the ablation, and
  `summary.json` now records `candidate_sampling.sampling_provenance` and
  `.max_tokens_provenance` so a bundle states its own derivation.

### Artifacts

| | Path |
|---|---|
| Published bundle | `/mnt/cephfs/hoangduy/results/glm53-aa-lcr-v11/glm53-w4afp8-aa-lcr-v11-full-t1p95-2x-r1/` |
| Checkpoint sqlite | `/mnt/cephfs/hoangduy/aa-lcr-v11-work/glm53-w4afp8-aa-lcr-v11-full-t1p95-2x-r1/run.sqlite` |
| Runbook | [`docs/runbooks/aa-lcr-v11.md`](../runbooks/aa-lcr-v11.md) |

Bundle files: `summary.json`, `report.md`, `candidates.jsonl`,
`judgments.jsonl`, `run-manifest.json`, `files.sha256`.

One operational note: the Job's stdout carried only the admission-gate line. The
manifests use `exec > >(tee -a "$LOG") 2>&1`, and the container can exit before
that process substitution flushes, so the completion JSON was lost from both
`kubectl logs` and the CephFS runlog. The Job still exited 0 and the bundle is
complete. Worth fixing with a `wait` on the tee PID if the runlog is ever the
only record.
