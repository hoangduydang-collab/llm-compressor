# GLM-5.3 GPTQ full7 quality evaluation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated native-GPTQ full7 arm, fail-closed CPU preflight, exact whole-suite token accounting, and a three-way report against the corrected historical AWQ and PhalaCloud artifacts.

**Architecture:** The benchmark repository owns the model profile and lm-eval protocol. The llm-compressor repository owns Rancher rendering, candidate checkpoint gates, SGLang lifecycle, token-counter normalization, and the offline three-way comparison. A CPU-only staging job must pass before a separately approved TP8 GPU pod can launch.

**Tech Stack:** Bash, Python 3.11+, pytest, lm-evaluation-harness 0.4.10, SGLang 0.5.17, Kubernetes, PyYAML, safetensors.

## Global constraints

- Design source: `docs/superpowers/specs/2026-09-13-glm53-gptq-full7-quality-design.md`.
- Candidate checkpoint is exactly `/mnt/cephfs/hoangduy/results/glm53-ep-gptq-w4afp8/full-ep8/20260912t183612z/output/304b8051cfb2b260b61ce0cbe330e02a98e73639-gptq-W4AFP8/20260912-184239/checkpoint`.
- Historical AWQ result is exactly `full7-20260901t064327z`; never use the invalid AWQ arm under `full7-20260831t135418z`.
- Historical PhalaCloud result is exactly `full7-20260831t135418z`.
- Full7 means all seven full populations, seed 0, reasoning enabled, `GENERAL_MAX_GEN_TOKS=32768`, `GENERAL_NUM_CONCURRENT=16`, and no chat template on loglikelihood.
- The 32,768-token override is explicitly diagnostic historical compatibility, not a native-task or leaderboard protocol.
- Serve with SGLang 0.5.17, TP8, W4AFP8, FP8 KV, context 65,536, memory fraction 0.75, chunked prefill 2,048, no speculative decoding, and shared-expert fusion disabled.
- Retain exact whole-suite counter deltas and per-sample outputs. Never claim exact per-task or per-sample token usage.
- A quality drop greater than 0.02 is reported after all tasks finish; it never aborts the suite.
- Do not touch collaborator checkpoints, caches, jobs, or workspaces.
- Do not claim GPUs until `scripts/gpu-free.sh --verify` passes and the user approves the displayed free node.
- Keep unrelated local changes and `benchmarks/benchmarks.bundle` out of every commit.

---

### Task 1: Add the dedicated GPTQ benchmark profile

**Files:**
- Create in benchmarks repo: `configs/glm/glm-5.3-w4afp8-gptq.sh`
- Create in benchmarks repo: `quality/orchestrator/tests/test_glm53_gptq_profile.py`

**Interfaces:**
- Consumes: the existing corrected `configs/glm/glm-5.3-w4afp8-ours.sh` protocol.
- Produces: a profile named `glm-5.3-w4afp8-gptq` with candidate-specific identity and an operator-supplied `SERVED_TOKENIZER_REVISION`.

- [ ] **Step 1: Write the failing profile test**

```python
from pathlib import Path

from quality.orchestrator import config as CFG
from quality.orchestrator import plan as P


ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "configs/glm/glm-5.3-w4afp8-gptq.sh"
CHECKPOINT = (
    "/mnt/cephfs/hoangduy/results/glm53-ep-gptq-w4afp8/full-ep8/"
    "20260912t183612z/output/"
    "304b8051cfb2b260b61ce0cbe330e02a98e73639-gptq-W4AFP8/"
    "20260912-184239/checkpoint"
)
TASKS = [
    "gsm8k", "ifeval", "gpqa_diamond_cot_zeroshot", "mmlu",
    "arc_challenge", "hellaswag", "truthfulqa_mc2",
]


def test_gptq_profile_has_dedicated_identity(monkeypatch):
    monkeypatch.setenv("SERVED_TOKENIZER_REVISION", "a" * 64)
    profile = CFG.profile_from_config(str(PROFILE))
    assert profile["profile"] == "glm-5.3-w4afp8-gptq"
    assert profile["model_name"] == "glm-5.3-w4afp8-gptq"
    assert profile["model_path"] == CHECKPOINT
    assert profile["model_tokenizer"] == CHECKPOINT
    assert profile["served_tokenizer"].endswith("/tok/gptq")
    assert profile["served_tokenizer_revision"] == "a" * 64
    assert profile["quant_recipe"] == "inhouse-ep-gptq-w4afp8"


def test_gptq_profile_preserves_historical_full7_contract(monkeypatch):
    monkeypatch.setenv("SERVED_TOKENIZER_REVISION", "b" * 64)
    profile = CFG.profile_from_config(str(PROFILE))
    plan = P.plan_run(profile)
    assert plan.suite("general").tasks == TASKS
    assert profile["general_log_samples"] is True
    assert profile["general_max_gen_toks"] == 32768
    assert profile["general_num_concurrent"] == 16
    assert profile["general_client_timeout"] == 3600
    assert profile["general_chat_template_on_loglikelihood"] is False
```

- [ ] **Step 2: Run the tests and verify RED**

Run from `C:\Users\hoangduy.dang\AI lab\benchmarks`:

```powershell
python -m pytest quality/orchestrator/tests/test_glm53_gptq_profile.py -v
```

Expected: FAIL because `configs/glm/glm-5.3-w4afp8-gptq.sh` does not exist.

- [ ] **Step 3: Implement the derived profile**

The profile must snapshot candidate-specific operator overrides before sourcing
the AWQ profile, then replace identity only:

```bash
#!/usr/bin/env bash
# Native EP-GPTQ candidate. Protocol is inherited byte-for-byte from the
# corrected historical AWQ full7 profile; only artifact identity changes.
_GPTQ_DEFAULT="/mnt/cephfs/hoangduy/results/glm53-ep-gptq-w4afp8/full-ep8/20260912t183612z/output/304b8051cfb2b260b61ce0cbe330e02a98e73639-gptq-W4AFP8/20260912-184239/checkpoint"
_GPTQ_MODEL_PATH="${MODEL_PATH:-$_GPTQ_DEFAULT}"
_GPTQ_MODEL_TOKENIZER="${MODEL_TOKENIZER:-$_GPTQ_MODEL_PATH}"
_GPTQ_SERVED_REV="${SERVED_TOKENIZER_REVISION:-}"

source "$(dirname "${BASH_SOURCE[0]}")/glm-5.3-w4afp8-ours.sh"

MODEL_NAME="glm-5.3-w4afp8-gptq"
MODEL_PATH="$_GPTQ_MODEL_PATH"
MODEL_TOKENIZER="$_GPTQ_MODEL_TOKENIZER"
SERVED_TOKENIZER="/mnt/cephfs/hoangduy/results/glm53-quality-paired/tok/gptq"
SERVED_TOKENIZER_REVISION="$_GPTQ_SERVED_REV"
QUANT_SOURCE="offline-w4afp8-inhouse-ep-gptq"
QUANT_RECIPE="inhouse-ep-gptq-w4afp8"

unset _GPTQ_DEFAULT _GPTQ_MODEL_PATH _GPTQ_MODEL_TOKENIZER _GPTQ_SERVED_REV
```

- [ ] **Step 4: Run profile and existing budget/path tests**

```powershell
python -m pytest `
  quality/orchestrator/tests/test_glm53_gptq_profile.py `
  quality/orchestrator/tests/test_log_samples_and_budget_knobs.py `
  quality/general/tests/test_ifeval_generation_fidelity.py -v
```

Expected: PASS. The dry-run argv must include `--log_samples` and
`max_gen_toks=32768` only for generative invocations.

- [ ] **Step 5: Commit the benchmarks change**

```powershell
git add configs/glm/glm-5.3-w4afp8-gptq.sh quality/orchestrator/tests/test_glm53_gptq_profile.py
git commit -m "feat(glm53): add native GPTQ full7 profile"
```

Expected: one benchmarks commit; `benchmarks.bundle` remains untracked.

---

### Task 2: Add testable three-arm CPU preflight logic

**Files:**
- Create: `pipeline/glm53_quality_preflight.py`
- Create: `pipeline/tests/test_glm53_quality_preflight.py`
- Modify: `pipeline/k8s/stage-glm53-quality-eval.sh`

**Interfaces:**
- Produces: `tokenizer_report(artifacts: Mapping[str, Path]) -> dict` and
  `protocol_report(dry_runs: Mapping[str, str], identities: Iterable[str]) -> dict`.
- CLI:
  `python -m pipeline.glm53_quality_preflight tokenizers --arm name=path ... --out FILE`
  and
  `python -m pipeline.glm53_quality_preflight protocols --arm name=path ... --identity TEXT ... --out FILE`.
- Exit 0 means no blocking tokenizer/template or protocol differences; exit 1 means refusal.

- [ ] **Step 1: Write failing unit tests**

```python
import json
from pathlib import Path

from pipeline.glm53_quality_preflight import protocol_report, tokenizer_report


def _tok(root: Path, *, template="same", local=False):
    root.mkdir()
    (root / "tokenizer.json").write_text('{"model":"same"}')
    (root / "chat_template.jinja").write_text(template)
    (root / "tokenizer_config.json").write_text(json.dumps({
        "model_max_length": 65536,
        "is_local": local,
    }))


def test_tokenizer_report_allows_only_inert_config_differences(tmp_path):
    ours, phala, gptq = (tmp_path / name for name in ("ours", "phala", "gptq"))
    _tok(ours, local=True)
    _tok(phala)
    _tok(gptq, local=True)
    report = tokenizer_report({"ours": ours, "phala": phala, "gptq": gptq})
    assert report["ok"] is True
    assert report["blocking_differences"] == []
    assert len(report["digests"]["gptq"]) == 64


def test_tokenizer_report_refuses_candidate_template_change(tmp_path):
    ours, phala, gptq = (tmp_path / name for name in ("ours", "phala", "gptq"))
    _tok(ours)
    _tok(phala)
    _tok(gptq, template="different")
    report = tokenizer_report({"ours": ours, "phala": phala, "gptq": gptq})
    assert report["ok"] is False
    assert any("gptq:chat_template.jinja" in item
               for item in report["blocking_differences"])


def test_protocol_report_normalizes_identity_but_not_protocol():
    base = "profile : glm-5.3-w4afp8-ours\nconcurrency=16\nmax_gen_toks=32768\n"
    candidate = base.replace("ours", "gptq")
    assert protocol_report(
        {"ours": base, "gptq": candidate},
        ["glm-5.3-w4afp8-ours", "glm-5.3-w4afp8-gptq"],
    )["ok"] is True
    changed = candidate.replace("concurrency=16", "concurrency=8")
    assert protocol_report(
        {"ours": base, "gptq": changed},
        ["glm-5.3-w4afp8-ours", "glm-5.3-w4afp8-gptq"],
    )["ok"] is False
```

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest pipeline/tests/test_glm53_quality_preflight.py -v
```

Expected: collection FAIL because `pipeline.glm53_quality_preflight` does not exist.

- [ ] **Step 3: Implement the preflight module**

Implement these exact public functions and a two-subcommand CLI:

```python
INERT_TOKENIZER_CONFIG_KEYS = {
    "is_local", "local_files_only", "_name_or_path", "name_or_path",
    "tokenizer_file", "auto_map",
}
BLOCKING_FILES = ("tokenizer.json", "chat_template.jinja")


def tokenizer_report(artifacts: Mapping[str, Path]) -> dict:
    """Hash tokenizer-only trees and compare every non-ours arm with ours."""


def protocol_report(
    dry_runs: Mapping[str, str],
    identities: Iterable[str],
) -> dict:
    """Normalize declared identity strings and compare every arm with ours."""


def main(argv: list[str] | None = None) -> int:
    """Write sorted, indented JSON and return 0 only when report['ok'] is true."""
```

Requirements:

- Hash every file in each tokenizer-only directory for
  `SERVED_TOKENIZER_REVISION`.
- Compare `tokenizer.json` and `chat_template.jinja` byte-for-byte.
- Compare `tokenizer_config.json` semantically after removing only
  `INERT_TOKENIZER_CONFIG_KEYS`.
- Preserve `generation_config.json` differences as non-blocking evidence.
- Normalize model names, checkpoint paths, `/tok/<arm>` paths, and per-arm
  served-tokenizer digests in dry-run text; do not normalize concurrency,
  tasks, generation kwargs, few-shot values, or chat-template behavior.
- Include a bounded unified diff in every protocol refusal.

- [ ] **Step 4: Run tests and verify GREEN**

```powershell
python -m pytest pipeline/tests/test_glm53_quality_preflight.py -v
```

Expected: PASS.

- [ ] **Step 5: Extend the staging script**

Make `GPTQ` opt-in so historical two-arm staging remains unchanged:

```bash
GPTQ="${GPTQ:-}"
ARMS="ours phala"
[ -n "$GPTQ" ] && ARMS="$ARMS gptq"
```

Replace each hard-coded `for arm in ours phala` with `for arm in $ARMS`, add
`gptq) SRC="$GPTQ"` to source selection, and:

1. Build `/mnt/cephfs/hoangduy/results/glm53-quality-paired/tok/gptq`.
2. Run all three profile dry-runs, passing the computed GPTQ digest as
   `SERVED_TOKENIZER_REVISION`.
3. Invoke `pipeline.glm53_quality_preflight protocols` for protocol parity.
4. Invoke `pipeline.glm53_quality_preflight tokenizers` for tokenizer/template
   parity and write exactly `$OUT/tokenizer-parity.json`.
5. Run the existing exact native gate:

```bash
if [ -n "$GPTQ" ]; then
  note "step 0d: exact native GPTQ checkpoint verification"
  PYTHONPATH="$STAGE_ROOT" "$PY" -m pipeline.native_sglang_save "$GPTQ" \
    > "$OUT/gptq-native-validation.json" 2> "$OUT/gptq-native-validation.err"
  gate gptq_native_checkpoint $?
fi
```

When `GPTQ` is set, require `BENCHMARKS_REF` and `GPTQ_PROFILE_SHA256`, verify
the staged `configs/glm/glm-5.3-w4afp8-gptq.sh` SHA256 exactly, and write both
declared values plus the measured profile SHA256 to
`$OUT/gptq-profile-provenance.json`. A mismatch is a gating failure.

Do not remove the existing AWQ/Phala gates.

- [ ] **Step 6: Run focused and shell-syntax tests**

```powershell
python -m pytest pipeline/tests/test_glm53_quality_preflight.py pipeline/tests/test_native_sglang_save.py -q
& 'C:\Program Files\Git\bin\bash.exe' -n pipeline/k8s/stage-glm53-quality-eval.sh
```

Expected: tests PASS and Bash exits 0.

- [ ] **Step 7: Commit the preflight change**

```powershell
git add pipeline/glm53_quality_preflight.py pipeline/tests/test_glm53_quality_preflight.py pipeline/k8s/stage-glm53-quality-eval.sh
git commit -m "feat: gate GPTQ full7 candidate before GPU launch"
```

---

### Task 3: Render reproducible CPU staging and GPTQ GPU manifests

**Files:**
- Create: `pipeline/k8s/glm53-gptq-full7-stage.yaml.tmpl`
- Create: `pipeline/k8s/render_quality_stage.py`
- Create: `pipeline/tests/test_render_quality_stage.py`
- Modify: `pipeline/k8s/render_arm.py`
- Modify: `pipeline/k8s/glm53-quality-arm.yaml.tmpl`
- Modify: `pipeline/tests/test_render_arm.py`

**Interfaces:**
- CPU renderer:
  `render_quality_stage.main(argv: list[str] | None = None) -> int`.
- GPU renderer gains `--arm gptq`, `--context-length`, and
  `--served-tokenizer-revision`.
- CPU renderer also requires `--benchmarks-ref` and `--profile-sha256`.
- The GPU pod exports `CTX`, `SERVED_TOKENIZER_REVISION`,
  `FULL7_AWQ_RESULT`, and `FULL7_PHALA_RESULT`.

- [ ] **Step 1: Write failing renderer tests**

Add to `pipeline/tests/test_render_arm.py`:

```python
def test_render_gptq_full7_contract(tmp_path: Path):
    out = tmp_path / "gptq.yaml"
    rc = render_arm.main([
        "--arm", "gptq",
        "--model", "/mnt/cephfs/hoangduy/results/gptq/checkpoint",
        "--run-tag", "gptq-full7-t",
        "--ref", "deadbeef",
        "--out", str(out),
        "--reasoning", "reasoning",
        "--context-length", "65536",
        "--served-tokenizer-revision", "a" * 64,
    ])
    assert rc == 0
    doc = yaml.safe_load(out.read_text())
    env = {e["name"]: e.get("value") for e in doc["spec"]["containers"][0]["env"]
           if "value" in e}
    assert env["PROFILE"].endswith("glm-5.3-w4afp8-gptq.sh")
    assert env["CTX"] == "65536"
    assert env["SERVED_TOKENIZER_REVISION"] == "a" * 64
    assert "full7-20260901t064327z" in env["FULL7_AWQ_RESULT"]
    assert "full7-20260831t135418z" in env["FULL7_PHALA_RESULT"]
```

Create `pipeline/tests/test_render_quality_stage.py`:

```python
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline/k8s"))
import render_quality_stage  # noqa: E402


def test_stage_manifest_requests_no_gpu_and_pins_candidate(tmp_path):
    out = tmp_path / "stage.yaml"
    rc = render_quality_stage.main([
        "--checkpoint", "/mnt/cephfs/hoangduy/results/gptq/checkpoint",
        "--run-tag", "t",
        "--ref", "deadbeef",
        "--benchmarks-ref", "1" * 40,
        "--profile-sha256", "2" * 64,
        "--out", str(out),
    ])
    assert rc == 0
    doc = yaml.safe_load(out.read_text())
    container = doc["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container["env"] if "value" in e}
    assert env["GPTQ"].endswith("/checkpoint")
    assert env["BENCHMARKS_REF"] == "1" * 40
    assert env["GPTQ_PROFILE_SHA256"] == "2" * 64
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    assert "@@" not in out.read_text()
```

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest pipeline/tests/test_render_arm.py pipeline/tests/test_render_quality_stage.py -v
```

Expected: FAIL because `gptq` is not accepted and the staging renderer does not exist.

- [ ] **Step 3: Implement the CPU staging template and renderer**

The template must define a namespaced Job with:

```yaml
metadata:
  name: hd-stage-glm53-gptq-@@RUN_TAG@@
  namespace: evaluation
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 43200
```

The container must use `lmsysorg/sglang:v0.5.17`, request 8 CPUs and 32 GiB
memory, request no GPU, mount `model-cache-shared` at `/mnt/cephfs`, clone and
checkout `@@REF@@`, and run:

```bash
GPTQ="@@CHECKPOINT@@" bash pipeline/k8s/stage-glm53-quality-eval.sh
```

`render_quality_stage.py` must follow `render_arm.py`'s safeguards: require an
absolute `/mnt/` checkpoint, reject remaining placeholders, parse persisted
YAML, re-read `GPTQ`, validate a 40-hex benchmarks commit plus a 64-hex profile
digest, and print the persisted Job name/ref/checkpoint/benchmarks provenance.

- [ ] **Step 4: Extend the GPU renderer and template**

In `render_arm.py`:

```python
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AWQ_RESULT = "/mnt/cephfs/hoangduy/results/glm53-quality-paired/full7-20260901t064327z/results/glm-5.3-w4afp8-ours/sglang/quality/general.glm53-full7-20260901t064327z.json"
_PHALA_RESULT = "/mnt/cephfs/hoangduy/results/glm53-quality-paired/full7-20260831t135418z/results/glm-5.3-w4afp8-phala/sglang/quality/general.glm53-full7-20260831t135418z.json"
```

Add `gptq` to `--arm`, add the two CLI arguments, and refuse a GPTQ render
unless context is exactly `65536` and the tokenizer revision is 64 lowercase
hex. Persist empty comparison paths for non-GPTQ arms and the fixed paths for
GPTQ.

Add template environment entries and pass them into the arm script:

```yaml
- {name: CTX, value: "@@CTX@@"}
- {name: SERVED_TOKENIZER_REVISION, value: "@@SERVED_TOKENIZER_REVISION@@"}
- {name: FULL7_AWQ_RESULT, value: "@@FULL7_AWQ_RESULT@@"}
- {name: FULL7_PHALA_RESULT, value: "@@FULL7_PHALA_RESULT@@"}
```

- [ ] **Step 5: Run renderer tests**

```powershell
python -m pytest pipeline/tests/test_render_arm.py pipeline/tests/test_render_quality_stage.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit manifest rendering**

```powershell
git add pipeline/k8s/glm53-gptq-full7-stage.yaml.tmpl pipeline/k8s/render_quality_stage.py pipeline/tests/test_render_quality_stage.py pipeline/k8s/render_arm.py pipeline/k8s/glm53-quality-arm.yaml.tmpl pipeline/tests/test_render_arm.py
git commit -m "feat: render GPTQ full7 staging and arm pods"
```

---

### Task 4: Normalize exact whole-suite token counters

**Files:**
- Create: `pipeline/quality_token_usage.py`
- Create: `pipeline/tests/test_quality_token_usage.py`
- Modify: `pipeline/k8s/glm53_quality_arm.sh`
- Create: `pipeline/tests/test_glm53_quality_runner_contract.py`

**Interfaces:**
- Produces:
  `parse_snapshot(text: str) -> dict[tuple[str, str], float]` and
  `summarize(before: str, after: str) -> dict`.
- CLI:
  `python -m pipeline.quality_token_usage --before FILE --after FILE --out FILE`.
- Returns exit 0 when a summary is written, including when status is
  `untrusted`; malformed numeric input returns nonzero.

- [ ] **Step 1: Write failing token-accounting tests**

```python
from pipeline.quality_token_usage import summarize


BEFORE = """\
sglang:prompt_tokens_total{model_name="gptq"} 100
sglang:generation_tokens_total{model_name="gptq"} 40
sglang:cached_tokens_total{model_name="gptq"} 10
sglang:num_requests_total{model_name="gptq"} 4
sglang:num_aborted_requests_total{model_name="gptq"} 0
"""
AFTER = """\
sglang:prompt_tokens_total{model_name="gptq"} 250
sglang:generation_tokens_total{model_name="gptq"} 90
sglang:cached_tokens_total{model_name="gptq"} 35
sglang:num_requests_total{model_name="gptq"} 9
sglang:num_aborted_requests_total{model_name="gptq"} 1
"""


def test_summarize_exact_counter_deltas():
    report = summarize(BEFORE, AFTER)
    assert report["status"] == "trusted"
    assert report["totals"]["prompt_tokens"] == 150
    assert report["totals"]["generated_tokens"] == 50
    assert report["totals"]["cached_tokens"] == 25
    assert report["totals"]["requests"] == 5
    assert report["totals"]["aborted_requests"] == 1


def test_summarize_marks_missing_counter_untrusted():
    report = summarize(BEFORE, AFTER.replace(
        'sglang:cached_tokens_total{model_name="gptq"} 35\n', ""))
    assert report["status"] == "untrusted"
    assert report["missing_after"]


def test_summarize_marks_counter_reset_untrusted():
    report = summarize(BEFORE, AFTER.replace(
        'sglang:generation_tokens_total{model_name="gptq"} 90',
        'sglang:generation_tokens_total{model_name="gptq"} 2'))
    assert report["status"] == "untrusted"
    assert report["counter_resets"]
```

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest pipeline/tests/test_quality_token_usage.py -v
```

Expected: collection FAIL because `pipeline.quality_token_usage` does not exist.

- [ ] **Step 3: Implement token normalization**

Use `(metric_name, raw_label_set)` as the series key so no label value is
dropped or guessed. Define this exact logical mapping:

```python
COUNTERS = {
    "sglang:prompt_tokens_total": "prompt_tokens",
    "sglang:generation_tokens_total": "generated_tokens",
    "sglang:cached_tokens_total": "cached_tokens",
    "sglang:num_requests_total": "requests",
    "sglang:num_aborted_requests_total": "aborted_requests",
}
```

The JSON must include:

```python
{
    "schema_version": 1,
    "status": "trusted" | "untrusted",
    "totals": {...},
    "series": [
        {"metric": ..., "labels": ..., "before": ..., "after": ..., "delta": ...}
    ],
    "missing_before": [...],
    "missing_after": [...],
    "counter_resets": [...],
}
```

Refuse NaN/Infinity and malformed sample lines. A missing series or negative
delta is evidence of incomplete accounting, never zero.

- [ ] **Step 4: Pin runner behavior with a static contract test**

```python
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / "k8s/glm53_quality_arm.sh").read_text()


def test_runner_summarizes_metrics_after_general_suite():
    assert '"$BVENV/bin/python" -m pipeline.quality_token_usage' in SCRIPT
    assert "metrics-before.txt" in SCRIPT
    assert "metrics-after.txt" in SCRIPT
    assert "token-usage.json" in SCRIPT


def test_manifest_does_not_claim_chat_template_on_loglikelihood():
    assert '"chat_template_on_loglikelihood": False' in SCRIPT
    assert "MC tasks scored as loglikelihood with a chat template" not in SCRIPT


def test_reasoning_separation_is_a_gate():
    assert 'gate reasoning_shape "$reasoning_rc"' in SCRIPT
    assert 'reasoning_shape gate failed' in SCRIPT
```

- [ ] **Step 5: Verify the static test is RED**

```powershell
python -m pytest pipeline/tests/test_glm53_quality_runner_contract.py -v
```

Expected: FAIL because token summarization and corrected provenance are absent.

- [ ] **Step 6: Integrate the summary after `scrape_metrics after`**

In `glm53_quality_arm.sh`, add:

```bash
artifact_rc=0
if [ "$rc" = 0 ]; then
  PYTHONPATH="$REPO" "$BVENV/bin/python" -m pipeline.quality_token_usage \
    --before "$CLIENT/metrics-before.txt" \
    --after "$CLIENT/metrics-after.txt" \
    --out "$CLIENT/token-usage.json" \
    2>&1 | tee "$CLIENT/token-usage.log"
  token_rc=${PIPESTATUS[0]}
  gate token_usage "$token_rc"
  [ "$token_rc" = 0 ] || artifact_rc=1
fi
```

Also change the harness manifest to record
`"chat_template_on_loglikelihood": False`, `log_samples: true`, and the
historical compatibility note. Remove the stale sentence claiming MC tasks
used a chat template.

Turn the existing reasoning-separation observation into a capability gate. The
probe must require a non-empty final `content`, a `reasoning` or
`reasoning_content` field, no leaked `<think>`/`</think>` marker in final
content, and a successful HTTP/JSON response. Record `reasoning_shape=0` on
success; otherwise stop before the suite with `reasoning_shape gate failed`.

- [ ] **Step 7: Run focused tests and Bash syntax**

```powershell
python -m pytest pipeline/tests/test_quality_token_usage.py pipeline/tests/test_glm53_quality_runner_contract.py -v
& 'C:\Program Files\Git\bin\bash.exe' -n pipeline/k8s/glm53_quality_arm.sh
```

Expected: PASS.

- [ ] **Step 8: Commit token accounting**

```powershell
git add pipeline/quality_token_usage.py pipeline/tests/test_quality_token_usage.py pipeline/k8s/glm53_quality_arm.sh pipeline/tests/test_glm53_quality_runner_contract.py
git commit -m "feat: retain exact full7 token counter deltas"
```

---

### Task 5: Build and integrate the three-way quality comparison

**Files:**
- Create: `pipeline/compare_glm53_full7.py`
- Create: `pipeline/tests/test_compare_glm53_full7.py`
- Modify: `pipeline/k8s/glm53_quality_arm.sh`
- Modify: `pipeline/tests/test_glm53_quality_runner_contract.py`

**Interfaces:**
- Produces:
  `compare(candidate: dict, awq: dict, phala: dict, threshold: float = 0.02) -> dict`.
- CLI:
  `python -m pipeline.compare_glm53_full7 --candidate FILE --awq FILE --phala FILE --out FILE --threshold 0.02`.
- A valid diagnostic regression returns exit 0. Missing/mismatched evidence returns nonzero.

- [ ] **Step 1: Write failing comparison tests**

```python
import copy

import pytest

from pipeline.compare_glm53_full7 import TASKS, compare


def _doc(profile, value=0.8, n=100):
    return {
        "kind": "general",
        "profile": profile,
        "payload": {
            "tasks": {
                task: {
                    "metric": "acc",
                    "value": value,
                    "n": n,
                    "num_fewshot": 0,
                    "higher_is_better": True,
                }
                for task in TASKS
            }
        },
    }


def test_compare_reports_both_deltas_and_flags_without_failure():
    candidate = _doc("gptq", 0.77)
    report = compare(candidate, _doc("awq", 0.80), _doc("phala", 0.79))
    assert report["conclusion"] == "diagnostic_regression"
    assert report["tasks"]["gsm8k"]["delta_vs_awq"] == pytest.approx(-0.03)
    assert report["tasks"]["gsm8k"]["delta_vs_phala"] == pytest.approx(-0.02)
    assert report["tasks"]["gsm8k"]["regression_vs_awq"] is True
    assert report["tasks"]["gsm8k"]["regression_vs_phala"] is False


def test_compare_refuses_population_mismatch():
    candidate = _doc("gptq")
    awq = _doc("awq")
    awq["payload"]["tasks"]["mmlu"]["n"] = 99
    with pytest.raises(ValueError, match="population"):
        compare(candidate, awq, _doc("phala"))


def test_compare_refuses_metric_mismatch():
    candidate = _doc("gptq")
    phala = _doc("phala")
    phala["payload"]["tasks"]["hellaswag"]["metric"] = "acc_norm"
    with pytest.raises(ValueError, match="metric"):
        compare(candidate, _doc("awq"), phala)


def test_compare_requires_all_tasks():
    candidate = _doc("gptq")
    del candidate["payload"]["tasks"]["ifeval"]
    with pytest.raises(ValueError, match="ifeval"):
        compare(candidate, _doc("awq"), _doc("phala"))
```

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest pipeline/tests/test_compare_glm53_full7.py -v
```

Expected: collection FAIL because `pipeline.compare_glm53_full7` does not exist.

- [ ] **Step 3: Implement the comparator**

Define:

```python
TASKS = (
    "gsm8k", "ifeval", "gpqa_diamond_cot_zeroshot", "mmlu",
    "arc_challenge", "hellaswag", "truthfulqa_mc2",
)
```

For every task, require all three rows, identical metric/filter/few-shot
identity, non-null equal populations, numeric finite values, and
`higher_is_better is not False`. Emit source path plus SHA256 in the CLI
artifact. Round deltas to 12 decimal places before applying
`delta < -threshold`; exactly `-0.02` is not greater than two points and is not
flagged.

Output:

```python
{
    "schema_version": 1,
    "threshold": 0.02,
    "conclusion": "no_large_regression" | "diagnostic_regression",
    "sources": {...},
    "tasks": {
        task: {
            "metric": ...,
            "n": ...,
            "gptq": ...,
            "awq": ...,
            "phala": ...,
            "delta_vs_awq": ...,
            "delta_vs_phala": ...,
            "regression_vs_awq": ...,
            "regression_vs_phala": ...,
        }
    },
    "flagged_tasks": [...],
}
```

- [ ] **Step 4: Run comparison tests**

```powershell
python -m pytest pipeline/tests/test_compare_glm53_full7.py -v
```

Expected: PASS.

- [ ] **Step 5: Add runner finalization**

After token normalization, only for the GPTQ arm:

```bash
if [ "$rc" = 0 ] && [ "$ARM" = gptq ]; then
  CANDIDATE_RESULT="$ROOT/results/glm-5.3-w4afp8-gptq/sglang/quality/general.$RUN_ID.json"
  PYTHONPATH="$REPO" "$BVENV/bin/python" -m pipeline.compare_glm53_full7 \
    --candidate "$CANDIDATE_RESULT" \
    --awq "$FULL7_AWQ_RESULT" \
    --phala "$FULL7_PHALA_RESULT" \
    --out "$CLIENT/full7-three-way.json" \
    --threshold 0.02 \
    2>&1 | tee "$CLIENT/full7-three-way.log"
  compare_rc=${PIPESTATUS[0]}
  gate full7_three_way "$compare_rc"
  [ "$compare_rc" = 0 ] || artifact_rc=1
fi
```

Change final exit to succeed only when both the suite and artifact assembly
succeed:

```bash
[ "$rc" = 0 ] && [ "$aa_rc" = 0 ] && [ "$artifact_rc" = 0 ]
```

Extend `test_glm53_quality_runner_contract.py` to assert the candidate path,
both fixed baseline environment variables, and `--threshold 0.02`.

- [ ] **Step 6: Run focused integration tests**

```powershell
python -m pytest `
  pipeline/tests/test_compare_glm53_full7.py `
  pipeline/tests/test_quality_token_usage.py `
  pipeline/tests/test_glm53_quality_runner_contract.py `
  pipeline/tests/test_render_arm.py -v
& 'C:\Program Files\Git\bin\bash.exe' -n pipeline/k8s/glm53_quality_arm.sh
```

Expected: PASS.

- [ ] **Step 7: Commit comparison finalization**

```powershell
git add pipeline/compare_glm53_full7.py pipeline/tests/test_compare_glm53_full7.py pipeline/k8s/glm53_quality_arm.sh pipeline/tests/test_glm53_quality_runner_contract.py
git commit -m "feat: compare GPTQ full7 with historical baselines"
```

---

### Task 6: Verify, publish code, stage, and launch

**Files:**
- Modify if needed: `docs/glm53-w4afp8-rancher-evaluation.md`
- Generate locally, do not commit: `.k8s-rendered/glm53-gptq-full7-stage-<tag>.yaml`
- Generate locally, do not commit: `.k8s-rendered/glm53-gptq-full7-<tag>.yaml`

**Interfaces:**
- Consumes: clean commits from Tasks 1–5.
- Produces: one CPU staging Job, then one separately approved 8-GPU GPTQ Pod,
  followed by `token-usage.json` and `full7-three-way.json`.

- [ ] **Step 1: Run the full local verification set**

In llm-compressor:

```powershell
python -m pytest `
  pipeline/tests/test_glm53_quality_preflight.py `
  pipeline/tests/test_render_quality_stage.py `
  pipeline/tests/test_render_arm.py `
  pipeline/tests/test_quality_token_usage.py `
  pipeline/tests/test_compare_glm53_full7.py `
  pipeline/tests/test_glm53_quality_runner_contract.py `
  pipeline/tests/test_native_sglang_save.py -q
& 'C:\Program Files\Git\bin\bash.exe' -n pipeline/k8s/stage-glm53-quality-eval.sh
& 'C:\Program Files\Git\bin\bash.exe' -n pipeline/k8s/glm53_quality_arm.sh
git diff --check
```

In benchmarks:

```powershell
python -m pytest `
  quality/orchestrator/tests/test_glm53_gptq_profile.py `
  quality/orchestrator/tests/test_log_samples_and_budget_knobs.py `
  quality/general/tests/test_ifeval_generation_fidelity.py `
  quality/general/tests/test_chat_template_per_path.py -q
git diff --check
```

Expected: every test PASS, both Bash checks exit 0, and both diff checks are clean.

- [ ] **Step 2: Record both exact source commits**

```powershell
git -C "C:\Users\hoangduy.dang\AI lab\llm-compressor" rev-parse HEAD
git -C "C:\Users\hoangduy.dang\AI lab\benchmarks" rev-parse HEAD
git -C "C:\Users\hoangduy.dang\AI lab\llm-compressor" status --short
git -C "C:\Users\hoangduy.dang\AI lab\benchmarks" status --short
```

Expected: only pre-existing unrelated files appear. Record both hashes in the
launch handoff.

- [ ] **Step 3: Ask before pushing**

Ask the user for explicit permission to push both reviewable branches. The
Rancher pod clones llm-compressor by commit; do not launch against an unpushed
commit. Do not force-push.

- [ ] **Step 4: Ask before cluster writes, then stage the benchmarks tree**

After approval, archive the local benchmarks working tree while excluding
`.git`, `benchmarks.bundle`, caches, and bytecode; copy it through a new owned
CPU staging pod into `/mnt/cephfs/hoangduy/projects/benchmarks`. Record the
benchmarks commit hash and profile SHA256 alongside `stage-gates.txt`.

Do not remove or overwrite collaborator paths. Updating
`/mnt/cephfs/hoangduy/projects/benchmarks` is an owned-tree write and still
requires explicit approval immediately before execution.

- [ ] **Step 5: Render and run CPU-only staging**

```powershell
$tag = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ').ToLower()
$ref = git rev-parse HEAD
$benchRef = git -C '..\benchmarks' rev-parse HEAD
$profileSha = (Get-FileHash '..\benchmarks\configs\glm\glm-5.3-w4afp8-gptq.sh' -Algorithm SHA256).Hash.ToLower()
New-Item -ItemType Directory -Force .k8s-rendered | Out-Null
python pipeline/k8s/render_quality_stage.py `
  --checkpoint '/mnt/cephfs/hoangduy/results/glm53-ep-gptq-w4afp8/full-ep8/20260912t183612z/output/304b8051cfb2b260b61ce0cbe330e02a98e73639-gptq-W4AFP8/20260912-184239/checkpoint' `
  --run-tag $tag --ref $ref --benchmarks-ref $benchRef `
  --profile-sha256 $profileSha `
  --out ".k8s-rendered/glm53-gptq-full7-stage-$tag.yaml"
kubectl apply -f ".k8s-rendered/glm53-gptq-full7-stage-$tag.yaml"
```

This Job requests no GPU. Wait for completion, then require every gate,
including `gptq_native_checkpoint=0`, `profile_renders=0`, `arm_parity=0`, and
`template_parity=0`.

- [ ] **Step 6: Read the staged GPTQ tokenizer revision**

Read the exact 64-hex GPTQ digest from:

```text
/mnt/cephfs/hoangduy/results/glm53-quality-paired/tokenizer-parity.json
```

Do not infer or copy the AWQ digest.

- [ ] **Step 7: Run authoritative Rancher availability**

From workspace root:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' scripts/gpu-free.sh --verify
```

Require agreement between the six-namespace report and independent Rancher
accounting. Report total free GPUs, per-node free GPUs, fully free nodes, and
largest schedulable pod. If methods disagree or the command fails, stop.

- [ ] **Step 8: Ask the user to approve the specific 8-GPU claim**

Show the exact selected fully free node and the proposed pod name. Do not apply
the GPU manifest before approval.

- [ ] **Step 9: Render and launch the GPTQ full7 arm**

After approval:

```powershell
$runTag = "gptq-full7-" + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ').ToLower()
$ref = git rev-parse HEAD
python pipeline/k8s/render_arm.py `
  --arm gptq `
  --model '/mnt/cephfs/hoangduy/results/glm53-ep-gptq-w4afp8/full-ep8/20260912t183612z/output/304b8051cfb2b260b61ce0cbe330e02a98e73639-gptq-W4AFP8/20260912-184239/checkpoint' `
  --run-tag $runTag --ref $ref `
  --reasoning reasoning `
  --context-length 65536 `
  --served-tokenizer-revision $gptqTokenizerRevision `
  --node $approvedNode `
  --out ".k8s-rendered/glm53-gptq-full7-$runTag.yaml"
kubectl apply -f ".k8s-rendered/glm53-gptq-full7-$runTag.yaml"
```

Expected persisted manifest: 8 GPUs, approved node, `arm=gptq`, no `LIMIT`, no
AA GPQA, reasoning enabled, context 65,536, and `HOLD_AFTER` empty.

- [ ] **Step 10: Monitor to terminal state**

Monitor the owned pod and durable run log without touching other workloads.
Report transitions for load, health, throughput, loglikelihood gate, full7
suite, token normalization, and three-way comparison. Treat Rancher API
timeouts as control-plane uncertainty, not model failure.

- [ ] **Step 11: Validate final artifacts**

Require:

```text
client-gptq/gates.txt
client-gptq/harness_manifest.json
client-gptq/metrics-before.txt
client-gptq/metrics-after.txt
client-gptq/token-usage.json
client-gptq/full7-three-way.json
results/glm-5.3-w4afp8-gptq/sglang/quality/general.<run-id>.json
```

Check seven full-population task rows, retained sample paths, token accounting
status, both historical source hashes, all deltas, and any >2-point flags.
Report token accounting as `untrusted` if the artifact says so; never repair it
by assumption.

- [ ] **Step 12: Write the measured status update and commit it**

Add a dated status document containing the exact run ID, commits, checkpoint,
serve contract, elapsed time, seven scores, both deltas, flags, and aggregate
token usage. Clearly label the 32,768-token override as historical
compatibility and the scores as internal/non-leaderboard.

```powershell
git add docs/status/<dated-glm53-gptq-full7-result>.md
git commit -m "docs: record GLM-5.3 GPTQ full7 quality result"
```

Ask before pushing the result commit.
