# MiniMax-M3 Native Humming W4A8 Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed, explicitly selected Humming W4A8 serving path for the existing MiniMax-M3 GPTQ checkpoint so it can be qualified against the current CUTLASS W4A8 path without changing the checkpoint or benchmark.

**Architecture:** Keep CUTLASS as the default in the existing HTTP launcher. The `humming` selector adds vLLM's structured `--quantization humming` option, verifies the exact checkpoint/runtime contract, installs only the missing Humming activation-admission patch, and requires positive runtime backend attestation before any benchmark starts. Reuse vLLM 0.24.0 and Humming 0.1.10 loading, repacking, dynamic E4M3 quantization, and indexed MoE kernels; add no CUDA arithmetic or benchmark implementation.

**Tech Stack:** Python 3.11, pytest, Bash, vLLM 0.24.0, humming-kernels 0.1.10, compressed-tensors GPTQ, H100/SM90, Slurm `srun`, existing aiperf benchmark repository.

## Global Constraints

- Read and follow `AGENTS.md` and `PLANNER_EXECUTOR_PROTOCOL.md` before each implementation or executor phase.
- Search the checked-out source and pinned upstream source before adding any new mechanism.
- Preserve the user's untracked `AGENTS.md`; never stage or modify it.
- Work on `duy-branch`. The planner commits and pushes each completed implementation task.
- Keep `M3_W4A8_BACKEND=cutlass` as the default and current behavior.
- Reject unsupported or ambiguous configurations. Never silently fall back from a requested Humming arm to CUTLASS, Marlin, eager mode, or unquantized experts.
- Do not modify checkpoint files, quantization recipes, benchmark workloads, benchmark scoring, or the benchmark repository.
- Do not write or modify CUDA kernel arithmetic in this plan.
- Use Humming's default indexed MoE GEMM. Reject `grouped_contiguous` during first qualification.
- Keep `VLLM_HUMMING_USE_F16_ACCUM=0` for first qualification.
- Require Humming's installed files to match its wheel `RECORD`; reject the
  separate `LLMC_NVFP4_W4A8_G16_V1` source overlay or any other local Humming
  mutation.
- Use a dedicated `cache-m3-gptq-w4a8-v1` Humming JIT cache namespace so the
  direct GPTQ experiment cannot reuse the NVFP4 specialization's cache.
- All local implementation tests are CPU-only. The planner does not launch GPU work.
- Cluster execution requires a committed `READY_FOR_EXECUTOR` packet, a persistent controller, and top-level `srun`; never use `sbatch`.
- The first executor packet stops after TP8/EP load, backend attestation, correctness smoke, and graphs-on stability. The existing performance benchmark is authorized only by a later packet after the planner accepts the returned qualification evidence.

---

## File and Interface Map

| File | Responsibility after this plan | Public or test-facing interface |
| --- | --- | --- |
| `PROJECT_GOALS.md` | Durable Goal 7 and current-session focus | Goal status text |
| `docs/superpowers/specs/2026-07-25-minimax-m3-humming-native-w4a8-backend-design.md` | Approved design record | Status marker |
| `pipeline/slurm/patch_vllm_m3_serve.py` | Existing M3 patches plus optional Humming activation-admission patch | `--humming`, `--check --humming`, `_patch_humming_supports_activation`, `ensure_vllm_m3_humming_patch` |
| `pipeline/m3_humming_w4a8.py` | Deep, fail-closed checkpoint/runtime preflight and server-log attestation | `evaluate_preflight`, `classify_backend_log`, `preflight`, `attest` CLI subcommands |
| `pipeline/slurm/run_vllm_http_serve_smoke.sh` | Structured backend selector and Humming pre-launch gate | `M3_W4A8_BACKEND=cutlass\|humming`, `PRINT_EFFECTIVE_CONFIG=1` |
| `pipeline/slurm/perf_eval_arm.sh` | Existing benchmark arm with a Humming-only post-readiness attestation gate | `M3_W4A8_BACKEND` inherited from controller |
| `pipeline/tests/test_patch_vllm_m3_serve.py` | Pure patch-function regression coverage | pytest |
| `pipeline/tests/test_m3_humming_w4a8.py` | Pure preflight and attestation coverage | pytest |
| `pipeline/tests/test_run_vllm_http_serve_smoke.py` | CPU dry-run launcher contract | pytest + Bash |
| `pipeline/tests/test_perf_eval_humming_contract.py` | Guard that attestation precedes the existing benchmark | pytest |
| `M3_HUMMING_W4A8_HANDOFF.md` | One active, protocol-complete qualification packet | `READY_FOR_EXECUTOR` packet |

The new Python helper is intentionally small and deep. Shell remains responsible
for environment setup and process launch; Python owns nested checkpoint metadata,
runtime version discovery, and log classification.

## Exact Runtime Contracts

### Backend selector

```text
M3_W4A8_BACKEND=cutlass   # default, adds no quantization CLI argument
M3_W4A8_BACKEND=humming   # adds exactly: --quantization humming
```

Any other value exits before site-packages patching, checkpoint mutation, GPU
preflight, or server launch. `EXTRA_VLLM_ARGS` may not contain
`--quantization` or `--quantization=...`; this prevents two competing selectors.

### Humming checkpoint contract

The preflight accepts exactly one `config_groups` entry targeting `Linear` with:

```json
{
  "quant_method": "compressed-tensors",
  "format": "pack-quantized",
  "weights": {
    "num_bits": 4,
    "type": "int",
    "symmetric": true,
    "strategy": "group",
    "group_size": 128,
    "dynamic": false
  },
  "input_activations": {
    "num_bits": 8,
    "type": "float",
    "symmetric": true,
    "strategy": "token",
    "group_size": null,
    "dynamic": true
  }
}
```

The full serving-ABI report from `pipeline.m3_serve_abi.analyze_checkpoint` must
also be valid, the ignore list must be non-empty, and the routed-expert inventory
must contain packed quantized modules. Multiple `Linear` groups are rejected as
ambiguous even when one appears compatible.

### Runtime contract

First qualification requires:

```text
vllm.__version__ == 0.24.0
importlib.metadata.version("humming-kernels") == 0.1.10
torch.cuda.get_device_capability() == (9, 0)
VLLM_HUMMING_USE_F16_ACCUM in {"", "0", "false", "False"}
VLLM_HUMMING_MOE_GEMM_TYPE in {"", "indexed"}
Humming package files match their installed wheel RECORD hashes
HUMMING_CACHE_DIR basename == cache-m3-gptq-w4a8-v1
all normal M3 patches installed
Humming activation-admission patch installed
```

Dry-run launcher tests do not import vLLM, Humming, or torch and do not perform
runtime preflight.

### Backend attestation contract

Attestation consumes the preserved preflight JSON and server log. It requires:

- preflight `valid=true` and `backend=humming`;
- a vLLM startup argument/config marker that resolves quantization to `humming`;
- exactly one supported Humming MoE selection:
  `Using indexed gemm for humming moe`;
- no `Using grouped_contiguous gemm for humming moe` in first qualification;
- no `Using CUTLASS W4A8 MoE backend`;
- no case-insensitive `Using Marlin`;
- no `UnquantizedFusedMoEMethod`.

It records, but does not require, Humming compile/cache markers. An absent or
ambiguous positive marker is a failed attestation.

---

### Task 1: Record the approved goal and design status

**Files:**

- Modify: `PROJECT_GOALS.md`
- Modify: `docs/superpowers/specs/2026-07-25-minimax-m3-humming-native-w4a8-backend-design.md`

- [ ] **Step 1: Update the durable goal record**

Add this goal after Goal 6:

```markdown
7. **Native Humming W4A8 serving on Hopper** — *Active: implementation planned.*
   Qualify Humming's existing GPTQ W4A8 path as a possible faster backend for
   the in-house MiniMax-M3 checkpoint on H100. Preserve packed INT4 group-128
   weights, dynamic per-token E4M3 activations, TP8 plus expert parallelism,
   graphs-on serving, and the existing production benchmark contract. CUTLASS
   remains the default until Humming passes fail-closed backend attestation,
   correctness, stability, and the paired performance decision. See
   `docs/superpowers/specs/2026-07-25-minimax-m3-humming-native-w4a8-backend-design.md`.
```

Set `Last reviewed` to `2026-07-25`. Change `Current session focus` to Goal 7,
while stating that Goals 1 and 2 remain active long-term work.

- [ ] **Step 2: Mark the design approved**

Replace its draft status with:

```markdown
- Status: APPROVED — 2026-07-25
```

- [ ] **Step 3: Verify documentation consistency**

Run:

```bash
rg -n "Native Humming W4A8|Status: APPROVED|Current session focus" \
  PROJECT_GOALS.md \
  docs/superpowers/specs/2026-07-25-minimax-m3-humming-native-w4a8-backend-design.md
```

Expected: Goal 7, the approved status, and the Goal 7 session focus are all
present; no draft status remains.

- [ ] **Step 4: Commit and push**

```bash
git add PROJECT_GOALS.md \
  docs/superpowers/specs/2026-07-25-minimax-m3-humming-native-w4a8-backend-design.md
git commit -m "docs: activate Humming W4A8 backend goal"
git push origin duy-branch
```

Expected: only the two documentation files are committed; `AGENTS.md` remains
untracked.

---

### Task 2: Add the optional Humming activation-admission patch

**Files:**

- Modify: `pipeline/slurm/patch_vllm_m3_serve.py`
- Modify: `pipeline/tests/test_patch_vllm_m3_serve.py`

- [ ] **Step 1: Write the failing patch-function tests**

Extend the test import:

```python
from pipeline.slurm.patch_vllm_m3_serve import (
    _BOUNDARY_BLOCK,
    _LOAD_AUDIT_BLOCK,
    _PROBE_BLOCK,
    _patch_append_load_audit,
    _patch_humming_supports_activation,
)
```

Add a minimal target fixture anchored to `HummingExpertsBase`:

```python
HUMMING_EXPERTS_SOURCE = """
class HummingExpertsBase(FusedMoEExperts):
    @classmethod
    def _supports_activation(cls, activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
        ]
"""
```

Add tests that assert:

1. the function inserts
   `MoEActivation.SWIGLUOAI_UNINTERLEAVE,` inside this class's list;
2. applying it twice is idempotent;
3. a different class with the enum already present does not cause a false
   "already patched" result;
4. a changed or missing Humming method returns `found=False`;
5. the output still compiles when prefixed with minimal enum/base-class stubs.

- [ ] **Step 2: Run the tests and observe the expected failure**

```bash
python -m pytest pipeline/tests/test_patch_vllm_m3_serve.py -q
```

Expected: collection fails because `_patch_humming_supports_activation` does not
exist.

- [ ] **Step 3: Implement a class-scoped, idempotent patch**

Add:

```python
_HUMMING_SWIGLU_MARK = (
    "llmc M3 Humming W4A8 SWIGLUOAI_UNINTERLEAVE admission patch"
)


def _patch_humming_supports_activation(
    text: str,
) -> tuple[str, bool, bool]:
    """Admit M3's activation only in HummingExpertsBase."""
```

Implementation rules:

- find `class HummingExpertsBase`;
- bound the search to that class body, ending at the next top-level `class`;
- find its `_supports_activation` list;
- insert the enum immediately after `MoEActivation.SWIGLUOAI,`;
- append the distinct marker as an inline comment on the inserted line;
- report already patched only when the enum is inside that exact method;
- return `(text, False, False)` for every changed upstream layout.

Do not copy the activation formula or the clamp constants into this patch.

- [ ] **Step 4: Add the optional patch target and programmatic interface**

Add:

```python
def _humming_patch_targets(
    vllm_dir: Path,
) -> list[tuple[str, Path, object]]:
    return [
        (
            "Humming W4A8 SWIGLU support",
            vllm_dir
            / "model_executor/layers/fused_moe/experts/fused_humming_moe.py",
            _patch_humming_supports_activation,
        )
    ]


def ensure_vllm_m3_humming_patch(*, apply: bool = True) -> None:
    """Apply or verify the optional Humming activation-admission patch."""
```

`ensure_vllm_m3_humming_patch` must raise `RuntimeError` for a missing file,
changed layout, or unpatched check. Do not add the Humming file to
`_patch_targets`; normal CUTLASS-only patching must remain usable when Humming is
absent.

- [ ] **Step 5: Add `--humming` CLI behavior**

Add:

```python
ap.add_argument(
    "--humming",
    action="store_true",
    help="also apply/check the Humming W4A8 activation-admission patch",
)
```

When selected, extend the current target list with `_humming_patch_targets`.
Without it, the current target list and exit behavior must be byte-for-byte
equivalent. `--check --humming` exits nonzero if either normal M3 patches or the
Humming patch are missing.

- [ ] **Step 6: Run focused verification**

```bash
python -m pytest pipeline/tests/test_patch_vllm_m3_serve.py -q
python -m ruff check \
  pipeline/slurm/patch_vllm_m3_serve.py \
  pipeline/tests/test_patch_vllm_m3_serve.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 7: Commit and push**

```bash
git add pipeline/slurm/patch_vllm_m3_serve.py \
  pipeline/tests/test_patch_vllm_m3_serve.py
git commit -m "feat: admit M3 activation in Humming experts"
git push origin duy-branch
```

---

### Task 3: Build the fail-closed Humming preflight and attestor

**Files:**

- Create: `pipeline/m3_humming_w4a8.py`
- Create: `pipeline/tests/test_m3_humming_w4a8.py`
- Reuse: `pipeline/m3_serve_abi.py`
- Reuse: `pipeline/slurm/patch_vllm_m3_serve.py`

- [ ] **Step 1: Write the exact accepted checkpoint fixture**

In the new test file, define a complete compressed-tensors config containing:

- MiniMax-M3 architecture metadata;
- `quant_method=compressed-tensors`;
- `format=pack-quantized`;
- one `Linear` group with the exact W4A8 contract above;
- a non-empty ignore list covering shared experts, router, and `lm_head`.

Define:

```python
def valid_abi_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "valid": True,
        "format": "pack-quantized",
        "inventory": {"quantized_modules": 64},
        "components": {"routed_experts": {"quantized": 64}},
        "errors": [],
    }
```

Add a focused assertion that
`classify_module("language_model.model.layers.3.block_sparse_moe.experts.0.w1")`
returns the existing key `routed_experts`, and use that exact key in the fixture.

- [ ] **Step 2: Write failing preflight-policy tests**

Specify these interfaces:

```python
@dataclass(frozen=True)
class RuntimeFacts:
    vllm_version: str
    humming_version: str
    device_capability: tuple[int, int]
    humming_source_integrity: str
    humming_source_mismatches: tuple[str, ...]
    humming_cache_dir: str
    f16_accum: str
    moe_gemm_type: str
    normal_patch_status: str
    humming_patch_status: str


def evaluate_preflight(
    config: Mapping[str, object],
    abi_report: Mapping[str, object],
    runtime: RuntimeFacts,
) -> dict[str, object]:
    ...
```

Test the accepted report:

```python
assert report["valid"] is True
assert report["backend"] == "humming"
assert report["reason_codes"] == []
assert report["details"]["weight_group_size"] == 128
assert report["details"]["activation_strategy"] == "token"
```

Parameterize one-field rejection cases for:

- vLLM version;
- Humming version;
- capability;
- Humming wheel-`RECORD` integrity and any detected
  `LLMC_NVFP4_W4A8_G16_V1` overlay;
- Humming cache basename;
- FP16 accumulation;
- grouped-contiguous GEMM;
- either patch status;
- quantization method and format;
- absent or multiple `Linear` groups;
- every weight field;
- every activation field;
- empty ignore metadata;
- invalid ABI report;
- zero routed-expert quantized modules.

Each rejection must have `valid=false`, `backend=humming`, and one stable
uppercase reason code. It must not name CUTLASS or Marlin as a fallback.

- [ ] **Step 3: Write failing log-attestation tests**

Specify:

```python
def classify_backend_log(
    text: str,
    preflight: Mapping[str, object],
) -> dict[str, object]:
    ...
```

Cover:

- accepted single-quoted vLLM config plus indexed GEMM marker;
- accepted JSON-formatted quantization marker;
- missing quantization marker;
- missing GEMM marker;
- grouped-contiguous marker;
- CUTLASS marker;
- Marlin marker;
- unquantized marker;
- two contradictory GEMM markers;
- invalid or non-Humming preflight;
- optional compile/cache markers recorded under `details`.

The accepted result includes:

```python
{
    "valid": True,
    "backend": "humming",
    "gemm_type": "indexed",
    "reason_codes": [],
}
```

Implement quantization recognition with only these two forms:

```python
re.compile(r"""["']quantization["']\s*:\s*["']humming["']""")
re.compile(r"\bquantization=humming\b")
```

Record compile/cache evidence as complete log lines matching both
case-insensitive `humming` and `compile|cache`; these lines are diagnostic and
do not replace either required positive backend marker.

- [ ] **Step 4: Run tests and observe the expected failure**

```bash
python -m pytest pipeline/tests/test_m3_humming_w4a8.py -q
```

Expected: collection fails because `pipeline.m3_humming_w4a8` does not exist.

- [ ] **Step 5: Implement pure policy evaluation**

Create constants:

```python
EXPECTED_VLLM_VERSION = "0.24.0"
EXPECTED_HUMMING_VERSION = "0.1.10"
EXPECTED_DEVICE_CAPABILITY = (9, 0)
```

Implement one ordered list of predicates so reason codes are deterministic.
Return JSON-serializable dictionaries with:

```text
schema_version
valid
backend
reason_codes
details
```

Do not import torch, vLLM, or Humming in the pure functions.

- [ ] **Step 6: Implement runtime discovery and checkpoint preflight**

Implement:

```python
def preflight_checkpoint(checkpoint: Path) -> dict[str, object]:
    config = json.loads((checkpoint / "config.json").read_text())
    abi_report = analyze_checkpoint(checkpoint)
    # discover package versions, SM capability, env policy, and patch checks
    return evaluate_preflight(config, abi_report, runtime)
```

Runtime discovery must:

- read `vllm.__version__`;
- read `humming-kernels` via `importlib.metadata.version`;
- walk the distribution's hashed `humming/` entries from
  `importlib.metadata.distribution("humming-kernels").files`, recompute each
  file's declared hash, and report every mismatch; reject an empty or
  unhashable file set rather than treating it as pristine;
- separately scan the installed Humming sources for
  `LLMC_NVFP4_W4A8_G16_V1` so the reason is explicit;
- read `torch.cuda.get_device_capability`;
- resolve `HUMMING_CACHE_DIR` and require basename
  `cache-m3-gptq-w4a8-v1`;
- call `ensure_vllm_m3_patches(apply=False)`;
- call `ensure_vllm_m3_humming_patch(apply=False)`;
- convert any discovery exception into a structured `DISCOVERY_ERROR` report.

It must never apply a patch or modify a checkpoint.

- [ ] **Step 7: Implement the CLI**

Use subcommands:

```text
python -m pipeline.m3_humming_w4a8 preflight \
  --checkpoint /path/to/checkpoint \
  --out /path/to/humming-preflight.json

python -m pipeline.m3_humming_w4a8 attest \
  --preflight /path/to/humming-preflight.json \
  --log /path/to/serve.log \
  --out /path/to/backend-attestation.json
```

Both commands write indented, sorted JSON with a trailing newline. Return codes:

```text
0  valid
1  discovered but contract/attestation rejected
2  missing/unreadable input or discovery error
```

- [ ] **Step 8: Add CLI tests without GPU imports**

Monkeypatch `preflight_checkpoint` for the preflight CLI. Use temporary JSON and
log files for attestation. Assert output file contents and all three return-code
classes.

- [ ] **Step 9: Run focused verification**

```bash
python -m pytest \
  pipeline/tests/test_m3_humming_w4a8.py \
  pipeline/tests/test_m3_serve_abi.py -q
python -m ruff check \
  pipeline/m3_humming_w4a8.py \
  pipeline/tests/test_m3_humming_w4a8.py
```

Expected: all tests pass; existing ABI tests remain green.

- [ ] **Step 10: Commit and push**

```bash
git add pipeline/m3_humming_w4a8.py \
  pipeline/tests/test_m3_humming_w4a8.py
git commit -m "feat: add Humming W4A8 preflight attestation"
git push origin duy-branch
```

---

### Task 4: Add the structured backend selector to the HTTP launcher

**Files:**

- Modify: `pipeline/slurm/run_vllm_http_serve_smoke.sh`
- Create: `pipeline/tests/test_run_vllm_http_serve_smoke.py`

- [ ] **Step 1: Write a CPU-only launcher harness**

Create a temporary checkpoint directory containing:

```json
{
  "architectures": ["MiniMaxM3ForCausalLM"],
  "model_type": "minimax_m3"
}
```

Run the launcher with:

```python
subprocess.run(
    ["bash", str(LAUNCHER)],
    cwd=REPO_ROOT,
    env={
        **os.environ,
        "PRINT_EFFECTIVE_CONFIG": "1",
        "CKPT": str(checkpoint),
        "MODEL_ID": str(checkpoint),
        "LOG": str(tmp_path / "serve.log"),
        "PID_FILE": str(tmp_path / "serve.pid"),
    },
    text=True,
    capture_output=True,
    check=False,
)
```

- [ ] **Step 2: Write failing selector tests**

Assert:

1. absent selector returns zero, prints `M3_W4A8_BACKEND=cutlass`, and omits
   `--quantization humming`;
2. `humming` returns zero, prints `VLLM_HUMMING_USE_F16_ACCUM=0`,
   `VLLM_HUMMING_MOE_GEMM_TYPE=indexed`,
   `HUMMING_CACHE_DIR` ending in `cache-m3-gptq-w4a8-v1`, and includes exactly
   one `--quantization humming`;
3. an unknown backend exits nonzero and does not print `EFFECTIVE_ARGV`;
4. `EXTRA_VLLM_ARGS=--quantization marlin` exits nonzero;
5. `EXTRA_VLLM_ARGS=--quantization=humming` exits nonzero;
6. a caller-provided true FP16-accum value exits nonzero;
7. a caller-provided `grouped_contiguous` GEMM type exits nonzero.

- [ ] **Step 3: Run tests and observe the expected failures**

```bash
python -m pytest pipeline/tests/test_run_vllm_http_serve_smoke.py -q
```

Expected: selector assertions fail against the current launcher.

- [ ] **Step 4: Parse and validate the selector before side effects**

Add:

```bash
M3_W4A8_BACKEND="${M3_W4A8_BACKEND:-cutlass}"
case "$M3_W4A8_BACKEND" in
  cutlass) ;;
  humming) ;;
  *)
    echo "ERROR: M3_W4A8_BACKEND must be cutlass or humming; got: $M3_W4A8_BACKEND" >&2
    exit 2
    ;;
esac
```

Place validation before patch application, checkpoint patching, GPU preflight,
and PID reuse checks. Reject either spelling of a raw `--quantization` option in
`EXTRA_VLLM_ARGS`.

- [ ] **Step 5: Enforce first-qualification precision and GEMM policy**

For Humming:

```bash
_HUMMING_M3_CACHE_ROOT="${HUMMING_M3_W4A8_CACHE_ROOT:-$HOME/.humming}"
export HUMMING_CACHE_DIR="$_HUMMING_M3_CACHE_ROOT/cache-m3-gptq-w4a8-v1"

case "${VLLM_HUMMING_USE_F16_ACCUM:-0}" in
  ""|0|false|False) export VLLM_HUMMING_USE_F16_ACCUM=0 ;;
  *) echo "ERROR: Humming qualification requires FP32 accumulation" >&2; exit 2 ;;
esac

case "${VLLM_HUMMING_MOE_GEMM_TYPE:-indexed}" in
  ""|indexed) export VLLM_HUMMING_MOE_GEMM_TYPE=indexed ;;
  *) echo "ERROR: first Humming qualification requires indexed MoE GEMM" >&2; exit 2 ;;
esac
```

Allow `HUMMING_M3_W4A8_CACHE_ROOT` to replace only the parent directory while
preserving the required final basename. Do not export Humming-specific variables
in the CUTLASS arm.

- [ ] **Step 6: Integrate the optional patch and preflight**

For a real Humming launch:

1. check/apply normal patches plus `--humming`;
2. run the final `--check --humming`;
3. run:

```bash
python -m pipeline.m3_humming_w4a8 preflight \
  --checkpoint "$MODEL_CKPT" \
  --out "${LOG}.humming-preflight.json"
```

Exit before `nohup` if any command fails. In `PRINT_EFFECTIVE_CONFIG=1`, print
the intended preflight artifact path but skip imports and runtime discovery.

- [ ] **Step 7: Add the structured vLLM argument and observability**

After creating `ARGS`, add:

```bash
if [[ "$M3_W4A8_BACKEND" == "humming" ]]; then
  ARGS+=(--quantization humming)
fi
```

Print the selector in normal startup metadata and `EFFECTIVE_ENV`. Print the two
Humming policy variables only for Humming.

- [ ] **Step 8: Run focused verification**

```bash
python -m pytest pipeline/tests/test_run_vllm_http_serve_smoke.py -q
bash -n pipeline/slurm/run_vllm_http_serve_smoke.sh
```

Expected: all selector tests pass and Bash syntax is valid.

- [ ] **Step 9: Commit and push**

```bash
git add pipeline/slurm/run_vllm_http_serve_smoke.sh \
  pipeline/tests/test_run_vllm_http_serve_smoke.py
git commit -m "feat: select Humming W4A8 serving explicitly"
git push origin duy-branch
```

---

### Task 5: Gate the existing performance arm on Humming attestation

**Files:**

- Modify: `pipeline/slurm/perf_eval_arm.sh`
- Create: `pipeline/tests/test_perf_eval_humming_contract.py`

- [ ] **Step 1: Write a failing source-contract test**

Read `perf_eval_arm.sh` as text and assert:

- it contains `M3_W4A8_BACKEND`;
- the Humming attestation command appears after the local `/v1/models`
  readiness success check;
- the attestation command appears before `performance/scripts/preflight.sh` and
  `performance/scripts/run_performance.sh`;
- its output is `$C/backend-attestation.json`;
- remote BF16 mode is not attested as Humming;
- the file contains no `sbatch`.

This test checks orchestration order only. Log semantics remain covered by
`test_m3_humming_w4a8.py`.

- [ ] **Step 2: Run the test and observe the expected failure**

```bash
python -m pytest pipeline/tests/test_perf_eval_humming_contract.py -q
```

Expected: failure because the arm does not call the attestor.

- [ ] **Step 3: Add the post-readiness, pre-benchmark gate**

Immediately after local readiness:

```bash
if [ "${M3_W4A8_BACKEND:-cutlass}" = humming ]; then
  note "attest Humming backend before benchmark"
  python -m pipeline.m3_humming_w4a8 attest \
    --preflight "$C/serve.log.humming-preflight.json" \
    --log "$C/serve.log" \
    --out "$C/backend-attestation.json" \
    >>"$C/client.log" 2>&1
  rc=$?
  [ "$rc" = 0 ] || {
    note "Humming backend attestation failed rc=$rc"
    exit 1
  }
fi
```

Confirm the preflight filename exactly matches the launcher's `${LOG}`-derived
path. Do not change any benchmark command, profile, workload, or score.

- [ ] **Step 4: Preserve the attestation path**

Add the attestation path to `client.log` and leave it in the existing arm result
root. Do not copy it into the benchmark repository.

- [ ] **Step 5: Run focused verification**

```bash
python -m pytest \
  pipeline/tests/test_perf_eval_humming_contract.py \
  pipeline/tests/test_m3_humming_w4a8.py -q
bash -n pipeline/slurm/perf_eval_arm.sh
```

Expected: all tests pass and Bash syntax is valid.

- [ ] **Step 6: Commit and push**

```bash
git add pipeline/slurm/perf_eval_arm.sh \
  pipeline/tests/test_perf_eval_humming_contract.py
git commit -m "feat: attest Humming before performance runs"
git push origin duy-branch
```

---

### Task 6: Run integrated local verification

**Files:**

- Verify all files changed in Tasks 1–5

- [ ] **Step 1: Run the complete focused test set**

```bash
python -m pytest \
  pipeline/tests/test_patch_vllm_m3_serve.py \
  pipeline/tests/test_m3_serve_abi.py \
  pipeline/tests/test_m3_humming_w4a8.py \
  pipeline/tests/test_run_vllm_http_serve_smoke.py \
  pipeline/tests/test_perf_eval_humming_contract.py -q
```

Expected: all tests pass with zero skips caused by missing GPU packages.

- [ ] **Step 2: Run adjacent regression tests**

```bash
python -m pytest \
  pipeline/tests/test_hopper_nvfp4_w4a8_preflight.py \
  pipeline/tests/test_hopper_nvfp4_w4a8_probe_contract.py \
  pipeline/tests/test_humming_nvfp4_w4a8_patch.py -q
```

Expected: the separate NVFP4 specialization remains green; this feature does
not alter its Humming patch set or fallback policy.

- [ ] **Step 3: Run static checks**

```bash
python -m ruff check \
  pipeline/m3_humming_w4a8.py \
  pipeline/slurm/patch_vllm_m3_serve.py \
  pipeline/tests/test_m3_humming_w4a8.py \
  pipeline/tests/test_patch_vllm_m3_serve.py \
  pipeline/tests/test_run_vllm_http_serve_smoke.py \
  pipeline/tests/test_perf_eval_humming_contract.py
bash -n pipeline/slurm/run_vllm_http_serve_smoke.sh
bash -n pipeline/slurm/perf_eval_arm.sh
```

Expected: no lint or syntax errors.

- [ ] **Step 4: Verify default/experimental command delta**

Run the two dry-run commands against a temporary MiniMax config and diff only
their `EFFECTIVE_ENV` and `EFFECTIVE_ARGV` sections.

Expected differences:

```text
M3_W4A8_BACKEND
VLLM_HUMMING_USE_F16_ACCUM
VLLM_HUMMING_MOE_GEMM_TYPE
HUMMING_CACHE_DIR
--quantization humming
Humming preflight artifact path
```

No TP, EP, model length, KV-cache, block size, parser, CUDA-graph, or shared
expert setting may differ.

- [ ] **Step 5: Inspect repository state**

```bash
git status --short
git diff --check
git diff --stat origin/duy-branch...HEAD
```

Expected: no staged or unstaged tracked changes; only the user's untracked
`AGENTS.md` may appear.

---

### Task 7: Publish the first TP8/EP qualification packet

**Files:**

- Create: `M3_HUMMING_W4A8_HANDOFF.md`
- Reference: `PLANNER_EXECUTOR_PROTOCOL.md`
- Reference: `pipeline/slurm/run_vllm_http_serve_smoke.sh`
- Reference: `pipeline/slurm/smoke_chat_completions.sh`

- [ ] **Step 1: Capture the implementation base**

Run:

```bash
git rev-parse HEAD
git status --short
```

Use the full SHA as the packet's base implementation commit. The only permitted
untracked path is repository-root `AGENTS.md`; every tracked change is a stop.

- [ ] **Step 2: Write a protocol-complete packet**

Use the canonical template from `PLANNER_EXECUTOR_PROTOCOL.md` with:

```text
Protocol version: 1
State: READY_FOR_EXECUTOR
Packet revision: 2026-07-25-r1
Branch: duy-branch
Repository: /mnt/nfs/hoangduy/projects/llm-compressor
Environment: /mnt/nfs/hoangduy/venvs/quant
Checkpoint:
  /mnt/nfs/hoangduy/projects/llm-compressor/artifacts/
  m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay
Topology: 1 exclusive node, 8 H100 GPUs, TP8, EP enabled
Launch: persistent tmux controller with one top-level srun
Time limit: 03:00:00
Backend: M3_W4A8_BACKEND=humming
Graphs: enabled
```

The one decision question is:

> Does the pinned Humming path load this GPTQ W4A8 checkpoint on TP8/EP,
> positively attest indexed Humming MoE selection, and complete repeated
> graphs-on HTTP correctness smokes without non-finite, empty, or failed output?

- [ ] **Step 3: Make the packet fail closed before allocation**

The controller-side preflight must verify:

- branch and base commit;
- no tracked changes and no untracked files except root `AGENTS.md`;
- checkpoint `config.json` and safetensors index;
- quant venv;
- `vllm==0.24.0`;
- `humming-kernels==0.1.10`;
- Humming wheel-`RECORD` integrity, with no NVFP4 source-overlay marker;
- `PRINT_EFFECTIVE_CONFIG=1` emits the Humming selector and exactly one
  `--quantization humming`;
- no result-root collision.

The H100 capability and installed patch checks run inside the `srun` allocation
before server launch.

- [ ] **Step 4: Provide exact launch and monitoring commands**

The packet must create a UTC run ID, a fresh durable root under:

```text
/mnt/nfs/hoangduy/results/m3-humming-w4a8-qualification/$RUN_ID
```

Launch one detached `tmux` controller that owns one top-level:

```text
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8
     --cpus-per-task=192 --time=03:00:00 --kill-on-bad-exit=1
```

Inside the allocation:

1. apply/check `patch_vllm_m3_serve.py --humming --probe`;
2. start the existing HTTP launcher with Humming, TP8, EP, graphs on,
   `M3_LOAD_AUDIT=1`, `M3_MOE_PROBE=1`, and preserved log/PID paths;
3. wait for `/v1/models`;
4. run the attestor;
5. execute ten sequential invocations of the existing
   `smoke_chat_completions.sh` with fixed prompt and token limit;
6. preserve every response and return code;
7. stop the server and record process/scheduler state.

Monitoring commands must be non-owning: `tmux capture-pane`, `tail`, `squeue`,
and `sacct`. The packet must not tell the executor to attach interactively.

- [ ] **Step 5: Set exact success and stop gates**

Required success:

- preflight JSON `valid=true`;
- attestation JSON `valid=true`, `backend=humming`, `gemm_type=indexed`;
- server reaches readiness with normal graphs-on capture;
- all ten smoke commands return zero;
- all ten responses are non-empty;
- no non-finite diagnostic marker;
- no server death, CUDA error, illegal memory access, or backend fallback marker;
- server log, preflight, attestation, responses, versions, command, scheduler
  record, and return codes are preserved.

Stop and return without retry on:

- any version/config/patch/attestation mismatch;
- compile/load/capture/server failure;
- any output or diagnostic gate failure;
- any need to change topology, backend, accumulation, GEMM type, eager mode,
  checkpoint, or code.

Pre-authorized retries: none. Allowed adaptations: none. Performance benchmark:
explicitly prohibited in this packet.

- [ ] **Step 6: Define the evidence return**

Require an executor evidence document using the protocol template, raw small
logs committed when reasonably sized, and large artifacts recorded with
absolute path, byte size, and SHA-256. The executor sets
`RETURNED_FOR_ANALYSIS`, commits, pushes, and stops.

- [ ] **Step 7: Validate the packet mechanically**

Run:

```bash
rg -n "READY_FOR_EXECUTOR|Base Git commit|Decision question|srun|tmux|sbatch|Pre-authorized retries|Allowed adaptations|Stop-and-return|Return contract|performance" \
  M3_HUMMING_W4A8_HANDOFF.md
```

Expected:

- every required protocol section is present;
- `srun` and `tmux` are present;
- `sbatch` appears only in a prohibition such as “do not use sbatch”;
- performance work is explicitly unauthorized;
- no angle-bracket template fields remain.

- [ ] **Step 8: Commit and push the packet**

```bash
git add M3_HUMMING_W4A8_HANDOFF.md
git commit -m "docs: hand off Humming W4A8 qualification"
git push origin duy-branch
```

Expected: the executor can begin without choosing commands, paths, topology,
gates, retry policy, or evidence format.

---

### Task 8: Planner decision after executor return

**Files:**

- Modify later: `M3_HUMMING_W4A8_HANDOFF.md`
- Modify later: `PROJECT_GOALS.md`
- Reuse unchanged: `pipeline/slurm/perf_eval_arm.sh`
- Reuse unchanged: the benchmark repository's existing workload and scoring

This task is not executable until the first packet returns with
`RETURNED_FOR_ANALYSIS`.

- [ ] **Step 1: Verify the evidence commit and raw artifacts**

Confirm actual revision, topology, package versions, commands, scheduler state,
preflight, attestation, ten responses, logs, hashes, and return codes. Treat
missing raw evidence as an unqualified result.

- [ ] **Step 2: Record the qualification verdict**

If any qualification gate failed, keep CUTLASS as default, record the first
localized failure, and end the current arm. Do not authorize a changed setting
or a CUDA fix from the same packet.

If all gates passed, mark the Humming serving path qualified for performance
comparison only.

- [ ] **Step 3: Author a separate paired-performance packet**

The new packet must reuse the existing benchmark unchanged and compare the same
GPTQ checkpoint twice:

```text
control:   M3_W4A8_BACKEND=cutlass
candidate: M3_W4A8_BACKEND=humming
```

Keep profile, TP8/EP topology, model length, KV-cache dtype, request workload,
warmup, concurrency sweep, generation parameters, serving defaults, and scoring
identical. Use fresh result roots and the existing `perf_eval_arm.sh`; solve
result-path collision by sequencing the two local arms unless the existing
benchmark already exposes a proven collision-free paired-run mechanism.

The Humming arm must pass its pre-benchmark attestation. The packet's primary
decision remains production agentic/high-concurrency throughput and TTFT;
concurrency 1 is a secondary guardrail.

- [ ] **Step 4: Stop at the next executor boundary**

Commit and push the new `READY_FOR_EXECUTOR` packet. Do not launch it locally.

---

## Final Plan Verification

Before claiming this plan is ready:

- [ ] Every approved design requirement maps to a task or global constraint.
- [ ] CUTLASS is the default in every example and planned code path.
- [ ] Humming selection, checkpoint eligibility, runtime precision, activation
      admission, and backend identity all fail closed.
- [ ] The existing benchmark is reused; no benchmark script or scoring logic is
      created.
- [ ] No CUDA arithmetic or custom kernel work appears.
- [ ] Local tests do not require GPU packages.
- [ ] The first executor packet uses persistent `tmux` plus top-level `srun`,
      never `sbatch`, and stops before performance.
- [ ] The performance packet is conditional on planner acceptance of the first
      executor return.
- [ ] Search the plan for unresolved placeholders:

```bash
python -c 'from pathlib import Path; import re; p=Path("docs/superpowers/plans/2026-07-25-minimax-m3-humming-native-w4a8-backend.md"); s=p.read_text(); bad=["T"+"BD", "T"+"ODO", "FIX"+"ME", "similar"+" to", "appropriate error"+" handling"]; angle=chr(60)+"[^"+chr(62)+"]+"+chr(62); hits=[x for x in bad if x in s]+re.findall(angle, s); print(hits); raise SystemExit(bool(hits))'
```

Expected: no matches.
