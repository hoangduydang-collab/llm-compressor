# Hopper NVFP4 W4A8 Humming Specialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:executing-plans` task-by-task. This plan is deliberately local and
> sequential because the repository currently forbids unrequested subagents.

**Goal:** Add an opt-in, fail-closed Humming `0.1.10` specialization that keeps
NVFP4 E2M1 weights packed, dynamically quantizes activations to E4M3, applies
each original group-16 weight scale to only its K16 half, and executes Hopper
FP8 WGMMA with FP32 accumulation and BF16 output.

**Architecture:** This repository owns a deterministic overlay for the pinned
Humming wheel instead of copying Humming's GEMM framework. The overlay adds one
exact policy flag, expands the existing scale-register path from two to four
E4M3 scales per WGMMA N16 subblock, applies those scales while the converted B
fragment is still in E4M3 registers, and disables Humming's ordinary
post-accumulator group scaling for that policy. A pure-Python bit-level oracle,
an exact source-version gate, and an H100 probe make the change testable without
silently affecting Marlin or other Humming configurations.

**Tech stack:** Python 3.11, pytest, Humming kernels `0.1.10`, CUDA 12/13,
PyTorch FP8, SM90 WGMMA, vLLM compressed-tensors, Bash, Slurm `srun`.

## Global constraints

- Work from `duy-branch`; preserve the unrelated user changes in
  `PROJECT_GOALS.md` and the untracked `AGENTS.md`.
- Do not vendor or recreate Humming's repacker, TMA pipeline, WGMMA mainloop,
  activation quantizer, or vLLM adapter.
- The overlay accepts only the exact unmodified Humming `0.1.10` LF-normalized
  source hashes recorded below, or a source tree already carrying this
  overlay's marker. Backups preserve the original bytes, including CRLF.
- The new policy is exactly: WGMMA, A=`float8e4m3`, B=`float4e2m1`,
  BS=`float8e4m3`, input group size `0`, weight group size `16`,
  `GROUP_TENSOR`, symmetric weights, and FP32 accumulation.
- Unsupported source versions, dtypes, scale types, zero points, differing
  fused global scales, non-SM90 devices, or failed layout checks stop the W4A8
  arm before model mutation. They do not alter the ordinary NVFP4/Marlin W4A16
  launch arm.
- Preserve two distinct group-16 scales. Never average, merge, or select one
  scale for the full K32 WGMMA fragment.
- Use the numerical contract
  `B8 = Q_E4M3(E2M1(B4) * group_scale / 8)` and compensate once with
  `effective_global_scale = checkpoint_inverse_global_scale * 8`.
- Keep transformed persistent bytes for packed weight plus compact group scale
  plus global scale at or below `1.10x` their checkpoint bytes.
- Local tests are CPU-only. GPU compile, SASS, correctness, memory, and timing
  evidence must run through the canonical planner/executor packet using a
  top-level `srun`; never emit `sbatch`.

## Pinned source contract

The overlay targets Humming tag `v0.1.10`, commit
`4351af3a8fcdce1a8dee50104ba49566af2427fb`, with these SHA-256 values:

| Relative file | Unmodified SHA-256 |
| --- | --- |
| `humming/config/config.py` | `20406dc0ce4fcb957a035c4f04fc0f1c41746519be495ce39bcb2ff270d327e4` |
| `humming/kernel/humming.py` | `4f51361a17de489366fd849fa305ebc6d055b117531352c5963985f19ff09804` |
| `humming/layer.py` | `31a6d7c0394683b9c1236fd93dc275e19a289b66488a57ca8b531fe30f29c8dd` |
| `humming/include/humming/memory/s2r_loader/loader_bs.cuh` | `4f3d64befa294c90a23e05f3cde28474ee46cae4ac9f36ac605dd8aec76480ae` |
| `humming/include/humming/arith/mainloop_arith.cuh` | `1ca6831856827cebb1402bc167a4468a1002862ebe0f498b81017539618a8a5d` |
| `humming/include/humming/mma/wgmma.cuh` | `77f89c87c8404818e1596592f4423d3c5609d1fcbad17a2f2fd373d92fe1383a` |

---

### Task 1: Bit-exact NVFP4-to-E4M3 reference and memory contract

**Files:**

- Create: `pipeline/hopper_nvfp4_w4a8/__init__.py`
- Create: `pipeline/hopper_nvfp4_w4a8/reference.py`
- Test: `pipeline/tests/test_hopper_nvfp4_w4a8_reference.py`

**Interfaces:**

- `decode_e2m1(code: int) -> Fraction`
- `decode_e4m3fn(code: int) -> Fraction | None`
- `encode_e4m3fn(value: Fraction) -> int`
- `convert_e2m1_group_to_e4m3(codes, scale_code) -> tuple[int, ...]`
- `persistent_byte_report(n: int, k: int, num_global_scales: int = 1) -> dict`

- [ ] **Step 1: Write failing datatype tests**

Test all 16 E2M1 codepoints against
`[0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6]`. Test E4M3FN
zero, subnormal, normal, `448`, NaN encodings, saturation, sign symmetry, and
round-to-nearest-even midpoint cases using `fractions.Fraction`.

- [ ] **Step 2: Write the K16 isolation sentinel**

Build 32 E2M1 values with identical nonzero codepoints in both halves but use
asymmetric scale codes (`1.0` for K0:16, `2.0` for K16:32). Assert the first 16
outputs equal `Q_E4M3(B4/8)`, the second 16 equal `Q_E4M3(B4/4)`, and swapping
the scale codes swaps only the corresponding half.

- [ ] **Step 3: Confirm RED**

Run: `python -m pytest -q pipeline/tests/test_hopper_nvfp4_w4a8_reference.py`

Expected: collection fails because the package does not exist.

- [ ] **Step 4: Implement the minimal pure-Python oracle**

Enumerate all finite E4M3FN codes as exact `Fraction` values. Encode by nearest
absolute distance, then round ties to the candidate with an even mantissa LSB;
saturate finite overflow to signed `448`. Decode E2M1 by a fixed 16-entry tuple.
Apply `scale / 8` separately to each submitted 16-value group.

`persistent_byte_report` computes:

```python
checkpoint = n * k // 2 + n * k // 16 + 4 * num_global_scales
transformed = n * k // 2 + n * k // 16 + 4 * num_global_scales
return {"checkpoint": checkpoint, "transformed": transformed,
        "ratio": transformed / checkpoint}
```

Reject non-byte-aligned shapes and any group whose length is not 16.

- [ ] **Step 5: Confirm GREEN**

Run: `python -m pytest -q pipeline/tests/test_hopper_nvfp4_w4a8_reference.py`

Expected: all all-codepoint, rounding, K16-isolation, and byte-accounting tests
pass without importing torch.

### Task 2: Fail-closed, idempotent Humming source overlay

**Files:**

- Create: `pipeline/hopper_nvfp4_w4a8/humming_patch.py`
- Create: `pipeline/slurm/patch_humming_nvfp4_w4a8.py`
- Test: `pipeline/tests/test_humming_nvfp4_w4a8_patch.py`

**Interfaces:**

- `PatchTarget(relative_path, pristine_sha256, transform)`
- `patch_humming_tree(root: Path, *, apply: bool) -> PatchReport`
- CLI: `python pipeline/slurm/patch_humming_nvfp4_w4a8.py [--root PATH] [--check] [--json PATH]`

- [ ] **Step 1: Write failing overlay tests**

Use minimal source fixtures containing every exact anchor. Assert the overlay:

- changes all six targets and adds marker `LLMC_NVFP4_W4A8_G16_V1`;
- is byte-for-byte idempotent on a second application;
- refuses a source with a changed anchor or unknown hash;
- `--check` returns nonzero for pristine/unrecognized trees and zero only for a
  completely patched tree;
- never modifies any file if validation of one target fails;
- writes `.llmc-orig-v0.1.10` backups only during the first successful apply.

- [ ] **Step 2: Confirm RED**

Run: `python -m pytest -q pipeline/tests/test_humming_nvfp4_w4a8_patch.py`

Expected: collection fails because `humming_patch` does not exist.

- [ ] **Step 3: Implement validation-before-write and atomic replacement**

Resolve the installed root from `Path(humming.__file__).parent` unless `--root`
is supplied. Before writing, read all six files, classify each as pristine,
patched, or unknown, and run every transform in memory. Abort the entire apply
if any file is missing, has an unknown hash, lacks its exact anchor, or has only
part of the marker set. After validation, create backups with exclusive-create
semantics and replace each file through a same-directory temporary file.

The JSON report records package version, root, commit/tag contract, per-file
before/after hashes, status, and whether any write occurred. The CLI requires
installed distribution `humming-kernels==0.1.10` when resolving automatically;
an explicit fixture root bypasses package discovery but not file hashes/anchors.

- [ ] **Step 4: Confirm GREEN**

Run: `python -m pytest -q pipeline/tests/test_humming_nvfp4_w4a8_patch.py`

Expected: all transactional, hash, idempotence, backup, CLI, and marker tests
pass.

### Task 3: Add the exact Humming A8/E2M1/g16 policy

> **Implementation correction:** the ordinary S2R path cannot load four bytes
> from one K16 row. The specialization gathers the same N positions from two
> adjacent compact K16 scale rows, then interleaves them per N16 WGMMA fragment.
> G2S storage remains unchanged.

**Files transformed by the Task 2 overlay:**

- Humming: `humming/config/config.py`
- Humming: `humming/kernel/humming.py`
- Humming: `humming/layer.py`
- Humming: `humming/include/humming/memory/s2r_loader/loader_bs.cuh`
- Humming: `humming/include/humming/arith/mainloop_arith.cuh`
- Humming: `humming/include/humming/mma/wgmma.cuh`
- Test: `pipeline/tests/test_humming_nvfp4_w4a8_patch.py`

- [ ] **Step 1: Add failing policy-contract assertions to overlay tests**

Assert the patched fixture contains all of the following behaviors, not merely
the marker:

- generated C++ `LayerConfig::kUseNvfp4W4a8G16`;
- the old scale assertion is relaxed only when that flag is true and
  `use_f16_accum` is false;
- four BS values per N16 subblock for this policy;
- pre-WGMMA FP8 B scaling by the four distinct scale bytes and `0.125f`;
- no post-C group scale and no accumulator reset for this policy;
- exactly one load-time `global_scale * 8.0` compensation.

- [ ] **Step 2: Confirm the new assertions fail**

Run: `python -m pytest -q pipeline/tests/test_humming_nvfp4_w4a8_patch.py`

Expected: the policy-contract assertions fail against transforms that only add
the patch framework/marker.

- [ ] **Step 3: Add the Python policy flag and guarded scale allowance**

In `LayerConfig._cpp_extra_names`, add `use_nvfp4_w4a8_g16`. At the end of
`LayerConfig.__post_init__`, define it as the conjunction:

```python
self.mma_type == MmaType.WGMMA
and self.a_dtype == dtypes.float8e4m3
and self.b_dtype == dtypes.float4e2m1
and self.bs_dtype == dtypes.float8e4m3
and self.input_scale_group_size == 0
and self.weight_scale_group_size == 16
and self.weight_scale_type == WeightScaleType.GROUP_TENSOR
and not self.has_zero_point
```

In `HummingKernel.check_scale`, replace only the weight-scale minimum assertion
with:

```python
if self.weight_scale_group_size > 0:
    allow_nvfp4_w4a8_g16 = (
        self.use_nvfp4_w4a8_g16 and not self.use_f16_accum
    )
    assert (
        self.weight_scale_group_size >= 256 // self.a_dtype.num_bits
        or allow_nvfp4_w4a8_g16
    )
```

- [ ] **Step 4: Reuse the existing compact scale loader with four values**

In `loader_bs.cuh`, change `kNumScalesPerSubBlock` so the new policy selects
`4`, while every old condition remains unchanged. Add a policy-only S2R gather
that loads the same two N-lane bytes from `group_index` and
`group_index + 1`, then interleaves each N16 fragment as two K0:16 bytes followed
by two K16:32 bytes. In `mainloop_arith.cuh`, make `kNumBSPerSubBlock` select `4`
for the same flag. Do not change the G2S loader or
`prepare_humming_weight_scale`; both already preserve compact storage.

The four bytes for transform iteration `j` represent two output-row positions
for K0:16 followed by the same two positions for K16:32. The GPU sentinel in
Task 5 is authoritative for the final byte permutation.

- [ ] **Step 5: Scale converted E4M3 B registers before WGMMA**

At the top of `MainloopArithmetic`, add the policy constant and inherit/use
`F8Conversion<ElementA>`. In `may_apply_bs_and_zp_on_b`, add a compile-time
branch for the exact policy. For each of the four 32-bit B registers, read its
matching E4M3 scale byte from `bs[buffer_id]` at `j * 4 + register_id`, convert
each of that register's four E4M3 values to float, multiply by
`float(scale) * 0.125f`, and construct E4M3 again. Add static assertions for
E4M3 A/BS, E2M1 B, WGMMA, g16, no zero point, and four registers.

The scalar conversion is the proof-first implementation. Vectorizing the four
conversions is a later optimization allowed only after exact correctness.

- [ ] **Step 6: Prevent double scaling and accumulator partition loss**

In `may_apply_as_and_bs_on_wgmma_c`, immediately return for
`kUseNvfp4W4a8G16`; group scaling has already happened on B.

In `wgmma.cuh::run`, add `!LayerConfig::kUseNvfp4W4a8G16` to
`kApplyScaleOnC` and both group-boundary `scale_d` branches. This keeps the
ordinary accumulator live across K32 iterations instead of resetting it at
every g16 boundary. Also exclude the policy in `final_regs_c_as_ptr()` so the
epilogue reads the live ordinary accumulator (`regs_c[0]`), not the unused
post-scale accumulator (`regs_c[1]`).

- [ ] **Step 7: Apply the single `/8` compensation during transformation**

In `HummingLayerMethod.transform_humming_layer`, after reading `global_scale`
and before setting parameters, require it for the special policy and replace it
with `global_scale * 8.0`. Do not mutate checkpoint tensors in place. The
`GROUP_TENSOR` predicate excludes Humming's differing-fused-global fold-to-group
case from this path.

- [ ] **Step 8: Confirm GREEN and compile Python outputs**

Run:

```powershell
python -m pytest -q pipeline/tests/test_humming_nvfp4_w4a8_patch.py
python -m compileall -q pipeline/hopper_nvfp4_w4a8 pipeline/slurm/patch_humming_nvfp4_w4a8.py
```

Expected: all overlay tests pass, and both Python modules compile. C++/CUDA
compilation is deliberately deferred to the SM90 gate.

### Task 4: Opt-in preflight and installer integration

**Files:**

- Create: `pipeline/hopper_nvfp4_w4a8/preflight.py`
- Create: `pipeline/slurm/install_humming_nvfp4_w4a8.sh`
- Test: `pipeline/tests/test_hopper_nvfp4_w4a8_preflight.py`

**Interfaces:**

- `PreflightInput` consumes device capability, Humming version/patch report,
  normalized layer metadata, and transformed byte counts.
- `evaluate_preflight(...) -> PreflightReport`
- CLI emits one JSON object and exits `0` only for `backend=humming_w4a8_g16`.

- [ ] **Step 1: Write failing preflight tests**

Cover one accepting case and a separate rejecting reason for SM != 90, package
version mismatch, incomplete patch, non-E2M1 weight, non-E4M3 activation/scale,
g != 16, zero point, non-`GROUP_TENSOR`, missing/differing fused global scale,
F16 accumulation, and transformed ratio above `1.10`.

Each rejection must return `backend=marlin_w4a16`, `eligible=false`, and one
stable reason code; it must never claim an in-process runtime retry.

- [ ] **Step 2: Confirm RED**

Run: `python -m pytest -q pipeline/tests/test_hopper_nvfp4_w4a8_preflight.py`

Expected: collection fails because `preflight` does not exist.

- [ ] **Step 3: Implement pure decision logic and JSON CLI**

Keep decision logic torch-free. Hardware/package discovery belongs in the CLI
adapter and is converted to normalized input first. The accepted report records
SM, package version, source hashes, activation/weight/scale dtype, group sizes,
accumulator/output dtype, checkpoint/transformed bytes, global-scale cardinality,
and exact backend.

- [ ] **Step 4: Add a dedicated installer, not a production default switch**

`install_humming_nvfp4_w4a8.sh` activates `/mnt/nfs/hoangduy/venvs/quant`,
verifies `humming-kernels==0.1.10`, runs the overlay, selects and verifies a
specialization-owned Humming JIT namespace without recursive cache deletion,
runs `--check`, and prints the separate Marlin command as the fallback. It must not edit
`install_vllm_m3_serve.sh` or make W4A8 the default before qualification.

- [ ] **Step 5: Confirm GREEN**

Run:

```powershell
python -m pytest -q pipeline/tests/test_hopper_nvfp4_w4a8_preflight.py
rg -n "sbatch" pipeline/slurm/install_humming_nvfp4_w4a8.sh
```

Expected: tests pass and `rg` returns no matches.

### Task 5: Exact SM90 compile/correctness/memory probe

**Files:**

- Create: `pipeline/hopper_nvfp4_w4a8/gpu_probe.py`
- Create: `pipeline/slurm/run_hopper_nvfp4_w4a8_probe_srun.sh`
- Test: `pipeline/tests/test_hopper_nvfp4_w4a8_probe_contract.py`

- [ ] **Step 1: Write failing launcher/probe contract tests**

Static tests require: one top-level `srun --nodes=1 --ntasks=1 --gres=gpu:1`,
no `sbatch`, a unique result root, exact Git revision capture, patch `--check`,
SM90 check, deterministic seeds, return-code capture, JSON validation, and no
model download or full quality evaluation.

- [ ] **Step 2: Confirm RED**

Run: `python -m pytest -q pipeline/tests/test_hopper_nvfp4_w4a8_probe_contract.py`

Expected: tests fail because the probe and launcher do not exist.

- [ ] **Step 3: Implement the one-layer probe**

Use a dense Humming layer with `N=128`, `K=128`, BF16 output, dynamic per-token
E4M3 A, packed E2M1 B, E4M3 g16 scales, one global scale, WGMMA, and FP32
accumulation. Exercise Humming's public `transform_humming_layer` using its
packed-int32 checkpoint ABI, then run `M in [1, 8, 32]`. Include:

- all 16 E2M1 codes;
- asymmetric alternating scale codes across every adjacent K16 pair;
- a K16 isolation case where only one half's scale changes;
- per-output-row sentinels spanning every N16 fragment and both K16 halves;
- random finite cases with a fixed seed;
- repeated launch determinism;
- transformed tensor byte counts before and after Humming preparation.

Compare against two references: exact emulated A8 x pre-rounded B8 for kernel
contract, and dequantized BF16 NVFP4 for approximation drift. Emit max absolute,
max relative, cosine, finite fraction, and per-shape pass/fail. Treat the exact
emulation comparison as the correctness gate; record the BF16 comparison but do
not tune its tolerance after seeing results.

- [ ] **Step 4: Capture generated-code evidence**

Record Humming kernel config and cubin path, use `cuobjdump --dump-sass` on that
cubin, and require an SM90 FP8 `WGMMA.MMA_ASYNC` instruction. Absence of WGMMA,
any persistent FP8-expanded weight tensor, nonfinite output, a failed K16
isolation assertion, or memory ratio above `1.10` fails the probe.

- [ ] **Step 5: Run local contract verification**

Run:

```powershell
python -m pytest -q pipeline/tests/test_hopper_nvfp4_w4a8_probe_contract.py
python -m compileall -q pipeline/hopper_nvfp4_w4a8/gpu_probe.py
```

Expected: CPU contract tests and Python compilation pass. Do not claim CUDA
correctness locally.

### Task 6: Canonical executor packet and continuation gate

**Files:**

- Create: `HOPPER_NVFP4_W4A8_HANDOFF.md`
- Modify: `PROJECT_GOALS.md` only if the user's existing edit can be preserved
  exactly and the new goal is not already present; otherwise leave it untouched.

- [ ] **Step 1: Write the execution packet in `PLANNER_ANALYSIS` state**

Use protocol version 1 and one decision question: “Does the patched Humming
A8/E2M1/g16 specialization compile on SM90 and satisfy exact K16 scale
isolation, numerical, packed-memory, and WGMMA instruction gates?” Include the
exact branch, base commit, environment, one-GPU resource contract, setup,
preflight, dry-run, `srun` launch, monitoring, evidence files, stop conditions,
and return schema. A pre-GPU failure returns blocker evidence and launches
nothing else.

- [ ] **Step 2: Add fixed gates**

The packet authorizes only the one-layer probe. Required gates are:

- overlay version/source check passes;
- device capability is exactly SM90;
- JIT compilation and all process return codes are zero;
- K16 isolation and exact-emulation comparisons pass for every M;
- all outputs are finite and deterministic;
- transformed persistent byte ratio is `<= 1.10`;
- SASS contains FP8 WGMMA;
- raw JSON, stdout/stderr, SASS, environment versions, scheduler output, and
  SHA-256 manifest are preserved.

The packet explicitly forbids model download, full-model serving, performance
continuation, MoE work, re-quantization, and quality evaluation.

- [ ] **Step 3: Verify the complete local suite**

Run:

```powershell
python -m pytest -q pipeline/tests/test_hopper_nvfp4_w4a8_reference.py pipeline/tests/test_humming_nvfp4_w4a8_patch.py pipeline/tests/test_hopper_nvfp4_w4a8_preflight.py pipeline/tests/test_hopper_nvfp4_w4a8_probe_contract.py
python -m compileall -q pipeline/hopper_nvfp4_w4a8 pipeline/slurm/patch_humming_nvfp4_w4a8.py
git diff --check
```

Expected: all focused CPU tests pass, compilation succeeds, and Git reports no
whitespace errors.

- [ ] **Step 4: Review, commit, and push planner artifacts**

Review the final diff without staging `PROJECT_GOALS.md` or `AGENTS.md` unless
Task 6 Step 1 explicitly required the former. Commit the implementation and
packet on `duy-branch`, push `duy-branch`, then update the packet to
`READY_FOR_EXECUTOR` only when its base commit is the exact pushed commit.

## Post-probe decision boundary

- If compile, exact correctness, layout, or compact-memory gates fail, stop and
  analyze the raw evidence. Do not begin timing or CUTLASS work.
- If all Task 5 gates pass, issue a new packet for target-shape latency against
  W4A16 Marlin and load-expanded W8A8. That later packet must include activation
  quantization time and requires at least `15%` improvement over Marlin in one
  decision-relevant M bucket without exceeding `10%` regression in the primary
  bucket.
- Only after correctness, memory, and latency continuation gates pass may a
  separate plan cover vLLM selector integration, dense model serving, quality,
  or fused MoE.
