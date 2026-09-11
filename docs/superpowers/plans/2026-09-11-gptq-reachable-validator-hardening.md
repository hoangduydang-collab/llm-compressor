# GPTQ Reachable Validator Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the single-file representative and sharded full checkpoints emitted by the GLM-5.3 EP GPTQ chain before another relaunch.

**Architecture:** Reuse `pipeline.serve_ignore.weight_map_of` as the sole checkpoint-layout contract. Feed its tensor-to-file map into the existing exhaustive key and scale checks.

**Tech Stack:** Python, pytest, PyTorch, safetensors, Kubernetes read-only replay

## Global Constraints

- Test only checkpoint layouts and validation branches reached by this chain.
- Preserve existing key, scale, phase, memory, quantization, and hold semantics.
- Do not modify serving/conversion paths or add malformed third-party artifact tests.
- Preserve unrelated working-tree changes.

---

### Task 1: Support both emitted checkpoint layouts

**Files:**
- Modify: `pipeline/validate_glm53_ep_gptq.py`
- Modify: `pipeline/tests/test_validate_glm53_ep_gptq.py`

**Interfaces:**
- Consumes: `weight_map_of(checkpoint: Path) -> dict[str, str]`
- Produces: `validate_checkpoint` results for single-file and sharded checkpoints

- [ ] **Step 1: Add real safetensors fixtures and failing tests**

Add imports:

```python
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
```

Add a tensor fixture:

```python
def _complete_tensors(layers=(3,), experts=1):
    tensors = {}
    for key in _complete_keys(layers=layers, experts=experts):
        if key.endswith(".weight_packed"):
            tensors[key] = torch.zeros((1, 1), dtype=torch.int32)
        elif key.endswith(".weight_scale"):
            tensors[key] = torch.ones((1,), dtype=torch.float32)
        else:
            tensors[key] = torch.tensor([1, 1], dtype=torch.int64)
    return tensors
```

Add tests:

```python
def test_validate_checkpoint_accepts_single_file_emitted_layout(tmp_path: Path):
    save_file(_complete_tensors(), tmp_path / "model.safetensors")
    assert validator.validate_checkpoint(tmp_path, layers=(3,), experts=1) == []


def test_validate_checkpoint_accepts_sharded_emitted_layout(tmp_path: Path):
    tensors = _complete_tensors()
    keys = sorted(tensors)
    first = {key: tensors[key] for key in keys[::2]}
    second = {key: tensors[key] for key in keys[1::2]}
    save_file(first, tmp_path / "model-00001-of-00002.safetensors")
    save_file(second, tmp_path / "model-00002-of-00002.safetensors")
    weight_map = {
        key: (
            "model-00001-of-00002.safetensors"
            if key in first
            else "model-00002-of-00002.safetensors"
        )
        for key in keys
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    assert validator.validate_checkpoint(tmp_path, layers=(3,), experts=1) == []


def test_validate_checkpoint_reads_bad_scale_from_single_file(tmp_path: Path):
    tensors = _complete_tensors()
    scale = next(key for key in tensors if key.endswith(".weight_scale"))
    tensors[scale] = torch.tensor([0.0], dtype=torch.float32)
    save_file(tensors, tmp_path / "model.safetensors")
    assert validator.validate_checkpoint(tmp_path, layers=(3,), experts=1) == [
        "weight scales contain 1 non-finite-or-non-positive value(s)"
    ]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest pipeline/tests/test_validate_glm53_ep_gptq.py -q
```

Expected: the three filesystem tests fail because `_checkpoint_index` requires
`model.safetensors.index.json`.

- [ ] **Step 3: Use the canonical weight map**

Remove `_checkpoint_index`. Change scale loading to:

```python
def _scale_values(
    checkpoint: Path,
    weight_map: Mapping[str, str],
) -> Iterable[float]:
    from safetensors import safe_open

    wanted = {key for key in weight_map if key.endswith(".weight_scale")}
    by_shard: dict[str, list[str]] = {}
    for key in wanted:
        by_shard.setdefault(weight_map[key], []).append(key)
    for shard, shard_keys in sorted(by_shard.items()):
        with safe_open(str(checkpoint / shard), framework="pt", device="cpu") as src:
            for key in sorted(shard_keys):
                tensor = src.get_tensor(key).reshape(-1)
                yield from tensor.tolist()
```

Change checkpoint validation to:

```python
def validate_checkpoint(
    checkpoint: Path, *, layers: Iterable[int], experts: int = 256
) -> list[str]:
    from pipeline.serve_ignore import weight_map_of

    weight_map = weight_map_of(checkpoint)
    errors = validate_checkpoint_keys(weight_map, layers=layers, experts=experts)
    if not errors:
        errors.extend(
            validate_scale_values(_scale_values(checkpoint, weight_map))
        )
    return errors
```

- [ ] **Step 4: Verify GREEN and focused regressions**

Run:

```bash
python -m pytest \
  pipeline/tests/test_validate_glm53_ep_gptq.py \
  pipeline/tests/test_render_glm53_ep_gptq_chain.py \
  pipeline/tests/test_compare_glm53_ep_gptq.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit only implementation files**

```bash
git add pipeline/validate_glm53_ep_gptq.py \
  pipeline/tests/test_validate_glm53_ep_gptq.py
git diff --cached --check
git commit -m "fix: validate emitted GPTQ checkpoint layouts"
```

---

### Task 2: Replay only reachable live gates

**Files:**
- No repository changes

**Interfaces:**
- Consumes: saved representative EP4 checkpoint and rank metrics
- Produces: pre-relaunch evidence for validator and logit loader

- [ ] **Step 1: Run the exact fixed validator against the saved EP4 artifact**

Pipe the committed `pipeline/validate_glm53_ep_gptq.py` into the held Pod's
venv Python, with the explicit checkpoint, layers `3,4`, 256 experts, four-rank
run directory, and `/dev/null` output. Expected: `"ok": true`, no key/scale/phase
errors, and non-null `peak_cuda_bytes`.

- [ ] **Step 2: Check GPU occupancy and request explicit approval**

Run `nvidia-smi --query-compute-apps` in the held Pod. Show the exact GPU-0
load/forward command and wait for approval.

- [ ] **Step 3: Exercise the exact logit-loader path**

Call `_last_token_logits` on the saved EP4 checkpoint, source tokenizer, and
the committed default prompt with `CUDA_VISIBLE_DEVICES=0`. Expected: one finite
`float32` logits array and exit 0.

- [ ] **Step 4: Push and perform queue-before-delete relaunch**

After all evidence passes, push the commits, submit a replacement pinned to
`gpu04`, verify it is Pending behind the holder, delete the holder, and verify
the replacement runs on `gpu04`.
