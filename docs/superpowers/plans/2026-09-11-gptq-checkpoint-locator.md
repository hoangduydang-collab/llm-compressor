# GPTQ Checkpoint Locator Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Rancher chain find checkpoints at the pipeline's actual `<lane>/<model>/<timestamp>/checkpoint` path.

**Architecture:** Preserve the existing `checkpoint_for` interface and newest-candidate behavior. Change only its discovery from a one-level glob to recursive checkpoint-directory discovery.

**Tech Stack:** Kubernetes YAML template, embedded Python, pytest, Bash

## Global Constraints

- Preserve quantization, validation, lane ordering, resources, and failure-hold behavior.
- Fail closed if no checkpoint directory exists.
- Preserve unrelated working-tree changes.

---

### Task 1: Recursively discover lane checkpoints

**Files:**
- Modify: `pipeline/tests/test_render_glm53_ep_gptq_chain.py`
- Modify: `pipeline/k8s/glm53-ep-gptq-chain.yaml.tmpl`

**Interfaces:**
- Consumes: lane output root passed as `checkpoint_for "$RUN_ROOT/lanes/<lane>"`
- Produces: newest recursively discovered directory named `checkpoint`

- [ ] **Step 1: Add failing rendered-template assertions**

After obtaining `script`, add:

```python
assert '.rglob("checkpoint")' in script
assert '.glob("*/checkpoint")' not in script
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest pipeline/tests/test_render_glm53_ep_gptq_chain.py -q
```

Expected: failure because the rendered script still contains
`glob("*/checkpoint")`.

- [ ] **Step 3: Implement the minimal fix**

In `checkpoint_for`, replace:

```python
pathlib.Path(sys.argv[1]).glob("*/checkpoint"),
```

with:

```python
(
    path
    for path in pathlib.Path(sys.argv[1]).rglob("checkpoint")
    if path.is_dir()
),
```

Keep newest-modification-time sorting and the existing no-candidate failure.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest pipeline/tests/test_render_glm53_ep_gptq_chain.py -q
python pipeline/k8s/render_glm53_ep_gptq_chain.py \
  --run-tag locatorverify \
  --ref aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --out .k8s-rendered/locator-verify.yaml
python -c "import pathlib,yaml; p=pathlib.Path('.k8s-rendered/locator-verify.yaml'); d=yaml.safe_load(p.read_text()); pathlib.Path('.k8s-rendered/locator-verify.body.sh').write_text(d['spec']['template']['spec']['containers'][0]['args'][0])"
bash -n .k8s-rendered/locator-verify.body.sh
```

Expected: 3 tests pass and `bash -n` exits 0.

- [ ] **Step 5: Commit only implementation files**

```bash
git add pipeline/tests/test_render_glm53_ep_gptq_chain.py \
  pipeline/k8s/glm53-ep-gptq-chain.yaml.tmpl
git diff --cached --check
git commit -m "fix: discover nested GPTQ checkpoints"
```
