# GPTQ Venv Launcher Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every distributed GPTQ lane runs with the venv interpreter that contains the editable `llmcompressor` install.

**Architecture:** Keep the existing Job and venv setup. Replace the image-level `torchrun` executable with `/work/venv/bin/python -m torch.distributed.run`, and extend environment preflight to import `llmcompressor`.

**Tech Stack:** Kubernetes YAML template, Bash, Python, pytest

## Global Constraints

- Do not change the quantization recipe, lane order, resource request, or failure-hold behavior.
- Do not install another PyTorch copy or add an entrypoint wrapper.
- Preserve unrelated working-tree changes.

---

### Task 1: Bind distributed ranks to the venv

**Files:**
- Modify: `pipeline/tests/test_render_glm53_ep_gptq_chain.py`
- Modify: `pipeline/k8s/glm53-ep-gptq-chain.yaml.tmpl`

**Interfaces:**
- Consumes: rendered container Bash body from `render_glm53_ep_gptq_chain.main`
- Produces: distributed lane command using `/work/venv/bin/python -m torch.distributed.run`

- [ ] **Step 1: Write the failing regression assertions**

Add these assertions after `script = container["args"][0]`:

```python
assert "/work/venv/bin/python -m torch.distributed.run" in script
assert "\n                torchrun " not in script
assert "import llmcompressor" in script
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest pipeline/tests/test_render_glm53_ep_gptq_chain.py -q
```

Expected: the new venv-launcher assertion fails because the rendered script still contains bare `torchrun`.

- [ ] **Step 3: Implement the minimal launcher fix**

In `run_lane`, replace:

```bash
torchrun --nproc_per_node="$ranks" -m pipeline.run \
```

with:

```bash
/work/venv/bin/python -m torch.distributed.run \
  --nproc_per_node="$ranks" -m pipeline.run \
```

In `environment-preflight`, import the editable package:

```python
import llmcompressor
```

- [ ] **Step 4: Verify GREEN and rendered Bash syntax**

Run:

```bash
python -m pytest pipeline/tests/test_render_glm53_ep_gptq_chain.py -q
python pipeline/k8s/render_glm53_ep_gptq_chain.py \
  --run-tag test --ref aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --out .k8s-rendered/venv-launch-test.yaml
python -c "import pathlib,yaml; p=pathlib.Path('.k8s-rendered/venv-launch-test.yaml'); d=yaml.safe_load(p.read_text()); pathlib.Path('.k8s-rendered/venv-launch-test.body.sh').write_text(d['spec']['template']['spec']['containers'][0]['args'][0])"
bash -n .k8s-rendered/venv-launch-test.body.sh
```

Expected: all pytest tests pass and `bash -n` exits 0.

- [ ] **Step 5: Commit only the fix files**

```bash
git add pipeline/tests/test_render_glm53_ep_gptq_chain.py \
  pipeline/k8s/glm53-ep-gptq-chain.yaml.tmpl
git diff --cached --check
git commit -m "fix: launch GPTQ ranks with venv Python"
```
