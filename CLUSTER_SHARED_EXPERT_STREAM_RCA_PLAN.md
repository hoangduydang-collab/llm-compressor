# MiniMax-M3 shared-expert stream CUDA-graph RCA plan

> **For the cluster agent:** Execute this runbook in order. Before the first
> live run, show the user `hostname`, `nvidia-smi`, the exact result path, and
> the three matrix commands, then obtain one approval covering the GPU claim,
> result writes, and cleanup of only the process groups started by this matrix.

**Goal:** Test whether the HTTP CUDA-graph illegal memory access (IMA) is caused
by a missing main-stream join after MiniMax-M3 shared experts run on vLLM's
auxiliary CUDA stream.

**Scope:** Evidence collection only. Do not edit vLLM, the checkpoint, or
site-packages. Do not install packages. Do not use `CUDA_LAUNCH_BLOCKING=1`.

**Estimated workload:** Three conditions, each with three sequential TP8 HTTP
starts. Use one free 8-GPU node. Allow several hours.

---

## 1. Why this experiment is the next step

The h119 run `20260710-051009` established:

- async CUDA + graphs + breakable graphs failed in two of three baseline trials;
- the same configuration passed with `CUDA_LAUNCH_BLOCKING=1`, so timing changes
  mask the bug;
- graphs off passed;
- breakable graphs disabled passed;
- the CUDA core dump named a BF16 elementwise-add kernel, not a collective:
  `at::native::vectorized_elementwise_kernel<8, ...CUDAFunctor_add<BFloat16>>`.

The vLLM fork's shared-expert implementation contains a matching race candidate:

```python
with torch.cuda.stream(self._stream):
    output = self._layer(shared_experts_input)
    current_stream().wait_stream(self._stream)
```

Inside that context, `current_stream()` is `self._stream`, so the operation is a
self-wait. The original/main stream is not made to wait before it consumes
`shared_output` in:

```python
result = shared_output + fused_output
```

The batch-size fingerprint is independently testable:

- the default shared-expert auxiliary-stream threshold is `tokens <= 256`;
- vLLM captures the 51 default CUDA-graph sizes largest-first;
- 16 sizes are greater than 256, so size 256 begins after about `16/51`;
- changing the threshold to 128 leaves 32 sizes greater than 128, so the
  transition should move to about `32/51`.

This plan therefore varies only the shared-expert stream behavior while keeping
async CUDA, breakable CUDA graphs, model, TP/EP layout, and serve flags fixed.

### Reference contract

- Fork implementation:
  <https://github.com/toncao/vllm/blob/minimax-m3-compressed-tensors/vllm/model_executor/layers/fused_moe/runner/shared_experts.py>
- MoE combine point:
  <https://github.com/toncao/vllm/blob/minimax-m3-compressed-tensors/vllm/model_executor/layers/fused_moe/runner/moe_runner.py>
- Environment defaults:
  <https://github.com/toncao/vllm/blob/minimax-m3-compressed-tensors/vllm/envs.py>
- PyTorch `wait_stream` semantics:
  <https://docs.pytorch.org/docs/stable/generated/torch.Stream.wait_stream.html>

These sources are the baseline. The hypothesis is not considered confirmed
until the threshold fingerprint and stream-disable control agree.

---

## 2. Safety and reproducibility gates

Run from the NFS clone:

```bash
cd /mnt/nfs/hoangduy/llm-compressor
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export HOME=/mnt/nfs/hoangduy
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"
```

Record repository and environment identity:

```bash
git status --short --branch
git rev-parse HEAD
python -c "import torch, vllm; print('torch', torch.__version__); print('vllm', vllm.__version__)"
python pipeline/slurm/patch_vllm_m3_serve.py --check
```

Required preflight result:

- the intended branch/commit is checked out;
- no tracked local changes affect the harness;
- patch check reports `4/4`;
- FlashInfer is at least `0.6.10`;
- the same checkpoint used by the first matrix exists at
  `/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4`.

Verify that the installed vLLM has the exact control knobs and suspect code:

```bash
env -u VLLM_DISABLE_SHARED_EXPERTS_STREAM \
    -u VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD \
python - <<'PY'
import inspect
import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

print("disable_default =", envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM)
print("threshold_default =", envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD)
print(inspect.getsource(SharedExperts._run_in_aux_stream))
assert envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM is False
assert envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD == 256
source = inspect.getsource(SharedExperts._run_in_aux_stream)
assert "current_stream().wait_stream(self._stream)" in source
PY
```

If any assertion fails, stop. Save the output and report that this runbook does
not match the installed vLLM; do not reinterpret results from a different
implementation.

Check the node before claiming GPUs:

```bash
hostname
nvidia-smi
FORCE=0 bash pipeline/slurm/free_gpus.sh
```

The cluster agent must now show this context and ask for approval. The matrix
starts CUDA work, writes under `/mnt/nfs/hoangduy`, and terminates only process
groups it launched. Never signal another user's process.

After approval, and only if all eight GPUs are free, pin the workload:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

---

## 3. Fixed experiment contract

All three conditions must use:

```text
checkpoint:       cyankiwi/MiniMax-M3-AWQ-INT4
TP:               8
expert parallel:  enabled
max model length: 8192
GPU utilization:  0.9
graphs:           enabled
breakable graphs: enabled/default
async CUDA:       enabled (CUDA_LAUNCH_BLOCKING unset)
trials:           async_baseline_1, async_baseline_2, async_baseline_3
```

Do not add `DEBUG_CUDAGRAPH=1`, coredumps, Compute Sanitizer, or source patches.
Those change timing or introduce a second variable.

After approval, create one session ID and result root:

```bash
export RCA_SESSION="$(date +%Y%m%d-%H%M%S)"
export RCA_RESULTS_ROOT=/mnt/nfs/hoangduy/logs/m3-cudagraph-shared-stream
export RCA_CASES=async_baseline_1,async_baseline_2,async_baseline_3
echo "RCA_SESSION=$RCA_SESSION"
echo "RCA_RESULTS_ROOT=$RCA_RESULTS_ROOT"
```

For every condition below:

1. Preserve the exact `RUN_ID`.
2. Write `condition.env` before launch.
3. Run all three trials even if one fails; the runner classifies each trial.
4. Do not reuse or rename a run directory.

---

## 4. Condition A — threshold 256 control

This explicitly reproduces the fork default. It establishes the current
failure rate and expected transition near `16/51`.

```bash
export RUN_ID="${RCA_SESSION}-control-t256"
export RUN_DIR="$RCA_RESULTS_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
cat >"$RUN_DIR/condition.env" <<EOF
condition=control-t256
VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256
CUDA_LAUNCH_BLOCKING=unset
TORCH_USE_CUDA_DSA=unset
MATRIX_CASES=$RCA_CASES
EOF

env -u CUDA_LAUNCH_BLOCKING \
    -u TORCH_USE_CUDA_DSA \
    VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256 \
    RESULTS_ROOT="$RCA_RESULTS_ROOT" \
    RUN_ID="$RUN_ID" \
    MATRIX_CASES="$RCA_CASES" \
    bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
```

Expected signal: at least one trial reports an IMA after approximately 16 graph
sizes have completed. A pass is allowed because the original failure was flaky.

If all three control trials pass, do not immediately reject the hypothesis.
Complete Conditions B and C, then follow the bounded-rerun rule in Section 8.

---

## 5. Condition B — disable only the shared-expert auxiliary stream

This preserves asynchronous CUDA and breakable CUDA graphs while removing the
suspected overlap.

```bash
export RUN_ID="${RCA_SESSION}-stream-disabled"
export RUN_DIR="$RCA_RESULTS_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
cat >"$RUN_DIR/condition.env" <<EOF
condition=stream-disabled
VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256
CUDA_LAUNCH_BLOCKING=unset
TORCH_USE_CUDA_DSA=unset
MATRIX_CASES=$RCA_CASES
EOF

env -u CUDA_LAUNCH_BLOCKING \
    -u TORCH_USE_CUDA_DSA \
    VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 \
    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256 \
    RESULTS_ROOT="$RCA_RESULTS_ROOT" \
    RUN_ID="$RUN_ID" \
    MATRIX_CASES="$RCA_CASES" \
    bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
```

Expected signal: all three trials capture `51/51`, become ready, and return a
chat response. In `summary.json`, each should be `server_ready` with
`chat_ok: true`.

This is a narrow workaround test, not proof by itself. Because one original
baseline trial passed, the threshold-shift condition is also required.

---

## 6. Condition C — move the overlap threshold to 128

This keeps the auxiliary stream enabled but changes where it first becomes
eligible during largest-first graph capture.

```bash
export RUN_ID="${RCA_SESSION}-control-t128"
export RUN_DIR="$RCA_RESULTS_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
cat >"$RUN_DIR/condition.env" <<EOF
condition=control-t128
VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=128
CUDA_LAUNCH_BLOCKING=unset
TORCH_USE_CUDA_DSA=unset
MATRIX_CASES=$RCA_CASES
EOF

env -u CUDA_LAUNCH_BLOCKING \
    -u TORCH_USE_CUDA_DSA \
    VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=128 \
    RESULTS_ROOT="$RCA_RESULTS_ROOT" \
    RUN_ID="$RUN_ID" \
    MATRIX_CASES="$RCA_CASES" \
    bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
```

Expected signal: failing trials move from approximately `16/51` to
approximately `32/51`. A trial may still pass because the race is asynchronous.

The important result is a repeatable movement of the failure boundary, not the
classifier's `graph_ima_collective` label. That label is based on symbols found
anywhere in a log; successful runs also contain collective symbols.

---

## 7. Produce the comparison artifact

Create one machine-readable comparison from the three runs:

```bash
python - "$RCA_RESULTS_ROOT" "$RCA_SESSION" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
session = sys.argv[2]
conditions = [
    f"{session}-control-t256",
    f"{session}-stream-disabled",
    f"{session}-control-t128",
]
rows = []

for run_id in conditions:
    run_dir = root / run_id
    condition_file = run_dir / "condition.env"
    condition_text = condition_file.read_text(encoding="utf-8")
    condition = dict(
        line.split("=", 1)
        for line in condition_text.splitlines()
        if line and "=" in line
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    by_case = {row["case"]: row for row in summary["trials"]}

    for case in ("async_baseline_1", "async_baseline_2", "async_baseline_3"):
        trial = by_case[case]
        log_path = run_dir / case / "serve.log"
        text = log_path.read_text(encoding="utf-8", errors="replace")
        normalized = text.replace("\r", "\n")
        progress = [
            (int(done), int(total))
            for done, total in re.findall(r"(\d+)\s*/\s*(\d+)", normalized)
            if int(total) == 51
        ]
        signal_lines = [
            line[-500:]
            for line in normalized.splitlines()
            if (
                "Capturing CUDA graphs" in line
                or "Graph capturing finished" in line
                or "illegal memory access" in line.lower()
            )
        ][-20:]
        rows.append(
            {
                "condition": condition["condition"],
                "run_id": run_id,
                "case": case,
                "disable_shared_experts_stream": condition[
                    "VLLM_DISABLE_SHARED_EXPERTS_STREAM"
                ],
                "threshold": condition[
                    "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD"
                ],
                "verdict": trial.get("verdict"),
                "server_ready": trial.get("server_ready"),
                "chat_ok": trial.get("chat_ok"),
                "ima": trial.get("ima"),
                "last_capture_progress": list(progress[-1]) if progress else None,
                "capture_finished": "Graph capturing finished" in normalized,
                "signal_lines": signal_lines,
            }
        )

report = {
    "session": session,
    "result_root": str(root),
    "hypothesis": "missing main-stream join after shared-expert auxiliary stream",
    "rows": rows,
}
out = root / f"{session}-comparison.json"
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
print(f"\ncomparison={out}")
PY
```

Then verify artifact completeness:

```bash
python - "$RCA_RESULTS_ROOT" "$RCA_SESSION" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
session = sys.argv[2]
comparison = root / f"{session}-comparison.json"
report = json.loads(comparison.read_text(encoding="utf-8"))
assert len(report["rows"]) == 9, len(report["rows"])
for row in report["rows"]:
    assert row["verdict"] is not None, row
    run_dir = root / row["run_id"] / row["case"]
    for name in ("serve.log", "chat.json", "meta.json", "result.json"):
        assert (run_dir / name).exists(), run_dir / name
print("ARTIFACT CHECK PASS: 9 classified trials with complete per-trial files")
print(comparison)
PY
```

Do not delete logs, failed core files, PID evidence, or intermediate JSON.

---

## 8. Decision rules and bounded reruns

### Strong confirmation

Classify the shared-expert stream race as **strongly confirmed** only if:

1. `stream-disabled` is `server_ready` with successful chat in all three trials;
2. at least one `control-t256` trial reproduces IMA near `16/51`; and
3. at least one `control-t128` trial reproduces IMA near `32/51`, rather than
   remaining near `16/51`.

Allow a one-step progress-bar offset because a crash can happen after a size is
launched but before tqdm records completion:

- threshold 256 expected window: `15/51` through `17/51`;
- threshold 128 expected window: `31/51` through `33/51`.

### Bounded rerun when the control is too lucky

If all three `control-t256` trials pass, rerun that condition once with a new
run ID, for six total threshold-256 observations:

```bash
export RUN_ID="${RCA_SESSION}-control-t256-rerun"
export RUN_DIR="$RCA_RESULTS_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
cat >"$RUN_DIR/condition.env" <<EOF
condition=control-t256-rerun
VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256
CUDA_LAUNCH_BLOCKING=unset
TORCH_USE_CUDA_DSA=unset
MATRIX_CASES=$RCA_CASES
EOF

env -u CUDA_LAUNCH_BLOCKING \
    -u TORCH_USE_CUDA_DSA \
    VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256 \
    RESULTS_ROOT="$RCA_RESULTS_ROOT" \
    RUN_ID="$RUN_ID" \
    MATRIX_CASES="$RCA_CASES" \
    bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
```

Do not perform more than this one extra three-trial run without a new user
decision.

### Reject or narrow the hypothesis

- If any `stream-disabled` trial still produces an IMA, the missing
  shared-expert join is not sufficient to explain the failure. Preserve logs
  and stop before patching.
- If threshold 128 still fails near `16/51`, verify worker inheritance of the
  environment variables. If inheritance is confirmed, reject the proposed
  threshold fingerprint.
- If all async trials pass across all conditions, report **inconclusive**. Do
  not claim a fix from absence of a flaky failure.
- If failures move with the threshold but stream-disabled chat is incorrect,
  report a shared-stream correctness dependency and keep the hypothesis open;
  do not call it resolved.

---

## 9. Required cluster-agent report

Write the report to:

```bash
export RCA_REPORT="$RCA_RESULTS_ROOT/${RCA_SESSION}-RCA_REPORT.md"
echo "$RCA_REPORT"
```

The report must contain:

1. host, repository commit, torch/vLLM/FlashInfer versions, and checkpoint;
2. the exact three condition values;
3. all nine verdicts, `chat_ok`, and last capture progress values;
4. links/paths to each `summary.json` and the comparison JSON;
5. classification: strongly confirmed, rejected, narrowed, or inconclusive;
6. evidence for that classification using the decision rules above;
7. any rerun and why it was allowed;
8. the proposed next action, without implementing it.

Also append a concise result paragraph to the top CUDA-graph RCA section of
`BUGS_AND_FIXES.md`. Do not overwrite the earlier matrix result.

After checking the report and documentation for consistency, commit only the
repository documentation change according to the repository's normal commit
protocol. Report the commit hash and ask before pushing; do not push merely
because the experiment completed.

If the result is strongly confirmed, the proposed durable fix is:

```python
main_stream = current_stream()
with torch.cuda.stream(self._stream):
    output = self._layer(shared_experts_input)
main_stream.wait_stream(self._stream)
```

That code is a **proposal**, not part of this experiment. It must be implemented
in the vLLM fork/repository with a regression test, then validated by rerunning
the threshold-256 control. Do not patch `site-packages` in this run.

The narrow operational workaround, pending a source fix, is:

```bash
VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
```

This is preferable to global `CUDA_LAUNCH_BLOCKING=1` only if all three
stream-disabled trials pass capture and chat.

---

## 10. Stop conditions

Stop and report immediately if:

- GPUs are occupied by another user;
- the installed source does not contain the expected env controls or suspect
  self-wait;
- patch/version/checkpoint preflight fails;
- a trial fails for OOM, missing model files, authentication, or disk quota
  instead of CUDA IMA;
- cleanup would require signaling a process not proven to belong to
  `hoangduy`;
- the result root is outside `/mnt/nfs/hoangduy`.

Do not:

- kill unknown or collaborator processes;
- modify vLLM/checkpoint/site-packages during the matrix;
- enable `DEBUG_CUDAGRAPH`, `CUDA_LAUNCH_BLOCKING`, or Compute Sanitizer;
- relabel a `masked_pass` as fixed;
- infer causality from `graph_ima_collective` alone;
- push from the cluster unless the user separately authorizes it.

---

## Paste-ready prompt for the cluster agent

```text
Read CLUSTER_SHARED_EXPERT_STREAM_RCA_PLAN.md completely. Execute it in order.
Obey its safety/approval gate before GPU work or NFS writes. Keep the model,
async CUDA, breakable graphs, TP8/EP, and HTTP launcher fixed; vary only the
shared-expert stream controls. Produce all three run directories, the comparison
JSON, the RCA report, and the concise BUGS_AND_FIXES.md update. Do not patch
vLLM or site-packages. Apply the decision rules literally and stop on any listed
stop condition.
```
