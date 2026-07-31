"""Fail-closed harness check (eval contract) — runs BEFORE any eval request.

Records and verifies, machine-readably: harness version, task registry
resolution, tokenizer + chat-template hashes (must be IDENTICAL across arms),
generation params, serving topology, dataset cache presence, endpoint health,
and the sample-manifest rule. Exits non-zero on any failure.
"""

import hashlib
import json
import os
import sys
import urllib.request

OUT = sys.argv[1]
BF16_DIR = "/mnt/nfs/hoangduy/hf_assets/Qwen/Qwen3-30B-A3B-Instruct-2507"
W2A16_DIR = (
    "/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/"
    "Qwen3-30B-A3B-Instruct-2507-autoround-W2A16-g128-ddp8"
)
ENDPOINTS = {
    "qwen3-30b-bf16": "http://localhost:8410",
    "qwen3-30b-w2a16": "http://localhost:8411",
}
TASKS = ["gpqa_diamond_cot_zeroshot", "ifeval"]
DATASET_CACHES = [
    "/mnt/nfs/hoangduy/cache/huggingface/datasets/Idavidrein___gpqa",
    "/mnt/nfs/hoangduy/cache/huggingface/datasets/google___if_eval",
]

failures = []
record = {"check": "sub4bit-w2a16-vs-bf16 gpqa+ifeval paired subset"}

# 1. harness version
from importlib.metadata import version  # noqa: E402

lm_eval_version = version("lm_eval")
record["lm_eval_version"] = lm_eval_version
if lm_eval_version != "0.4.10":
    failures.append(f"lm_eval version {lm_eval_version} != pinned 0.4.10")

# 2. task registry resolution + versions
import lm_eval.tasks as lm_tasks  # noqa: E402

tm = lm_tasks.TaskManager()
task_index = {}
for t in TASKS:
    if t not in tm.all_tasks:
        failures.append(f"task {t} not in lm-eval registry")
    else:
        task_index[t] = tm.task_index.get(t, {}).get("yaml_path", "builtin")
record["tasks"] = task_index
record["num_fewshot"] = {"gpqa_diamond_cot_zeroshot": 0, "ifeval": 0}
record["metrics"] = {
    "gpqa_diamond_cot_zeroshot": "exact_match,flexible-extract",
    "ifeval": "prompt_level_strict_acc",
}

# 3. tokenizer + chat template must be SEMANTICALLY identical across arms.
# (tokenizer.json bytes differ between arms from save_pretrained re-serialization;
# verified semantically equal 2026-07-31 — compare vocab/encodings, record file
# hashes for provenance only.)
from transformers import AutoTokenizer  # noqa: E402

PROBES = [
    "Hello world",
    "What is 17+25? éè中文 test",
    "def f(x):\n    return x**2",
    "<|im_start|>user",
]
toks, hashes = {}, {}
for label, d in (("bf16", BF16_DIR), ("w2a16", W2A16_DIR)):
    tok = AutoTokenizer.from_pretrained(d)
    toks[label] = tok
    tj = os.path.join(d, "tokenizer.json")
    vocab_canon = json.dumps(sorted(tok.get_vocab().items()), ensure_ascii=False)
    hashes[label] = {
        "tokenizer_json_sha256_provenance": hashlib.sha256(
            open(tj, "rb").read()
        ).hexdigest(),
        "vocab_sha256": hashlib.sha256(vocab_canon.encode()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(
            (tok.get_chat_template() or "").encode()
        ).hexdigest(),
        "vocab_size": tok.vocab_size,
    }
record["tokenizer"] = hashes
for key in ("vocab_sha256", "chat_template_sha256"):
    if hashes["bf16"][key] != hashes["w2a16"][key]:
        failures.append(f"{key} differs between arms")
if any(
    toks["bf16"].encode(p) != toks["w2a16"].encode(p) for p in PROBES
):
    failures.append("probe encodings differ between arms")
if toks["bf16"].special_tokens_map != toks["w2a16"].special_tokens_map:
    failures.append("special_tokens_map differs between arms")

# 4. generation / sampling params (paired: identical for both arms)
record["generation"] = {
    "temperature": 0,
    "seed": 0,
    "max_gen_toks": 4096,
    "num_concurrent": 16,
    "reasoning_mode": "nonreasoning (Instruct-2507, no thinking tags)",
}

# 5. serving topology
record["serving"] = {
    "backend": "vllm 0.26.0 serve-sub4 (candidate: +PR48918 port +humming main)",
    "bf16": "TP2, 2xH100, port 8410",
    "w2a16": "TP1, 1xH100, port 8411",
    "same_node": True,
}

# 6. dataset caches present (offline mode)
for d in DATASET_CACHES:
    if not os.path.isdir(d):
        failures.append(f"dataset cache missing: {d}")
record["dataset_caches"] = DATASET_CACHES
record["hf_offline"] = {
    "HF_HOME": os.environ.get("HF_HOME"),
    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
}

# 7. endpoints healthy and serving the expected model names
for name, base in ENDPOINTS.items():
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
            served = [m["id"] for m in json.load(r)["data"]]
        if name not in served:
            failures.append(f"{base} serves {served}, expected {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"endpoint {base} not healthy: {exc}")
record["endpoints"] = ENDPOINTS

# 8. sample manifest rule + comparability statement
record["sample_manifest"] = (
    "lm-eval --limit 100 --seed 0: first 100 docs in the task's deterministic "
    "doc order for the pinned task version; identical across arms by construction "
    "(paired subset)."
)
record["comparability"] = (
    "PAIRED-SUBSET ONLY: valid for W2A16-vs-BF16 model-to-model decisions. NOT "
    "directly comparable to public leaderboard numbers (greedy decoding, "
    "100-sample subsets, max_gen_toks 4096)."
)

record["failures"] = failures
with open(OUT, "w") as f:
    json.dump(record, f, indent=2)
print(json.dumps({"failures": failures}, indent=2))
print("HARNESS_CHECK:", "FAIL" if failures else "PASS")
sys.exit(1 if failures else 0)
