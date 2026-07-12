"""Run and compare MiniMax-M3 teacher-forced prompt-logprob probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from pipeline.evalsuite.distributional import (
    compare_distributional_records,
    normalize_prompt_logprobs,
)
from pipeline.evalsuite.probe_corpus import build_probe_corpus


def _canonical_sha256(data: object) -> str:
    payload = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_probe_corpus(
    path: Path,
    prompts: list[dict],
    *,
    tokenizer_sha256: str,
    dataset: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    content = {
        "schema_version": 1,
        "seed": seed,
        "tokenizer_sha256": tokenizer_sha256,
        "dataset": dataset,
        "prompts": prompts,
    }
    content["sha256"] = _canonical_sha256(content)
    _write_json(path, content)
    return content


def load_probe_corpus(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("prompts"), list):
        raise ValueError("unsupported distributional corpus schema")
    declared = data.get("sha256")
    payload = {key: value for key, value in data.items() if key != "sha256"}
    computed = _canonical_sha256(payload)
    if declared != computed:
        raise ValueError(
            f"distributional corpus sha256 mismatch: declared={declared} computed={computed}"
        )
    return data


def _complete_probe_prompts(
    records: list[dict],
    corpus: list[dict],
    corpus_sha256: str,
) -> tuple[list[dict], set[str]]:
    by_prompt: dict[str, list[dict]] = {}
    for record in records:
        if record.get("corpus_sha256") != corpus_sha256:
            raise ValueError("existing distributional records use a different corpus")
        by_prompt.setdefault(str(record.get("prompt_id")), []).append(record)

    kept: list[dict] = []
    complete: set[str] = set()
    for prompt in corpus:
        prompt_id = str(prompt["prompt_id"])
        values = by_prompt.get(prompt_id, [])
        expected_positions = set(range(1, len(prompt["prompt_token_ids"])))
        positions = {int(record["position"]) for record in values}
        tokens_match = all(
            int(record["observed_token_id"])
            == int(prompt["prompt_token_ids"][int(record["position"])])
            for record in values
            if int(record["position"]) in expected_positions
        )
        if positions == expected_positions and tokens_match:
            complete.add(prompt_id)
            kept.extend(values)
    kept.sort(key=lambda record: (str(record["prompt_id"]), int(record["position"])))
    return kept, complete


def probe_with_engine(
    engine,
    *,
    sampling_params,
    corpus: list[dict],
    corpus_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    existing = _read_jsonl(output_path) if output_path.is_file() else []
    records, complete = _complete_probe_prompts(existing, corpus, corpus_sha256)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output_path)

    with output_path.open("a", encoding="utf-8") as handle:
        for prompt in corpus:
            if str(prompt["prompt_id"]) in complete:
                continue
            outputs = engine.generate(
                [{"prompt_token_ids": prompt["prompt_token_ids"]}],
                sampling_params,
                use_tqdm=False,
            )
            if len(outputs) != 1:
                raise ValueError(
                    f"expected one vLLM output for {prompt['prompt_id']}, "
                    f"got {len(outputs)}"
                )
            prompt_records = normalize_prompt_logprobs(
                outputs[0],
                {
                    "prompt_id": prompt["prompt_id"],
                    "length_bucket": prompt["length_bucket"],
                    "corpus_sha256": corpus_sha256,
                },
            )
            for record in prompt_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            records.extend(prompt_records)

    summary = {
        "schema_version": 1,
        "corpus_sha256": corpus_sha256,
        "prompts": len(corpus),
        "resumed_prompts": len(complete),
        "tokens": len(records),
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _prepare_minimax_runtime(model: Path, model_source: str) -> dict[str, Any]:
    from pipeline._env import ensure_writable_caches
    from pipeline.serve_verify import (
        _is_w4a8_moe_scheme,
        _read_quant_config,
        apply_minimax_m3_serve_env,
    )

    result: dict[str, Any] = {
        "cache_redirects": ensure_writable_caches(),
        "environment": apply_minimax_m3_serve_env(model),
        "runtime_patches": [],
    }
    if _is_w4a8_moe_scheme(_read_quant_config(model)):
        from pipeline.vllm_m3_patches import (
            patch_vllm_w4a8_swigluoai_uninterleave,
            read_swiglu_params,
        )

        limit, alpha, beta = read_swiglu_params(model, model_source)
        result["runtime_patches"] = patch_vllm_w4a8_swigluoai_uninterleave(
            limit, alpha, beta
        )
    return result


def build_vllm_engine_args(args: argparse.Namespace, model: Path) -> dict[str, Any]:
    """Build the shared vLLM probe arguments, including distributed layout."""
    result = {
        "model": str(model),
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "enable_expert_parallel": True,
        "block_size": 128,
        "kv_cache_dtype": args.kv_cache_dtype,
    }
    if args.distributed_executor_backend:
        result["distributed_executor_backend"] = args.distributed_executor_backend
    return result


def run_vllm_probe(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model).resolve()
    corpus = load_probe_corpus(Path(args.corpus))
    runtime = _prepare_minimax_runtime(model, args.model_source)

    from vllm import LLM, SamplingParams

    llm_kwargs = build_vllm_engine_args(args, model)
    engine = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=args.top_k,
        logprobs=args.top_k,
        seed=args.seed,
    )
    output = Path(args.out)
    summary = probe_with_engine(
        engine,
        sampling_params=sampling,
        corpus=corpus["prompts"],
        corpus_sha256=corpus["sha256"],
        output_path=output,
    )
    summary.update(
        {
            "model": str(model),
            "model_source": args.model_source,
            "runtime": runtime,
            "engine_args": llm_kwargs,
            "sampling": {
                "temperature": 0.0,
                "max_tokens": 1,
                "top_k": args.top_k,
                "seed": args.seed,
            },
            "python": platform.python_version(),
            "pid": os.getpid(),
        }
    )
    _write_json(output.with_suffix(".summary.json"), summary)
    return summary


def build_corpus(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_source,
        trust_remote_code=True,
    )
    tokenizer_payload = tokenizer.backend_tokenizer.to_str()
    tokenizer_sha256 = hashlib.sha256(tokenizer_payload.encode("utf-8")).hexdigest()
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        revision=args.dataset_revision,
    )
    prompts = build_probe_corpus(dataset[args.text_column], tokenizer, seed=args.seed)
    return write_probe_corpus(
        Path(args.out),
        prompts,
        tokenizer_sha256=tokenizer_sha256,
        dataset={
            "id": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "revision": args.dataset_revision,
            "text_column": args.text_column,
        },
        seed=args.seed,
    )


def compare(args: argparse.Namespace) -> dict[str, Any]:
    result = compare_distributional_records(
        _read_jsonl(Path(args.reference)),
        _read_jsonl(Path(args.candidate)),
    )
    _write_json(Path(args.out), result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus = subparsers.add_parser("build-corpus")
    corpus.add_argument("--model-source", required=True)
    corpus.add_argument("--dataset", default="Salesforce/wikitext")
    corpus.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    corpus.add_argument("--split", default="test")
    corpus.add_argument("--dataset-revision")
    corpus.add_argument("--text-column", default="text")
    corpus.add_argument("--seed", type=int, default=42)
    corpus.add_argument("--out", required=True)
    corpus.set_defaults(func=build_corpus)

    run = subparsers.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--model-source", required=True)
    run.add_argument("--corpus", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--tensor-parallel-size", type=int, default=8)
    run.add_argument("--pipeline-parallel-size", type=int, default=1)
    run.add_argument("--max-model-len", type=int, default=65_536)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.add_argument("--kv-cache-dtype", default="fp8")
    run.add_argument("--distributed-executor-backend")
    run.add_argument("--top-k", type=int, default=20)
    run.add_argument("--seed", type=int, default=42)
    run.set_defaults(func=run_vllm_probe)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--reference", required=True)
    comparison.add_argument("--candidate", required=True)
    comparison.add_argument("--out", required=True)
    comparison.set_defaults(func=compare)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
