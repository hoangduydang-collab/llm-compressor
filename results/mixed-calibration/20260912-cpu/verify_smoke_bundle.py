"""Verify the synthetic smoke bundle with the actual GLM tokenizer on CPU."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from transformers import AutoTokenizer

from pipeline.calibration import (
    build_calibration_dataset_with_partition,
    calibration_partition_manifest,
)
from pipeline.calibration_bundle import tokenizer_identity
from pipeline.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision
    )
    manifest = json.loads((Path(args.bundle) / "manifest.json").read_text())
    assert manifest["realized_counts"] == {"generic": 4, "coding_agentic": 4}
    assert manifest["num_samples"] == 8
    assert manifest["max_seq_length"] == 128
    results = {}
    for name in (
        "glm53_ep_gptq_w4afp8_full_mixed",
        "glm53_distributed_w4afp8_awq_full_mixed",
    ):
        config = load_config(f"pipeline/configs/{name}.yaml")
        calibration = replace(
            config.calibration,
            prepared_dataset=args.bundle,
            num_samples=8,
            max_seq_length=128,
        )
        dataset, partition = build_calibration_dataset_with_partition(
            calibration, tokenizer
        )
        results[config.quantization.method] = calibration_partition_manifest(
            dataset, partition
        )
    assert results["gptq"]["token_ids_sha256"] == results["awq"]["token_ids_sha256"]
    windows = []
    for row, metadata in zip(dataset, manifest["samples"]):
        if metadata["source"] != "coding_agentic":
            continue
        decoded = tokenizer.decode(row["input_ids"])
        assert metadata["offset"] > 0
        # Check actual substantive text/tool payload, not only assistant markup.
        assert any(
            text in decoded
            for text in (
                "inclusive upper bound",
                "inspect the failing test",
                "one extra item",
                "regression tests including",
                "<tool_call>Bash",
                "<tool_call>Edit",
            )
        ), decoded
        windows.append({**metadata, "preview": decoded[:240]})
    before = tokenizer_identity(tokenizer)["sha256"]
    tokenizer(
        "test runtime tokenizer settings",
        padding="max_length",
        truncation=True,
        max_length=32,
    )
    assert tokenizer_identity(tokenizer)["sha256"] == before
    output = {
        "result": "PASS",
        "tokenizer": "GLM-5.3-BF16@304b8051cfb2b260b61ce0cbe330e02a98e73639",
        "counts": manifest["realized_counts"],
        "num_samples": 8,
        "max_seq_length": 128,
        "consumer_partitions": results,
        "selected_agent_windows": windows,
        "runtime_tokenizer_settings_do_not_change_identity": True,
        "scope": (
            "CPU synthetic data and real tokenizer; "
            "no model forward or quantization run"
        ),
    }
    with Path(args.output).open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
