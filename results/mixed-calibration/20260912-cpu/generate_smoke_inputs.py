"""Recreate synthetic CPU smoke inputs without storing repeated text in Git."""

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    generic = [
        {
            "id": str(i),
            "messages": [
                {
                    "role": "user",
                    "content": "Explain how to organize a project. " * 120 + str(i),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Start by separating concerns "
                        "and writing a clear interface. "
                    )
                    * 60,
                },
            ],
        }
        for i in range(8)
    ]
    swe = []
    for i in range(8):
        session_id = f"synthetic-session-{i}"
        turns = [
            ("metadata", "system_event", "not calibration content " * 1000, {}),
            (
                "user",
                "user_prompt",
                "Investigate the boundary failure. "
                + "Existing project overview and setup. " * 700,
                {},
            ),
            (
                "assistant",
                "assistant_thinking",
                (
                    "The failure suggests an inclusive upper bound. "
                    "Check the existing test and inspect the loop. "
                )
                * 20,
                {},
            ),
            (
                "assistant",
                "assistant_response",
                "I will inspect the failing test and the loop.",
                {},
            ),
            (
                "tool_use",
                "tool_use",
                "",
                {
                    "tool_name": "Bash",
                    "tool_call_id": f"call-{i}-0",
                    "tool_input_json": json.dumps(
                        {"command": f"pytest tests/test_boundary_{i}.py"}
                    ),
                },
            ),
            (
                "tool_result",
                "tool_result",
                "AssertionError: expected 3, got 4. "
                + "Stack frame and execution context. " * 200,
                {"tool_name": "Bash", "tool_call_id": f"call-{i}-0"},
            ),
            (
                "assistant",
                "assistant_response",
                (
                    "The loop includes one extra item. "
                    "Change while index <= limit to while index < limit, "
                    "preserving the empty case. "
                )
                * 30,
                {},
            ),
            (
                "tool_use",
                "tool_use",
                "",
                {
                    "tool_name": "Edit",
                    "tool_call_id": f"call-{i}-1",
                    "tool_input_json": json.dumps(
                        {
                            "file_path": f"src/boundary_{i}.py",
                            "old_string": "while index <= limit:",
                            "new_string": "while index < limit:",
                        }
                    ),
                },
            ),
            (
                "tool_result",
                "tool_result",
                "File updated successfully.",
                {"tool_name": "Edit", "tool_call_id": f"call-{i}-1"},
            ),
            (
                "assistant",
                "assistant_response",
                (
                    "Rerun the regression tests including empty input, "
                    "single input and maximum boundary. "
                )
                * 25,
                {},
            ),
        ]
        for n, (role, kind, content, extra) in enumerate(turns):
            row = dict(
                session_id=session_id,
                turn_id=f"{session_id}#{n}",
                turn_number=n,
                role=role,
                turn_type=kind,
                content=content,
                tool_name=None,
                tool_call_id=None,
                tool_input_json=None,
            )
            row.update(extra)
            swe.append(row)
    for name, rows in [("generic", generic), ("swe", swe)]:
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    config = dict(
        num_samples=8,
        max_seq_length=128,
        seed=42,
        sources=[
            dict(
                name="generic",
                weight=0.5,
                dataset_id="json",
                dataset_split="train",
                dataset_data_files=str(output / "generic.jsonl"),
                format="messages",
                id_column="id",
            ),
            dict(
                name="coding_agentic",
                weight=0.5,
                dataset_id="json",
                dataset_split="train",
                dataset_data_files=str(output / "swe.jsonl"),
                format="swe_chat",
                id_column="session_id",
            ),
        ],
    )
    (output / "mix.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
