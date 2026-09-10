"""Render and validate the immutable GLM-5.3 EP GPTQ Rancher chain."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "glm53-ep-gptq-chain.yaml.tmpl"
DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _dns_label(value: str, field: str) -> str:
    if len(value) > 63 or not DNS_LABEL.fullmatch(value):
        raise ValueError(f"{field} must be a Kubernetes DNS label: {value!r}")
    return value


def render(*, run_tag: str, ref: str, node: str = "") -> str:
    _dns_label(run_tag, "run tag")
    if not COMMIT.fullmatch(ref):
        raise ValueError("ref must be a full 40-character lowercase Git commit")
    if node:
        _dns_label(node, "node")
        node_selector = (
            "      nodeSelector:\n"
            f"        kubernetes.io/hostname: {node}"
        )
    else:
        node_selector = "      # nodeSelector intentionally omitted"

    text = TEMPLATE.read_text(encoding="utf-8")
    text = (
        text.replace("@@RUN_TAG@@", run_tag)
        .replace("@@REPO_REF@@", ref)
        .replace("@@NODESELECTOR@@", node_selector)
    )
    unresolved = sorted(set(re.findall(r"@@[A-Z_]+@@", text)))
    if unresolved:
        raise ValueError(f"unresolved template tokens: {unresolved}")
    document = yaml.safe_load(text)
    if document["spec"]["template"]["spec"]["containers"][0]["resources"][
        "requests"
    ]["nvidia.com/gpu"] != 8:
        raise ValueError("rendered chain must request exactly eight GPUs")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--node", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    text = render(run_tag=args.run_tag, ref=args.ref, node=args.node)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    persisted = yaml.safe_load(args.out.read_text(encoding="utf-8"))
    container = persisted["spec"]["template"]["spec"]["containers"][0]
    print(f"job={persisted['metadata']['name']}")
    print(f"ref={args.ref}")
    print(f"gpus={container['resources']['requests']['nvidia.com/gpu']}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
