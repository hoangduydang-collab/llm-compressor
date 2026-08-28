"""Compare a recipe's quantization SCOPE against the upstream vendor release.

Motivation
----------
A W4A8 checkpoint is meant to differ from the vendor's FP8 release in exactly one
way: the routed-expert weights carry 4 bits instead of 8. Every other decision --
which modules get scales at all, which stay at source precision -- should either
match the vendor or be a divergence we chose on purpose and can name.

Nothing in the pipeline checks that today. The recipe's `ignore` list and
`fp8_dynamic_targets` are hand-written regexes over a 59k-module tree, and the
failure mode is silent: a pattern that is too broad leaves a whole component in
BF16 and every existing gate still passes (the int4 side is untouched, the fold
audit only looks at what was folded, and quality evals absorb a few percent).
That is how the r8 checkpoint shipped with its FP8 modules shadowed.

Method
------
Upstream ground truth is the RELEASED WEIGHT INDEX, not `config.json`. A module
carrying `<name>.weight_scale_inv` was block-FP8 quantized; one carrying only
`<name>.weight` was left at source precision. `modules_to_not_convert` is a
statement of intent that can disagree with the artifact -- e.g. GLM-5.3 does not
list `self_attn.indexer.weights_proj` yet ships it in BF16, because its output
dim is 32 and a [128,128] block grid does not tile it. Reading the index avoids
having to model the vendor quantizer's skip rules at all.

Our ground truth is `re.match` for `re:`-prefixed targets and exact string
equality otherwise, which is precisely
`compressed_tensors.utils.match.match_name`. Order matters and mirrors
`pipeline/recipe.py`: the FP8_DYNAMIC modifier is constructed with
`targets=fp8_dynamic_targets` and NO ignore list, so an fp8 target wins over the
main modifier's `ignore`.

Two structural facts are handled rather than flagged:

* The MTP layer (index == num_hidden_layers) is reported separately. transformers
  builds `num_hidden_layers` decoder layers and models no MTP for any
  architecture, so those modules never exist in our process. Upstream quantizes
  them; we cannot, and the fix is a post-hoc graft, not a recipe change.
* The router (`mlp.gate`) is a bare Parameter on GlmMoeDsaTopkRouter rather than
  an nn.Linear, so `targets="Linear"` never reaches it. Upstream also leaves it
  at source precision, so this agrees -- but for two different reasons, and the
  recipe keeps it in `ignore` explicitly so it stays out of scope if that
  changes. (Note this is about QUANTIZING the router. The router must still be an
  AWQ *balance* layer, which is a different list; see
  tests/llmcompressor/modifiers/awq/test_mla_moe_router_balanced.py.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Divergences we have decided on, keyed by (component, upstream, ours). Anything
# not in here is a finding, not a policy.
#
# The indexer entry is the recipe's stated RED LINE: GLM's DSA indexer selects
# which tokens attend, and our own long-context retrieval regression from
# touching indexer precision is why the whole indexer stays BF16. Worth knowing
# when weighing it: upstream's choice here is not a quality judgement at all --
# it quantizes wq_b [4096,2048] and wk [128,6144] because the block grid tiles
# them, and skips weights_proj [32,6144] because it does not.
ACCEPTED_DIVERGENCES: dict[tuple[str, str, str], str] = {
    ("indexer.wq_b", "fp8", "src"): "RED LINE: DSA indexer stays BF16 (token-selection quality)",
    ("indexer.wk", "fp8", "src"): "RED LINE: DSA indexer stays BF16 (token-selection quality)",
}

# The ONE difference a W4A8 checkpoint is meant to have from the vendor's FP8
# release: the routed experts carry 4 bits instead of 8. Deliberately narrow --
# an earlier revision of this file treated `fp8 -> int4` as intended for any
# component, which would have waved through int4 on the MLA projections (they are
# in `fp8_dynamic_targets` today, so a typo dropping that entry sends them to the
# int4 modifier rather than to BF16, and the failure would not have been visible
# in the divergence list at all).
INTENDED_INT4_COMPONENTS = frozenset(
    {"routed expert gate_proj", "routed expert up_proj", "routed expert down_proj"}
)

NORM_SUFFIXES = (
    "input_layernorm",
    "post_attention_layernorm",
    "q_a_layernorm",
    "kv_a_layernorm",
    "k_norm",
    "shared_head.norm",
    "enorm",
    "hnorm",
)

_LAYER_RE = re.compile(r"(?:.*\.)?layers\.(\d+)\.(.*)$")


def match_name(name: str, target: str) -> bool:
    """Mirror of compressed_tensors.utils.match.match_name (re.match, not fullmatch)."""
    if target.startswith("re:"):
        return re.match(target.removeprefix("re:"), name) is not None
    return target == name


def layer_index(module: str) -> int | None:
    m = _LAYER_RE.match(module)
    return int(m.group(1)) if m else None


def component_of(module: str) -> str:
    """Structural label for a module, independent of any recipe."""
    if module in ("lm_head", "model.embed_tokens", "model.norm"):
        return module
    m = _LAYER_RE.match(module)
    if not m:
        return "OTHER:" + module
    rest = m.group(2)
    if rest.endswith(NORM_SUFFIXES):
        return "norm"
    if rest == "mlp.gate":
        return "router (mlp.gate)"
    if rest.startswith("self_attn.indexer."):
        return "indexer." + rest.split("self_attn.indexer.", 1)[1]
    if rest == "self_attn.indexers_proj":
        return "indexers_proj"
    if rest.startswith("self_attn."):
        return "MLA " + rest.split("self_attn.", 1)[1]
    if re.match(r"mlp\.experts\.\d+\.", rest):
        return "routed expert " + rest.rsplit(".", 1)[1]
    if rest.startswith("mlp.shared_experts."):
        return "shared expert " + rest.rsplit(".", 1)[1]
    if rest.startswith("mlp."):
        return "dense mlp " + rest.rsplit(".", 1)[1]
    if rest == "eh_proj":
        return "MTP eh_proj"
    return "OTHER:" + rest


def classify_upstream(weight_map: dict[str, str]) -> dict[str, str]:
    """module -> 'fp8' | 'src', from the presence of a block scale in the index."""
    scaled = {
        key[: -len(".weight_scale_inv")]
        for key in weight_map
        if key.endswith(".weight_scale_inv")
    }
    return {
        key[: -len(".weight")]: ("fp8" if key[: -len(".weight")] in scaled else "src")
        for key in weight_map
        if key.endswith(".weight")
    }


def classify_recipe(module: str, ignore: list[str], fp8_targets: list[str]) -> str:
    """'int4' | 'fp8' | 'src', under `targets="Linear"` plus the recipe's lists."""
    component = component_of(module)
    if component == "norm" or component.startswith("OTHER:"):
        return "src"
    if component in ("model.embed_tokens", "model.norm"):
        return "src"  # not Linear
    if component == "router (mlp.gate)":
        return "src"  # bare Parameter, and in `ignore` besides
    # recipe.py builds the FP8 modifier with targets=fp8_dynamic_targets and no
    # ignore, so an fp8 target wins over the main modifier's ignore list.
    if any(match_name(module, t) for t in fp8_targets):
        return "fp8"
    if any(match_name(module, t) for t in ignore):
        return "src"
    return "int4"


@dataclass
class Divergence:
    component: str
    upstream: str
    ours: str
    count: int
    example: str
    note: str = ""


@dataclass
class Report:
    depth: int
    rows: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    mtp_rows: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    covered_layers: set[int] = field(default_factory=set)
    real: list[Divergence] = field(default_factory=list)
    accepted: list[Divergence] = field(default_factory=list)
    layer_restricted: list[Divergence] = field(default_factory=list)
    mtp: list[Divergence] = field(default_factory=list)

    @property
    def partial(self) -> bool:
        """True when the recipe deliberately targets a subset of layers.

        Every smoke and probe config on this model carries a negative-lookahead
        pattern restricting the run to one or two sampled layers, and inside a
        sampled layer the FP8 targets are enumerated by hand for just a couple of
        modules. Such a recipe makes no scope claim at all, so measuring it
        against the vendor release produces true statements about the resulting
        checkpoint but not findings about the patterns. It is reported and does
        not fail; only a full-scope recipe is held to the comparison.
        """
        return self.covered_layers != set(range(self.depth))

    @property
    def ok(self) -> bool:
        return self.partial or not self.real


def compare(weight_map: dict[str, str], recipe: dict, depth: int) -> Report:
    quant = recipe["quantization"]
    ignore = list(quant.get("ignore") or [])
    fp8_targets = list(quant.get("fp8_dynamic_targets") or [])

    upstream = classify_upstream(weight_map)
    ours = {m: classify_recipe(m, ignore, fp8_targets) for m in upstream}

    def is_mtp(module: str) -> bool:
        index = layer_index(module)
        return index is not None and index >= depth

    report = Report(depth=depth)
    # Layers at or beyond `depth` are the MTP head, which is never built, so a
    # target matching one of them is not coverage.
    report.covered_layers = {
        layer_index(m)
        for m, verdict in ours.items()
        if verdict in ("int4", "fp8") and layer_index(m) is not None and not is_mtp(m)
    }

    # Group MTP separately from the body so a divergence is never reported as
    # part-real, part-out-of-reach.
    grouped: dict[tuple[str, str, str, bool], list[str]] = defaultdict(list)
    for module, up in upstream.items():
        our = ours[module]
        rows = report.mtp_rows if is_mtp(module) else report.rows
        rows[component_of(module)][(up, our)] += 1
        grouped[(component_of(module), up, our, is_mtp(module))].append(module)

    for (component, up, our, mtp_only), modules in grouped.items():
        if up == our:
            continue  # agreement
        if up == "fp8" and our == "int4" and component in INTENDED_INT4_COMPONENTS:
            continue  # the one intended difference
        div = Divergence(component, up, our, len(modules), sorted(modules)[0])
        if mtp_only:
            div.note = "MTP layer: transformers does not build it, so it is out of our reach"
            report.mtp.append(div)
        elif (component, up, our) in ACCEPTED_DIVERGENCES:
            div.note = ACCEPTED_DIVERGENCES[(component, up, our)]
            report.accepted.append(div)
        elif all((layer_index(m) not in report.covered_layers) for m in modules):
            div.note = "outside the layers this recipe targets (partial/smoke scope)"
            report.layer_restricted.append(div)
        else:
            report.real.append(div)
    return report


def format_report(report: Report, label: str) -> str:
    out = [f"### {label}", ""]
    verdict_of = {
        ("fp8", "int4"): "fp8 -> int4",
        ("fp8", "fp8"): "match (fp8)",
        ("src", "src"): "match (source precision)",
        ("fp8", "src"): "DIVERGENCE: upstream quantizes, we do not",
        ("src", "fp8"): "DIVERGENCE: we quantize, upstream does not",
        ("src", "int4"): "DIVERGENCE: we quantize, upstream does not",
    }
    def table(rows: dict[str, Counter]) -> list[str]:
        lines = [f"    {'component':<32} {'n':>6}  {'up':<4} {'ours':<5} verdict"]
        for component in sorted(rows, key=lambda c: (-sum(rows[c].values()), c)):
            for (up, our), n in sorted(rows[component].items(), key=lambda kv: -kv[1]):
                verdict = verdict_of.get((up, our), "?")
                if (up, our) == ("fp8", "int4"):
                    verdict += (
                        " INTENDED" if component in INTENDED_INT4_COMPONENTS
                        else " DIVERGENCE: int4 on a component that is not a routed expert"
                    )
                lines.append(f"    {component:<32} {n:>6}  {up:<4} {our:<5} {verdict}")
        return lines

    out += [f"  BODY (layers 0-{report.depth - 1} and the non-layer modules)"]
    out += table(report.rows)
    if report.mtp_rows:
        out += [
            "",
            f"  MTP HEAD (layer {report.depth}) -- not built by transformers, so nothing",
            "  here reflects work our pipeline can do; it is what a graft would have to",
            "  supply. Upstream quantizes it.",
        ]
        out += table(report.mtp_rows)
    for title, items in (
        (
            "differences from upstream inside the sampled layers (partial recipe: "
            "true of the checkpoint, not findings about the patterns)"
            if report.partial
            else "UNEXPLAINED DIVERGENCES",
            report.real,
        ),
        ("accepted divergences", report.accepted),
        ("out of scope: MTP layer", report.mtp),
        ("out of scope: layers this recipe does not target", report.layer_restricted),
    ):
        if not items:
            continue
        out += ["", f"    {title}:"]
        for d in sorted(items, key=lambda d: -d.count):
            out.append(f"      [{d.count:>6}] {d.component}: upstream={d.upstream} ours={d.ours}")
            out.append(f"               e.g. {d.example}")
            if d.note:
                out.append(f"               {d.note}")
    out += [
        "",
        f"    layers targeted by this recipe: {len(report.covered_layers)} of {report.depth}",
    ]
    if report.partial:
        out.append(
            "    RESULT: PARTIAL SCOPE -- this recipe targets a subset of layers, so it"
        )
        out.append(
            "            makes no scope claim to check. Gate a full-scope recipe instead."
        )
    else:
        out.append(
            f"    RESULT: {'scope matches upstream' if not report.real else 'SCOPE MISMATCH'}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="pipeline config yaml")
    parser.add_argument(
        "--upstream-index",
        required=True,
        help="upstream release model.safetensors.index.json (the FP8 release)",
    )
    parser.add_argument(
        "--upstream-config", required=True, help="upstream release config.json"
    )
    parser.add_argument("--label", default=None)
    args = parser.parse_args(argv)

    import yaml

    recipe = yaml.safe_load(Path(args.recipe).read_text())
    weight_map = json.loads(Path(args.upstream_index).read_text())["weight_map"]
    config = json.loads(Path(args.upstream_config).read_text())
    report = compare(weight_map, recipe, int(config["num_hidden_layers"]))
    label = args.label or f"{recipe.get('name', Path(args.recipe).stem)}"
    print(format_report(report, label))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
