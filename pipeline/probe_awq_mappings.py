"""Diagnose AWQ smooth/balance mapping resolution for MiniMax-M3 on a meta model.

The sequential AWQ path resolves mappings via
``compressed_tensors.utils.match.match_modules_set``, which groups a smooth layer
with its balance layers *per parent context* (i.e. per decoder layer). If a balance
target does not match uniformly in every layer, the per-layer group never closes and
all smooth layers collapse into one set, producing::

    ValueError: AWQ needs to match a single smoothlayer for each mapping ...

This probe builds the model on the meta device (no weights, seconds) and reports, for
each registered M3 AWQ mapping:

* how many modules each target matches (and a few example names),
* whether every sparse decoder layer exposes ``self_attn.indexer.{q,k}_proj``,
* the number of groups ``match_modules_set`` yields and any group with >1 smooth.

Usage::

    python -m pipeline.probe_awq_mappings --config pipeline/configs/minimax_m3.yaml
"""

from __future__ import annotations

import argparse
import re

from pipeline.config import load_config
from pipeline.minimax_m3_config import (
    load_minimax_m3_vl_config,
    get_minimax_m3_awq_mappings,
)


def _build_meta_model(cfg):
    import transformers
    from accelerate import init_empty_weights

    config = load_minimax_m3_vl_config(
        cfg.model.id, trust_remote_code=cfg.model.trust_remote_code
    )
    model_cls = getattr(transformers, cfg.model.auto_class)
    with init_empty_weights():
        model = model_cls.from_config(config, trust_remote_code=cfg.model.trust_remote_code)
    return model


def _simulate_linearized_experts(model, n_experts: int) -> int:
    """Replace fused ``mlp.experts`` with a ModuleList mimicking ``linearize_moe``.

    ``linearize_moe`` turns the fused 3D experts into a ``ModuleList`` of per-expert
    modules exposing split ``gate_proj`` / ``up_proj`` / ``down_proj`` Linears. That only
    happens on the real load, so this stubs the same *names* (tiny Linears) on the meta
    model to validate the routed-expert mapping grouping (post-attention-norm MoE-input;
    the up->down mapping was removed in r6 — see minimax_m3_config) without a full
    load. Expert count does not affect grouping, so a small N is used for speed.
    """
    from torch import nn

    replaced = 0
    for name, _module in list(model.named_modules()):
        if not name.endswith(".mlp.experts") or "language_model" not in name:
            continue
        experts = nn.ModuleList()
        for _ in range(n_experts):
            expert = nn.Module()
            expert.gate_proj = nn.Linear(1, 1)
            expert.up_proj = nn.Linear(1, 1)
            expert.down_proj = nn.Linear(1, 1)
            experts.append(expert)
        model.set_submodule(name, experts)
        replaced += 1
    return replaced


def _report_indexer_coverage(module_names: list[str]) -> None:
    layer_re = re.compile(r"\.layers\.(\d+)\.")
    all_layers: set[int] = set()
    indexer_layers: set[int] = set()
    for name in module_names:
        m = layer_re.search(name)
        if not m:
            continue
        idx = int(m.group(1))
        all_layers.add(idx)
        if ".self_attn.indexer." in name:
            indexer_layers.add(idx)
    sparse_like = sorted(i for i in all_layers if i >= 3)
    missing = [i for i in sparse_like if i not in indexer_layers]
    print("== indexer coverage ==")
    print(f"  decoder layers seen: {min(all_layers)}..{max(all_layers)}")
    print(f"  layers with self_attn.indexer.*: {sorted(indexer_layers)[:5]}"
          f"{' ...' if len(indexer_layers) > 5 else ''} (count={len(indexer_layers)})")
    print(f"  sparse-range layers (>=3) WITHOUT indexer: "
          f"{missing[:10]}{' ...' if len(missing) > 10 else ''} (count={len(missing)})")
    print()


def _match_name(name: str, target: str) -> bool:
    if target.startswith("re:"):
        return re.match(target.removeprefix("re:"), name) is not None
    return target == name


def _report_mapping(model, mapping, module_names: list[str]) -> None:
    from compressed_tensors.utils.match import match_modules_set

    print(f"== mapping smooth={mapping.smooth_layer} ==")
    targets = (mapping.smooth_layer, *mapping.balance_layers)
    for target in targets:
        hits = [n for n in module_names if _match_name(n, target)]
        example = hits[:3]
        print(f"  target {target!r}: {len(hits)} matches e.g. {example}")

    try:
        groups = list(match_modules_set(model, targets))
    except Exception as exc:  # noqa: BLE001 - report, don't crash the probe
        print(f"  match_modules_set RAISED: {type(exc).__name__}: {str(exc)[:200]}")
        print()
        return

    multi = [g for g in groups if len(g[0]) > 1]
    print(f"  groups yielded: {len(groups)}; groups with >1 smooth: {len(multi)}")
    if multi:
        print(f"  FIRST bad group smooth count: {len(multi[0][0])}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="probe AWQ mapping resolution (meta)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--simulate-linearized",
        type=int,
        default=0,
        metavar="N",
        help="stub N per-expert Linears (gate/up/down) to validate routed-expert "
        "mapping grouping on meta (mimics linearize_moe). 0 disables (default).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.model_id:
        cfg.model.id = args.model_id

    print(f"[probe] building meta model for {cfg.model.id} ...")
    model = _build_meta_model(cfg)
    if args.simulate_linearized > 0:
        n = _simulate_linearized_experts(model, args.simulate_linearized)
        print(f"[probe] simulated linearized experts on {n} MoE layers "
              f"({args.simulate_linearized} experts each)")
    module_names = [name for name, _ in model.named_modules()]
    print(f"[probe] total modules: {len(module_names)}\n")

    _report_indexer_coverage(module_names)

    fused_experts = [n for n in module_names if re.search(r"\.mlp\.experts\.gate_up_proj$", n)]
    if fused_experts:
        print("== routed experts (meta) ==")
        print(f"  fused experts modules present: {len(fused_experts)} "
              f"e.g. {fused_experts[:2]}")
        print("  NOTE: the MoE-input mapping targets split `mlp.experts.N.*_proj`, "
              "which exist only AFTER linearize_moe on the real load. 0 matches / an "
              "'incomplete set' here for that mapping is EXPECTED on the meta model; "
              "validate it in the smoke run. Attention mappings are "
              "authoritative here.")
        print()

    for mapping in get_minimax_m3_awq_mappings():
        _report_mapping(model, mapping, module_names)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
