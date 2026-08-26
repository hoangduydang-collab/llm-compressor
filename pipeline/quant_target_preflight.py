"""Fail-closed pre-flight for a quantization config's target/ignore regexes.

Answers, before any GPU is allocated: given this pipeline config, *exactly which
modules* will GPTQ/AWQ quantize, which will go FP8, and which stay untouched?

It exists because the failure mode it catches is silent. A regex written for one
model family and reused on another does not raise — it matches nothing, the run
succeeds, every int4-side gate passes, and the checkpoint quietly carries BF16
where FP8 was intended. That is what all three of MiniMax-M3's
`fp8_dynamic_targets` do when pointed at GLM-5.2: zero matches, because M3 nests
under `language_model.`, uses dense `(q|k|v|o)_proj` where GLM uses MLA, and
fuses `gate_up_proj` where GLM does not.

Method: build a dimension-shrunken stand-in of the target architecture that
keeps every *structural* property the regexes can see — the real
`num_hidden_layers`, the real dense/MoE split, shared-expert and indexer
presence, fused-vs-unfused projections — then linearize the MoE and apply the
config's own patterns to the resulting module names. No weights are downloaded;
only the architecture config is read.

Usage:
    python -m pipeline.quant_target_preflight --config pipeline/configs/foo.yaml
    python -m pipeline.quant_target_preflight --config ... --arch-config DIR
    python -m pipeline.quant_target_preflight --config ... --json out.json

Exit codes: 0 all checks passed, 1 a check failed, 2 could not build a stand-in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import torch
import yaml

# Dimensions shrunk to this scale; structure is preserved exactly.
_TINY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_routed_experts": 2,
    "num_local_experts": 2,
    "num_experts": 2,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "vocab_size": 256,
    "q_lora_rank": 32,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "index_head_dim": 16,
    "index_n_heads": 2,
}

# Structural fields that must survive shrinking, or the regexes see a different
# model than the run will.
_PRESERVE = (
    "num_hidden_layers",
    "first_k_dense_replace",
    "n_shared_experts",
    "model_type",
    "architectures",
    "indexer_types",
    "index_topk_freq",
    "moe_layer_freq",
    "num_nextn_predict_layers",
)

# Name-based role classification. Deliberately pattern-based rather than
# class-based so it works across DeepSeek-like MoE families without a registry.
_ROLES = {
    "routed_expert": r"\.mlp\.experts\.\d+\.",
    "indexer": r"\.self_attn\.indexer\.",
    "attention": r"\.self_attn\.(?!indexer\.)",
    "shared_expert": r"\.shared_experts\.",
    "dense_mlp": r"\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)$",
    "head": r"^lm_head$",
}


def classify(name: str) -> str:
    for role, pat in _ROLES.items():
        if re.search(pat, name):
            return role
    return "other"


def match(patterns: Iterable[str], names: Iterable[str]) -> set[str]:
    """Apply llm-compressor target syntax ('re:' prefix, else exact) to names."""
    names = list(names)
    out: set[str] = set()
    for p in patterns or []:
        rx = p[3:] if p.startswith("re:") else "^" + re.escape(p) + "$"
        out |= {n for n in names if re.search(rx, n)}
    return out


def build_stand_in(arch_config) -> list[str]:
    """Return the Linear module names of a shrunken stand-in, MoE linearized."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from llmcompressor.modeling.moe.linearize import linearize_moe

    preserved = {
        k: getattr(arch_config, k)
        for k in _PRESERVE
        # model_type is passed positionally; architectures is derived
        if hasattr(arch_config, k) and k not in ("model_type", "architectures")
    }
    cfg = AutoConfig.for_model(arch_config.model_type, **{
        **{k: v for k, v in _TINY.items() if hasattr(arch_config, k)},
        **preserved,
    })
    # Some configs derive per-layer lists from dims; re-assert the preserved ones.
    for k, v in preserved.items():
        try:
            setattr(cfg, k, v)
        except Exception:  # noqa: BLE001 - read-only property, nothing to do
            pass
    cfg.dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)
    linearize_moe(model)
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="pipeline config yaml")
    ap.add_argument(
        "--arch-config",
        help=(
            "path to the model's config.json "
            "(default: resolve model.id via AutoConfig)"
        ),
    )
    ap.add_argument("--json", dest="json_out", help="write the resolved sets here")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    quant = cfg.get("quantization") or {}
    model_id = (cfg.get("model") or {}).get("id")

    from transformers import AutoConfig

    if args.arch_config:
        arch = AutoConfig.from_pretrained(args.arch_config, trust_remote_code=True)
    else:
        arch = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    try:
        names = build_stand_in(arch)
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        print(
            f"could not build a stand-in for {arch.model_type}: {exc}",
            file=sys.stderr,
        )
        return 2

    roles: dict[str, set[str]] = {}
    for n in names:
        roles.setdefault(classify(n), set()).add(n)

    ignore = match(quant.get("ignore"), names)
    fp8 = match(quant.get("fp8_dynamic_targets"), names)
    weight_q = set(names) - ignore  # what the GPTQ/AWQ modifier is left holding

    print(f"config      : {args.config}")
    n_layers = getattr(arch, "num_hidden_layers", "?")
    print(f"model       : {model_id}  ({arch.model_type}, {n_layers} layers)")
    print(f"stand-in    : {len(names)} Linear modules")
    print()
    print("module roles:")
    for role in sorted(roles):
        print(f"  {role:16} {len(roles[role]):5}")
    print()

    failures: list[str] = []

    print(f"{quant.get('method', '?')} / {quant.get('scheme', '?')} targets "
          f"(everything not ignored): {len(weight_q)}")
    for role in sorted(roles):
        n = len(roles[role] & weight_q)
        if n:
            print(f"  {role:16} {n:5}")
    print()

    print(f"fp8_dynamic_targets: {len(fp8)}")
    for p in quant.get("fp8_dynamic_targets") or []:
        hit = match([p], names)
        if not hit:
            failures.append(f"fp8 target matches nothing (dead pattern): {p}")
        print(f"  [{'OK' if hit else 'DEAD':4}] {len(hit):5}  {p}")
    print()

    print("ignore list:")
    for p in quant.get("ignore") or []:
        hit = match([p], names)
        print(f"  [{'OK' if hit else 'none':4}] {len(hit):5}  {p}")
    print()

    invariants = [
        ("weight-quant and FP8 sets are disjoint", not (weight_q & fp8)),
        ("lm_head is not quantized", not ({"lm_head"} & (weight_q | fp8))),
        ("indexer is never quantized",
         not (roles.get("indexer", set()) & (weight_q | fp8))),
        ("no routed expert is in the FP8 set",
         not (roles.get("routed_expert", set()) & fp8)),
        ("at least one routed expert is weight-quantized",
         bool(roles.get("routed_expert", set()) & weight_q)),
    ]
    for label, ok in invariants:
        print(f"  [{'PASS' if ok else 'FAIL':4}] {label}")
        if not ok:
            failures.append(label)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "config": args.config,
                    "model_id": model_id,
                    "model_type": arch.model_type,
                    "num_hidden_layers": getattr(arch, "num_hidden_layers", None),
                    "n_linears": len(names),
                    "roles": {k: sorted(v) for k, v in roles.items()},
                    "weight_quant_targets": sorted(weight_q),
                    "fp8_targets": sorted(fp8),
                    "ignored": sorted(ignore),
                    "failures": failures,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S)")
        for f in failures:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
