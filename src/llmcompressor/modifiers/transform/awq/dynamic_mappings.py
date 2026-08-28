"""
Dynamic AWQ mapping builders for hybrid attention models.

Models with hybrid attention (mix of full self-attention and linear/Gated
DeltaNet attention) need layer-index-specific AWQ mappings that vary by
model size. This module provides runtime detection and mapping generation
for such architectures (e.g. Qwen3Next, Qwen3.5).
"""

import re
from collections.abc import Callable

from loguru import logger
from torch.nn import Module

from llmcompressor.modifiers.transform.awq.mappings import (
    AWQ_MAPPING_REGISTRY,
    AWQMapping,
    default_mappings,
)
from llmcompressor.modifiers.transform.utils.hybrid_attention import (
    get_hybrid_attention_config,
)
from llmcompressor.modifiers.utils.pytorch_helpers import is_moe_model

__all__ = ["AWQ_DYNAMIC_MAPPING_REGISTRY", "get_layer_mappings_from_model"]


def get_layer_mappings_from_model(model: Module) -> list[AWQMapping]:
    """
    Infer AWQ mappings from a model. Checks the dynamic mapping registry
    first (for models needing runtime-generated mappings), then falls back
    to the static registry, then to default mappings.

    :param model: the model to infer mappings for
    :return: list of AWQMapping for the model
    """
    model_name = model.__class__.__name__

    if model_name in AWQ_DYNAMIC_MAPPING_REGISTRY:
        mappings = AWQ_DYNAMIC_MAPPING_REGISTRY[model_name](model)
        if mappings is not None:
            return mappings

    if model_name in AWQ_MAPPING_REGISTRY:
        return AWQ_MAPPING_REGISTRY[model_name]

    logger.info(
        f"Architecture {model_name} not found in mappings. "
        f"Using default mappings: {default_mappings}"
    )
    return default_mappings


def build_hybrid_attention_mappings(model: Module) -> list[AWQMapping] | None:
    """
    Dynamically build AWQ mappings for models with hybrid attention
    (full self-attention + linear/Gated DeltaNet attention), such as
    Qwen3Next and Qwen3.5.

    Reads layer_types from the model config to determine which layers use
    full vs linear attention, then inspects the model's module names to
    detect the correct linear attention projection names and MLP structure.

    Returns None if the model is not a hybrid attention model.
    """
    result = get_hybrid_attention_config(model)
    if result is None:
        return None

    layer_types, num_layers = result

    full_indices = [i for i in range(num_layers) if layer_types[i] == "full_attention"]
    linear_indices = [
        i for i in range(num_layers) if layer_types[i] == "linear_attention"
    ]

    if not full_indices or not linear_indices:
        logger.warning(
            "Hybrid attention model detected but could not find indices for "
            "both full and linear attention layers. Falling back."
        )
        return None

    full_re = "|".join(str(i) for i in full_indices)
    linear_re = "|".join(str(i) for i in linear_indices)

    linear_proj_names = _detect_linear_attn_projections(model)
    is_moe = is_moe_model(model)

    mappings = []

    # Full attention layers: input_layernorm -> q/k/v_proj
    mappings.append(
        AWQMapping(
            f"re:.*layers\\.({full_re})\\.input_layernorm$",
            [
                "re:.*self_attn.q_proj$",
                "re:.*self_attn.k_proj$",
                "re:.*self_attn.v_proj$",
            ],
        )
    )

    # Linear attention layers: input_layernorm -> linear_attn projections
    if linear_proj_names:
        mappings.append(
            AWQMapping(
                f"re:.*layers\\.({linear_re})\\.input_layernorm$",
                [f"re:.*linear_attn.{p}$" for p in linear_proj_names],
            )
        )

    # MLP mappings depend on whether the model uses MoE
    if is_moe:
        mappings.append(
            AWQMapping(
                "re:.*post_attention_layernorm$",
                [
                    # TODO: should add "re:.*mlp.gate.weight$" but is a Parameter
                    "re:.*mlp.experts.*.gate_proj$",
                    "re:.*mlp.experts.*.up_proj$",
                    "re:.*mlp.shared_expert_gate$",
                    "re:.*mlp.shared_expert.gate_proj$",
                    "re:.*mlp.shared_expert.up_proj$",
                ],
            )
        )
    else:
        mappings.append(
            AWQMapping(
                "re:.*post_attention_layernorm$",
                ["re:.*gate_proj$", "re:.*up_proj$"],
            )
        )

    mappings.append(AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]))

    logger.info(
        f"Built dynamic hybrid attention AWQ mappings: "
        f"{len(full_indices)} full-attention layers, "
        f"{len(linear_indices)} linear-attention layers, "
        f"linear projections: {linear_proj_names}, MoE: {is_moe}"
    )

    return mappings


def build_step3p5_mappings(model: Module) -> list[AWQMapping] | None:
    """
    Dynamically build AWQ mappings for Step3p5 models.

    Step-3.5-Flash uses dense FFN layers early in the stack and MoE FFN layers
    later in the stack. The dense and MoE post-attention mappings must be
    layer-index-specific so AWQ only groups each norm with balance layers that
    exist in the same decoder layer.
    """
    dense_indices, moe_indices = _detect_step3p5_ffn_layer_indices(model)

    if not dense_indices and not moe_indices:
        logger.warning(
            "Step3p5 model detected but dense/MoE FFN layer indices could not be "
            "inferred. Falling back."
        )
        return None

    mappings = [
        AWQMapping(
            "re:.*input_layernorm$",
            [
                "re:.*self_attn.q_proj$",
                "re:.*self_attn.k_proj$",
                "re:.*self_attn.v_proj$",
                "re:.*self_attn.g_proj$",
            ],
        ),
        AWQMapping("re:.*self_attn.v_proj$", ["re:.*self_attn.o_proj$"]),
    ]

    if dense_indices:
        dense_re = "|".join(str(i) for i in dense_indices)
        mappings.append(
            AWQMapping(
                f"re:.*layers\\.({dense_re})\\.post_attention_layernorm$",
                [
                    "re:.*mlp.gate_proj$",
                    "re:.*mlp.up_proj$",
                ],
            )
        )

    if moe_indices:
        moe_re = "|".join(str(i) for i in moe_indices)
        mappings.append(
            AWQMapping(
                f"re:.*layers\\.({moe_re})\\.post_attention_layernorm$",
                [
                    "re:.*moe.gate$",
                    "re:.*moe.gate_proj$",
                    "re:.*moe.up_proj$",
                    "re:.*share_expert.gate_proj$",
                    "re:.*share_expert.up_proj$",
                ],
            )
        )

    # The packed moe.up_proj -> moe.down_proj path is intentionally excluded
    # because AWQ's smooth-layer update assumes a 1D/2D smooth weight.
    mappings.append(
        AWQMapping(
            "re:.*(mlp|share_expert).up_proj$",
            ["re:.*(mlp|share_expert).down_proj$"],
        )
    )

    logger.info(
        f"Built dynamic Step3p5 AWQ mappings: "
        f"{len(dense_indices)} dense layers, {len(moe_indices)} MoE layers"
    )

    return mappings


# Relative module path of a decoder-layer member, e.g.
# "model.layers.3.mlp.experts.0.gate_proj" -> idx 3, rel "mlp.experts.0.gate_proj".
_LAYER_MEMBER_PATTERN = re.compile(r"(?:^|\.)layers\.(?P<idx>\d+)\.(?P<rel>.+)$")

# Candidate balance layers for the MoE-block input of an MLA mixed dense/MoE
# stack, as (AWQ pattern, matcher against the layer-relative module path).
# Ordered most-specific-first for readability only; inclusion is decided by
# presence, not order.
_MLA_MOE_BALANCE_CANDIDATES: list[tuple[str, re.Pattern]] = [
    # THE ROUTER. It consumes post_attention_layernorm exactly like the experts
    # do, so when AWQ divides that norm by s the router must be multiplied by s
    # or its logits shift and top-k expert selection changes. Being exempt from
    # QUANTIZATION (it is in every recipe's ignore list) does not exempt it from
    # this: see BUGS_AND_FIXES.md, "GLM-5.2 AWQ leaves the MoE router
    # uncompensated". MiniMax-M3 has always balanced it
    # (pipeline/minimax_m3_config.py) and asserts so in a test.
    ("re:.*mlp[.]gate$", re.compile(r"^mlp\.gate$")),
    ("re:.*mlp[.]shared_experts[.]gate_up_proj$",
     re.compile(r"^mlp\.shared_experts\.gate_up_proj$")),
    ("re:.*mlp[.]shared_experts[.]gate_proj$",
     re.compile(r"^mlp\.shared_experts\.gate_proj$")),
    ("re:.*mlp[.]shared_experts[.]up_proj$",
     re.compile(r"^mlp\.shared_experts\.up_proj$")),
    ("re:.*mlp[.]experts[.][0-9]+[.]gate_proj$",
     re.compile(r"^mlp\.experts\.\d+\.gate_proj$")),
    ("re:.*mlp[.]experts[.][0-9]+[.]up_proj$",
     re.compile(r"^mlp\.experts\.\d+\.up_proj$")),
]


# Candidate balance layers for the ATTENTION input (post input_layernorm) of an
# MLA stack. The DSA indexer's projections are here for the same reason the router
# is in the MoE-input set: they consume the smoothed norm's output and must be
# multiplied by s to preserve the function.
#
# Verified against transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py, not
# assumed. GlmMoeDsaDecoderLayer.forward computes
# `hidden_states = self.input_layernorm(hidden_states)` and hands that tensor to
# self_attn, which passes it VERBATIM to `self.indexer(hidden_states, q_resid, ...)`;
# inside the indexer, `self.wk(hidden_states)` and
# `self.weights_proj(hidden_states)` consume it. So wk and weights_proj see exactly
# what q_a_proj and kv_a_proj_with_mqa see.
#
# Leaving them out would feed them x/s, changing the DSA index scores and
# therefore WHICH TOKENS ARE ATTENDED -- the uncompensated-router defect, one
# block earlier.
#
# THIS IS A LATENT TRAP, NOT A SHIPPED DEFECT. Read the skip logic in
# awq/base.py before concluding otherwise: a mapping is `continue`d when
# `any_targeted` is False, i.e. when neither the smooth layer nor ANY balance
# layer is int-quantized (`_is_grid_search_targeted` deliberately excludes
# float-schemed modules). On today's GLM recipes every attention-side balance
# layer -- q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj -- is FP8_DYNAMIC, so
# all three attention mappings are skipped and those norms are never divided by s.
# No GLM checkpoint we have produced has an uncompensated indexer. Contrast the
# MoE-input mapping, whose routed experts ARE int4: that one applies, which is why
# the router defect was real and measured at 1.08e-1 to 2.42e-1.
#
# What makes it worth fixing anyway is how cheaply it goes live: move anything in
# the attention block to int4 -- deliberately, or by a typo dropping an entry from
# `fp8_dynamic_targets`, which sends those projections THROUGH to the int4
# modifier rather than to BF16 -- and `any_targeted` flips to True, the fold
# applies, and the indexer is silently uncompensated. Nothing would catch it:
# pipeline/m3_checkpoint_scale_audit.py audits only post_attention_layernorm,
# mlp.gate and the shared experts, so no gate looks at input_layernorm or
# q_a_layernorm at all.
_MLA_ATTN_INPUT_BALANCE_CANDIDATES: list[tuple[str, re.Pattern]] = [
    ("re:.*(q|q_a)_proj$", re.compile(r"^self_attn\.(q|q_a)_proj$")),
    ("re:.*kv_a_proj_with_mqa$", re.compile(r"^self_attn\.kv_a_proj_with_mqa$")),
    ("re:.*self_attn[.]indexer[.]wk$", re.compile(r"^self_attn\.indexer\.wk$")),
    (
        "re:.*self_attn[.]indexer[.]weights_proj$",
        re.compile(r"^self_attn\.indexer\.weights_proj$"),
    ),
]

# Candidate balance layers for the q_a_layernorm output. `q_resid =
# q_a_layernorm(q_a_proj(x))` is consumed by q_b_proj AND by the indexer's wq_b.
_MLA_QA_BALANCE_CANDIDATES: list[tuple[str, re.Pattern]] = [
    ("re:.*q_b_proj$", re.compile(r"^self_attn\.q_b_proj$")),
    ("re:.*self_attn[.]indexer[.]wq_b$", re.compile(r"^self_attn\.indexer\.wq_b$")),
]

_INDEXER_PRESENT = re.compile(r"^self_attn\.indexer(\.|$)")


def _layer_members(model: Module) -> dict[int, set[str]]:
    """Layer index -> set of module paths relative to that decoder layer."""
    members: dict[int, set[str]] = {}
    for name, _ in model.named_modules():
        match = _LAYER_MEMBER_PATTERN.search(name)
        if match is not None:
            members.setdefault(int(match.group("idx")), set()).add(match.group("rel"))
    return members


def _present_in_all(
    candidates: list[tuple[str, re.Pattern]],
    indices: list[int],
    members: dict[int, set[str]],
) -> list[str]:
    """Candidate patterns that match a module in EVERY one of ``indices``.

    The same rule ``match_modules_set`` applies: a mapping closes per layer only
    when every balance pattern finds something there, so emitting a pattern that
    is absent from one layer in scope makes AWQ resolve zero mappings.
    """
    return [
        pattern
        for pattern, matcher in candidates
        if all(any(matcher.match(rel) for rel in members[index]) for index in indices)
    ]


def _partitioned_mappings(
    smooth_rel: str,
    candidates: list[tuple[str, re.Pattern]],
    members: dict[int, set[str]],
    smooth_matcher: re.Pattern,
) -> list[AWQMapping]:
    """One mapping per group of layers that share a balance set.

    Layers are grouped by whether they carry a DSA indexer, because only 21 of
    GLM-5.2's 78 layers do (`indexer_types`), and an unscoped indexer pattern is
    the exact trap the router fell into: absent from the other 57 layers, it stops
    ``match_modules_set`` from closing any set. When every layer agrees the
    mapping is left UNSCOPED, which keeps the common case's patterns simple and
    preserves the previous behaviour for models with no indexer at all.
    """
    scoped = [i for i, rel in members.items() if any(smooth_matcher.match(r) for r in rel)]
    if not scoped:
        return []
    with_indexer = sorted(i for i in scoped if any(_INDEXER_PRESENT.match(r) for r in members[i]))
    without = sorted(i for i in scoped if i not in set(with_indexer))

    if not with_indexer or not without:
        balance = _present_in_all(candidates, sorted(scoped), members)
        return [AWQMapping(f"re:.*{smooth_rel}$", balance)] if balance else []

    mappings = []
    for group in (with_indexer, without):
        balance = _present_in_all(candidates, group, members)
        if not balance:
            continue
        scope = "|".join(str(i) for i in group)
        mappings.append(AWQMapping(f"re:.*layers[.]({scope})[.]{smooth_rel}$", balance))
    return mappings


def _detect_mla_dense_moe_layers(
    members: dict[int, set[str]],
) -> tuple[list[int], list[int]]:
    """Split decoder layers into dense-MLP and MoE.

    Detected from the modules that actually exist rather than read from
    ``first_k_dense_replace``, because what matters is what
    ``match_modules_set`` will find -- and the module tree can differ from the
    config after ``linearize_moe`` fuses/unfuses experts.
    """
    dense, moe = [], []
    for index, rel in members.items():
        if any(_MLA_MOE_BALANCE_CANDIDATES[0][1].match(r) for r in rel):
            moe.append(index)
        elif "mlp.gate_proj" in rel or "mlp.gate_up_proj" in rel:
            dense.append(index)
    return sorted(dense), sorted(moe)


def build_mla_mixed_dense_moe_mappings(model: Module) -> list[AWQMapping] | None:
    """AWQ mappings for an MLA model with a mixed dense/MoE stack (GLM-5.2).

    Replaces the static ``_mla_mixed_dense_moe_mappings``, which had to drop the
    router from the MoE-input balance set: its patterns were unscoped, so a
    router pattern -- absent from the dense prefix layers -- stopped
    ``match_modules_set`` from ever closing a per-layer set and AWQ resolved zero
    mappings. Dropping it silenced that error but broke function preservation for
    routing.

    Scoping the smooth-layer pattern to the MoE layer indices removes the
    conflict, so the router can be balanced where it exists and is simply absent
    from the dense mapping. This is exactly what
    ``pipeline/minimax_m3_config.py`` does for MiniMax-M3 via
    ``_M3_SPARSE_LAYER``.

    Every balance pattern is emitted only if it matches a module in EVERY MoE
    layer. That is the same rule ``match_modules_set`` applies, so a
    shared-expert form that some architecture spells differently (or does not
    have) cannot reintroduce the zero-mappings failure.

    Returns None when no MoE layers are detected, so a dense model falls through
    to the static registry unchanged.
    """
    members = _layer_members(model)
    if not members:
        return None
    dense_indices, moe_indices = _detect_mla_dense_moe_layers(members)
    if not moe_indices:
        return None

    mappings = [
        *_partitioned_mappings(
            "input_layernorm",
            _MLA_ATTN_INPUT_BALANCE_CANDIDATES,
            members,
            re.compile(r"^input_layernorm$"),
        ),
        *_partitioned_mappings(
            "q_a_layernorm",
            _MLA_QA_BALANCE_CANDIDATES,
            members,
            re.compile(r"^self_attn\.q_a_layernorm$"),
        ),
        AWQMapping("re:.*kv_a_layernorm$", ["re:.*kv_b_proj$"]),
    ]

    if dense_indices:
        dense_re = "|".join(str(i) for i in dense_indices)
        mappings.append(
            AWQMapping(
                f"re:.*layers[.]({dense_re})[.]post_attention_layernorm$",
                ["re:.*mlp[.]gate_proj$", "re:.*mlp[.]up_proj$"],
            )
        )

    balance = [
        pattern
        for pattern, matcher in _MLA_MOE_BALANCE_CANDIDATES
        if all(
            any(matcher.match(rel) for rel in members[index]) for index in moe_indices
        )
    ]
    if not any(pattern.endswith("mlp[.]gate$") for pattern in balance):
        # The router is the whole point of this builder. If it is not uniformly
        # present, fall back rather than emit a mapping that silently omits it.
        logger.warning(
            "MLA mixed dense/MoE mappings: router (mlp.gate) not present in every "
            f"MoE layer {moe_indices[:5]}...; falling back to static mappings"
        )
        return None
    mappings.append(
        AWQMapping(
            f"re:.*layers[.]({'|'.join(str(i) for i in moe_indices)})"
            "[.]post_attention_layernorm$",
            balance,
        )
    )
    mappings.append(AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]))

    indexer_indices = sorted(
        i for i, rel in members.items() if any(_INDEXER_PRESENT.match(r) for r in rel)
    )
    logger.info(
        f"Built MLA mixed dense/MoE AWQ mappings: {len(dense_indices)} dense + "
        f"{len(moe_indices)} MoE layers, {len(indexer_indices)} with a DSA indexer; "
        f"MoE-input balance set: {balance}"
    )
    for mapping in mappings:
        if "layernorm" in mapping.smooth_layer:
            logger.info(
                f"  attention-side mapping: {mapping.smooth_layer} -> "
                f"{mapping.balance_layers}"
            )
    return mappings


AWQ_DYNAMIC_MAPPING_REGISTRY: dict[str, Callable[[Module], list[AWQMapping] | None]] = {
    "Glm4MoeLiteForCausalLM": build_mla_mixed_dense_moe_mappings,
    "GlmMoeDsaForCausalLM": build_mla_mixed_dense_moe_mappings,
    "Qwen3NextForCausalLM": build_hybrid_attention_mappings,
    "Qwen3_5ForCausalLM": build_hybrid_attention_mappings,
    "Qwen3_5ForConditionalGeneration": build_hybrid_attention_mappings,
    "Qwen3_5MoeForCausalLM": build_hybrid_attention_mappings,
    "Qwen3_5MoeForConditionalGeneration": build_hybrid_attention_mappings,
    "Step3p5ForCausalLM": build_step3p5_mappings,
}


def _detect_linear_attn_projections(model: Module) -> list[str]:
    """
    Detect the linear attention projection names by inspecting the first
    linear_attention layer's submodules.

    Different architectures use different projection layouts:
      - Qwen3Next: in_proj_qkvz, in_proj_ba
      - Qwen3.5:   in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
    """
    proj_names = []
    for name, _ in model.named_modules():
        if ".linear_attn." not in name:
            continue
        # Extract the submodule name after linear_attn.
        sub = name.rsplit("linear_attn.", 1)[-1]
        # Only include input projection layers (in_proj_*)
        if sub.startswith("in_proj_"):
            proj_names.append(sub)
    # Deduplicate while preserving order (same projections repeat per layer)
    return list(dict.fromkeys(proj_names))


_STEP3P5_FFN_LAYER_PATTERN = re.compile(
    r"(?:^|\.)layers\.(?P<idx>\d+)\.(?P<ffn>mlp|moe|share_expert)(?:\.|$)"
)


def _detect_step3p5_ffn_layer_indices(model: Module) -> tuple[list[int], list[int]]:
    """
    Detect which Step3p5 decoder layers use dense MLPs and which use MoE blocks.
    """
    dense_indices: set[int] = set()
    moe_indices: set[int] = set()

    for name, _ in model.named_modules():
        match = _STEP3P5_FFN_LAYER_PATTERN.search(name)
        if match is None:
            continue

        layer_idx = int(match.group("idx"))
        ffn_type = match.group("ffn")
        if ffn_type == "mlp":
            dense_indices.add(layer_idx)
        else:
            moe_indices.add(layer_idx)

    return sorted(dense_indices), sorted(moe_indices)
