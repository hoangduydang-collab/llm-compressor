"""Mixed int4+fp8 recipes (M3 r8a): float-schemed modules must not activate
or shape AWQ's grid search, while still being folded at apply time.

Root cause regression (2026-07-23, r8a smoke): the appended FP8_DYNAMIC
modifier attaches quantization schemes before AWQ runs, so fp8 attention /
shared experts (a) turned previously-skipped mappings "targeted" and (b)
entered the grid-search loss and duo-scaling weight means — fp8-carrying
layers chose smoothing-scale medians ~0.33 (vs ~0.9 in the pure-int r6 run)
with 27% of columns outside the (0.2, 5.0) plausibility band.
"""

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from torch.nn import Linear

from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier
from llmcompressor.modifiers.transform.awq.base import _is_grid_search_targeted

INT4_ARGS = QuantizationArgs(
    num_bits=4, type="int", strategy="group", group_size=2, symmetric=True
)
FP8_ARGS = QuantizationArgs(
    num_bits=8, type="float", strategy="channel", symmetric=True, dynamic=False
)


def _schemed(linear: Linear, args: QuantizationArgs) -> Linear:
    linear.quantization_scheme = QuantizationScheme(targets=["Linear"], weights=args)
    return linear


def _m3_like_model() -> torch.nn.ModuleDict:
    """input_layernorm -> fp8 attention; post_attention -> int4 experts +
    fp8 shared expert (the r8a topology)."""
    return torch.nn.ModuleDict(
        {
            "decoder": torch.nn.ModuleDict(
                {
                    "input_layernorm": torch.nn.LayerNorm(4),
                    "self_attn": torch.nn.ModuleDict(
                        {
                            "q_proj": _schemed(Linear(4, 4), FP8_ARGS),
                            "k_proj": _schemed(Linear(4, 4), FP8_ARGS),
                            "v_proj": _schemed(Linear(4, 4), FP8_ARGS),
                        }
                    ),
                    "post_attention_layernorm": torch.nn.LayerNorm(4),
                    "mlp": torch.nn.ModuleDict(
                        {
                            "experts": torch.nn.ModuleList(
                                [
                                    torch.nn.ModuleDict(
                                        {
                                            "gate_proj": _schemed(
                                                Linear(4, 2), INT4_ARGS
                                            ),
                                            "up_proj": _schemed(
                                                Linear(4, 2), INT4_ARGS
                                            ),
                                        }
                                    )
                                    for _ in range(2)
                                ]
                            ),
                            "shared_experts": torch.nn.ModuleDict(
                                {"gate_up_proj": _schemed(Linear(4, 4), FP8_ARGS)}
                            ),
                        }
                    ),
                }
            )
        }
    )


def test_is_grid_search_targeted():
    assert _is_grid_search_targeted(_schemed(Linear(4, 4), INT4_ARGS))
    assert not _is_grid_search_targeted(_schemed(Linear(4, 4), FP8_ARGS))
    assert not _is_grid_search_targeted(Linear(4, 4))


def test_fp8_only_mapping_is_skipped_and_mixed_mapping_keeps_fp8_for_fold():
    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm$",
                ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
            ),
            AWQMapping(
                "re:.*post_attention_layernorm$",
                [
                    "re:.*shared_experts[.]gate_up_proj$",
                    "re:.*experts[.][0-9]+[.]gate_proj$",
                    "re:.*experts[.][0-9]+[.]up_proj$",
                ],
            ),
        ],
    )
    model = _m3_like_model()
    awq._set_resolved_mappings(model)

    smooth_names = {m.smooth_name for m in awq._resolved_mappings}
    # fp8-only balance set -> mapping not "targeted", skipped entirely
    assert not any("input_layernorm" in n and "post" not in n for n in smooth_names)
    # int4 experts keep the post-attention mapping alive...
    post = [
        m
        for m in awq._resolved_mappings
        if "post_attention_layernorm" in m.smooth_name
    ]
    assert len(post) == 1
    balance_names = set(post[0].balance_names)
    # ...and the fp8 shared expert STAYS a balance layer (apply-time
    # compensation fold), it is only excluded from the grid objective.
    assert "decoder.mlp.shared_experts.gate_up_proj" in balance_names
    shared = model["decoder"]["mlp"]["shared_experts"]["gate_up_proj"]
    grid_targeted = [
        b for b in post[0].balance_layers if _is_grid_search_targeted(b)
    ]
    assert shared not in grid_targeted
    assert len(grid_targeted) == 4  # 2 experts x (gate + up)
