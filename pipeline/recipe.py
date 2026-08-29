"""Map a (method, scheme) pair to an llm-compressor recipe (list of modifiers).

The weight algorithm (GPTQ / AWQ / AutoRound) and any conditioning transform
(SmoothQuant / SpinQuant rotation) are orthogonal to the *scheme*, which decides
the on-disk numeric format (W4AFP8 / W4A8 / ...). This builder composes the two.

References (existing examples this mirrors):
  - GPTQ W4AFP8:  examples/quantization_w4a8_fp8/llama3_example.py
  - AWQ  W4AFP8:  examples/awq/w4a8_fp8_llama_example.py
  - SmoothQuant:  examples/quantization_w8a8_int8/llama3_example.py
"""

from pipeline.config import QuantizationConfig


def assert_fp8_rest_scheme(name: str) -> None:
    """Fail closed unless ``name`` is an 8-bit FLOAT weight preset.

    This guards more than a typo. ``fp8_dynamic_targets`` covers the attention
    projections, shared experts and dense MLPs -- everything the main W4 modifier
    ignores -- and AWQ decides whether to grid-search a mapping by asking whether
    any balance layer is integer-quantized (``_is_grid_search_targeted`` returns
    False for float schemes). So a name like "W4A16" here would not merely change
    a dtype: it would silently make the attention-side mappings eligible for
    smoothing and int4-quantize projections the recipe is documented to keep at 8
    bits, with nothing downstream to catch it. Checking the resolved scheme rather
    than the string means a future preset rename cannot slip through either.
    """
    from compressed_tensors.quantization import preset_name_to_scheme

    # Allowlist first. The resolved-scheme check below is necessary but NOT
    # sufficient: MXFP8 also resolves to 8-bit float weights, so it passes that
    # test, yet it is a Blackwell-native microscaling format (1x32 blocks, E8M0
    # scales) that SM90 does not implement in hardware and that SGLang's
    # `quant_method: w4afp8` loader does not expect. Naming the two supported
    # presets is the honest statement of intent; the check after it still guards
    # against one of them being redefined upstream.
    supported = ("FP8_DYNAMIC", "FP8_BLOCK")
    if name not in supported:
        raise ValueError(
            f"fp8_scheme={name!r} is not supported for the FP8-rest pass. Use one "
            f"of {supported}: FP8_DYNAMIC for per-output-channel, FP8_BLOCK for "
            f"per-128x128 (what zai-org, DeepSeek-V3 and PhalaCloud's "
            f"SGLang-served W4AFP8 all ship). Other 8-bit float presets such as "
            f"MXFP8 resolve fine but do not serve on SM90."
        )
    try:
        resolved = preset_name_to_scheme(name, ["Linear"])
    except Exception as err:  # unknown preset, or compressed-tensors renamed it
        raise ValueError(
            f"fp8_scheme={name!r} is not a known compressed-tensors preset "
            f"({type(err).__name__}: {err}). Use FP8_DYNAMIC (per-channel) or "
            f"FP8_BLOCK (per-128x128, what SGLang and zai-org ship)."
        ) from err
    weights = resolved.weights
    if weights is None or weights.type != "float" or weights.num_bits != 8:
        got = "None" if weights is None else f"{weights.type}/{weights.num_bits}-bit"
        raise ValueError(
            f"fp8_scheme={name!r} resolves to {got} weights, but this pass must "
            f"stay 8-bit float: it targets the attention/shared-expert/dense "
            f"modules, and a non-float scheme there would change which AWQ "
            f"mappings apply as well as the on-disk format. Use FP8_DYNAMIC or "
            f"FP8_BLOCK."
        )


def build_recipe(quant: QuantizationConfig) -> list:
    """Return a list of llm-compressor modifiers for ``quant``.

    Imports are deferred so that importing this module does not require a full
    llm-compressor install (useful for unit-testing config handling).
    """
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

    method = quant.method
    scheme = quant.scheme
    ignore = list(quant.ignore)

    def with_fp8_rest(recipe: list) -> list:
        # r8: FP8_DYNAMIC (W8A8) on modules the main modifier ignores — e.g.
        # M3 attention / shared experts / dense MLPs, which dominate the
        # remaining BF16 weight traffic once the routed experts are int4.
        # Data-free (RTN weights, dynamic per-token activations), so it adds
        # no calibration cost; mirrors
        # examples/quantization_non_uniform/quantization_multiple_modifiers.py
        if quant.fp8_dynamic_targets:
            assert_fp8_rest_scheme(quant.fp8_scheme)
            recipe.append(
                QuantizationModifier(
                    targets=list(quant.fp8_dynamic_targets),
                    scheme=quant.fp8_scheme,
                )
            )
        return recipe

    def gptq() -> object:
        kwargs: dict = {"targets": "Linear", "scheme": scheme, "ignore": ignore}
        if quant.gptq_dampening_frac is not None:
            kwargs["dampening_frac"] = quant.gptq_dampening_frac
        if quant.gptq_offload_hessians:
            kwargs["offload_hessians"] = True
        return GPTQModifier(**kwargs)

    def awq_then_quant() -> list:
        # AWQ computes the smoothing scales; QuantizationModifier applies the
        # actual weight/activation quantization for the chosen scheme.
        return [
            AWQModifier(duo_scaling=quant.awq_duo_scaling),
            QuantizationModifier(targets=["Linear"], scheme=scheme, ignore=ignore),
        ]

    def smoothquant() -> object:
        return SmoothQuantModifier(smoothing_strength=quant.smoothquant_strength)

    def quant_only() -> object:
        return QuantizationModifier(targets=["Linear"], scheme=scheme, ignore=ignore)

    if method == "gptq":
        return with_fp8_rest([gptq()])
    if method == "awq":
        return with_fp8_rest(awq_then_quant())
    if method == "smoothquant+gptq":
        return with_fp8_rest([smoothquant(), gptq()])
    if method == "smoothquant+awq":
        return with_fp8_rest([smoothquant(), *awq_then_quant()])
    if method == "quant_only":
        return with_fp8_rest([quant_only()])
    if method == "autoround":
        from llmcompressor.modifiers.autoround import AutoRoundModifier

        return with_fp8_rest(
            [AutoRoundModifier(targets="Linear", scheme=scheme, ignore=ignore)]
        )
    if method in ("spinquant+gptq", "spinquant+awq"):
        # Rotation transforms spread outliers across weights AND activations,
        # which is the most promising lever for low-bit activations (W4A8/W4AFP8).
        from llmcompressor.modifiers.transform import SpinQuantModifier

        rotation = SpinQuantModifier(rotations=["R1", "R2"])
        tail = [gptq()] if method.endswith("gptq") else awq_then_quant()
        return with_fp8_rest([rotation, *tail])

    raise ValueError(f"unhandled method {method!r}")


def describe_recipe(quant: QuantizationConfig) -> dict:
    """A JSON-serializable summary of the recipe, for run metadata."""
    return {
        "method": quant.method,
        "scheme": quant.scheme,
        "ignore": list(quant.ignore),
        "smoothquant_strength": quant.smoothquant_strength,
        "awq_duo_scaling": quant.awq_duo_scaling,
        "gptq_dampening_frac": quant.gptq_dampening_frac,
        "gptq_offload_hessians": quant.gptq_offload_hessians,
        "fp8_dynamic_targets": list(quant.fp8_dynamic_targets),
        "fp8_scheme": quant.fp8_scheme,
    }
