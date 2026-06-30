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

    def gptq() -> object:
        kwargs: dict = {"targets": "Linear", "scheme": scheme, "ignore": ignore}
        if quant.gptq_dampening_frac is not None:
            kwargs["dampening_frac"] = quant.gptq_dampening_frac
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
        return [gptq()]
    if method == "awq":
        return awq_then_quant()
    if method == "smoothquant+gptq":
        return [smoothquant(), gptq()]
    if method == "smoothquant+awq":
        return [smoothquant(), *awq_then_quant()]
    if method == "quant_only":
        return [quant_only()]
    if method == "autoround":
        from llmcompressor.modifiers.autoround import AutoRoundModifier

        return [AutoRoundModifier(targets="Linear", scheme=scheme, ignore=ignore)]
    if method in ("spinquant+gptq", "spinquant+awq"):
        # Rotation transforms spread outliers across weights AND activations,
        # which is the most promising lever for low-bit activations (W4A8/W4AFP8).
        from llmcompressor.modifiers.transform import SpinQuantModifier

        rotation = SpinQuantModifier(rotations=["R1", "R2"])
        tail = [gptq()] if method.endswith("gptq") else awq_then_quant()
        return [rotation, *tail]

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
    }
